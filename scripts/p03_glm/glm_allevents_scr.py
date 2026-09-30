#!/usr/bin/env python
# encoding: utf-8
"""
glm_allevents_scr.py

SCR factorial GLM with every trial event modeled, mirroring the fMRI single-trial
model (cue, expectation rating, stimulus, outcome rating). Preprocessing is
identical to glm_factorial_scr.py (MAD>5 -> NaN + interpolate, run-SD scaling,
ITI-mean offset, joint min-max scaling of the design, OLS + intercept).

--model selects the regressor set, so all variants share one code path:
    stim       6 stimulus regressors (3 stim x 2 cue)       == glm_factorial_scr.py
    cuestim    + high_cue, low_cue                           == glm_factorial_and_cue_scr.py
    allevents  + high_cue, low_cue, expectrating, actualrating   (fMRI-like)
    singletrial
               one heat regressor per trial only (design of the old
               nobaseline/glm_singletrial table); one output row per trial
    singletrial_allevents
               one heat regressor per trial + high_cue, low_cue, expectrating,
               actualrating as nuisance; one output row per trial, same columns as
               the old nobaseline/glm_singletrial table

In 'stim', cue/rating periods fall into the implicit baseline; in 'allevents'
only the ITI does, as in the fMRI model.

--outlier (default depends on --model):
    trim_interp  values > 5 MAD from the run median -> NaN -> linear interpolation
                 (manuscript condition GLM; default for stim/cuestim/allevents)
    winsor_iqr   cap at Q1 - 5*IQR / Q3 + 5*IQR (feature-engine Winsorizer, as in the
                 Oct 2024 glm_singletrial.py; default for singletrial / singletrial_allevents)
Event onsets: exact in the condition models (as glm_factorial_scr.py); rounded to whole
seconds in the single-trial models (as glm_singletrial.py), so 'singletrial' reproduces
the old nobaseline/glm_singletrial table.

--plot saves two PNGs per run to <save-dir>/plots/<model>/: the signal with the
convolved regressors, and the signal with the model fit.

Each event is a boxcar over its recorded duration convolved with the PsPM SCRF.
Per-run VIFs of the design are saved (vif_<regressor> columns); the SCRF is slow
and events are seconds apart, so check collinearity before interpreting betas.

Usage:
    python glm_allevents_scr.py --scl-dir <physio01_SCL> --model allevents --save-dir <out>
    python glm_allevents_scr.py --scl-dir <physio01_SCL> --model singletrial_allevents --save-dir <out> --plot
"""

import os, glob, re, json, argparse, warnings
from os.path import join, dirname, abspath
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import convolve
from scipy.interpolate import interp1d
from sklearn import linear_model

HERE = dirname(abspath(__file__))
warnings.filterwarnings("ignore", message="Mean of empty slice")  # see ITI note below

# %%----------------------------------------------------------------------------
#                               functions (verbatim from glm_factorial_scr.py)
# ------------------------------------------------------------------------------
def extract_meta(basename):
    sub_ind = int(re.search(r'sub-(\d+)', basename).group(1))
    ses_ind = int(re.search(r'ses-(\d+)', basename).group(1))
    run_ind = int(re.search(r'run-(\d+)', basename).group(1))
    runtype = re.search(r'runtype-(.*?)_', basename).group(1)
    return sub_ind, ses_ind, run_ind, runtype

def trim_mad(data, threshold):
    """set samples more than threshold x MAD from the median to NaN (trimming, not
    winsorizing; named winsorize_mad in glm_factorial_scr.py)"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold_value = threshold * mad
    data[data < median-threshold_value] = np.nan
    data[data > median+threshold_value] = np.nan
    return data

def winsorize_iqr(pdf, fold=5):
    """cap at Q1 - fold*IQR / Q3 + fold*IQR, as glm_singletrial.py (Oct 2024) did with
    feature_engine's Winsorizer(capping_method='iqr'); pandas fallback applies the same rule"""
    try:
        from feature_engine.outliers import Winsorizer
        return Winsorizer(capping_method='iqr', tail='both', fold=fold).fit_transform(pdf), 'feature_engine'
    except ImportError:
        q1, q3 = pdf.quantile(0.25), pdf.quantile(0.75)
        return pdf.clip(lower=q1 - fold * (q3 - q1), upper=q3 + fold * (q3 - q1), axis=1), 'pandas'

def interpolate_data(data):
    time_points = np.arange(len(data))
    valid = ~np.isnan(data)
    interp_func = interp1d(time_points[valid], data[valid], kind='linear', fill_value="extrapolate")
    return interp_func(time_points)

def merge_qc_scl(qc_fname, scl_flist):
    qc = pd.read_csv(qc_fname)
    qc['sub'] = qc['src_subject_id']
    qc['ses'] = qc['session_id']
    qc['run'] = qc['param_run_num']
    qc['task'] = qc['param_task_name']
    qc_sub = qc.loc[qc['Signal quality'] == 'include',
                    ['sub', 'ses', 'run', 'task', 'Signal quality']]

    scl_file = pd.DataFrame()
    scl_file['filename'] = pd.DataFrame(scl_flist)
    scl_file['sub'] = scl_file['filename'].str.extract(r'sub-(\d+)').astype(int)
    scl_file['ses'] = scl_file['filename'].str.extract(r'ses-(\d+)').astype(int)
    scl_file['run'] = scl_file['filename'].str.extract(r'run-(\d+)').astype(int)
    scl_file['task'] = scl_file['filename'].str.extract(r'runtype-(\w+)_').astype(str)
    return pd.merge(scl_file, qc_sub, on=['sub', 'ses', 'run', 'task'], how='inner')

def adjust_baseline(data, baseline):
    if baseline > 0:
        return data - baseline
    else:
        return data + abs(baseline)

def event_regressor(js, event, trial_index, n, scr, samplingrate, data_points_per_second,
                    shift_time=0, round_sec=False):
    """boxcar over each selected trial's event duration, convolved with the SCRF;
    round_sec rounds onsets/offsets to whole seconds, as glm_singletrial.py (Oct 2024) did"""
    signal = np.zeros(n)
    starts = np.array(js[event]['start'])[trial_index] / samplingrate
    stops = np.array(js[event]['stop'])[trial_index] / samplingrate
    if round_sec:
        starts, stops = np.round(starts), np.round(stops)
    for start, stop in zip(starts, stops):
        signal[int((start + shift_time) * data_points_per_second):
               int((stop + shift_time) * data_points_per_second)] = 1
    return convolve(signal, scr, mode='full')[:n]

def vif(X):
    """variance inflation factor per column (X: time x regressors, intercept added)"""
    Xc = np.column_stack([np.ones(len(X)), X])
    out = []
    for j in range(1, Xc.shape[1]):
        others = np.delete(Xc, j, axis=1)
        resid = Xc[:, j] - others @ np.linalg.lstsq(others, Xc[:, j], rcond=None)[0]
        r2 = 1 - resid.var() / Xc[:, j].var()
        out.append(np.inf if r2 >= 1 else 1 / (1 - r2))
    return out

# %%----------------------------------------------------------------------------
#                               arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--scl-dir', required=True)
parser.add_argument('--save-dir', required=True)
parser.add_argument('--model', default='allevents',
                    choices=['stim', 'cuestim', 'allevents', 'singletrial', 'singletrial_allevents'])
parser.add_argument('--outlier', choices=['trim_interp', 'winsor_iqr'],
                    help='default: winsor_iqr for singletrial models, trim_interp otherwise')
parser.add_argument('--plot', action='store_true', help='save 2 PNGs per run')
parser.add_argument('--baselinecorrect', default='False', choices=['True', 'False'])
parser.add_argument('--task', default='pain')
parser.add_argument('--qc', default=join(HERE, '..', '..', 'data', 'QC_EDA_new.csv'))
parser.add_argument('--scrf', default=join(HERE, 'pspm-scrf_td-25.txt'))
args = parser.parse_args()

scl_dir, save_dir, task = args.scl_dir, args.save_dir, args.task
Path(save_dir).mkdir(parents=True, exist_ok=True)
singletrial = args.model.startswith('singletrial')
default_outlier = 'winsor_iqr' if singletrial else 'trim_interp'
outlier = args.outlier or default_outlier
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = join(save_dir, 'plots', args.model)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

stim_list = ['high_stim-high_cue', 'high_stim-low_cue',
             'med_stim-high_cue', 'med_stim-low_cue',
             'low_stim-high_cue', 'low_stim-low_cue']
cue_list = ['high_cue', 'low_cue']
reg_list = {'stim': stim_list,
            'cuestim': cue_list + stim_list,
            'allevents': cue_list + ['expectrating'] + stim_list + ['actualrating'],
            'singletrial': [],
            'singletrial_allevents': cue_list + ['expectrating', 'actualrating']}[args.model]
# singletrial models: reg_list holds the nuisance regressors; per-trial heat regressors are added per run
samplingrate = 2000
data_points_per_second = 25
scr = pd.read_csv(args.scrf, sep='\t').squeeze()

# glob file list _______________________________________________________________
pattern = f'*{task}_epochstart--3_epochend-20_baselinecorrect-{args.baselinecorrect}_samplingrate-25_physio-eda.txt'
scl_flist = sorted(glob.glob(join(scl_dir, '**', pattern), recursive=True))
merged_df = merge_qc_scl(args.qc, scl_flist)
filtered_list = sorted(merged_df.filename)
print(f"model={args.model}  outlier={outlier}  regressors: "
      f"{(['heat trial x N'] if singletrial else []) + reg_list}")
print(f"after QC 'include': {len(filtered_list)} runs / {merged_df['sub'].nunique()} subs")

# %%----------------------------------------------------------------------------
#                               glm estimation
# ------------------------------------------------------------------------------
rows, skipped = [], []
winsor_impl = None
for scl_fpath in filtered_list:
    basename = os.path.basename(scl_fpath)
    rundir = os.path.dirname(scl_fpath)
    sub_ind, ses_ind, run_ind, runtype = extract_meta(basename)
    sub, ses, run = f"sub-{sub_ind:04d}", f"ses-{ses_ind:02d}", f"run-{run_ind:02d}"
    json_fname = f"{sub}_{ses}_{run}_runtype-{runtype}_samplingrate-2000_onset.json"
    meta_glob = glob.glob(join(scl_dir, sub, ses, basename.split('epochend')[0] + "*scltimecourse.csv"))
    if not os.path.exists(join(rundir, json_fname)) or not meta_glob:
        skipped.append(basename)
        continue

    pdf = pd.read_csv(scl_fpath, sep='\t', header=None)
    with open(join(rundir, json_fname)) as json_file:
        js = json.load(json_file)

    # remove outlier ___________________________________________________________
    if outlier == 'trim_interp':      # remove samples > 5 MAD from the median, then interpolate
        trimmed = trim_mad(pdf, threshold=5)
        clean_signal = interpolate_data(trimmed.to_numpy().flatten())
    else:                             # cap samples beyond Q1-5*IQR / Q3+5*IQR (no interpolation)
        capped, winsor_impl = winsorize_iqr(pdf, fold=5)
        clean_signal = np.asarray(capped, dtype=float).flatten()

    # run-SD scaling + ITI offset ______________________________________________
    # NOTE kept verbatim from glm_factorial_scr.py: the ITI windows use start/25
    # instead of start/80 (onsets are 2000 Hz samples), so this offset is not a
    # true ITI mean. It is one constant per run and is absorbed by the intercept.
    iti_intervals = []
    for i in range(1, len(js['event_cue']['start'])):
        iti_intervals.append((js['event_actualrating']['stop'][i-1], js['event_cue']['start'][i]))
    scaled = pd.DataFrame(clean_signal / np.nanstd(clean_signal))
    averages = []
    for start, stop in iti_intervals:
        filtered_values = scaled.loc[start/25:(stop-1)/25] if stop > start else pd.Series(dtype='float64')
        averages.append(np.nanmean(filtered_values))
    scaled_offset = adjust_baseline(scaled, np.nanmean(averages))
    n = len(scaled_offset)

    # metadata _________________________________________________________________
    metadf = pd.read_csv(meta_glob[0])
    metadf['condition'] = metadf['param_stimulus_type'].astype(str) + '-' + metadf['param_cue_type'].astype(str)
    all_trials = np.arange(len(js['event_cue']['start']))

    # design ___________________________________________________________________
    total_regressor, trial_names = [], []
    if singletrial:
        for t in range(len(metadf)):
            cue_t = str(metadf.loc[t, 'param_cue_type']).replace('_cue', '')
            stim_t = str(metadf.loc[t, 'param_stimulus_type']).replace('_stim', '')
            trial_names.append((f"epoch-stim_trial-{t:03d}_cue-{cue_t}_stim-{stim_t}",
                                f"trial-{t + 1:03d}", cue_t, stim_t))
            total_regressor.append(event_regressor(js, 'event_stimuli', [t], n, scr,
                                                   samplingrate, data_points_per_second,
                                                   round_sec=True))
    for reg in reg_list:
        if reg in stim_list:
            idx = metadf.loc[metadf['condition'] == reg].index.values
            event = 'event_stimuli'
        elif reg in cue_list:
            idx = metadf.loc[metadf['param_cue_type'] == reg].index.values
            event = 'event_cue'
        else:
            idx = all_trials
            event = f'event_{reg}'
        total_regressor.append(event_regressor(js, event, idx, n, scr, samplingrate, data_points_per_second,
                                               round_sec=singletrial))

    Xmatrix = np.vstack(total_regressor)
    normalized_Xmatrix = (Xmatrix - Xmatrix.min()) / (Xmatrix.max() - Xmatrix.min())

    # linear regression ________________________________________________________
    X_r = np.array(normalized_Xmatrix).T
    Y_r = np.array(scaled_offset).reshape(-1, 1)
    fit = linear_model.LinearRegression().fit(X_r, Y_r)

    modelfit, vifs, coef = fit.score(X_r, Y_r), vif(X_r), fit.coef_[0]
    if singletrial:
        # one row per trial; columns as in the old nobaseline/glm_singletrial table
        # (singletrial_name counts from trial-000, singletrial_index from trial-001)
        nt = len(trial_names)
        nuis = {f'nuisance_{r}': coef[nt + k] for k, r in enumerate(reg_list)}
        for k, (name, index, cue_t, stim_t) in enumerate(trial_names):
            rows.append({'filename': basename, 'sub': sub, 'ses': ses, 'run': run, 'runtype': runtype,
                         'cuetype': cue_t, 'stimtype': stim_t, 'singletrial_name': name,
                         'singletrial_index': index, 'intercept': fit.intercept_[0], 'beta': coef[k],
                         'modelfit': modelfit, 'vif': vifs[k], **nuis})
    else:
        row = {'filename': basename, 'sub': sub, 'ses': ses, 'run': run, 'runtype': runtype,
               'intercept': fit.intercept_[0]}
        row.update({r: coef[k] for k, r in enumerate(reg_list)})
        row['modelfit'] = modelfit
        row.update({f'vif_{r}': v for r, v in zip(reg_list, vifs)})
        rows.append(row)

    if args.plot:
        t_sec = np.arange(n) / data_points_per_second
        heat = np.zeros(n, bool)
        for a, b in zip(js['event_stimuli']['start'], js['event_stimuli']['stop']):
            heat[int(a / samplingrate * data_points_per_second):int(b / samplingrate * data_points_per_second)] = True
        n_heat = len(trial_names) if singletrial else len(stim_list)
        heat_cue = [tn[2] for tn in trial_names] if singletrial else \
                   [c.split('-')[1].replace('_cue', '') for c in stim_list]
        for kind in ('design', 'modelfitted'):
            fig, ax = plt.subplots(figsize=(11, 3.2))
            ax.fill_between(t_sec, 0, 1, where=heat, transform=ax.get_xaxis_transform(),
                            color='#e6e5df', linewidth=0, label='heat')
            y = Y_r.ravel()
            ax.plot(t_sec, y, color='#1f1f1e', linewidth=0.6, label='SCL (scaled)')
            if kind == 'design':
                lo, hi = np.percentile(y, [1, 99])
                for k in range(X_r.shape[1]):
                    is_heat = k < n_heat
                    high = is_heat and heat_cue[k] == 'high'
                    ax.plot(t_sec, lo + X_r[:, k] * (hi - lo), linewidth=1,
                            color=('#eb6834' if high else '#2a78d6') if is_heat else '#9a9890')
                ax.plot([], [], color='#eb6834', label='heat, high cue')
                ax.plot([], [], color='#2a78d6', label='heat, low cue')
                if reg_list:
                    ax.plot([], [], color='#9a9890', label='cue / rating')
            else:
                ax.plot(t_sec, fit.predict(X_r).ravel(), color='#2a78d6', linewidth=1.5,
                        label=f'model fit (R2 = {modelfit:.2f})')
            ax.set_xlim(0, t_sec[-1]); ax.set_xlabel('time (s)'); ax.set_ylabel('SCL (run SD units)')
            ax.set_title(f"{sub} {ses} {run}  {args.model}", loc='left', fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
            fig.tight_layout()
            suffix = '' if kind == 'design' else '_modelfitted'
            fig.savefig(join(plot_dir, basename[:-4] + suffix + '.png'), dpi=110)
            plt.close(fig)

betadf = pd.DataFrame(rows)
tag = '' if outlier == default_outlier else f'_outlier-{outlier.replace("_", "")}'
out = join(save_dir, f'glm-{args.model}_task-{task}{tag}_baselinecorrect-{args.baselinecorrect}_scr.tsv')
betadf.to_csv(out, sep='\t')
n_runs = betadf['filename'].nunique() if len(betadf) else 0
print(f"GLM fit: {n_runs} runs / {betadf['sub'].nunique() if len(betadf) else 0} subs"
      f"{f' / {len(betadf)} trials' if singletrial else ''} -> {out}")
if len(betadf) and singletrial:
    print(f"single-trial VIF: median {betadf.vif.median():.2f}, max {betadf.vif.max():.2f}")
elif len(betadf):
    v = betadf.filter(like='vif_')
    print("median VIF per regressor:\n" + v.median().round(2).to_string())
if skipped:
    print(f"skipped {len(skipped)} QC'd runs missing onset json or metadata csv:")
    for s in skipped:
        print("   ", s)

with open(out.replace('.tsv', '.json'), 'w') as f:
    json.dump({"source_code": "scripts/p03_glm/glm_allevents_scr.py",
               "model": args.model,
               "regressors": (['heat per trial'] if singletrial else []) + reg_list,
               "scl_dir": scl_dir, "baselinecorrect": args.baselinecorrect,
               "qc": os.path.abspath(args.qc),
               "outlier": {"trim_interp": "MAD>5 set to NaN, linear interpolation",
                           "winsor_iqr": "capped at Q1-5*IQR / Q3+5*IQR (feature-engine Winsorizer 'iqr')"}[outlier],
               "outlier_implementation": winsor_impl,
               "onsets": "rounded to whole seconds" if singletrial else "exact",
               "scaling": "divide by run SD",
               "regressor": "event boxcars (recorded durations) x PsPM SCRF; design jointly min-max scaled",
               "samplingrate_of_onsettime": 2000, "samplingrate_of_SCL": 25}, f, indent=4)
