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

In 'stim', cue/rating periods fall into the implicit baseline; in 'allevents'
only the ITI does, as in the fMRI model.

Each event is a boxcar over its recorded duration convolved with the PsPM SCRF.
Per-run VIFs of the design are saved (vif_<regressor> columns); the SCRF is slow
and events are seconds apart, so check collinearity before interpreting betas.

Usage:
    python glm_allevents_scr.py --scl-dir <physio01_SCL> --model allevents --save-dir <out>
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

def winsorize_mad(data, threshold):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold_value = threshold * mad
    data[data < median-threshold_value] = np.nan
    data[data > median+threshold_value] = np.nan
    return data

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

def event_regressor(js, event, trial_index, n, scr, samplingrate, data_points_per_second, shift_time=0):
    """boxcar over each selected trial's event duration, convolved with the SCRF"""
    signal = np.zeros(n)
    starts = np.array(js[event]['start'])[trial_index] / samplingrate
    stops = np.array(js[event]['stop'])[trial_index] / samplingrate
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
parser.add_argument('--model', default='allevents', choices=['stim', 'cuestim', 'allevents'])
parser.add_argument('--baselinecorrect', default='False', choices=['True', 'False'])
parser.add_argument('--task', default='pain')
parser.add_argument('--qc', default=join(HERE, '..', '..', 'data', 'QC_EDA_new.csv'))
parser.add_argument('--scrf', default=join(HERE, 'pspm-scrf_td-25.txt'))
args = parser.parse_args()

scl_dir, save_dir, task = args.scl_dir, args.save_dir, args.task
Path(save_dir).mkdir(parents=True, exist_ok=True)

stim_list = ['high_stim-high_cue', 'high_stim-low_cue',
             'med_stim-high_cue', 'med_stim-low_cue',
             'low_stim-high_cue', 'low_stim-low_cue']
cue_list = ['high_cue', 'low_cue']
reg_list = {'stim': stim_list,
            'cuestim': cue_list + stim_list,
            'allevents': cue_list + ['expectrating'] + stim_list + ['actualrating']}[args.model]
samplingrate = 2000
data_points_per_second = 25
scr = pd.read_csv(args.scrf, sep='\t').squeeze()

# glob file list _______________________________________________________________
pattern = f'*{task}_epochstart--3_epochend-20_baselinecorrect-{args.baselinecorrect}_samplingrate-25_physio-eda.txt'
scl_flist = sorted(glob.glob(join(scl_dir, '**', pattern), recursive=True))
merged_df = merge_qc_scl(args.qc, scl_flist)
filtered_list = sorted(merged_df.filename)
print(f"model={args.model}  regressors: {reg_list}")
print(f"after QC 'include': {len(filtered_list)} runs / {merged_df['sub'].nunique()} subs")

# %%----------------------------------------------------------------------------
#                               glm estimation
# ------------------------------------------------------------------------------
rows, skipped = [], []
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
    winsor_mad = winsorize_mad(pdf, threshold=5)
    winsor_interp = interpolate_data(winsor_mad.to_numpy().flatten())

    # run-SD scaling + ITI offset ______________________________________________
    # NOTE kept verbatim from glm_factorial_scr.py: the ITI windows use start/25
    # instead of start/80 (onsets are 2000 Hz samples), so this offset is not a
    # true ITI mean. It is one constant per run and is absorbed by the intercept.
    iti_intervals = []
    for i in range(1, len(js['event_cue']['start'])):
        iti_intervals.append((js['event_actualrating']['stop'][i-1], js['event_cue']['start'][i]))
    winsor_scaled = pd.DataFrame(winsor_interp / np.nanstd(winsor_interp))
    averages = []
    for start, stop in iti_intervals:
        filtered_values = winsor_scaled.loc[start/25:(stop-1)/25] if stop > start else pd.Series(dtype='float64')
        averages.append(np.nanmean(filtered_values))
    winsor_physio = adjust_baseline(winsor_scaled, np.nanmean(averages))
    n = len(winsor_physio)

    # metadata _________________________________________________________________
    metadf = pd.read_csv(meta_glob[0])
    metadf['condition'] = metadf['param_stimulus_type'].astype(str) + '-' + metadf['param_cue_type'].astype(str)
    all_trials = np.arange(len(js['event_cue']['start']))

    # design ___________________________________________________________________
    total_regressor = []
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
        total_regressor.append(event_regressor(js, event, idx, n, scr, samplingrate, data_points_per_second))

    Xmatrix = np.vstack(total_regressor)
    normalized_Xmatrix = (Xmatrix - Xmatrix.min()) / (Xmatrix.max() - Xmatrix.min())

    # linear regression ________________________________________________________
    X_r = np.array(normalized_Xmatrix).T
    Y_r = np.array(winsor_physio).reshape(-1, 1)
    fit = linear_model.LinearRegression().fit(X_r, Y_r)

    row = {'filename': basename, 'sub': sub, 'ses': ses, 'run': run, 'runtype': runtype,
           'intercept': fit.intercept_[0]}
    row.update({r: fit.coef_[0][k] for k, r in enumerate(reg_list)})
    row['modelfit'] = fit.score(X_r, Y_r)
    row.update({f'vif_{r}': v for r, v in zip(reg_list, vif(X_r))})
    rows.append(row)

betadf = pd.DataFrame(rows)
out = join(save_dir, f'glm-{args.model}_task-{task}_baselinecorrect-{args.baselinecorrect}_scr.tsv')
betadf.to_csv(out, sep='\t')
print(f"GLM fit: {len(betadf)} runs / {betadf['sub'].nunique()} subs -> {out}")
if len(betadf):
    v = betadf.filter(like='vif_')
    print("median VIF per regressor:\n" + v.median().round(2).to_string())
if skipped:
    print(f"skipped {len(skipped)} QC'd runs missing onset json or metadata csv:")
    for s in skipped:
        print("   ", s)

with open(out.replace('.tsv', '.json'), 'w') as f:
    json.dump({"source_code": "scripts/p03_glm/glm_allevents_scr.py",
               "model": args.model, "regressors": reg_list,
               "scl_dir": scl_dir, "baselinecorrect": args.baselinecorrect,
               "qc": os.path.abspath(args.qc),
               "outlier": "MAD>5 set to NaN, linear interpolation",
               "scaling": "divide by run SD",
               "regressor": "event boxcars (recorded durations) x PsPM SCRF; design jointly min-max scaled",
               "samplingrate_of_onsettime": 2000, "samplingrate_of_SCL": 25}, f, indent=4)
