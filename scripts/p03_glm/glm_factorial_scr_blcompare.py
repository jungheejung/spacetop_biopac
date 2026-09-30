#!/usr/bin/env python
# encoding: utf-8
"""
glm_factorial_scr_blcompare.py

Baseline-correction robustness check for the SCR factorial GLM.

Runs the *current* glm_factorial_scr.py pipeline (MAD>5 -> NaN + interpolate,
run-SD scaling, ITI-mean baseline, 6 condition boxcars x PsPM SCRF, OLS) on
either SCL input variant, so the two output tables differ ONLY in the upstream
spacetop-prep baseline correction:

    --baselinecorrect True   trial-wise: subtract mean EDA in the 3 s before each
                             heat onset (p01_grouplevel_01SCL.py, --baselinecorrect True)
    --baselinecorrect False  no trial-wise correction (what the manuscript uses)

Differences from glm_factorial_scr.py (intentional, none affect the betas):
  - paths are CLI arguments; QC csv + SCRF default to this repo
  - no per-run plots
  - modelfit (R^2) stored per run (original overwrote the whole column each loop)
  - logs QC'd runs whose onset json / metadata csv is missing instead of crashing

Usage:
    python glm_factorial_scr_blcompare.py \
        --scl-dir  <.../physio01_SCL dir holding *_physio-eda.txt, *_onset.json, *_scltimecourse.csv> \
        --baselinecorrect False \
        --save-dir <output dir>
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

# %%----------------------------------------------------------------------------
#                               arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--scl-dir', required=True)
parser.add_argument('--baselinecorrect', required=True, choices=['True', 'False'])
parser.add_argument('--save-dir', required=True)
parser.add_argument('--task', default='pain')
parser.add_argument('--qc', default=join(HERE, '..', '..', 'data', 'QC_EDA_new.csv'))
parser.add_argument('--scrf', default=join(HERE, 'pspm-scrf_td-25.txt'))
args = parser.parse_args()

scl_dir, save_dir, task = args.scl_dir, args.save_dir, args.task
Path(save_dir).mkdir(parents=True, exist_ok=True)

cond_list = ['high_stim-high_cue', 'high_stim-low_cue',
             'med_stim-high_cue', 'med_stim-low_cue',
             'low_stim-high_cue', 'low_stim-low_cue']
stim_dict = {c: 1 for c in cond_list}
shift_time = 0
data_points_per_second = 25
scr = pd.read_csv(args.scrf, sep='\t').squeeze()

# glob file list _______________________________________________________________
pattern = f'*{task}_epochstart--3_epochend-20_baselinecorrect-{args.baselinecorrect}_samplingrate-25_physio-eda.txt'
scl_flist = sorted(glob.glob(join(scl_dir, '**', pattern), recursive=True))
merged_df = merge_qc_scl(args.qc, scl_flist)
filtered_list = sorted(merged_df.filename)
n_scl_subs = len(set(re.search(r'sub-\d+', f).group(0) for f in scl_flist))
print(f"SCL files found: {len(scl_flist)} runs / {n_scl_subs} subs")
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
    samplingrate = 2000
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

    # run-level ITI baseline ___________________________________________________
    # NOTE kept verbatim for replication: onsets are 2000 Hz samples, so the 25 Hz
    # index should be start/80, not start/25 -- windows land 3.2x too late and most
    # are empty. Harmless for the condition betas: it subtracts one constant per
    # run, which the OLS intercept absorbs.
    iti_intervals = []
    for i in range(1, len(js['event_cue']['start'])):
        iti_intervals.append((js['event_actualrating']['stop'][i-1], js['event_cue']['start'][i]))

    winsor_scaled = pd.DataFrame(winsor_interp / np.nanstd(winsor_interp))
    averages = []
    for start, stop in iti_intervals:
        filtered_values = winsor_scaled.loc[start/25:(stop-1)/25] if stop > start else pd.Series(dtype='float64')
        averages.append(np.nanmean(filtered_values))
    winsor_physio = adjust_baseline(winsor_scaled, np.nanmean(averages))

    # metadata _________________________________________________________________
    metadf = pd.read_csv(meta_glob[0])
    metadf['condition'] = metadf['param_stimulus_type'].astype(str) + '-' + metadf['param_cue_type'].astype(str)

    # convolve stimulus boxcars with canonical SCR _____________________________
    total_regressor = []
    for cond in cond_list:
        signal = np.zeros(len(winsor_physio))
        cond_index = metadf.loc[metadf['condition'] == cond].index.values
        event_start_time = np.array(js['event_stimuli']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_stimuli']['stop'])[cond_index]/samplingrate
        for start, stop in zip(event_start_time, event_stop_time):
            signal[int((start + shift_time) * data_points_per_second):
                   int((stop + shift_time) * data_points_per_second)] = stim_dict[cond]
        total_regressor.append(convolve(signal, scr, mode='full')[:len(signal)])

    Xmatrix = np.vstack(total_regressor)
    normalized_Xmatrix = (Xmatrix - Xmatrix.min()) / (Xmatrix.max() - Xmatrix.min())

    # linear regression ________________________________________________________
    X_r = np.array(normalized_Xmatrix).T
    Y_r = np.array(winsor_physio).reshape(-1, 1)
    reg = linear_model.LinearRegression().fit(X_r, Y_r)

    row = {'filename': basename, 'sub': sub, 'ses': ses, 'run': run, 'runtype': runtype,
           'intercept': reg.intercept_[0]}
    row.update({c: reg.coef_[0][k] for k, c in enumerate(cond_list)})
    row['modelfit'] = reg.score(X_r, Y_r)
    rows.append(row)

betadf = pd.DataFrame(rows, columns=['filename', 'sub', 'ses', 'run', 'runtype', 'intercept']
                      + cond_list + ['modelfit'])
out = join(save_dir, f'glm-factorial_task-{task}_baselinecorrect-{args.baselinecorrect}_scr.tsv')
betadf.to_csv(out, sep='\t')
print(f"GLM fit: {len(betadf)} runs / {betadf['sub'].nunique()} subs -> {out}")
if skipped:
    print(f"skipped {len(skipped)} QC'd runs missing onset json or metadata csv:")
    for s in skipped:
        print("   ", s)

with open(out.replace('.tsv', '.json'), 'w') as f:
    json.dump({"source_code": "scripts/p03_glm/glm_factorial_scr_blcompare.py",
               "scl_dir": scl_dir,
               "baselinecorrect": args.baselinecorrect,
               "qc": os.path.abspath(args.qc),
               "outlier": "MAD>5 set to NaN, linear interpolation",
               "scaling": "divide by run SD; subtract mean ITI level",
               "regressor": "6 stimulus-condition boxcars (heat duration) x PsPM SCRF",
               "samplingrate_of_onsettime": 2000, "samplingrate_of_SCL": 25}, f, indent=4)
