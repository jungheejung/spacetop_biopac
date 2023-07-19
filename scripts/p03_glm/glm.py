#!/usr/bin/env python
# encoding: utf-8

# ----------------------------------------------------------------------
#                               libraries
# ----------------------------------------------------------------------
import os, glob, re
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import convolve
from sklearn import linear_model
import nilearn
from nilearn import glm
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def extract_meta(basename):
    # basename = os.path.basename(fname)
    sub_ind = int(re.search(r'sub-(\d+)', basename).group(1))
    ses_ind = int(re.search(r'ses-(\d+)', basename).group(1))
    run_ind = int(re.search(r'run-(\d+)', basename).group(1))
    runtype = re.search(r'runtype-(.*?)_', basename).group(1)
    return sub_ind, ses_ind, run_ind, runtype

# ----------------------------------------------------------------------
#                               parameters
# ----------------------------------------------------------------------
scl_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
save_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/glm'
TR = 0.46
task = 'pain'
# ======= TODO: make code generic
# glob files
# extract info
# add this info to a table
# add file basename to a table
# split this into pandas
# /dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL/sub-0015/ses-01
scl_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
scl_flist = sorted(glob.glob(join(scl_dir,'**', '**', 'sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'), recursive=True))
                #    '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'
# %% ======= NOTE: create empty dataframe
df_column = ['filename', 'sub', 'ses', 'run', 'runtype', 'intercept', 'coef'] 
betadf = pd.DataFrame(index=range(len(scl_flist)), columns=df_column)

for ind, scl_fpath in enumerate(sorted(scl_flist)):
    # ======= NOTE: load data
    # pdf_fname = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'
    # sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
    # 'sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'
    # sub-0017_ses-03_run-05_runtype-pain_samplingrate-2000_onset.json
    # jsonfname = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_samplingrate-2000_onset.json'

    basename = os.path.basename(scl_fpath)
    dirname = os.path.dirname(scl_fpath)
    sub_ind, ses_ind, run_ind, runtype = extract_meta(basename)
    json_fname = f"sub-{sub_ind:04d}_ses-{ses_ind:02d}_run-{run_ind:02d}_runtype-{runtype}_samplingrate-2000_onset.json"
    samplingrate = re.search(r'samplingrate-(\d+)_', json_fname).group(1)
    pdf = pd.read_csv(scl_fpath, sep='\t', header=None)
    with open(join(dirname, json_fname)) as json_file:
        js = json.load(json_file)
    betadf.at[ind, 'filename'] = basename


    # %% ======= NOTE: fetch HRF curve
    # convolve 
    TR = 0.46
    hrf_duration = 20  # Duration of the HRF in seconds
    # hrf_oversampling = 1/25 #1/25  # Number of samples per second
    hrf_values = nilearn.glm.first_level.glover_hrf(1/TR, 
                                                    # oversampling=50, 
                                                    time_length=hrf_duration, onset=0)
    # plt.plot(hrf_values)


    # %% ======= NOTE: construct event regressors
    onset_sec = np.array(js['event_stimuli']['start'])/samplingrate
    total_runlength_sec = 400; data_points_per_second = 25
    shift_time = 3
    array_length = total_runlength_sec * data_points_per_second
    signal = np.zeros(len(pdf)) #np.zeros(total_runlength_sec * data_points_per_second)
    event_time = np.array(js['event_stimuli']['start'])/samplingrate
    eventtime_shift = event_time + shift_time
    event_indices = (eventtime_shift * data_points_per_second).astype(int)
    signal[event_indices[:len(pdf)]] = 1


    # ======= NOTE: convolve 
    convolved_signal = convolve(signal, hrf_values, mode='full')[:len(signal)]
    X = pdf[0]
    index = X.index
    y = convolved_signal
    plt.plot(index, X )
    plt.plot(index, y*10)


    # %%======= NOTE: linear modeling dataframe
    X_r = np.array(X).reshape(-1,1)
    Y_r = np.array(y).reshape(-1,1)
    reg = linear_model.LinearRegression().fit(X_r, Y_r)
    reg.score(X_r, Y_r)
    print(f"coefficient: {reg.coef_}")
    print(f"intercept: {reg.intercept_}")

    betadf.at[ind, 'coef'] = reg.coef_
    betadf.at[ind, 'intercept'] = reg.intercept_

# ======= NOTE:  extract metadata and save dataframe
betadf['sub']= betadf['filename'].str.extract(r'(sub-\d+)')
betadf['ses'] = betadf['filename'].str.extract(r'(ses-\d+)')
betadf['run'] = betadf['filename'].str.extract(r'(run-\d+)')
betadf['runtype'] = betadf['filename'].str.extract(r'runtype-(\w+)_')
Path(join(save_dir)).mkdir(parents=True, exist_ok=True)
betadf.to_csv(join(save_dir, f'glm-{task}.tsv'))
# TODO: save metadata in json
{"shift":3, 
 "samplingrate_of_onsettime": 2000, 
 "samplingrate_of_SCL": 25, 
 "TR": 0.46, 
 "source_code": "scripts/p03_glm/glm.py",
 "regressor": "stimulus condition convolve"}