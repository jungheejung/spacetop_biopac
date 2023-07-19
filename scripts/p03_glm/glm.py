#!/usr/bin/env python
# encoding: utf-8

# %%----------------------------------------------------------------------
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

def filter_good_data(filenames, baddata_df):
    """
    Args:
        filenames (list): list of single trial file names
        baddata_df: (pd.dataframe): loaded from bad data json, converted as pandas 

    Returns:
        good_data (pd.DataFrame): dataframe, excluding bad data sub/ses/run 
    """
    # Create DataFrame from filenames
    df = pd.DataFrame({'filename': filenames})
    
    # Extract sub, ses, run from the filenames
    df['sub'] = df['filename'].str.extract(r'sub-(\d+)')
    df['ses'] = df['filename'].str.extract(r'ses-(\d+)')
    df['run'] = df['filename'].str.extract(r'run-(\d+)')
    
    # Convert the columns to numeric
    df['sub'] = pd.to_numeric(df['sub'])
    df['ses'] = pd.to_numeric(df['ses'])
    df['run'] = pd.to_numeric(df['run'])
    
    # Merge with index DataFrame and filter good data
    merged = df.merge(baddata_df, on=['sub', 'ses', 'run'], how='left', indicator=True)
    good_data = merged[merged['_merge'] != 'both']
    good_data = good_data.drop('_merge', axis=1)
    
    return good_data


# %% prototype
def merge_qc_scl(qc_fname, scl_flist):
    # qc_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/QC_EDA_new.csv'
    qc = pd.read_csv(qc_fname)
    qc['sub'] = qc['src_subject_id']
    qc['ses'] = qc['session_id']
    qc['run'] = qc['param_run_num']
    qc['task'] = qc['param_task_name']

    qc_sub = qc.loc[qc['Signal quality'] == 'include', ['sub', 'ses', 'run', 'task', 'Signal quality']]
    
    scl_file = pd.DataFrame()
    scl_file['filename'] = pd.DataFrame(scl_flist)
    scl_file['sub'] = scl_file['filename'].str.extract(r'sub-(\d+)').astype(int)
    scl_file['ses'] = scl_file['filename'].str.extract(r'ses-(\d+)').astype(int)
    scl_file['run'] = scl_file['filename'].str.extract(r'run-(\d+)').astype(int)
    scl_file['task'] = scl_file['filename'].str.extract(r'runtype-(\w+)_').astype(str)

    # Merge temp_df with the qc DataFrame on 'sub', 'ses', 'run', and 'runtype'
    # merged_df = pd.merge(temp_df, qc, on=['sub', 'ses', 'run', 'runtype'], how='inner')
    merged_df = pd.merge(scl_file, qc_sub, on=['sub', 'ses', 'run', 'task'], how='inner')
    
    return merged_df
# %%----------------------------------------------------------------------
#                               parameters
# ----------------------------------------------------------------------
scl_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
save_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/glm'
scl_dir = '/Volumes/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
save_dir = '/Volumes/spacetop_projects_cue/analysis/physio/glm'
qc_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/QC_EDA_new.csv'
qc = pd.read_csv(qc_fname)
TR = 0.46
task = 'pain'
# ======= TODO: make code generic
# glob files
# extract info
# add this info to a table
# add file basename to a table
# split this into pandas
# /dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL/sub-0015/ses-01
# scl_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*{task}_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'), recursive=True))
                #    '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'
# %% ======= NOTE: create empty dataframe
df_column = ['filename', 'sub', 'ses', 'run', 'runtype', 'intercept', 'coef'] 

merged_df = merge_qc_scl(qc_fname, scl_flist)
filtered_list = list(merged_df.filename)
betadf = pd.DataFrame(index=range(len(filtered_list)), columns=df_column)
for ind, scl_fpath in enumerate(sorted(filtered_list)):
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
    samplingrate = int(re.search(r'samplingrate-(\d+)_', json_fname).group(1))
    pdf = pd.read_csv(scl_fpath, sep='\t', header=None)
    with open(join(dirname, json_fname)) as json_file:
        js = json.load(json_file)
    betadf.at[ind, 'filename'] = basename


    # ======= NOTE: fetch HRF curve
    # convolve 
    TR = 0.46
    hrf_duration = 20  # Duration of the HRF in seconds
    # hrf_oversampling = 1/25 #1/25  # Number of samples per second
    hrf_values = nilearn.glm.first_level.glover_hrf(1/TR, 
                                                    # oversampling=50, 
                                                    time_length=hrf_duration, onset=0)
    # plt.plot(hrf_values)


    # ======= NOTE: construct event regressors
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
    norm_X = (X - np.mean(X)) / np.std(X)
    index = norm_X.index
    y = convolved_signal
    plt.plot(index, norm_X )
    plt.plot(index, y*200)
    plt.savefig(join(save_dir, basename[:-4]+'.png'))
    plt.close()
    #======= NOTE: linear modeling dataframe
    X_r = np.array(norm_X).reshape(-1,1)
    Y_r = np.array(y*200).reshape(-1,1)
    reg = linear_model.LinearRegression().fit(X_r, Y_r)
    reg.score(X_r, Y_r)
    print(f"coefficient: {reg.coef_[0][0]}, intercept: {reg.intercept_[0]}")

    betadf.at[ind, 'coef'] = reg.coef_[0][0]
    betadf.at[ind, 'intercept'] = reg.intercept_[0]

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