#!/usr/bin/env python
# encoding: utf-8


# %%----------------------------------------------------------------------------
#                               libraries
# ------------------------------------------------------------------------------
import os, glob, re, json
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import convolve
from scipy import stats
from scipy.interpolate import interp1d
from sklearn import linear_model
import nilearn
from nilearn import glm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import Winsorizer
import seaborn as sns



# %%----------------------------------------------------------------------------
#                               functions
# ------------------------------------------------------------------------------
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

def winsorize_mad(data, threshold=3.5):
    # winsorized_data = data
    # median = np.median(data)
    # mad = stats.median_abs_deviation(data)
    # # mad = np.median(np.abs(data - median))
    # # threshold_value = threshold * mad
    # lower_bound = median - threshold * mad
    # upper_bound = median + threshold * mad
    # winsorized_data[winsorized_data < -threshold_value] = np.nan
    # winsorized_data[winsorized_data > threshold_value] = np.nan
    # lower_proportion = np.sum(data < lower_bound) / len(data)
    # upper_proportion = np.sum(data > upper_bound) / len(data)
    wz = Winsorizer(capping_method='mad', tail='both', fold=threshold)
    winsorized_data = wz.fit_transform(data)
    # winsorized_data = stats.mstats.winsorize(data, limits=(lower_proportion, upper_proportion))
    # winsorized_data = np.clip(data, median - threshold_value, median + threshold_value)
    return winsorized_data


def interpolate_data(data):
    time_points = np.arange(len(data))
    valid = ~np.isnan(data)  # Mask of valid (non-NaN) data points
    interp_func = interp1d(time_points[valid], data[valid], kind='linear', fill_value="extrapolate")
    return interp_func(time_points)

# prototype
def merge_qc_scl(qc_fname, scl_flist):
    """
    Merges quality control (QC) data with skin conductance level (SCL) file information based on subject ID, session, run number, and task name.

    This function reads a QC file and a list of SCL file names. It filters the QC data to include only entries marked for inclusion. It then extracts relevant information from both the QC data and SCL file names, such as subject ID, session, run number, and task name, and merges the two datasets on these dimensions.

    Parameters:
    - qc_fname (str): File path to the QC CSV file. The QC file should contain columns for subject ID   (`src_subject_id`), session ID (`session_id`), run number (`param_run_num`), task name (`param_task_name`), and signal quality (`Signal quality`).
    - scl_flist (list of str): A list containing the file names of SCL files. These file names should include patterns that allow extraction of subject ID (`sub-<id>`), session ID (`ses-<id>`), run number (`run-<id>`), and task name (`runtype-<task_name>_`).

    Returns:
    - DataFrame: A pandas DataFrame resulting from the inner merge of the QC data (filtered by 'Signal quality' == 'include') and the SCL file information on subject ID, session, run number, and task name. This merged DataFrame contains information only for those records present in both datasets and marked for inclusion in the QC data.

    Example:
    >>> qc_fname = 'path/to/QC_EDA_new.csv'
    >>> scl_flist = ['file_sub-01_ses-1_run-1_runtype-task1_.csv', 'file_sub-02_ses-2_run-2_runtype-task2_.csv']
    >>> merged_df = merge_qc_scl(qc_fname, scl_flist)
    >>> print(merged_df)

    Note:
    - The function assumes specific formatting for both the QC file columns and the SCL file name patterns.
    - The merge is performed as an inner join, meaning only records that match across both datasets (and are marked 'include' in QC) will be included in the output DataFrame.
    """
    # qc_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/QC_EDA_new.csv'
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

    # Merge temp_df with the qc DataFrame on 'sub', 'ses', 'run', and 'runtype'
    merged_df = pd.merge(scl_file, qc_sub, on=['sub', 'ses', 'run', 'task'], how='inner')
    
    return merged_df

def boxcar_function(x, start, end):
    """
    Boxcar function that returns 1 for x in the interval [start, end] and 0 otherwise.

    Parameters:
    - x: The input value.
    - start: The start of the interval where the function returns 1.
    - end: The end of the interval where the function returns 1.

    Returns:
    - int: 1 if start <= x <= end, otherwise 0.
    """
    if start <= x <= end:
        return 1
    else:
        return 0
def adjust_baseline(data, baseline):
    if baseline > 0:
        return data - baseline  # Subtract if baseline is positive
    else:
        return data + abs(baseline)  # Add the absolute value if baseline is negative
# %%----------------------------------------------------------------------------
#                               parameters
# ------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10  # You can also set a default font size if desired

scl_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL'                                   
save_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/glm_cueepoch'

scl_dir = '/Volumes/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL' 
save_dir = '/Volumes/spacetop_projects_cue/analysis/physio/nobaseline/glm_cueepoch'                

# # local
# scl_dir = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'  
# scl_dir = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL' 
# save_dir = '/Users/h/Desktop'      


qc_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/QC_EDA_new.csv'
qc = pd.read_csv(qc_fname)
TR = 0.46
task = 'pain'

# glob file list _______________________________________________________________
scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*{task}_epochstart--3_epochend-20_baselinecorrect-False_samplingrate-25_physio-eda.txt'), 
                             recursive=True))

# create empty dataframe _______________________________________________________
df_column = ['filename', 'sub', 'ses', 'run', 'runtype', 'intercept'] 
cond_list = [
            'high_stim-high_cue', 'high_stim-low_cue',
             'med_stim-high_cue', 'med_stim-low_cue',
             'low_stim-high_cue', 'low_stim-low_cue']
merged_df = merge_qc_scl(qc_fname, scl_flist)
filtered_list = list(merged_df.filename)
betadf = pd.DataFrame(index=range(len(filtered_list)), columns=df_column + cond_list)
Path(join(save_dir)).mkdir(parents=True, exist_ok=True)

# %%----------------------------------------------------------------------------
#                               glm estimation
# ------------------------------------------------------------------------------

for ind, scl_fpath in enumerate(sorted(filtered_list)):

    basename = os.path.basename(scl_fpath)
    dirname = os.path.dirname(scl_fpath)
    sub_ind, ses_ind, run_ind, runtype = extract_meta(basename)
    json_fname = f"sub-{sub_ind:04d}_ses-{ses_ind:02d}_run-{run_ind:02d}_runtype-{runtype}_samplingrate-2000_onset.json"
    samplingrate = int(re.search(r'samplingrate-(\d+)_', json_fname).group(1))
    pdf = pd.read_csv(scl_fpath, sep='\t', header=None)
    with open(join(dirname, json_fname)) as json_file:
        js = json.load(json_file)
    betadf.at[ind, 'filename'] = basename

    # remove outlier ___________________________________________________________
    winsor_mad = winsorize_mad(pdf, threshold=5)

    # baseline correction ____________________________
    # Adjusting the calculation for the new requirement
    # The first interval starts from 0 to the first 'event_cue' 'start'
    iti_intervals = []#[(0, js['event_cue']['start'][0])]

    # For subsequent intervals, use the previous 'event_actualrating' 'stop' time to the current 'event_cue' 'start' time
    for i in range(1, len(js['event_cue']['start'])):
        iti_interval = (js['event_actualrating']['stop'][i-1], js['event_cue']['start'][i])
        iti_intervals.append(iti_interval)

    winsor_scaled = winsor_mad/ np.nanstd(winsor_mad)
    # averages
    averages = []
    for start, stop in iti_intervals:
        # Assuming the index is directly comparable to the start and stop times
        filtered_values = winsor_scaled.loc[start/25:(stop-1)/25] if stop > start else pd.Series(dtype='float64')
        averages.append(np.nanmean(filtered_values))
    baseline_value = np.nanmean(averages)


    winsor_physio = adjust_baseline(winsor_scaled, baseline_value)
    # (winsor_physio - np.mean(winsor_physio))/np.nanstd(winsor_physio)
    # winsor_physio_interp = interpolate_data(winsor_physio)

    # fetch SCR curve __________________________________________________________
    pspm_scr = pd.read_csv('/Users/h/Documents/projects_local/spacetop_biopac/scripts/p03_glm/pspm-scrf_td-25.txt', sep='\t')
    scr = pspm_scr.squeeze()


    # extract metadata _________________________________________________________
    sub = f"sub-{sub_ind:04d}"
    ses = f"ses-{ses_ind:02d}"
    run = f"run-{run_ind:02d}"
    meta_fname = glob.glob(join(scl_dir, sub, ses, basename.split('epochend')[0] + "*scltimecourse.csv"))[0]
    metadf = pd.read_csv(meta_fname)
    metadf['condition'] = metadf['param_stimulus_type'].astype(str) + '-' + metadf['param_cue_type'].astype(str) 


    # cond_type ________________________________________________________________
    total_runlength_sec = 400; data_points_per_second = 25
    shift_time = 0
    array_length = total_runlength_sec * data_points_per_second
    signal = np.zeros(len(winsor_physio)) #np.zeros(total_runlength_sec * data_points_per_second)

    stim_dict = {"high_cue":1,
                 "low_cue":1,
                "high_stim-high_cue": 1,
                 "high_stim-low_cue": 1,
                 "med_stim-high_cue": 1,
                 "med_stim-low_cue": 1,
                 "low_stim-high_cue": 1,
                 "low_stim-low_cue": 1
                }

    # convolve onsets with boxcar and canonical SCR ____________________________
    #       UPDATE 02/27/2024. 
    #       We're creating boxcar function based on the stimulus duration
    #       prior to this update, only the event onset was convolved with the SCR function. 
    #       in other words, the convolved SCR was just a blip.
    total_regressor = []
    boxcar = []


    for cond in ['high_cue', 'low_cue']:
        signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['param_cue_type'] == cond].index.values
        event_start_time = np.array(js['event_cue']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_cue']['stop'])[cond_index]/samplingrate
        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            signal[start_index:stop_index] = 1 #stim_dict[cond]
        # Convolve the signal with the scr
        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)

    for cond in cond_list:
        signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['condition'] == cond].index.values
        event_start_time = np.array(js['event_stimuli']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_stimuli']['stop'])[cond_index]/samplingrate
        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            signal[start_index:stop_index] = stim_dict[cond]
        # Convolve the signal with the scr
        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)


    boxcar_total = np.sum(np.stack(boxcar), axis=0)
    boxcar_summed = (boxcar_total > 0).astype(int)
    # plot convolved signal ____________________________________________________
    Xmatrix = np.vstack(total_regressor)
    normalized_Xmatrix =  (Xmatrix - Xmatrix.min()) / (Xmatrix.max() - Xmatrix.min())
    # Xmatrix
    y = winsor_physio #pdf[0]
    total_time = len(y) / 25
    index = np.arange(len(y))
    x_seconds = index / 25  # Convert index to seconds
    index = np.arange(len(y))#y.index
    cue_stim_cond = ['high_cue', 'low_cue','high_stim-high_cue',
 'high_stim-low_cue',
 'med_stim-high_cue',
 'med_stim-low_cue',
 'low_stim-high_cue',
 'low_stim-low_cue']
    for cond_ind in np.arange(len(cue_stim_cond)):
        plt.plot(x_seconds, normalized_Xmatrix[cond_ind].T)
    # TODO; display x ticks in seconds
 

    boxheight = (np.max(winsor_physio) - np.min(winsor_physio)) * 0.25
    plt.xlim(0, total_time)
    y_min, y_max = plt.ylim()
    plt.plot(x_seconds, y, '#2F2f2f', label='SCR signal',alpha=1, linewidth=.5) #signal
    boxcar_height = np.max(boxcar_summed)
    plt.fill_between(x_seconds,  
                     y_min, y_max, where=boxcar_summed > 0, 
                #  boxcar_summed*y_min, boxcar_summed *y_max,
                 edgecolor=None,
                 facecolor='#9f9f9f', alpha=0.3, label='heat onset')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("SCR amplitude (A.U.)", fontsize=14)
    plt.title(f"{sub} {ses} {run}\nSkin conductance GLM fit", fontsize=18)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    sns.despine()
    
    plt.savefig(join(save_dir, basename[:-4]+'.png'))
    plt.show()
    plt.close()

    # linear regression ________________________________________________________
    X_r = np.array(normalized_Xmatrix).T
    Y_r = np.array(y).reshape(-1,1)
    reg = linear_model.LinearRegression().fit(X_r, Y_r)
    modelfit = reg.score(X_r, Y_r)
    print(f"coefficient: {reg.coef_[0][0]}, {reg.coef_[0][1]}, {reg.coef_[0][2]}, \
          {reg.coef_[0][3]}, {reg.coef_[0][4]}, {reg.coef_[0][5]}, \
          {reg.coef_[0][6]}, {reg.coef_[0][7]}, \
          intercept: {reg.intercept_[0]}")

    betadf.at[ind, cue_stim_cond[0]] = reg.coef_[0][0]
    betadf.at[ind, cue_stim_cond[1]] = reg.coef_[0][1]
    betadf.at[ind, cue_stim_cond[2]] = reg.coef_[0][2]
    betadf.at[ind, cue_stim_cond[3]] = reg.coef_[0][3]
    betadf.at[ind, cue_stim_cond[4]] = reg.coef_[0][4]
    betadf.at[ind, cue_stim_cond[5]] = reg.coef_[0][5]
    betadf.at[ind, cue_stim_cond[6]] = reg.coef_[0][6]
    betadf.at[ind, cue_stim_cond[7]] = reg.coef_[0][7]
    betadf.at[ind, 'intercept'] = reg.intercept_[0]
    betadf['modelfit'] = modelfit
    # visualizing model fit results ____________________________________________
    # convolve onset boxcars, multiply it with model fitted coefficients
    total_regressor = []
    boxcar = []

    for cond in ['high_cue', 'low_cue']:
        signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['param_cue_type'] == cond].index.values
        event_start_time = np.array(js['event_cue']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_cue']['stop'])[cond_index]/samplingrate
        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            signal[start_index:stop_index] = stim_dict[cond]
        # Convolve the signal with the scr
        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)


    for cond in cond_list:
        signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['condition'] == cond].index.values
        event_start_time = np.array(js['event_stimuli']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_stimuli']['stop'])[cond_index]/samplingrate
        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            signal[start_index:stop_index] = stim_dict[cond]
        # Convolve the signal with the scr
        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)

# convolve
    scr_normalized = scr/np.sum(scr)
    predicted_total_signal = []

    for ind, cond in enumerate(['high_cue', 'low_cue']):
        print(ind)
        predicted_signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['param_cue_type'] == cond].index.values
        event_start_time = np.array(js['event_cue']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_cue']['stop'])[cond_index]/samplingrate

        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            predicted_signal[start_index:stop_index] = stim_dict[cond] * reg.coef_[0][ind]

        predicted_convolved_signal = convolve(predicted_signal, scr_normalized, 
                                              mode='full')[:len(predicted_signal)]
        predicted_total_signal.append(predicted_convolved_signal)
    for ind, cond in enumerate(cond_list):
        print(ind)
        predicted_signal  = np.zeros(len(winsor_physio)) 
        cond_index = metadf.loc[metadf['condition'] == cond].index.values
        event_start_time = np.array(js['event_stimuli']['start'])[cond_index]/samplingrate
        event_stop_time = np.array(js['event_stimuli']['stop'])[cond_index]/samplingrate

        for start, stop in zip(event_start_time, event_stop_time):
            start_index = int((start + shift_time) * data_points_per_second)
            stop_index = int((stop + shift_time) * data_points_per_second)
            predicted_signal[start_index:stop_index] = stim_dict[cond] * reg.coef_[0][ind]

        predicted_convolved_signal = convolve(predicted_signal, scr_normalized, 
                                              mode='full')[:len(predicted_signal)]
        predicted_total_signal.append(predicted_convolved_signal)

    # PLOT2 convolved signal ____________________________________________________


    # boxcar_summed = np.sum(np.stack(boxcar), axis=0) * np.mean(reg.coef_[0])

    predictedXmatrix = np.vstack(predicted_total_signal)
    y = winsor_physio
    boxheight = (np.max(winsor_physio) - np.min(winsor_physio)) * 0.25
    index = np.arange(len(y))
    x_seconds = index / 25
    color = {
        'high_cue': 'black', 
        'low_cue': 'gray',
        'high_stim-high_cue':'#DB2A04',
        'high_stim-low_cue': '#521240',
        'med_stim-high_cue': '#A33B10',
        'med_stim-low_cue': '#E3A833',
        'low_stim-high_cue': '#3343AD',
        'low_stim-low_cue': '#5DA5F8'}
    plt.figure(figsize=(10, 5))  # Set the figure size as desired
    for cond_ind, cond_name in enumerate(cue_stim_cond):
        plt.plot(x_seconds, predictedXmatrix[cond_ind].T, color[cond_name],
                 linestyle=(0, (1, 1)), linewidth=2, label=list(color.keys())[cond_ind])
    plt.plot(x_seconds, y, '#2F2f2f', label='SCR signal',alpha=1, linewidth=.5) #signal
    y_min, y_max = plt.ylim()
    boxcar_height = np.max(boxcar_summed)
    plt.fill_between(x_seconds,  
                 boxcar_summed*y_min, boxcar_summed *y_max,
                 edgecolor=None,
                 facecolor='#9f9f9f', alpha=0.3, label='heat onset')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("SCR amplitude (A.U.)", fontsize=14)
    plt.title(f"{sub} {ses} {run}\nSkin conductance GLM fit", fontsize=18)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    sns.despine()
    
    plt.savefig(join(save_dir, basename[:-4]+'_modelfitted.png'))
    plt.show()
    plt.close()

# extract metadata and save dataframe __________________________________________
betadf['sub']= betadf['filename'].str.extract(r'(sub-\d+)')
betadf['ses'] = betadf['filename'].str.extract(r'(ses-\d+)')
betadf['run'] = betadf['filename'].str.extract(r'(run-\d+)')
betadf['runtype'] = betadf['filename'].str.extract(r'runtype-(\w+)_')

betadf.to_csv(join(save_dir, f'glm-factorialcue_task-{task}_scr.tsv'), sep='\t')
# TODO: save metadata in json
json_fname = join(save_dir, f'glm-factorialcue_task-{task}_scr.json')

json_content = {"shift":3, 
 "samplingrate_of_onsettime": 2000, 
 "samplingrate_of_SCL": 25, 
 "TR": 0.46, 
 "source_code": "scripts/p03_glm/glm_factorial_scr.py",
 "regressor": "stimulus condition convolve w/ 1) canonical scr \
 2) boxcar for onset duration"}

with open(json_fname, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)



# %%
