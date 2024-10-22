#!/usr/bin/env python
# encoding: utf-8


# %%----------------------------------------------------------------------------
#                               libraries
# ------------------------------------------------------------------------------
import os
import glob
import re
import json
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import convolve
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import Winsorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.interpolate import interp1d
# %%----------------------------------------------------------------------------
#                               functions
# ------------------------------------------------------------------------------
def extract_meta(basename):
    """Extract subject, session, run, and runtype from file name."""
    sub_ind = int(re.search(r'sub-(\d+)', basename).group(1))
    ses_ind = int(re.search(r'ses-(\d+)', basename).group(1))
    run_ind = int(re.search(r'run-(\d+)', basename).group(1))
    runtype = re.search(r'runtype-(.*?)_', basename).group(1)
    return sub_ind, ses_ind, run_ind, runtype

def filter_good_data(filenames, baddata_df):
    """Filter out bad data from filenames based on a bad data DataFrame."""
    df = pd.DataFrame({'filename': filenames})
    df[['sub', 'ses', 'run']] = df['filename'].str.extract(r'sub-(\d+)_ses-(\d+)_run-(\d+)', expand=True).astype(int)
    merged = df.merge(baddata_df, on=['sub', 'ses', 'run'], how='left', indicator=True)
    return merged[merged['_merge'] != 'both'].drop('_merge', axis=1)

def winsorize_mad(data, threshold=3.5):
    """Winsorize data based on (MAD) method."""
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
    # wz = Winsorizer(capping_method='iqr', tail='both', fold=threshold)
    return wz.fit_transform(data)


def interpolate_data(data):
    """Interpolate missing data points (NaNs) linearly."""
    valid = ~np.isnan(data)
    interp_func = interp1d(np.arange(len(data))[valid], data[valid], kind='linear', fill_value="extrapolate")
    return interp_func(np.arange(len(data)))

def merge_qc_scl(qc_fname, scl_flist):
    """Merge QC and SCL data based on subject, session, and run identifiers.
        Example:
    >>> qc_fname = 'path/to/QC_EDA_new.csv'
    >>> scl_flist = ['file_sub-01_ses-1_run-1_runtype-task1_.csv', 'file_sub-02_ses-2_run-2_runtype-task2_.csv']
    >>> merged_df = merge_qc_scl(qc_fname, scl_flist)
    >>> print(merged_df)
    """
    qc = pd.read_csv(qc_fname)
    qc = qc.loc[qc['Signal quality'] == 'include', ['src_subject_id', 'session_id', 'param_run_num', 'param_task_name']]
    qc.columns = ['sub', 'ses', 'run', 'task']
    
    scl_file = pd.DataFrame({'filename': scl_flist})
    scl_file[['sub', 'ses', 'run', 'task']] = scl_file['filename'].str.extract(r'sub-(\d+)_ses-(\d+)_run-(\d+)_runtype-(\w+)_')
    scl_file = scl_file.astype({'sub': int, 'ses': int, 'run': int})
    
    return pd.merge(scl_file, qc, on=['sub', 'ses', 'run', 'task'], how='inner')


def boxcar_function(x, start, end):
    """Return 1 if x is between start and end, otherwise return 0."""
    return 1 if start <= x <= end else 0

def adjust_baseline(data, baseline):
    """Adjust data based on a provided baseline value."""
    return data - baseline if baseline > 0 else data + abs(baseline)

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath) as json_file:
        return json.load(json_file)
    
    
# %%----------------------------------------------------------------------------
#                               parameters
# ------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10  # You can also set a default font size if desired

# Directories and file paths
physio_dir = Path('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_physio')
cue_dir = Path('/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue')
scl_dir = cue_dir / 'analysis/physio/nobaseline/physio01_SCL'
save_dir = cue_dir / 'analysis/physio/nobaseline/glm_singletrial'
qc_fname = physio_dir / 'data/QC_EDA_new.csv'

# Load QC data
qc = pd.read_csv(qc_fname)
TR = 0.46
task = 'pain'

# Get list of SCL files
scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*{task}_epochstart--3_epochend-20_baselinecorrect-False_samplingrate-25_physio-eda.txt'), 
                             recursive=True))

# create empty dataframe _______________________________________________________
df_column = ['filename', 'sub', 'ses', 'run', 'runtype', 'intercept', 'beta', 'singletrial_name', 'singletrial_index']
all_dfs = []
Path(save_dir).mkdir(parents=True, exist_ok=True)

# merge dataframe

merged_df = merge_qc_scl(qc_fname, scl_flist)
filtered_list = list(merged_df.filename)

# %%----------------------------------------------------------------------------
#                               glm estimation
# ------------------------------------------------------------------------------

for ind, scl_fpath in enumerate(sorted(filtered_list)):
    
    basename = os.path.basename(scl_fpath)
    sub_ind, ses_ind, run_ind, runtype = extract_meta(basename)
    json_fname = f"sub-{sub_ind:04d}_ses-{ses_ind:02d}_run-{run_ind:02d}_runtype-{runtype}_samplingrate-2000_onset.json"
    
    samplingrate = int(re.search(r'samplingrate-(\d+)_', json_fname).group(1))
    pdf = pd.read_csv(scl_fpath, sep='\t', header=None)
    js = load_json(join(os.path.dirname(scl_fpath), json_fname))
   

    # # Winsorize data and adjust baseline _____________________________________
    winsor_mad = winsorize_mad(pdf, threshold=5)

    # baseline correction using ITIs ___________________________________________
    iti_intervals = [] #[(0, js['event_cue']['start'][0])]

    # For subsequent intervals, use the previous 'event_actualrating' 'stop' time to the current 'event_cue' 'start' time
    for i in range(1, len(js['event_cue']['start'])):
        iti_interval = (js['event_actualrating']['stop'][i-1], js['event_cue']['start'][i])
        iti_intervals.append(iti_interval)

    winsor_scaled = winsor_mad/ np.nanstd(winsor_mad)

    averages = []
    for start, stop in iti_intervals:
        # Assuming the index is directly comparable to the start and stop times
        filtered_values = winsor_scaled.loc[start/25:(stop-1)/25] if stop > start else pd.Series(dtype='float64')
        averages.append(np.nanmean(filtered_values))
    baseline_value = np.nanmean(averages)

    winsor_physio = adjust_baseline(winsor_scaled, baseline_value)


    # fetch SCR curve __________________________________________________________
    pspm_scr = pd.read_csv(Path(physio_dir, 'scripts', 'p03_glm', 'pspm-scrf_td-25.txt'), sep='\t')
    scr = pspm_scr.squeeze()


    # extract metadata _________________________________________________________
    sub = f"sub-{sub_ind:04d}"
    ses = f"ses-{ses_ind:02d}"
    run = f"run-{run_ind:02d}"
    meta_fname = glob.glob(join(scl_dir, sub, ses, basename.split('epochend')[0] + "*scltimecourse.csv"))[0]
    metadf = pd.read_csv(meta_fname)
    metadf['condition'] = metadf['param_stimulus_type'].astype(str) + '-' + metadf['param_cue_type'].astype(str) 

    betadf = pd.DataFrame(index=range(len(metadf)), columns=df_column)# + trial_list)
    betadf[ 'filename'] = basename
    # cond_type ________________________________________________________________
    total_runlength_sec = 400; data_points_per_second = 25
    shift_time = 0
    array_length = total_runlength_sec * data_points_per_second
    signal = np.zeros(len(winsor_physio)) #np.zeros(total_runlength_sec * data_points_per_second)


    # convolve onsets with boxcar and canonical SCR ____________________________
    #       UPDATE 02/27/2024. 
    #       We're creating boxcar function based on the stimulus duration
    #       prior to this update, only the event onset was convolved with the SCR function. 
    #       in other words, the convolved SCR was just a blip.
    total_regressor = []
    boxcar = []
    condition_name_list = []


    for trial_index in range(len(metadf)):  # Iterate over each trial
        trial_str = f"trial-{trial_index:03d}"  # Format trial index
        cue_type = metadf.loc[trial_index, 'param_cue_type']  # Get the cue type for the trial
        stim_type = metadf.loc[trial_index, 'param_stimulus_type']  # Get the stimulus type for the trial
        condition_name = f"epoch-stim_{trial_str}_cue-{cue_type.replace('_cue', '')}_stim-{stim_type.replace('_stim', '')}"

        signal  = np.zeros(len(winsor_physio)) 
        event_start_time = np.round(np.array(js['event_stimuli']['start'])[trial_index]/samplingrate)
        event_stop_time = np.round(np.array(js['event_stimuli']['stop'])[trial_index]/samplingrate)

        start_index = int((event_start_time + shift_time) * data_points_per_second)
        stop_index = int((event_stop_time + shift_time) * data_points_per_second)
        signal[start_index:stop_index] = 1 

        # Convolve the signal with the scr
        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)        
        condition_name_list.append(condition_name)




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

    for cond_ind in np.arange(len(metadf)):
        plt.plot(x_seconds, normalized_Xmatrix[cond_ind].T)

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
    trial_list = [f"trial-{i:03d}" for i in range(1, len(metadf)+1)]
    for beta_ind in range(X_r.shape[-1]):
        betadf.at[beta_ind, 'beta'] = reg.coef_[0][beta_ind]
        betadf.at[beta_ind, 'singletrial_index'] = trial_list[beta_ind]
    betadf['singletrial_name'] = condition_name_list
    betadf.at[beta_ind, 'intercept'] = reg.intercept_[0]
    betadf['modelfit'] = modelfit
    # visualizing model fit results ____________________________________________
    # convolve onset boxcars, multiply it with model fitted coefficients
    total_regressor = []
    boxcar = []

# TODO: 10/21/2024. Adapt code
    for trial_index in range(len(metadf)): 
    # for cond in ['high_cue', 'low_cue']:
        signal  = np.zeros(len(winsor_physio)) 
        trial_str = f"trial-{trial_index:03d}"
        cue_type = metadf.loc[trial_index, 'param_cue_type']
        stim_type = metadf.loc[trial_index, 'param_stimulus_type']
        condition_name = f"epoch-stim_{trial_str}_cue-{cue_type.replace('_cue', '')}_stim-{stim_type.replace('_stim', '')}"
        # cond_index = metadf.loc[metadf['param_cue_type'] == cond].index.values
        event_start_time = np.round(np.array(js['event_stimuli']['start'])[trial_index]/samplingrate)
        event_stop_time = np.round(np.array(js['event_stimuli']['stop'])[trial_index]/samplingrate)

        start_index = int((event_start_time + shift_time) * data_points_per_second)
        stop_index = int((event_stop_time + shift_time) * data_points_per_second)
        signal[start_index:stop_index] = 1 

        convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]
        total_regressor.append(convolved_signal)
        boxcar.append(signal)



# convolve
    scr_normalized = scr/np.sum(scr)
    predicted_total_signal = []

    # cue trials _______________________________________________________________
    # for trial_index in range(len(metadf)): 
    #     print(trial_index)
    #     predicted_signal  = np.zeros(len(winsor_physio)) 

    #     event_start_time = np.round(np.array(js['event_cue']['start'])[trial_index]/samplingrate)
    #     event_stop_time = np.round(np.array(js['event_cue']['stop'])[trial_index]/samplingrate)

    #     start_index = int((event_start_time + shift_time) * data_points_per_second)
    #     stop_index = int((event_stop_time + shift_time) * data_points_per_second)

    #     predicted_signal[start_index:stop_index] = 1 * reg.coef_[0][trial_index]

    #     predicted_convolved_signal = convolve(predicted_signal, scr_normalized, 
    #                                           mode='full')[:len(predicted_signal)]
    #     predicted_total_signal.append(predicted_convolved_signal)
    # stim trials _______________________________________________________________
    for trial_index in range(len(metadf)): 
        print(trial_index)
        predicted_signal  = np.zeros(len(winsor_physio)) 

        event_start_time = np.round(np.array(js['event_stimuli']['start'])[trial_index]/samplingrate)
        event_stop_time = np.round(np.array(js['event_stimuli']['stop'])[trial_index]/samplingrate)
        start_index = int((event_start_time + shift_time) * data_points_per_second)
        stop_index = int((event_stop_time + shift_time) * data_points_per_second)

        predicted_signal[start_index:stop_index] = 1 * reg.coef_[0][trial_index]
        predicted_convolved_signal = convolve(predicted_signal, scr_normalized, 
                                              mode='full')[:len(predicted_signal)]
        predicted_total_signal.append(predicted_convolved_signal)

    # PLOT2 convolved signal ____________________________________________________
    predictedXmatrix = np.vstack(predicted_total_signal)
    y = winsor_physio
    boxheight = (np.max(winsor_physio) - np.min(winsor_physio)) * 0.25
    index = np.arange(len(y))
    x_seconds = index / 25
    plt.figure(figsize=(10, 5))  # Set the figure size as desired
    for beta_ind in range(len(predictedXmatrix)):#cond_ind, cond_name in enumerate(trial_list):
        plt.plot(x_seconds, predictedXmatrix[beta_ind].T, 
                 linestyle=(0, (1, 1)), linewidth=2) #, label=list(color.keys())[cond_ind])
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
    sns.despine()
    plt.savefig(join(save_dir, basename[:-4]+'_modelfitted.png'))
    plt.show()
    plt.close()

    # TODO
    # I need to append this betadf to a new betadf
    all_dfs.append(betadf)

# extract metadata and save dataframe __________________________________________
final_df = pd.concat(all_dfs, ignore_index=True)

final_df['sub']= final_df['filename'].str.extract(r'(sub-\d+)')
final_df['ses'] = final_df['filename'].str.extract(r'(ses-\d+)')
final_df['run'] = final_df['filename'].str.extract(r'(run-\d+)')
final_df['runtype'] = final_df['filename'].str.extract(r'runtype-(\w+)_')

final_df.to_csv(join(save_dir, f'glm-singletrial_task-{task}_scr.tsv'), sep='\t')
# TODO: save metadata in json
json_fname = join(save_dir, f'glm-singletrial_task-{task}_scr.json')

json_content = {"shift":3, 
 "samplingrate_of_onsettime": 2000, 
 "samplingrate_of_SCL": 25, 
 "TR": 0.46, 
 "source_code": "scripts/p03_glm/glm_singletrial.py",
 "regressor": "stimulus condition convolve w/ 1) canonical scr \
 2) boxcar for onset duration"}

with open(json_fname, 'w') as json_file:
    json.dump(json_content, json_file, indent=4)



# %%
