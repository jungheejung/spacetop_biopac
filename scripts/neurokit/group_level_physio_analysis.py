## assuming that we have all of the run-bids formatted acq.
# load data
# [x] TODO: append task info (PVC) from metadata
# [x] TODO: move biopac final data into fmriprep: spacetop_data/data/sub/ses/physio
# [x] TODO: move behavioral data preprocessed into fmriprep:  spacetop_data/data/sub/ses/beh
# [x] TODO:allow to skip broken files
# [x] TODO: allow to skip completed files
# baseline correct
# filter signal
# extract mean signals
# export as .csv file
# extract timeseries signal
# export as .csv file

# 
# %% libraries _________________________________________________________________________________________
from tkinter import Variable
import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob, sys
from pathlib import Path
from os.path import join
import itertools
from statistics import mean
import logging
from datetime import datetime

<<<<<<< HEAD
cluster = sys.argv[1] 
slurm_ind = int(sys.argv[2]) 
print(f"slurm_ind: {slurm_ind}")
=======
cluster = sys.argv[1]
slurm_ind = sys.argv[2]
>>>>>>> e5cc8a0a0c9df33cfcbcc78944763b104e8dbadb
pwd = os.getcwd()
main_dir = Path(pwd).parents[1]

if cluster == 'discovery':
    main_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_biopac/'
else:
    main_dir = '/Users/h/Dropbox/projects_dropbox/spacetop_biopac'

sys.path.append(os.path.join(main_dir, 'scripts'))
sys.path.insert(0, os.path.join(main_dir, 'scripts'))
print(sys.path)
import utils
from utils import preprocess

__author__ = "Heejung Jung"
__copyright__ = "Spatial Topology Project"
__credits__ = [
    "Heejung"
]  # people who reported bug fixes, made suggestions, etc. but did not actually write the code.
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Heejung Jung"
__email__ = "heejung.jung@colorado.edu"
__status__ = "Development"

plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size'] = 14

# %% set parameters
pwd = os.getcwd()
main_dir = pwd
flaglist = []

if cluster == 'discovery':
<<<<<<< HEAD
    biopac_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop/biopac/dartmouth/b04_finalbids/task-social' #'/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir =  '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
=======
    biopac_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/physio/physio01_raw'  #'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/beh/d02_preproc-beh'  # '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
>>>>>>> e5cc8a0a0c9df33cfcbcc78944763b104e8dbadb
    cuestudy_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social'
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
else:
    #biopac_dir = '/Volumes/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    biopac_dir = '/Volumes/spacetop/biopac/dartmouth/b04_finalbids/task-social'
    beh_dir = '/Volumes/spacetop_projects_social/data/beh/d02_preproc-beh'  # '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
    cuestudy_dir = '/Volumes/spacetop_projects_social'
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
sub_list = []
biopac_list = next(os.walk(biopac_dir))[1]
remove_int = [1, 2, 3, 4, 5, 6]
#remove_int = list(np.arange(78))
remove_list = [f"sub-{x:04d}" for x in remove_int]
include_int = list(np.arange(slurm_ind * 10 + 1, (slurm_ind + 1) * 10, 1))
include_list = [f"sub-{x:04d}" for x in include_int]
sub_list = [x for x in biopac_list if x not in remove_list]
sub_list = [x for x in sub_list if x in include_list]
ses_list = [1, 3, 4]
run_list = [1, 2, 3, 4, 5, 6]
sub_ses = list(itertools.product(sorted(sub_list), ses_list, run_list))

date = datetime.now().strftime("%m-%d-%Y")
txt_filename = os.path.join(log_dir, f"s02-biopac_flaglist_{date}.txt")

#if os.path.exists(txt_filename):
#    os.remove(txt_filename)

formatter = logging.Formatter("%(levelname)s - %(message)s")
handler = logging.FileHandler(txt_filename)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logging.getLogger().addHandler(handler)
logging.getLogger().addHandler(ch)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
print(f"starting logger and for loop")


def _extract_bids(fname):
    entities = dict(
<<<<<<< HEAD
    match.split('-', 1)
    for match in fname.split('_')
    if '-' in match
    )    
=======
        match.split('-', 1) for match in fname.split('_') if '-' in match)
>>>>>>> e5cc8a0a0c9df33cfcbcc78944763b104e8dbadb
    sub_num = int(entities['sub'])
    ses_num = int(entities['ses'])
    if 'run' in entities['run'].split('-'):
        run_list = entities['run'].split('-')
        run_list.remove('run')
        run_num = run_list[0]
        task_type = run_list[-1]
    else:
        run_num = int(entities['run'].split('-')[0])
        run_type = entities['run'].split('-')[-1]
    return sub_num, ses_num, run_num, task_type
# %%____________________________________________________________________________________________________
flag = []
for i, (sub, ses_ind, run_ind) in enumerate(sub_ses):

    # NOTE: open physio dataframe (check if exists) __________________________________________________________
    try:
        ses = f"ses-{ses_ind:02d}"
        run = f"run-{run_ind:02d}"
        logger.info(
            f"\n\n__________________{sub} {ses} {run}__________________")

        physio_flist = glob.glob(
            join(
                biopac_dir, sub, ses,
                f"{sub}_{ses}_task-social_*{run}*_recording-ppg-eda_physio.acq"
            ))
        physio_fpath = physio_flist[0]
        #string = 'Python is great and Java is also great'
        sub_num, ses_num, run_num, task_type = _extract_bids(os.path.basename(physio_fpath))
    except IndexError:
        logger.error(f"\tmissing physio file - {sub} {ses} {run} DOES NOT exist")
        print("no biopac file exists")
        continue



    # NOTE: identify physio file for corresponding sub/ses/run _______________________________________________
    physio_fname = os.path.basename(physio_fpath)
    logger.info({physio_fname})
    task = [match for match in physio_fname.split('_') if "task" in match][0]
    # DEP: physio_df, spacetop_samplingrate = nk.read_acqknowledge(physio_fpath) output file is pandas
    physio_df = pd.read_csv(physio_fpath)
    spacetop_samplingrate = 2000


    # NOTE: identify behavioral file for corresponding sub/ses/run ____________________________________
    beh_fpath = glob.glob(
        join(beh_dir, sub, ses,
             f"{sub}_{ses}_task-social_{run}*_beh.csv"))
    try:  # # if len(beh_fpath) != 0:
        beh_fname = beh_fpath[0]
    except IndexError:
        logger.error(
            "missing behavioral file: {sub} {ses} {run} DOES NOT exist")
        continue
    beh_df = pd.read_csv(beh_fname)
    task_type = ([
        match for match in os.path.basename(beh_fname).split('_')
        if "run" in match
    ][0]).split('-')[2]
    print(f"{sub} {ses} {run} {task_type} ____________________")
    metadata_df = beh_df[[
        'src_subject_id', 'session_id', 'param_task_name', 'param_run_num',
        'param_cue_type', 'param_stimulus_type', 'param_cond_type'
    ]]

    # NOTE: merge fixation columns (some files look different) handle slight changes in biopac dataframe
    if 'fixation-01' in physio_df.columns:
        physio_df[
            'fixation'] = physio_df['fixation-01'] + physio_df['fixation-02']
    # NOTE: baseline correct _________________________________________________________________________________
    # 1) extract fixations:
    fix_bool = physio_df['fixation'].astype(bool).sum()
    print(
        f"confirming the number of fixation non-szero timepoints: {fix_bool}")
    print(f"this amounts to {fix_bool/spacetop_samplingrate} seconds")

    # baseline correction method 1: use the first 6 dummy scan period
    baseline_method01 = physio_df['Skin Conductance (EDA) - EDA100C-MRI'].loc[
        0:5520].mean()
    physio_df['EDA_corrected_01tr'] = physio_df[
        'Skin Conductance (EDA) - EDA100C-MRI'] - baseline_method01

    # baseline correction method 02: use the fixation period from the entire run
    mask = physio_df['fixation'].astype(bool)
    baseline_method02 = physio_df['Skin Conductance (EDA) - EDA100C-MRI'].loc[
        mask].mean()
    physio_df['EDA_corrected_02fixation'] = physio_df[
        'Skin Conductance (EDA) - EDA100C-MRI'] - baseline_method02

    # TODO method 03: per trial, grab the baseline fixation average signal

    print(f"baseline using the 6 TR: {baseline_method01}")
    print(f"baseline using fixation from entire run: {baseline_method02}")

    # NOTE: extract epochs ___________________________________________________________________________________
    # extract epochs :: cue
    utils.preprocess._binarize_channel(physio_df,
                                       source_col='cue',
                                       new_col='cue',
                                       threshold=None,
                                       binary_high=5,
                                       binary_low=0)
    dict_cue = utils.preprocess._identify_boundary(physio_df, 'cue')
    print(f"* total number of trials: {len(dict_cue['start'])}")

    # extract epochs :: expect rating
    utils.preprocess._binarize_channel(physio_df,
                                       source_col='expect',
                                       new_col='expectrating',
                                       threshold=None,
                                       binary_high=5,
                                       binary_low=0)
    dict_expectrating = utils.preprocess._identify_boundary(
        physio_df, 'expectrating')
    print(f"* total number of trials: {len(dict_expectrating['start'])}")

    # extract epochs :: stimulus delivery
    event_key = 'stimuli'
    utils.preprocess._binarize_channel(physio_df,
                                       source_col='administer',
                                       new_col=event_key,
                                       threshold=None,
                                       binary_high=5,
                                       binary_low=0)
    dict_stimuli = utils.preprocess._identify_boundary(physio_df, event_key)
    physio_df[physio_df[event_key].diff() != 0].index
    print(
        f"* total number of {event_key} trials: {len(dict_stimuli['start'])}")

    # extract epochs :: actual rating
    utils.preprocess._binarize_channel(physio_df,
                                       source_col='actual',
                                       new_col='actualrating',
                                       threshold=None,
                                       binary_high=5,
                                       binary_low=0)
    dict_actualrating = utils.preprocess._identify_boundary(
        physio_df, 'actualrating')
    print(f"* total number of trials: {len(dict_actualrating['start'])}")

    # NOTE: TTL extraction ___________________________________________________________________________________
    if task_type == 'pain':
        final_df = pd.DataFrame()
        # binarize TTL channels (raise error if channel has no TTL, despite being a pain run)
        try:
            utils.preprocess._binarize_channel(
                physio_df,
                source_col='TSA2 TTL - CBLCFMA - Current Feedback M',
                new_col='ttl',
                threshold=None,
                binary_high=5,
                binary_low=0)
        except:
            logger.error(
                f"this pain run doesn't have any TTLs {sub} {ses} {run}")
            continue

        dict_ttl = utils.preprocess._identify_boundary(physio_df, 'ttl')
        ttl_onsets = list(
            np.array(dict_ttl['start']) +
            (np.array(dict_ttl['stop']) - np.array(dict_ttl['start'])) / 2)
        print(
            f"ttl onsets: {ttl_onsets}, length of ttl onset is : {len(ttl_onsets)}"
        )

        # create onset dataframe template ______________________________________________________________
        df_onset = pd.DataFrame({
            'expect_start': dict_expectrating['start'],
            'actual_end': dict_actualrating['stop'],
            'stim_start': np.nan,
            'stim_end': np.nan
        })

        df_stim = pd.DataFrame({
            'stim_start': dict_stimuli['start'],
            'stim_end': dict_stimuli['stop']
        })
        for i in range(len(df_stim)):
            idx = pd.IntervalIndex.from_arrays(df_onset['expect_start'],
                                               df_onset['actual_end'])
            start_val = df_stim.iloc[i][df_stim.columns.get_loc('stim_start')]
            interval_idx = df_onset[idx.contains(start_val)].index[0]
            df_onset.iloc[interval_idx,
                          df_onset.columns.get_loc('stim_start')] = start_val

            end_val = df_stim.iloc[i][df_stim.columns.get_loc('stim_end')]
            interval_idx = df_onset[idx.contains(end_val)].index[0]
            df_onset.iloc[interval_idx,
                          df_onset.columns.get_loc('stim_end')] = end_val
            print(
                f"this is the {i}-th iteration. stim value is {start_val}, and is in between index {interval_idx}"
            )

        # define empty TTL data frame
        df_ttl = pd.DataFrame(np.nan,
                              index=np.arange(len(df_onset)),
                              columns=['ttl_1', 'ttl_2', 'ttl_3', 'ttl_4'])

        # identify which set of TTLs fall between expect and actual
        pad = 1  # seconds. you may increase the value to have a bigger event search interval
        df_onset['expect_start_interval'] = df_onset['expect_start'] - pad
        df_onset['actual_end_interval'] = df_onset['actual_end'] + pad
        idx = pd.IntervalIndex.from_arrays(df_onset['expect_start_interval'],
                                           df_onset['actual_end_interval'])

        for i in range(len(ttl_onsets)):

            val = ttl_onsets[i]
            print(f"{i}-th value: {val}")
            empty_cols = []
            try:
                interval_idx = df_onset[idx.contains(val)].index
                if len(interval_idx) == 0:
                    trim = val - spacetop_samplingrate * 2
                    interval_idx = df_onset[idx.contains(trim)].index
                    logger.info(
                        f"this TTL does not belong to any event boundary")
                # try:
                #     trim = val+spacetop_samplingrate*2
                #     interval_idx = df_onset[idx.contains(trim)].index
                #     flaglist.append(f"this TTL does not belong to any event boundary")
                # else:
                #     trim = val+spacetop_samplingrate*2
                #     interval_idx = df_onset[idx.contains(trim)].index
                #     flaglist.append(f"this TTL does not belong to any event boundary")
                interval_idx = interval_idx[0]
                print(f"\t\t* interval index: {interval_idx}")
            except:
                print(f"this TTL does not belong to any event boundary")
                logger.error(
                    f"this TTL does not belong to any event boundary")
                continue
            mask = df_ttl.loc[[interval_idx]].isnull()
            empty_cols = list(
                itertools.compress(np.array(df_ttl.columns.to_list()),
                                   mask.values[0]))
            print(f"\t\t* empty columns: {empty_cols}")
            df_ttl.loc[df_ttl.index[interval_idx], str(empty_cols[0])] = val
            print(
                f"\t\t* this is the row where the value -- {val} -- falls. on the {interval_idx}-th row"
            )

        # merge :: merge df_onset and df_ttl -> final output: final df
        final_df = pd.merge(df_onset,
                            df_ttl,
                            left_index=True,
                            right_index=True)
        final_df['ttl_r1'] = final_df['ttl_1'] - final_df['stim_start']
        final_df['ttl_r2'] = final_df['ttl_2'] - final_df['stim_start']
        final_df['ttl_r3'] = final_df['ttl_3'] - final_df['stim_start']
        final_df['ttl_r4'] = final_df['ttl_4'] - final_df['stim_start']

        ttl2 = final_df['ttl_2'].values.tolist()
        plateau_start = np.ceil(ttl2).astype(pd.Int64Dtype)
        # TODO: before we merge the data, we have to figure out a way to remove the nans
        # [x] identify row with nan in ttl2 column
        # [x] for plateau remove items with that index
        any_nans = np.argwhere(np.isnan(ttl2)).tolist()
        flat_nans = [item for sublist in any_nans for item in sublist]
        for ind in flat_nans:
            plateau_start = np.delete(plateau_start, ind)
        metadata_df.drop(flat_nans, axis=0, inplace=True)
        metadata_df['trail_num'] = metadata_df.index + 1

        # create a dictionary for neurokit. this will serve as the events
        event_stimuli = {
            'onset': np.array(plateau_start).astype(pd.Int64Dtype),
            'duration': np.repeat(spacetop_samplingrate * 5, 12),
            'label': np.array(np.arange(12)),
            'condition': beh_df['param_stimulus_type'].values.tolist()
        }
        # TODO: interim plot to check if TTL matches with signals
        run_physio = physio_df[[
            'EDA_corrected_02fixation', 'Pulse (PPG) - PPG100C', 'ttl'
        ]]
        # run_physio
        stim_plot = nk.events_plot(event_stimuli, run_physio)
    else:
        event_stimuli = {
            'onset': np.array(dict_stimuli['start']),
            'duration': np.repeat(spacetop_samplingrate * 5, 12),
            'label': np.array(np.arange(12), dtype='<U21'),
            'condition': beh_df['param_stimulus_type'].values.tolist()
        }
        # TODO: plot the ttl and visulize the alignment
        # interim plot to check if TTL matches with signals
        run_physio = physio_df[[
            'EDA_corrected_02fixation', 'Pulse (PPG) - PPG100C', 'stimuli'
        ]]
        #run_physio
        stim_plot = nk.events_plot(event_stimuli, run_physio)

    # NOTE: neurokit analysis :+: HIGHLIGHT :+: filter signal ________________________________________________

    # IF you want to use raw signal
    # eda_signal = nk.signal_sanitize(run_physio["Skin Conductance (EDA) - EDA100C-MRI"])
    # eda_raw_plot = plt.plot(run_df["Skin Conductance (EDA) - EDA100C-MRI"])
    # NOTE: PHASIC_____________________________________________________________________________
    amp_min = 0.01
    scr_signal = nk.signal_sanitize(
        physio_df['Skin Conductance (EDA) - EDA100C-MRI'])
    scr_filters = nk.signal_filter(scr_signal,
                                   sampling_rate=spacetop_samplingrate,
                                   highcut=1,
                                   method="butterworth",
                                   order=2)  # ISABEL: Detrend
    scr_detrend = nk.signal_detrend(scr_filters)

    scr_decomposed = nk.eda_phasic(nk.standardize(scr_detrend),
                                   sampling_rate=spacetop_samplingrate)

    scr_peaks, info = nk.eda_peaks(scr_decomposed["EDA_Phasic"].values,
                                   sampling_rate=spacetop_samplingrate,
                                   method="neurokit",
                                   amplitude_min=amp_min)
    scr_signals = pd.DataFrame({
        "EDA_Raw": scr_signal,
        "EDA_Clean": scr_filters
    })
    scr_processed = pd.concat([scr_signals, scr_decomposed, scr_peaks], axis=1)
    try:
        scr_epochs = nk.epochs_create(scr_processed,
                                      event_stimuli,
                                      sampling_rate=spacetop_samplingrate,
                                      epochs_start=0,
                                      epochs_end=5,
                                      baseline_correction=True)  #
    except:
        print("has NANS in the datafram")
        continue
    scr_phasic = nk.eda_eventrelated(scr_epochs)

    # NOTE:  TONIC ________________________________________________________________________________
    scl_signal = nk.signal_sanitize(physio_df['EDA_corrected_02fixation'])
    scl_filters = nk.signal_filter(scl_signal,
                                   sampling_rate=spacetop_samplingrate,
                                   highcut=1,
                                   method="butterworth",
                                   order=2)  # ISABEL: Detrend
    scl_detrend = nk.signal_detrend(scl_filters)
    scl_decomposed = nk.eda_phasic(nk.standardize(scl_detrend),
                                   sampling_rate=spacetop_samplingrate)
    scl_signals = pd.DataFrame({
        "EDA_Raw": scl_signal,
        "EDA_Clean": scl_filters
    })
    scl_processed = pd.concat([scl_signals, scl_decomposed['EDA_Tonic']],
                              axis=1)
    try:
        scl_epoch = nk.epochs_create(scl_processed['EDA_Tonic'],
                                     event_stimuli,
                                     sampling_rate=spacetop_samplingrate,
                                     epochs_start=-1,
                                     epochs_end=8,
                                     baseline_correction=False)
    except:
        print("has NANS in the datafram")
        continue

    #  NOTE: concatenate dataframes __________________________________________________________________________
    bio_df = pd.concat([
        physio_df[[
            'trigger', 'fixation', 'cue', 'expect', 'administer', 'actual',
            'ttl'
        ]], scr_processed
    ],
                       axis=1)
    fig_save_dir = join(cuestudy_dir, 'data', 'physio', 'qc', sub, ses)
    Path(fig_save_dir).mkdir(parents=True, exist_ok=True)

    fig_savename = f"{sub}_{ses}_{run}-{task_type}_physio-scr-scl.png"
    processed_fig = nk.events_plot(
        event_stimuli,
        bio_df[['administer', 'EDA_Tonic', 'EDA_Phasic', 'SCR_Peaks']])
    plt.show()
    # Tonic level ______________________________________________________________________________________
    
    # 1. append columns to the begining (trial order, trial type)
    # NOTE: eda_epochs_level -> scl_epoch
    metadata_tonic = pd.DataFrame(
        index=list(range(len(scl_epoch))),
        columns=['trial_order', 'iv_stim', 'mean_signal'])
    try:
        for ind in range(len(scl_epoch)):
            metadata_tonic.iloc[
                ind, metadata_tonic.columns.
                get_loc('mean_signal')] = scl_epoch[ind]["Signal"].mean()
            metadata_tonic.iloc[
                ind, metadata_tonic.columns.
                get_loc('trial_order')] = scl_epoch[ind]['Label'].unique()[0]
            metadata_tonic.iloc[
                ind, metadata_tonic.columns.
                get_loc('iv_stim')] = scl_epoch[ind]["Condition"].unique()[0]
    except:
        for ind in range(len(scl_epoch)):
            metadata_tonic.iloc[
                ind,
                metadata_tonic.columns.get_loc('mean_signal')] = scl_epoch[str(
                    ind)]["Signal"].mean()
            metadata_tonic.iloc[
                ind,
                metadata_tonic.columns.get_loc('trial_order')] = scl_epoch[str(
                    ind)]['Label'].unique()[0]
            metadata_tonic.iloc[
                ind, metadata_tonic.columns.get_loc('iv_stim')] = scl_epoch[
                    str(ind)]["Condition"].unique()[0]
    # 2. eda_level_timecourse
    eda_level_timecourse = pd.DataFrame(
        index=list(range(len(scl_epoch))),
        columns=['time_' + str(col) for col in list(np.arange(18000))])
    try:
        for ind in range(len(scl_epoch)):
            eda_level_timecourse.iloc[
                ind, :] = scl_epoch[str(ind)]['Signal'].to_numpy().reshape(
                    1, 18000
                )  
    except:
        for ind in range(len(scl_epoch)):
            eda_level_timecourse.iloc[
                ind, :] = scl_epoch[ind]['Signal'].to_numpy().reshape(
                    1, 18000
                )  


    tonic_df = pd.concat([metadata_df, metadata_tonic], axis=1)
    tonic_timecourse = pd.concat(
        [metadata_df, metadata_tonic, eda_level_timecourse], axis=1)
    # NOTE: save tonic data __________________________________________________________________________________
    save_dir = join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', sub,
                    ses)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tonic_fname = f"{sub}_{ses}_{run}-{task_type}_epochstart--1_epochend-8_physio-scl.csv"
    tonictime_fname = f"{sub}_{ses}_{run}-{task_type}_epochstart--1_epochend-8_physio-scltimecourse.csv"
    tonic_df.to_csv(join(save_dir, tonic_fname))
    tonic_timecourse.to_csv(join(save_dir, tonictime_fname))

    # NOTE: save phasic data _________________________________________________________________________________
    metadata_df = metadata_df.reset_index(drop=True)
    scr_phasic = scr_phasic.reset_index(drop=True)
    phasic_meta_df = pd.concat(
        [metadata_df, scr_phasic], axis=1
    )  
    phasic_fname = f"{sub}_{ses}_{run}-{task_type}_epochstart-0_epochend-5_physio-scr.csv"
    phasic_meta_df.to_csv(join(save_dir, phasic_fname))
    print(f"{sub}_{ses}_{run}-{task_type} finished")
    #plt.clf()

<<<<<<< HEAD
    print(f"complete {sub} {ses} {run}")
# %%
=======

>>>>>>> e5cc8a0a0c9df33cfcbcc78944763b104e8dbadb
