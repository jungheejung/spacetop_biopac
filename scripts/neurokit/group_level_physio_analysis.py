## assuming that we have all of the run-bids formatted acq.
# load data
# [x] TODO: append task info (PVC) from metadata
# [x] TODO: move biopac final data into fmriprep: spacetop_data/data/sub/ses/physio
# [x] TODO: move behavioral data preprocessed into fmriprep:  spacetop_data/data/sub/ses/beh
# [ ] TODO:allow to skip broken files
# [ ] TODO: allow to skip completed files
# baseline correct
# filter signal
# extract mean signals
# export as .csv file
# extract timeseries signal
# export as .csv file

# load data
# %%
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

cluster = sys.argv[1] 
slurm_ind = sys.argv[2] 
pwd = os.getcwd()
main_dir = Path(pwd).parents[1]
#discovery = 0
if cluster == 'discovery':
    main_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_biopac/'
else:
    main_dir = '/Users/h/Dropbox/projects_dropbox/spacetop_biopac'

sys.path.append(os.path.join(main_dir, 'scripts'))
sys.path.insert(0,os.path.join(main_dir, 'scripts'))
print(sys.path)

import utils
from utils import preprocess
__author__ = "Heejung Jung"
__copyright__ = "Spatial Topology Project"
__credits__ = ["Heejung"] # people who reported bug fixes, made suggestions, etc. but did not actually write the code.
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Heejung Jung"
__email__ = "heejung.jung@colorado.edu"
__status__ = "Development" 

plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size'] = 14

# %% set parameters
pwd = os.getcwd()
# sub-0051_ses-03_run-02
main_dir = pwd
# sub_num = 51; ses_num = 3; run_num = 2
# sub = f"sub-{sub_num:04d}";
# ses = f"ses-{ses_num:02d}"
# run = f"run-{run_num-1:02d}"
flaglist = []

if cluster == 'discovery':
    biopac_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir =  '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
    cuestudy_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social'
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
else:
    biopac_dir = '/Volumes/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir =  '/Volumes/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
    cuestudy_dir = '/Volumes/spacetop_projects_social' 
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
sub_list = []
biopac_list = next(os.walk(biopac_dir))[1]  
remove_int = [1,2,3,4,5,6]
#remove_int = list(np.arange(78))

remove_list = [f"sub-{x:04d}" for x in remove_int]
include_int = list(np.arange(slurm_ind*10+1,(slurm_ind+1)*10,1 ))
include_list = [f"sub-{x:04d}" for x in include_int]
sub_list = [x for x in biopac_list if x not in remove_list]
sub_list = [x for x in sub_list if x  in include_list]
#sub_list = ['sub-0029']
ses_list = [1,3,4]
run_list = [1,2,3,4,5,6]
sub_ses = list(itertools.product(sorted(sub_list), ses_list, run_list))

date = datetime.now().strftime("%m-%d-%Y")

txt_filename = os.path.join(
    log_dir, f"s02-biopac_flaglist_{date}.txt"
)

if os.path.exists(txt_filename):
    os.remove(txt_filename)


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
# %%
# beh_fname = glob.glob(join(main_dir, 'data', '*', '*', 'beh', f"*_task-social_*_beh.csv"))[0]
# sub-0005_ses-01_task-social_run-01_recording-ppg-eda_physio.acq
flag = []
for i, (sub, ses_ind, run_ind) in enumerate(sub_ses):

# check if biopac file exists
    try:
        #print(sub, ses_ind, run_ind)
        ses = f"ses-{ses_ind:02d}"
        run = f"run-{run_ind:02d}"
        logger.info(f"\n\n__________________{sub} {ses} {run}__________________")
        # biopac_flist = glob.glob(join(biopac_ttl_dir, sub, ses, "*ttl.csv"))
        physio_flist = glob.glob(join(biopac_dir, sub, ses, f"{sub}_{ses}_task-social_{run}*_recording-ppg-eda_physio.acq"))
        physio_fpath = physio_flist[0]
    except:
        logger.error(f"\tno biopac file exists")
        # with open(join(log_dir, "flag_{date}.txt"), "a") as logfile:
        #     traceback.print_exc(file=logfile)
        continue
    
    try:
        save_dir = join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', sub, ses)
        phasic_fname = f"{sub}_{ses}_{run}_epochstart-0_epochend-9_physio-phasictonic.csv"
        if not os.path.exists(join(save_dir, phasic_fname)):
            pass
    except:
        save_dir = join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', sub, ses)
        phasic_fname = f"{sub}_{ses}_{run}_epochstart-0_epochend-9_physio-phasictonic.csv"
        logger.warning(f"aborting: this job was complete for {sub}_{ses}_{run}")
        continue
# if output derivative already exists, skip loop:

# for physio_fpath in sorted(physio_flist):
    # physio_fpath = physio_flist[0]
    physio_fname = os.path.basename(physio_fpath)
    logger.info({physio_fname})

    task = [match for match in physio_fname.split('_') if "task" in match][0]


    # physio_df, spacetop_samplingrate = nk.read_acqknowledge(physio_fpath)
    physio_df = pd.read_csv(physio_fpath)
    spacetop_samplingrate = 2000
    beh_fpath = glob.glob(join(beh_dir, sub, ses, f"{sub}_{ses}_task-social_{run}*_beh.csv"))
    if len(beh_fpath) != 0:
        beh_fname = beh_fpath[0]
        beh_df = pd.read_csv(beh_fname)
        task_type = ([
            match for match in os.path.basename(beh_fname).split('_')
            if "run" in match
        ][0]).split('-')[2]
        print(f"{sub} {ses} {run} {task_type} ____________________")
        metadata_df = beh_df[['src_subject_id', 'session_id', 'param_task_name', 'param_run_num', 
        'param_cue_type', 'param_stimulus_type', 'param_cond_type' ]]

        # merge fixation columns (some files look different)
        if 'fixation-01' in physio_df.columns:
            physio_df['fixation'] = physio_df['fixation-01'] + physio_df['fixation-02'] 
        # baseline correct ________________________________________________________________________________
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
        #physio_df['EDA_corrected_01tr'].plot()
        #physio_df['EDA_corrected_02fixation'].plot()

        # extract epochs ________________________________________________________________________________
        # extract epochs :: cue
        utils.preprocess._binarize_channel(physio_df,
                                            source_col='cue',
                                            new_col='cue',
                                            threshold=None,
                                            binary_high=5,
                                            binary_low=0)
        dict_cue = utils.preprocess._identify_boundary(physio_df, 'cue')
        cue_freq = len(dict_cue['start'])
        print(f"* total number of trials: {cue_freq}")

        # extract epochs :: expect rating
        utils.preprocess._binarize_channel(physio_df,
                                            source_col='expect',
                                            new_col='expectrating',
                                            threshold=None,
                                            binary_high=5,
                                            binary_low=0)

        dict_expectrating = utils.preprocess._identify_boundary(
            physio_df, 'expectrating')
        expectrating_freq = len(dict_expectrating['start'])
        print(f"* total number of trials: {expectrating_freq}")

        # extract epochs :: stimulus delivery
        event_key = 'stimuli'
        utils.preprocess._binarize_channel(physio_df,
                                            source_col='administer',
                                            new_col= event_key,
                                            threshold=None,
                                            binary_high=5,
                                            binary_low=0)
        dict_stimuli = utils.preprocess._identify_boundary(physio_df, event_key)
        physio_df[physio_df[event_key].diff() != 0].index
        stim_freq = len(dict_stimuli['start'])
        print(f"* total number of {event_key} trials: {stim_freq}")

        # extract epochs :: actual rating
        utils.preprocess._binarize_channel(physio_df,
                                            source_col='actual',
                                            new_col='actualrating',
                                            threshold=None,
                                            binary_high=5,
                                            binary_low=0)
        dict_actualrating = utils.preprocess._identify_boundary(
            physio_df, 'actualrating')
        actualrating_freq = len(dict_actualrating['start'])
        print(f"* total number of trials: {actualrating_freq}")

        # TTL extraction ________________________________________________________________________________
        if task_type == 'pain':
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

            final_df = pd.DataFrame()
            if 'TTL' in physio_df:
                utils.preprocess._binarize_channel(
                    physio_df,
                    source_col='TSA2 TTL - CBLCFMA - Current Feedback M',
                    new_col='ttl',
                    threshold=None,
                    binary_high=5,
                    binary_low=0)
            else:
                flaglist.append(f"this pain run doesn't have any TTLs {sub} {ses} {run}")
                continue
            dict_ttl = utils.preprocess._identify_boundary(physio_df, 'ttl')

            ttl_onsets = list(dict_ttl['start'] +
                                (dict_ttl['stop'] - dict_ttl['start']) / 2)
            print(
                f"ttl onsets: {ttl_onsets}, length of ttl onset is : {len(ttl_onsets)}"
            )

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

            # calculate TTL onsets
            dict_ttl = utils.preprocess._identify_boundary(physio_df, 'ttl')

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
                interval_idx = df_onset[idx.contains(val)].index[0]
                print(f"\t\t* interval index: {interval_idx}")
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
            plateau_start = np.ceil(ttl2).astype(int)
            #plateau_start

        # create a dictionary for neurokit. this will serve as the events
        if task_type == 'pain':
            ttl2 = final_df['ttl_2'].values.tolist()
            plateau_start = np.ceil(ttl2).astype(int)
            event_stimuli = {
                'onset': np.array(plateau_start),
                'duration': np.repeat(spacetop_samplingrate * 5, 12),
                'label': np.array(np.arange(12)),
                'condition': beh_df['param_stimulus_type'].values.tolist()
            }
                # interim plot to check if TTL matches with signals
            run_physio = physio_df[[
                'EDA_corrected_02fixation', 'Pulse (PPG) - PPG100C', 'ttl'
            ]]
            #run_physio
            #plot = nk.events_plot(event_stimuli, run_physio)
        else:
            event_stimuli = {
                'onset': np.array(dict_stimuli['start']),
                'duration': np.repeat(spacetop_samplingrate * 5, 12),
                'label': np.array(np.arange(12), dtype='<U21'),
                'condition': beh_df['param_stimulus_type'].values.tolist()
            }
                # interim plot to check if TTL matches with signals
            run_physio = physio_df[[
                'EDA_corrected_02fixation', 'Pulse (PPG) - PPG100C', 'stimuli'
            ]]
            #run_physio
            #plot = nk.events_plot(event_stimuli, run_physio)



        # filter signal ________________________________________________________________________________

        # IF you want to use raw signal
        # eda_signal = nk.signal_sanitize(run_physio["Skin Conductance (EDA) - EDA100C-MRI"])
        # eda_raw_plot = plt.plot(run_df["Skin Conductance (EDA) - EDA100C-MRI"])

        #  USE baseline corrected signal
        eda_signal = nk.signal_sanitize(physio_df["EDA_corrected_02fixation"])
        eda_filters = nk.signal_filter(eda_signal,
                                    sampling_rate=spacetop_samplingrate,
                                    highcut=1,
                                    method="butterworth",
                                    order=2)

        #eda_raw_plot = plt.plot(physio_df["EDA_corrected_02fixation"])
        #eda_filters_plot = plt.plot(eda_filters)
        #plt.title('baseline_corrected vs. baseline_corrected + filtered signal')
        #plt.show()

        # 2)  decompose signla

        eda_decomposed = nk.eda_phasic(nk.standardize(eda_filters),
                                    sampling_rate = spacetop_samplingrate)
        #eda_decomposed_plot = eda_decomposed.plot()

        eda_peaks, info = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values,
                                    sampling_rate = spacetop_samplingrate,
                                    method = "neurokit",
                                    amplitude_min = 0.02)
        info["sampling_rate"] = spacetop_samplingrate

        signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_filters})
        eda_processed = pd.concat([signals, eda_decomposed, eda_peaks], axis=1)
        eda_level_signal = eda_processed["EDA_Tonic"]  # for skin conductance level

        # 3) signal type:
        ### * Interim: `eda_epoch` Define epochs for EDA signal
        # * eda_epochs: snipping out segments based on start of heat pain stimulus (plateau?) with eda_processed
        # * eda_epochs_BL: eda_epochs but with baseline correction (necessary?)
        # * eda_epochs_level: snipping out segments based on plateau of heat pain stimulus with tonic channel of eda_processed for skin conductance level
        # * eda_epochs_physioBL: snipping out segments based on trigger column (beginning of experiment) for extraction of physio baseline correction


        # TODO: %% eda_epochs_level is the same as eda_epochs_tonic_decomposed. Is there a difference? or is this a matter of being copied over?
        eda_epochs_BL = nk.epochs_create(eda_processed, 
                                        event_stimuli, 
                                        sampling_rate=spacetop_samplingrate, 
                                        epochs_start=0, 
                                        epochs_end=9,
                                        baseline_correction=False)

        # tonic component
        eda_epochs_level = nk.epochs_create(eda_level_signal, 
                                            event_stimuli, 
                                            sampling_rate=spacetop_samplingrate, 
                                            epochs_start=-1, 
                                            epochs_end=8,
                                            baseline_correction=False)

        eda_phasic_BL = nk.eda_eventrelated(eda_epochs_BL)

        # TODO: save plot and later QC with an RA

        eda_tonic_BL = nk.eda_intervalrelated(eda_epochs_BL)

        #plot_eda_phasic = nk.events_plot(event_stimuli, 
        #                                eda_processed[["EDA_Tonic", "EDA_Phasic"]])


        #  concatenate dataframes ____________________________________________________________
        bio_df = pd.concat([physio_df[['trigger', 'fixation', 'cue', 'expect', 'administer', 'actual']],eda_processed], axis =1) 
        fig_save_dir = join(cuestudy_dir, 'data', 'physio', 'qc', sub, ses)
        Path(fig_save_dir).mkdir( parents=True, exist_ok=True )

        fig_savename = f"{sub}_{ses}_{run}_physio-edatonic-edaphasic.png"
        processed_fig = nk.events_plot(event_stimuli, 
                                        bio_df[['administer', 'EDA_Tonic', 'EDA_Phasic', 'SCR_Peaks' ]])
        #@suppress
        #fig = processed_fig[0].get_figure()
        #processed_fig.savefig(join(fig_save_dir, fig_savename))
        #processed_fig.show()
        #plt.close()
        
        #eda_processed.plot(subplots = True)


        # Tonic level ________________________________________________________________________________
        # 1. append columns to the begining (trial order, trial type)
        metadata_tonic = pd.DataFrame(index = list(range(len(eda_epochs_level))),
                                columns=['trial_order', 'iv_stim', 'mean_signal'])         
        for ind in range(len(eda_epochs_level)):
            metadata_tonic.iloc[ind, metadata_tonic.columns.get_loc('mean_signal')] = eda_epochs_level[str(ind)]["Signal"].mean()
            metadata_tonic.iloc[ind, metadata_tonic.columns.get_loc('trial_order')] = eda_epochs_level[str(ind)]['Label'].unique()[0]
            metadata_tonic.iloc[ind, metadata_tonic.columns.get_loc('iv_stim')] = eda_epochs_level[str(ind)]["Condition"].unique()[0]
        # 2. eda_level_timecourse
        eda_level_timecourse = pd.DataFrame(index = list(range(len(eda_epochs_level))),
                                columns= ['time_' + str(col) for col in list(np.arange(18000))])         
        for ind in range(len(eda_epochs_level)):
            eda_level_timecourse.iloc[ind,:] = eda_epochs_level[str(ind)]['Signal'].to_numpy().reshape(1,18000)# eda_timecourse.reset_index(drop=True, inplace=True)
        tonic_df = pd.concat([metadata_tonic,eda_level_timecourse ], axis = 1)
        tonic_meta_df = pd.concat([metadata_df, tonic_df], axis = 1)
        # 
        save_dir = join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', sub, ses)
        Path(save_dir).mkdir( parents=True, exist_ok=True )
        tonic_fname = f"{sub}_{ses}_{run}_epochstart--1_epochend-8_physio-edatonic.csv"
        tonic_meta_df.to_csv(join(save_dir, tonic_fname))


        #  Phasic: ________________________________________________________________________________
        phasic_meta_df =  pd.concat([metadata_df, eda_phasic_BL],axis = 1)
        phasic_fname = f"{sub}_{ses}_{run}_epochstart-0_epochend-9_physio-phasictonic.csv"
        phasic_meta_df.to_csv(join(save_dir, phasic_fname))
        print(f"{sub}_{ses}_{run} finished")
        plt.clf()


    else:
        flaglist.append(f"{sub} {ses} {run}")
    
    




    print(f"complete {sub} {ses} {run}")
# %%
