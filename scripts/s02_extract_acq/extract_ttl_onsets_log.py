#!/usr/bin/env python3
"""
# split runs with "fMRI trigger" e.g. binarize the values and identify start stop indices
# ttl onsets
#   convert each event into trial number
# do this based on two events: expect and actual

# TODO:
# * identify BIDS scheme for physio data
# * flag files without ANISO
# * flag files with less than 5 runs
# ** for those runs, we need to manually assign run numbers (biopac will collect back to back)
# * change main_dir directory when running on discovery
# TODO: create metadata, of folders and how columns were calculated
# * remove unnecessary print statements
TODO (4/9):
don't run code for already successful files. how do I indicate this?
what's the best way to keep track of successful completes vs debug required files/
#
"""

# %% libraries ________________________
import neurokit2 as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os, glob, shutil, datetime
from pathlib import Path
import json
import re
import logging

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


def _binarize_channel(data, origin_col, new_col, threshold, binary_h,
                      binary_l):
    """
    data: pandas dataframe. acquisition file
    origin_col: columns with raw signal
    new_col: new columns for saving binarized origin_col values in
    threshold: int. threshold for binarizing values within pandas column
    binary_h, binary_l: two numbers. e.g. 5, 0 or 1, 0

    """
    data.loc[data[origin_col] > threshold, new_col] = binary_h
    data.loc[data[origin_col] <= threshold, new_col] = binary_l


def _identifyshifts(data, col_name, shift_ind=1):
    """
    detect transitions in a pandas dataframe (when values switch from 0 to 1; 1 to 0)
    data: dataframe (pandas)
    col_name: (str) name of column within data
    shift_ind: how many rows do you allow for detecting transitions? default = 1
    return: two list of indices where a transition is detected
    1) the beginning of the transition
    2) the end of the transition
    """
    shift_start = data[data[col_name] > data[col_name].shift(shift_ind)].index
    shift_stop = data[data[col_name] < data[col_name].shift(shift_ind)].index
    return shift_start, shift_stop


# %% directories ___________________________________
# current_dir = os.getcwd()
# main_dir = Path(current_dir).parents[1]

# %% temporary
main_dir = '/Volumes/spacetop'
print(main_dir)
save_dir = os.path.join(main_dir, 'biopac', 'dartmouth', 'b03_extract_ttl')
print(save_dir)
# %% filename __________________________
# filename ='/Users/h/Dropbox/projects_dropbox/spacetop_biopac/data/sub-0026/SOCIAL_spacetop_sub-0026_ses-01_task-social_ANISO.acq'
acq_list = glob.glob(os.path.join(main_dir, 'biopac', 'dartmouth',
                                  'b02_sorted', 'sub-' + ('[0-9]' * 4), '*',
                                  '*task-social*_physio.acq'),
                     recursive=True)
flaglist = []
runmeta = pd.read_csv(
    '/Users/h/Dropbox/projects_dropbox/social_influence_analysis/data/spacetop_task-social_run-metadata.csv'
)

# %% logger parameters __________________________________________________
txt_filename = os.path.join(
    save_dir, f'biopac_flaglist_{datetime.date.today().isoformat()}.txt')

formatter = logging.Formatter('%(levelname)s - %(message)s')
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
# create formatter and add it to the handler

# TODO: skip sub-0001, sub-0003, sub-0004, sub-0005

# %%
# new = sorted(acq_list)[160:]
# for ind, acq in enumerate(new):  #sorted(acq_list):
for ind, acq in enumerate(sorted(acq_list)):  #sorted(acq_list):
    acq_fname = os.path.basename(acq)
    sub = [match for match in acq_fname.split('_') if "sub" in match][0]
    ses = [match for match in acq_fname.split('_')
           if "ses" in match][0]  # 'ses-03'
    task = [match for match in acq_fname.split('_') if "task" in match][0]
    logger.info(f"\n\n__________________{sub} {ses} {task}__________________")

    # [ checkpoint 1 ] Can you load the data acq file? ____________________________________
    try:
        spacetop_data, spacetop_samplingrate = nk.read_acqknowledge(acq)
        logger.info(f"\t[ checkpoint 1 ] can you load data acq.? SUCCESS")
    except KeyError:
        logger.error(
            f"\t[ checkpoint 1 ] can you load data acq.? failed to read acquisition file: {acq}"
        )
        continue

    # ____________________________________ 1. identify run transitions ____________________________________
    # 1-1) _____________ binarize MR triggers (872 TRs per run) _____________
    # [ checkpoint 2 ]  Does .acq file have MR TR aniso data?  ____________________________________
    try:
        spacetop_data['mr_aniso'] = spacetop_data[
            'fMRI Trigger - CBLCFMA - Current Feedba'].rolling(
                window=3).mean()
        logger.info(
            f"\t[ checkpoint 2 ] Does .acq file have MR TR aniso data? SUCCESS"
        )
    except KeyError:
        logger.error(
            f"\t[ checkpoint 2 ] Does .acq file have MR TR aniso data? file may not have fmri aniso data"
        )
        continue

    try:
        _binarize_channel(spacetop_data,
                          origin_col='mr_aniso',
                          new_col='spike',
                          threshold=40,
                          binary_h=5,
                          binary_l=0)
    except ValueError:
        logger.error(f"\tfile may not have any acquired physiodata")
        continue
    start_spike, stop_spike = _identifyshifts(spacetop_data, 'spike')
    # create new column that has MR trigger pulse signal - bin_spike
    spacetop_data['bin_spike'] = 0
    spacetop_data.loc[start_spike, 'bin_spike'] = 5
    # 1-2) _____________ binarize runs (6 runs in total, less than or more than dependeing on data acquisition) _____________
    spacetop_data['mr_aniso_boxcar'] = spacetop_data[
        'fMRI Trigger - CBLCFMA - Current Feedba'].rolling(window=2000).mean()
    mid_val = (np.max(spacetop_data['mr_aniso_boxcar']) -
               np.min(spacetop_data['mr_aniso_boxcar'])) / 4
    _binarize_channel(spacetop_data,
                      origin_col='mr_aniso_boxcar',
                      new_col='mr_boxcar',
                      threshold=mid_val,
                      binary_h=5,
                      binary_l=0)
    start_df, stop_df = _identifyshifts(spacetop_data, 'mr_boxcar')

    logger.info(f"\t* start_df: {start_df}")
    logger.info(f"\t* stop_df: {stop_df}")
    logger.info(f"\t* number of run transitions: {len(start_df)}")

    # 1-3) adjust one TR (remove it!)_____________
    sdf = spacetop_data.copy()
    sdf.loc[start_df, 'bin_spike'] = 0
    nstart_df, nstop_df = _identifyshifts(sdf, 'bin_spike')

    sdf['adjusted_boxcar'] = sdf['bin_spike'].rolling(window=2000).mean()
    mid_val = (np.max(sdf['adjusted_boxcar']) -
               np.min(sdf['adjusted_boxcar'])) / 4
    _binarize_channel(sdf,
                      origin_col='adjusted_boxcar',
                      new_col='adjust_run',
                      threshold=mid_val,
                      binary_h=5,
                      binary_l=0)
    astart_df, astop_df = _identifyshifts(sdf, 'adjust_run')
    # [ checkpoint 3 ] Can you identify number of run transitions?  ____________________________________
    try:
        len(astart_df) == len(astop_df)
        logger.info(
            f"[ checkpoint 3 ] Can you identify number of run transitions? SUCCESS"
        )
        logger.info(f"\t* adjusted start_df: {astart_df}")
        logger.info(f"\t* adjusted stop_df: {astop_df}")
        # TODO:
        # remove blips from turning off the TSA2.
        # need to grab the runmeta information earlier
        logger.info(f"\t* durations (start - stop_df): {astop_df - astart_df}")
    except:
        logger.error(
            f"[ checkpoint 3 ] Can you identify number of run transitions? length of start onsets and stop onsets differ - please investigate"
        )
        continue

    # ____________________________________ 2. identify ttl events based on TTL column ____________________________________
    sdf['TTL'] = sdf['TSA2 TTL - CBLCFMA - Current Feedback M'].rolling(
        window=2000).mean()
    sdf.loc[sdf['TTL'] > 5, 'ttl_aniso'] = 5
    sdf.loc[sdf['TTL'] <= 5, 'ttl_aniso'] = 0

    # %% EV stimuli ::
    mid_val = (np.max(sdf['administer']) - np.min(sdf['administer'])) / 2
    # sdf.loc[sdf['administer'] > mid_val, 'stimuli'] = 5
    # sdf.loc[sdf['administer'] <= mid_val, 'stimuli'] = 0
    _binarize_channel(sdf,
                      origin_col='administer',
                      new_col='stimuli',
                      threshold=mid_val,
                      binary_h=5,
                      binary_l=0)

    df_transition = pd.DataFrame({'start_df': astart_df, 'stop_df': astop_df})

    # ____________________________________ 3. remove runs if shorter than 300 s ____________________________________
    # TODO: if run smaller than 300s, drop and remove from pandas
    # or even better, cross check if the run is incomplete but use
    # POP item from start_df, stop_df
    # spacetop_data.at[start_df[r]:stop_df[r], 'run_num'] = r+1
    list_astart = list(astart_df)
    list_astop = list(astop_df)
    for r in range(len(astart_df)):
        if (astop_df[r] - astart_df[r]) / 2000 < 300:
            sdf.drop(sdf.index[astart_df[r]:astop_df[r]], axis=0, inplace=True)
            list_astart.pop(r)
            list_astop.pop(r)

    # # ____________________________________ 4. identify runs with TTL signal ____________________________________
    ttl_bool = []
    acq_runs_with_ttl = []
    new_meta_run_with_ttl = []
    for r in range(len(list_astart)):
        bool_val = np.unique(
            sdf.iloc[list_astart[r]:list_astop[r],
                     # df_transition.start_df[r]:df_transition.stop_df[r],
                     sdf.columns.get_loc('ttl_aniso')]).any()
        ttl_bool.append(bool_val)
        sdf.at[list_astart[r]:list_astop[r], 'run_num'] = r + 1
    if not any(ttl_bool):
        logging.warning(
            f"[ checkpoint 4 ] Can you identify ttl events based on ttl column? >>  no runs with TTL in this {acq} file"
        )
        continue

    acq_runs_with_ttl = [i for i, x in enumerate(ttl_bool) if x]
    # print(f"acq_runs_with_ttl: {acq_runs_with_ttl}")
    # logger.info(f"acq_runs_with_ttl stop_df: run-{int(acq_runs_with_ttl) + 1}")

    # TODO: check if runs_with_ttl matches the index from the ./social_influence-analysis/data/spacetop_task-social_run-metadata.csv
    # TODO: if run is NaN in runmeta, drop it from ttl_list_from_meta\
    # [ checkpoint 4 ] Can you identify ttl events based on ttl column?  ____________________________________
    # try:
    #     acq_runs_with_ttl
    # except:
    #     logger.debug("\t[ checkpoint 4 ] Can you identify ttl events based on ttl column? >> no runs with TTL")
    #     continue

    run_list_crosscheck = []
    a = runmeta.loc[(runmeta['sub'] == sub) & (runmeta['ses'] == ses)]
    for r in acq_runs_with_ttl:
        run_key = a[f"run-{r+1:02d}"].values[0]
        run_list_crosscheck.append(run_key)
    run_mask = pd.isnull(run_list_crosscheck)
    update_ttl_runs = [
        d for (d, remove) in zip(acq_runs_with_ttl, run_mask) if not remove
    ]

    ttl_list_from_meta = a.columns[a.eq('pain').any()]
    meta_runs_with_ttl = [
        int(re.findall('\d+', s)[0]) for s in list(ttl_list_from_meta)
    ]
    new_meta_run_with_ttl[:] = [m - 1 for m in meta_runs_with_ttl]

    if any(run_mask):  # if there's any run with nan in metadata
        logger.warning(
            "\t[ checkpoint 5-2 ] match metadata? >> SUCCESS. NaNs in metadata - exclude runs from acquisition file"
        )
    elif ~any(
            run_mask
    ) and update_ttl_runs == new_meta_run_with_ttl:  # no nans. # but no TTLs
        logger.info(
            f"\t[ checkpoint 5-1 ] match metadata? >> SUCCESS. No NaNs. pain runs in metadata and pain runs in .acq MATCH"
        )
    elif ~any(run_mask) and update_ttl_runs != new_meta_run_with_ttl:
        logger.error(
            '\t[ checkpoint 5 ] match metadata? mismatch of .acq & metadata .csv'
        )
        logger.error(f"\tcheck {acq}")
        logger.error(f"\tacq_runs_with_ttl: {acq_runs_with_ttl}")
        logger.error(f"\tnew_meta_run_with_ttl: {new_meta_run_with_ttl}")
        continue

    # a = runmeta.loc[(runmeta['sub'] == sub) & (runmeta['ses'] == ses)]
    # ttl_list_from_meta = a.columns[a.eq('pain').any()]
    # meta_runs_with_ttl = [
    #     int(re.findall('\d+', s)[0]) for s in list(ttl_list_from_meta)
    # ]
    # new_meta_run_with_ttl[:] = [m - 1 for m in meta_runs_with_ttl]

    # print(f"new_meta_run_with_ttl: {new_meta_run_with_ttl}")
    # logger.info(f"new_meta_run_with_ttl: {new_meta_run_with_ttl}")
    # ____________________________________ identify TTL signals and trials ____________________________________
    # run_len = len(df_transition)
    # if run_len == 6:

    # print(f"runs with ttl: {acq_runs_with_ttl}", file = f)
    for i, run_num in enumerate(update_ttl_runs):
        run = f"run-{run_num + 1:02d}"
        logging.info(f"__________________ pain run {run} __________________")
        run_subset = sdf.loc[sdf['run_num'] == run_num + 1]
        # print(len(run_subset) / spacetop_samplingrate)
        try:
            300 < len(run_subset) / spacetop_samplingrate < 450
            logger.info(
                f"\t[ checkpoint 6 ] Does each run last for 398s? >> YES, in between 300 ~ 450 seconds."
            )
        except:
            logger.warning(
                f'\t[ checkpoint 6 ] Does each run last for 398s? >> NO, run length is shorter than 380s or longer than 410s'
            )
            continue

        run_df = run_subset.reset_index()
        # identify events :: expect and actual _________________
        # identify events :: stimulli _________________
        # try:
        start_expect, stop_expect = _identifyshifts(run_df, 'expect')
        start_actual, stop_actual = _identifyshifts(run_df, 'actual')
        start_stim, stop_stim = _identifyshifts(run_df, 'stimuli')

        try:
            not start_expect.empty and not start_actual.empty
            logger.info(
                f"[ checkpoint 7 ] Do we get transitions for events: expect actual stimulus? >> YES"
            )
            logger.info(f"type of start_stim: {type(start_stim)}")
            logger.info(f"start_stim: {start_stim}")
            # continue
        except:
            # print("continue")
            logger.critical(
                f"[ checkpoint 7 ] Do we get transitions for events: expect actual stimulus? >> NO, can't identify transitions for start_expect/start_actual/start_stim"
            )
            continue

        events = nk.events_create(
            event_onsets=list(start_stim),
            event_durations=list(
                (stop_stim - start_stim) / spacetop_samplingrate))

        # transform events :: transform to onset _________________
        expect_start = list(start_expect) / spacetop_samplingrate
        actual_end = list(stop_actual) / spacetop_samplingrate
        stim_start = list(start_stim) / spacetop_samplingrate
        stim_end = list(stop_stim) / spacetop_samplingrate
        stim_onset = events['onset'] / spacetop_samplingrate

        # build pandas dataframe _________________
        df_onset = pd.DataFrame({
            'expect_start': expect_start,
            'actual_end': actual_end,
            'stim_start': np.nan,
            'stim_end': np.nan
        })

        df_stim = pd.DataFrame({
            'stim_start': stim_start,
            'stim_end': stim_end
        })
        # ____________________________________ identify boundary conditions and assign TTL event to specific trial ____________________________________
        # based on information of "expect, actual" events, we will assign a trial number to stimulus events
        # RESOURCE: https://stackoverflow.com/questions/62300474/filter-all-rows-in-a-pandas-dataframe-where-a-given-value-is-between-two-columnv
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
            # print(
            # f"this is the {i}-th iteration. stim value is {start_val}, and is in between index {interval_idx}"
            # )
        start_ttl, stop_ttl = _identifyshifts(run_df, 'ttl_aniso')
        # start_ttl = run_df[run_df['ttl_aniso'] > run_df['ttl_aniso'].shift(1)]
        # stop_ttl = run_df[run_df['ttl_aniso'] < run_df['ttl_aniso'].shift(1)]
        # TODO:
        # if start_ttl and stop_ttl mismatch, then add the last onset to stop_ttl
        if len(start_ttl) > len(stop_ttl):
            # last_row = run_df.iloc[-1]
            stop_ttl = stop_ttl.values
            stop_ttl = np.append(stop_ttl, run_df.index.values[-1])
        else:
            pass
            # stop_ttl.append(run_df.index.values[-1], axis=1)

        ttl_onsets = list(start_ttl +
                          (stop_ttl - start_ttl) / 2) / spacetop_samplingrate
        # print(f"ttl onsets: {ttl_onsets}, length of ttl onset is : {len(ttl_onsets)}")
        # define empty TTL data frame
        df_ttl = pd.DataFrame(np.nan,
                              index=np.arange(len(df_onset)),
                              columns=['ttl_1', 'ttl_2', 'ttl_3', 'ttl_4'])
        # identify which set of TTLs fall between expect and actual
        pad = 4  # seconds. you may increase the value to have a bigger event search interval
        df_onset['expect_start_interval'] = df_onset['expect_start']
        df_onset['actual_end_interval'] = df_onset['actual_end'] + pad
        adjusted = df_onset['actual_end_interval']
        try:
            adjusted.iloc[-1, :] = len(run_subset) / 2000
        except IndexError:
            logger.critical(
                f"[ checkpoint 7 ] can't identify transitions for start_expect/start_actual/start_stim"
            )

            continue

        a_idx = pd.IntervalIndex.from_arrays(df_onset['expect_start_interval'],
                                             adjusted)
        for i in range(len(ttl_onsets)):
            try:
                val = ttl_onsets[i]
                # print(f"{i}-th value: {val}")
                empty_cols = []
                interval_idx = df_onset[a_idx.contains(val)].index[0]

                # print(f"\t\t* interval index: {interval_idx}")
                mask = df_ttl.loc[[interval_idx]].isnull()
                empty_cols = list(
                    itertools.compress(np.array(df_ttl.columns.to_list()),
                                       mask.values[0]))
                # print(f"\t\t* empty columns: {empty_cols}")
                df_ttl.loc[df_ttl.index[interval_idx],
                           str(empty_cols[0])] = val  #
            except IndexError:
                logger.debug(
                    f"\tIndexError: index 0 is out of bounds for axis 0 with size 0"
                )
                continue
            # print(
            # f"\t\t* this is the row where the value -- {val} -- falls. on the {interval_idx}-th row"
            # )
        logger.info(f"\t* successfully assigned TTL to dataframe")
        fdf = pd.merge(df_onset, df_ttl, left_index=True, right_index=True)
        fdf['ttl_r1'] = fdf['ttl_1'] - fdf['stim_start']
        fdf['ttl_r2'] = fdf['ttl_2'] - fdf['stim_start']
        fdf['ttl_r3'] = fdf['ttl_3'] - fdf['stim_start']
        fdf['ttl_r4'] = fdf['ttl_4'] - fdf['stim_start']
        fdf['plateau_dur'] = fdf['ttl_r3'] - fdf['ttl_r2']
        save_filename = f"{sub}_{ses}_{task}_{run}_physio-ttl.csv"
        new_dir = os.path.join(save_dir, task, sub, ses)

        Path(new_dir).mkdir(parents=True, exist_ok=True)
        fdf.reset_index(inplace=True)
        fdf = fdf.rename(columns={'index': 'trial_num'})
        fdf.to_csv(os.path.join(new_dir, save_filename), index=False)
        logger.info(f'\t[SUCCESS] :: saved to {new_dir} for {run}')
