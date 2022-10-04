# %%
# import neurokit2 as nk
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
import json
# %%
# biopac_dir = '/Volumes/spacetop_projects_social/data/physio/01_raw-physio'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
# beh_dir =  '/Volumes/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
# cuestudy_dir = '/Volumes/spacetop_projects_social' 
# log_dir = join(cuestudy_dir, "scripts", "logcenter")

discovery=1
if discovery:
    biopac_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir =  '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
    cuestudy_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_social'
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
else:
    biopac_dir = '/Volumes/spacetop_projects_social/data/physio/physio01_raw'#'/Volumes/spacetop/biopac/dartmouth/b04_finalbids/'
    beh_dir =  '/Volumes/spacetop_projects_social/data/beh/d02_preproc-beh'# '/Volumes/spacetop_projects_social/data/d02_preproc-beh'
    cuestudy_dir = '/Volumes/spacetop_projects_social' 
    log_dir = join(cuestudy_dir, "scripts", "logcenter")
# glob files
save_dir = join(cuestudy_dir, 'data', 'physio', 'physio02_preproc')
# sub-0026_ses-04_run-06-pain_epochstart--1_epochend-8_physio-scl
tonic_flist = glob.glob(join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', '*', '*', 
f"*_epochstart--1_epochend-8_physio-scl.csv"))
# tonic_flist = glob.glob(join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', '*', '*', 
# f"*_epochstart--1_epochend-8_physio-edatonic.csv"))

tonic_group_df = pd.DataFrame()
for tonic_fpath in tonic_flist:
    tonic_df = pd.read_csv(tonic_fpath)
    tonic_group_df = pd.concat([tonic_group_df, tonic_df])
# tonic_group_df.to_csv(join(save_dir, f"group_epochstart--1_epochend-8_physio-edatonic.csv"))
tonic_group_df.to_csv(join(save_dir, f"group_epochstart--1_epochend-8_physio-scl.csv"))
with open(join(save_dir,'group_epochstart--1_epochend-8_physio-scl.json'), 'w', encoding='utf-8') as f:
    json.dump(tonic_flist, f, ensure_ascii=False, indent=4)


# %% Phasic: ________________________________________________________________________________
# sub-0026_ses-04_run-06-pain_epochstart-0_epochend-5_physio-scr
phasic_flist = glob.glob(join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', '*', '*', 
f"*_epochstart-0_epochend-5_physio-scr.csv"))
phasic_group_df = pd.DataFrame()
for phasic_fpath in phasic_flist:
    phasic_df = pd.read_csv(phasic_fpath)
    phasic_group_df = pd.concat([phasic_group_df, phasic_df])
phasic_group_df.to_csv(join(save_dir, f"group_epochstart-0_epochend-5_physio-scr.csv"))
with open(join(save_dir,'group_epochstart-0_epochend-5_physio-scr.json'), 'w', encoding='utf-8') as f:
    json.dump(phasic_flist, f, ensure_ascii=False, indent=4)
# stack files
# savee filename in mettadata

# %% REpair mistake
phasic_flist = glob.glob(join(cuestudy_dir, 'data', 'physio', 'physio02_preproc', '*', '*', 
f"*_epochstart-0_epochend-5_physio-scr.csv"))
phasic_group_df = pd.DataFrame()
for phasic_fpath in phasic_flist:
    phasic_df = pd.read_csv(phasic_fpath)
    former = phasic_df[['Unnamed: 0', 'src_subject_id', 'session_id', 'param_task_name',
        'param_run_num', 'param_cue_type', 'param_stimulus_type',
        'param_cond_type']].iloc[0:12]

    latter = phasic_df[['Label', 'Condition', 'Event_Onset',
        'EDA_Peak_Amplitude', 'EDA_SCR', 'SCR_Peak_Amplitude',
        'SCR_Peak_Amplitude_Time', 'SCR_RiseTime', 'SCR_RecoveryTime']].iloc[12:25].reset_index(drop=True)

    phasic_patch = pd.concat([former, latter], axis = 1)
    phasic_group_df = pd.concat([phasic_group_df, phasic_patch], axis = 0)
phasic_group_df = phasic_group_df.loc[:, ~phasic_group_df.columns.str.contains('^Unnamed')]
phasic_group_df.to_csv(join(save_dir, f"group_epochstart-0_epochend-9_physio-phasic.csv"), index=False)
with open(join(save_dir,'group_epochstart-0_epochend-9_physio-phasic.json'), 'w', encoding='utf-8') as f:
    json.dump(phasic_flist, f, ensure_ascii=False, indent=4)