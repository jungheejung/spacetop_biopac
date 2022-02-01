# lesson
# 1. think about how to sort files, what steps need to come first
# 1. os path
# 2. glob
# purpose
# acq files - haphazardly saved on biopac pc
# sort and reorder based on BIDS scheme

# TODO: # if file doesn't match convention
# flag it 
# 1) correct file naming convention - lives in correct folder > skip
# 2) correct file naming conventino - lives in INcorrect folder
# 3) incorrect file naming convention - > dump into text file and fix it ourselves
# 4) number of files - dictionries - for every task, and session
# %% libraries ___________________________________
# from msilib.schema import Directory
import os 
import glob
from pathlib import Path
import shutil
import re

# %% directories ___________________________________
current_dir = os.getcwd()
main_dir = Path(current_dir).parents[1]
print(main_dir)

# %% Heejung Jan 19
# glob anything below data Directory 
# grab basename 
# extract keywords
acq_list = glob.glob(os.path.join(main_dir, 'sandbox/**', '*.acq'), recursive = True)
for acq in acq_list:
    filename  = os.path.basename(acq)
    sub = [match for match in filename.split('_') if "sub" in match][0] # 'sub-0056'
    ses = [match for match in filename.split('_') if "ses" in match][0] # 'ses-03'
    task = [match for match in filename.split('_') if "task" in match][0]
    # r_sub = re.compile(".*sub")
    # sub_string = list(filter(r_sub.match, filename.split("_")))
    # print(sub_string)
    print(sub, ses, task)
    new_dir = os.path.join(main_dir, 'data', sub, ses)
    Path(new_dir).mkdir( parents=True, exist_ok=True )
    shutil.move(acq, new_dir)

# %% EXAMPLE 3. almost works, but one caveat - relying on the order of the "split" does not help. 
# hint FRACTIONAL_POSNER, FRACTIONAL_SPUNT _____________________________________________________________________________________
acq_list = glob.glob(os.path.join(main_dir, 'sandbox/**', '*.acq'), recursive = True)
for acq in acq_list:
    filename  = os.path.basename(acq)
    sub = filename.split('_')[2] # 'sub-0056'
    ses = filename.split('_')[3] # 'ses-03'
    task = filename.split('_')[4]
    print(sub, ses, task)
    new_dir = os.path.join(main_dir, 'data', sub, ses)
    Path(new_dir).mkdir( parents=True, exist_ok=True )
    shutil.move(acq, new_dir)
# %% EXAMPLE 2. get list of files _____________________________________________________________________________________
files = glob.glob(os.path.join(main_dir, 'sandbox', 'sub-00*','ses-*', '*.acq'))
for file in files:
    filename  = os.path.basename(file) # 'ALIGNVIDEOS_spacetop_sub-0056_ses-03_task-alignvideos_ANISO.acq'
    sub = filename.split('_')[2] # 'sub-0056'
    ses = filename.split('_')[3] # 'ses-03'
    task = filename.split('_')[4]
    print(sub, ses, task)
    # move accordingly
    # make a directory, if session folder doesn't exist
    new_dir = os.path.join(main_dir, 'sandbox', sub, ses)
    Path(new_dir).mkdir( parents=True, exist_ok=True )
# after making direrctory, use shutil. to move

# %% EXAMPLE 1. _____________________________________________________________________________________
example_file = '/Users/h/Dropbox/projects_dropbox/spacetop_biopac/sandbox/sub-0056/ALIGNVIDEOS_spacetop_sub-0056_ses-01_task-alignvideos_ANISO.acq'
filename  = os.path.basename(example_file) # 'ALIGNVIDEOS_spacetop_sub-0056_ses-03_task-alignvideos_ANISO.acq'
sub = filename.split('_')[2] # 'sub-0056'
ses = filename.split('_')[3] # 'ses-03'
task = filename.split('_')[4]
print(sub, ses, task)
# move accordingly
# %%make a directory, if session folder doesn't exist
new_dir = os.path.join(main_dir, 'sandbox', sub, ses)
print(new_dir, sub, ses)


#%%
Path(new_dir).mkdir( parents=True, exist_ok=True )
# %%
shutil.move(example_file, new_dir)
# %%
