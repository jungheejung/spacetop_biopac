
# %%
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
import neurokit2 as nk

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


# prototype
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
save_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/glm/factorial'
scl_dir = '/Volumes/spacetop_projects_cue/analysis/physio/physio01_SCL' #'/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL'                                            sub-0015_ses-01_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
save_dir = '/Volumes/spacetop_projects_cue/analysis/physio/glm/factorial'
qc_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/QC_EDA_new.csv'
qc = pd.read_csv(qc_fname)
TR = 0.46
task = 'pain'

scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*sub-0017_ses-03_run-02*_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'), recursive=True))

# %%======= NOTE: create empty dataframe
df_column = ['filename', 'sub', 'ses', 'run', 'runtype', 'intercept'] 
cond_list = ['high_stim-high_cue', 'high_stim-low_cue',
             'med_stim-high_cue', 'med_stim-low_cue',
             'low_stim-high_cue', 'low_stim-low_cue']
merged_df = merge_qc_scl(qc_fname, scl_flist)
filtered_list = list(merged_df.filename)
betadf = pd.DataFrame(index=range(len(filtered_list)), columns=df_column + cond_list)
Path(join(save_dir)).mkdir(parents=True, exist_ok=True)


# %%
pdf = pd.read_csv(scl_flist[0], sep='\t', header=None)

# %%----------------------------------------------------------------------
#        outlier validation 1st iteration with another participant
# ------------------------------------------------------------------------
# TODO: isabel - check neurokit paramters regarding nk.find_outliers
outlier_bool = nk.find_outliers(pdf, exclude=2, side='both', method='sd')
import matplotlib.pyplot as plt

# Original data
x = list(range(len(pdf)))
plt.scatter(x, pdf, color='blue', label='Data')

column_values = pdf.iloc[:, 0].values
outlier_data = [column_values[i] if outlier else None for i, outlier in enumerate(outlier_bool)]
plt.scatter(x, outlier_data, color='red', label='Outliers')

plt.legend()
plt.show()

# %%----------------------------------------------------------------------
#        outlier validation 2nd iteration with another participant
# ------------------------------------------------------------------------
scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*sub-0034_ses-01_run-03*_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'), recursive=True))
pdf = pd.read_csv(scl_flist[0], sep='\t', header=None)

outlier_bool = nk.find_outliers(pdf, exclude=2, side='both', method='sd')
import matplotlib.pyplot as plt

# Original data
x = list(range(len(pdf)))
plt.scatter(x, pdf, color='blue', label='Data')

column_values = pdf.iloc[:, 0].values
outlier_data = [column_values[i] if outlier else None for i, outlier in enumerate(outlier_bool)]
plt.scatter(x, outlier_data, color='red', label='Outliers')

plt.legend()
plt.show()

# %%

# %% outlier validation 3rd iteration with another participant
# %%----------------------------------------------------------------------
#         outlier validation 3rd iteration with another participant
# ------------------------------------------------------------------------
scl_flist = sorted(glob.glob(join(scl_dir,'**', f'*sub-0051_ses-03_run-02*_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'), recursive=True))
pdf = pd.read_csv(scl_flist[0], sep='\t', header=None)

outlier_bool = nk.find_outliers(pdf, exclude=2, side='both', method='sd')
import matplotlib.pyplot as plt

# Original data
x = list(range(len(pdf)))
plt.scatter(x, pdf, color='blue', label='Data')

column_values = pdf.iloc[:, 0].values
outlier_data = [column_values[i] if outlier else None for i, outlier in enumerate(outlier_bool)]
plt.scatter(x, outlier_data, color='red', label='Outliers')

plt.legend()
plt.show()
# %%
# %%----------------------------------------------------------------------
#                     outlier removal and interpolation
# ----------------------------------------------------------------------

# Step 1: Create a dummy dataset and a corresponding boolean list
datavalues = pdf.iloc[:, 0].values
outlier_bool = nk.find_outliers(pdf, exclude=2, side='both', method='sd')
# example) outlier_bool = [False, False, True, False, True, False, False, False]

# Step 2: Identify indices of `True` values
outlier_indices = [i for i, val in enumerate(outlier_bool) if val]

# Step 3: Define ranges around each `True` index
ranges = []
current_start = outlier_indices[0]

for i in range(1, len(outlier_indices)):
    if outlier_indices[i] - outlier_indices[i-1] > 1:  # Check for jumps
        ranges.append((current_start, outlier_indices[i-1]))
        current_start = outlier_indices[i]

# Add the last range
ranges.append((current_start, outlier_indices[-1]))

print(ranges)
# Step 4: Interpolate and impute
# TODO: Isabel. validate that this is the right way to approach the outlier removals
# TODO: line 191, 194
interpolate_window = 10
for start, end in ranges:
    # Calculate interpolated value
    interpolated_value = (datavalues[start-interpolate_window] + datavalues[end+interpolate_window]) / 2
    
    # Impute the value at the outlier index
    datavalues[start-1:end+1] = interpolated_value

print(datavalues)

# %%
plt.scatter(x, datavalues, color='blue', label='interpolation')
# plt.scatter(x, outlier_data, color='red', label='Outliers')
plt.legend()
plt.show()
# %%
