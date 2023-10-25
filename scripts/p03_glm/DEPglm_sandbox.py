#!/usr/bin/env python
# encoding: utf-8

# NOTE:
# I need the entire signal (just like the brain data)
# e.g.: /Volumes/spacetop_projects_cue/analysis/physio/physio01_SCL/sub-0015/ses-01/sub-0015_ses-01_run-06_runtype-vicarious_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt
# e.g. /Volumes/spacetop_projects_cue/analysis/physio/physio01_SCL/sub-0015/ses-01/sub-0015_ses-01_run-06_runtype-vicarious_samplingrate-2000_onset.json
# then I need to convolve it with boxcars
# %%
# ----------------------------------------------------------------------
#                               libraries
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.signal import convolve
from sklearn import linear_model
import nilearn
from nilearn import glm
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
## %%load dataframe

# pdf = pd.read_csv("/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_ttlindex-1_physio-scltimecourse.csv")
# raw signal continuous. not broken up for trials
pdf_fname = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.txt'
pdf = pd.read_csv(pdf_fname, sep='\t', header=None)
jsonfname = '/Users/h/Documents/projects_local/sandbox/physioresults/physio01_SCL/sub-0017/ses-03/sub-0017_ses-03_run-05_runtype-pain_samplingrate-2000_onset.json'
# js = json.loads(jsonfname)
with open(jsonfname) as json_file:
    js = json.load(json_file)

# %% ======= NOTE: fetch HRF curve
# convolve 
tr = 25
hrf_duration = 20  # Duration of the HRF in seconds
hrf_oversampling = 1/25 #1/25  # Number of samples per second
hrf_values = nilearn.glm.first_level.glover_hrf(1/0.46, 
                                                # oversampling=50, 
                                                time_length=hrf_duration, onset=0)
plt.plot(hrf_values)

# %% PSPM SCR curve
pspm_scr = pd.read_csv('/Users/h/Documents/projects_local/spacetop_biopac/scripts/p03_glm/pspm-scrf_td-25.txt', sep='\t')
plt.plot(pspm_scr.squeeze())
scr = pspm_scr.squeeze()
# # %%
# ======= NOTE: Assuming your DataFrame is named 'df'
# for i in np.arange(1,12):
# column_names = pdf.columns
# start_index = pdf.columns.get_loc('time_0')
# selected_columns = pdf.iloc[:, start_index:-1]
# edasignal = np.array(selected_columns.iloc[0,:])
# convolved_signal = convolve(edasignal, hrf_values)[:len(edasignal)]

# %% ======= NOTE: construct event regressors
onset_sec = np.array(js['event_stimuli']['start'])/2000
total_dur_seconds = 400
data_points_per_second = 25
array_length = total_dur_seconds * data_points_per_second
signal = np.zeros(len(pdf)) #np.zeros(total_dur_seconds * data_points_per_second)
event_time = np.array(js['event_stimuli']['start'])/2000
eventtime_shift = event_time 
event_indices = (eventtime_shift * data_points_per_second).astype(int)
signal[event_indices[:len(pdf)]] = 1


# ======= NOTE: convolve 
# hrf_values = hrf_values
convolved_signal = convolve(signal, scr, mode='full')[:len(signal)]

# %%
# plt.plot(signal)
# plt.plot(edasignal)
plt.plot(convolved_signal)
# %%
plt.plot(hrf_values)

# %% -------------------------------------------------------------------
#                 local test
# ----------------------------------------------------------------------
  
# %%
# ======= NOTE: eventwise dataframe
# column_names = pdf.columns
# start_index = pdf.columns.get_loc('time_0')
# selected_columns = pdf.iloc[:, start_index:-1]
# X = selected_columns.T[:len(hrf_values)]
# y = hrf_values
# reg = linear_model.LinearRegression().fit(selected_columns.T[:len(hrf_values)], hrf_values)
# reg.score(X, y)
# reg.coef_
# reg.intercept_
# # %%
# plt.plot(selected_columns.T)
# # reg.predict(np.array([[3, 5]]))
# # append metadata
# pdf.columns[0:20]

# %%======= NOTE: timewise dataframe

X = pdf[0]
index = X.index
y = convolved_signal
plt.plot(index, X )
plt.plot(index, y*10)
# %%
X_r = np.array(X).reshape(-1,1)
Y_r = np.array(y).reshape(-1,1)
reg = linear_model.LinearRegression().fit(X_r, Y_r)
reg.score(X_r, Y_r)
print(f"coefficient: {reg.coef_}")
print(f"intercept: {reg.intercept_}")


# pdf.columns[0:20]
# %%
