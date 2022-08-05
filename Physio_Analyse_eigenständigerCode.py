#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:25:14 2022

@author: isabelneumann
"""
# SPACETOP NEUROKIT ANALYSES

##############################################################################

# %% Load packages
import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size']= 14

##############################################################################

#%% parameters 
run_num= 2

# %% Get data
spacetop_data, spacetop_samplingrate = nk.read_acqknowledge('/Users/isabelneumann/Desktop/CANlab/Heejung/Neurokit/data/sub-0051/ses-03/SOCIAL_spacetop_sub-0051_ses-03_task-social_ANISO.acq')
print(spacetop_data.columns)
bdf = pd.read_csv('/Users/isabelneumann/Desktop/CANlab/Heejung/Neurokit/data/behavioral_sub-0051-0053/sub-0051/ses-03/sub-0051_ses-03_task-social_run-02-pain_beh.csv')

# %%Organize data: trigger onsets
mid_val = (np.max(spacetop_data['trigger']) - np.min(spacetop_data['trigger']))/2
spacetop_data.loc[spacetop_data['trigger'] > mid_val, 'fmri_trigger'] = 5
spacetop_data.loc[spacetop_data['trigger'] <= mid_val, 'fmri_trigger'] = 0

start_df = spacetop_data[spacetop_data['fmri_trigger'] > spacetop_data['fmri_trigger'].shift(1)].index
stop_df = spacetop_data[spacetop_data['fmri_trigger'] < spacetop_data['fmri_trigger'].shift(1)].index
print(start_df)
print(stop_df)

# %% Organize data: transition
df_transition = pd.DataFrame({
                        'start_df': start_df, 
                        'stop_df': stop_df
                        })
run_subset = spacetop_data[df_transition.start_df[run_num]: df_transition.stop_df[run_num]]
run_df = run_subset.reset_index()

# %% Process EDA -> nachdem die ganzen Punkte extrahiert wurden (noch ändern!)
## hier noch ausprobieren 
eda_signal = nk.signal_sanitize(run_df["Skin Conductance (EDA) - EDA100C-MRI"])

#eda_filters, info = nk.signal_filter(run_df["Skin Conductance (EDA) - EDA100C-MRI"], 
                                   #sampling_rate=spacetop_samplingrate, 
                                   # highcut=1, method="butterworth", order=4)

eda_cleaned, info = nk.eda_process(run_df["Skin Conductance (EDA) - EDA100C-MRI"], 
                                    sampling_rate=spacetop_samplingrate) 
eda_decomposed = nk.eda_phasic(nk.standardize(run_df["Skin Conductance (EDA) - EDA100C-MRI"]), 
                                sampling_rate=spacetop_samplingrate, 
                                method='highpass')
plot_test = eda_decomposed.plot
plt.show(plot_test)
eda_peaks, info = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values,
                                sampling_rate=spacetop_samplingrate, 
                                method='neurokit', 
                                amplitude_min = 0.02)                         

#%% 
eda_process, info = nk.eda_process(run_df["Skin Conductance (EDA) - EDA100C-MRI"], 
                            sampling_rate= spacetop_samplingrate, 
                            method='neurokit')

# %% Start & end of condition 
# ab hier ist wieder "normal"
expect_start = run_df[run_df['expect'] > run_df['expect'].shift(1)]

expect_end = run_df[run_df['expect'] < run_df['expect'].shift(1)]


# %% Extract start & end for creating events
# expect_start_list = expect_start["index"].values.tolist()
expect_start_list = expect_start.index.tolist()
print(expect_start_list)
expect_end_list = expect_end.index.tolist()
event_labels = [1,2,1,3,5,6,5,4,4,6,2,3] # muss man irgendwie in events reinkriegen 


# %% Define events
expect_events = nk.events_create(event_onsets=expect_start_list, 
                                 event_durations = 8000, # brauche ich die Länge überhaupt?
                                 event_conditions=event_labels)
print(expect_events)

#%% Process EDA signal 
# Sanitize input
eda_signal = nk.signal_sanitize(run_df["Skin Conductance (EDA) - EDA100C-MRI"])
# Preprocess
eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=spacetop_samplingrate, method='neurokit')
eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=spacetop_samplingrate)
# Find peaks
peak_signal, info = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values,
                            sampling_rate=spacetop_samplingrate,
                            method='neurokit',
                            amplitude_min=0.02)
info["sampling_rate"] = spacetop_samplingrate  # Add sampling rate in dict info
# Store
signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})
signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)


#%%
# Filter als erstes = das macht nk.clean (nur dass die noch Butterdingens nehmen)
# Raw signal und filter signal plotten lassen als Vergleich_: 
#  signals = pd.DataFrame({"EDA_Raw": eda,
                             # "EDA_BioSPPy": nk.eda_clean(eda, sampling_rate=100,method='biosppy'),
                             # "EDA_NeuroKit": nk.eda_clean(eda, sampling_rate=100,
                              #method='neurokit')})
      #@savefig p_eda_clean.png scale=100%


#eda_df_clean, info = nk.signal_filter(eda_signal, sampling_rate=spacetop_sampling_rate, 
                     #highcut=1, method="butterworth", order=4)

# eda phasic 
# Segmente rausschneiden = epochs 
# eda peaks (neurokit als methode, min amplitude kann man angeben), evtl. standardize (helpful bei high inter-ind. differences)
# Kontrolle: Process besteht aus folgenden Schritten: preprocess (clean & decompose phasic), find peaks, s. Link aber reicht das so wie wir es in WÜ machen oder besser anderes Protokoll nehmen?
# https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/eda/eda_process.py 
# Problem nur: wie SCL? s. https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/eda/eda_intervalrelated.py
# Weiterverarbeitung: normalization! 

# %% ppg-> clean 

run_df_clean, info = nk.bio_process(eda=run_df["Skin Conductance (EDA) - EDA100C-MRI"], 
                                    ppg=run_df["Pulse (PPG) - PPG100C"],
                                    sampling_rate = spacetop_samplingrate)
plot_clean_signal = run_df_clean[["EDA_Tonic", "EDA_Phasic", "PPG_Clean", "PPG_Rate"]].plot(subplots=True)

###############################################################################

# Expect condition 

# %% Start & end of condition 
expect_start = run_df[run_df['expect'] > run_df['expect'].shift(1)]

expect_end = run_df[run_df['expect'] < run_df['expect'].shift(1)]


# %% Extract start & end for creating events
# expect_start_list = expect_start["index"].values.tolist()
expect_start_list = expect_start.index.tolist()
print(expect_start_list)
expect_end_list = expect_end.index.tolist()
event_labels = [1,2,1,3,5,6,5,4,4,6,2,3] # muss man irgendwie in events reinkriegen 


# %% Define events
expect_events = nk.events_create(event_onsets=expect_start_list, 
                                 event_durations = 8000, # brauche ich die Länge überhaupt?
                                 event_conditions=event_labels)
print(expect_events)

plot_expect = nk.events_plot(expect_events, run_df_clean['EDA_Clean'])
plot_eda_expect = nk.events_plot(expect_events, run_df["Skin Conductance (EDA) - EDA100C-MRI"])
plot_ppg_expect = nk.events_plot(expect_events, run_df["Pulse (PPG) - PPG100C"])


# %% Create epochs 
expect_epochs = nk.epochs_create(run_df_clean, 
                                 expect_events, 
                                 sampling_rate=spacetop_samplingrate, 
                                 epochs_start=-1, epochs_end=8) # kann man auch end_list nehmen? / aktuell zu kurz um intervalrelated zu machen

plot_epochs_expect = nk.epochs_plot(expect_epochs)
for epoch in expect_epochs.values():
    nk.signal_plot(epoch[["EDA_Clean", "SCR_Height"]], 
    title = epoch['Condition'].values[0],
    standardize=True)
# %% Analyze features 
expect_analysis = nk.bio_analyze(expect_epochs, sampling_rate=spacetop_samplingrate, method="event-related")

print(expect_analysis)
expect_analysis.to_csv("results_expect.csv")

# %% reshape pandas from 0 - 11 
expect_analysis.reset_index(inplace=True)

#%% 
b_p = pd.merge(expect_analysis, bdf, left_index=True, right_index = True)
b_p.to_csv("firstresults.csv")

##############################################################################

# %%Cue condition 

## Start & end of condition 
cue_start = run_df[run_df['cue'] > run_df['cue'].shift(1)]
cue_end = run_df[run_df['cue'] < run_df['cue'].shift(1)]


## Extract start & end for creating events
cue_start_list = cue_start["index"].values.tolist()
cue_end_list = cue_end["index"].values.tolist()
cue_event_labels = [1,2,1,3,5,6,5,4,4,6,2,3] # muss man irgendwie in events reinkriegen 


## Define events
cue_events = nk.events_create(event_onsets=cue_start_list, 
                              event_durations = 8000, # brauche ich die Länge überhaupt?
                              event_conditions=cue_event_labels)
print(cue_events) 

plot_eda_cue = nk.events_plot(cue_events, spacetop_data["Skin Conductance (EDA) - EDA100C-MRI"])
plot_ppg_cue = nk.events_plot(cue_events, spacetop_data["Pulse (PPG) - PPG100C"])


## Create epochs 
cue_epochs = nk.epochs_create(run_df_clean, 
                              cue_events, 
                              sampling_rate=spacetop_samplingrate, 
                              epochs_start=-1, epochs_end=8) # kann man auch end_list nehmen? / aktuell zu kurz um intervalrelated zu machen


## Analyze features 
cue_analysis = nk.bio_analyze(cue_epochs, sampling_rate=spacetop_samplingrate)

print(cue_analysis)
cue_analysis.to_csv("results_cue.csv")


# deeper into ppg: 
# run_df_ppg, info = nk.ppg_process(spacetop_data["Pulse (PPG) - PPG100C"], 
                                  # sampling_rate = spacetop_samplingrate)

#spacetop_epochs_ppg = nk.epochs_create(run_df_ppg, 
                          #spacetop_events, 
                          #sampling_rate=spacetop_samplingrate, 
                          #epochs_start=-1, epochs_end=8)

#df_ppg = nk.ppg_intervalrelated(run_df_ppg, sampling_rate = spacetop_samplingrate)

#for epoch in spacetop_epochs.values():
    # Plot scaled signals
    #nk.signal_plot(epoch[['Skin Conductance (EDA) - EDA100C-MRI', 'Pulse (PPG) - PPG100C',"expect"]], 
                   #title=epoch['Condition'].values[0],  # Extract condition name
                   #standardize=True) 


# spacetop_events = nk.events_find(event_channel = run_df['expect'],
                                 #start_at = start_list, 
                                 #end_at = end_list,
                                # event_labels = event_labels)


# irgendwie durch liste durchiterieren?

#plot = nk.events_plot(events_spacetop)
                                     
                                     #events = {"onset": []}
#if start_at>0: 
    #events_spacetop["onset"] = events_spacetop["onset"][events_spacetop["onset"] >= start_at]
    
#if end_at is not None
    #events_spacetop["onset"] = events_spacetop["onset"][events_spacetop["onset"] <= end_at]

                                 #event_labels = event_labels) 

# mit heejung
#events_spacetop = nk.events_find(event_channel = run_df['expect'],
                                 #start_at = expect_start['expect'], 
                                 #end_at = expect_end['expect'])
                                 
 #duration_min = 7900,
 #duration_max = 8100, 

# plot = nk.events_plot(events, spacetop_data['expect'])






# %%
