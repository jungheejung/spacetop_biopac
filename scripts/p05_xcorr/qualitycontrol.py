# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk
import numpy as np
import json
# %%
# loading in a example dataset
# TODO: make code generic later
df = pd.read_csv('/Users/h/Documents/projects_local/sandbox/physiodata/sub-0081/ses-04/sub-0081_ses-04_task-cue_run-03-pain_recording-ppg-eda-trigger_physio.tsv', sep='\t')

# %% Resample data
source_samplingrate=2000
dest_samplingrate=25
resamp = nk.signal_resample(
            df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=dest_samplingrate)
    
# %% Resample the data and check ______________________

df.head()
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)  # Share the x-axis
axs[0].plot(df['physio_eda'])
axs[0].set_title('EDA Signal')
axs[0].set_ylabel('EDA')

# Plot `resamp` on the second subplot
axs[1].plot(resamp)
axs[1].set_title('Resampled Signal')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlabel('Time or Samples')  
plt.tight_layout() 
plt.show()

# %% spectrogram ____________________________________________
# f: Array of sample frequencies.
# t: Array of segment times.
# Sxx: Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
fs = 25 # sampling frequency
Nx = len(resamp); nsc = np.floor(Nx/4); nov = np.floor(nsc/2); nff = int(max(256, 2 ** np.ceil(np.log2(nsc))))
nsc = fs * 10 # 10 second window
nff = 2 * Nx

# %% ground truth ____________________________________________
# Let's check if the spectrogram is acting as intended
# If we have a 1hz sinewave, and plot the spectrogram as time on the horizontal axis, frequency on the vertical axis. we should expect to see a horizontal line for the 1 hz value.  
# 1) here are the parameters for a sine wave
frequency = 1  # Frequency of the sine wave in Hz
sampling_rate = 25  # Sampling rate in Hz
duration = len(resamp)/25  # Duration in seconds
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
sine_wave = np.sin(2 * np.pi * frequency * t)
# 2) Plotting the sine wave
plt.plot(t[:100], sine_wave[:100])
plt.title('1 Second Sine Wave')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 3) spectrogram for 1hz sinewave ____________________________________________
fs = 1
f, t, Sxx = signal.spectrogram(sine_wave, fs=sampling_rate, 
                             nfft=nff)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.title('Spectrogram of 1 Hz Sine Wave (400 Seconds, Sampled at 25 Hz)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# %% spectrogram for subject data _________________________________________
# nsc: how quickly would these signals vary
# the lowest frequency 
# start with a wider window
# sine wave of 1 hz. 

f, t, Sxx = signal.spectrogram(resamp, fs=sampling_rate, 
                             nfft=nff)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.title('Raw data (full range of frequencies)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


f, t, Sxx = signal.spectrogram(resamp, fs=sampling_rate, 
                             nfft=nff)
plt.pcolormesh(t, f[:500], Sxx[:500], shading='gouraud')
plt.title('Raw data (first 500 frequencies)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
sns.despine()
# plt.show()

# 2) load onset data and mark data
json_fname = '/Users/h/Documents/projects_local/spacetop_biopac/data/sub-0081_ses-04_run-03_runtype-pain_samplingrate-2000_onset.json'
onset_samplingrate=2000
with open(json_fname, 'r') as file:
    onset = json.load(file)


onset_start_sec = [x/onset_samplingrate for x in onset['event_stimuli']['start']]
onset_stop_sec = [x/onset_samplingrate for x in onset['event_stimuli']['stop']]
# x_coords = [20, 40, 60, 80]
# for time in onset_start_sec:
#     plt.axvline(x=time, color='g', linestyle='--')  
# for time in onset_stop_sec:
#     plt.axvline(x=time, color='r', linestyle='--') 
plt.ylim([0, max(f[:500])])  # Assuming your frequency range is positive

for x in range(len(onset_start_sec)):
    plt.axvspan(onset_start_sec[x], onset_stop_sec[x], color='red', alpha=0.2)  # Adjust the color and alpha as needed

# plt.show()
# ... (your spectrogram plot code)

# Set the y-axis limit to leave some space at the bottom for the bars if needed

# Now draw the shaded bars with zorder < 0
# for x in range(len(onset_start_sec)):
#     plt.axvspan(onset_start_sec[x], onset_stop_sec[x], ymin=-0.05, ymax=0, color='red', alpha=0.5, zorder=-1)

plt.show()

# %%

##########
# ROI list
##########
# extract brain
from netneurotools import datasets as nntdata
# extract time series
from nilearn.maskers import NiftiMapsMasker,  NiftiLabelsMasker
from nilearn import datasets
fmri_fname = '/Users/h/Documents/projects_local/sandbox/fmriprep_bold/sub-0081/sub-0081_ses-04_task-social_acq-mb8_run-3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confounds_fname = '/Users/h/Documents/projects_local/sandbox/fmriprep_bold/sub-0081/sub-0081_ses-04_task-social_acq-mb8_run-4_desc-confounds_timeseries.tsv'
# schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
masker = NiftiLabelsMasker(labels_img=schaefer['maps'], 
                           labels=schaefer['labels']
                        #    standardize=True, 
                        #    high_pass=128,
                        #    t_r=0.46
                           )
# %%
# subset confounds
confounds = pd.read_csv(confounds_fname,sep='\t')
filter_col = [col for col in confounds if col.startswith("motion")]
default_csf_24dof = [
    "csf",
    "trans_x",
    "trans_x_derivative1",
    "trans_x_power2",
    "trans_x_derivative1_power2",
    "trans_y",
    "trans_y_derivative1",
    "trans_y_derivative1_power2",
    "trans_y_power2",
    "trans_z",
    "trans_z_derivative1",
    "trans_z_derivative1_power2",
    "trans_z_power2",
    "rot_x",
    "rot_x_derivative1",
    "rot_x_derivative1_power2",
    "rot_x_power2",
    "rot_y",
    "rot_y_derivative1",
    "rot_y_derivative1_power2",
    "rot_y_power2",
    "rot_z",
    "rot_z_derivative1",
    "rot_z_derivative1_power2",
    "rot_z_power2",
]
filter_col.extend(default_csf_24dof)
dummy = pd.DataFrame(np.eye(len(confounds))).loc[:, 0:5]
dummy.rename(
    columns={
        0: "dummy_00",
        1: "dummy_01",
        2: "dummy_02",
        3: "dummy_03",
        4: "dummy_04",
        5: "dummy_05",
    },
    inplace=True,
)
subset_confounds = pd.concat([confounds[filter_col], dummy], axis=1)

print("grabbed all the confounds and fmri data")
subset_confounds.head()
time_series = masker.fit_transform(fmri_fname,
                                confounds=subset_confounds.fillna(0))

# %% cross correlation 
# adjusted to ROI and PHYSIO
first_roi = time_series.T[0]
TR=0.46
fmri_samplingrate = 1/0.46
# resamp
physio_tr = nk.signal_resample(
            df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=fmri_samplingrate)


outlier_bool = nk.find_outliers(first_roi, exclude=1, side='both', method='sd')
import matplotlib.pyplot as plt

# Original data
x = list(range(len(first_roi)))
plt.scatter(x, first_roi, color='blue', label='Data')

column_values = first_roi
outlier_data = [column_values[i] if outlier else None for i, outlier in enumerate(outlier_bool)]
plt.scatter(x, outlier_data, color='red', label='Outliers')
plt.title('outlier validation: 1st iteration with participant "sub-0081')
plt.legend()
plt.show()

# mask boolean array
# first_roi_dropoutlier  = np.where(outlier_bool, first_roi, np.nan)
first_roi_dropoutlier = np.where(outlier_bool, np.nan, first_roi)
plt.plot(first_roi_dropoutlier)
plt.show()
# plt.plot(physio_tr)
# %% cross correlation Matt's class example, 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, correlate
from scipy.signal.windows import hann
# Generate signals
Fs = 500
tvec = np.arange(0, 2, 1/Fs)
data1 = np.sin(2 * np.pi * 8 * tvec) + 0.1 * np.random.randn(len(tvec))
data2 = np.sin(2 * np.pi * 8 * tvec + np.pi/4) + 0.1 * np.random.randn(len(tvec))

# Plot raw signals
plt.figure()
plt.subplot(221)
plt.plot(tvec, data1, 'r', tvec, data2, 'b')
plt.legend(['signal 1', 'signal 2'])
plt.title('raw signals')

# Compute and plot PSD
window = hann(250)
noverlap = 125
nfft = len(data1)

f, Pxx = welch(data1, Fs, window=window, noverlap=noverlap, nfft=nfft)
_, Pyy = welch(data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(222)
plt.plot(f, np.abs(Pxx), 'r', f, np.abs(Pyy), 'b')
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('PSD')

# Compute and plot cross-spectrum
f, Pxy = csd(data1, data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(223)
plt.plot(f, np.abs(Pxy))
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('cross-spectrum')

# Compute and plot cross-correlation
maxlags = 100
acf = correlate(data1, data2, mode='full', method='auto')
acf /= len(data1)  # Normalizing
lags = np.arange(-maxlags, maxlags + 1) * (1./Fs)

plt.subplot(224)
plt.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1])
plt.grid(True)
plt.xlabel('time lag (s)')
plt.ylabel('correlation (r)')
plt.title('xcorr')

# Show the figure
plt.tight_layout()
plt.show()



# modify this for physio and fmri _______________
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, correlate
from scipy.signal.windows import hann
# Generate signals
Fs = 1/TR #1/TR
 
physio_standardized = (physio_tr - np.nanmean(physio_tr)) / np.nanstd(physio_tr)
fmri_standardized = (first_roi_dropoutlier - np.nanmean(first_roi_dropoutlier))/np.nanstd(first_roi_dropoutlier)
tvec = np.arange(0, len(physio_standardized) / Fs, 1/Fs)
data1 = physio_standardized
data2 = np.nan_to_num(fmri_standardized)

# Plot raw signals
plt.figure()
plt.subplot(221)
plt.plot(tvec, data1, 'r', tvec, data2, 'b')
plt.legend(['physio', f'fmri ROI'])
plt.xlabel('time (s)')
plt.title('raw signals')

# Compute and plot PSD
ws = int(Fs * 15)
window = hann(ws)
noverlap = ws // 2
nfft = len(tvec)

f, Pxx = welch(data1, Fs, window=window, noverlap=noverlap, nfft=nfft)
_, Pyy = welch(data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(222)
plt.plot(f, np.abs(Pxx), 'r', f, np.abs(Pyy), 'b')
plt.xlim([0, 1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('PSD')

# Compute and plot cross-spectrum
f, Pxy = csd(data1, data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(223)
plt.plot(f, np.abs(Pxy))
plt.xlim([0, Fs/2]) # Nyquist frequency is the upper bound
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('cross-spectrum')

# Compute and plot cross-correlation
maxlags = int(Fs * 30)
acf = correlate(data1, data2, mode='full', method='auto')
acf /= len(data1)  # Normalizing
lags = np.arange(-maxlags, maxlags + 1) * (1./Fs)

plt.subplot(224)
plt.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1])
plt.grid(True)
plt.xlabel('time lag (s)')
plt.ylabel('correlation (r)')
plt.title('xcorr')

# Show the figure
plt.tight_layout()
plt.show()



# modify this for physio and fmri _______________
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, correlate
from scipy.signal.windows import hann
# Generate signals
Fs = 1/TR #1/TR
 
physio_standardized = (physio_tr - np.nanmean(physio_tr)) / np.nanstd(physio_tr)
fmri_standardized = (first_roi_dropoutlier - np.nanmean(first_roi_dropoutlier))/np.nanstd(first_roi_dropoutlier)
tvec = np.arange(0, len(physio_standardized[:500]) / Fs, 1/Fs)
data1 = physio_standardized[:500]
data2 = np.nan_to_num(fmri_standardized[:500])

# Plot raw signals
plt.figure()
plt.subplot(221)
plt.plot(tvec, data1, 'r', tvec, data2, 'b')
plt.legend(['physio', f'fmri ROI'])
plt.xlabel('time (s)')
plt.title('raw signals')

# Compute and plot PSD
ws = int(Fs * 30)
window = hann(ws)
noverlap = ws // 2
nfft = len(tvec)

f, Pxx = welch(data1, Fs, window=window, noverlap=noverlap, nfft=nfft)
_, Pyy = welch(data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(222)
plt.plot(f, np.abs(Pxx), 'r', f, np.abs(Pyy), 'b')
plt.xlim([0, 1])
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('PSD')

# Compute and plot cross-spectrum
f, Pxy = csd(data1, data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

plt.subplot(223)
plt.plot(f, np.abs(Pxy))
plt.xlim([0, Fs/2]) # Nyquist frequency is the upper bound
plt.xlabel('Frequency (Hz)')
plt.ylabel('power')
plt.title('cross-spectrum')

# Compute and plot cross-correlation
maxlags = int(Fs * 30)
acf = correlate(data1, data2, mode='full', method='auto')
acf /= len(data1)  # Normalizing
lags = np.arange(-maxlags, maxlags + 1) * (1./Fs)

plt.subplot(224)
plt.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1])
plt.grid(True)
plt.xlabel('time lag (s)')
plt.ylabel('correlation (r)')
plt.title('xcorr')

# Show the figure
plt.tight_layout()
plt.show()
# %%Matt's class _____________________________
# https://rcweb.dartmouth.edu/~mvdm/wiki/doku.php?id=analysis:course:week8&s[]=cross&s[]=correlation
import numpy as np
import matplotlib.pyplot as plt

Fs = 500  # Sampling frequency
dt = 1.0 / Fs  # Time interval between samples
t = [0, 2]  # Time vector from 0 to 2 seconds
tvec = np.arange(t[0], t[1], dt)

f1 = 8  # Frequency of the sine wave
# Generate the sine wave and add some noise
data1 = np.sin(2 * np.pi * f1 * tvec) + 0.1 * np.random.randn(tvec.size)

# Compute the autocorrelation function
acf = np.correlate(data1, data1, mode='full') / data1.size
lags = np.arange(-len(data1) + 1, len(data1)) * dt  # Lags converted to time

# Only take the second half of the autocorrelation function, as the first half is symmetric
half = len(acf) // 2
acf = acf[half:]
lags = lags[half:]

# Plot the autocorrelation function
plt.plot(lags, acf)
plt.grid(True)
plt.xlabel('Lags (seconds)')
plt.ylabel('Autocorrelation coefficient')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

Fs = 500  # Sampling frequency
dt = 1.0 / Fs  # Time interval between samples
tvec = np.arange(0, 2, dt)  # Time vector from 0 to 2 seconds, excluding the endpoint

f2 = 8  # Frequency of the sine wave
# Generate the sine wave with a phase shift and add some noise
data2 = np.sin(2 * np.pi * f2 * tvec + np.pi / 4) + 0.1 * np.random.randn(tvec.size)

# Compute the cross-correlation function
ccf = np.correlate(data1, data2, mode='full') / data1.size
lags = np.arange(-len(data1) + 1, len(data1)) * dt  # Lags converted to time

# Only take the second half of the cross-correlation function, as the first half is symmetric
half = len(ccf) // 2
ccf = ccf[half:]
lags = lags[half:]

# Plot the cross-correlation function
plt.plot(lags, ccf)
plt.grid(True)
plt.xlabel('Lags (seconds)')
plt.ylabel('Cross-correlation coefficient')
plt.show()

