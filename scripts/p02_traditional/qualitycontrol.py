# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import neurokit2 as nk
import numpy as np
# %%
df = pd.read_csv('/Users/h/Documents/projects_local/sandbox/physiodata/sub-0081/ses-04/sub-0081_ses-04_task-cue_run-03-pain_recording-ppg-eda-trigger_physio.tsv', sep='\t')

# %%
source_samplingrate=2000
dest_samplingrate=25
resamp = nk.signal_resample(
            df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=dest_samplingrate)
    
# %% Resample the data and check ______________________

df.head()
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)  # Share the x-axis
# plt.plot(df['physio_eda'])
# plt.plot(resamp)
# Plot `df['physio_eda']` on the first subplot
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
# f: Array of sample frequencies.
# t: Array of segment times.
# Sxx: Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
# %% spectrogram
fs = 25 # sampling frequency
Nx = len(resamp); nsc = np.floor(Nx/4); nov = np.floor(nsc/2); nff = int(max(256, 2 ** np.ceil(np.log2(nsc))))
nsc = fs * 10 # 10 second window
nff = 2 * Nx

# %% ground truth
# Parameters for the sine wave
frequency = 1  # Frequency of the sine wave in Hz
sampling_rate = 25  # Sampling rate in Hz
duration = len(resamp)/25  # Duration in seconds

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Sine wave
sine_wave = np.sin(2 * np.pi * frequency * t)
# Plotting the sine wave
plt.plot(t[:100], sine_wave[:100])
plt.title('1 Second Sine Wave')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# spectrogram ______
fs = 1
f, t, Sxx = signal.spectrogram(sine_wave, fs=sampling_rate, 
                             nfft=nff)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.title('Spectrogram of 1 Hz Sine Wave (400 Seconds, Sampled at 25 Hz)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# %%
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
plt.title('Raw data (first 200 frequencies)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
# %%
# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
# ax1.plot(df['physio_eda'])
# ax1.set_ylabel('Signal')

# Pxx, freqs, bins, im = ax2.specgram(resamp, NFFT=nff, Fs=fs, noverlap=nov)

# # The `specgram` method returns 4 objects. They are:
# # - Pxx: the periodogram
# # - freqs: the frequency vector
# # - bins: the centers of the time bins
# # - im: the .image.AxesImage instance representing the data in the plot
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Frequency (Hz)')
# ax2.set_xlim(0, 20)

# plt.show()
# %%
