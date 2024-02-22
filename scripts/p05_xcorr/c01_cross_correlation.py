# %% libraries
import pandas as pd

from scipy import signal
import neurokit2 as nk
import numpy as np
import os, json, re, glob
from os.path import join
from netneurotools import datasets as nntdata
from neuromaps.parcellate import Parcellater
import neuromaps
from neuromaps import datasets as neuromaps_datasets
from neuromaps.datasets import fetch_annotation, fetch_fslr
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti
from neuromaps.transforms import fsaverage_to_fslr
# extract time series
from nilearn.maskers import NiftiMapsMasker,  NiftiLabelsMasker
from nilearn import datasets

import matplotlib.pyplot as plt
from scipy.signal import welch, csd, correlate, coherence
from scipy.signal.windows import hann
import matplotlib.gridspec as gridspec
import seaborn as sns
# %%

# %% 0. parameters
physio_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue'
physio_dir = '/Users/h/Documents/projects_local/sandbox/physiodata'
fmri_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/fmriprep/results/fmriprep/'
fmri_dir = '/Users/h/Documents/projects_local/sandbox/fmriprep_bold'
save_dir = '/Users/h/Documents/projects_local/sandbox'
task = 'pain'
# %% 1. glob physio data

physio_flist = glob.glob(join(physio_dir, '**', f'sub-*_ses-*_task-cue_run-03-{task}_recording-ppg-eda-trigger_physio.tsv'), recursive=True)

# %%
for i, physio_fname in enumerate(physio_flist):

    # 2. Extract bids info
    matches = re.search(r"sub-(\d+)/ses-(\d+)/.*_run-(\d+)-", physio_fname)
    if matches:
        sub = f"sub-{matches.group(1)}"
        ses = f"ses-{matches.group(2)}"
        run = f"run-{matches.group(3)}"
    else:
        sub, ses, run = None, None, None

    # 3-1. load physio data ________________________________________________
    df = pd.read_csv(physio_fname, sep='\t')

    source_samplingrate=2000
    dest_samplingrate=25
    resamp = nk.signal_resample(
                df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=dest_samplingrate)

    # 3-2. load brain data ________________________________________________
    print(f"step 3-2: load brain data _____________")
    fmri_fname = join(fmri_dir,sub, ses, 'func', f"{sub}_{ses}_task-social_acq-mb8_run-{int(matches.group(3))}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" )


    confounds_fname = join(fmri_dir, sub, ses, 'func', f'{sub}_{ses}_task-social_acq-mb8_run-{int(matches.group(3))}_desc-confounds_timeseries.tsv')

    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    masker = NiftiLabelsMasker(labels_img=schaefer['maps'], 
                            labels=schaefer['labels']
                            #    standardize=True, 
                            #    high_pass=128,
                            #    t_r=0.46
                            )

    # 3-3. subset confounds
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
            5: "dummy_05"
        },
        inplace=True,
    )
    subset_confounds = pd.concat([confounds[filter_col], dummy], axis=1)

    print("grabbed all the confounds and fmri data")
    subset_confounds.head()
    time_series = masker.fit_transform(fmri_fname,
                                    confounds=subset_confounds.fillna(subset_confounds.median()))

    # 3-4. resample physio to fmri TR
    TR=0.46
    fmri_samplingrate = 1/0.46
    # resamp physio data to TR sampling rate
    physio_tr = nk.signal_resample(
                df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=fmri_samplingrate)
    
    # 4. loop through ROI and calculate xcorr ________________________________________________
    # 4-1. create dataframe to store ROI data
    print(f"step 4: calculate xcorr _____________")
    roi_df = pd.DataFrame(index=range(time_series.shape[1]), columns=['sub', 'ses', 'run', 'roi', 'Maximum Correlation Value', 'Time Lag (s)'])
    for roi in range(time_series.shape[1]):
        # remove outlier
        second_roi = time_series.T[roi]

        outlier_bool = nk.find_outliers(second_roi, exclude=1, side='both', method='sd')
                                        
        column_values = second_roi
        outlier_data = [column_values[i] if outlier else None for i, outlier in enumerate(outlier_bool)]

        second_roi_dropoutlier = np.where(outlier_bool, np.nan, second_roi)


        # 4-2. plot and save

        Fs = 1/TR #1/TR
        
        physio_standardized = (physio_tr - np.nanmean(physio_tr)) / np.nanstd(physio_tr)
        fmri_standardized = (second_roi_dropoutlier - np.nanmean(second_roi_dropoutlier))/np.nanstd(second_roi_dropoutlier)
        tvec = np.arange(0, len(physio_standardized) / Fs, 1/Fs)
        data1 = physio_standardized
        data2 = np.nan_to_num(fmri_standardized)

        # 4-2. plot parameters
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, :]) # Wide subplot on row 1
        ax2 = fig.add_subplot(gs[1, 0]) # Subplot 1 on row 2
        ax3 = fig.add_subplot(gs[1, 1]) # Subplot 2 on row 2
        ax4 = fig.add_subplot(gs[1, 2]) # Subplot 3 on row 2
        ax5 = fig.add_subplot(gs[1, 3]) # Subplot 3 on row 2

        # 4-2.A: Plot raw signals ______________________________
        ax1.plot(tvec, data1, 'r', linewidth=1, alpha=0.7)
        ax1.plot(tvec, data2, 'b', linewidth=1, alpha=0.7)
        ax1.legend(['physio', f'fmri ROI'])
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('amplitude normalized (a.u.)')
        ax1.set_title('raw signals')

        # 4-2.B:  Compute and plot PSD ______________________________
        ws = int(Fs * 15)
        window = hann(ws)
        noverlap = ws // 2
        nfft = len(tvec)

        f, Pxx = welch(data1, Fs, window=window, noverlap=noverlap, nfft=nfft)
        _, Pyy = welch(data2, Fs, window=window, noverlap=noverlap, nfft=nfft)

        ax2.plot(f, np.abs(Pxx), 'r', f, np.abs(Pyy), 'b')
        ax2.set_xlim([0, 1])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('power')
        ax2.set_title('PSD')

        # 4-2.C: Compute and plot cross-spectrum
        f, Pxy = csd(data1, data2, Fs, window=window, noverlap=noverlap, nfft=nfft)
        ax3.plot(f, np.abs(Pxy))
        ax3.set_xlim([0, Fs/2]) # Nyquist frequency is the upper bound
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('power')
        ax3.set_title('cross-spectrum')

        # 4-2.D: Compute and plot cross-correlation
        maxlags = int(Fs * 30)
        acf = correlate(data1, data2, mode='full', method='auto')
        acf /= len(data1)  # Normalizing
        lags = np.arange(-maxlags, maxlags + 1) * (1./Fs)

        ax4.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1])
        ax4.grid(True)
        ax4.set_xlabel('time lag (s)')
        ax4.set_ylabel('correlation (r)')
        ax4.set_title('xcorr')
        ax4.text(0.5, -0.2, '<-- SCR leads       BOLD leads -->', ha='center', va='center', transform=ax4.transAxes)


        # 4-2.E: Coherence
        f_coh, Cxy = coherence(data1, data2, Fs, window=hann(ws), noverlap=noverlap, nfft=nfft)
        ax5.plot(f_coh, Cxy)
        ax5.set_xlim([0, Fs/2])  # Limit to Nyquist frequency
        ax5.set_xlabel('frequency [Hz]')
        ax5.set_ylabel('Coherence')
        ax5.set_title('Coherence Spectrum')
        plt.tight_layout()
        sns.despine()
        # plt.show()

        # calculate xcorr and save in dataframe _________________________
        # Slicing acf and lags for the plot range
        acf_sliced = acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1]
        lags_sliced = lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1]

        # Find the maximum correlation value and corresponding time lag
        max_acf_value = np.max(acf_sliced)
        max_acf_index = np.argmax(acf_sliced)
        max_lag_time = lags_sliced[max_acf_index]


        # Create a DataFrame to store these values _________________________
        # df = pd.DataFrame({'Maximum Correlation Value': [max_acf_value],
        #                    'Time Lag (s)': [max_lag_time]})

        roi_df.iloc[roi] = [sub, ses, run, roi, max_acf_value, max_lag_time]
        save_fname = join(save_dir, f"{sub}_{ses}_{run}_runtype-{task}_xcorr-fmri-physio.tsv")
        roi_df.to_csv(save_fname,sep='\t')


# %% plot concatenated dataframe
    # roi_df['Maximum Correlation Value']
    #     loading_data = data[:, comp_ind-1]
# loading = parc.inverse_transform(roi_df['Maximum Correlation Value'])
# maxval = np.max(loading_data);    minval = np.min(loading_data); sd = np.std(loading_data)
# plot_brain_surfaces(image=loading, cbar_label=cbar_label, cmap=cmap, color_range=(minval+sd, maxval-sd))
# plt.show()
# %% 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
# Define the colors at specific points
colors = [
    (-1.8, "#120041"),  # Start with blue at -1.9
    (-1.2, "#2500fa"),
    (-0.6, "#84c6fd"),  # Start with blue at -1.9
    (0, "white"),    # Transition to white at 0
    (0.4, "#d50044"),
    (0.8, "#ff0000"),    # Start transitioning to red just after 0 towards 1.2
    (1.2, "#ffd400")  # End with yellow at 1.2
]

colors_with_opacity = [
    (-1.8, to_rgba("#3661ab", alpha=1.0)),  # Fully opaque
    (-0.9, to_rgba("#63a4ff", alpha=0.8)),  # Fully opaque
    # (-0.1, to_rgba("#008bff", alpha=0.6)),  # Fully opaque
    (0, to_rgba("white", alpha=1.0)),       # Fully opaque
    # (0.1, to_rgba("#d50044", alpha=0.6)),   # 30% opacity
    (0.6, to_rgba("#ffa300", alpha=0.8)),   # 60% opacity
    (1.2, to_rgba("#ff0000", alpha=1.0))    # Fully opaque
]



# Normalize the points to the [0, 1] interval
norm_points = np.linspace(-1.9, 1.2, len(colors_with_opacity))
norm_colors = [c[1] for c in colors_with_opacity]
norm_points = (norm_points - norm_points.min()) / (norm_points.max() - norm_points.min())

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("custom_gradient", list(zip(norm_points, norm_colors)))

# Create a gradient image
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Plot the gradient
fig, ax = plt.subplots(figsize=(6, 2))
ax.imshow(gradient, aspect='auto', cmap=cmap)
ax.set_axis_off()

plt.show()
# %%
from nilearn import image, plotting
from surfplot import Plot
from neuromaps.transforms import fsaverage_to_fslr
import glob

def plot_brain_surfaces(image, cbar_label='INSERT LABEL', cmap='viridis', color_range=None):
    """
    Plot brain surfaces with the given data.

    Parameters:
    - TST: Tuple of (left hemisphere data, right hemisphere data) to be plotted.
    - cbar_label: Label for the color bar.
    - cmap: Colormap for the data.
    - color_range: Optional. Tuple of (min, max) values for the color range. If not provided, the range is auto-detected.
    """
    surfaces_fslr = fetch_fslr()
    lh_fslr, rh_fslr = surfaces_fslr['inflated']
    
    p = Plot(surf_lh=lh_fslr,
             surf_rh=rh_fslr, 
             size=(1000, 200), 
             zoom=1.2, layout='row', 
             views=['lateral', 'medial', 'ventral', 'posterior'], 
             mirror_views=True, brightness=.7)
    p.add_layer({'left': image[0], 
            'right': image[1]}, 
            cmap=cmap, cbar=True,
            color_range=color_range,
            cbar_label=cbar_label
            ) # YlOrRd_r

    cbar_kws = dict(outer_labels_only=True, pad=.02, n_ticks=2, decimals=3)
    fig = p.build(cbar_kws=cbar_kws)
    return(fig)
    # fig.show()

# Example usage:
# TST = (left_hemisphere_data, right_hemisphere_data)
# plot_brain_surfaces(TST, cbar_label='gradient', cmap='viridis', color_range=(0, .15))

# %%
xcorr_fname = '/Users/h/Documents/projects_local/sandbox/sub-0081_ses-04_run-03_runtype-pain_xcorr-fmri-physio.tsv'
roi_df  =pd.read_csv(xcorr_fname, '\t')
schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
parc = Parcellater(dlabel_to_gifti(schaefer), 'fsLR')
roidata = roi_df['Maximum Correlation Value']
maxval = np.max(roidata);    minval = np.min(roidata); sd = np.std(roidata)
plot_brain_surfaces(image=roidata, cbar_label='xcorr sub-0081_ses-04_run-03', cmap=cmap, color_range=(minval+sd, maxval-sd))
plt.show()