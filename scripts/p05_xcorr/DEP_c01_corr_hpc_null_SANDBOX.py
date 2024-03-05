# %%
import pandas as pd
import argparse
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
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import welch, csd, correlate, coherence
from scipy.signal.windows import hann
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.interpolate import interp1d
from feature_engine.outliers import Winsorizer

# %%
__author__ = "Heejung Jung"
__copyright__ = "Spatial Topology Project"
__credits__ = ["Heejung"] # people who reported bug fixes, made suggestions, etc. but did not actually write the code.
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Heejung Jung"
__email__ = "heejung.jung@colorado.edu"
__status__ = "Development" 

# 0. argparse __________________________________________________________________
parser = argparse.ArgumentParser()
parser.add_argument("--slurm-id", type=int,
                    help="specify slurm array id")
parser.add_argument("--physio-dir", type=str,
                    help="directory for physio data (BIDS)")
parser.add_argument("--fmriprep-dir", type=str,
                    help="directory for physio data (BIDS)")
parser.add_argument("--save-dir", 
type=str,
                    help="directory for saving Xcorr data")
parser.add_argument("-r", "--runtype",
                    choices=['pain','vicarious','cognitive','all'], 
                    help="specify runtype name (e.g. pain, cognitive, variance)")

args = parser.parse_args()
print(args.slurm_id)
slurm_id = args.slurm_id # e.g. 1, 2
physio_dir = args.physio_dir
fmriprep_dir = args.fmriprep_dir
save_top_dir = args.save_dir
runtype = args.runtype
lag_window = 5
# %% 0. parameters
#sub_folders = next(os.walk(physio_dir))#[0]
# _, sub_folders, _ = next(os.walk(physio_dir))
# sub_list = [i for i in sorted(sub_folders) if i.startswith('sub-')]
# sub = sub_list[slurm_id]#f'sub-{sub_list[slurm_id]:04d}'
physio_dir = '/Users/h/Documents/projects_local/sandbox/physiodata'
# fmriprep_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/fmriprep/results/fmriprep/'
fmriprep_dir = '/Users/h/Documents/projects_local/sandbox/fmriprep_bold'
save_dir = '/Users/h/Documents/projects_local/sandbox'
save_top_dir = '/Users/h/Desktop'
runtype = 'pain'
sub = 'sub-0081'
save_dir = join(save_top_dir, sub)
Path(save_dir).mkdir(parents=True, exist_ok=True)
#physio_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue'
# physio_dir = '/Users/h/Documents/projects_local/sandbox/physiodata'
SCL_epoch_start = -3
SCL_epoch_end = 20
baselinecorrect = "False"
dest_samplingrate = 25
# fmriprep_dir = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/fmriprep/results/fmriprep/'
# fmriprep_dir = '/Users/h/Documents/projects_local/sandbox/fmriprep_bold'
# save_dir = '/Users/h/Documents/projects_local/sandbox'
runtype = 'pain'


def winsorize_mad(data, threshold=3.5):
    wz = Winsorizer(capping_method='mad', tail='both', fold=threshold)
    winsorized_data = wz.fit_transform(data)
    return winsorized_data

def interpolate_data(data):
    time_points = np.arange(len(data))
    valid = ~np.isnan(data)  # Mask of valid (non-NaN) data points
    interp_func = interp1d(time_points[valid], data[valid], kind='linear', fill_value="extrapolate")
    return interp_func(time_points)

# %% 1. glob physio data

# * baseline corrected
# * detrend
# * low pass filter of 1


# physio_flist = glob.glob(join(physio_dir, '**', '{sub}_ses-*_run-*_runtype-{runtype}_epochstart--3_epochend-20_baselinecorrect-True_samplingrate-25_physio-eda.tsv'), recursive=True)

physio_flist = glob.glob(join(physio_dir, '**', f"{sub}_*_runtype-{runtype}_epochstart-{SCL_epoch_start}_epochend-{SCL_epoch_end}_baselinecorrect-{baselinecorrect}_samplingrate-{dest_samplingrate}_physio-eda.tsv"), recursive=True)
print(physio_flist)
# /Volumes/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL/sub-0015/ses-01/sub-0015_ses-01_run-01_runtype-pain_epochstart--3_epochend-20_baselinecorrect-False_samplingrate-25_physio-eda.tsv
# %%
physio_flist = ['/Volumes/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL/sub-0081/ses-04/sub-0081_ses-04_run-03_runtype-pain_epochstart--3_epochend-20_baselinecorrect-False_samplingrate-25_physio-eda.tsv']
for i, physio_fname in enumerate(physio_flist):

    # 2. Extract bids info
    matches = re.search(r"sub-(\d+)_ses-(\d+)_run-(\d+)_", physio_fname)
    # matches = re.search(r"sub-(\d+)_ses-(\d+)_.*_run-(\d+)-", os.path.basename(physio_fname))
    if matches:
        sub = f"sub-{matches.group(1)}"
        ses = f"ses-{matches.group(2)}"
        run = f"run-{matches.group(3)}"
    else:
        sub, ses, run = None, None, None
    print(f"step 2: extract bids info {sub} {ses} {run}")
    # 3-1. load physio data ________________________________________________
    df = pd.read_csv(physio_fname, sep='\t')
    # resample physio data (2000hz to 25hz)
    TR =0.46
    Fs = 1/TR #1/TR
    source_samplingrate=25
    dest_samplingrate=1/0.46
    physio_resamp = nk.signal_resample(
                df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=source_samplingrate, desired_sampling_rate=dest_samplingrate)

    # filter
    # butterworth
    # detrend
    # 3-2. load brain data ________________________________________________
    print(f"step 3-2: load brain data _____________")
    fmri_fname = join(fmriprep_dir,sub, ses, 'func', f"{sub}_{ses}_task-social_acq-mb8_run-{int(matches.group(3))}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" )


    confounds_fname = join(fmriprep_dir, sub, ses, 'func', f'{sub}_{ses}_task-social_acq-mb8_run-{int(matches.group(3))}_desc-confounds_timeseries.tsv')

    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, data_dir=save_top_dir)
    #masker = NiftiLabelsMasker(labels_img=schaefer['maps'], 
    #                        labels=schaefer['labels']
                            #    standardize=True, 
                            #    high_pass=128,
                            #    t_r=0.46
                            #)
    masker = NiftiLabelsMasker(labels_img=join(save_top_dir, 'schaefer_2018', 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'),
                            high_pass=1/128,
                               t_r=0.46 )
    #        labels=join(save_top_dir, 'schaefer_2018', 'Schaefer2018_400Parcels_7Networks_order.txt'))
    # 3-3. subset confounds
    print(f"3-3 confound subset from fmriprep")
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
    # print(subset_confounds.head())
    time_series = masker.fit_transform(fmri_fname, confounds=subset_confounds.fillna(subset_confounds.median()))

    # 3-4. resample physio to fmri TR
    physio_samplingrate = 25
    TR=0.46
    fmri_samplingrate = 1/0.46
    # resamp physio data to TR sampling rate
    # physio_tr = nk.signal_resample(
    #             df['physio_eda'].to_numpy(),  method='interpolation', sampling_rate=physio_samplingrate, desired_sampling_rate=fmri_samplingrate)
    # physio_center = physio_tr - np.nanmean(physio_tr)
    # physio_filter = nk.signal_filter(physio_center, 
    #                                  sampling_rate=dest_samplingrate,
    #                                  highcut=Fs//2,
    #                                  lowcut=.01,
    #                                  method="butterworth",
    #                                  order=2)
    # physio_filter = resamp #df['physio_eda'].to_numpy()
    # print(f"physio filter shape: {physio_filter.shape}")
    # physio_detrend = nk.signal_detrend(physio_filter, 
    #                                    method="polynomial", 
    #                                    order=0)

    # 4. loop through ROI and calculate xcorr ________________________________________________
    # 4-1. create dataframe to store ROI data
    print(f"step 4: calculate xcorr _____________")
    # roi_df = pd.DataFrame(index=range(time_series.shape[1]), columns=['sub', 'ses', 'run', 'roi', 'trial','Maximum Correlation Value', 'Time Lag (s)'])
    roi_data = []
    for roi_ind in range(time_series.shape[1]):
        # remove outlier
        roi = time_series.T[roi_ind]


        fmri_outlier = winsorize_mad(roi.reshape(-1,1), threshold=7)
        physio_outlier = winsorize_mad(physio_resamp.reshape(-1,1), threshold=7)
        # fmri_outlier = interpolate_data(winsor_physio)

        # 4-2. plot and save

        Fs = 1/TR #1/TR
        
        physio_standardized = (physio_outlier - np.nanmean(physio_outlier) )/ np.nanstd(physio_outlier)
        fmri_standardized = (fmri_outlier - np.nanmean(fmri_outlier))/np.nanstd(fmri_outlier)
        total_length = len(fmri_standardized)

        from scipy.interpolate import interp1d
        # mean_data1 = np.nanmean(physio_standardized)
        # mean_data2 = np.nanmean(fmri_standardized)

        # # Replace NaN values with the computed mean
        # data1 = np.where(np.isnan(physio_standardized), mean_data1, physio_standardized)
        # data2 = np.where(np.isnan(fmri_standardized), mean_data2, fmri_standardized)
        def interpolate_data(data):
            time_points = np.arange(len(data))
            valid = ~np.isnan(data)  # Mask of valid (non-NaN) data points
            interp_func = interp1d(time_points[valid], data[valid], kind='linear', fill_value="extrapolate")
            return interp_func(time_points)


# identify trials _____
        df_metajson = join(os.path.dirname(physio_fname), f"{sub}_{ses}_{run}_runtype-{runtype}_samplingrate-2000_onset.json")

        with open(df_metajson, 'r') as file:
            df_meta = json.load(file)
            print(df_meta)
        match = re.search(r"samplingrate-(\d+)", df_metajson)

        if match:
            # Extract the number following "samplingrate-"
            metadata_sampling_rate = int(match.group(1))
            print("Sampling rate:", metadata_sampling_rate)
        else:
            print("Sampling rate not found")
        conversion_factor = metadata_sampling_rate / fmri_samplingrate
        converted_df_meta = {}
        for event_type, times in df_meta.items():
            converted_df_meta[event_type] = {}
            for key, value in times.items():
                converted_df_meta[event_type][key] = [x / conversion_factor for x in value]
        # extract stimuli time from onset values
                
        # TODO: baseline value
        iti_intervals = []
        # For subsequent intervals, use the previous 'event_actualrating' 'stop' time to the current 'event_cue' 'start' time
        for i in range(1, len(converted_df_meta['event_cue']['start'])):
            iti_interval = (converted_df_meta['event_actualrating']['stop'][i-1], converted_df_meta['event_cue']['start'][i])
            iti_intervals.append(iti_interval)
        padding_seconds = 2.5  # seconds
        padding_samples = padding_seconds * fmri_samplingrate 
        physio_iti = []
        fmri_iti = []
        for i, (start, stop) in enumerate(iti_intervals):
            start_index = max(0, int(start) - padding_samples)
            stop_index = min(len(physio_standardized) - 1, np.round(stop)  + padding_samples)
            physio_segment = physio_standardized.loc[int(start_index):int(stop_index), 'x0']
            fmri_segment = fmri_standardized.loc[int(start_index):int(stop_index), 'x0']
            physio_iti.append(physio_segment)
            fmri_iti.append(fmri_segment)    
        # fmri_shave = fmri_standardized[6:]
        # physio_shave= physio_standardized[6:total_length] 
        # tvec = np.arange(0, len(physio_shave) / Fs, 1/Fs)
        # print(f"tvec: {len(tvec)}, physio:{physio_shave.shape}, fmri:{fmri_shave.shape}")


        ####      
        # event_stimuli_start = converted_df_meta['event_stimuli']['start']
        # event_stimuli_stop =  converted_df_meta['event_stimuli']['stop']
 # Convert seconds to samples based on sampling rate
        # beh_meta = pd.read_csv(join(physio_dir,sub, ses, f"{sub}_{ses}_{run}_runtype-{runtype}_epochstart--3_epochend-20_baselinecorrect-False_physio-scl.csv" ))
        beh_meta = pd.read_csv(join('/Volumes/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL',sub, ses, f"{sub}_{ses}_{run}_runtype-{runtype}_epochstart--3_epochend-20_baselinecorrect-False_physio-scl.csv" ))

# extract physio and fmri data based on onset time
        differences = [stop - start for start, stop in iti_intervals]

        differences
        # physio_iti = np.mean(np.vstack(physio_iti), axis=0).T
        # Interpolate missing values
        # data1 = interpolate_data(physio_standardized)
        # data2 = interpolate_data(fmri_standardized)
        for i in np.arange(len(physio_iti)):
            data1 = physio_iti[i]
            data2 = fmri_iti[i]
            tvec = np.arange(0, len(data1) / Fs, 1/Fs)
            tvec = tvec[:len(data1)]
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
            ws = int(Fs * lag_window)
            ws = len(data1)
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
            # stimulus_duration_sec = len()  # Example: 10 seconds
            # half_stimulus_length_samples = (Fs * stimulus_duration_sec) // 2  # Convert half duration to samples
            # maxlags = int(half_stimulus_length_samples)

            maxlags = len(data1)//2#int(Fs * lag_window)
            # maxlags = int(5 * Fs)
            acf = correlate(data1, data2, mode='full', method='auto') # matlab: xcorr
            # acf /= len(data1)  # Normalizing
            norm_factor = np.sqrt(np.sum(data1**2) * np.sum(data2**2))
            ccf = acf / norm_factor
            lags = np.arange(-maxlags, maxlags + 1) * (1./Fs)

            max_lags_index = np.where((lags >= -maxlags) & (lags <= maxlags))
            ccf = ccf[max_lags_index]
            lags = lags[max_lags_index]
            convert_lags = lags*(1./Fs);
            # ax4.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf[len(acf)//2-maxlags:len(acf)//2+maxlags+1])
            # ax4.plot(lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1], acf_normalized[len(acf_normalized)//2-maxlags:len(acf_normalized)//2+maxlags+1])
            ax4.plot(convert_lags, ccf)
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
            plt.show()
            # fig.savefig(join(save_dir, f"{sub}_{ses}_{run}_runtype-{runtype}_roi-{roi}_xcorr-fmri-physio.png"))
            # plt.close(fig)
            # calculate xcorr and save in dataframe _________________________
            # Slicing acf and lags for the plot range
            acf_sliced = ccf[len(ccf)//2-maxlags:len(ccf)//2+maxlags+1]
            lags_sliced = lags[len(lags)//2-maxlags:len(lags)//2+maxlags+1]

            # Find the maximum correlation value and corresponding time lag
            absolute_values = [abs(number) for number in acf_sliced]  # Convert all numbers to their 
            max_acf_value = np.max(absolute_values)
            max_acf_index = np.argmax(absolute_values)
            max_lag_time = lags_sliced[max_acf_index]

        # Create a DataFrame to store these values _________________________
            roi_data.append({
                'sub': sub,
                'ses': ses,
                'run': run,
                'roi': roi_ind,
                'index': i,  # Assuming 'i' is an index, renamed to avoid confusion
                'max_acf_value': max_acf_value,
                'max_lag_time': max_lag_time
            })
    roi_df = pd.DataFrame(roi_data)
    save_fname = join(save_top_dir, f"{sub}_{ses}_{run}_runtype-{runtype}_xcorr-fmri-physio.tsv")
    roi_df.to_csv(save_fname,sep='\t')
plt.close('all')
print("complete!")
