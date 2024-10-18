# biopac_spacetop
[google docs](https://docs.google.com/document/d/1MG7KvyxD-4ATWAHJz7R3X1sWljrlrJyaTsE_qUBBRy8/edit)

## Order of operations
1. Convert ACQ files to BIDS format `/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue/sub-0104/ses-04`
2. Preprocess data
   1) First, we exclude outliers using by winsorizing the data with a threshold of 5 median absolute deviation. Anything below or above that threshold was convert to nans and interpolated using scipy's interpolate function.
   2)  Next, I create a boxcar based on the stimulus epoch delivery. On average, the stimulus duration is about 9 seconds on average.
   3)   Third, I use the canonical SCR from Bach's PSPM. From that, we convolve the SCR function to the constructed event time boxcar function.
3. Construct GLM with onset times and cue x stim combination.
This is fed in the model for beta coefficient estimation, construct a glm,

Purpose:
* organize files in BIDS scheme
* extract biopac signals from template
* analyze per trial and condition 

Some changes in the code
* Preprocess the .acq files using spacetop_prep
* Move here to do the analyses


Folder structure
* p01_SCLextraction
* p02_traditional
* p03_glm
* p04_wani

* 
