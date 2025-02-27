#!/bin/bash -l
#SBATCH --job-name=trialxco
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=06:00:00
#SBATCH -o ./log_trial/xcorr_%A_%a.o
#SBATCH -e ./log_trial/xcorr_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-133%30

conda activate biopac
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}
ID=$((SLURM_ARRAY_TASK_ID-1))
MAINDIR='/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue'
PHYSIO="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL"
FMRIPREP="/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/fmriprep/results/fmriprep/"
SAVE="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/xcorr_trial"


python ${PWD}/c01_corr_hpc_pertrial.py \
--slurm-id ${ID} \
--physio-dir ${PHYSIO} \
--fmriprep-dir ${FMRIPREP} \
--save-dir ${SAVE} \
-r "pain"
