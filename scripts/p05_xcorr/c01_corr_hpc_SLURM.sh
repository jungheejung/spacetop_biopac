#!/bin/bash -l
#SBATCH --job-name=glm
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=03:00:00
#SBATCH -o ./log_xcorr/xcorr_%A_%a.o
#SBATCH -e ./log_xcorr/xcorr_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=1-133%30

conda activate biopac
echo "SLURMSARRAY: " ${SLURM_ARRAY_TASK_ID}
ID=$((SLURM_ARRAY_TASK_ID-1))
MAINDIR='/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue'
PHYSIO="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL"
FMRIPREP="/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/derivatives/fmriprep/results/fmriprep/"
SAVE="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/xcorr_canlab2023"
MASK="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/data/atlas"

python /dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_physio/scripts/p05_xcorr/c01_corr_hpc.py \
--slurm-id ${ID} \
--physio-dir ${PHYSIO} \
--fmriprep-dir ${FMRIPREP} \
--save-dir ${SAVE} \
--mask-dir ${MASK} \
-r "pain"
