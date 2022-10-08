#!/bin/bash -l
#SBATCH --job-name=biopac
#SBATCH --nodes=1
#SBATCH --task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=01:00:00
#SBATCH -o ./log/biopac_%A_%a.o
#SBATCH -e ./log/biopac_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=4-14 

conda activate biopac
#${SLURM_ARRAY_TASK_ID}
PWD=${PWD}
cluster="discovery" # local
python ${PWD}/group_level_physio_analysis.py ${cluster} ${SLURM_ARRAY_TASK_ID}
