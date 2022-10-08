#!/bin/bash -l
#SBATCH --job-name=biopac
#SBATCH --nodes=1
#SBATCH --task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=05:00:00
#SBATCH -o ./log/biopac_%A_%a.o
#SBATCH -e ./log/biopac_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

conda activate biopac
PWD=${PWD}
cluster="discovery" # local
python ${PWD}/stack_group_level.py
