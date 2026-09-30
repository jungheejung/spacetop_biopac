#!/bin/bash -l
#SBATCH --job-name=scr_singletrial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=04:00:00
#SBATCH -o ./log/scr_singletrial_%j.o
#SBATCH -e ./log/scr_singletrial_%j.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

# Single-trial SCR GLMs on the no-baseline SCL, one heat regressor per trial:
#   singletrial            heat only (design of the old nobaseline/glm_singletrial table)
#   singletrial_allevents  + cue and rating regressors as nuisance
# Outliers capped at Q1-5*IQR / Q3+5*IQR (as the Oct 2024
# glm_singletrial.py). Each writes its own table + 2 PNGs per run; the old
# nobaseline/glm_singletrial table is not touched.
# Submit from scripts/p03_glm:   mkdir -p log && sbatch glm_singletrial_allevents_scr.sh

conda activate biopac

CUE=/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue
SAVE_DIR=${CUE}/analysis/physio/glm_allevents

for d in ${CUE}/analysis/physio/nobaseline/physio01_SCL ${CUE}/analysis/physio_nobaseline/physio01_SCL; do
  [ -d "$d" ] && { SCL_DIR=$d; break; }
done
echo "SCL dir: ${SCL_DIR}"

for MODEL in singletrial singletrial_allevents; do
  python ${SLURM_SUBMIT_DIR}/glm_allevents_scr.py \
    --scl-dir "${SCL_DIR}" --model ${MODEL} --baselinecorrect False \
    --save-dir ${SAVE_DIR} --plot
done
