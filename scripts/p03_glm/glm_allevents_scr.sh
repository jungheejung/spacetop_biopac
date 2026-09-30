#!/bin/bash -l
#SBATCH --job-name=scr_allevents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=03:00:00
#SBATCH -o ./log/scr_allevents_%j.o
#SBATCH -e ./log/scr_allevents_%j.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

# Fit the SCR GLM with three regressor sets on the no-baseline SCL:
#   stim (manuscript), cuestim (+cue), allevents (+cue, +ratings; fMRI-like)
# Submit from scripts/p03_glm:   mkdir -p log && sbatch glm_allevents_scr.sh

conda activate biopac

CUE=/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue
SAVE_DIR=${CUE}/analysis/physio/glm_allevents

# SCL dir moved between commits; use the first candidate that exists.
for d in ${CUE}/analysis/physio/nobaseline/physio01_SCL ${CUE}/analysis/physio_nobaseline/physio01_SCL; do
  [ -d "$d" ] && { SCL_DIR=$d; break; }
done
echo "SCL dir: ${SCL_DIR}"

for MODEL in stim cuestim allevents; do
  python ${SLURM_SUBMIT_DIR}/glm_allevents_scr.py \
    --scl-dir "${SCL_DIR}" --model ${MODEL} --baselinecorrect False --save-dir ${SAVE_DIR}
done
