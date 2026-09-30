#!/bin/bash -l
#SBATCH --job-name=scr_blcompare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH -o ./log/scr_blcompare_%j.o
#SBATCH -e ./log/scr_blcompare_%j.e
#SBATCH --account=DBIC
#SBATCH --partition=standard

# Run the current SCR factorial GLM on both SCL variants (trial-wise baseline
# corrected vs not), so the two beta tables differ only in baseline correction.
# Submit from scripts/p03_glm:   mkdir -p log && sbatch glm_factorial_scr_blcompare.sh

conda activate biopac

CUE=/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue
SAVE_DIR=${CUE}/analysis/physio/glm_baselinecompare

# SCL dirs moved between commits; use the first candidate that exists.
first_dir () { for d in "$@"; do [ -d "$d" ] && { echo "$d"; return; }; done; }
SCL_TRUE=$(first_dir  ${CUE}/analysis/physio/physio01_SCL_25s \
                      ${CUE}/analysis/physio/physio01_SCL)
SCL_FALSE=$(first_dir ${CUE}/analysis/physio_nobaseline/physio01_SCL \
                      ${CUE}/analysis/physio/nobaseline/physio01_SCL)

for pair in "True:${SCL_TRUE}" "False:${SCL_FALSE}"; do
  BL=${pair%%:*}; DIR=${pair#*:}
  N=$(find "${DIR}" -name "*pain_epochstart--3_epochend-20_baselinecorrect-${BL}_samplingrate-25_physio-eda.txt" 2>/dev/null | wc -l)
  echo "baselinecorrect-${BL}: ${DIR}  (${N} pain runs)"
  if [ -z "${DIR}" ] || [ "${N}" -eq 0 ]; then
    echo "  !! no baselinecorrect-${BL} files found -- skipping"; continue
  fi
  python ${SLURM_SUBMIT_DIR}/glm_factorial_scr_blcompare.py \
    --scl-dir "${DIR}" --baselinecorrect ${BL} --save-dir ${SAVE_DIR}
done
