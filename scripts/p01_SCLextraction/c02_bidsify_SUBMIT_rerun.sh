#!/bin/bash -l
# Rerun BIDS run-splitting (spacetop-prep c02_save_separate_run.py) for participants
# whose raw .acq exists in physio02_sort but never reached physio03_bids/task-cue:
#   30, 40, ..., 130 -- skipped by the old sublist() stride bug (spacetop-prep d244b98,
#   2023-05-17); c02 was last run before that fix, so every multiple of 10 was dropped.
# One subject per array task (SLURM_ARRAY_TASK_ID == subject number, STRIDE=1).
# Uses the shared lab copy of spacetop-prep unchanged.
# Next: sbatch --array=30,40,50,60,70,80,90,100,120,130%5 s01_extractSCL_SUBMIT_rerun.sh
#SBATCH --job-name=physio_c02
#SBATCH --nodes=1
#SBATCH --task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=04:00:00
#SBATCH -o ./log/c02_%A_%a.o
#SBATCH -e ./log/c02_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=30,40,50,60,70,80,90,100,120,130%5

conda activate biopac

PREP=/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/scripts/spacetop_prep/physio
TOPDIR="/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio"
METADATA="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/data/spacetop_task-social_run-metadata.csv"

# With the pre-fix sublist(), STRIDE=1 selects nobody -- stop instead of finishing silently.
if grep -q "slurm_id \* stride + 1" ${PREP}/utils/initialize.py; then
  echo "ABORT: ${PREP}/utils/initialize.py has the old sublist() (stride bug)."; exit 1
fi
SUB=$(printf "sub-%04d" ${SLURM_ARRAY_TASK_ID})
ls -d ${TOPDIR}/physio02_sort/${SUB} || { echo "no physio02_sort for ${SUB}"; exit 0; }

cd ${PREP}   # c02 reads the rename json files relative to here
python ${PREP}/c02_save_separate_run.py \
--topdir ${TOPDIR} \
--metadata ${METADATA} \
--slurm_id ${SLURM_ARRAY_TASK_ID} \
--stride 1 \
--sub-zeropad 4 \
--task task-social \
--run-cutoff 300 \
--colnamechange ${PREP}/c02_changecolumn.json \
--tasknamechange ${PREP}/c02_changetaskname.json \
--exclude-sub 1 2 3 4 5 6

ls ${TOPDIR}/physio03_bids/task-cue/${SUB}/*/ 2>/dev/null | head
