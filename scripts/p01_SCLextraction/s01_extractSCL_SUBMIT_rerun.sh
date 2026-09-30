#!/bin/bash -l
# Rerun SCL extraction (no baseline) for participants never processed:
#   30..130 (x10): skipped by the old sublist() stride bug (fixed in spacetop-prep d244b98, 2023-05-17)
#   57 58 59 68 127 129: batch jobs hit the 1.5 h limit before reaching them
#   122 132 133: have SCL output but need QC; rerun keeps outputs consistent
# Requires the POST-fix sublist() on Discovery; with the old one, STRIDE=1 selects nobody.
# Check: python -c "import inspect,spacetop_prep.physio.utils.initialize as i;print(inspect.getsource(i.sublist))"
#SBATCH --job-name=physio
#SBATCH --nodes=1
#SBATCH --task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=06:00:00
#SBATCH -o ./log/physio03_%A_%a.o
#SBATCH -e ./log/physio03_%A_%a.e
#SBATCH --account=DBIC
#SBATCH --partition=standard
#SBATCH --array=30,40,50,60,70,80,90,100,120,130,57,58,59,68,127,129,122,132,133%5

conda activate biopac

PROJECT_DIR="/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue"
PHYSIO_DIR="/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue"
BEH_DIR="${PROJECT_DIR}/data/beh/beh02_preproc"
OUTPUT_LOGDIR="${PROJECT_DIR}/scripts/logcenter"
OUTPUT_SAVEDIR="${PROJECT_DIR}/analysis/physio/nobaseline"
METADATA="${PROJECT_DIR}/data/spacetop_task-social_run-metadata.csv"
CHANNELJSON="/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/scripts/spacetop_prep/physio/p01_channel.json"
SLURM_ID=${SLURM_ARRAY_TASK_ID}
STRIDE=1   # one subject per array task: SLURM_ARRAY_TASK_ID == subject number
ZEROPAD=4
TASK="task-cue"
SAMPLINGRATE=2000
DOWNSAMPLE=25
TTL_INDEX=1
SCL_EPOCH_START=-3
SCL_EPOCH_END=20

python ${PWD}/s01_extractSCL_nobaselinecorrect.py \
--input-physiodir ${PHYSIO_DIR} \
--input-behdir ${BEH_DIR} \
--output-logdir ${OUTPUT_LOGDIR} \
--output-savedir ${OUTPUT_SAVEDIR} \
--metadata ${METADATA} \
--dictchannel ${CHANNELJSON} \
--slurm-id ${SLURM_ID} \
--slurm-stride ${STRIDE} \
--bids-zeropad ${ZEROPAD} \
--bids-task ${TASK} \
--event-name "event_stimuli" \
--prior-event "event_expectrating" \
--later-event "event_actualrating" \
--source-samplingrate ${SAMPLINGRATE} \
--dest-samplingrate ${DOWNSAMPLE} \
--scl-epochstart ${SCL_EPOCH_START} \
--scl-epochend ${SCL_EPOCH_END} \
--ttl-index ${TTL_INDEX} \
--baselinecorrect "False" \
--exclude-sub 1 2 3 4 5 6
