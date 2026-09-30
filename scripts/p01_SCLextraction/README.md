# SCL pipeline: raw EDA → GLM-ready skin conductance

End-to-end steps for task-cue EDA, including how to add new participants and check
that nothing was dropped. Run everything on Discovery in the `biopac` conda env.

```
physio02_sort (.acq)  ──c02──►  physio03_bids (run .tsv)  ──s01──►  physio01_SCL (25 Hz EDA + onsets)
        │                                                              │
        └──────────── check_scl_completeness.py ◄───────────────────────┘
                                   │
                     QC (s03_qc) ──► data/QC_EDA_new.csv ──► p03_glm
```

## Paths (Discovery)

| What | Path |
|---|---|
| Raw, sorted `.acq` | `/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio02_sort` |
| BIDS runs | `/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue` |
| spacetop-prep (shared lab copy) | `/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/scripts/spacetop_prep` |
| Behavior | `/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/data/beh/beh02_preproc` |
| Run metadata | `/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/data/spacetop_task-social_run-metadata.csv` |
| SCL output | `/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue/analysis/physio/nobaseline/physio01_SCL` |
| Failed-run archive | `…/analysis/physio/nobaseline/physio01_SCL_failed` (+ `failed_runs.tsv`) |
| QC sheet | `data/QC_EDA_new.csv` (this repo) |

## 1. Split raw recordings into BIDS runs (spacetop-prep `c02`)

`c02_save_separate_run.py` finds run boundaries in each session's `.acq` and writes one
`.tsv` per run to `physio03_bids/task-cue`.

- All subjects: spacetop-prep `physio/c02_SUBMIT.sh` (batches of 10 subjects per array task).
- Specific subjects: `c02_bidsify_SUBMIT_rerun.sh` here. Set `--array` to the subject
  numbers (one subject per task), e.g. `sbatch --array=141,142%5 c02_bidsify_SUBMIT_rerun.sh`.

A session is skipped (`ERROR - number of complete runs do not match scan notes`) when the
number of detected runs differs from the metadata, e.g. a restarted run. Check the
segment lengths in the `c02_*.e` log (cue runs are ~401 s) and handle it by hand.

## 2. Extract SCL (`s01_extractSCL_nobaselinecorrect.py`)

Per run: downsample 2000 → 25 Hz, mean-center, band-pass 0.01–2 Hz (2nd-order
Butterworth), no trial-wise baseline correction. Writes per run to `physio01_SCL/<sub>/<ses>/`:

| File | Used by |
|---|---|
| `*_baselinecorrect-False_samplingrate-25_physio-eda.txt` | GLM signal |
| `*_samplingrate-2000_onset.json` | GLM event onsets |
| `*_scltimecourse.csv` | GLM trial metadata |

- All subjects: `s01_extractSCL_SUBMIT.sh` (batches of 10, 1.5 h limit).
- Specific subjects: `s01_extractSCL_SUBMIT_rerun.sh` (one subject per task, 6 h limit),
  e.g. `sbatch --array=141,142%5 s01_extractSCL_SUBMIT_rerun.sh`.

**Failed runs** (e.g. no event markers, 9 of 12 heat trials detected) are logged and
skipped; the subject's other runs still run. Partial files of a failed run are moved to
`physio01_SCL_failed/<sub>/<ses>/` and the error is appended to
`physio01_SCL_failed/failed_runs.tsv`. The archive sits outside `physio01_SCL` on purpose:
the GLM globs `physio01_SCL/**`, so partial files there would be picked up.

## 3. Check completeness (`check_scl_completeness.py`)

```bash
python check_scl_completeness.py                  # every subject -> scl_completeness.csv
python check_scl_completeness.py --subs 141 142   # only these
```

One row per subject with pain-run counts per stage (`expected`, `acq`, `bids`, `eda`,
`json`, `tc`, `glm_ready`, `qc_rows`, `qc_include`, `failed`) and a status:

| Status | Meaning | Action |
|---|---|---|
| `OK` | every BIDS run GLM-ready and QC'd | none |
| `NEEDS_QC` | GLM-ready, not in QC sheet | step 4 |
| `PARTIAL` | some runs missing and not failed | rerun step 2 for the subject |
| `NO_SCL` | BIDS exists, no SCL output | rerun step 2 |
| `FAILED_RUNS` | every missing run failed (see `error`) | data problem; rerunning won't help |
| `NOT_BIDS` | raw exists, never split | step 1 |
| `NO_ACQ` | no raw data | nothing to recover |

**Done** = no `PARTIAL`, `NO_SCL` or `NOT_BIDS` left. Also scan logs for crashes:
`grep -l -E "Traceback|Error|TIME LIMIT" log/*.e`.

## 4. QC (`../s03_qc/qc01_plot_eda_by_rating.py`)

Plots raw EDA of runs previously rated include / borderline / exclude next to unrated runs,
with quality metrics, and writes `qc_to_rate.csv` in the QC sheet format.

```bash
python ../s03_qc/qc01_plot_eda_by_rating.py --out-dir <dir> --subs 141 142
```

Fill `Signal quality` in `qc_to_rate.csv` and append the rows to `data/QC_EDA_new.csv`.
The GLM keeps **only** runs rated exactly `include`; `borderline/ acceptable` and any other
value (e.g. `good`) are dropped.

## 5. GLM (`../p03_glm`)

- `glm_factorial_scr.py`: manuscript model (6 stim × cue regressors); hardcoded paths, and
  crashes if a QC'd run lacks its `scltimecourse.csv`.
- `glm_allevents_scr.py --model {stim,cuestim,allevents}`: same preprocessing, optional cue
  and rating regressors; skips incomplete runs. `glm_allevents_scr.sh` runs all three.
- `glm_factorial_scr_blcompare.py`: baseline-correction comparison.

## Known pitfalls

- **`sublist()` stride bug.** spacetop-prep versions before `d244b98` (2023-05-17) skip
  every 10th subject (10, 20, 30, …) when batching; the original `c02` run predates the fix,
  so those subjects had to be re-split. With the old version, `STRIDE=1` selects nobody and
  the job finishes silently (`sub list: []`). The rerun scripts abort if they detect it.
- **Time limit.** 10 subjects per task can exceed 1.5 h; late subjects in a batch are lost
  silently. Use the rerun scripts (6 h, one subject per task).
- **`c02_SUBMIT.sh`** passes `--exclude_sub`, but the script only accepts `--exclude-sub`.
- **Conda env** is `biopac` (spacetop-prep scripts say `physio`, which does not exist).
- **Metadata** lives in `spacetop_projects_cue/data/`, not `spacetop_projects_social`.
- **Empty `sub list: []`** for a subject can also mean it has no folder in the input dir;
  check with `ls` before debugging.
