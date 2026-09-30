#!/usr/bin/env python
"""
check_scl_completeness.py

Per-subject pain-run counts at every physio stage, to confirm the SCL pipeline
is complete and to show where any subject stops:

    expected  runs labelled 'pain' in the run metadata
    acq       subject has raw .acq in physio02_sort (0/1)
    bids      physio03_bids/task-cue pain run files
    eda       SCL *baselinecorrect-False*_physio-eda.txt   (GLM input)
    json      *_onset.json                                 (GLM onsets)
    tc        *scltimecourse.csv                           (GLM trial metadata)
    qc        runs in QC_EDA_new.csv / of which 'include'

A run is GLM-ready when it has eda + json + tc. Status column:
    OK            GLM-ready runs == BIDS runs
    NO_ACQ        no raw data -> nothing to recover
    NOT_BIDS      raw exists, c02 never split it   -> c02_bidsify_SUBMIT_rerun.sh
    NO_SCL        BIDS exists, no SCL output       -> s01_extractSCL_SUBMIT_rerun.sh
    PARTIAL       some BIDS runs lack SCL output   -> check that subject's .e log
    NEEDS_QC      GLM-ready but no QC rows         -> add to QC_EDA_new.csv

Run on Discovery (biopac env):
    python check_scl_completeness.py [--qc ../../data/QC_EDA_new.csv] [--subs 30 40 57]
Writes scl_completeness.csv next to this script.
"""
import argparse, glob, re
from os.path import join, dirname, abspath, exists
import pandas as pd

HERE = dirname(abspath(__file__))
DATA = '/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio'
CUE = '/dartfs-hpc/rc/lab/C/CANlab/labdata/projects/spacetop_projects_cue'

p = argparse.ArgumentParser()
p.add_argument('--sort-dir', default=join(DATA, 'physio02_sort'))
p.add_argument('--bids-dir', default=join(DATA, 'physio03_bids', 'task-cue'))
p.add_argument('--scl-dir', default=join(CUE, 'analysis', 'physio', 'nobaseline', 'physio01_SCL'))
p.add_argument('--metadata', default=join(CUE, 'data', 'spacetop_task-social_run-metadata.csv'))
p.add_argument('--qc', default=join(HERE, '..', '..', 'data', 'QC_EDA_new.csv'))
p.add_argument('--subs', nargs='+', type=int, help='only these subject numbers')
a = p.parse_args()

def runs(pattern):
    """set of (ses, run) pairs from files matching pattern"""
    out = set()
    for f in glob.glob(pattern, recursive=True):
        s, r = re.search(r'ses-(\d+)', f), re.search(r'run-(\d+)', f)
        if s and r:
            out.add((int(s.group(1)), int(r.group(1))))
    return out

meta = pd.read_csv(a.metadata)
meta_long = meta.melt(id_vars=['sub', 'ses'], var_name='run', value_name='type')
expected = meta_long[meta_long['type'] == 'pain'].groupby('sub').size()

qc = pd.read_csv(a.qc) if exists(a.qc) else pd.DataFrame(columns=['src_subject_id', 'param_task_name', 'Signal quality'])
qc = qc[qc['param_task_name'] == 'pain']
qc['sub'] = qc['src_subject_id'].map(lambda x: f'sub-{int(x):04d}')

subs = set(expected.index) | {d.split('/')[-1] for d in glob.glob(join(a.sort_dir, 'sub-*'))}
if a.subs:
    subs = {f'sub-{n:04d}' for n in a.subs}
subs = sorted(s for s in subs if int(s[4:]) > 6)   # 1-6 are pilots

rows = []
for sub in subs:
    acq = bool(glob.glob(join(a.sort_dir, sub, '**', '*task-social*.acq'), recursive=True))
    bids = runs(join(a.bids_dir, sub, '**', f'{sub}_*run-*-pain*physio*'))
    eda = runs(join(a.scl_dir, sub, '**', '*runtype-pain_*baselinecorrect-False_samplingrate-25_physio-eda.txt'))
    js = runs(join(a.scl_dir, sub, '**', '*runtype-pain_samplingrate-2000_onset.json'))
    tc = runs(join(a.scl_dir, sub, '**', '*runtype-pain_*scltimecourse.csv'))
    ready = eda & js & tc
    q = qc[qc['sub'] == sub]
    n_inc = int((q['Signal quality'] == 'include').sum())

    if not acq and not bids:
        status = 'NO_ACQ'
    elif not bids:
        status = 'NOT_BIDS'
    elif not ready:
        status = 'NO_SCL'
    elif len(ready) < len(bids):
        status = 'PARTIAL'
    elif q.empty:
        status = 'NEEDS_QC'
    else:
        status = 'OK'
    rows.append(dict(sub=sub, expected=int(expected.get(sub, 0)), acq=int(acq), bids=len(bids),
                     eda=len(eda), json=len(js), tc=len(tc), glm_ready=len(ready),
                     qc_rows=len(q), qc_include=n_inc, status=status,
                     missing_scl=' '.join(f'ses-{s:02d}_run-{r:02d}' for s, r in sorted(bids - ready))))

df = pd.DataFrame(rows)
out = join(HERE, 'scl_completeness.csv')
df.to_csv(out, index=False)

pd.set_option('display.width', 200, 'display.max_rows', 500, 'display.max_colwidth', 60)
print(df.to_string(index=False))
print('\n=== status counts ===')
print(df['status'].value_counts().to_string())
print(f"\nGLM-ready subjects: {(df.glm_ready > 0).sum()}   "
      f"with >=1 QC 'include' run: {((df.glm_ready > 0) & (df.qc_include > 0)).sum()}")
print(f'saved -> {out}')
