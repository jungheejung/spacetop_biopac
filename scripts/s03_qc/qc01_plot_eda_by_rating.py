#!/usr/bin/env python
"""
qc01_plot_eda_by_rating.py

Visual EDA QC aid: plot pain runs the original rater removed (exclude, borderline)
next to runs they kept (include), in one format, plus runs not yet rated, so new
runs can be rated against the same reference.

For each run (raw physio_eda from physio03_bids, block-averaged 2000 -> 25 Hz):
  - time course in microsiemens with heat periods shaded (event_stimuli channel)
  - simple quality metrics, printed in the panel title and saved to CSV:
      mean_uS, sd_uS        level / variability
      pct_flat              % of 1-s windows with SD < 0.002 uS (flat / disconnected)
      pct_outlier           % samples > 5 MAD from run median (what the GLM interpolates)
      jumps_per_min         abrupt steps: |diff| > 10 x MAD(diff) and > 0.05 uS
      heat_resp_z           mean (2-8 s post-onset minus 3 s pre-onset) / run SD
      n_heat                detected heat onsets (12 expected)

Outputs (--out-dir):
  qc_<label>.pdf            6 runs per page; labels: include, borderline, exclude, unrated
  qc_metrics.csv            one row per run
  qc_metrics_by_label.png   metric distributions by label (unrated overlaid)
  qc_to_rate.csv            unrated runs in QC_EDA_new.csv format -> fill 'Signal quality'

Run on Discovery (biopac env); reading every run takes a while, so use sbatch or an
interactive node:
  python qc01_plot_eda_by_rating.py --out-dir <dir>
  python qc01_plot_eda_by_rating.py --out-dir <dir> --n-ref 0            # all rated runs
  python qc01_plot_eda_by_rating.py --out-dir <dir> --subs 30 40 60 122  # rate only these
"""
import argparse, glob, re
from os.path import join, dirname, abspath, basename
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = dirname(abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--bids-dir', default='/dartfs-hpc/rc/lab/C/CANlab/labdata/data/spacetop_data/physio/physio03_bids/task-cue')
p.add_argument('--qc', default=join(HERE, '..', '..', 'data', 'QC_EDA_new.csv'))
p.add_argument('--out-dir', required=True)
p.add_argument('--subs', nargs='+', type=int, help='unrated subjects to plot (default: every unrated run)')
p.add_argument('--n-ref', type=int, default=40, help='rated runs sampled per label; 0 = all')
p.add_argument('--seed', type=int, default=1)
a = p.parse_args()
Path(a.out_dir).mkdir(parents=True, exist_ok=True)

FS, FS_OUT = 2000, 25
BLOCK = FS // FS_OUT
LABELS = ['include', 'borderline', 'exclude', 'unrated']
COLOR = {'include': '#2a78d6', 'borderline': '#eb6834', 'exclude': '#1baf7a', 'unrated': '#eda100'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5df'

# ---- run list + QC label ------------------------------------------------------
files = sorted(glob.glob(join(a.bids_dir, 'sub-*', 'ses-*', '*task-cue_run-*-pain_*physio*.tsv*')))
runs = pd.DataFrame({'fpath': files})
runs['sub'] = runs.fpath.str.extract(r'sub-(\d+)').astype(int)
runs['ses'] = runs.fpath.str.extract(r'ses-(\d+)').astype(int)
runs['run'] = runs.fpath.str.extract(r'run-(\d+)').astype(int)
runs = runs[runs['sub'] > 6]                                    # 1-6 are pilots

qc = pd.read_csv(a.qc)
qc = qc[qc.param_task_name == 'pain'].rename(columns={
    'src_subject_id': 'sub', 'session_id': 'ses', 'param_run_num': 'run', 'Signal quality': 'rating'})
qc['label'] = qc['rating'].astype(str).str.lower().map(
    lambda r: 'include' if r in ('include', 'good') else 'borderline' if r.startswith('borderline')
    else 'exclude' if r == 'exclude' else 'unrated')
runs = runs.merge(qc[['sub', 'ses', 'run', 'label']], on=['sub', 'ses', 'run'], how='left')
runs['label'] = runs['label'].fillna('unrated')

rated = runs[runs.label != 'unrated']
if a.n_ref > 0:
    rated = rated.groupby('label', group_keys=False).apply(
        lambda g: g.sample(min(len(g), a.n_ref), random_state=a.seed))
unrated = runs[runs.label == 'unrated']
if a.subs:
    unrated = unrated[unrated['sub'].isin(a.subs)]
todo = pd.concat([rated, unrated]).sort_values(['label', 'sub', 'ses', 'run']).reset_index(drop=True)
print('runs to plot:\n' + todo.label.value_counts().reindex(LABELS).fillna(0).astype(int).to_string())

# ---- load + metrics ---------------------------------------------------------------
def load(fpath):
    d = pd.read_csv(fpath, sep='\t', usecols=['physio_eda', 'event_stimuli'])
    n = len(d) // BLOCK * BLOCK
    eda = d.physio_eda.to_numpy()[:n].reshape(-1, BLOCK).mean(1)
    stim = d.event_stimuli.to_numpy()[:n].reshape(-1, BLOCK).max(1) > 2.5
    return eda, stim

def metrics(eda, stim):
    med = np.median(eda); mad = np.median(np.abs(eda - med)) or np.nan
    win = eda[:len(eda) // FS_OUT * FS_OUT].reshape(-1, FS_OUT)
    d = np.diff(eda); dmad = np.median(np.abs(d - np.median(d)))
    jumps = np.sum((np.abs(d) > 10 * dmad) & (np.abs(d) > 0.05))
    onsets = np.flatnonzero(np.diff(stim.astype(int)) == 1) + 1
    sd = eda.std()
    resp = [eda[o + 2 * FS_OUT:o + 8 * FS_OUT].mean() - eda[o - 3 * FS_OUT:o].mean()
            for o in onsets if o >= 3 * FS_OUT and o + 8 * FS_OUT <= len(eda)]
    return dict(mean_uS=eda.mean(), sd_uS=sd,
                pct_flat=100 * np.mean(win.std(1) < 0.002),
                pct_outlier=100 * np.mean(np.abs(eda - med) > 5 * mad),
                jumps_per_min=jumps / (len(eda) / FS_OUT / 60),
                heat_resp_z=np.mean(resp) / sd if resp and sd > 0 else np.nan,
                n_heat=len(onsets))

rows, traces = [], {}
for i, r in todo.iterrows():
    try:
        eda, stim = load(r.fpath)
    except Exception as e:
        print(f'  skip {basename(r.fpath)}: {e}')
        continue
    traces[i] = (eda, stim)
    rows.append({**r.drop('fpath').to_dict(), **metrics(eda, stim), 'file': basename(r.fpath)})
    if len(rows) % 25 == 0:
        print(f'  loaded {len(rows)}/{len(todo)}')
m = pd.DataFrame(rows, index=list(traces))
m.to_csv(join(a.out_dir, 'qc_metrics.csv'), index=False)

# ---- galleries ------------------------------------------------------------------
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
PER_PAGE = 6
for label in LABELS:
    sel = m[m.label == label]
    if sel.empty:
        continue
    with PdfPages(join(a.out_dir, f'qc_{label}.pdf')) as pdf:
        for start in range(0, len(sel), PER_PAGE):
            page = sel.iloc[start:start + PER_PAGE]
            fig, axes = plt.subplots(PER_PAGE, 1, figsize=(8.5, 11), squeeze=False)
            for ax, (i, r) in zip(axes[:, 0], page.iterrows()):
                eda, stim = traces[i]
                t = np.arange(len(eda)) / FS_OUT
                ax.fill_between(t, 0, 1, where=stim, transform=ax.get_xaxis_transform(),
                                color=GRID, linewidth=0)
                ax.plot(t, eda, color=COLOR[label], linewidth=0.8)
                ax.set_title(f"sub-{r['sub']:04d} ses-{r.ses:02d} run-{r.run:02d}  [{label}]   "
                             f"flat {r.pct_flat:.0f}%  outlier {r.pct_outlier:.1f}%  "
                             f"jumps {r.jumps_per_min:.1f}/min  heat resp {r.heat_resp_z:.2f} SD  "
                             f"n_heat {r.n_heat}", loc='left', fontsize=7.5, color=INK)
                ax.set_xlim(0, t[-1]); ax.set_ylabel('EDA (uS)')
                ax.spines[['top', 'right']].set_visible(False)
            for ax in axes[len(page):, 0]:
                ax.axis('off')
            axes[min(len(page), PER_PAGE) - 1, 0].set_xlabel('time (s)   shaded = heat')
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f'saved qc_{label}.pdf ({len(sel)} runs)')

# ---- metric distributions by label --------------------------------------------------
cols = ['sd_uS', 'pct_flat', 'pct_outlier', 'jumps_per_min', 'heat_resp_z', 'n_heat']
present = [l for l in LABELS if (m.label == l).any()]
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
rng = np.random.default_rng(a.seed)
for ax, c in zip(axes.flat, cols):
    for k, l in enumerate(present):
        v = m.loc[m.label == l, c].dropna()
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=COLOR[l],
                   edgecolor='white', linewidth=0.5, alpha=0.85)
        if len(v):
            ax.hlines(v.median(), k - 0.3, k + 0.3, color=INK, linewidth=2)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present)
    ax.set_title(c, loc='left', color=INK)
    ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('EDA quality metrics by QC rating (bar = median)', x=0.01, ha='left', color=INK)
fig.tight_layout()
fig.savefig(join(a.out_dir, 'qc_metrics_by_label.png'), dpi=150)
print('saved qc_metrics_by_label.png')
print(m.groupby('label')[cols].median().reindex(present).round(3).to_string())

# ---- rating template --------------------------------------------------------------------
tmpl = m[m.label == 'unrated'][['sub', 'ses', 'run']].rename(columns={
    'sub': 'src_subject_id', 'ses': 'session_id', 'run': 'param_run_num'})
tmpl['param_task_name'] = 'pain'
tmpl['Signal quality'] = ''
tmpl['Notes'] = ''
tmpl.to_csv(join(a.out_dir, 'qc_to_rate.csv'), index=False)
print(f"saved qc_to_rate.csv ({len(tmpl)} runs to rate)")
