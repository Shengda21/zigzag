"""GPU-day hyperparameter scan: p=97, addition only (P_S), grid over
weight_decay x train_fraction, to find the region where grokking actually
happens before committing GPU time to the full 701/702 x seed1-5 queue.

This script only *decides execution parameters* for queue_grok.py -- it is
not part of the confirmatory pipeline and its output is never used to judge
whether an event occurred (see PREREGISTRATION_GROKKING_DRAFT.md, "扫描...
只决定执行参数，不参与判定").

python scan_grok.py                          # full grid, 20000 steps each
python scan_grok.py --dryrun                 # CPU smoke test, p=13, 300 steps
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
GROK_DIR = HERE.parent
ROOT = GROK_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import run_grok as RG    # noqa: E402
import zz_analyze as ZA  # noqa: E402

WD_GRID = (0.1, 0.5, 1.0)
FRAC_GRID = (0.3, 0.4, 0.5)
GROK_THRESHOLD = 0.9
SCAN_WORLD = 701
SCAN_TASK = 'P_S'


def one_cell(spec, world, wd, frac, steps, eval_every, ckpt_every, out_root, dryrun_p=None):
    out_dir = out_root / f'scan_wd{wd}_frac{frac}'
    args = SimpleNamespace(spec=spec, world=world, seed=1, steps=steps, eval_every=eval_every,
                            ckpt_every=ckpt_every, wd=wd, lr=1e-3, frac=frac, layers=1, d_model=128,
                            n_heads=4, d_ff=512, batch_size=512, only_tasks=SCAN_TASK, out=str(out_dir),
                            dryrun=False, dryrun_p=dryrun_p, cpu=False, resume=True)
    RG.run(args)
    complex_path = GROK_DIR / 'data' / spec / f'world{world}' / 'COMPLEX.json'
    res = ZA.analyze_run(out_dir, str(complex_path), n_perm=0)
    steps_list, traj = res['steps'], res['trajectory'][SCAN_TASK]
    grok_step = next((s for s, v in zip(steps_list, traj) if v >= GROK_THRESHOLD), None)
    return dict(wd=wd, frac=frac, out_dir=str(out_dir), final_heldout_acc=traj[-1] if traj else None,
                grok_step=grok_step, grokked=grok_step is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='MODP')
    ap.add_argument('--world', type=int, default=SCAN_WORLD)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--ckpt-every', type=int, default=2000)
    ap.add_argument('--runs-dir', default=None)
    ap.add_argument('--dryrun', action='store_true')
    a = ap.parse_args()
    steps, eval_every, ckpt_every, dryrun_p = a.steps, a.eval_every, a.ckpt_every, None
    if a.dryrun:
        steps, eval_every, ckpt_every, dryrun_p = 300, 50, 150, 13
    runs_dir = Path(a.runs_dir) if a.runs_dir else GROK_DIR / 'runs' / 'scan'
    runs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for wd, frac in itertools.product(WD_GRID, FRAC_GRID):
        print(f'=== scan wd={wd} frac={frac} ===')
        r = one_cell(a.spec, a.world, wd, frac, steps, eval_every, ckpt_every, runs_dir, dryrun_p=dryrun_p)
        results.append(r)
        print(f'  grok_step={r["grok_step"]} final_acc={r["final_heldout_acc"]}')

    grid = {f'wd{r["wd"]}_frac{r["frac"]}': ('grokked@' + str(r['grok_step'])) if r['grokked'] else '未grok'
            for r in results}
    summary = dict(spec=a.spec, world=a.world, task=SCAN_TASK, steps=steps, wd_grid=WD_GRID, frac_grid=FRAC_GRID,
                   grok_threshold=GROK_THRESHOLD, results=results, grid=grid,
                   scope='execution-parameter scan only; not part of the confirmatory pipeline')
    (runs_dir / 'SCAN_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf8')
    print(json.dumps(grid, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
