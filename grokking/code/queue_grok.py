"""GPU-day queue: run every (world, seed) pair for a spec, seed-major (all
worlds at seed=1 first, so a partial run still gives >=1 seed per world),
and write COMPLETE.json per run with the final held-out accuracy of every
edge/face, the placebo accuracy, the last train-batch token accuracy and a
grok-moment estimate per cell (first checkpoint whose held-out accuracy on
zz_analyze's own trajectory crosses 0.9).

python queue_grok.py --spec MODP --worlds 701,702 --seeds 1,2,3,4,5
python queue_grok.py --spec MODP_ADDONLY --worlds 703,704 --seeds 1,2,3,4,5 --preset classic
python queue_grok.py --spec MODP --worlds 701 --seeds 1 --dryrun   (smoke test)
"""
from __future__ import annotations
import argparse
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

GROK_THRESHOLD = 0.9


def run_args(spec, world, seed, out, **overrides):
    """Matches run_grok.py's argparse Namespace shape. Hyperparameters default
    to None so run_grok.run()'s own --preset/spec-default resolution applies
    (queue_grok only forces a value when the caller explicitly overrides it),
    exactly as if these were unset CLI flags."""
    base = dict(spec=spec, world=world, seed=seed, steps=None, eval_every=None, ckpt_every=None,
                wd=None, lr=None, frac=None, layers=None, d_model=None, n_heads=None, d_ff=None,
                batch_size=None, preset=None, only_tasks=None, out=str(out), dryrun=False,
                dryrun_p=None, cpu=False, resume=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def complete_json(spec, world, out_dir):
    complex_path = GROK_DIR / 'data' / spec / f'world{world}' / 'COMPLEX.json'
    complex_def = json.loads(complex_path.read_text(encoding='utf8'))
    res = ZA.analyze_run(out_dir, str(complex_path), n_perm=0)
    steps, traj = res['steps'], res['trajectory']
    cells = complex_def['taught_edges'] + complex_def['faces'] + complex_def['control_cells']

    def grok_step(cell):
        for s, v in zip(steps, traj.get(cell, [])):
            if v >= GROK_THRESHOLD:
                return s
        return None
    final_acc = {c: (traj[c][-1] if traj.get(c) else None) for c in cells}
    grok_at = {c: grok_step(c) for c in complex_def['taught_edges'] + complex_def['faces']}

    train_log = out_dir / 'TRAIN_LOG.jsonl'
    train_acc = None
    if train_log.exists():
        lines = [json.loads(s) for s in train_log.read_text(encoding='utf8').splitlines() if s.strip()]
        if lines:
            train_acc = lines[-1]['train_tok_acc']

    out = dict(spec=spec, world=world, out_dir=str(out_dir), steps=steps,
               final_heldout_accuracy=final_acc, placebo_accuracy=final_acc.get('P_X'),
               train_token_accuracy_last_batch=train_acc, grok_step_estimate=grok_at,
               grok_threshold=GROK_THRESHOLD,
               note='grok_step_estimate = first evaluated checkpoint whose held-out cell score (the '
                    'same score zz_analyze.py computes: min(override) for faces, per-task accuracy for '
                    'edges) is >= grok_threshold; this is a single-threshold convenience estimate, NOT '
                    'the pre-registered multi-threshold/stability event criterion in '
                    'PREREGISTRATION_GROKKING_DRAFT.md.')
    (out_dir / 'COMPLETE.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf8')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='MODP')
    ap.add_argument('--worlds', default='701,702')
    ap.add_argument('--seeds', default='1,2,3,4,5')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--eval-every', type=int, default=None)
    ap.add_argument('--ckpt-every', type=int, default=None)
    ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--preset', choices=['classic'], default=None,
                     help='classic: 1 layer, full-batch AdamW, wd=1.0, 50000 steps -- see run_grok.py '
                          '--preset and README "最小规格与经典设置". Explicit flags above still win.')
    ap.add_argument('--runs-dir', default=None)
    ap.add_argument('--dryrun', action='store_true')
    a = ap.parse_args()
    worlds = [int(w) for w in a.worlds.split(',')]
    seeds = [int(s) for s in a.seeds.split(',')]
    runs_dir = Path(a.runs_dir) if a.runs_dir else GROK_DIR / 'runs'

    overrides = dict(preset=a.preset)
    if a.wd is not None:
        overrides['wd'] = a.wd
    if a.layers is not None:
        overrides['layers'] = a.layers
    if a.steps:
        overrides['steps'] = a.steps
    if a.eval_every:
        overrides['eval_every'] = a.eval_every
    if a.ckpt_every:
        overrides['ckpt_every'] = a.ckpt_every
    if a.dryrun:
        overrides.update(dryrun=True, dryrun_p=13)

    results = []
    for seed in seeds:            # seed-major: every world gets seed=1 before anyone gets seed=2
        for world in worlds:
            out_dir = runs_dir / f'{a.spec}_world{world}_seed{seed}'
            print(f'=== {a.spec} world{world} seed{seed} -> {out_dir} ===')
            args = run_args(a.spec, world, seed, out_dir, **overrides)
            RG.run(args)
            res = complete_json(a.spec, world, out_dir)
            results.append(res)
            print(f'  grok_step_estimate={res["grok_step_estimate"]}')
    (runs_dir / f'{a.spec}_QUEUE_SUMMARY.json').write_text(
        json.dumps(dict(spec=a.spec, worlds=worlds, seeds=seeds, results=results), ensure_ascii=False, indent=1),
        encoding='utf8')


if __name__ == '__main__':
    main()
