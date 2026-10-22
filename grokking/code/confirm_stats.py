"""Confirmation-stage statistics for the MODP grokking line (see
PREREGISTRATION_GROKKING_DRAFT.md "6. 确认阶段判据 (GPU 证据后修订)"). Built
entirely on top of the unmodified crosslingual/code/zz_analyze.py:
analyze_run()'s own event_ledger() already computes "first checkpoint >=
threshold, stable at the next checkpoint too" per cell per theta -- this
module just reads that ledger for the specific rule the confirmation stage
uses (only the 7 arithmetic edges, a +-4-checkpoint tolerance window instead
of the analyzer's built-in +-1, a placebo-must-not-enter check, H0 merge
events, and multi-seed reproduction). No changes to zz_analyze.py.

python confirm_stats.py <run_dir> <complex.json> [--threshold 0.9] [--tol-checkpoints 4] [--perm 1000]
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROK_DIR = HERE.parent
ROOT = GROK_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import zz_analyze as ZA  # noqa: E402

# The 7 edges GPU evidence (701/702, --preset classic, 100k steps) showed
# actually require arithmetic to answer: 4 direct ops (P_S add, P_D sub,
# P_M mul, P_Q div) + 3 "same primitive on a transformed pair" compositions
# (Pn_D, Ps_S, Pi_Q). The other real edges (P_N, P_Pn, P_Ps, P_I, P_Pi) are
# single-element negation/inversion or pair construction/swap -- simpler,
# not part of the primary confirmatory event set, still reported descriptively.
ARITHMETIC_EDGES = ('P_S', 'P_D', 'P_M', 'P_Q', 'Pn_D', 'Ps_S', 'Pi_Q')
DEFAULT_THRESHOLD = 0.9
PLACEBO_CELL = 'P_X'


def edge_birth_steps(res, edges=ARITHMETIC_EDGES, threshold=DEFAULT_THRESHOLD):
    """{edge: first_step or None}. 'birth' = event_ledger's own
    first_step >= threshold AND stable (present at the next checkpoint
    too) -- read directly off analyze_run()'s ledger, nothing re-derived."""
    out = {}
    for e in edges:
        entry = res['ledger']['cells'].get(e, {}).get(str(threshold))
        out[e] = entry['first_step'] if entry and entry.get('first_step') is not None and entry.get('stable') else None
    return out


def arithmetic_synchrony(res, edges=ARITHMETIC_EDGES, threshold=DEFAULT_THRESHOLD, tol_checkpoints=4):
    """Confirmation-stage synchrony statistic: '算术边在 +-4 个检查点窗口内
    进入的条数'. For each observed checkpoint, count how many of `edges`
    have a stable birth step within +-tol_checkpoints*eval_every of it;
    return the checkpoint with the maximum count."""
    steps = res['steps']
    births = edge_birth_steps(res, edges, threshold)
    if len(steps) < 2:
        return dict(best_step=None, count=0, tol_steps=None, tol_checkpoints=tol_checkpoints,
                     eval_every=None, per_edge=births, threshold=threshold)
    eval_every = steps[1] - steps[0]
    tol_steps = tol_checkpoints * eval_every
    counts = {s: sum(1 for v in births.values() if v is not None and abs(v - s) <= tol_steps) for s in steps}
    best_step = max(counts, key=lambda s: counts[s])
    return dict(best_step=best_step, count=counts[best_step], tol_steps=tol_steps,
                tol_checkpoints=tol_checkpoints, eval_every=eval_every, per_edge=births, threshold=threshold)


def placebo_never_enters(res, threshold=DEFAULT_THRESHOLD, cell=PLACEBO_CELL):
    """Confirmation-stage requirement '安慰剂 P_X 不得进入': True (pass) iff
    P_X never has a stable birth at `threshold`."""
    entry = res['ledger']['cells'].get(cell, {}).get(str(threshold))
    entered = bool(entry and entry.get('first_step') is not None and entry.get('stable'))
    return dict(never_enters=not entered, threshold=threshold, cell=cell, ledger_entry=entry)


def h0_merge_events(res, cells=None):
    """Per-threshold list of transitions cw_zigzag.classify_h0_change tagged
    'merge' (a new edge connected two previously-separate components),
    already computed by analyze_run()'s per_theta[th]['transitions'];
    optionally restricted to transitions whose added cells intersect
    `cells` (e.g. the 7 arithmetic edges)."""
    out = {}
    for th, v in res['per_theta'].items():
        merges = [t for t in v['transitions'] if t.get('h0') == 'merge']
        if cells is not None:
            merges = [t for t in merges if set(t.get('added', ())) & set(cells)]
        out[th] = merges
    return out


def multi_seed_reproduction(births_by_seed, edges=ARITHMETIC_EDGES, tol_checkpoints=4,
                             eval_every=None, min_seed_frac=0.8):
    """births_by_seed: {seed: {edge: step_or_None}}, e.g. one
    edge_birth_steps(...) call per seed's analyze_run result. An edge
    'reproduces' (confirmation-stage '跨 >=4/5 种子复现') if it has a stable
    birth in >= min_seed_frac (default 0.8 = 4/5) of the seeds AND those
    birth steps all land within +-tol_checkpoints*eval_every of their
    median (eval_every must be supplied -- it is not carried by
    births_by_seed itself)."""
    n_seeds = len(births_by_seed)
    need = max(1, int(round(min_seed_frac * n_seeds)))
    tol_steps = tol_checkpoints * eval_every if eval_every else None
    out = {}
    for e in edges:
        vals = [births_by_seed[s][e] for s in births_by_seed if births_by_seed[s].get(e) is not None]
        med = statistics.median(vals) if vals else None
        within_tol = all(abs(v - med) <= tol_steps for v in vals) if (vals and tol_steps is not None) else None
        reproduced = len(vals) >= need and (within_tol is not False)
        out[e] = dict(n_seeds_total=n_seeds, n_seeds_born=len(vals), need=need, steps=vals,
                       median=med, within_tol=within_tol, reproduced=reproduced)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('complex_json')
    ap.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument('--tol-checkpoints', type=int, default=4)
    ap.add_argument('--perm', type=int, default=0)
    a = ap.parse_args()
    res = ZA.analyze_run(a.run_dir, a.complex_json, n_perm=a.perm)
    births = edge_birth_steps(res, threshold=a.threshold)
    sync = arithmetic_synchrony(res, threshold=a.threshold, tol_checkpoints=a.tol_checkpoints)
    placebo = placebo_never_enters(res, threshold=a.threshold)
    h0 = h0_merge_events(res, cells=ARITHMETIC_EDGES)
    out = dict(run=str(a.run_dir), threshold=a.threshold, edge_births=births, synchrony=sync,
               placebo=placebo, h0_merge_events_arithmetic={k: len(v) for k, v in h0.items()})
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
