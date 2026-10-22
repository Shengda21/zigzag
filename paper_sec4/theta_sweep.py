"""Threshold sweep and delta-robust event counts for the FORGET3 arms (paper: threshold-stability section).

(1) Fine threshold grid: for theta in 0.50..0.95 (step 0.01) the union-bridge H1 barcode of the K_B subcomplex,
    stored with node indices (even = checkpoint, odd = bridge).
(2) Bootstrap perturbation size: item-cluster bootstrap (B = 1000) of all cell scores; for each replicate the
    sup-norm distance to the point estimate over all cells and checkpoints, and its 0.95 quantile; summary statistics.
(3) delta-robust cokernel counts along the union zigzag at theta = 0.8 (and 0.6..0.9) for delta in {0, 0.02, 0.05, 0.10}:
      kappa_fwd  = classes of H_k(K_i u K_{i+1}) alive across [theta-delta, theta+delta] not reached from K_i(theta-delta)
      kappa_bwd  = same, not reached from K_{i+1}(theta-delta)
      kappa_both = same, not reached from K_i(theta-delta) (+) K_{i+1}(theta-delta)   (robust bridge-only classes)
    births_i = kappa_fwd - kappa_both, deaths_i = kappa_bwd - kappa_both.  Reported for k = 0, 1.

  python theta_sweep.py  ->  theta_sweep.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5 / 'crosslingual' / 'code'))
import cw_zigzag as CW      # noqa: E402
import zz_analyze as ZA     # noqa: E402
import zz_robust as R       # noqa: E402

RUNS = V5 / 'cloud_return_round4' / 'complex_worlds' / 'runs' / 'RCN12_FORGET3'
FINE = [round(0.50 + 0.01 * i, 2) for i in range(46)]            # 0.50 .. 0.95
DELTAS = [0.0, 0.02, 0.05, 0.10]
B, SEED = 1000, 20260925


def image_cols(cx, src, tgt, k):
    return [int(c) for c in cx.induced(src, tgt, k)]


def kappa(cx, w_hi, w_lo, sources_lo, k):
    """dim (I + J) - dim J with I = im(H_k(w_hi) -> H_k(w_lo)), J = sum_j im(H_k(src_j) -> H_k(w_lo))."""
    I = image_cols(cx, w_hi, w_lo, k)
    J = []
    for s in sources_lo:
        J.extend(image_cols(cx, s, w_lo, k))
    return CW.rank(I + J) - CW.rank(J)


def robust_counts(cx, masks_by_theta, theta, delta, k):
    """per transition i: (births, deaths, bridge_only) at scale delta."""
    lo, hi = masks_by_theta(theta - delta), masks_by_theta(theta + delta)
    T = len(lo)
    out = []
    for i in range(T - 1):
        w_lo, w_hi = lo[i] | lo[i + 1], hi[i] | hi[i + 1]
        both = kappa(cx, w_hi, w_lo, [lo[i], lo[i + 1]], k)
        fwd = kappa(cx, w_hi, w_lo, [lo[i]], k)
        bwd = kappa(cx, w_hi, w_lo, [lo[i + 1]], k)
        out.append((fwd - both, bwd - both, both))
    return out


def analyse_arm(run_dir, spec, full_boot):
    evals = ZA.load_run(run_dir)
    sm = R.ScoreModel(spec, evals)
    cx_full = CW.CWComplex([(n, d, list(b)) for n, d, b in spec['cells']])
    sub = R.Sub(spec['cells'], R.K_B_CELLS).bind(cx_full)
    mk = R.Masker(cx_full)
    S = sm.scores(sm.weights(B if full_boot else 0, SEED))
    arr = np.stack([S[n] for n in cx_full.names], axis=1)         # [B+1, cells, T]
    steps = sm.steps
    res = dict(run=run_dir.name, steps=steps)
    # (1) fine threshold sweep, K_B union H1 barcode (point estimate)
    sweep = {}
    for th in FINE:
        masks = [sub.remap(m) for m in mk.masks(arr[0], th)]
        sweep[f'{th:.2f}'] = [list(b) for b in R.fast_barcode(sub.cx, masks, 1, 'union')]
    res['sweep_KB_H1'] = sweep
    # (2) bootstrap perturbation sizes
    if full_boot:
        dev = np.abs(arr[1:] - arr[0][None])                        # [B, cells, T]
        sup = dev.reshape(B, -1).max(axis=1)
        q95 = np.quantile(dev.reshape(B, -1), 0.95, axis=1)
        # per-cell standard deviation at checkpoints where the point score lies in [0.5, 0.95]
        mid = (arr[0] >= 0.5) & (arr[0] <= 0.95)
        sd_cell = arr[1:].std(axis=0)                               # [cells, T]
        res['bootstrap_eps'] = dict(B=B, sup_median=float(np.median(sup)), sup_q95=float(np.quantile(sup, 0.95)),
                                    q95_median=float(np.median(q95)), sd_mid_scores_median=float(np.median(sd_cell[mid])) if mid.any() else None,
                                    sd_mid_scores_q95=float(np.quantile(sd_cell[mid], 0.95)) if mid.any() else None,
                                    n_items_per_task={t: len(k) for t, k in list(sm.items.items())[:6]})
    # (3) robust counts along the union zigzag (full complex and K_B), several theta and delta
    rc = {}
    for label, cx, remap in (('full', cx_full, lambda m: m), ('K_B', sub.cx, sub.remap)):
        cache = {}

        def mbt(th, cx=cx, remap=remap):
            key = round(th, 4)
            if key not in cache:
                cache[key] = [remap(m) for m in mk.masks(arr[0], key)]
            return cache[key]
        rc[label] = {}
        for th in (0.6, 0.7, 0.8, 0.9):
            rc[label][str(th)] = {}
            for d in DELTAS:
                if th - d < 0 or th + d > 1:
                    continue
                rc[label][str(th)][str(d)] = {f'H{k}': robust_counts(cx, mbt, th, d, k) for k in (0, 1)}
    res['robust_counts'] = rc
    return res


def main():
    spec = json.loads((V5 / 'complex_worlds' / 'data' / 'RCN12_FORGET3' / 'world635' / 'COMPLEX.json').read_text(encoding='utf8'))
    out = []
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and any(d.glob('eval_*.jsonl*')):
            r = analyse_arm(d, spec, full_boot=True)
            out.append(r)
            print(r['run'], 'eps:', r['bootstrap_eps'])
            for th in ('0.8',):
                for dl in ('0.0', '0.05', '0.1'):
                    c = r['robust_counts']['K_B'][th][dl]['H1']
                    ev = [(r['steps'][i + 1], b, dd, bo) for i, (b, dd, bo) in enumerate(c) if b or dd or bo]
                    print(f'   K_B H1 theta={th} delta={dl}: (to_step, births, deaths, bridge_only) =', ev)
    (Path(__file__).with_suffix('.json')).write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
