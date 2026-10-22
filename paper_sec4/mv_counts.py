"""Numerical check of the bridge-comparison theorem on the FORGET3 confirmation arms.

For every adjacent checkpoint pair (A = K_i(theta), B = K_{i+1}(theta)) and every arm:
  * u_k  = number of union-bridge-only intervals in degree k   (zigzag with union bridges)
  * n_k  = number of intersection-bridge-only intervals in degree k
  * d_k  = rank of the Mayer-Vietoris connecting map  H_k(A u B) -> H_{k-1}(A n B)
           = dim ker( H_{k-1}(A n B) -> H_{k-1}(A) (+) H_{k-1}(B) )
  Theorem: u_k = n_{k-1} = d_k.  Also: the checkpoint-restricted barcodes of the union and
  intersection zigzags coincide (bridge-only intervals removed, node indices mapped to checkpoints).
Historical union: spurious classes of H_1(K~_t) = beta_1(K~_t) - rank( (+)_{s<=t} H_1(K_s) -> H_1(K~_t) ),
  and the step-wise increments equal rank of the connecting map of the pair (K~_{t-1}, K_t).

  python mv_counts.py  ->  mv_counts.json  (+ a short console report)
"""
from __future__ import annotations
import collections
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
THETAS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def stacked_rank(cx, src, tgts, k):
    """rank of H_k(src) -> (+)_j H_k(tgt_j), all inclusions; returns (dim H_k(src), rank)."""
    reps, _ = cx.homology(src, k)
    cols = []
    for r_idx in range(len(reps)):
        v, shift = 0, 0
        for t in tgts:
            col = cx.induced(src, t, k)[r_idx]
            v |= int(col) << shift
            shift += len(cx.homology(t, k)[0])
        cols.append(v)
    return len(reps), CW.rank(cols)


def connecting_rank(cx, A, B, k):
    """rank of MV connecting map H_k(A u B) -> H_{k-1}(A n B) = dim ker(H_{k-1}(A n B) -> H_{k-1}(A) (+) H_{k-1}(B))."""
    if k == 0:
        return 0
    dim, rk = stacked_rank(cx, A & B, [A, B], k - 1)
    return dim - rk


def bridge_only(bars, node):
    return sum(1 for s, e in bars if s == e == node)


def restricted(bars):
    out = []
    for s, e in bars:
        a, b = (s + 1) // 2, e // 2
        if a <= b:
            out.append((a, b))
    return sorted(out)


def spurious_history(cx, masks, k=1):
    """per t: beta_k of the historical union, the number of classes not supported by any K_s (s<=t),
    and the connecting-map rank of the pair (K~_{t-1}, K_t)."""
    out = []
    acc = 0
    for t, m in enumerate(masks):
        prev = acc
        acc |= m
        beta = len(cx.homology(acc, k)[0])
        # rank of (+)_{s<=t} H_k(K_s) -> H_k(K~_t)
        cols = []
        for s in range(t + 1):
            cols.extend(int(c) for c in cx.induced(masks[s], acc, k))
        supported = CW.rank(cols)
        inc = connecting_rank(cx, prev, m, k) if t else 0
        out.append(dict(t=t, beta=beta, spurious=beta - supported, connecting_rank_prev_new=inc))
    return out


def analyse_arm(run_dir, spec):
    evals = ZA.load_run(run_dir)
    sm = R.ScoreModel(spec, evals)
    cx_full = CW.CWComplex([(n, d, list(b)) for n, d, b in spec['cells']])
    S = sm.scores(sm.weights(0, 0))
    arr = np.stack([S[n][0] for n in cx_full.names], axis=0)     # [cells, T] point estimate (replicate 0)
    steps = sm.steps
    res = dict(run=run_dir.name, steps=steps, thetas={})
    sub = R.Sub(spec['cells'], R.K_B_CELLS).bind(cx_full)
    for th in THETAS:
        row = {}
        for label, cx, remap in (('full', cx_full, lambda m: m), ('K_B', sub.cx, sub.remap)):
            mk = R.Masker(cx_full)
            masks = [remap(m) for m in mk.masks(arr, th)]
            T = len(masks)
            per_bridge = []
            ok_theorem = True
            for i in range(T - 1):
                A, B = masks[i], masks[i + 1]
                rec = dict(i=i, step_a=steps[i], step_b=steps[i + 1])
                for k in (0, 1, 2):
                    u = bridge_only(R.fast_barcode(cx, masks, k, 'union'), 2 * i + 1)
                    n_prev = bridge_only(R.fast_barcode(cx, masks, k - 1, 'intersection'), 2 * i + 1) if k >= 1 else None
                    d = connecting_rank(cx, A, B, k)
                    rec[f'u{k}'] = u
                    rec[f'd{k}'] = d
                    if k >= 1:
                        rec[f'n{k-1}'] = n_prev
                        ok_theorem &= (u == d == n_prev)
                    else:
                        ok_theorem &= (u == 0)
                per_bridge.append(rec)
            restricted_equal = all(restricted(R.fast_barcode(cx, masks, k, 'union')) == restricted(R.fast_barcode(cx, masks, k, 'intersection')) for k in (0, 1, 2))
            nontrivial = [b for b in per_bridge if any(b[f'u{k}'] for k in (1, 2))]
            hist = spurious_history(cx, masks, 1)
            hist_ok = all(h['spurious'] == sum(x['connecting_rank_prev_new'] for x in hist[:h['t'] + 1]) for h in hist) or None
            row[label] = dict(theorem_holds_all_bridges=ok_theorem, restricted_barcodes_equal=restricted_equal,
                              bridges_with_bridge_only_classes=nontrivial,
                              union_H1_bars=[list(b) for b in R.fast_barcode(cx, masks, 1, 'union')],
                              intersection_H1_bars=[list(b) for b in R.fast_barcode(cx, masks, 1, 'intersection')],
                              hist_union_final=hist[-1], hist_union_spurious_series=[h['spurious'] for h in hist],
                              hist_increments_match_cumulative=hist_ok)
        res['thetas'][str(th)] = row
    return res


def main():
    spec = json.loads((V5 / 'complex_worlds' / 'data' / 'RCN12_FORGET3' / 'world635' / 'COMPLEX.json').read_text(encoding='utf8'))
    out = []
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and any(d.glob('eval_*.jsonl*')):
            r = analyse_arm(d, spec)
            out.append(r)
            x = r['thetas']['0.8']
            print(r['run'], '| theorem holds (full/K_B):', x['full']['theorem_holds_all_bridges'], x['K_B']['theorem_holds_all_bridges'],
                  '| restricted barcodes equal:', x['full']['restricted_barcodes_equal'], x['K_B']['restricted_barcodes_equal'],
                  '| K_B union H1 bars:', x['K_B']['union_H1_bars'], '| K_B intersection H1 bars:', x['K_B']['intersection_H1_bars'],
                  '| K_B bridge-only at:', [(b['step_a'], b['step_b'], b['u1'], b['d1']) for b in x['K_B']['bridges_with_bridge_only_classes']],
                  '| hist-union H1 final (K_B):', x['K_B']['hist_union_final'], '| increments match:', x['K_B']['hist_increments_match_cumulative'])
    all_ok = all(r['thetas'][t][l]['theorem_holds_all_bridges'] and r['thetas'][t][l]['restricted_barcodes_equal'] for r in out for t in r['thetas'] for l in ('full', 'K_B'))
    print('ALL ARMS x THETAS x {full,K_B}: theorem and restricted-barcode equality hold:', all_ok)
    (Path(__file__).with_suffix('.json')).write_text(json.dumps(dict(all_ok=all_ok, arms=out), indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
