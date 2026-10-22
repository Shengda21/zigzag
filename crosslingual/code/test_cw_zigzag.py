"""Tests for cw_zigzag.py.

Cross-validates the general `CWComplex` implementation against the original
hard-coded eleven-cell algorithms in
``reproduction/analysis_dependencies/full_topology.py`` (imported via
``sys.path.insert``, not modified), then exercises the product-with-interval
construction, the event classifiers, the score-threshold closure semantics,
and the permutation null model.

Run: ``python test_cw_zigzag.py``. Prints a JSON summary and exits 0 iff
every check passed (any failed assertion raises and aborts with a non-zero
exit code).
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'reproduction', 'analysis_dependencies')))

import cw_zigzag as cz  # noqa: E402
import full_topology as ft  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------
# 1. Cross-validation of CWComplex(eleven_cell_complex) against full_topology.py
# --------------------------------------------------------------------------

def test_cross_validation() -> dict:
    cx = cz.eleven_cell_complex()
    check(cx.n == 11, 'eleven_cell_complex must have 11 cells')
    check(tuple(cx.dims) == tuple(ft.DIMS), 'dims mismatch vs full_topology.DIMS')
    check(tuple(cx.bound) == tuple(ft.BOUND), 'bound mismatch vs full_topology.BOUND')
    check(tuple(cx.closure) == tuple(ft.CLOSURE), 'closure mismatch vs full_topology.CLOSURE')

    # is_closed agreement over all 2**11 masks.
    n_masks = 1 << 11
    for mask in range(n_masks):
        check(cx.is_closed(mask) == ft.is_closed(mask), f'is_closed mismatch at mask={mask}')

    closed = [m for m in range(n_masks) if ft.is_closed(m)]
    check(len(closed) == 83, f'expected 83 closed subcomplexes, got {len(closed)}')

    # betti() vs full_topology.homology dimensions (per-k), with a Euler
    # characteristic cross-check independent of full_topology as well.
    for m in closed:
        b_gen = cx.betti(m)
        b_gen_padded = b_gen + [0] * (3 - len(b_gen))
        b_ref = [len(ft.homology(m, k)[0]) for k in (0, 1, 2)]
        check(b_gen_padded == b_ref, f'betti mismatch at mask={m}: {b_gen_padded} vs {b_ref}')
        euler = sum((-1) ** cx.dims[i] for i in range(11) if m >> i & 1)
        check(euler == b_gen_padded[0] - b_gen_padded[1] + b_gen_padded[2],
              f'Euler characteristic mismatch at mask={m}')

    # 300 random closed-subcomplex sequences (length 2-5): barcode and
    # endpoints must agree exactly with full_topology, for k in {0,1} and
    # bridge in {union, intersection}.
    rng = random.Random(20260922)
    n_sequences = 0
    n_barcode_checks = 0
    n_endpoint_checks = 0
    for _ in range(300):
        length = rng.randrange(2, 6)
        masks = tuple(rng.choice(closed) for _ in range(length))
        n_sequences += 1
        for k, bridge in itertools.product((0, 1), ('union', 'intersection')):
            bc_gen = sorted(cx.barcode(masks, k, bridge))
            bc_ref = sorted(ft.barcode(masks, k, bridge))
            check(bc_gen == bc_ref, f'barcode mismatch masks={masks} k={k} bridge={bridge}')
            n_barcode_checks += 1

            ep_gen = cx.endpoints(masks, k, bridge)
            ep_ref = ft.endpoints(masks, k, bridge)
            check(np.array_equal(ep_gen, ep_ref),
                  f'endpoints mismatch masks={masks} k={k} bridge={bridge}: {ep_gen} vs {ep_ref}')
            n_endpoint_checks += 1

            for cens in (False, True):
                ep_gen_c = cx.endpoints(masks, k, bridge, exclude_boundary=cens)
                ep_ref_c = ft.endpoints(masks, k, bridge, exclude_boundary=cens)
                check(np.array_equal(ep_gen_c, ep_ref_c),
                      f'endpoints(exclude_boundary={cens}) mismatch masks={masks} k={k} bridge={bridge}')

    return {
        'closed_subcomplexes': len(closed),
        'is_closed_checks': n_masks,
        'betti_checks': len(closed),
        'random_sequences': n_sequences,
        'barcode_checks': n_barcode_checks,
        'endpoint_checks': n_endpoint_checks,
    }


# --------------------------------------------------------------------------
# 2. full_topology.tests() toy assertions, reproduced on the general class.
# --------------------------------------------------------------------------

def test_toy_assertions_from_full_topology() -> dict:
    cx = cz.eleven_cell_complex()
    check(all(cx._boundary(b) == 0 for b in cx.bound), 'boundary-squared-zero failed')

    A = cx.closure[9] ^ (1 << 9)   # fA's boundary triangle (edges+vertices, no fA itself)
    B = cx.closure[10] ^ (1 << 10)  # fB's boundary triangle
    check(cx.barcode((A, A)) == ((0, 2),), 'barcode((A,A)) mismatch')
    check(cx.barcode((A, B)) == ((0, 1), (1, 2)), 'barcode((A,B)) mismatch')

    edge1 = A ^ (1 << 4)
    edge2 = A ^ (1 << 5)
    check(cx.barcode((edge1, edge2)) == ((1, 1),), 'barcode((edge1,edge2)) mismatch')
    check(cx.barcode((A, A | (1 << 9))) == ((0, 0),), 'barcode((A, A|fA)) mismatch (clean fill)')
    check(cx.barcode((A, A ^ (1 << 4))) == ((0, 1),), 'barcode((A, A^eFD)) mismatch (edge removal)')

    outer = 15 | sum(1 << i for i in (4, 5, 7, 8))
    check(len(cx.homology(outer, 1)[0]) == 1, 'outer-ring H1 should have dimension 1')
    check([len(cx.homology(2047, k)[0]) for k in (0, 1, 2)] == [1, 0, 0],
          'full complex betti should be [1,0,0]')

    # Independent known-interval quivers, exercising the module-level decompose().
    rng = random.Random(1729)
    n_quivers = 100
    for _ in range(n_quivers):
        n = rng.randrange(2, 10)
        truth = []
        for _j in range(rng.randrange(1, 12)):
            a = rng.randrange(n)
            b = rng.randrange(a, n)
            truth.append((a, b))
        active = [[j for j, (a, b) in enumerate(truth) if a <= i <= b] for i in range(n)]
        dims = list(map(len, active))
        arrows = []
        for i in range(n - 1):
            s, t = (i, i + 1) if rng.randrange(2) else (i + 1, i)
            arrows.append((s, t, tuple((1 << active[t].index(j)) if j in active[t] else 0 for j in active[s])))
        check(sorted(cz.decompose(dims, arrows)) == sorted(truth), 'decompose() failed a known-interval quiver')

    return {'boundary_squared_zero': True, 'outer_ring_h1': 1, 'full_betti': [1, 0, 0],
            'known_interval_quivers': n_quivers}


# --------------------------------------------------------------------------
# 3. Product-with-interval complex.
# --------------------------------------------------------------------------

def test_product_complex() -> dict:
    cx = cz.crosslingual_complex()
    check(cx.n == 33, f'crosslingual_complex should have 33 cells, got {cx.n}')

    def build(names):
        mask = 0
        for nm in names:
            mask |= 1 << cx.index(nm)
        return mask

    en_names = [nm for nm in cx.names if nm.endswith('@en')]
    zh_names = [nm for nm in cx.names if nm.endswith('@zh')]
    vX = ['vF', 'vD', 'vP', 'vR']
    eXY = ['eFD', 'eDP', 'eFP', 'eFR', 'eRP']

    full_mask = (1 << cx.n) - 1
    check(cx.is_closed(full_mask), 'full 33-cell mask should be closed')
    check(cx.betti(full_mask) == [1, 0, 0, 0], f'full betti mismatch: {cx.betti(full_mask)}')

    en_mask = build(en_names)
    check(cx.is_closed(en_mask), 'en-only mask should be closed')
    check(cx.betti(en_mask) == [1, 0, 0], f'en-only betti mismatch: {cx.betti(en_mask)}')

    # en full + 4 zh vertices + 4 bridge edges vX@I -> beta0=1, beta1=0.
    step1 = build(en_names + [f'{v}@zh' for v in vX] + [f'{v}@I' for v in vX])
    check(cx.is_closed(step1), 'step1 mask should be closed')
    check(cx.betti(step1) == [1, 0, 0], f'step1 betti mismatch: {cx.betti(step1)}')

    # + eFD@zh (no eFD@I) -> beta1=1 (a fresh cycle through the zh side).
    step2 = step1 | build(['eFD@zh'])
    check(cx.is_closed(step2), 'step2 mask should be closed')
    check(cx.betti(step2) == [1, 1, 0], f'step2 betti mismatch: {cx.betti(step2)}')

    # + eFD@I -> beta1=0, and the transition must classify as clean_fill.
    step3 = step2 | build(['eFD@I'])
    check(cx.is_closed(step3), 'step3 mask should be closed')
    check(cx.betti(step3) == [1, 0, 0], f'step3 betti mismatch: {cx.betti(step3)}')
    ev = cz.classify_h1_change(cx, step2, step3)
    check(ev['category'] == 'clean_fill', f'step2->step3 should classify as clean_fill, got {ev}')
    check(ev['h1_delta'] == -1 and ev['added'] == ['eFD@I'] and ev['removed'] == [],
          f'clean_fill event details unexpected: {ev}')

    # en full + zh full (incl. fA@zh, fB@zh), no @I cells at all -> two
    # disjoint copies, beta0=2.
    both_mask = build(en_names + zh_names)
    check(cx.is_closed(both_mask), 'both-sides mask should be closed')
    check(cx.betti(both_mask) == [2, 0, 0], f'disjoint-copies betti mismatch: {cx.betti(both_mask)}')

    # + 4 bridge vertices vX@I -> beta0=1. beta1 is cross-checked against an
    # independent Euler-characteristic computation (NOTE: this comes out to
    # 3, not the "5" mentioned in the task's illustrative prose; 8 vertices,
    # 14 edges [5+5 base edges + 4 vX@I bridges], 4 faces gives Euler = -2,
    # which forces beta1 = 1 - Euler = 3 given beta0=1, beta2=0 -- so 3 is
    # the value consistent with the Euler-Poincare identity that is itself
    # independently verified for every closed subcomplex in
    # test_cross_validation above).
    with_v_bridges = both_mask | build([f'{v}@I' for v in vX])
    check(cx.is_closed(with_v_bridges), 'vertex-bridge mask should be closed')
    betti_vb = cx.betti(with_v_bridges)
    dims_by_k = [sum(1 for i in range(cx.n) if with_v_bridges >> i & 1 and cx.dims[i] == kk) for kk in (0, 1, 2)]
    euler_vb = dims_by_k[0] - dims_by_k[1] + dims_by_k[2]
    check(betti_vb[0] - betti_vb[1] + (betti_vb[2] if len(betti_vb) > 2 else 0) == euler_vb,
          f'Euler-Poincare check failed for vertex-bridge state: {betti_vb} vs cell counts {dims_by_k}')
    check(betti_vb[0] == 1, f'vertex-bridge beta0 should be 1, got {betti_vb}')
    check(betti_vb[1] == 3, f'vertex-bridge beta1 should be 3 (Euler-checked), got {betti_vb}')

    # + 5 square edges eXY@I (fA@I, fB@I still missing) -> beta1=0, and
    # beta2=2 (two hollow triangular prisms, one per base face).
    with_e_bridges = with_v_bridges | build([f'{e}@I' for e in eXY])
    check(cx.is_closed(with_e_bridges), 'edge-bridge mask should be closed')
    betti_eb = cx.betti(with_e_bridges)
    check(betti_eb == [1, 0, 2], f'edge-bridge betti mismatch: {betti_eb}')

    return {
        'n_cells': cx.n,
        'full_betti': [1, 0, 0, 0],
        'en_only_betti': [1, 0, 0],
        'step1_betti': [1, 0, 0],
        'step2_betti': [1, 1, 0],
        'step3_betti': [1, 0, 0],
        'step2_to_step3_event': ev,
        'disjoint_copies_betti': [2, 0, 0],
        'vertex_bridges_betti': betti_vb,
        'edge_bridges_betti': betti_eb,
    }


# --------------------------------------------------------------------------
# 4. mask_from_scores closure semantics.
# --------------------------------------------------------------------------

def test_mask_from_scores_closure() -> dict:
    cx = cz.eleven_cell_complex()
    scores = {nm: 0.9 for nm in cx.names}
    scores['eDP'] = 0.1  # one boundary edge of fA is low
    mask = cx.mask_from_scores(scores, 0.5)
    cells = cx.cells_in(mask)
    check('fA' not in cells, 'fA should not enter: its boundary edge eDP is below threshold')
    check('eDP' not in cells, 'eDP itself should not enter (score below threshold)')
    check('fB' in cells, 'fB should enter: its whole closure is above threshold')
    check(cx.is_closed(mask), 'mask_from_scores must always return a closed subcomplex')

    # Sanity-check against local_topology.mask_from_scores on the same complex.
    sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..', '..', 'reproduction', 'analysis_dependencies')))
    import local_topology as lt  # noqa: E402
    score_vec = [scores[nm] for nm in cx.names]
    check(cx.names == lt.CELLS, 'cell order must match local_topology.CELLS for this comparison')
    check(mask == lt.mask_from_scores(score_vec, 0.5), 'mask_from_scores disagrees with local_topology reference')

    return {'cells_in_mask': cells, 'fA_excluded': True, 'matches_local_topology': True}


# --------------------------------------------------------------------------
# 5. Permutation null model: synchronized vs staggered transitions.
# --------------------------------------------------------------------------

def test_permutation_null() -> dict:
    n = 11
    cells = [(f'x{i}', 0, []) for i in range(n)]
    cx = cz.CWComplex(cells)
    names = cx.names
    T = n + 1  # 12 checkpoints -> 11 transitions, one per cell in the staggered case

    # Scenario A: every cell jumps 0.2 -> 0.8 simultaneously at transition 5.
    sync = {nm: [0.2] * 6 + [0.8] * 6 for nm in names}
    # Scenario B: cell i alone jumps 0.2 -> 0.8 at transition i (i = 0..10).
    stagger = {nm: [0.2 if t <= i else 0.8 for t in range(T)] for i, nm in enumerate(names)}

    n_perm = 200
    seed = 12345
    res_sync = cz.permutation_null(cx, sync, thresholds=[0.5], n_perm=n_perm, seed=seed, k=0, bridge='union')
    res_stag = cz.permutation_null(cx, stagger, thresholds=[0.5], n_perm=n_perm, seed=seed, k=0, bridge='union')

    p_sync_at_5 = res_sync['p_values'][5]
    p_stag_min = min(res_stag['p_values'].values())
    print(f'[permutation_null] synchronized jump, p at t=5: {p_sync_at_5:.6f}')
    print(f'[permutation_null] staggered jumps, min p over all t: {p_stag_min:.6f} '
          f'(argmin t={min(res_stag["p_values"], key=res_stag["p_values"].get)})')
    check(p_sync_at_5 < p_stag_min,
          f'synchronized p-value {p_sync_at_5} should be well below the staggered minimum {p_stag_min}')

    # block grouping sanity check: with all 11 cells forced into one block,
    # each permutation replicate either leaves the whole synchronized jump
    # exactly at t=5 (coin miss, ~50% per boundary) or moves it as one
    # intact unit to a neighboring transition. That first outcome alone
    # reproduces the full observed statistic, so the null distribution at
    # t=5 now puts much more mass at-or-above the observed value than the
    # unblocked (independent-per-cell) null did -- the p-value should rise
    # noticeably compared to `p_sync_at_5` (it is no longer testing whether
    # the cells moved *together*, only whether they moved *at all*).
    res_sync_block = cz.permutation_null(cx, sync, thresholds=[0.5], n_perm=n_perm, seed=seed, k=0,
                                          bridge='union', block=[list(names)])
    p_sync_block = res_sync_block['p_values'][5]
    print(f'[permutation_null] synchronized jump with block=[all cells], p at t=5: {p_sync_block:.6f}')

    return {
        'n_perm': n_perm, 'seed': seed,
        'p_sync_at_t5': p_sync_at_5,
        'p_stagger_min': p_stag_min,
        'p_sync_at_t5_blocked': p_sync_block,
        'bonferroni_alpha': res_sync['bonferroni_alpha'],
    }


def main() -> dict:
    results = {}
    results['cross_validation'] = test_cross_validation()
    print('[1/5] cross-validation against full_topology.py: OK', results['cross_validation'])
    results['toy_assertions'] = test_toy_assertions_from_full_topology()
    print('[2/5] full_topology.tests() toy assertions reproduced: OK')
    results['product_complex'] = test_product_complex()
    print('[3/5] product-with-interval (33-cell) complex: OK')
    results['mask_from_scores'] = test_mask_from_scores_closure()
    print('[4/5] mask_from_scores closure semantics: OK')
    results['permutation_null'] = test_permutation_null()
    print('[5/5] permutation_null synchronized-vs-staggered: OK')
    return results


if __name__ == '__main__':
    out = main()
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print('ALL TESTS PASSED')
