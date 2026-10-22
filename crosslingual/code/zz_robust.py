"""Revised statistics of PREREGISTRATION_FORGET3_CONFIRM2_20260924.md §2.

S2  item-cluster bootstrap: every cell score is a weighted accuracy over item
    clusters (task, item); one multinomial resample per task and replicate is
    applied to all checkpoints of a run; masks, barcodes, event classes and the
    pre-registered criteria are recomputed per replicate.  Replicate 0 is the
    point estimate (all weights 1).  Support s_c = share of replicates 1..B in
    which criterion c holds; "robust" = point estimate holds and s_c >= 0.95.
S3  temporal-organisation test: E_tot = total zigzag endpoints (H0 + H1, union
    bridges, averaged over the threshold grid) of the point-estimate trajectory
    against uniformly permuted checkpoint orders (same permutation for all
    cells); one-sided lower tail.

Line-specific criteria: `forget3` (RCN12_FORGET3 and the time-scaled
RCN12_FORGET3_X2, criteria 1-7 and report R3 of §3/§4), `grok` and `bio`
(post-hoc descriptions, see the functions).  Nothing here is tuned on the
confirmation worlds; parameters are the pre-registered ones.

  python zz_robust.py --line forget3 --runs <dir with world*_S*> --design <COMPLEX.json> --out <dir> [--boot 1000 --perm 1000]
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import random
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw_zigzag as CW          # noqa: E402
import zz_analyze as ZA         # noqa: E402

THETA = tuple(ZA.THETA)
ROBUST = 0.95
MIN_THETAS = 5


# ------------------------------------------------------------------ scores
def item_key(r):
    k = r.get('item', r.get('root'))
    return json.dumps(k, sort_keys=True) if isinstance(k, (list, dict)) else str(k)


class ScoreModel:
    """Cell scores as weighted accuracies over item clusters (same definitions as
    zz_analyze.design_from_spec: vertex read-out, edge conversion, face = min over
    override variants of the joint task, or the route score for routes faces)."""

    def __init__(self, spec, evals, blanket=False):
        self.steps = sorted(evals)
        T = len(self.steps)
        vocab = spec.get('vocabulary', 'rcn')
        main_t = set(spec['main_templates'])
        routes = spec.get('face_routes', {}) if spec.get('face_readout') == 'routes' else {}
        self.cells = [(n, d, list(b)) for n, d, b in spec['cells']]
        by = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))   # task -> t -> item -> rows
        for ti, s in enumerate(self.steps):
            for r in evals[s]:
                if r.get('vocabulary') == vocab and r.get('template') in main_t:
                    by[r['task']][ti][item_key(r)].append(r)
        self.items = {}           # cluster task -> list of item keys (bootstrap unit)
        self.comp = {}            # cell -> list of (task, idx array, correct [T, n], total [n])

        def component(task, pred=lambda r: True):
            keys = sorted({k for ti in by[task] for k in by[task][ti]})
            if task not in self.items:
                self.items[task] = keys
            pos = {k: i for i, k in enumerate(self.items[task])}
            n = len(self.items[task])
            corr = np.zeros((T, n))
            tot = None
            for ti in range(T):
                cnt = np.zeros(n)
                for k, rs in by[task].get(ti, {}).items():
                    sel = [r for r in rs if pred(r)]
                    corr[ti, pos[k]] = sum(r['correct'] for r in sel)
                    cnt[pos[k]] = len(sel)
                if tot is None:
                    tot = cnt
                assert np.array_equal(cnt, tot), f'probe set changes across checkpoints for {task}'
            return (task, corr, tot)

        for n, d, _ in self.cells:
            if d == 0:
                self.comp[n] = [] if blanket else [component('v' + n)]
            elif d == 1:
                self.comp[n] = [component(n)]
            elif n in routes:
                tasks = routes[n]
                keys = sorted(set.intersection(*[{k for ti in by[t] for k in by[t][ti]} for t in tasks]))
                ptask = 'route:' + n
                self.items[ptask] = keys
                corr = np.zeros((T, len(keys)))
                for ti in range(T):
                    for i, k in enumerate(keys):
                        corr[ti, i] = float(all(all(r['correct'] for r in by[t][ti].get(k, [])) for t in tasks))
                self.comp[n] = [(ptask, corr, np.ones(len(keys)))]
            else:
                self.comp[n] = [component(n, lambda r, o=o: r.get('override') == o) for o in (False, True)]

    def weights(self, B, seed):
        """Replicate 0 = ones; replicates 1..B = multinomial item-cluster resamples per task."""
        rng = np.random.default_rng(seed)
        W = {}
        for task, keys in self.items.items():
            n = len(keys)
            w = np.ones((B + 1, n))
            if n:
                w[1:] = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
            W[task] = w
        return W

    def scores(self, W):
        """cell -> array [B+1, T] of scores."""
        out = {}
        for n, d, _ in self.cells:
            comps = self.comp[n]
            if not comps:
                out[n] = np.ones((next(iter(W.values())).shape[0], len(self.steps)))
                continue
            vals = []
            for task, corr, tot in comps:
                w = W[task]
                den = w @ tot
                num = w @ corr.T
                vals.append(np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-12), 0.0))
            out[n] = np.minimum.reduce(vals) if len(vals) > 1 else vals[0]
        return out


class Masker:
    """Vectorised closure-min thresholding (identical to CWComplex.mask_from_scores)."""

    def __init__(self, cx):
        self.cx = cx
        n = len(cx.names)
        self.cl = np.zeros((n, n), dtype=bool)
        for i, c in enumerate(cx.closure):
            for j in range(n):
                self.cl[i, j] = bool(c >> j & 1)
        self.bits = [1 << i for i in range(n)]

    def masks(self, S, theta):
        """S: [n_cells, T] scores -> list of T int masks."""
        raw = S >= theta - 1e-12
        present = ~((~raw)[None, :, :] & self.cl[:, :, None]).any(axis=1)
        out = []
        for t in range(S.shape[1]):
            idx = np.nonzero(present[:, t])[0]
            out.append(sum(self.bits[i] for i in idx))
        return out


# ------------------------------------------------------------------ helpers
def closed_mask(cx, names):
    m = 0
    for c in names:
        m |= cx.closure[cx.index(c)]
    return m


class Sub:
    """Closed subcomplex as its own CWComplex; remap(mask of K*) -> mask of the subcomplex.
    Homology of K_t(theta) cap K_S computed here equals the homology of the restricted mask in K*."""

    def __init__(self, cells, keep):
        keep = set(keep)
        self.cells = [(n, d, list(b)) for n, d, b in cells if n in keep]
        self.cx = CW.CWComplex(self.cells)
        self.src = None
        self.names = [c[0] for c in self.cells]

    def bind(self, cx):
        self.src = [cx.index(n) for n in self.names]
        return self

    def remap(self, m):
        out = 0
        for j, i in enumerate(self.src):
            if m >> i & 1:
                out |= 1 << j
        return out


def fast_barcode(cx, masks, k=1, bridge='union'):
    """cx.barcode on the run-length-compressed sequence, mapped back to the original node indices.
    Inside a run of identical masks every map (and every bridge) is the identity, so each interval
    simply extends over the whole run; this is exact, not an approximation (checked in tests)."""
    masks = [int(m) for m in masks]
    runs = []                                  # (mask, first checkpoint, last checkpoint)
    for i, m in enumerate(masks):
        if runs and runs[-1][0] == m:
            runs[-1][2] = i
        else:
            runs.append([m, i, i])
    comp = cx.barcode(tuple(r[0] for r in runs), k, bridge)

    def start(a):
        j = a // 2
        return 2 * runs[j][1] if a % 2 == 0 else 2 * runs[j][2] + 1

    def end(b):
        j = b // 2
        return 2 * runs[j][2] if b % 2 == 0 else 2 * runs[j][2] + 1
    return tuple((start(a), end(b)) for a, b in comp)


def make_classifier(cx):
    @lru_cache(maxsize=400000)
    def h1(m0, m1):
        d = CW.classify_h1_change(cx, m0, m1)
        return d['h1_delta'], d['category']

    @lru_cache(maxsize=400000)
    def h0(m0, m1):
        d = CW.classify_h0_change(cx, m0, m1)
        return d['h0_delta'], d.get('category')
    return h1, h0


def e_tot(cx, masks_by_theta, order):
    tot = 0.0
    for th, masks in masks_by_theta.items():
        seq = [masks[i] for i in order]
        for k in (0, 1):
            tot += float(cx.endpoints(seq, k=k, bridge='union').sum())
    return tot / len(masks_by_theta)


def temporal_test(cx, masks_by_theta, n_perm, seed):
    T = len(next(iter(masks_by_theta.values())))
    obs = e_tot(cx, masks_by_theta, list(range(T)))
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        o = list(range(T))
        rng.shuffle(o)
        null.append(e_tot(cx, masks_by_theta, o))
    null = np.array(null)
    return dict(E_obs=obs, null_median=float(np.median(null)) if n_perm else None, null_min=float(null.min()) if n_perm else None,
                p_lower=float((1 + (null <= obs).sum()) / (n_perm + 1)) if n_perm else None, n_perm=n_perm)


# ------------------------------------------------------------------ FORGET3 criteria (§3 / §4)
B_ORIG = ('B1B2', 'B2B3', 'B1B3', 'B3B4')
B_REP = tuple(e + 'r' for e in B_ORIG)
K_B_CELLS = ('B1', 'B2', 'B3', 'B4') + B_ORIG + B_REP + ('B2B4', 'fB123', 'fB234')
K_C_CELLS = ('C1', 'C2', 'C3', 'C4', 'C1C2', 'C2C3', 'C1C3', 'C3C4', 'fC123')


class Forget3:
    names = ('c1_teach', 'c2_death', 'c3_replacement', 'c4_recovery', 'c5_zigzag', 'c6_mechanism', 'c7_placebo')
    success = ('c1_teach', 'c2_death', 'c3_replacement', 'c4_recovery', 'c5_zigzag', 'c7_placebo')     # amendment 3: c6 is report item R5

    def __init__(self, cx, spec, steps):
        self.cx, self.steps = cx, list(steps)
        self.F = int(spec.get('time_scale', 1))
        cells = [(n, d, list(b)) for n, d, b in spec['cells']]
        kb = [cx.names[i] for i in range(len(cx.names)) if closed_mask(cx, [c for c in K_B_CELLS if c in cx.names]) >> i & 1]
        kc = [cx.names[i] for i in range(len(cx.names)) if closed_mask(cx, K_C_CELLS) >> i & 1]
        self.SB, self.SC = Sub(cells, kb).bind(cx), Sub(cells, kc).bind(cx)
        self.h1B, _ = make_classifier(self.SB.cx)
        self.h1C, _ = make_classifier(self.SC.cx)
        self.h1, self.h0 = make_classifier(cx)
        self.ctrl = tuple(spec.get('control_cells', ()))
        self.T = len(self.steps)

    def at(self, s):
        return self.steps.index(s * self.F)

    def has(self, m, c):
        return bool(m >> self.cx.index(c) & 1)

    def trans(self, masks, sub, lo, hi, want):
        """any transition i->i+1 with lo < step[i+1] <= hi whose H1 change in the subcomplex satisfies want(delta, cat)"""
        F = self.F
        S, h1 = (self.SB, self.h1B) if sub == 'B' else (self.SC, self.h1C)
        for i in range(self.T - 1):
            if lo * F < self.steps[i + 1] <= hi * F:
                if want(*h1(S.remap(masks[i]), S.remap(masks[i + 1]))):
                    return True
        return False

    def zigzag_ok(self, masks):
        F, st, T = self.F, self.steps, self.T
        seq = tuple(self.SB.remap(m) for m in masks)
        bars = fast_barcode(self.SB.cx, seq, 1, 'union')
        wins = [(256 * F, 288 * F), (416 * F, 480 * F), (608 * F, 768 * F)]

        def hits(bar, w):
            return any(w[0] <= st[j // 2] <= w[1] for j in range(bar[0], bar[1] + 1) if j % 2 == 0)
        cand = [[b for b in bars if hits(b, w)] for w in wins]
        def ck(bar):         # checkpoints covered by a bar (even nodes); bridge-only nodes are transitional (amendment 2)
            return [j // 2 for j in range(bar[0], bar[1] + 1) if j % 2 == 0]
        three = any(max(ck(b1)) < min(ck(b2)) and max(ck(b2)) < min(ck(b3)) for b1 in cand[0] for b2 in cand[1] for b3 in cand[2])
        mono, acc = [], 0
        for m in seq:
            acc |= m
            mono.append(acc)
        mb = fast_barcode(self.SB.cx, mono, 1, 'union')
        persists = any(st[b[0] // 2] <= 288 * F and b[1] == 2 * (T - 1) for b in mb)
        return three and persists, dict(bars=[list(b) for b in bars], mono_bars=[list(b) for b in mb])

    def evaluate(self, mb, detail=False):
        """mb: theta -> list of masks.  Returns {criterion: bool} (+ details)."""
        F = self.F
        m08, m06 = mb[0.8], mb[0.6]
        r, det = {}, {}
        r['c1_teach'] = all(any(self.has(m08[t], e) for t in range(self.T) if self.steps[t] <= 256 * F) for e in B_ORIG)
        gone = all(not self.has(mb[th][self.at(352)], e) for th in THETA for e in B_ORIG)
        n_loss = [th for th in THETA if self.trans(mb[th], 'B', 288, 352, lambda d, c: d < 0 and c == 'edge_loss')]
        r['c2_death'] = gone and len(n_loss) >= MIN_THETAS
        rep_in = all(any(self.has(m08[t], e) for t in range(self.T) if self.steps[t] <= 416 * F) for e in B_REP)
        n_inc = [th for th in THETA if self.trans(mb[th], 'B', 288, 416, lambda d, c: c == 'h1_increase')]
        r['c3_replacement'] = rep_in and len(n_inc) >= MIN_THETAS
        r['c4_recovery'] = (all(not self.has(m06[self.at(544)], e) for e in B_REP)
                            and all(self.has(m08[self.at(608)], e) for e in B_ORIG))
        zz = {th: self.zigzag_ok(mb[th]) for th in THETA}
        r['c5_zigzag'] = sum(v[0] for v in zz.values()) >= MIN_THETAS

        n_mech = [th for th in THETA if self.trans(mb[th], 'C', 128, 384, lambda d, c: d < 0 and c == 'clean_fill')]
        r['c6_mechanism'] = len(n_mech) >= MIN_THETAS and len(n_loss) >= MIN_THETAS     # amendment 1: mechanism distinction only
        if detail:      # report R4 (amendment 1): face reopening / refilling
            det['R4_reopen_thetas'] = [th for th in THETA if self.trans(mb[th], 'C', 352, 480, lambda d, c: c == 'h1_increase')]
            det['R4_refill_thetas'] = [th for th in THETA if self.trans(mb[th], 'C', 544, 672, lambda d, c: d < 0 and c == 'clean_fill')]
        r['c7_placebo'] = not any(self.has(m, c) for th in THETA for m in mb[th] for c in self.ctrl)
        if detail:
            det.update(thetas_edge_loss=n_loss, thetas_h1_increase=n_inc, thetas_mechanism=n_mech,
                       zigzag={str(th): dict(ok=v[0], **v[1]) for th, v in zz.items()},
                       first_in_08={e: next((self.steps[t] for t in range(self.T) if self.has(m08[t], e)), None) for e in B_ORIG + B_REP + ('fC123',)})
        return r, det

    def transfer_window(self, evals, mb):
        """Report R3: H0 merge through C4A1 vs the jump of the untrained chain C3->C4->A1->A2."""
        task = 'chain_C3_C4_A1_A2'
        accs = [ZA.acc([r for r in evals[s] if r['task'] == task]) for s in self.steps]
        w = 4        # 64 steps at the 16-step grid
        deltas = [accs[i + w] - accs[i] for i in range(self.T - w)]
        best = int(np.argmax(deltas))
        merges = {}
        for th in THETA:
            m = mb[th]
            for i in range(self.T - 1):
                d, cat = self.h0(m[i], m[i + 1])
                if cat == 'merge' and self.has(m[i + 1], 'C4A1') and not self.has(m[i], 'C4A1'):
                    merges[str(th)] = self.steps[i + 1]
                    break
        m08 = merges.get('0.8')
        inside = m08 is not None and self.steps[best] < m08 <= self.steps[best + w]
        return dict(task=task, accuracy=accs, best_window=[self.steps[best], self.steps[best + w]], best_delta=deltas[best],
                    second_delta=float(sorted(deltas)[-2]) if len(deltas) > 1 else None, merge_step_by_theta=merges, merge_inside_best_window_08=inside)


# ------------------------------------------------------------------ public checkpoints (Pythia)
class PublicScoreModel:
    """public_ckpt cell scores (cell_scores.py): vertex/edge = accuracy over all rows of the task
    (every template and item); face = share of (item, template) pairs whose two composition paths
    are both correct.  Clusters: (task, item) and (face, item)."""

    def __init__(self, cells, evals):
        self.steps = sorted(evals)
        T = len(self.steps)
        self.cells = cells
        ve = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0])))
        fc = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
        for ti, s in enumerate(self.steps):
            for r in evals[s]:
                if r['kind'] in ('vertex', 'edge'):
                    x = ve[r['task']][ti][str(r['item'])]
                    x[0] += r['correct']
                    x[1] += 1
                elif r['kind'] == 'composition':
                    fc[r['face']][ti][(str(r['item']), r['template'])][r['path_index']] = r['correct']
        self.items, self.comp = {}, {}
        for n, d, _ in cells:
            if d < 2:
                src = ve[n]
                keys = sorted({k for ti in src for k in src[ti]})
                corr = np.array([[src[ti][k][0] if k in src[ti] else 0 for k in keys] for ti in range(T)], dtype=float)
                tot = np.array([src[0][k][1] if k in src[0] else 0 for k in keys], dtype=float)
            else:
                src = fc[n]
                keys = sorted({k[0] for ti in src for k in src[ti]})
                pos = {k: i for i, k in enumerate(keys)}
                corr = np.zeros((T, len(keys)))
                tot = np.zeros(len(keys))
                for ti in range(T):
                    cnt = np.zeros(len(keys))
                    for (it, tpl), paths in src[ti].items():
                        if 0 in paths and 1 in paths:
                            cnt[pos[it]] += 1
                            corr[ti, pos[it]] += int(paths[0] == 1 and paths[1] == 1)
                    if ti == 0:
                        tot = cnt
            task = ('face:' if d == 2 else '') + n
            self.items[task] = keys
            self.comp[n] = [(task, corr, tot)]

    weights = ScoreModel.weights
    scores = ScoreModel.scores


class PublicLine:
    """P1 robust exits, P2 zigzag vs monotonised finite bars, P4 robust H1-decrease / H0-merge events."""

    def __init__(self, cx, steps):
        self.cx, self.steps, self.T = cx, list(steps), len(steps)
        self.h1, self.h0 = make_classifier(cx)

    def events(self, mb):
        ex, h1d, h0m = set(), set(), set()
        n = len(self.cx.names)
        for th in THETA:
            m = mb[th]
            for t in range(self.T - 2):
                gone = m[t] & ~m[t + 1] & ~m[t + 2]
                for i in range(n):
                    if gone >> i & 1:
                        ex.add((self.cx.names[i], th, t))
            for t in range(self.T - 1):
                d, cat = self.h1(m[t], m[t + 1])
                if d < 0:
                    h1d.add((cat, th, t))
                d0, cat0 = self.h0(m[t], m[t + 1])
                if cat0 == 'merge':
                    h0m.add((th, t))
        return ex, h1d, h0m

    @staticmethod
    def near(ev, pool, key):
        """ev matched in pool if an event with the same key and |t' - t| <= 1 exists."""
        k, t = key(ev)
        return any(key(x)[0] == k and abs(key(x)[1] - t) <= 1 for x in pool)

    def bars(self, mb):
        out = {}
        for th in THETA:
            m = mb[th]
            mono, acc = [], 0
            for x in m:
                acc |= x
                mono.append(acc)
            last = 2 * (self.T - 1)
            row = {}
            for k in (0, 1):
                zz = fast_barcode(self.cx, m, k, 'union')
                mo = fast_barcode(self.cx, mono, k, 'union')
                row[f'H{k}'] = dict(zigzag_finite=sum(b[1] < last for b in zz), mono_finite=sum(b[1] < last for b in mo),
                                    zigzag_bars=len(zz), mono_bars=len(mo))
            out[str(th)] = row
        return out


# ------------------------------------------------------------------ grokking (post hoc description)
ARITH = ('P_S', 'P_D', 'P_M', 'P_Q', 'Pn_D', 'Ps_S', 'Pi_Q')
ADDITIVE = ('P_S', 'P_D', 'Pn_D', 'Ps_S')


class GrokLine:
    names = ('g1_all_born', 'g2_additive_same_checkpoint', 'g3_placebo_never')

    def __init__(self, cx, steps):
        self.cx, self.steps, self.T = cx, list(steps), len(steps)

    def birth(self, m, c):
        i = self.cx.index(c)
        return next((t for t in range(self.T - 1) if m[t] >> i & 1 and m[t + 1] >> i & 1), None)

    def evaluate(self, mb, detail=False):
        m = mb[0.8]
        b = {c: self.birth(m, c) for c in ARITH if c in self.cx.names}
        r = dict(g1_all_born=all(v is not None for v in b.values()),
                 g2_additive_same_checkpoint=len({b[c] for c in ADDITIVE}) == 1 and b[ADDITIVE[0]] is not None,
                 g3_placebo_never=not any(x >> self.cx.index('P_X') & 1 for th in THETA for x in mb[th]))
        det = dict(birth_step_08={c: (self.steps[v] if v is not None else None) for c, v in b.items()}) if detail else {}
        return r, det


# ------------------------------------------------------------------ driver
def analyse_run(args):
    run_dir, design, line, B, n_perm, seed, time_scale = args
    t0 = time.time()
    spec = json.loads(Path(design).read_text(encoding='utf8'))
    if time_scale:
        spec['time_scale'] = time_scale        # COMPLEX.json does not carry the spec's time_scale (X2 worlds)
    evals = ZA.load_run(run_dir)
    if line == 'public':
        cells = ([(v, 0, []) for v in spec['vertices']] + [(e['name'], 1, [e['src'], e['dst']]) for e in spec['edges']]
                 + [(f['name'], 2, list(f['cycle'])) for f in spec['faces']])
        sm = PublicScoreModel(cells, evals)
        cx = CW.CWComplex(cells)
    else:
        blanket = spec.get('vertex_present_mode') == 'blanket' or os.environ.get('ZZ_VERTICES_PRESENT') == '1' or line == 'grok'
        sm = ScoreModel(spec, evals, blanket=blanket)
        cx = CW.CWComplex([(n, d, list(b)) for n, d, b in spec['cells']])
    mk = Masker(cx)
    W = sm.weights(B, seed)
    S = sm.scores(W)
    names = list(cx.names)
    arr = np.stack([S[n] for n in names], axis=1)        # [B+1, n_cells, T]
    crit = Forget3(cx, spec, sm.steps) if line == 'forget3' else GrokLine(cx, sm.steps) if line == 'grok' else None
    point_masks = {th: mk.masks(arr[0], th) for th in THETA}
    res = dict(run=Path(run_dir).name, design=str(design), steps=sm.steps, B=B, seed=seed, time_scale=spec.get('time_scale', 1))
    res['scores_point'] = {n: [round(float(x), 4) for x in arr[0][i]] for i, n in enumerate(names)}
    if crit is not None:
        point, det = crit.evaluate(point_masks, detail=True)
        hits = collections.Counter()
        for b in range(1, B + 1):
            rb, _ = crit.evaluate({th: mk.masks(arr[b], th) for th in THETA})
            hits.update(k for k, v in rb.items() if v)
        support = {k: hits[k] / B if B else None for k in crit.names}
        robust = {k: bool(point[k] and B and support[k] >= ROBUST) for k in crit.names}
        success_keys = getattr(crit, 'success', crit.names)
        res.update(point=point, support=support, robust=robust, all_robust=all(robust[k] for k in success_keys), success_criteria=list(success_keys), detail=det)
        if line == 'forget3':
            res['transfer_R3'] = crit.transfer_window(evals, point_masks)
    if line == 'public':
        pl = PublicLine(cx, sm.steps)
        ex0, h10, h00 = pl.events(point_masks)
        cnt_ex, cnt_h1, cnt_h0 = collections.Counter(), collections.Counter(), collections.Counter()
        for b in range(1, B + 1):
            exb, h1b, h0b = pl.events({th: mk.masks(arr[b], th) for th in THETA})
            for e in ex0:
                cnt_ex[e] += pl.near(e, exb, lambda x: ((x[0], x[1]), x[2]))
            for e in h10:
                cnt_h1[e] += pl.near(e, h1b, lambda x: ((x[0], x[1]), x[2]))
            for e in h00:
                cnt_h0[e] += pl.near(e, h0b, lambda x: (x[0], x[1]))
        st = sm.steps
        rob_ex = sorted([dict(cell=e[0], theta=e[1], exit_after_step=st[e[2]], support=cnt_ex[e] / B) for e in ex0 if B and cnt_ex[e] / B >= ROBUST],
                        key=lambda d: (d['theta'], d['exit_after_step']))
        per_theta = {str(th): sum(1 for d in rob_ex if d['theta'] == th) for th in THETA}
        res.update(P1=dict(point_exits=len(ex0), robust_exits=rob_ex, robust_per_theta=per_theta,
                           H_P1=sum(v >= 1 for v in per_theta.values()) >= MIN_THETAS),
                   P2=pl.bars(point_masks),
                   P4=dict(h1_decrease=[dict(category=e[0], theta=e[1], to_step=st[e[2] + 1], support=cnt_h1[e] / B if B else None)
                                        for e in sorted(h10, key=lambda x: (x[1], x[2]))],
                           h0_merge=[dict(theta=e[0], to_step=st[e[1] + 1], support=cnt_h0[e] / B if B else None,
                                          added=cx.cells_in(point_masks[e[0]][e[1] + 1] & ~point_masks[e[0]][e[1]])) for e in sorted(h00)]))
    res['temporal_S3'] = temporal_test(cx, point_masks, n_perm, seed + 1)
    res['seconds'] = round(time.time() - t0, 1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--line', choices=('forget3', 'grok', 'public'), default='forget3')
    p.add_argument('--runs', required=True, help='directory holding run subdirectories (eval_*.jsonl[.gz])')
    p.add_argument('--design', required=True, help='COMPLEX.json (any world of the spec; cells are shared)')
    p.add_argument('--only', default='', help='comma-separated run names')
    p.add_argument('--out', required=True)
    p.add_argument('--boot', type=int, default=1000)
    p.add_argument('--perm', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--procs', type=int, default=4)
    p.add_argument('--time-scale', type=int, default=0, help='criteria windows multiplier (2 for RCN12_FORGET3_X2)')
    a = p.parse_args()
    runs = sorted(d for d in Path(a.runs).iterdir() if d.is_dir() and any(d.glob('eval_*.jsonl*')))
    if a.only:
        keep = set(a.only.split(','))
        runs = [d for d in runs if d.name in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), a.design, a.line, a.boot, a.perm, a.seed, a.time_scale) for d in runs]
    with Pool(min(a.procs, len(jobs))) as pool:
        results = pool.map(analyse_run, jobs)
    for r in results:
        (out / f"{r['run']}.robust.json").write_text(json.dumps(r, indent=1), encoding='utf8')
        s3 = r['temporal_S3']
        if 'point' in r:
            extra = {k: (int(r['point'][k]), round(r['support'][k], 3)) for k in r['point']}
        elif 'P1' in r:
            extra = dict(H_P1=r['P1']['H_P1'], robust_per_theta=r['P1']['robust_per_theta'])
        else:
            extra = ''
        print(r['run'], extra, 'S3 p=', s3['p_lower'], f"({r['seconds']}s)", flush=True)
    summ = dict(line=a.line, runs=[r['run'] for r in results], boot=a.boot, perm=a.perm,
                arms_all_robust=[r['run'] for r in results if r.get('all_robust')],
                arms_S3_ok=[r['run'] for r in results if (r['temporal_S3']['p_lower'] or 1) <= 0.01])
    (out / 'ROBUST_SUMMARY.json').write_text(json.dumps(summ, indent=1), encoding='utf8')
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
