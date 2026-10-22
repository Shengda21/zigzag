"""H0 barcodes of the Pythia-1.4B capability complex at theta = 0.6 (paper §4.5, Fig. 4 lower panel).

Zigzag (union bridges) vs union-monotonised ordinary persistence, with an item-cluster bootstrap
support for every finite interval: an interval is matched in a replicate if that replicate's barcode
of the same kind has a finite interval whose start and end nodes are each within 2 nodes (one
checkpoint) of it.  `in_mono` marks zigzag intervals that also occur (same nodes) in the monotonised
barcode; the zigzag-only finite intervals are the non-monotone ones (here: a component that splits
off when an edge briefly drops below theta and merges back when it returns).

  python pythia_h0_bars.py   ->  pythia_h0_bars.json
"""
import json
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5 / 'crosslingual' / 'code'))
import cw_zigzag as CW      # noqa: E402
import zz_analyze as ZA     # noqa: E402
import zz_robust as R       # noqa: E402

THETA, B, SEED = 0.6, 1000, 20260925
spec = json.loads((V5 / 'public_ckpt' / 'spec' / 'COMPLEX.json').read_text(encoding='utf8'))
cells = ([(v, 0, []) for v in spec['vertices']] + [(e['name'], 1, [e['src'], e['dst']]) for e in spec['edges']]
         + [(f['name'], 2, list(f['cycle'])) for f in spec['faces']])
cx = CW.CWComplex(cells)
import os
_RUN = Path(os.environ['ZZ_PYTHIA_RUN']) if os.environ.get('ZZ_PYTHIA_RUN') else V5 / 'cloud_return_round4' / 'public_ckpt' / 'runs' / 'pythia-1.4b'
_OUT = Path(os.environ['ZZ_PYTHIA_H0_OUT']) if os.environ.get('ZZ_PYTHIA_H0_OUT') else Path(__file__).with_suffix('.json')
evals = ZA.load_run(_RUN)   # env ZZ_PYTHIA_RUN overrides the run directory (round 5: merged 55-checkpoint run)
sm = R.PublicScoreModel(cells, evals)
steps = sm.steps
T = len(steps)
mk = R.Masker(cx)
S = sm.scores(sm.weights(B, SEED))
arr = np.stack([S[n] for n in cx.names], axis=1)           # [B+1, cells, T]


def barcodes(masks):
    mono, acc = [], 0
    for m in masks:
        acc |= m
        mono.append(acc)
    return R.fast_barcode(cx, masks, 0, 'union'), R.fast_barcode(cx, mono, 0, 'union')


def finite(bars):
    return [b for b in bars if b[1] < 2 * (T - 1)]


def matched(bar, pool):
    return any(abs(bar[0] - c[0]) <= 2 and abs(bar[1] - c[1]) <= 2 for c in pool)


def node_step(j):
    return steps[j // 2] if j % 2 == 0 else f'{steps[j // 2]}/{steps[j // 2 + 1]}'


pt_masks = mk.masks(arr[0], THETA)
zz0, mo0 = barcodes(pt_masks)
fz0, fm0 = finite(zz0), finite(mo0)
hit_z = np.zeros(len(fz0))
hit_m = np.zeros(len(fm0))
more = 0
diffs = []
for b in range(1, B + 1):
    zz, mo = barcodes(mk.masks(arr[b], THETA))
    fz, fm = finite(zz), finite(mo)
    hit_z += [matched(x, fz) for x in fz0]
    hit_m += [matched(x, fm) for x in fm0]
    more += len(fz) > len(fm)
    diffs.append(len(fz) - len(fm))


def describe(bar, sup, kind):
    s, e = bar
    return dict(kind=kind, start_node=s, end_node=e, start=node_step(s), end=node_step(e), finite=e < 2 * (T - 1),
                ends_on_bridge=e % 2 == 1, support=None if sup is None else round(float(sup), 3))


mono_nodes = {(x[0], x[1]) for x in mo0}
out = dict(theta=THETA, B=B, steps=steps,
           zigzag=[dict(describe(x, hit_z[fz0.index(x)] / B if x in fz0 else None, 'zigzag'), in_mono=(x[0], x[1]) in mono_nodes) for x in sorted(zz0)],
           mono=[describe(x, hit_m[fm0.index(x)] / B if x in fm0 else None, 'mono') for x in sorted(mo0)],
           counts=dict(zigzag_bars=len(zz0), zigzag_finite=len(fz0), mono_bars=len(mo0), mono_finite=len(fm0)),
           bootstrap=dict(share_zigzag_more_finite=more / B, diff_quantiles=[float(q) for q in np.quantile(diffs, [.05, .5, .95])]))
_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf8')
print(json.dumps(out['counts']), json.dumps(out['bootstrap']))
for d in out['zigzag'] + out['mono']:
    if d['finite']:
        print(d['kind'], d['start'], '->', d['end'], 'bridge-end' if d['ends_on_bridge'] else 'checkpoint-end', 'support', d['support'])
