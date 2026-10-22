"""Cross-arm summary of the FORGET3 confirmation arms for the examples section of the manuscript.

Reads zigzag_v5/paper_sec4/mv_counts.json and theta_sweep.json (produced by mv_counts.py / theta_sweep.py) and prints:
  * per arm and theta: number of union-bridge-only H1 classes (transition classes) and at which bridges;
  * per arm: historical-union beta_1, supported dimension, artifact dimension at the end (K_B, theta = 0.8);
  * delta-robust K_B H1 events at theta = 0.8 for delta in {0, 0.02, 0.05, 0.10}, grouped into named events, with the
    number of arms (out of 6) in which each event is robust.
Writes forget3_summary.json next to this script (used by make_figures.py).
"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent          # repository root (mirrors the zigzag_v5 layout)
mv = json.loads((V5 / 'paper_sec4' / 'mv_counts.json').read_text(encoding='utf8'))
sw = json.loads((V5 / 'paper_sec4' / 'theta_sweep.json').read_text(encoding='utf8'))

out = dict(arms={}, events={})
print('== transition classes (union-only H1 bars, K_B) per arm and theta ==')
for arm in mv['arms']:
    row = {}
    for th, d in arm['thetas'].items():
        br = [(b['step_a'], b['step_b'], b['u1']) for b in d['K_B']['bridges_with_bridge_only_classes'] if b['u1']]
        row[th] = br
    hist = arm['thetas']['0.8']['K_B']['hist_union_final']
    out['arms'][arm['run']] = dict(transition_by_theta=row, hist_beta1=hist['beta'], hist_artifact=hist['spurious'],
                                   hist_supported=hist['beta'] - hist['spurious'],
                                   union_H1_bars_08=arm['thetas']['0.8']['K_B']['union_H1_bars'],
                                   intersection_H1_bars_08=arm['thetas']['0.8']['K_B']['intersection_H1_bars'])
    print(arm['run'], {th: v for th, v in row.items() if v}, '| hist beta1', hist['beta'], 'artifact', hist['spurious'])

print('\n== delta-robust K_B H1 events at theta = 0.8 ==')


def name_event(step, kind):
    if kind == 'death' and 288 < step <= 352:
        return 'death of the original loop (forgetting, 288-352)'
    if kind == 'birth' and step <= 288:
        return 'birth of the original loop (learning, <=288)'
    if kind == 'birth' and 288 < step <= 416:
        return 'birth of the replacement loop (336-416)'
    if kind == 'death' and 480 < step <= 512:
        return 'death of the replacement loop (480-512)'
    if kind == 'birth' and step >= 496:
        return 'birth of the recovered loop (>=496)'
    if kind == 'transition':
        return f'transition class at bridge ending {step}'
    return f'{kind} at {step}'


counts = collections.defaultdict(lambda: collections.defaultdict(set))
for arm in sw:
    steps = arm['steps']
    rc = arm['robust_counts']['K_B']['0.8']
    for dl, d in rc.items():
        for i, (b, dd, tr) in enumerate(d['H1']):
            to_step = steps[i + 1]
            if b:
                counts[name_event(to_step, 'birth')][dl].add(arm['run'])
            if dd:
                counts[name_event(to_step, 'death')][dl].add(arm['run'])
            if tr:
                counts[name_event(to_step, 'transition')][dl].add(arm['run'])
deltas = sorted({dl for ev in counts.values() for dl in ev}, key=float)
for ev in sorted(counts):
    out['events'][ev] = {dl: sorted(counts[ev][dl]) for dl in deltas}
    print(f'{ev:55s}', {dl: len(counts[ev][dl]) for dl in deltas})
eps = {arm['run']: arm['bootstrap_eps'] for arm in sw}
out['bootstrap_eps'] = eps
print('\n== bootstrap eps (median over replicates): sup, q95, sd of mid-range scores ==')
for r, e in eps.items():
    print(f"{r}: sup_median={e['sup_median']:.3f} sup_q95={e['sup_q95']:.3f} q95_median={e['q95_median']:.3f} sd_mid={e['sd_mid_scores_median']:.3f} sd_mid_q95={e['sd_mid_scores_q95']:.3f}")
(HERE / 'forget3_summary.json').write_text(json.dumps(out, indent=1), encoding='utf8')
