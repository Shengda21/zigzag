"""Post-hoc check of Definition 3.3 (ii) conditions for the grokking held-out edge births
(paper §4.4): threshold coverage (>= 5/7 thresholds) and checkpoint refinement (birth on the
original 250-step grid within delta_t = 1000 steps of the birth on the 1000-step analysis grid).
Edge cell score = held-out accuracy over templates 0/1 = summary 'modp:edge:task<edge>' accuracy
(checked equal on a sample row set). Vertices are present by construction (declared amendment).

  python check_grok_tier2.py   ->  check_grok_tier2.json
"""
import glob
import json
import re
from pathlib import Path

V5 = Path(__file__).resolve().parents[1]
RUNS = V5 / 'cloud_return_round2' / 'grokking' / 'runs_confirm'
TH = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
E = ('P_S', 'P_D', 'P_M', 'P_Q', 'Pn_D', 'Ps_S', 'Pi_Q')
DELTA = 1000


def birth(steps, vals, th):
    """first checkpoint with score >= th that is still >= th at the next checkpoint"""
    return next((steps[i] for i in range(len(steps) - 1) if vals[i] >= th and vals[i + 1] >= th), None)


out = dict(pairs=[], delta_t=DELTA, thresholds=TH)
for d in sorted(glob.glob(str(RUNS / 'MODP_world70*_seed*'))):
    fs = sorted(glob.glob(d + '/eval_*.summary.json'), key=lambda f: int(re.findall(r'eval_(\d+)', f)[0]))
    st = [int(re.findall(r'eval_(\d+)', f)[0]) for f in fs]
    S = [json.load(open(f, encoding='utf8')) for f in fs]
    idx4 = [i for i in range(len(st)) if i % 4 == 0 or i == len(st) - 1]      # the 1000-step analysis grid (ZZ_EVAL_STRIDE=4)
    for e in E:
        v = [s[f'modp:edge:task{e}']['accuracy'] for s in S]
        per = {}
        for th in TH:
            b4 = birth([st[i] for i in idx4], [v[i] for i in idx4], th)
            b1 = birth(st, v, th)
            per[str(th)] = dict(birth_1000=b4, birth_250=b1, refined_ok=b4 is not None and b1 is not None and abs(b4 - b1) <= DELTA)
        cov = sum(p['birth_1000'] is not None for p in per.values())
        ref = sum(p['refined_ok'] for p in per.values())
        out['pairs'].append(dict(run=Path(d).name, edge=e, thresholds_born=cov, thresholds_refinement_ok=ref,
                                 meets_ii_conditions=cov >= 5 and ref >= 5, per_theta=per))
n = sum(p['meets_ii_conditions'] for p in out['pairs'])
out['summary'] = dict(pairs=len(out['pairs']), meets=n,
                      coverage_hist={k: sum(p['thresholds_born'] == k for p in out['pairs']) for k in range(8)})
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=1), encoding='utf8')
print(json.dumps(out['summary']))
