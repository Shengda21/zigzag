"""Ground-truth topology of RCN12 and a synthetic positive/negative-control
scenario pushed through the spec-driven analysis.

Scenario (world 601 probes, checkpoints every 16 steps, 0..512):
  taught cells pass 32 steps after their introduce_step;
  latent relabel bridges A3B1 / B4C1 pass at 320 / 352 (after C4A1 is taught at 256);
  latent composition edge A1A3 and face fB234 never pass; control edges never pass;
  chains pass once all their edges pass.
Expected: fC123 clean_fill at 160; A3B1 birth = H0 merge at 320 (beta0 2->1);
B4C1 birth closes the cross-domain cycle at 352 (beta1 +1); ledger shows A1A3,
fB234 and both controls never entering; final betti = [1, 3].
"""
from __future__ import annotations
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
ROOT = R.parent
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
sys.path.insert(0, str(HERE))
import zz_score as Z          # noqa: E402
import zz_analyze as A        # noqa: E402
import spec_world as S        # noqa: E402

SPEC = json.loads((R / 'specs' / 'RCN12.json').read_text(encoding='utf8'))
DATA = R / 'data' / 'RCN12' / 'world601'
PASS_AT = {'A3B1': 320, 'B4C1': 352}
NEVER = {'A1A3', 'B2B4', 'fB234', 'fA123', 'fA143', 'A2B2', 'C2B4'}


def pass_step(r, intro):
    task = r['task']
    if r['kind'] == 'vertex':
        return 16
    if task in NEVER:
        return None
    if task in PASS_AT:
        return PASS_AT[task]
    if r['kind'] in ('edge', 'joint'):
        return intro[task] + 32
    if r['kind'] == 'repair':
        return SPEC['repair_introduce_step'] + 32
    if r['kind'] == 'transfer':
        steps = []
        for e in r['requires']:
            if e in NEVER:
                return None
            steps.append(PASS_AT.get(e, intro.get(e, 0) + 32))
        return max(steps)
    raise ValueError(r['kind'])


def main():
    gt = json.loads((DATA / 'GROUND_TRUTH.json').read_text(encoding='utf8'))
    assert gt['phases']['taught_all'] == [2, 2] and gt['bridge_sequence']['A3B1'] == [1, 2] and gt['bridge_sequence']['A3B1+B4C1'] == [1, 3], gt
    assert gt['earliest_possible_step'] == {'A1A3': 0, 'B2B4': 64, 'A3B1': 256, 'B4C1': 256, 'fA123': 0, 'fA143': 64, 'fB234': 64}, gt['earliest_possible_step']
    intro = {e['name']: e['introduce_step'] for e in SPEC['edges'] if e['kind'] == 'taught'}
    intro.update({f['name']: f['introduce_step'] for f in SPEC['faces'] if f['kind'] in ('taught', 'taught_inconsistent')})
    probes = [json.loads(s) for s in (DATA / 'PROBES.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    root = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun')) / 'spec_test'
    if root.exists():
        shutil.rmtree(root)
    run = root / 'world601_SYN'
    run.mkdir(parents=True)
    for step in range(0, 513, 16):
        with (run / f'eval_{step:04d}.jsonl').open('w', encoding='utf8') as f:
            for r in probes:
                ps = pass_step(r, intro)
                pred = r['gold'] if (ps is not None and step >= ps) else 'X'
                f.write(json.dumps(dict(r, prediction=pred, step=step, **Z.score(r, pred)), ensure_ascii=False) + '\n')
    (run / 'COMPLETE.json').write_text('{}')
    res = A.analyze_run(run, str(DATA / 'COMPLEX.json'), n_perm=40, seed=3)
    A.write_outputs(res, root / 'out')
    pt = res['per_theta']['0.8']
    b0 = [b[0] for b in pt['betti']]
    b1 = [b[1] for b in pt['betti']]
    steps = res['steps']
    i = {s: k for k, s in enumerate(steps)}
    assert b1[i[512]] == 3 and b0[i[512]] == 1, (b0, b1)
    assert b0[i[304]] == 2 and b0[i[320]] == 1, b0                     # first bridge merges
    assert b1[i[336]] == 2 and b1[i[352]] == 3, b1                     # second bridge closes the long cycle
    fills = [t for t in pt['transitions'] if t['h1'] == 'clean_fill']
    assert any(t['to_step'] == 160 and t['added'] == ['fC123'] for t in fills), fills
    merges = [t for t in pt['transitions'] if t['h0'] == 'merge']
    assert any(t['to_step'] == 320 for t in merges), merges
    led = res['ledger']
    assert led['cells']['A3B1']['0.8']['first_step'] == 320 and led['cells']['B4C1']['0.8']['first_step'] == 352
    assert led['cells']['A1A3']['0.8']['first_step'] is None and led['cells']['fB234']['0.8']['first_step'] is None
    assert led['control_violations'] == {}, led['control_violations']
    tr = {t['step']: t for t in res['transfer']}
    assert tr[352]['chain_A1_A2_A3_B1'] == 1.0 and tr[304]['chain_A1_A2_A3_B1'] == 0.0
    assert tr[512]['control_edges'] == 0.0
    print(json.dumps(dict(status='SPEC_WORLD_TESTS_PASSED', ground_truth=gt['phases'], bridge_sequence=gt['bridge_sequence'],
                          beta0=b0, beta1=b1, verdict=res['synchrony']['verdict'])))


if __name__ == '__main__':
    main()
