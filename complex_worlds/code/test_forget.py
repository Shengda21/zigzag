"""RCN12_FORGET (768 steps): ground truth, curriculum audit and a synthetic
forgetting/recovery scenario pushed through the spec-driven analysis.

Scenario (world 611 probes, checkpoints every 16 steps): a taught cell is
correct from introduce+32 until retire+16 (drop), correct again from
reintroduce+32; latent and control cells never; route rows follow their edges;
chain-format practice rows exist in the stream. Expected (theta=0.8): the
B-triangle H1 class has two disjoint zigzag intervals ([96,288] and
[512,768]), edge_loss + H0 split at 304, fC123 clean_fill at 160 and 576;
union-monotonised ordinary persistence keeps the B triangle alive to the end.
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
import cw_zigzag as CW        # noqa: E402
import spec_world as S        # noqa: E402

SPEC = json.loads((R / 'specs' / 'RCN12_FORGET.json').read_text(encoding='utf8'))
DATA = R / 'data' / 'RCN12_FORGET' / 'world611'
CELLS = {**{e['name']: e for e in SPEC['edges']}, **{f['name']: f for f in SPEC['faces']}}
NEVER = {'A1A3', 'B2B4', 'fB234', 'fA123', 'fA143', 'A2B2', 'C2B4'}
STEPS = SPEC['steps']


def taught_present(name, step):
    c = CELLS[name]
    if c['kind'] not in ('taught', 'taught_inconsistent'):
        return False
    if step < c.get('introduce_step', 0) + 32:
        return False
    r, back = c.get('retire_step'), c.get('reintroduce_step')
    if r is not None and step >= r + 16 and (back is None or step < back + 32):
        return False
    return True


def correct(r, step):
    k = r['kind']
    if k == 'vertex':
        return step >= 16
    if k in ('edge', 'joint'):
        return r['task'] not in NEVER and taught_present(r['task'], step)
    if k == 'route':
        x, y, z = CELLS[r['face']]['path']
        yz = next(e['name'] for e in SPEC['edges'] if e['src'] == y and e['dst'] == z)
        return taught_present(yz, step)
    if k == 'repair':
        return taught_present('fC123', step) and step >= SPEC['repair_introduce_step'] + 32
    if k == 'transfer':
        return all(e not in NEVER and taught_present(e, step) for e in r['requires'])
    raise ValueError(k)


def union_persistence_h1(cx, masks):
    acc = 0
    filt = []
    for m in masks:
        acc |= m
        filt.append(acc)
    dims = [len(cx.homology(m, 1)[0]) for m in filt]
    arrows = [(i, i + 1, cx.induced(filt[i], filt[i + 1], 1)) for i in range(len(filt) - 1)]
    return sorted(CW.decompose(dims, arrows)), dims


def main():
    assert STEPS == 768
    gt = json.loads((DATA / 'GROUND_TRUTH.json').read_text(encoding='utf8'))
    assert gt['step_phases'] == SPEC['expected_phases'], gt['step_phases']
    probes = [json.loads(s) for s in (DATA / 'PROBES.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    assert sum(r['kind'] == 'route' for r in probes) == 4 * 2 * 32
    stream = [json.loads(s) for s in (DATA / 'TRAIN_STREAM.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    batch = sum(SPEC['per_step'].values())
    assert len(stream) == batch * STEPS
    for i, r in enumerate(stream):
        if r['kind'] in ('edge', 'joint'):
            assert S.active_at(CELLS[r['task']], i // batch), (r['id'], i // batch)
        if r['kind'] == 'transfer':
            assert all(S.active_at(CELLS[e], i // batch) for e in r['requires']), r['id']
    joint = [r for r in stream if r['kind'] == 'joint']
    assert joint and all(isinstance(r['override'], bool) for r in joint) and 0.4 <= sum(r['override'] for r in joint) / len(joint) <= 0.6
    assert sum(r['kind'] == 'transfer' for r in stream) > 0                     # chain-format practice present
    assert not any(r['kind'] in ('edge', 'joint') and r['task'] in ('B1B2', 'B2B3', 'B1B3', 'B3B4') for r in stream[288 * batch:480 * batch])
    assert any(r['kind'] == 'edge' and r['task'] == 'B1B2' for r in stream[480 * batch:512 * batch])
    assert not any(r['kind'] == 'joint' and r['task'] == 'fC123' for r in stream[352 * batch:544 * batch])
    root = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun')) / 'forget_test'
    if root.exists():
        shutil.rmtree(root)
    run = root / 'world611_SYN'
    run.mkdir(parents=True)
    for step in range(0, STEPS + 1, 16):
        with (run / f'eval_{step:04d}.jsonl').open('w', encoding='utf8') as f:
            for r in probes:
                pred = r['gold'] if correct(r, step) else 'X'
                f.write(json.dumps(dict(r, prediction=pred, step=step, **Z.score(r, pred)), ensure_ascii=False) + '\n')
    (run / 'COMPLETE.json').write_text('{}')
    res = A.analyze_run(run, str(DATA / 'COMPLEX.json'), n_perm=30, seed=5)
    A.write_outputs(res, root / 'out')
    steps = res['steps']
    i = {s: k for k, s in enumerate(steps)}
    pt = res['per_theta']['0.8']
    b0 = [b[0] for b in pt['betti']]
    b1 = [b[1] for b in pt['betti']]
    expect = {160: (3, 2), 288: (2, 2), 320: (5, 1), 384: (5, 2), 528: (2, 3), 592: (2, 2), 768: (2, 2)}
    for s, (e0, e1) in expect.items():
        assert (b0[i[s]], b1[i[s]]) == (e0, e1), (s, b0[i[s]], b1[i[s]], (e0, e1))
    trans = pt['transitions']
    assert any(t['to_step'] == 304 and t['h1'] == 'edge_loss' and t['h0'] == 'split' for t in trans), [t for t in trans if t['to_step'] == 304]
    fills = [t['to_step'] for t in trans if t['h1'] == 'clean_fill' and 'fC123' in t['added']]
    assert fills == [160, 576], fills
    cx = A.design_from_spec(str(DATA / 'COMPLEX.json'))['complex']()
    masks = [cx.mask_from_scores({c: res['trajectory'][c][k] for c in cx.names}, 0.8) for k in range(len(steps))]
    zz = cx.barcode(tuple(masks), 1, 'union')
    zz_ck = [[steps[j // 2] for j in range(a, b + 1) if j % 2 == 0] for a, b in zz]
    long_bars = sorted((c[0], c[-1]) for c in zz_ck if len(c) >= 2)
    u_bars, u_dims = union_persistence_h1(cx, masks)
    u_steps = sorted((steps[a], steps[b]) for a, b in u_bars)
    print(json.dumps(dict(status='FORGET_TESTS_PASSED', beta0=b0, beta1=b1, zigzag_H1_bars=long_bars, union_monotonised_bars=u_steps,
                          transfer_final={k: v for k, v in res['transfer'][-1].items() if k.startswith('chain')}, verdict=res['synchrony']['verdict'])))
    assert (96, 288) in long_bars and (512, 768) in long_bars, long_bars             # zigzag: the B triangle dies and is reborn
    assert (240, 560) in long_bars or any(a >= 224 and b <= 576 for a, b in long_bars), long_bars   # C hole reopened while fC123 forgotten
    assert not any(a >= 96 and b < 768 and a < 400 for a, b in u_steps if (a, b) != (32, 144)), u_steps   # union: no B death, no C reopening
    assert sum(1 for a, b in u_steps if b == 768) >= 2, u_steps


if __name__ == '__main__':
    main()
