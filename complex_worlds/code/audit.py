"""CPU audit of spec-generated worlds: hashes, stream composition, curriculum
(no cell before its introduce_step, no latent/control cell ever trained),
held-out roots, probe coverage of every cell, scorer/gold consistency, exact
chat-template token budgets."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
ROOT = R.parent
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
import zz_score as Z          # noqa: E402
sys.path.insert(0, str(HERE))
from spec_world import active_at   # noqa: E402

META = ROOT.parent / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15'


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def budget(tok, rs):
    pp = [tok.apply_chat_template([dict(role='user', content=r['prompt'])], tokenize=False, add_generation_prompt=True) for r in rs]
    ff = [tok.apply_chat_template([dict(role='user', content=r['prompt']), dict(role='assistant', content=r['gold'])], tokenize=False) for r in rs]
    pl = [len(x) for x in tok(pp, add_special_tokens=False)['input_ids']]
    fl = [len(x) for x in tok(ff, add_special_tokens=False)['input_ids']]
    return dict(prompt_max=max(pl), full_max=max(fl), answer_max=max(f - p for p, f in zip(pl, fl)))


def audit_world(spec, d, tok):
    m = json.loads((d / 'MANIFEST.json').read_text(encoding='utf8'))
    for name, h in m['hashes'].items():
        assert sha(d / name) == h, name
    cx = json.loads((d / 'COMPLEX.json').read_text(encoding='utf8'))
    stream = rows(d / 'TRAIN_STREAM.jsonl')
    probes = rows(d / 'PROBES.jsonl')
    batch, steps = m['batch'], m['steps']
    assert len(stream) == batch * steps
    cells = {**{e['name']: e for e in spec['edges']}, **{f['name']: f for f in spec['faces']}}
    forbidden = set(cx['latent_cells']) | set(cx['control_cells'])
    for i, r in enumerate(stream):
        s = i // batch
        if r['kind'] == 'edge' or r['kind'] == 'joint':
            assert r['task'] not in forbidden, r['id']
            assert active_at(cells[r['task']], s), (r['id'], s)          # introduce_step and retire/reintroduce windows
        if r['kind'] == 'repair':
            assert s >= spec['repair_introduce_step'], r['id']
    per_step = collections.Counter(r['kind'] for r in stream[:batch])
    assert sum(per_step.values()) == batch
    joint_rows = [r for r in stream if r['kind'] == 'joint']
    assert all(isinstance(r['override'], bool) and isinstance(r['id'], str) for r in joint_rows), 'joint rows: override must be bool, id str (DEFECT_20260923)'
    if joint_rows:
        frac = sum(r['override'] for r in joint_rows) / len(joint_rows)
        assert 0.4 <= frac <= 0.6, f'override fraction {frac:.2f} not balanced'
    assert len({r['id'] for r in stream}) == len(stream), 'duplicate training ids'
    for i, r in enumerate(stream):
        if r['kind'] == 'transfer':          # chain-format practice rows
            assert all(active_at(cells[e], i // batch) for e in r['requires']), r['id']
    held = set(map(tuple, m['split']['structure'])) | set(map(tuple, m['split']['independent']))
    for r in stream:
        if len(r['root']) == 2:
            assert tuple(r['root']) not in held, r['id']
    covered = {r['task'] for r in probes}
    for n, dim, _ in cx['cells']:
        assert (('v' + n) if dim == 0 else n) in covered, n
    assert all(tuple(r['root']) in held for r in probes)
    for r in probes:
        if r['kind'] == 'route':
            assert r['item'] == r['root'] and r['face'] in cx['face_routes'] and r['task'] in cx['face_routes'][r['face']], r['id']
    for r in stream + probes:
        assert Z.score(r, r['gold'])['correct'] == 1, r['id']
    for r in probes[:500]:
        assert Z.score(r, r['gold'] + '|9')['valid_format'] == 0
    budgets = {}
    if tok is not None:
        for name, rs in [('TRAIN_STREAM.jsonl', stream), ('PROBES.jsonl', probes)]:
            b = budget(tok, rs)
            assert b['full_max'] <= 512 and b['answer_max'] <= 96, (name, b)
            budgets[name] = b
    return dict(world=m['world'], train=len(stream), probes=len(probes), probe_kinds=dict(collections.Counter(r['kind'] for r in probes)),
                cells=len(cx['cells']), latent=len(cx['latent_cells']), control=len(cx['control_cells']), curriculum_ok=True,
                latent_or_control_in_stream=0, held_out_leakage=0, token_budgets=budgets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', default='RCN12')
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    spec = json.loads((R / 'specs' / f'{a.spec}.json').read_text(encoding='utf8'))
    out = R / 'data' / a.spec
    worlds = sorted(int(d.name[5:]) for d in out.glob('world*'))
    if a.quick:
        for w in worlds:
            m = json.loads((out / f'world{w}' / 'MANIFEST.json').read_text(encoding='utf8'))
            for name, h in m['hashes'].items():
                assert sha(out / f'world{w}' / name) == h, name
        print('HASHES_OK', worlds)
        return
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(META)
    except Exception as e:      # noqa: BLE001
        print('tokenizer unavailable:', e)
    res = [audit_world(spec, out / f'world{w}', tok) for w in worlds]
    (out / 'CPU_AUDIT.json').write_text(json.dumps(dict(status='PASS_INPUT_ONLY_NOT_TRAINED', spec=a.spec, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                        worlds=res), ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status='PASS', worlds=[(r['world'], r['train'], r['probes'], {k: v['full_max'] for k, v in r['token_budgets'].items()}) for r in res])))


if __name__ == '__main__':
    main()
