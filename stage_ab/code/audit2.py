"""CPU audit of the complete 20260922 Stage A/B package (no model, no network).

Extends audit.py with the COMMON / ATOMIC_EVAL files, zz_score self-consistency
on every gold answer, exact chat-template token budgets for every training and
evaluation row, and leakage checks between COMMON training roots and the
held-out target roots. `--quick` only verifies hashes (used on the GPU box).
"""
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
DATA = R / 'data'
sys.path.insert(0, str(HERE))
import zz_score as Z   # noqa: E402

META = R.parents[1] / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15'
FILES = ['FORMAT_TRAIN.jsonl', 'FORMAT_EVAL.jsonl', 'DEPENDENT.jsonl', 'INDEPENDENT.jsonl', 'EVAL_TARGET.jsonl', 'COMMON.jsonl', 'ATOMIC_EVAL.jsonl']


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def token_budget(tok, rs):
    pp = [tok.apply_chat_template([dict(role='user', content=r['prompt'])], tokenize=False, add_generation_prompt=True) for r in rs]
    ff = [tok.apply_chat_template([dict(role='user', content=r['prompt']), dict(role='assistant', content=r['gold'])], tokenize=False) for r in rs]
    pl = [len(x) for x in tok(pp, add_special_tokens=False)['input_ids']]
    fl = [len(x) for x in tok(ff, add_special_tokens=False)['input_ids']]
    ans = [f - p for p, f in zip(pl, fl)]
    return dict(prompt_max=max(pl), full_max=max(fl), answer_max=max(ans), full_min=min(fl))


def audit_world(w, tok):
    d = DATA / f'world{w}'
    m = json.loads((d / 'MANIFEST.json').read_text(encoding='utf-8'))
    assert set(m['hashes']) == set(FILES), set(m['hashes']) ^ set(FILES)
    for name, h in m['hashes'].items():
        assert sha(d / name) == h, name
    R_ = {f: rows(d / f) for f in FILES}
    fmt, fe, dep, ind, ev, com, at = (R_[f] for f in FILES)
    assert len(fmt) == 8192 and len(fe) == 64 and len(dep) == len(ind) == 16384 and len(ev) == 768
    assert len(com) == m['common_steps'] * m['common_batch'] == 32768 and len(at) == 640
    # vocabularies, conditions, templates
    assert all(x['vocabulary'] == 'aux' for x in fmt + fe + dep + ind) and all(x['vocabulary'] == 'target' for x in ev)
    assert {x['condition'] for x in dep} == {'DEPENDENT'} and {x['condition'] for x in ind} == {'INDEPENDENT'}
    assert {x['template'] for x in fmt + dep + ind + com} <= {0, 1, 2} and {x['template'] for x in ev} == {0, 1, 3}
    assert collections.Counter(x['vocabulary'] for x in com) == {'target': 16384, 'aux': 16384}
    assert collections.Counter((x['vocabulary'], x['template']) for x in at) == {('target', 0): 160, ('target', 1): 160, ('target', 3): 160, ('aux', 0): 160}
    assert all(x['task'] in Z.ATOMIC_EDGES for x in at) and {x['task'] for x in com} == set(('vF', 'vD', 'vP', 'vR') + Z.ATOMIC_EDGES)
    # pairing
    assert all(a['schema'] == b['schema'] and a['paired_prompt_tokens'] == b['paired_prompt_tokens'] and a['template'] == b['template']
               and a['domain'] == b['domain'] and a['kind'] == b['kind'] for a, b in zip(dep, ind))
    assert all(a['gold'] != b['gold'] or a['second_root'] != b['second_root'] for a, b in zip(dep, ind))
    # root hygiene
    ts = {k: set(map(tuple, v)) for k, v in m['target_split'].items()}
    as_ = {k: set(map(tuple, v)) for k, v in m['auxiliary_split'].items()}
    held_t = ts['structure'] | ts['independent']
    held_a = as_['structure'] | as_['independent']
    assert all(tuple(x['root']) in as_['train'] and tuple(x['second_root']) in as_['train'] for x in dep + ind)
    assert all(tuple(x['root']) in ts['structure'] for x in ev if x['kind'] == 'joint')
    assert all(tuple(x['root']) in ts['independent'] for x in ev if x['kind'] == 'repair')
    for x in com:
        if len(x['root']) == 2:
            assert tuple(x['root']) not in (held_t if x['vocabulary'] == 'target' else held_a), x['id']
    assert all(tuple(x['root']) in (ts if x['vocabulary'] == 'target' else as_)['structure'] for x in at)
    assert all(tuple(x['root']) in held_a for x in fe) and all(tuple(x['root']) in as_['train'] for x in fmt)
    # scorer agrees with gold everywhere; wrong answers are rejected
    for f, rs in R_.items():
        assert all(Z.score(x, x['gold'])['correct'] == 1 for x in rs), f
        assert all(Z.score(x, x['gold'] + '|9')['valid_format'] == 0 for x in rs[:200]), f
    # exact token budgets under the Qwen2.5 chat template
    budgets = {}
    if tok is not None:
        for f, rs in R_.items():
            b = token_budget(tok, rs)
            assert b['full_max'] <= 512 and b['answer_max'] <= 96, (f, b)
            budgets[f] = b
    counts = {f: len(rs) for f, rs in R_.items()}
    return dict(world=w, counts=counts, paired_prompt_tokens=True, target_joint_repair_train=0, hashes_match=True,
                common_target_records=16384, common_aux_records=16384, atomic_eval_records=640,
                held_out_leakage=0, scorer_gold_consistent=True, token_budgets=budgets, version=m['version'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.quick:
        for w in (401, 402):
            d = DATA / f'world{w}'
            m = json.loads((d / 'MANIFEST.json').read_text(encoding='utf-8'))
            for name, h in m['hashes'].items():
                assert sha(d / name) == h, name
        print('HASHES_OK')
        return
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(META)
    except Exception as e:      # noqa: BLE001
        print('tokenizer unavailable, skipping token budgets:', e)
    results = [audit_world(w, tok) for w in (401, 402)]
    out = dict(status='PASS_INPUT_ONLY_NOT_TRAINED', version='STAGE_AB_20260922', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               worlds=results, code_sha256={f: sha(HERE / f)[:16] for f in ['zz_score.py', 'zz_train.py', 'run_stage_ab.py', 'queue_stage_ab.py', 'prepare_common.py']})
    (DATA / 'STAGE_AB_CPU_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=out['status'], worlds=[(r['world'], r['counts'], {k: v['full_max'] for k, v in r['token_budgets'].items()}) for r in results]), ensure_ascii=False))


if __name__ == '__main__':
    main()
