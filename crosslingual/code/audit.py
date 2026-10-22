"""CPU audit of the cross-lingual package (no model, no network)."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CL = HERE.parent
ROOT = CL.parent
DATA = CL / 'data'
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(HERE))
import zz_score as Z          # noqa: E402
import templates as TP        # noqa: E402

META = ROOT.parent / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15'
FILES = ['SPEC_TRAIN.jsonl', 'FMT_TRAIN.jsonl', 'GEN_MULTI.jsonl', 'GEN_MONO.jsonl', 'PROBES.jsonl']
MAX_LEN, MAX_ANSWER = 1024, 96


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def budget(tok, rs):
    pp = [tok.apply_chat_template([dict(role='user', content=r['prompt'])], tokenize=False, add_generation_prompt=True) for r in rs]
    ff = [tok.apply_chat_template([dict(role='user', content=r['prompt']), dict(role='assistant', content=r['gold'])], tokenize=False) for r in rs]
    pl = [len(x) for x in tok(pp, add_special_tokens=False)['input_ids']]
    fl = [len(x) for x in tok(ff, add_special_tokens=False)['input_ids']]
    return dict(prompt_max=max(pl), full_max=max(fl), answer_max=max(f - p for p, f in zip(pl, fl)), full_mean=round(sum(fl) / len(fl), 1))


def audit_world(w, tok):
    d = DATA / f'world{w}'
    m = json.loads((d / 'MANIFEST.json').read_text(encoding='utf8'))
    assert set(m['hashes']) == set(FILES)
    for name, h in m['hashes'].items():
        assert sha(d / name) == h, name
    R_ = {f: rows(d / f) for f in FILES}
    spec, fm, multi, mono, probes = (R_[f] for f in FILES)
    steps = m['steps']
    assert len(spec) == steps * 32 and len(fm) == steps * 16 and len(multi) == len(mono) == steps * 16
    assert {r['lang'] for r in spec} == {'en'} and {r['template'] for r in spec} <= {0, 1, 2}
    assert {r['lang'] for r in fm} == set(TP.TRAIN_LANGS) and {r['lang'] for r in multi} == set(TP.TRAIN_LANGS) and {r['lang'] for r in mono} == {'en'}
    assert collections.Counter(r['lang'] for r in multi) == {l: steps * 16 // 3 + (1 if i < steps * 16 % 3 else 0) for i, l in enumerate(TP.TRAIN_LANGS)}
    assert all(a['kind'] == b['kind'] and a['item'] == b['item'] and a['paired_prompt_tokens'] == b['paired_prompt_tokens'] for a, b in zip(multi, mono))
    # probes: languages, templates, held-out roots and items
    held = set(map(tuple, m['split']['structure'])) | set(map(tuple, m['split']['independent']))
    for r in spec + fm:
        if len(r['root']) == 2:
            assert tuple(r['root']) not in held, r['id']
    pc = collections.Counter((r['vocabulary'], r['lang']) for r in probes)
    assert pc[('spec', 'en')] == 720 and all(pc[('spec', l)] == 512 for l in ('zh', 'ru', 'ja')) and all(pc[('general', l)] == 64 for l in TP.TRAIN_LANGS)
    assert all(tuple(r['root']) in held for r in probes if r['vocabulary'] == 'spec')
    hb = set(map(tuple, m['general_held']['belebele']))
    hx = set(m['general_held']['xquad'])
    assert all((tuple(r['item']) in hb) if r['kind'] == 'belebele' else (r['item'] in hx) for r in probes if r['vocabulary'] == 'general')
    assert not ({tuple(r['item']) for r in multi if r['kind'] == 'belebele'} & hb) and not ({r['item'] for r in multi if r['kind'] == 'xquad'} & hx)
    # every probe item exists in every probe language with the same gold (symbolic tasks) -> cross-lingual cells are well defined
    by = collections.defaultdict(dict)
    for r in probes:
        if r['vocabulary'] == 'spec':
            key = r['id'].replace(f"probe_{r['lang']}_{r['template']}_", '')
            by[key][(r['lang'], r['template'])] = r['gold']
    for key, langs in by.items():
        assert len({g for g in langs.values()}) == 1, key
        assert all(any(l == lang for l, _ in langs) for lang in ('en', 'zh', 'ru', 'ja')), key
    for f, rs in R_.items():
        assert all(Z.score(x, x['gold'])['correct'] == 1 for x in rs), f
    budgets = {}
    if tok is not None:
        for f, rs in R_.items():
            b = budget(tok, rs)
            assert b['full_max'] <= MAX_LEN and b['answer_max'] <= MAX_ANSWER, (f, b)
            budgets[f] = b
        for lang in TP.LANGS:
            budgets[f'probes_{lang}'] = budget(tok, [r for r in probes if r['lang'] == lang])
    return dict(world=w, counts={f: len(r) for f, r in R_.items()}, probe_counts={f'{a}:{b}': v for (a, b), v in pc.items()},
                cross_lingual_items=len(by), held_out_leakage=0, scorer_gold_consistent=True, token_budgets=budgets, version=m['version'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    if a.quick:
        for w in (501, 502):
            m = json.loads((DATA / f'world{w}' / 'MANIFEST.json').read_text(encoding='utf8'))
            for name, h in m['hashes'].items():
                assert sha(DATA / f'world{w}' / name) == h, name
        print('HASHES_OK')
        return
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(META)
    except Exception as e:      # noqa: BLE001
        print('tokenizer unavailable:', e)
    res = [audit_world(w, tok) for w in (501, 502)]
    out = dict(status='PASS_INPUT_ONLY_NOT_TRAINED', version='CROSSLINGUAL_20260922', time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), worlds=res)
    (DATA / 'CROSSLINGUAL_CPU_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status=out['status'], budgets={r['world']: {k: (v['full_max'], v['full_mean']) for k, v in r['token_budgets'].items()} for r in res})))


if __name__ == '__main__':
    main()
