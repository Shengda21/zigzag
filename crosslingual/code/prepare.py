"""Build the cross-lingual worlds 501/502 (no model, no network; raw general
data must already be in crosslingual/raw via download_general.py).

Per world the package contains:
  SPEC_TRAIN.jsonl   English-only specialised layer: vertex/edge/joint/repair
                     tasks on a fresh synthetic codebook, 512 steps x 32.
  FMT_TRAIN.jsonl    format-copy in en/zh/ru (shared by both conditions), 512 x 16.
  GEN_MULTI.jsonl    Belebele + XQuAD items, language cycling en/zh/ru, 512 x 16.
  GEN_MONO.jsonl     the same items in English; prompt tokens paired with MULTI.
  PROBES.jsonl       never trained: specialised probes in en/zh/ru/ja on held-out
                     structure roots (atomic, joint, repair) and held-out general
                     items in en/zh/ru.
  MANIFEST.json      codebook, root splits, general item splits, hashes, config.

Zero prior knowledge of the specialised layer is guaranteed by the synthetic
codebook; the general layer is real public data (CC-BY-SA-4.0) and may be
partially known to the base model, which is allowed for that layer.
"""
from __future__ import annotations
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CL = HERE.parent                      # crosslingual/
ROOT = CL.parent                      # zigzag_v5/
RAW = CL / 'raw'
OUT = CL / 'data'
sys.path.insert(0, str(ROOT / 'reproduction'))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(HERE))
import make_d1 as old                # noqa: E402
import zz_score as Z                 # noqa: E402
import templates as TP               # noqa: E402

WORLDS = (501, 502)
STEPS = 512
SPEC_PER_STEP, FMT_PER_STEP, GEN_PER_STEP = 32, 16, 16
SPEC_MIX = dict(vertex=4, edge=16, joint=6, repair=6)        # per step, sums to 32
PROBE_ROOTS = 16
GEN_HELD = dict(belebele=32, xquad=32)
XQUAD_WINDOW = dict(en=260, zh=130, ru=300)
EDGES = old.CELLS[4:9]
VERTS = old.CELLS[:4]
LANG_FILES = dict(belebele=dict(en='eng_Latn', zh='zho_Hans', ru='rus_Cyrl'), xquad=dict(en='en', zh='zh', ru='ru'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def jwrite(p, rs):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rs), encoding='utf8')
    return sha(p)


# ------------------------------------------------------------------ specialised
def vertex(book, x, c, lang, t, rid, split):
    seq = old.view(book, c, x)
    return dict(id=rid, task=f'v{c}', kind='base', vocabulary='spec', lang=lang, template=t, root=list(x), split=split,
                prompt=TP.vertex_prompt(lang, t, seq), gold=','.join(f'{book[c][a]:03d}' for a in x), schema=['digits'], field_length=len(x))


def edge(book, x, cell, lang, t, rid, split):
    s, d = cell[1:]
    return dict(id=rid, task=cell, kind='base', vocabulary='spec', lang=lang, template=t, root=list(x), split=split,
                prompt=TP.edge_prompt(lang, t, s, d, old.view(book, s, x)), gold=old.view(book, d, x), schema=[d], field_length=len(x))


def joint(book, x, z, domain, lang, t, rid, override, split):
    m = 'D' if domain == 'A' else 'R'
    active = z if override else x
    prompt = TP.joint_prompt(lang, t, old.view(book, 'F', x), m, old.view(book, m, z), override)
    return dict(id=rid, task='f' + domain, kind='joint', vocabulary='spec', lang=lang, domain=domain, template=t, root=list(x),
                active_root=list(active), override=override, split=split, prompt=prompt,
                gold='|'.join([old.view(book, 'P', x), old.view(book, m, active), old.view(book, 'P', active), str(int(x == active))]),
                schema=['P', m, 'P', 'binary'], field_length=len(x))


def repair(book, x, z, bad, domain, lang, t, rid, override, error, split):
    m = 'D' if domain == 'A' else 'R'
    active = z if override else x
    middle = bad if error == 1 else active
    end = bad if error in (1, 2) else active
    prompt = TP.repair_prompt(lang, t, old.view(book, 'F', x), m, old.view(book, m, z), override, old.view(book, m, middle), old.view(book, 'P', end))
    return dict(id=rid, task='repair', kind='repair', vocabulary='spec', lang=lang, domain=domain, template=t, root=list(x),
                active_root=list(active), wrong_root=list(bad), override=override, error=error, split=split, prompt=prompt,
                gold='|'.join([old.view(book, m, active), old.view(book, 'P', active), str(error)]), schema=[m, 'P', 'step'], field_length=len(x))


def fmt(book, x, lang, t, rid):
    a, b = x
    fields = [old.view(book, 'D', x), old.view(book, 'P', (a, b)), old.view(book, 'P', (b, a)), str(int(a == b))]
    return dict(id=rid, kind='format', vocabulary='spec', lang=lang, template=t, root=list(x), prompt=TP.format_prompt(lang, t, '|'.join(fields)),
                gold='|'.join(fields), schema=['D', 'P', 'P', 'binary'], field_length=2)


# --------------------------------------------------------------------- general
def load_general():
    bel = {}
    for lang, f in LANG_FILES['belebele'].items():
        bel[lang] = {(r['link'], r['question_number']): r for r in rows(RAW / 'belebele' / 'data' / f'{f}.jsonl')}
    keys = sorted(set(bel['en']) & set(bel['zh']) & set(bel['ru']))
    xq = {}
    for lang, f in LANG_FILES['xquad'].items():
        xq[lang] = {r['id']: r for r in rows(RAW / 'xquad' / f'xquad.{f}.jsonl')}
    ids = sorted(set(xq['en']) & set(xq['zh']) & set(xq['ru']))
    return bel, keys, xq, ids


def window(context, start, length, w, lang):
    lo, hi = max(0, start - w), min(len(context), start + length + w)
    seg = context[lo:hi]
    if lang != 'zh':
        if lo > 0:
            m = re.search(r'\s', seg)
            if m and m.start() < 60:
                seg = seg[m.start() + 1:]
        if hi < len(context):
            k = seg.rfind(' ')
            if k > len(seg) - 60:
                seg = seg[:k]
    return seg.strip()


def belebele_row(bel, key, lang, rid, split):
    r = bel[lang][key]
    return dict(id=rid, kind='belebele', vocabulary='general', lang=lang, template=0, item=list(key), split=split,
                prompt=TP.belebele_prompt(lang, r), gold='ABCD'[int(r['correct_answer_num']) - 1], schema=['letter'], field_length=1)


def xquad_row(xq, iid, lang, rid, split):
    r = xq[lang][iid]
    ans = r['answers']['text'][0]
    start = int(r['answers']['answer_start'][0])
    ctx = window(r['context'], start, len(ans), XQUAD_WINDOW[lang], lang)
    assert ans in ctx, (iid, lang)
    return dict(id=rid, kind='xquad', vocabulary='general', lang=lang, template=0, item=iid, split=split,
                prompt=TP.xquad_prompt(lang, ctx, r['question']), gold=ans.strip(), schema=['text'], field_length=1)


def pad_pair(rows_a, rows_b, tok):
    for a, b in zip(rows_a, rows_b):
        pa = a['prompt'] + TP.PAD_PHRASE[a['lang']]
        pb = b['prompt'] + TP.PAD_PHRASE[b['lang']]
        target = max(len(tok.encode(pa).ids), len(tok.encode(pb).ids)) + 4
        for row, text in ((a, pa), (b, pb)):
            n = len(tok.encode(text).ids)
            row['prompt'] = text + ' ~' * (target - n)
            row['paired_prompt_tokens'] = target
            assert len(tok.encode(row['prompt']).ids) == target


# ----------------------------------------------------------------------- world
def make_world(world, tok, bel, bkeys, xq, xids):
    rng = random.Random(900000 + world)
    book = old.book(world + 13001, 200)
    pairs = [(a, b) for a in range(32) for b in range(32)]
    rng.shuffle(pairs)
    split = dict(structure=pairs[:32], independent=pairs[32:64], train=pairs[64:])
    probe_struct = split['structure'][:PROBE_ROOTS]
    probe_ind = split['independent'][:PROBE_ROOTS]
    # specialised English stream: fixed per-step mix, templates 0/1/2 cycling
    spec = []
    for s in range(STEPS):
        block = []
        for j in range(SPEC_MIX['vertex']):
            x = (rng.randrange(32),) if j % 2 == 0 else rng.choice(split['train'])
            block.append(vertex(book, x, rng.choice('FDPR'), 'en', rng.choice((0, 1, 2)), f'spec_v_{s}_{j}', 'train'))
        for j in range(SPEC_MIX['edge']):
            x = (rng.randrange(32),) if j % 2 == 0 else rng.choice(split['train'])
            block.append(edge(book, x, rng.choice(EDGES), 'en', rng.choice((0, 1, 2)), f'spec_e_{s}_{j}', 'train'))
        for j in range(SPEC_MIX['joint']):
            x, z = rng.sample(split['train'], 2)
            block.append(joint(book, x, z, 'AB'[j % 2], 'en', rng.choice((0, 1, 2)), f'spec_j_{s}_{j}', bool((j // 2) % 2), 'train'))
        for j in range(SPEC_MIX['repair']):
            x, z, bad = rng.sample(split['train'], 3)
            block.append(repair(book, x, z, bad, 'AB'[j % 2], 'en', rng.choice((0, 1, 2)), f'spec_r_{s}_{j}', bool((j // 2) % 2), j % 3, 'train'))
        rng.shuffle(block)
        spec.extend(block)
    # format copy, three languages, shared
    fm = [fmt(book, rng.choice(split['train']), TP.TRAIN_LANGS[i % 3], rng.choice((0, 1)) if TP.TRAIN_LANGS[i % 3] != 'en' else rng.choice((0, 1, 2)), f'fmt_{i}')
          for i in range(STEPS * FMT_PER_STEP)]
    # general: held-out items first, then MULTI/MONO streams over the same item order
    bk = bkeys.copy()
    rng.shuffle(bk)
    xi = xids.copy()
    rng.shuffle(xi)
    held_b, train_b = bk[:GEN_HELD['belebele']], bk[GEN_HELD['belebele']:]
    held_x, train_x = xi[:GEN_HELD['xquad']], xi[GEN_HELD['xquad']:]
    multi, mono = [], []
    for i in range(STEPS * GEN_PER_STEP):
        lang = TP.TRAIN_LANGS[i % 3]
        if i % 2 == 0:
            key = train_b[(i // 2) % len(train_b)]
            multi.append(belebele_row(bel, key, lang, f'gen_b_{i}', 'train'))
            mono.append(belebele_row(bel, key, 'en', f'gen_b_{i}', 'train'))
        else:
            iid = train_x[(i // 2) % len(train_x)]
            multi.append(xquad_row(xq, iid, lang, f'gen_x_{i}', 'train'))
            mono.append(xquad_row(xq, iid, 'en', f'gen_x_{i}', 'train'))
    pad_pair(multi, mono, tok)
    # probes: specialised in four languages on held-out roots; general held-out in three
    probes = []
    for lang in TP.LANGS:
        for t in TP.PROBE_TEMPLATES[lang]:
            for j, x in enumerate(probe_struct):
                for c in 'FDPR':
                    probes.append(vertex(book, x, c, lang, t, f'probe_{lang}_{t}_v{c}_{j}', 'structure'))
                for c in EDGES:
                    probes.append(edge(book, x, c, lang, t, f'probe_{lang}_{t}_{c}_{j}', 'structure'))
                z = probe_struct[(j + 1) % PROBE_ROOTS]
                for d in 'AB':
                    for over in (False, True):
                        probes.append(joint(book, x, z, d, lang, t, f'probe_{lang}_{t}_f{d}_{j}_{int(over)}', over, 'structure'))
            if t == TP.PROBE_TEMPLATES[lang][0]:
                for j, x in enumerate(probe_ind):
                    z = probe_ind[(j + 1) % PROBE_ROOTS]
                    bad = probe_ind[(j + 2) % PROBE_ROOTS]
                    for d in 'AB':
                        for err in (0, 1, 2):
                            probes.append(repair(book, x, z, bad, d, lang, t, f'probe_{lang}_{t}_rep{d}_{j}_{err}', False, err, 'independent'))
    for lang in TP.TRAIN_LANGS:
        for k, key in enumerate(held_b):
            probes.append(belebele_row(bel, key, lang, f'probe_{lang}_b_{k}', 'held'))
        for k, iid in enumerate(held_x):
            probes.append(xquad_row(xq, iid, lang, f'probe_{lang}_x_{k}', 'held'))
    # hygiene
    held = set(split['structure']) | set(split['independent'])
    for r in spec + fm:
        if len(r['root']) == 2:
            assert tuple(r['root']) not in held, r['id']
    assert not ({tuple(r['item']) for r in multi if r['kind'] == 'belebele'} & set(map(tuple, held_b)))
    assert not ({r['item'] for r in multi if r['kind'] == 'xquad'} & set(held_x))
    assert all(Z.score(r, r['gold'])['correct'] for r in spec + fm + multi + mono + probes)
    assert all(a['kind'] == b['kind'] and a['item'] == b['item'] and a['paired_prompt_tokens'] == b['paired_prompt_tokens'] for a, b in zip(multi, mono))
    d = OUT / f'world{world}'
    hashes = {name: jwrite(d / name, rs) for name, rs in [('SPEC_TRAIN.jsonl', spec), ('FMT_TRAIN.jsonl', fm), ('GEN_MULTI.jsonl', multi),
                                                           ('GEN_MONO.jsonl', mono), ('PROBES.jsonl', probes)]}
    man = dict(version='CROSSLINGUAL_20260922', world=world, codebook=book, split=split, probe_structure=probe_struct, probe_independent=probe_ind,
               general_held=dict(belebele=held_b, xquad=held_x), general_train_counts=dict(belebele=len(train_b), xquad=len(train_x)),
               steps=STEPS, per_step=dict(spec=SPEC_PER_STEP, fmt=FMT_PER_STEP, gen=GEN_PER_STEP), spec_mix=SPEC_MIX,
               languages=dict(train_specialised=['en'], train_general_MULTI=list(TP.TRAIN_LANGS), train_general_MONO=['en'], train_format=list(TP.TRAIN_LANGS),
                              complex=['en', 'zh'], independent_transfer=['ru', 'ja']),
               probe_counts=dict(collections.Counter((r['vocabulary'], r['lang'], r['kind']) for r in probes)),
               hashes=hashes, seeds=[1, 2], checkpoint_every=16, base_seed=900000 + world,
               scope='Development worlds; conditions MULTI/MONO differ only in the language of the general layer')
    man['probe_counts'] = {f'{a}:{b}:{c}': v for (a, b, c), v in man['probe_counts'].items()}
    (d / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding='utf8')
    return man


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT.parent / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15' / 'tokenizer.json'))
    bel, bkeys, xq, xids = load_general()
    mans = [make_world(w, tok, bel, bkeys, xq, xids) for w in WORLDS]
    (OUT / 'CROSSLINGUAL_MANIFEST.json').write_text(json.dumps(dict(status='PREPARED_LOCAL_NOT_TRAINED', version='CROSSLINGUAL_20260922', worlds=list(WORLDS),
                                                                    raw_manifest_sha256=sha(RAW / 'RAW_MANIFEST.json'), manifests=mans), ensure_ascii=False, indent=2), encoding='utf8')
    for m in mans:
        print(m['world'], {k: v[:10] for k, v in m['hashes'].items()}, sum(int(v) for v in m['probe_counts'].values()), 'probes')


if __name__ == '__main__':
    main()
