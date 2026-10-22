"""COMMON atomic-edge foundation for worlds 401/402 plus a compact ATOMIC_EVAL.

The 20260921 package prepared format / DEPENDENT / INDEPENDENT / target-eval
files but no stage that teaches the target and auxiliary codebook edges, so the
target evaluation would have been at chance. Codebooks and root splits are read
from the existing MANIFEST.json; the frozen files are left byte-identical and
their hashes are re-verified here.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]            # zigzag_v5
DATA = HERE.parent / 'data'
sys.path.insert(0, str(ROOT / 'reproduction'))
sys.path.insert(0, str(HERE))
import make_d1 as old             # noqa: E402
import zz_score as Z              # noqa: E402

old.SUFFIX[3] = '依字段先后作答，使用竖线和逗号分隔，不添加解释。'   # identical to prepare.py TEMPLATES[3]
COMMON_STEPS = 512
BATCH = 64
REPLAY_PER_STEP = 16
TRAIN_TEMPLATES = (0, 1, 2)
EVAL_TEMPLATES = (0, 1, 3)
EDGES = old.CELLS[4:9]            # eFD eDP eFP eFR eRP
VERTICES = old.CELLS[:4]          # vF vD vP vR (digit read-out, keeps code reading stable)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(world: int) -> dict:
    d = DATA / f'world{world}'
    man = json.loads((d / 'MANIFEST.json').read_text(encoding='utf8'))
    frozen = {k: v for k, v in man['hashes'].items() if k not in ('COMMON.jsonl', 'ATOMIC_EVAL.jsonl')}
    for name, h in frozen.items():
        assert sha(d / name) == h, f'frozen file changed: {name}'
    if 'COMMON.jsonl' in man['hashes'] and (d / 'COMMON.jsonl').exists():
        return man
    target = {k: list(v) for k, v in man['target_codebook'].items()}
    aux = {k: list(v) for k, v in man['auxiliary_codebook'].items()}
    tsplit = {k: [tuple(x) for x in v] for k, v in man['target_split'].items()}
    asplit = {k: [tuple(x) for x in v] for k, v in man['auxiliary_split'].items()}
    rng = random.Random(world + 830000)
    train = []
    for i in range(COMMON_STEPS * BATCH):
        isaux = i % 2 == 1
        pool = (asplit if isaux else tsplit)['train']
        book = aux if isaux else target
        x = (rng.randrange(32),) if i % 4 < 2 else rng.choice(pool)
        cell = rng.choice(old.CELLS[:9])
        train.append(old.base(book, x, cell, rng.choice(TRAIN_TEMPLATES), f'common_{i}', 'aux' if isaux else 'target'))
    ev = []
    for j, x in enumerate(tsplit['structure']):
        for t in EVAL_TEMPLATES:
            for c in EDGES:
                ev.append(old.base(target, x, c, t, f'atomic_target_{j}_{c}_{t}', 'target'))
    for j, x in enumerate(asplit['structure']):
        for c in EDGES:
            ev.append(old.base(aux, x, c, 0, f'atomic_aux_{j}_{c}_0', 'aux'))
    held_t = set(tsplit['structure'] + tsplit['independent'])
    held_a = set(asplit['structure'] + asplit['independent'])
    for r in train:
        if len(r['root']) == 2:
            assert tuple(r['root']) not in (held_t if r['vocabulary'] == 'target' else held_a)
    assert all(old.score(r, r['gold'])['correct'] and Z.score(r, r['gold'])['correct'] for r in train + ev)
    assert len(train) == COMMON_STEPS * BATCH and len(ev) == 32 * 3 * 5 + 32 * 5
    hashes = dict(man['hashes'])
    hashes['COMMON.jsonl'] = old.jwrite(d / 'COMMON.jsonl', train)
    hashes['ATOMIC_EVAL.jsonl'] = old.jwrite(d / 'ATOMIC_EVAL.jsonl', ev)
    man.update(version='STAGE_AB_20260922', version_20260921='STAGE_AB_20260921', hashes=hashes,
               common_steps=COMMON_STEPS, common_extension_limit=1024, common_batch=BATCH,
               format_batch=BATCH, development_batch_new=BATCH, development_replay_per_step=REPLAY_PER_STEP,
               development_seeds=[1, 2], checkpoint_every=16, atomic_eval_records=len(ev),
               train_templates=list(TRAIN_TEMPLATES), eval_templates=list(EVAL_TEMPLATES),
               common_seed=world + 830000)
    old.dump(d / 'MANIFEST.json', man)
    return man


def main():
    mans = [build(w) for w in (401, 402)]
    top = DATA / 'STAGE_AB_MANIFEST.json'
    old.dump(top, dict(status='PREPARED_LOCAL_NOT_TRAINED', version='STAGE_AB_20260922', worlds=[401, 402],
                       manifests=mans, stage_COMMON='COMMON', stage_A='FORMAT', stage_B=['DEPENDENT', 'INDEPENDENT']))
    print(json.dumps({m['world']: {k: m['hashes'][k][:12] for k in ('COMMON.jsonl', 'ATOMIC_EVAL.jsonl')} for m in mans}))


if __name__ == '__main__':
    main()
