"""Cross-lingual arm runner (one arm per invocation; queue_crosslingual.py orders them).

Arm = {MULTI, MONO} x seed {1, 2}, from the base model, 512 steps. Each step
contains 32 English specialised records + 16 format-copy records (en/zh/ru) +
16 general records (MULTI: en/zh/ru cycling; MONO: the same items in English,
prompt-token paired). Slot order, specialised and format records are identical
between MULTI and MONO for the same seed, so the only difference is the
language of the general layer. Probes (en/zh/ru/ja specialised, en/zh/ru
general held-out) are evaluated every 16 steps and the adapter saved every 16.

Reuses stage_ab/code/zz_train.py and zz_score.py unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R.parent / 'stage_ab' / 'code'))
sys.path.insert(0, str(HERE))
import zz_score as Z          # noqa: E402
import zz_train as T          # noqa: E402

ARMS = ('MULTI1', 'MULTI2', 'MONO1', 'MONO2')
CFG = dict(steps=512, spec=32, fmt=16, gen=16, micro=16, eval_every=16, save_every=16, max_len=1024)


def model_path(arg):
    if arg:
        return arg
    cfg = R / 'REMOTE_CONFIG.json'
    if cfg.exists():
        return json.loads(cfg.read_text(encoding='utf8'))['model_path']
    return os.environ['MODEL_PATH']


def stream(data, cond, seed_value, steps, cfg):
    spec = T.rows(data / 'SPEC_TRAIN.jsonl')
    fm = T.rows(data / 'FMT_TRAIN.jsonl')
    gen = T.rows(data / f'GEN_{cond}.jsonl')
    rng = random.Random(seed_value)
    slots = []
    for s in range(steps):
        order = list(range(cfg['spec'] + cfg['fmt'] + cfg['gen']))
        rng.shuffle(order)
        slots.append(order)
    out = []
    for s in range(steps):
        block = spec[s * cfg['spec']:(s + 1) * cfg['spec']] + fm[s * cfg['fmt']:(s + 1) * cfg['fmt']] + gen[s * cfg['gen']:(s + 1) * cfg['gen']]
        assert len(block) == cfg['spec'] + cfg['fmt'] + cfg['gen'], s
        out.extend(block[k] for k in slots[s])
    pairing = dict(condition=cond, seed=seed_value, slots_sha256=hashlib.sha256(json.dumps(slots).encode()).hexdigest(), steps=steps,
                   per_step=dict(spec=cfg['spec'], fmt=cfg['fmt'], gen=cfg['gen']))
    return out, pairing


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--model')
    p.add_argument('--device', default='cuda')
    p.add_argument('--data', type=Path)
    p.add_argument('--runs', type=Path)
    p.add_argument('--eval-batch', type=int, default=128)
    p.add_argument('--dryrun', type=int, default=0)
    p.add_argument('--micro', type=int, default=0)
    p.add_argument('--save-every', type=int, default=0)
    a = p.parse_args()
    data = a.data or (R / 'data' / f'world{a.world}')
    runs = a.runs or (R / 'runs')
    out = runs / f'world{a.world}_{a.arm}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', out.name)
        return
    man = json.loads((data / 'MANIFEST.json').read_text(encoding='utf8'))
    for name, h in man['hashes'].items():
        assert T.sha(data / name) == h, name
    cfg = dict(CFG)
    cond, seed_idx = a.arm[:-1], int(a.arm[-1])
    steps = cfg['steps']
    if a.dryrun:
        steps = min(a.dryrun, steps)
        cfg.update(spec=2, fmt=1, gen=1, micro=2, eval_every=2, save_every=2)
    else:
        if a.micro:
            cfg['micro'] = a.micro
        if a.save_every:
            cfg['save_every'] = a.save_every
    batch = cfg['spec'] + cfg['fmt'] + cfg['gen']
    seed_value = a.world * 100 + seed_idx
    T.seed(seed_value)
    tok, model = T.load(model_path(a.model), a.device)
    T.seed(seed_value)
    train, pairing = stream(data, cond, seed_value, steps, cfg)
    T.dump(out / 'PAIRING.json', pairing)
    enc = T.encoded(tok, train, max_len=cfg['max_len'])
    ev = T.rows(data / 'PROBES.jsonl')
    if a.dryrun:
        keep = {}
        for r in ev:
            keep.setdefault((r['vocabulary'], r['lang'], r['kind'], r.get('task')), r)
        ev = list(keep.values())
    T.dump(out / 'run_manifest.json', dict(world=a.world, arm=a.arm, condition=cond, seed=seed_value, steps=steps, batch=batch, microbatch=cfg['micro'],
                                           eval_every=cfg['eval_every'], save_every=cfg['save_every'], data_hashes=man['hashes'],
                                           code_sha256=dict(run=T.sha(__file__), train=T.sha(R.parent / 'stage_ab' / 'code' / 'zz_train.py'), score=T.sha(R.parent / 'stage_ab' / 'code' / 'zz_score.py')),
                                           train_records=len(enc), eval_records=len(ev), dryrun=a.dryrun, started=time.time(),
                                           token_range=[min(len(x[0]) for x in enc), max(len(x[0]) for x in enc)], pairing=pairing))
    final, stopped = T.train_loop(model, tok, out, enc, steps, batch, cfg['micro'], a.device, eval_rows=ev, eval_every=cfg['eval_every'],
                                  save_every=cfg['save_every'], stop_flag=R / 'STOP_COMPUTE', seed_value=seed_value, eval_batch=a.eval_batch, keep_resume=bool(a.dryrun))
    if stopped:
        return
    summ = final['scores']
    # development gate (report only): English specialised layer learned on the held-out paraphrase
    en = {e: summ.get(f'spec:base:task{e}:lang{"en"}', {}).get('accuracy', 0.0) for e in Z.ATOMIC_EDGES}
    en3 = {e: summ.get(f'spec:base:task{e}:template3', {}).get('accuracy', 0.0) for e in Z.ATOMIC_EDGES}
    gate = dict(passed=min(en3.values()) >= 0.9, en_edges=en, en_edges_template3=en3, threshold=0.9,
                zh_edges={e: summ.get(f'spec:base:task{e}:langzh', {}).get('accuracy', 0.0) for e in Z.ATOMIC_EDGES},
                ru_edges={e: summ.get(f'spec:base:task{e}:langru', {}).get('accuracy', 0.0) for e in Z.ATOMIC_EDGES},
                ja_edges={e: summ.get(f'spec:base:task{e}:langja', {}).get('accuracy', 0.0) for e in Z.ATOMIC_EDGES},
                general={l: summ.get(f'general:belebele:lang{l}', {}).get('accuracy', 0.0) for l in ('en', 'zh', 'ru')})
    T.dump(out / f'GATE_{steps}.json', gate)
    T.dump(out / 'COMPLETE.json', dict(world=a.world, arm=a.arm, steps=steps, gate=gate, adapter_sha256=T.sha(out / 'final/adapter_model.safetensors'),
                                       finished=time.time(), dryrun=a.dryrun))
    print('COMPLETE', a.world, a.arm, 'en_gate=', gate['passed'], flush=True)


if __name__ == '__main__':
    main()
