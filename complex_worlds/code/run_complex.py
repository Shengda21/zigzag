"""Runner for spec-generated worlds (one arm = one world x one training seed).

The training stream is pre-built by spec_world.py in curriculum order
(TRAIN_STREAM.jsonl, 512 steps x 64); the seed only reshuffles the order inside
each step and drives LoRA dropout/initialisation. Probes (every declared cell,
transfer chains, repairs) are evaluated every 16 steps and the adapter saved
every 16. Reuses stage_ab/code/zz_train.py and zz_score.py unchanged.
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
import zz_score as Z          # noqa: E402
import zz_train as T          # noqa: E402

CFG = dict(micro=32, eval_every=16, save_every=16, max_len=512)


def model_path(arg):
    if arg:
        return arg
    cfg = R / 'REMOTE_CONFIG.json'
    if cfg.exists():
        return json.loads(cfg.read_text(encoding='utf8'))['model_path']
    return os.environ['MODEL_PATH']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', default='RCN12')
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--seed', type=int, required=True, choices=(1, 2))
    p.add_argument('--model')
    p.add_argument('--device', default='cuda')
    p.add_argument('--runs', type=Path)
    p.add_argument('--eval-batch', type=int, default=192)
    p.add_argument('--dryrun', type=int, default=0)
    p.add_argument('--micro', type=int, default=0)
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=0, help='override the 16-step evaluation grid (time-scaled specs)')
    p.add_argument('--format', choices=('chat', 'raw'), default='chat', help="'raw' = continued-pretraining mode: base model, documents, loss on all tokens, few-shot probes")
    p.add_argument('--fewshot', type=int, default=3, help='raw mode: demonstrations of the same kind prepended to every probe')
    a = p.parse_args()
    data = R / 'data' / a.spec / f'world{a.world}'
    runs = a.runs or (R / 'runs' / a.spec)
    out = runs / (f'world{a.world}_S{a.seed}' + ('_raw' if a.format == 'raw' else ''))
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', out.name)
        return
    man = json.loads((data / 'MANIFEST.json').read_text(encoding='utf8'))
    for name, h in man['hashes'].items():
        assert T.sha(data / name) == h, name
    batch = man['batch']
    steps = man['steps']
    micro = CFG['micro']
    eval_every, save_every = CFG['eval_every'], CFG['save_every']
    if a.dryrun:
        steps = min(a.dryrun, steps)
        batch, micro, eval_every, save_every = 4, 2, 2, 2
    else:
        micro = a.micro or micro
        save_every = a.save_every or save_every
        eval_every = a.eval_every or eval_every
    seed_value = a.world * 100 + a.seed
    T.seed(seed_value)
    tok, model = T.load(model_path(a.model), a.device)
    T.seed(seed_value)
    stream = T.rows(data / 'TRAIN_STREAM.jsonl')
    rng = random.Random(seed_value)
    train = []
    full_batch = man['batch']
    for s in range(steps):
        block = stream[s * full_batch:(s + 1) * full_batch]
        rng.shuffle(block)
        train.extend(block[:batch])
    enc = T.encoded(tok, train, max_len=CFG['max_len'] if a.format == 'chat' else 640, fmt=a.format)
    demos_for = None
    if a.format == 'raw':
        # few-shot demonstrations: k training documents of the same kind (never the same held-out item), fixed per probe id
        by_kind = {}
        for r in stream:
            by_kind.setdefault(r['kind'], []).append(r)
        by_kind.setdefault('route', by_kind.get('edge', []))          # route probes share the edge document format
        by_kind.setdefault('vertex', by_kind.get('vertex', []))

        def demos_for(r):
            pool = by_kind.get(r['kind']) or by_kind['edge']
            rng_d = random.Random(hashlib.sha256((str(seed_value) + r['id']).encode()).hexdigest())
            return ''.join(T.raw_doc(d) for d in rng_d.sample(pool, min(a.fewshot, len(pool))))
    ev = T.rows(data / 'PROBES.jsonl')
    if a.dryrun:
        keep = {}
        for r in ev:
            keep.setdefault((r['kind'], r['task'], r.get('override')), r)
        ev = list(keep.values())[:80]
    T.dump(out / 'run_manifest.json', dict(spec=a.spec, world=a.world, seed=seed_value, steps=steps, batch=batch, microbatch=micro, eval_every=eval_every, format=a.format, fewshot=a.fewshot if a.format == 'raw' else None,
                                           save_every=save_every, data_hashes=man['hashes'], stream_order_sha256=hashlib.sha256(json.dumps([r['id'] for r in train]).encode()).hexdigest(),
                                           code_sha256=dict(run=T.sha(__file__), train=T.sha(R.parent / 'stage_ab' / 'code' / 'zz_train.py'), score=T.sha(R.parent / 'stage_ab' / 'code' / 'zz_score.py')),
                                           train_records=len(enc), eval_records=len(ev), dryrun=a.dryrun, started=time.time(),
                                           token_range=[min(len(x[0]) for x in enc), max(len(x[0]) for x in enc)]))
    final, stopped = T.train_loop(model, tok, out, enc, steps, batch, micro, a.device, eval_rows=ev, eval_every=eval_every, save_every=save_every,
                                  stop_flag=R / 'STOP_COMPUTE', seed_value=seed_value, eval_batch=a.eval_batch, keep_resume=bool(a.dryrun),
                                  fmt=a.format, demos_for=demos_for)
    if stopped:
        return
    summ = final['scores']
    cx = json.loads((data / 'COMPLEX.json').read_text(encoding='utf8'))
    taught3 = {e: summ.get(f'rcn:edge:task{e}:template3', {}).get('accuracy', 0.0) for e in cx['taught_edges']}
    control = {e: summ.get(f'rcn:edge:task{e}', {}).get('accuracy', 0.0) for e in cx['control_cells']}
    latent = {c: summ.get(f'rcn:edge:task{c}', summ.get(f'rcn:joint:task{c}', {})).get('accuracy', 0.0) for c in cx['latent_cells']}
    gate = dict(passed=min(taught3.values()) >= 0.9 and max(control.values(), default=0.0) <= 0.1, taught_edges_template3=taught3,
                control_edges=control, latent_cells_final=latent, thresholds=dict(taught_min=0.9, control_max=0.1))
    T.dump(out / f'GATE_{steps}.json', gate)
    T.dump(out / 'COMPLETE.json', dict(spec=a.spec, world=a.world, seed=seed_value, steps=steps, gate=gate,
                                       adapter_sha256=T.sha(out / 'final/adapter_model.safetensors'), finished=time.time(), dryrun=a.dryrun))
    print('COMPLETE', a.world, f'S{a.seed}', 'gate=', gate['passed'], flush=True)


if __name__ == '__main__':
    main()
