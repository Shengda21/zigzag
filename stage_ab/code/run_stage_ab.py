"""Stage A/B runner for worlds 401/402 (single H800, one stage per invocation).

Chain per world (queue_stage_ab.py orders them):
  COMMON      base model -> atomic edges of target+aux codebooks, 512 steps x 64,
              eval ATOMIC_EVAL every 128, gate: every target edge >= 0.9 on
              templates 0/1/3. Pre-planned single extension COMMON_EXT (+512)
              if the gate fails; no other retry.
  FORMAT      Stage A. From COMMON final; FORMAT_TRAIN 128 steps x 64; eval
              FORMAT_EVAL + ATOMIC_EVAL + EVAL_TARGET every 16 steps; gate =
              zz_score.stage_a_gate. Failure stops Stage B for that world.
  DEP{s}/IND{s}  Stage B, seeds s in {1,2}. From FORMAT final. Each step = 64
              new records (seeded permutation, identical across conditions)
              + 16 replay records from COMMON (identical across conditions).
              256 steps; eval every 16; adapter saved every 16; gate =
              zz_score.stage_b_gate judged per template / condition / domain.

`--dryrun N` shrinks every stage to N steps, tiny batches and a 24-row eval so
the whole code path (resume, gates, pairing files) can run on a CPU toy model.
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
sys.path.insert(0, str(HERE))
import zz_score as Z          # noqa: E402
import zz_train as T          # noqa: E402

STAGES = ('COMMON', 'COMMON_EXT', 'FORMAT', 'DEP1', 'DEP2', 'IND1', 'IND2')
CONFIG = dict(
    COMMON=dict(steps=512, batch=64, micro=32, eval_every=128, save_every=128, train='COMMON.jsonl', evals=['ATOMIC_EVAL.jsonl']),
    COMMON_EXT=dict(steps=512, batch=64, micro=32, eval_every=128, save_every=128, train='COMMON.jsonl', evals=['ATOMIC_EVAL.jsonl']),
    FORMAT=dict(steps=128, batch=64, micro=32, eval_every=16, save_every=16, train='FORMAT_TRAIN.jsonl', replay=16,
                evals=['FORMAT_EVAL.jsonl', 'ATOMIC_EVAL.jsonl', 'EVAL_TARGET.jsonl']),
    B=dict(steps=256, batch_new=64, replay=16, micro=20, eval_every=16, save_every=16,
           evals=['EVAL_TARGET.jsonl', 'ATOMIC_EVAL.jsonl', 'FORMAT_EVAL.jsonl']),
)


def model_path(arg):
    if arg:
        return arg
    cfg = R / 'REMOTE_CONFIG.json'
    if cfg.exists():
        return json.loads(cfg.read_text(encoding='utf8'))['model_path']
    return os.environ['MODEL_PATH']


def source_for(world, stage, runs):
    if stage == 'COMMON':
        return None
    if stage == 'COMMON_EXT':
        return runs / f'world{world}_COMMON'
    if stage == 'FORMAT':
        ext = runs / f'world{world}_COMMON_EXT'
        return ext if (ext / 'COMPLETE.json').exists() else runs / f'world{world}_COMMON'
    return runs / f'world{world}_FORMAT'


def require_gate(source, ignore):
    c = json.loads((source / 'COMPLETE.json').read_text(encoding='utf8'))
    if not ignore:
        assert c['gate']['passed'], f'{source.name}: gate failed, refusing to start next stage'
    return c


def b_stream(world, stage, data, man, rng_seed):
    """Seeded permutation of DEPENDENT/INDEPENDENT plus COMMON replay, with the
    permutation, replay indices and slot order shared by both conditions."""
    cond = 'DEPENDENT' if stage.startswith('DEP') else 'INDEPENDENT'
    new = T.rows(data / f'{cond}.jsonl')
    common = T.rows(data / 'COMMON.jsonl')
    cfg = CONFIG['B']
    rng = random.Random(rng_seed)
    perm = list(range(len(new)))
    rng.shuffle(perm)
    replay_idx = [rng.randrange(len(common)) for _ in range(cfg['steps'] * cfg['replay'])]
    slots = []
    for s in range(cfg['steps']):
        order = list(range(cfg['batch_new'] + cfg['replay']))
        rng.shuffle(order)
        slots.append(order)
    stream = []
    for s in range(cfg['steps']):
        block = [new[perm[s * cfg['batch_new'] + j]] for j in range(cfg['batch_new'])]
        block += [dict(common[replay_idx[s * cfg['replay'] + j]], id=f'replay_{s}_{j}') for j in range(cfg['replay'])]
        stream.extend(block[k] for k in slots[s])
    pairing = dict(condition=cond, seed=rng_seed, permutation_sha256=hashlib.sha256(json.dumps(perm).encode()).hexdigest(),
                   replay_sha256=hashlib.sha256(json.dumps(replay_idx).encode()).hexdigest(),
                   slots_sha256=hashlib.sha256(json.dumps(slots).encode()).hexdigest(),
                   new_per_step=cfg['batch_new'], replay_per_step=cfg['replay'], steps=cfg['steps'])
    return stream, pairing


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--model')
    p.add_argument('--device', default='cuda')
    p.add_argument('--data', type=Path)
    p.add_argument('--runs', type=Path)
    p.add_argument('--eval-batch', type=int, default=192)
    p.add_argument('--dryrun', type=int, default=0, help='shrink every stage to N steps (CPU toy model)')
    p.add_argument('--micro', type=int, default=0, help='execution override of the micro-batch (memory/throughput only)')
    p.add_argument('--save-every', type=int, default=0, help='execution override of adapter save interval (eval interval unchanged)')
    p.add_argument('--ignore-gates', action='store_true', help='dry-run only: continue the chain past failed gates')
    a = p.parse_args()
    data = a.data or (R / 'data' / f'world{a.world}')
    runs = a.runs or (R / 'runs')
    out = runs / f'world{a.world}_{a.stage}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', out.name)
        return
    man = json.loads((data / 'MANIFEST.json').read_text(encoding='utf8'))
    for name, h in man['hashes'].items():
        assert T.sha(data / name) == h, name
    stage = a.stage
    isb = stage[:3] in ('DEP', 'IND')
    cfg = dict(CONFIG['B'] if isb else CONFIG[stage])
    if isb:
        cfg['batch'] = cfg['batch_new'] + cfg['replay']
    steps = cfg['steps']
    if a.dryrun:
        steps = min(a.dryrun, steps)
        cfg.update(micro=2, eval_every=2, save_every=2)
        if isb:
            cfg.update(batch_new=3, replay=1, batch=4)
        else:
            cfg['batch'] = 4
        CONFIG['B'].update(steps=steps, batch_new=cfg.get('batch_new', 64), replay=cfg.get('replay', 16))
    else:
        if a.micro:
            cfg['micro'] = a.micro
        if a.save_every:
            cfg['save_every'] = a.save_every
    stage_seed = man['common_seed'] if stage.startswith('COMMON') else (a.world * 100 + (int(stage[-1]) if isb else 50))
    T.seed(stage_seed)
    tok, model = T.load(model_path(a.model), a.device)
    source = source_for(a.world, stage, runs)
    source_sha = None
    if source is not None:
        require_gate(source, a.ignore_gates)
        source_sha = T.load_adapter(model, source / 'final')
    T.seed(stage_seed)
    # ---- training stream
    pairing = None
    if isb:
        train, pairing = b_stream(a.world, stage, data, man, stage_seed)
        T.dump(out / 'PAIRING.json', pairing)
    elif stage == 'COMMON_EXT':
        train = T.rows(data / 'COMMON.jsonl')
        random.Random(stage_seed + 1).shuffle(train)
    elif stage == 'FORMAT':
        # EXECUTION_REPAIR_02 (2026-09-23): the first run trained 128 steps of pure format copy and every
        # target atomic edge fell to 0.0 by step 64 (catastrophic forgetting), so the pre-registered
        # atomic_edge_strict check failed in both worlds. Stage B already replays 16 COMMON records per
        # step; the same replay is applied here (48 format + 16 replay). No other change; no retry beyond this.
        fmt_rows = T.rows(data / cfg['train'])
        common = T.rows(data / 'COMMON.jsonl')
        replay = cfg['replay'] if not a.dryrun else 1
        per = cfg['batch'] - replay
        rng = random.Random(stage_seed + 2)
        replay_idx = [rng.randrange(len(common)) for _ in range(steps * replay)]
        train = []
        for s in range(steps):
            block = fmt_rows[s * per:(s + 1) * per] + [dict(common[replay_idx[s * replay + j]], id=f'replay_{s}_{j}') for j in range(replay)]
            rng.shuffle(block)
            train.extend(block)
        pairing = dict(repair='EXECUTION_REPAIR_02_format_replay', format_per_step=per, replay_per_step=replay,
                       replay_sha256=hashlib.sha256(json.dumps(replay_idx).encode()).hexdigest())
        T.dump(out / 'REPLAY.json', pairing)
    else:
        train = T.rows(data / cfg['train'])
    if a.dryrun:
        train = train[:steps * cfg['batch']]
    assert len(train) >= steps * cfg['batch'], (len(train), steps, cfg['batch'])
    enc = T.encoded(tok, train[:steps * cfg['batch']])
    # ---- evaluation rows
    ev = []
    for name in cfg['evals']:
        ev.extend(T.rows(data / name))
    if a.dryrun:
        keep = {}
        for r in ev:
            key = (r['vocabulary'], r['kind'], r.get('template'), r.get('condition'), r.get('domain'), r.get('task'))
            keep.setdefault(key, r)
        ev = list(keep.values())[:60]
    T.dump(out / 'run_manifest.json', dict(world=a.world, stage=stage, steps=steps, batch=cfg['batch'], microbatch=cfg['micro'],
                                           eval_every=cfg['eval_every'], save_every=cfg['save_every'], seed=stage_seed,
                                           source=str(source) if source else None, source_sha256=source_sha,
                                           data_hashes=man['hashes'], code_sha256=dict(run=T.sha(__file__), train=T.sha(HERE / 'zz_train.py'), score=T.sha(HERE / 'zz_score.py')),
                                           train_records=len(enc), eval_records=len(ev), dryrun=a.dryrun, started=time.time(),
                                           token_range=[min(len(x[0]) for x in enc), max(len(x[0]) for x in enc)], pairing=pairing))
    final, stopped = T.train_loop(model, tok, out, enc, steps, cfg['batch'], cfg['micro'], a.device, eval_rows=ev,
                                  eval_every=cfg['eval_every'], save_every=cfg['save_every'], stop_flag=R / 'STOP_COMPUTE',
                                  seed_value=stage_seed, eval_batch=a.eval_batch, keep_resume=bool(a.dryrun))
    if stopped:
        return
    summ = final['scores']
    if stage.startswith('COMMON'):
        gate = Z.atomic_gate(summ, 'target', (0, 1, 3), 0.9)
    elif stage == 'FORMAT':
        gate = Z.stage_a_gate(summ, summ)
    else:
        gate = Z.stage_b_gate(summ)
    T.dump(out / f'GATE_{steps}.json', gate)
    T.dump(out / 'COMPLETE.json', dict(world=a.world, stage=stage, steps=steps, gate=gate, source_sha256=source_sha,
                                       adapter_sha256=T.sha(out / 'final/adapter_model.safetensors'), finished=time.time(), dryrun=a.dryrun))
    print('COMPLETE', a.world, stage, 'gate_passed=', gate['passed'], flush=True)


if __name__ == '__main__':
    main()
