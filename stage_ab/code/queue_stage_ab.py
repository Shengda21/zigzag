"""GPU-exclusive queue for Stage A/B. Order is fixed before any result is seen:

  world401 COMMON [COMMON_EXT] FORMAT -> world402 COMMON [COMMON_EXT] FORMAT
  world401 DEP1 IND1 DEP2 IND2        -> world402 IND1 DEP1 IND2 DEP2

A failed COMMON gate (after the single pre-planned extension) or a failed
FORMAT (Stage A) gate skips Stage B for that world and is recorded in
QUEUE_SKIPPED.json; nothing is retried or tuned. STOP_COMPUTE halts cleanly.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
PY = sys.executable
WORLDS = [int(w) for w in os.environ.get('ZZ_WORLDS', '401,402').split(',')]
TAG = '_'.join(str(w) for w in WORLDS)
B_ORDER = {401: ['DEP1', 'IND1', 'DEP2', 'IND2'], 402: ['IND1', 'DEP1', 'IND2', 'DEP2']}


def complete(w, stage):
    p = R / 'runs' / f'world{w}_{stage}' / 'COMPLETE.json'
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else None


def run(w, stage, extra):
    if complete(w, stage):
        return complete(w, stage)
    if (R / 'STOP_COMPUTE').exists():
        raise RuntimeError('STOP_COMPUTE present')
    (R / 'logs').mkdir(exist_ok=True)
    with (R / 'logs' / f'{w}_{stage}.log').open('a', encoding='utf8') as log:
        cmd = [PY, '-u', str(HERE / 'run_stage_ab.py'), '--world', str(w), '--stage', stage] + extra
        p = subprocess.Popen(cmd, cwd=R, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false'))
        (R / f'RUNNING_{TAG}.json').write_text(json.dumps(dict(world=w, stage=stage, pid=p.pid, time=time.time())))
        code = p.wait()
    c = complete(w, stage)
    if code != 0 or c is None:
        if (R / 'STOP_COMPUTE').exists():
            raise RuntimeError('stopped by STOP_COMPUTE')
        raise RuntimeError(f'{w}/{stage} failed with exit {code}')
    return c


def main(extra):
    lock = None
    if os.name != 'nt':
        import fcntl
        lock = open(R / f'GPU_QUEUE_{TAG}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    skipped = {}
    for w in WORLDS:
        c = run(w, 'COMMON', extra)
        if not c['gate']['passed'] and '--ignore-gates' not in extra:
            c = run(w, 'COMMON_EXT', extra)
            if not c['gate']['passed']:
                skipped[w] = 'COMMON gate failed after pre-planned extension'
                continue
        f = run(w, 'FORMAT', extra)
        if not f['gate']['passed'] and '--ignore-gates' not in extra:
            skipped[w] = 'Stage A format gate failed'
    (R / f'STAGE_A_COMPLETE_{TAG}.json').write_text(json.dumps(dict(time=time.time(), skipped=skipped)))
    for w in WORLDS:
        if w in skipped:
            continue
        for arm in B_ORDER[w]:
            run(w, arm, extra)
    (R / f'QUEUE_SKIPPED_{TAG}.json').write_text(json.dumps(skipped))
    (R / f'QUEUE_COMPLETE_{TAG}.json').write_text(json.dumps(dict(time=time.time(), skipped=skipped)))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception:
        (R / f'QUEUE_ERROR_{TAG}.json').write_text(traceback.format_exc())
        raise
