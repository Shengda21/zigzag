"""Fixed-order queue: seed 1 of every world first, then seed 2.
  601 S1, 602 S1, 603 S1, 604 S1, 601 S2, 602 S2, 603 S2, 604 S2
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
SPEC = os.environ.get('ZZ_SPEC', 'RCN12')
WORLDS = [int(w) for w in os.environ.get('ZZ_WORLDS', '601,602,603,604').split(',')]   # RCN12_FORGET uses ZZ_SPEC=RCN12_FORGET ZZ_WORLDS=611,612
ORDER = [(w, 1) for w in WORLDS] + [(w, 2) for w in WORLDS]
TAG = SPEC + '_' + '_'.join(str(w) for w in WORLDS)


def main(extra):
    if os.name != 'nt':
        import fcntl
        lock = open(R / f'GPU_QUEUE_{TAG}.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (R / 'logs').mkdir(exist_ok=True)
    for w, seed in ORDER:
        out = R / 'runs' / SPEC / f'world{w}_S{seed}'
        if (out / 'COMPLETE.json').exists():
            continue
        if (R / 'STOP_COMPUTE').exists():
            raise RuntimeError('STOP_COMPUTE present')
        with (R / 'logs' / f'{SPEC}_{w}_S{seed}.log').open('a', encoding='utf8') as log:
            p = subprocess.Popen([PY, '-u', str(HERE / 'run_complex.py'), '--spec', SPEC, '--world', str(w), '--seed', str(seed)] + extra, cwd=R,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false'))
            (R / f'RUNNING_{TAG}.json').write_text(json.dumps(dict(spec=SPEC, world=w, seed=seed, pid=p.pid, time=time.time())))
            code = p.wait()
        if code != 0 or not (out / 'COMPLETE.json').exists():
            if (R / 'STOP_COMPUTE').exists():
                raise RuntimeError('stopped by STOP_COMPUTE')
            raise RuntimeError(f'{w}/S{seed} failed with exit {code}')
    (R / f'QUEUE_COMPLETE_{TAG}.json').write_text(json.dumps(dict(spec=SPEC, worlds=WORLDS, time=time.time())))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception:
        (R / f'QUEUE_ERROR_{TAG}.json').write_text(traceback.format_exc())
        raise
