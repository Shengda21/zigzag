"""Fixed-order queue for the cross-lingual arms. Seed 1 of every arm runs
before any seed 2, so a shortened rental still yields a complete one-seed set:

  501 MULTI1, 501 MONO1, 502 MONO1, 502 MULTI1, 501 MULTI2, 501 MONO2, 502 MONO2, 502 MULTI2
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
ORDER = [(501, 'MULTI1'), (501, 'MONO1'), (502, 'MONO1'), (502, 'MULTI1'), (501, 'MULTI2'), (501, 'MONO2'), (502, 'MONO2'), (502, 'MULTI2')]


def main(extra):
    if os.name != 'nt':
        import fcntl
        lock = open(R / 'GPU_QUEUE.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (R / 'logs').mkdir(exist_ok=True)
    for w, arm in ORDER:
        out = R / 'runs' / f'world{w}_{arm}'
        if (out / 'COMPLETE.json').exists():
            continue
        if (R / 'STOP_COMPUTE').exists():
            raise RuntimeError('STOP_COMPUTE present')
        with (R / 'logs' / f'{w}_{arm}.log').open('a', encoding='utf8') as log:
            p = subprocess.Popen([PY, '-u', str(HERE / 'run_crosslingual.py'), '--world', str(w), '--arm', arm] + extra, cwd=R,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false'))
            (R / 'RUNNING.json').write_text(json.dumps(dict(world=w, arm=arm, pid=p.pid, time=time.time())))
            code = p.wait()
        if code != 0 or not (out / 'COMPLETE.json').exists():
            if (R / 'STOP_COMPUTE').exists():
                raise RuntimeError('stopped by STOP_COMPUTE')
            raise RuntimeError(f'{w}/{arm} failed with exit {code}')
    (R / 'QUEUE_COMPLETE.json').write_text(json.dumps(dict(time=time.time())))


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Exception:
        (R / 'QUEUE_ERROR.json').write_text(traceback.format_exc())
        raise
