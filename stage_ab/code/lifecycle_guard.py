"""Finite rental guard for the Stage A/B remote root only (copied from the v5
guard; persistent data is never deleted).

Reads BUDGET.json (written by deploy_stage_ab.py): billing_start_unix,
compute_stop_unix, hard_stop_unix, rate_cny_per_hour. Writes STOP_COMPUTE at
compute_stop so runners checkpoint and exit; requests a platform shutdown at
hard_stop, or earlier once ALL_WORK_COMPLETE.json and LOCAL_RETURN_VERIFIED.json
both exist (the latter is uploaded by collect_stage_ab.py after local checks).
"""
import fcntl
import json
import os
import signal
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def dump(name, x):
    p = R / name
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(x, indent=2))
    q.replace(p)


def shutdown(reason):
    candidates = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().split(b'\0')
            if cmd[:4] == [b'/bin/supervisord', b'-c', b'/init/supervisor/supervisor.ini', b'']:
                candidates.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    dump('SHUTDOWN_REQUEST.json', dict(time=time.time(), reason=reason, supervisord_pid=candidates,
                                       local_verified=(R / 'LOCAL_RETURN_VERIFIED.json').exists()))
    if len(candidates) == 1:
        os.kill(candidates[0], signal.SIGTERM)
    else:
        os.system('shutdown -h now')


def main():
    lock = open(R / 'lifecycle_guard.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        b = json.loads((R / 'BUDGET.json').read_text())
        now = time.time()
        dump('LIFECYCLE_STATUS.json', dict(time=now, session_estimated_cny=(now - b['billing_start_unix']) * b['rate_cny_per_hour'] / 3600,
                                           compute_stop=b['compute_stop_unix'], hard_stop=b['hard_stop_unix']))
        if (R / 'ALL_WORK_COMPLETE.json').exists() and (R / 'LOCAL_RETURN_VERIFIED.json').exists():
            shutdown('All authorized work completed and local return verified')
            return
        if now >= b['compute_stop_unix'] and not (R / 'STOP_COMPUTE').exists():
            (R / 'STOP_COMPUTE').write_text('Budget reserve for transfer\n')
        if now >= b['hard_stop_unix']:
            shutdown('Finite rental hard cap; data retained on persistent disk')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
