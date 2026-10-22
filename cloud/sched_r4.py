"""Round-4 GPU scheduler (execution only; no scientific parameter is touched).
Runs the remaining LoRA arms with at most MAX_JOBS concurrent run_complex.py processes and
only when the GPU has at least MIN_FREE_MIB free, longest job first. Each job is skipped when its
COMPLETE.json exists; a failed job is retried up to 2 times (run_complex resumes from resume.pt)."""
import json
import os
import subprocess
import time
from pathlib import Path

B = Path('/root/zz2')
CW = B / 'complex_worlds'
PY = '/root/miniconda3/bin/python'
MAX_JOBS, MIN_FREE_MIB = 3, 22000
SFT = ['--micro', '16', '--save-every', '256', '--eval-batch', '256']
X2 = ['--format', 'raw', '--fewshot', '3', '--model', str(B / 'models/Qwen2.5-1.5B'), '--micro', '16', '--save-every', '512', '--eval-every', '32', '--eval-batch', '256']
JOBS = [
    ('x2_636', ['--spec', 'RCN12_FORGET3_X2', '--world', '636', '--seed', '1'] + X2, CW / 'runs/RCN12_FORGET3_X2/world636_S1_raw'),
    ('f3_636_S1', ['--spec', 'RCN12_FORGET3', '--world', '636', '--seed', '1'] + SFT, CW / 'runs/RCN12_FORGET3/world636_S1'),
    ('f3_635_S2', ['--spec', 'RCN12_FORGET3', '--world', '635', '--seed', '2'] + SFT, CW / 'runs/RCN12_FORGET3/world635_S2'),
    ('f3_637_S2', ['--spec', 'RCN12_FORGET3', '--world', '637', '--seed', '2'] + SFT, CW / 'runs/RCN12_FORGET3/world637_S2'),
    ('f3_636_S2', ['--spec', 'RCN12_FORGET3', '--world', '636', '--seed', '2'] + SFT, CW / 'runs/RCN12_FORGET3/world636_S2'),
]


def running():
    out = subprocess.run("ps -eo args | grep 'run_complex.py' | grep -v grep", shell=True, capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def free_mib():
    out = subprocess.run('nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits', shell=True, capture_output=True, text=True).stdout
    tot, used = [int(x) for x in out.split(',')]
    return tot - used


def log(msg):
    with open(B / 'logs/sched_r4.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def main():
    procs, tries = {}, {}
    while True:
        if (B / 'STOP_COMPUTE').exists() or (CW / 'STOP_COMPUTE').exists():
            log('STOP_COMPUTE present, scheduler exits')
            return
        for name, p in list(procs.items()):
            if p.poll() is not None:
                log(f'{name} exited {p.returncode}')
                del procs[name]
        live = running()

        def is_live(args):
            key = ' '.join(args[:6])          # --spec X --world N --seed S
            return any(key in l for l in live)
        pending = [j for j in JOBS if not (j[2] / 'COMPLETE.json').exists() and j[0] not in procs and not is_live(j[1]) and tries.get(j[0], 0) < 3]
        if not pending and not procs:
            (B / 'SCHED_R4_DONE.json').write_text(json.dumps(dict(time=time.time(), tries=tries)))
            log('all scheduled jobs finished')
            return
        if pending and len(running()) < MAX_JOBS and free_mib() >= MIN_FREE_MIB:
            name, args, out = pending[0]
            tries[name] = tries.get(name, 0) + 1
            env = dict(os.environ, OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            lf = open(B / f'logs/lane_{name}.log', 'a')
            procs[name] = subprocess.Popen([PY, '-u', 'code/run_complex.py'] + args, cwd=CW, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            log(f'start {name} try {tries[name]} pid {procs[name].pid} free_mib {free_mib()}')
            time.sleep(240)          # let the new job allocate before judging free memory again
        time.sleep(60)


if __name__ == '__main__':
    main()
