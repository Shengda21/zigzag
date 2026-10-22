"""Round 4 (2026-09-24, same H800 box): FORGET3 second confirmation on fresh
worlds 635/636/637 x seeds 1,2 (SFT), time-scaled continued-pretraining arms
(RCN12_FORGET3_X2, worlds 635/636, base model, raw few-shot), and Pythia
public checkpoints (inference only). Pre-registration:
complex_worlds/PREREGISTRATION_FORGET3_CONFIRM2_20260924.md and
public_ckpt/PREREGISTRATION_PYTHIA_20260924.md.

  python deploy_round4.py --upload
  python deploy_round4.py --smoke
  python deploy_round4.py --start --hours 36
  python deploy_round4.py --pythia            (after the final-checkpoint capability check)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
sys.path.insert(0, str(HERE))
from cloud_remote2 import connect, endpoint, run   # noqa: E402

BASE = endpoint().get('base', '/root/zz2')
SCRATCH = Path(os.environ.get('ZZ_SCRATCH', Path(os.environ.get('TEMP', '/tmp')) / 'zz_deploy'))
LINES = {
    'stage_ab_code': ['stage_ab/code'],
    'complex_worlds': ['complex_worlds/code', 'complex_worlds/specs', 'complex_worlds/data/RCN12_FORGET3', 'complex_worlds/data/RCN12_FORGET3_X2'],
    'public_ckpt': ['public_ckpt/code', 'public_ckpt/data', 'public_ckpt/spec'],
    'analysis_code': ['crosslingual/code'],
    'docs': ['complex_worlds/PREREGISTRATION_FORGET3_CONFIRM2_20260924.md', 'public_ckpt/PREREGISTRATION_PYTHIA_20260924.md'],
}
PY_ENV = "OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false"
Q = "cd {B}/complex_worlds && ZZ_SPEC=RCN12_FORGET3 ZZ_WORLDS=%d {py} -u code/queue_complex.py --micro 16 --save-every 256 --eval-batch 256"
X2 = ("cd {B}/complex_worlds && {py} -u code/run_complex.py --spec RCN12_FORGET3_X2 --world %d --seed 1 --format raw --fewshot 3 "
      "--model {B}/models/Qwen2.5-1.5B --micro 16 --save-every 512 --eval-every 32 --eval-batch 256")
LANES = {
    'f3c2_635': Q % 635,
    'f3c2_636': Q % 636,
    'f3c2_637': Q % 637,
    'x2_635': X2 % 635,
    'x2_636': X2 % 636,
}
SMOKE = [
    "cd {B}/complex_worlds && {py} -u code/run_complex.py --spec RCN12_FORGET3 --world 635 --seed 1 --dryrun 2 --runs {B}/complex_worlds/runs_smoke --eval-batch 64",
    ("cd {B}/complex_worlds && {py} -u code/run_complex.py --spec RCN12_FORGET3_X2 --world 635 --seed 1 --format raw --fewshot 3 "
     "--model {B}/models/Qwen2.5-1.5B --dryrun 2 --runs {B}/complex_worlds/runs_smoke --eval-batch 64"),
]
REVS = ['step%d' % s for s in list(range(1000, 11000, 1000)) + list(range(13000, 143000, 3000)) + [143000]]


def build_zips():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = {}
    for line, items in LINES.items():
        z = SCRATCH / f'r4_{line}.zip'
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for d in items:
                p = V5 / d
                files = [p] if p.is_file() else sorted(p.rglob('*'))
                for f in files:
                    rel = f.relative_to(V5)
                    if f.is_file() and '__pycache__' not in rel.parts and not f.name.endswith('.pyc') and 'runs' not in rel.parts:
                        zf.write(f, rel.as_posix())
        out[line] = z
        print(line, z.stat().st_size // 1024, 'KB')
    return out


def upload(c, e):
    zips = build_zips()
    py = e['remote_python']
    run(c, f'mkdir -p {BASE}/logs')
    with c.open_sftp() as s:
        for line, z in zips.items():
            s.put(str(z), f'{BASE}/{z.name}')
        s.put(str(HERE / 'zz_guard.py'), f'{BASE}/zz_guard.py')
    script = '\n'.join([
        f'cd {BASE} && for z in r4_*.zip; do {py} -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(\'.\')" $z && echo unzipped $z; done',
        'cat complex_worlds/REMOTE_CONFIG.json; echo',
        f'cd {BASE}/complex_worlds && {py} code/audit.py --spec RCN12_FORGET3 --quick && {py} code/audit.py --spec RCN12_FORGET3_X2 --quick',
        'sha256sum code/run_complex.py ../stage_ab/code/zz_train.py ../stage_ab/code/zz_score.py ../public_ckpt/code/run_public.py | cut -c1-12',
        f'df -h {BASE} | tail -1',
    ])
    code, out, err = run(c, script, timeout=900)
    print(out, err[-500:])
    assert code == 0 and 'HASHES_OK' in out, 'upload failed'


def smoke(c, e):
    py = e['remote_python']
    for t in SMOKE:
        cmd = t.format(B=BASE, py=py)
        code, out, err = run(c, f"export {PY_ENV}; {cmd} 2>&1 | tail -5", timeout=1800)
        print('----', cmd[:150])
        print(out)
        assert code == 0, 'smoke failed'
    run(c, f'rm -rf {BASE}/complex_worlds/runs_smoke')


def launch(c, e, lanes):
    py = e['remote_python']
    cmds = []
    for name, tpl in lanes.items():
        cmd = tpl.format(B=BASE, py=py)
        cmds.append(f"{PY_ENV} setsid nohup bash -c \"{cmd}\" > {BASE}/logs/lane_{name}.log 2>&1 < /dev/null &")
    cmds.append("sleep 5; ps -eo pid,etimes,args | grep -E 'queue_complex|run_complex|run_public|zz_guard' | grep -v grep")
    try:
        code, out, err = run(c, '\n'.join(cmds), timeout=60)
        print(out, err)
    except Exception as ex:      # noqa: BLE001
        print('launch command returned', repr(ex)[:120], '- verify with status')


def start(c, e, hours):
    py = e['remote_python']
    code, out, _ = run(c, "ps -eo pid,args | grep -E 'queue_complex|run_complex|zz_guard' | grep -v grep || true")
    assert not out.strip(), f'already running:\n{out}'
    now = time.time()
    budget = dict(rate_cny_per_hour=e.get('rate_cny_per_hour', 9.98), billing_start_unix=now, session_hours_limit=hours,
                  compute_stop_unix=now + hours * 3600 - 2400, hard_stop_unix=now + hours * 3600 - 600, source='deploy_round4.py --start',
                  note='round 4; no time cap, this limit is only a safety net')
    with c.open_sftp() as s:
        with s.open(f'{BASE}/BUDGET.json', 'w') as f:
            f.write(json.dumps(budget, indent=2))
    (HERE / 'BUDGET_SESSION_R4.json').write_text(json.dumps(budget, indent=2), encoding='utf8')
    run(c, f"cd {BASE} && (setsid nohup {py} -u zz_guard.py > logs/guard_r4.log 2>&1 < /dev/null &) ; sleep 1; echo ok", timeout=30)
    launch(c, e, LANES)


def pythia(c, e, model, revs, workers, batch):
    lanes = {}
    for i in range(workers):
        part = ','.join(revs[i::workers])
        lanes[f'pythia_{i}'] = ("cd {B}/public_ckpt/code && source /etc/network_turbo >/dev/null 2>&1; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ZZ_CURL_DL=1 ZZ_HF_TMP={B}/hf_tmp/w%d {py} -u run_public.py --model %s "
                                "--revisions %s --device cuda --batch_size %d --out {B}/public_ckpt/runs/%s") % (i, model, part, batch, model.split('/')[-1])
    launch(c, e, lanes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upload', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start', action='store_true')
    p.add_argument('--hours', type=float, default=36.0)
    p.add_argument('--pythia', action='store_true')
    p.add_argument('--model', default='EleutherAI/pythia-1.4b')
    p.add_argument('--revisions', default=','.join(REVS))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lane', action='append', default=[], help='relaunch only these LANES entries')
    a = p.parse_args()
    e = endpoint()
    c = connect()
    try:
        if a.upload:
            upload(c, e)
        if a.smoke:
            smoke(c, e)
        if a.start:
            start(c, e, a.hours)
        if a.lane:
            launch(c, e, {k: LANES[k] for k in a.lane})
        if a.pythia:
            pythia(c, e, a.model, a.revisions.split(','), a.workers, a.batch)
    finally:
        c.close()


if __name__ == '__main__':
    main()
