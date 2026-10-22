"""Local: upload the frozen Stage A/B package to the rented box and launch.

  python deploy_stage_ab.py --upload            # zip + code + bootstrap, run bootstrap
  python deploy_stage_ab.py --smoke             # 2-step GPU smoke test on the real model (~3 min), synchronous
  python deploy_stage_ab.py --start --hours 6 --rate 9.98   # write BUDGET.json, start guard + queue (nohup)

Nothing here reads gold answers or tunes anything; --start refuses to run if a
queue is already alive on the box.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R.parent / 'cloud'))
from cloud_remote2 import connect, endpoint, run   # noqa: E402

ZIP = R / 'STAGE_AB_INPUTS_20260922.zip'
CODE = ['zz_score.py', 'zz_train.py', 'run_stage_ab.py', 'queue_stage_ab.py', 'lifecycle_guard.py', 'audit2.py', 'bootstrap_remote.sh', 'prepare_common.py']


def upload(c, e):
    root = e['remote_root']
    assert ZIP.exists(), 'run make_package.py first'
    with c.open_sftp() as s:
        for d in ['', '/code', '/logs', '/runs']:
            try:
                s.mkdir(root + d)
            except IOError:
                pass
        s.put(str(ZIP), f'{root}/{ZIP.name}')
        for f in CODE:
            s.put(str(HERE / f), f'{root}/code/{f}')
        s.put(str(R.parents[1] / 'zigzag_v3' / 'model_metadata' / 'models' / 'qwen15' / 'config.json'), f'{root}/model_config_reference.json')
        s.put(str(R / 'STAGE_AB_RUN_PLAN_20260922.json'), f'{root}/STAGE_AB_RUN_PLAN_20260922.json')
    code, out, err = run(c, f"cd {root} && ROOT={root} PY={e['remote_python']} bash code/bootstrap_remote.sh", timeout=3600)
    print(out, err)
    assert code == 0 and 'BOOTSTRAP_OK' in out, 'bootstrap failed'


def smoke(c, e):
    root = e['remote_root']
    cmd = (f"cd {root} && OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false {e['remote_python']} -u code/run_stage_ab.py --world 401 --stage COMMON "
           f"--dryrun 2 --ignore-gates --runs {root}/runs_smoke --eval-batch 64 2>&1 | tail -12")
    code, out, err = run(c, cmd, timeout=1800)
    print(out, err)
    assert code == 0 and 'COMPLETE 401 COMMON' in out, 'smoke test failed'


def start(c, e, hours, rate):
    root = e['remote_root']
    code, out, _ = run(c, f"ps -eo pid,args | grep -E 'queue_stage_ab|lifecycle_guard' | grep -v grep || true")
    assert not out.strip(), f'already running:\n{out}'
    now = time.time()
    budget = dict(rate_cny_per_hour=rate, billing_start_unix=now, session_hours_limit=hours,
                  compute_stop_unix=now + hours * 3600 - 1800, hard_stop_unix=now + hours * 3600 - 600,
                  session_cost_limit_cny=round(rate * hours, 2), source='deploy_stage_ab.py --start')
    with c.open_sftp() as s:
        with s.open(f'{root}/BUDGET.json', 'w') as f:
            f.write(json.dumps(budget, indent=2))
    (R / 'BUDGET_20260922.json').write_text(json.dumps(budget, indent=2), encoding='utf8')
    py = e['remote_python']
    launch = (f"cd {root} && nohup {py} -u code/lifecycle_guard.py > logs/guard.log 2>&1 < /dev/null & "
              f"cd {root} && OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nohup {py} -u code/queue_stage_ab.py > logs/queue.log 2>&1 < /dev/null & "
              f"sleep 3; ps -eo pid,etimes,args | grep -E 'queue_stage_ab|lifecycle_guard' | grep -v grep")
    code, out, err = run(c, launch, timeout=60)
    print(out, err)
    assert 'queue_stage_ab' in out and 'lifecycle_guard' in out, 'launch failed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upload', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start', action='store_true')
    p.add_argument('--hours', type=float, default=6.0)
    p.add_argument('--rate', type=float, default=9.98)
    a = p.parse_args()
    e = endpoint()
    c = connect()
    try:
        if a.upload:
            upload(c, e)
        if a.smoke:
            smoke(c, e)
        if a.start:
            start(c, e, a.hours, a.rate)
    finally:
        c.close()


if __name__ == '__main__':
    main()
