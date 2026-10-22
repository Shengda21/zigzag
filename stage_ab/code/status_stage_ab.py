"""Local: one-shot status snapshot of the remote Stage A/B queue."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R.parent / 'cloud'))
from cloud_remote2 import connect, endpoint, run   # noqa: E402

REMOTE = r'''
import json, subprocess, time
from pathlib import Path
r = Path(%r); out = {'time': time.time()}
for name in ['RUNNING.json', 'STAGE_A_COMPLETE.json', 'QUEUE_SKIPPED.json', 'QUEUE_COMPLETE.json', 'QUEUE_ERROR.json', 'LIFECYCLE_STATUS.json', 'BUDGET.json', 'REMOTE_CONFIG.json']:
    p = r / name
    if p.exists():
        try: out[name] = json.loads(p.read_text())
        except Exception: out[name] = p.read_text()[-2000:]
out['stop_flag'] = (r / 'STOP_COMPUTE').exists()
out['processes'] = subprocess.getoutput("ps -eo pid,pcpu,etimes,args | grep -E 'run_stage_ab|queue_stage_ab|lifecycle_guard' | grep -v grep")
out['gpu'] = subprocess.getoutput('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')
out['disk'] = subprocess.getoutput("df -h %s | tail -1")
out['complete'] = sorted(p.parent.name for p in (r / 'runs').glob('*/COMPLETE.json'))
out['gates'] = {p.parent.name: json.loads(p.read_text())['gate']['passed'] for p in (r / 'runs').glob('*/COMPLETE.json')}
q = r / 'logs' / 'queue.log'
if q.exists(): out['queue_tail'] = q.read_text(errors='replace').splitlines()[-5:]
if (r / 'RUNNING.json').exists():
    d = json.loads((r / 'RUNNING.json').read_text()); run = r / 'runs' / f"world{d['world']}_{d['stage']}"
    t = run / 'train.jsonl'
    if t.exists():
        lines = t.read_text().splitlines()
        if lines: out['latest_train'] = json.loads(lines[-1])
    ev = sorted(run.glob('eval_*.summary.json'))
    if ev:
        s = json.loads(ev[-1].read_text()); out['latest_eval'] = dict(step=s['step'], **{k: round(v['accuracy'], 4) for k, v in s['scores'].items() if k.endswith(':all')})
print(json.dumps(out))
'''


def main():
    e = endpoint()
    c = connect()
    try:
        code, out, err = run(c, f"{e['remote_python']} - <<'PY'\n{REMOTE % (e['remote_root'], e['remote_root'])}\nPY", timeout=60)
        assert code == 0, err
        x = json.loads(out)
        (R / 'server_snapshot').mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        (R / 'server_snapshot' / f'STATUS_{stamp}.json').write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')
        (R / 'server_snapshot' / 'LATEST.json').write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(x, ensure_ascii=False, indent=2))
    finally:
        c.close()


if __name__ == '__main__':
    main()
