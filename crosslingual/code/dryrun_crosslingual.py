"""CPU dry-run of MULTI1/MONO1 on the toy model built by stage_ab dryrun_local."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R.parent / 'stage_ab' / 'code'))
from dryrun_local import build_toy    # noqa: E402
import zz_score as Z                  # noqa: E402


def main():
    scratch = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun')) / 'crosslingual'
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / 'runs').mkdir(parents=True)
    toy = build_toy(scratch / 'toy_model')
    env = dict(os.environ, MODEL_PATH=str(toy))
    base = [sys.executable, '-u', str(HERE / 'run_crosslingual.py'), '--device', 'cpu', '--dryrun', '3', '--runs', str(scratch / 'runs'), '--eval-batch', '8']
    for arm in ('MULTI1', 'MONO1'):
        r = subprocess.run(base + ['--world', '501', '--arm', arm], env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
        print(f'--- 501 {arm} exit={r.returncode}')
        print('\n'.join(r.stdout.splitlines()[-3:]))
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit(arm)
        out = scratch / 'runs' / f'world501_{arm}'
        for f in ['run_manifest.json', 'PAIRING.json', 'eval_0000.jsonl', 'GATE_3.json', 'COMPLETE.json', 'final/adapter_model.safetensors']:
            assert (out / f).exists(), f
    a = json.loads((scratch / 'runs/world501_MULTI1/PAIRING.json').read_text(encoding='utf8'))
    b = json.loads((scratch / 'runs/world501_MONO1/PAIRING.json').read_text(encoding='utf8'))
    assert a['slots_sha256'] == b['slots_sha256']
    ma = json.loads((scratch / 'runs/world501_MULTI1/run_manifest.json').read_text(encoding='utf8'))
    mb = json.loads((scratch / 'runs/world501_MONO1/run_manifest.json').read_text(encoding='utf8'))
    assert ma['train_records'] == mb['train_records']
    n = 0
    for f in (scratch / 'runs').glob('*/eval_*.jsonl'):
        recs, summ = Z.score_file(f)
        assert summ == json.loads(f.with_suffix('.summary.json').read_text(encoding='utf8'))['scores'], f
        n += 1
    ev = json.loads((scratch / 'runs/world501_MULTI1/eval_0000.summary.json').read_text(encoding='utf8'))['scores']
    langs = sorted({k.split(':lang')[1].split(':')[0] for k in ev if ':lang' in k})
    print(json.dumps(dict(status='CROSSLINGUAL_DRYRUN_PASSED', eval_files_rescored=n, probe_langs=langs, eval_rows=ma['eval_records'])))


if __name__ == '__main__':
    main()
