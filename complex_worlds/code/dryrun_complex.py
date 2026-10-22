"""CPU dry-run of run_complex.py on the toy model, followed by the spec-driven
analysis on the dry-run outputs (exercises the whole path: stream -> eval ->
cell scores -> masks -> barcodes -> ledger)."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
ROOT = R.parent
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
from dryrun_local import build_toy    # noqa: E402
import zz_score as Z                  # noqa: E402
import zz_analyze as A                # noqa: E402


def main():
    scratch = Path(os.environ.get('ZZ_DRYRUN_DIR', Path(os.environ.get('TEMP', '/tmp')) / 'zz_dryrun')) / 'complex'
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / 'runs').mkdir(parents=True)
    toy = build_toy(scratch / 'toy_model')
    env = dict(os.environ, MODEL_PATH=str(toy))
    cmd = [sys.executable, '-u', str(HERE / 'run_complex.py'), '--spec', 'RCN12', '--world', '601', '--seed', '1', '--device', 'cpu',
           '--dryrun', '3', '--runs', str(scratch / 'runs'), '--eval-batch', '8']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf8', errors='replace')
    print('--- 601 S1 exit=', r.returncode)
    print('\n'.join(r.stdout.splitlines()[-3:]))
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit('runner failed')
    out = scratch / 'runs' / 'world601_S1'
    for f in ['run_manifest.json', 'eval_0000.jsonl', 'GATE_3.json', 'COMPLETE.json', 'final/adapter_model.safetensors']:
        assert (out / f).exists(), f
    n = 0
    for f in out.glob('eval_*.jsonl'):
        recs, summ = Z.score_file(f)
        assert summ == json.loads(f.with_suffix('.summary.json').read_text(encoding='utf8'))['scores'], f
        n += 1
    spec_path = R / 'data' / 'RCN12' / 'world601' / 'COMPLEX.json'
    res = A.analyze_run(out, str(spec_path), n_perm=20, seed=1)
    A.write_outputs(res, scratch / 'analysis_601_S1')
    assert set(res['ledger']['cells']) >= {'A3B1', 'B4C1', 'A1A3', 'A2B2'}
    print(json.dumps(dict(status='COMPLEX_DRYRUN_PASSED', eval_files_rescored=n, cells=len(res['trajectory']), verdict=res['synchrony']['verdict'],
                          control_violations=res['ledger']['control_violations'])))


if __name__ == '__main__':
    main()
