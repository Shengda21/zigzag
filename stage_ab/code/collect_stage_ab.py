"""Local: download Stage A/B results, verify them independently, mark the
remote as safe to shut down.

  python collect_stage_ab.py            # results + final adapters
  python collect_stage_ab.py --adapters # also every step_XXXX adapter (large)

Verification: every eval_*.jsonl is re-scored with zz_score and must equal the
runner's summary; every final adapter sha256 must equal COMPLETE.json; run
manifests must reference the frozen data hashes. On success writes
cloud_return_20260922/LOCAL_RETURN_VERIFIED.json and uploads it (the guard then
shuts the box down once ALL_WORK_COMPLETE.json also exists).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import stat
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R.parent / 'cloud'))
sys.path.insert(0, str(HERE))
from cloud_remote2 import connect, endpoint, run   # noqa: E402
import zz_score as Z                                # noqa: E402

LOCAL = R / 'cloud_return_20260922'
KEEP = ('eval_', 'train.jsonl', 'run_manifest.json', 'GATE_', 'COMPLETE.json', 'PAIRING.json')


def walk(s, remote):
    for entry in s.listdir_attr(remote):
        p = f'{remote}/{entry.filename}'
        if stat.S_ISDIR(entry.st_mode):
            yield from walk(s, p)
        else:
            yield p, entry.st_size


def wanted(path, adapters):
    name = path.rsplit('/', 1)[-1]
    if '/runs/' in path:
        if any(name.startswith(k) or name == k for k in KEEP):
            return True
        if name == 'adapter_model.safetensors' or name == 'adapter_config.json' or name == 'state.json':
            return '/final/' in path or adapters
        return False
    return path.endswith('.json') or '/logs/' in path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--adapters', action='store_true')
    a = p.parse_args()
    e = endpoint()
    root = e['remote_root']
    c = connect()
    got = []
    try:
        with c.open_sftp() as s:
            files = [(pth, sz) for pth, sz in walk(s, root) if wanted(pth, a.adapters)]
            total = sum(sz for _, sz in files)
            print(f'{len(files)} files, {total / 1e6:.1f} MB')
            for pth, sz in files:
                rel = pth[len(root) + 1:]
                dst = LOCAL / rel
                if dst.exists() and dst.stat().st_size == sz:
                    got.append(rel)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                s.get(pth, str(dst))
                got.append(rel)
        # ---- independent verification
        checks = dict(eval_files=0, eval_mismatch=[], adapters=0, adapter_mismatch=[], manifests=0)
        for f in LOCAL.glob('runs/*/eval_*.jsonl'):
            recs, summ = Z.score_file(f)
            ref = json.loads(f.with_suffix('.summary.json').read_text(encoding='utf8'))['scores']
            checks['eval_files'] += 1
            if summ != ref:
                checks['eval_mismatch'].append(str(f.relative_to(LOCAL)))
        for f in LOCAL.glob('runs/*/COMPLETE.json'):
            comp = json.loads(f.read_text(encoding='utf8'))
            ad = f.parent / 'final' / 'adapter_model.safetensors'
            if ad.exists():
                checks['adapters'] += 1
                if hashlib.sha256(ad.read_bytes()).hexdigest() != comp['adapter_sha256']:
                    checks['adapter_mismatch'].append(f.parent.name)
        for f in LOCAL.glob('runs/*/run_manifest.json'):
            man = json.loads(f.read_text(encoding='utf8'))
            world = man['world']
            local = json.loads((R / 'data' / f'world{world}' / 'MANIFEST.json').read_text(encoding='utf8'))['hashes']
            assert man['data_hashes'] == local, f'data hash drift in {f.parent.name}'
            checks['manifests'] += 1
        ok = not checks['eval_mismatch'] and not checks['adapter_mismatch'] and checks['eval_files'] > 0
        verdict = dict(time=time.time(), files=len(got), verified=ok, **checks)
        (LOCAL / 'LOCAL_RETURN_VERIFIED.json' if ok else LOCAL / 'LOCAL_RETURN_FAILED.json').write_text(json.dumps(verdict, indent=2), encoding='utf8')
        print(json.dumps(verdict, indent=2))
        if ok:
            with c.open_sftp() as s:
                with s.open(f'{root}/LOCAL_RETURN_VERIFIED.json', 'w') as f:
                    f.write(json.dumps(verdict, indent=2))
            code, out, _ = run(c, f"test -f {root}/QUEUE_COMPLETE.json && echo QUEUE_DONE || echo QUEUE_NOT_DONE")
            print(out.strip())
    finally:
        c.close()


if __name__ == '__main__':
    main()
