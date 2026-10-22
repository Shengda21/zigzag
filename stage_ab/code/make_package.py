"""Build STAGE_AB_INPUTS_20260922.zip (data + code) and record its sha256.
The 20260921 zip is left untouched for provenance."""
import hashlib
import json
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
OUT = R / 'STAGE_AB_INPUTS_20260922.zip'
CODE = ['zz_score.py', 'zz_train.py', 'run_stage_ab.py', 'queue_stage_ab.py', 'lifecycle_guard.py', 'audit2.py',
        'bootstrap_remote.sh', 'prepare_common.py', 'test_zz_score.py']


def main():
    names = []
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted((R / 'data').rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(R).as_posix())
                names.append(f.relative_to(R).as_posix())
        for f in CODE:
            z.write(HERE / f, f'code/{f}')
            names.append(f'code/{f}')
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (R / 'STAGE_AB_PACKAGE_20260922.json').write_text(json.dumps(dict(
        zip=OUT.name, sha256=digest, bytes=OUT.stat().st_size, files=names,
        built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2), encoding='utf8')
    print(OUT.name, OUT.stat().st_size, digest[:16], len(names), 'files')


if __name__ == '__main__':
    main()
