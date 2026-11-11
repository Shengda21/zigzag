"""Freeze the fixed-bank baseline comparison before the first query."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK = HERE.parent / 'counterfactual_state_use_4301'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = HERE / 'manifest.json'
    assert not target.exists()
    result = {
        'status': 'FROZEN_BEFORE_R4401_QUERIES',
        'files': {name: sha(HERE / name) for name in
                  ('PROTOCOL.md', 'evaluate.py', 'run.sh', 'freeze.py')},
        'banks': {world: sha(BANK / f'bank_{world}.jsonl')
                  for world in ('r1603', 'r4004')},
        'source_bank_manifest_sha256': sha(BANK / 'manifest.json'),
    }
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'],
                      'manifest_sha256': sha(target), 'banks': result['banks']}))


if __name__ == '__main__':
    main()
