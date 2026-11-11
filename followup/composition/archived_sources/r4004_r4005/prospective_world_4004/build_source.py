"""Freeze a fixed-symbol, rule-text-free fourth-world source curriculum."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from prospective_world_4004.rules import reference, audit

SYMBOLS = dict(zip('ABCDE', 'WYVXU'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prompt(root, word):
    return ('Input: ' + ' '.join(map(str, root)) + '\nProgram: ' +
            ' '.join(SYMBOLS[x] for x in word) +
            '\nReturn the state after each instruction, separated by " | ". '
            'Return only digits, spaces, and separators.')


def main():
    assert not (HERE / 'source_manifest.json').exists()
    assert audit() == {'A': 8, 'B': 3, 'C': 8, 'D': 2, 'E': 2}
    original = BASE / 'prospective_support_3803' / 'source_train.jsonl'
    target = HERE / 'source_train.jsonl'
    counts = Counter()
    with original.open(encoding='utf-8') as source, target.open('w', encoding='utf-8') as sink:
        for line in source:
            row = json.loads(line)
            root, word = tuple(row['root']), row['word']
            assert len(set(word)) <= 2
            answer = ' | '.join(' '.join(map(str, s)) for s in reference(root, word))
            assert row['length'] == len(word)
            counts[row['population']] += 1
            record = {'id': 'train4004:' + str(row['step']) + ':' + str(row['slot']),
                      'root': root, 'word': word,
                      'prompt': prompt(root, word), 'answer': answer,
                      'length': len(word), 'step': row['step'], 'slot': row['slot'],
                      'population': row['population']}
            sink.write(json.dumps(record, separators=(',', ':')) + '\n')
    assert sum(counts.values()) == 16384
    manifest = {'status': 'FROZEN_HIDDEN_SOURCE_BEFORE_MODEL_TRAINING',
                'world': 'r4004', 'rows': 16384, 'populations': dict(counts),
                'original_sha256': sha(original),
                'files': {name: sha(HERE / name) for name in
                          ('PROTOCOL.md', 'rules.py', 'build_source.py', 'source_train.jsonl')}}
    (HERE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'rows': 16384,
                      'manifest_sha256': sha(HERE / 'source_manifest.json')}))


if __name__ == '__main__':
    main()
