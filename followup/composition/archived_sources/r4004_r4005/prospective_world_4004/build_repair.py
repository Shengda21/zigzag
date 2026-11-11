"""Freeze a balanced fresh unary/binary r4004 source repair bank."""
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from prospective_world_4004.rules import reference

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
    assert not (HERE / 'repair_manifest.json').exists()
    names = ('source_train.jsonl', 'topology_bank.jsonl', 'hard_bank.jsonl',
             'shell_train.jsonl', 'control_train.jsonl')
    occupied = set()
    for name in names:
        for line in (HERE / name).read_text().splitlines():
            row = json.loads(line)
            root = tuple(row['root'])
            occupied.add(root)
            occupied.update(reference(root, row['word']))
    prior = len(occupied)
    words = [a + b for a, b in itertools.permutations('ABCDE', 2)
             for _ in range(192)]
    words += [op for op in 'ABCDE' for _ in range(52 if op == 'A' else 51)]
    assert len(words) == 4096
    rng = random.Random(400420261115)
    rng.shuffle(words)
    data = []
    attempts = 0
    for index, word in enumerate(words):
        for trial in range(100000):
            root = tuple(rng.randrange(8) for _ in range(8))
            trace = reference(root, word)
            states = {root, *trace}
            if states & occupied:
                continue
            occupied.update(states)
            attempts += trial + 1
            data.append({'id': f'repair4004:{index}', 'root': root,
                'word': word, 'length': len(word), 'prompt': prompt(root, word),
                'answer': ' | '.join(' '.join(map(str, s)) for s in trace)})
            break
        else:
            raise RuntimeError(('repair sampling exhausted', index))
    path = HERE / 'repair_train.jsonl'
    path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n'
                            for r in data), encoding='utf-8')
    counts = Counter(r['word'] for r in data)
    assert all(counts[a + b] == 192 for a, b in itertools.permutations('ABCDE', 2))
    assert sum(counts[op] for op in 'ABCDE') == 256
    gate = {'status': 'FROZEN_R4004_REPAIR_BEFORE_UPDATES',
            'prior_states': prior, 'total_states': len(occupied),
            'rows': len(data), 'word_counts': dict(counts),
            'sampling_attempts': attempts,
            'files': {name: sha(HERE / name) for name in
                      ('REPAIR_PROTOCOL.md', 'build_repair.py', 'repair_train.jsonl')},
            'source_files': {name: sha(HERE / name) for name in names}}
    (HERE / 'repair_manifest.json').write_text(json.dumps(gate, indent=2))
    print(json.dumps({'status': gate['status'],
                      'manifest_sha256': sha(HERE / 'repair_manifest.json'),
                      'rows': len(data), 'states': len(occupied) - prior}))


if __name__ == '__main__':
    main()
