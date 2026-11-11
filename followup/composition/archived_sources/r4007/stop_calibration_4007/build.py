"""Freeze a state-disjoint, length-balanced stop-calibration bank."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
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


def long_word(rng, length):
    while True:
        labels = rng.sample('ABCDE', 3)
        word = ''.join(rng.choice(labels) for _ in range(length))
        if len(set(word)) == 3 and all(a != b for a, b in zip(word, word[1:])):
            return word


def main():
    assert not (HERE / 'manifest.json').exists()
    sources = (
        'prospective_world_4004/source_train.jsonl',
        'prospective_world_4004/repair_train.jsonl',
        'prospective_world_4004/topology_bank.jsonl',
        'prospective_world_4004/hard_bank.jsonl',
        'prospective_world_4004/shell_train.jsonl',
        'prospective_world_4004/control_train.jsonl',
        'prospective_world_4004/confirm_4005/topology_bank.jsonl',
        'prospective_world_4004/confirm_4005/hard_bank.jsonl',
        'prospective_world_4004/confirm_4005/shell_train.jsonl',
        'prospective_world_4004/confirm_4005/control_train.jsonl',
        'prospective_world_4004/extension_4006/train.jsonl',
        'prospective_world_4004/extension_4006/confirm_4006/topology_bank.jsonl',
        'prospective_world_4004/extension_4006/confirm_4006/hard_bank.jsonl',
        'prospective_world_4004/extension_4006/confirm_4006/shell_train.jsonl',
        'prospective_world_4004/extension_4006/confirm_4006/control_train.jsonl',
    )
    occupied = set()
    for name in sources:
        with (BASE / name).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                root = tuple(row['root'])
                occupied.add(root)
                occupied.update(reference(root, row['word']))
    prior = len(occupied)
    rng = random.Random(400720261115)
    words = [long_word(rng, length)
             for length, count in ((7, 1024), (8, 1024), (9, 512), (10, 512))
             for _ in range(count)]
    words += [rng.choice('ABCDE') for _ in range(512)]
    words += [''.join(rng.sample('ABCDE', 2)) for _ in range(512)]
    rng.shuffle(words)
    assert len(words) == 4096
    bank = []
    for index, word in enumerate(words):
        for _ in range(100000):
            root = tuple(rng.randrange(8) for _ in range(8))
            trace = reference(root, word)
            states = {root, *trace}
            if states & occupied:
                continue
            occupied.update(states)
            bank.append({'id': f'stop4007:{index}', 'root': root,
                         'word': word, 'length': len(word),
                         'prompt': prompt(root, word),
                         'answer': ' | '.join(' '.join(map(str, s)) for s in trace)})
            break
        else:
            raise RuntimeError(('bank sampling exhausted', index))
    counts = Counter(row['length'] for row in bank)
    assert counts == {1: 512, 2: 512, 7: 1024, 8: 1024, 9: 512, 10: 512}
    target = HERE / 'train.jsonl'
    target.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n'
                              for row in bank), encoding='utf-8')
    result = {'status': 'FROZEN_BEFORE_R4007_UPDATES', 'rows': 4096,
              'length_counts': dict(counts), 'prior_states': prior,
              'total_states': len(occupied),
              'source_sha256': {name: sha(BASE / name) for name in sources},
              'files': {name: sha(HERE / name) for name in
                        ('PROTOCOL.md', 'build.py', 'train.jsonl')}}
    (HERE / 'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'], 'rows': 4096,
                      'manifest_sha256': sha(HERE / 'manifest.json')}))


if __name__ == '__main__':
    main()
