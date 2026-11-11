"""Freeze model-blind reversible-intervention banks in r1603."""
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
from implicit_operator_trajectory_3101.rules import reference
from topological_depth_3701.build import make_item, VERTICES, EDGES, FACES
from essential_depth_3600.build import reachable4, valid

WORLD = 'r1603'
SYMBOLS = dict(zip('ABCDE', 'WYVXU'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prompt(root, word):
    return ('Input: ' + ' '.join(map(str, root)) + '\nProgram: ' +
            ' '.join(SYMBOLS[x] for x in word) +
            '\nReturn the state after each instruction, separated by " | ". '
            'Return only digits, spaces, and separators.')


def answer(trace):
    return ' | '.join(' '.join(map(str, state)) for state in trace)


def write_jsonl(name, bank):
    path = HERE / name
    with path.open('w', encoding='utf-8') as stream:
        for row in bank:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    return sha(path)


def occupied_prior():
    names = ('prospective_support_3803/source_train.jsonl',
             'prospective_support_3803/topology_bank.jsonl',
             'prospective_support_3803/hard_bank.jsonl',
             'prospective_support_3803/shell_train.jsonl',
             'prospective_support_3803/control_train.jsonl',
             'negative_control_3903/broken_train.jsonl')
    states = set()
    for name in names:
        for row in rows(BASE / name):
            root = tuple(row['root'])
            states.add(root)
            states.update(reference(root, row['word'], WORLD))
    return states, {name: sha(BASE / name) for name in names}


def make_probes(occupied):
    rng = random.Random(410320261110)
    seen = set()
    topology = []
    for dimension, cells in enumerate((VERTICES, EDGES, FACES)):
        for cell in cells:
            for item in range(128):
                root, words, traces, _ = make_item(rng, WORLD, cell,
                                                   dimension, occupied, seen)
                for member, (word, trace) in enumerate(zip(words, traces)):
                    topology.append({'id': f'topo4103:{cell}:{item}:{member}',
                        'world': WORLD, 'root': root, 'word': word,
                        'length': len(word), 'cell': cell, 'dimension': dimension,
                        'item': item, 'member': member,
                        'prompt': prompt(root, word), 'answer': answer(trace)})
    assert len(topology) == 5760
    occupied.update(seen)
    rng = random.Random(410320261111)
    hard = []
    used_words = set()
    for length in (7, 8):
        for pair in range(256):
            for _ in range(200000):
                a = ''.join(rng.choice('ABCDE') for _ in range(length))
                b = ''.join(rng.choice('ABCDE') for _ in range(length))
                if len(set(a)) != 5 or len(set(b)) != 5 or a == b or a[0] == b[0]:
                    continue
                if any(x == y for x, y in zip(a, a[1:])) or \
                   any(x == y for x, y in zip(b, b[1:])):
                    continue
                if a in used_words or b in used_words:
                    continue
                root = tuple(rng.randrange(8) for _ in range(8))
                if root in occupied:
                    continue
                ta, tb = valid(root, a, WORLD), valid(root, b, WORLD)
                if ta is None or tb is None or ta[-1] == tb[-1]:
                    continue
                short = reachable4(root, WORLD)
                if ta[-1] in short or tb[-1] in short:
                    continue
                states = {root, *ta, *tb}
                if states & occupied:
                    continue
                occupied.update(states)
                used_words.update((a, b))
                for member, (word, trace) in enumerate(((a, ta), (b, tb))):
                    hard.append({'id': f'hard4103:{length}:{pair}:{member}',
                        'world': WORLD, 'root': root, 'word': word,
                        'length': length, 'pair_id': pair, 'member': member,
                        'prompt': prompt(root, word), 'answer': answer(trace)})
                break
            else:
                raise RuntimeError(('hard sampling exhausted', length, pair))
    assert len(hard) == 1024
    return {'topology': write_jsonl('topology_bank.jsonl', topology),
            'hard': write_jsonl('hard_bank.jsonl', hard)}


def long_word(rng, length):
    while True:
        letters = rng.sample('ABCDE', 3)
        word = ''.join(rng.choice(letters) for _ in range(length))
        if len(set(word)) == 3 and all(a != b for a, b in zip(word, word[1:])):
            return word


def make_training(occupied):
    rng = random.Random(410320261112)
    words = [long_word(rng, length) for length in (7, 8) for _ in range(1536)]
    words += [rng.choice('ABCDE') for _ in range(512)]
    words += [''.join(rng.sample('ABCDE', 2)) for _ in range(512)]
    rng.shuffle(words)
    assert len(words) == 4096
    clean = []
    for index, word in enumerate(words):
        for _ in range(100000):
            root = tuple(rng.randrange(8) for _ in range(8))
            trace = reference(root, word, WORLD)
            states = {root, *trace}
            if states & occupied:
                continue
            occupied.update(states)
            clean.append({'id': f'train4103:{index}', 'world': WORLD,
                'root': root, 'word': word, 'length': len(word),
                'prompt': prompt(root, word), 'answer': answer(trace)})
            break
        else:
            raise RuntimeError(('train sampling exhausted', index))
    broken = [dict(row) for row in clean]
    donor = {}
    for length in (7, 8):
        indices = [i for i, row in enumerate(clean) if row['length'] == length]
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        donor.update({shuffled[i]: shuffled[(i + 1) % len(indices)]
                      for i in range(len(indices))})
    changed = 0
    for index, row in enumerate(broken):
        if row['length'] < 7:
            continue
        original = clean[index]['answer'].split(' | ')
        suffix = clean[donor[index]]['answer'].split(' | ')[3:]
        states = original[:3] + suffix
        changed += states != original
        row['answer'] = ' | '.join(states)
        row['donor_index'] = donor[index]
    assert changed >= 3000
    assert all(a['prompt'] == b['prompt'] and a['word'] == b['word'] and
               a['root'] == b['root'] for a, b in zip(clean, broken))
    return {'rows': 4096, 'long_rows': 3072, 'anchor_rows': 1024,
            'changed_long_answers': changed,
            'length_counts': dict(Counter(row['length'] for row in clean)),
            'clean_sha256': write_jsonl('clean_train.jsonl', clean),
            'broken_sha256': write_jsonl('broken_train.jsonl', broken)}


def main():
    assert not (HERE / 'manifest.json').exists()
    occupied, sources = occupied_prior()
    prior = len(occupied)
    probes = make_probes(occupied)
    training = make_training(occupied)
    complex_record = {'vertices': VERTICES, 'edges': EDGES, 'faces': FACES,
        'boundaries': {**{v: [] for v in VERTICES},
            **{e: list(e) for e in EDGES},
            **{f: [''.join(x) for x in itertools.combinations(f, 2)] for f in FACES}},
        'field': 'F2'}
    (HERE / 'complex.json').write_text(json.dumps(complex_record, indent=2))
    gate = {'status': 'PASS_R4103_MODEL_BLIND_BANKS',
            'prior_states': prior, 'total_states': len(occupied),
            'prior_sha256': sources, 'probes': probes, 'training': training}
    (HERE / 'cpu_gate.json').write_text(json.dumps(gate, indent=2))
    names = ('PROTOCOL.md', 'build.py', 'complex.json', 'cpu_gate.json',
             'topology_bank.jsonl', 'hard_bank.jsonl', 'clean_train.jsonl',
             'broken_train.jsonl')
    manifest = {'status': 'FROZEN_BEFORE_R4103_INTERVENTION_MODEL_QUERY',
                'files': {name: sha(HERE / name) for name in names}}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': manifest['status'],
                      'manifest_sha256': sha(HERE / 'manifest.json'),
                      'prior_states': prior, 'total_states': len(occupied),
                      'probes': probes, 'training': training}))


if __name__ == '__main__':
    main()
