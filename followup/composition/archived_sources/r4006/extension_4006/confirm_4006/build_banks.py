"""Freeze untouched fourth-world extension confirmation banks."""
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent.parent
sys.path.insert(0, str(BASE))
from prospective_world_4004.rules import reference as rule_reference, audit
from topological_depth_3701.build import face_word, VERTICES, EDGES, FACES

SYMBOLS = dict(zip('ABCDE', 'WYVXU'))
WORLD = 'r4004'


def reference(root, word, world=WORLD):
    assert world == WORLD
    return rule_reference(root, word)


def trace_if_effective(root, word, world):
    trace = reference(root, word, world)
    if any(a == b for a, b in zip((root, *trace[:-1]), trace)):
        return None
    if len({root, *trace}) != len(word) + 1:
        return None
    return trace


def make_item(rng, world, cell, dimension, old, seen):
    for attempt in range(100000):
        root = tuple(rng.randrange(8) for _ in range(8))
        if root in old or root in seen:
            continue
        if dimension == 0:
            words = [cell]
        elif dimension == 1:
            words = [cell, cell[::-1]]
        else:
            length = 7 if rng.randrange(2) == 0 else 8
            words = [face_word(rng, cell, length), face_word(rng, cell, length)]
            if words[0] == words[1] or words[0][0] == words[1][0]:
                continue
        traces = ([reference(root, word, world) for word in words]
                  if dimension == 1 else
                  [trace_if_effective(root, word, world) for word in words])
        if any(trace is None for trace in traces):
            continue
        if dimension == 2 and traces[0][-1] == traces[1][-1]:
            continue
        states = {root}
        for trace in traces:
            states.update(trace)
        if any(state in old or state in seen for state in states):
            continue
        seen.update(states)
        return root, words, traces, attempt + 1
    raise RuntimeError((world, cell, dimension, 'exhausted'))


def reachable4(root, world):
    seen = {root}
    frontier = {root}
    for _ in range(4):
        next_states = {reference(state, op, world)[0]
                       for state in frontier for op in 'ABCDE'}
        frontier = next_states - seen
        seen.update(next_states)
    return seen


def valid(root, word, world, short=None):
    trace = reference(root, word, world)
    if len({root, *trace}) != len(word) + 1:
        return None
    for index in range(len(word)):
        shorter = word[:index] + word[index + 1:]
        if reference(root, shorter, world)[-1] == trace[-1]:
            return None
    if short is not None and trace[-1] in short:
        return None
    return trace


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
    return ' | '.join(' '.join(map(str, s)) for s in trace)


def base_row(identifier, root, word, trace):
    return {'id': identifier, 'world': WORLD, 'root': root, 'word': word,
            'length': len(word), 'prompt': prompt(root, word), 'answer': answer(trace)}


def write_jsonl(name, data):
    path = HERE / name
    with path.open('w', encoding='utf-8') as stream:
        for row in data:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    return sha(path)


def prior_states():
    names = ['prospective_world_4004/source_train.jsonl',
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
             'prospective_support_3803/topology_bank.jsonl',
             'prospective_support_3803/hard_bank.jsonl']
    occupied = set()
    for name in names:
        for row in rows(BASE / name):
            root = tuple(row['root'])
            occupied.add(root)
            occupied.update(reference(root, row['word'], WORLD))
    return occupied, {name: sha(BASE / name) for name in names}


def build_topology(occupied):
    rng = random.Random(400620261120)
    seen = set()
    bank = []
    attempts = Counter()
    for dimension, cells in enumerate((VERTICES, EDGES, FACES)):
        for cell in cells:
            for item in range(128):
                root, words, traces, used = make_item(rng, WORLD, cell,
                                                      dimension, occupied, seen)
                attempts[str(dimension)] += used
                for member, (word, trace) in enumerate(zip(words, traces)):
                    row = base_row(f'topo4006:{cell}:{item}:{member}', root, word, trace)
                    row.update({'cell': cell, 'dimension': dimension,
                                'item': item, 'member': member})
                    bank.append(row)
    assert len(bank) == 5760
    occupied.update(seen)
    return {'rows': len(bank), 'new_states': len(seen),
            'sampling_attempts': dict(attempts),
            'sha256': write_jsonl('topology_bank.jsonl', bank)}


def random_word(rng, length):
    while True:
        word = ''.join(rng.choice('ABCDE') for _ in range(length))
        if len(set(word)) == 5 and all(a != b for a, b in zip(word, word[1:])):
            return word


def build_hard(occupied):
    rng = random.Random(400620261121)
    bank = []
    used_words = set()
    attempts = Counter()
    for length in (7, 8):
        for pair in range(256):
            for trial in range(200000):
                a, b = random_word(rng, length), random_word(rng, length)
                if a == b or a[0] == b[0] or a in used_words or b in used_words:
                    continue
                root = tuple(rng.randrange(8) for _ in range(8))
                if root in occupied:
                    continue
                ta = valid(root, a, WORLD)
                if ta is None:
                    continue
                tb = valid(root, b, WORLD)
                if tb is None or ta[-1] == tb[-1]:
                    continue
                short = reachable4(root, WORLD)
                if ta[-1] in short or tb[-1] in short:
                    continue
                states = {root, *ta, *tb}
                if states & occupied:
                    continue
                used_words.update((a, b))
                occupied.update(states)
                attempts[str(length)] += trial + 1
                for member, (word, trace) in enumerate(((a, ta), (b, tb))):
                    row = base_row(f'hard4006:{length}:{pair}:{member}', root, word, trace)
                    row.update({'pair_id': pair, 'member': member})
                    bank.append(row)
                break
            else:
                raise RuntimeError(('hard sampling exhausted', length, pair))
    assert len(bank) == 1024
    return {'rows': len(bank), 'pairs_per_length': 256,
            'sampling_attempts': dict(attempts),
            'sha256': write_jsonl('hard_bank.jsonl', bank)}


PAIRS = [('ACD', 'ACD', 683), ('ADE', 'ADE', 683),
         ('BCE', 'BCE', 683), ('BDE', 'BDE', 683),
         ('ACE', 'ABC', 682), ('BCD', 'CDE', 682)]


def template(rng, length):
    while True:
        indices = [rng.randrange(3) for _ in range(length)]
        if len(set(indices)) == 3 and all(a != b for a, b in zip(indices, indices[1:])):
            return indices


def effective(trace, root):
    return all(a != b for a, b in zip((root, *trace[:-1]), trace)) and \
           len({root, *trace}) == len(trace) + 1


def build_curricula(occupied):
    rng = random.Random(400620261122)
    special_patterns = {length: [template(rng, length) for _ in range(341)]
                        for length in (7, 8)}
    special_counts = {(shell, length): 0
                      for shell in ('ACE', 'BCD') for length in (7, 8)}
    specifications = []
    for type_id, (shell, control, count) in enumerate(PAIRS):
        count7 = 342 if type_id < 2 else 341
        lengths = [7] * count7 + [8] * (count - count7)
        for length in lengths:
            specifications.append((shell, control, length))
    assert len(specifications) == 4096
    rng.shuffle(specifications)
    banks = {'shell': [], 'control': []}
    attempts = Counter()
    for index, (shell, control, length) in enumerate(specifications):
        fixed_pattern = None
        if shell in ('ACE', 'BCD'):
            key = shell, length
            fixed_pattern = special_patterns[length][special_counts[key]]
            special_counts[key] += 1
        for trial in range(100000):
            pattern = fixed_pattern if fixed_pattern is not None else template(rng, length)
            shell_labels, control_labels = (
                ('ACE', 'ACB') if shell == 'ACE' else
                ('CDB', 'CDE') if shell == 'BCD' else
                (shell, control))
            words = {'shell': ''.join(shell_labels[j] for j in pattern),
                     'control': ''.join(control_labels[j] for j in pattern)}
            root = tuple(rng.randrange(8) for _ in range(8))
            if root in occupied:
                continue
            traces = {arm: reference(root, word, WORLD) for arm, word in words.items()}
            states = {root, *traces['shell'], *traces['control']}
            if states & occupied:
                continue
            occupied.update(states)
            attempts[shell + '/' + control] += trial + 1
            for arm in banks:
                row = base_row(f'{arm}4006:{index}', root, words[arm], traces[arm])
                row.update({'triple': shell if arm == 'shell' else control,
                            'index': index, 'arm': arm})
                banks[arm].append(row)
            break
        else:
            raise RuntimeError(('curriculum sampling exhausted', index, shell, control))
    shell, control = banks['shell'], banks['control']
    assert len(shell) == len(control) == 4096
    assert Counter(r['length'] for r in shell) == Counter({7: 2048, 8: 2048})
    assert all(a['root'] == b['root'] and a['length'] == b['length']
               for a, b in zip(shell, control))
    same = sum(a['word'] == b['word'] and a['answer'] == b['answer']
               for a, b in zip(shell, control))
    assert same == 2732
    occurrence = {arm: dict(Counter(ch for r in bank for ch in r['triple']))
                  for arm, bank in banks.items()}
    assert occurrence['shell'] == occurrence['control']
    assert set(special_counts.values()) == {341}
    token_counts = {arm: dict(Counter(ch for r in bank for ch in r['word']))
                    for arm, bank in banks.items()}
    assert token_counts['shell'] == token_counts['control']
    hashes = {arm: write_jsonl(f'{arm}_train.jsonl', bank) for arm, bank in banks.items()}
    return {'rows_per_arm': 4096, 'identical_examples': same,
            'triple_operator_incidence': occurrence,
            'token_operator_counts': token_counts,
            'sampling_attempts': dict(attempts), 'sha256': hashes}


def main():
    assert not (HERE / 'banks_manifest.json').exists()
    assert audit() == {'A': 8, 'B': 3, 'C': 8, 'D': 2, 'E': 2}
    occupied, sources = prior_states()
    prior_count = len(occupied)
    topology = build_topology(occupied)
    hard = build_hard(occupied)
    curricula = build_curricula(occupied)
    complex_record = {'vertices': VERTICES, 'edges': EDGES, 'faces': FACES,
        'boundaries': {**{v: [] for v in VERTICES},
            **{e: list(e) for e in EDGES},
            **{f: [''.join(x) for x in itertools.combinations(f, 2)] for f in FACES}},
        'field': 'F2'}
    (HERE / 'complex.json').write_text(json.dumps(complex_record, indent=2))
    gate = {'status': 'PASS_R4006_UNTOUCHED_CONFIRMATION_BANKS', 'world': WORLD,
            'prior_states': prior_count, 'total_states': len(occupied),
            'source_sha256': sources, 'topology': topology,
            'hard': hard, 'curricula': curricula}
    (HERE / 'banks_cpu_gate.json').write_text(json.dumps(gate, indent=2))
    names = ('PROTOCOL.md', 'build_banks.py', 'complex.json',
             'topology_bank.jsonl', 'hard_bank.jsonl', 'shell_train.jsonl',
             'control_train.jsonl', 'banks_cpu_gate.json')
    manifest = {'status': 'FROZEN_BEFORE_R4006_CONFIRMATION_MODEL_QUERY',
                'files': {name: sha(HERE / name) for name in names}}
    (HERE / 'banks_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': manifest['status'], 'manifest_sha256': sha(HERE / 'banks_manifest.json'),
                      'prior_states': prior_count, 'total_states': len(occupied),
                      'topology': topology, 'hard': hard, 'curricula': curricula}))


if __name__ == '__main__':
    main()
