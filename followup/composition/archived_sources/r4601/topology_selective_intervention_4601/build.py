"""Freeze equal-dose topology-selective training banks without model calls."""
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
from reversible_intervention_4103 import build as prior
from implicit_operator_trajectory_3101.rules import reference

WORLD = 'r1603'
TRIPLES = tuple(map(''.join, itertools.combinations('ABCDE', 3)))
CYCLE = frozenset(('ACD', 'ACE', 'ADE', 'BCD', 'BCE', 'BDE'))
ACYCLIC = frozenset(('ABC', 'ACD', 'ADE', 'BCE', 'BDE', 'CDE'))
assert len(TRIPLES) == 10 and len(CYCLE) == len(ACYCLIC) == 6


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(name, values):
    path = HERE / name
    with path.open('w', encoding='utf8') as f:
        for value in values:
            f.write(json.dumps(value, separators=(',', ':')) + '\n')
    return sha(path)


def random_pattern(rng, length):
    while True:
        pattern = tuple(rng.randrange(3) for _ in range(length))
        if len(set(pattern)) == 3 and all(a != b for a, b in zip(pattern, pattern[1:])):
            return pattern


def make_words(rng):
    result = []
    for triple in TRIPLES:
        for length in (7, 8):
            for _ in range(21):
                pattern = random_pattern(rng, length)
                for letters in itertools.permutations(triple):
                    result.append((''.join(letters[i] for i in pattern), triple))
    assert len(result) == 2520
    anchors = [(rng.choice('ABCDE'), None) for _ in range(788)]
    anchors += [(''.join(rng.sample('ABCDE', 2)), None) for _ in range(788)]
    result.extend(anchors)
    rng.shuffle(result)
    assert len(result) == 4096
    return result


def main():
    assert not (HERE / 'manifest.json').exists()
    rng = random.Random(460120261110)
    occupied, provenance = prior.occupied_prior()
    for name in ('reversible_intervention_4103/topology_bank.jsonl',
                 'reversible_intervention_4103/hard_bank.jsonl',
                 'reversible_intervention_4103/clean_train.jsonl',
                 'counterfactual_state_use_4301/bank_r1603.jsonl',
                 'conditional_branch_4501/bank_r1603.jsonl'):
        path = BASE / name
        provenance[name] = sha(path)
        for row in rows(path):
            root = tuple(row['root'])
            occupied.add(root)
            occupied.update(reference(root, row['word'], WORLD))
    words = make_words(rng)
    clean = []
    for index, (word, triple) in enumerate(words):
        for _ in range(100000):
            root = tuple(rng.randrange(8) for _ in range(8))
            trace = reference(root, word, WORLD)
            if {root, *trace} & occupied:
                continue
            occupied.update((root, *trace))
            clean.append({'id': f'train4601:{index}', 'world': WORLD,
                          'root': root, 'word': word, 'triple': triple,
                          'length': len(word), 'prompt': prior.prompt(root, word),
                          'answer': prior.answer(trace)})
            break
        else:
            raise RuntimeError(('root sampling exhausted', index))
    assert len(clean) == 4096
    donor = {}
    for length in (7, 8):
        donor[length] = []
        for _ in range(504):
            donor[length].append([' '.join(str(rng.randrange(8)) for _ in range(8))
                                  for _ in range(length - 3)])
    arms = {}
    for arm, keep in (('cycle', CYCLE), ('acyclic', ACYCLIC)):
        broken = [dict(row) for row in clean]
        counts = Counter()
        for length in (7, 8):
            indices = [i for i, row in enumerate(clean)
                       if row['length'] == length and row['triple'] not in keep]
            assert len(indices) == 504
            for j, index in enumerate(indices):
                answer = clean[index]['answer'].split(' | ')
                broken[index]['answer'] = ' | '.join(answer[:3] + donor[length][j])
                assert broken[index]['answer'] != clean[index]['answer']
                counts.update(clean[index]['word'])
        arms[arm] = {'sha256': save(f'train_{arm}.jsonl', broken),
                     'corrupted_operator_tokens': dict(sorted(counts.items())),
                     'corrupted_rows': 1008}
    assert arms['cycle']['corrupted_operator_tokens'] == arms['acyclic']['corrupted_operator_tokens']
    assert all(arms[x]['corrupted_operator_tokens'][o] > 0 for x in arms for o in 'ABCDE')
    assert [row['prompt'] for row in clean] == [row['prompt'] for row in broken]
    # The multiset of random late-state targets is shared between arms by length.
    for length in (7, 8):
        suffixes = []
        for arm in ('cycle', 'acyclic'):
            bank = list(rows(HERE / f'train_{arm}.jsonl'))
            suffixes.append(sorted(tuple(row['answer'].split(' | ')[3:]) for row in bank
                                   if row['length'] == length and row['triple'] not in
                                   (CYCLE if arm == 'cycle' else ACYCLIC)))
        assert suffixes[0] == suffixes[1]
    gate = {'status': 'PASS_MODEL_BLIND_EQUAL_DOSE_BANKS', 'world': WORLD,
            'source': provenance, 'clean_sha256': save('train_clean.jsonl', clean),
            'arms': arms, 'rows': 4096, 'three_operator_rows': 2520,
            'anchor_rows': 1576,
            'eval_banks': {name: sha(BASE / 'reversible_intervention_4103' / name)
                           for name in ('topology_bank.jsonl', 'hard_bank.jsonl')}}
    (HERE / 'cpu_gate.json').write_text(json.dumps(gate, indent=2))
    names = ('PROTOCOL.md', 'build.py', 'cpu_gate.json', 'train_clean.jsonl',
             'train_cycle.jsonl', 'train_acyclic.jsonl')
    manifest = {'status': 'FROZEN_BEFORE_R4601_MODEL_QUERY',
                'files': {name: sha(HERE / name) for name in names},
                'reference_banks': gate['eval_banks']}
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': manifest['status'],
                      'manifest_sha256': sha(HERE / 'manifest.json'),
                      'arms': arms}))


if __name__ == '__main__':
    main()
