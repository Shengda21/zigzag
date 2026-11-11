"""Freeze fresh five-operator programs and counterfactual trace prefixes."""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from implicit_operator_trajectory_3101.rules import reference as reference_1603
from essential_depth_3600.build import valid as valid_1603, reachable4 as reachable4_1603
from prospective_world_4004.rules import reference as reference_4004
from prospective_world_4004.confirm_4005.build_banks import (
    valid as valid_4004, reachable4 as reachable4_4004)

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


def reference(root, word, world):
    return (reference_1603(root, word, world) if world == 'r1603'
            else reference_4004(root, word))


def valid(root, word, world):
    return (valid_1603(root, word, world) if world == 'r1603'
            else valid_4004(root, word, world))


def reachable4(root, world):
    return (reachable4_1603(root, world) if world == 'r1603'
            else reachable4_4004(root, world))


def prompt(root, word):
    return ('Input: ' + ' '.join(map(str, root)) + '\nProgram: ' +
            ' '.join(SYMBOLS[x] for x in word) +
            '\nReturn the state after each instruction, separated by " | ". '
            'Return only digits, spaces, and separators.')


def answer(trace):
    return ' | '.join(' '.join(map(str, state)) for state in trace)


def prior(world):
    source = ('reversible_intervention_4103' if world == 'r1603'
              else 'reversible_intervention_4201')
    folder = BASE / source
    gate = json.loads((folder / 'cpu_gate.json').read_text())
    names = list(gate['prior_sha256']) + [source + '/' + name for name in
        ('topology_bank.jsonl', 'hard_bank.jsonl',
         'clean_train.jsonl', 'broken_train.jsonl')]
    occupied = set()
    used_words = set()
    digests = {}
    for name in names:
        path = BASE / name
        digests[name] = sha(path)
        for row in rows(path):
            root = tuple(row['root'])
            occupied.add(root)
            occupied.update(reference(root, row['word'], world))
            if len(set(row['word'])) == 5:
                used_words.add(row['word'])
    return occupied, used_words, digests


def draw_word(rng, length):
    while True:
        word = ''.join(rng.choice('ABCDE') for _ in range(length))
        if len(set(word)) == 5 and all(a != b for a, b in zip(word, word[1:])):
            return word


def build_world(world, occupied, used_words, rng):
    bank = []
    for length in (7, 8):
        for item in range(128):
            for attempt in range(200000):
                word = draw_word(rng, length)
                if word in used_words:
                    continue
                root = tuple(rng.randrange(8) for _ in range(8))
                if root in occupied:
                    continue
                trace = valid(root, word, world)
                if trace is None or trace[-1] in reachable4(root, world):
                    continue
                states = {root, *trace}
                if states & occupied:
                    continue
                k = 3 if (item + length) % 2 == 0 else 4
                true_current = trace[k - 1]
                true_next = trace[k]
                for _ in range(10000):
                    edit = tuple(rng.randrange(8) for _ in range(8))
                    if edit in occupied or edit in states:
                        continue
                    if sum(a != b for a, b in zip(edit, true_current)) < 5:
                        continue
                    edited_suffix = reference(edit, word[k:], world)
                    edited_next = edited_suffix[0]
                    if edited_next in (edit, true_next) or edited_next in occupied:
                        continue
                    if len({edit, *edited_suffix}) != len(edited_suffix) + 1:
                        continue
                    if set(edited_suffix) & (occupied | states):
                        continue
                    break
                else:
                    raise RuntimeError(('counterfactual exhausted', world, length, item))
                prefix_true = answer(trace[:k]) + ' | '
                prefix_edit = answer(trace[:k - 1] + [edit]) + ' | '
                assert len(prefix_true) == len(prefix_edit)
                assert prefix_true[:-len(answer([true_current])) - 3] == \
                       prefix_edit[:-len(answer([edit])) - 3]
                occupied.update(states)
                occupied.add(edit)
                occupied.update(edited_suffix)
                used_words.add(word)
                bank.append({'id': f'cf4301:{world}:{length}:{item}',
                    'world': world, 'root': root, 'word': word, 'length': length,
                    'k': k, 'prompt': prompt(root, word),
                    'prefix_true': prefix_true, 'prefix_edit': prefix_edit,
                    'true_current': true_current, 'edit_current': edit,
                    'true_suffix': answer(trace[k:]),
                    'edit_suffix': answer(edited_suffix),
                    'true_next': true_next, 'edit_next': edited_next})
                break
            else:
                raise RuntimeError(('program exhausted', world, length, item))
    assert len(bank) == 256
    return bank


def main():
    assert not (HERE / 'manifest.json').exists()
    files = {}
    for world, seed in (('r1603', 430120261119), ('r4004', 430120261120)):
        occupied, used_words, sources = prior(world)
        before = len(occupied)
        bank = build_world(world, occupied, used_words, random.Random(seed))
        path = HERE / f'bank_{world}.jsonl'
        path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n'
                                for row in bank), encoding='utf-8')
        files[path.name] = sha(path)
        (HERE / f'cpu_gate_{world}.json').write_text(json.dumps({
            'status': 'PASS_FRESH_COUNTERFACTUAL_BANK', 'world': world,
            'rows': len(bank), 'prior_states': before,
            'total_states': len(occupied), 'source_sha256': sources,
            'bank_sha256': files[path.name]}, indent=2))
    names = ('PROTOCOL.md', 'build.py', 'bank_r1603.jsonl',
             'bank_r4004.jsonl', 'cpu_gate_r1603.json',
             'cpu_gate_r4004.json')
    result = {'status': 'FROZEN_BEFORE_R4301_MODEL_QUERY',
              'files': {name: sha(HERE / name) for name in names}}
    (HERE / 'manifest.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'status': result['status'],
                      'manifest_sha256': sha(HERE / 'manifest.json'),
                      'bank_sha256': files}))


if __name__ == '__main__':
    main()
