"""Freeze a single-face forgetting gate and matched four-operator probes."""
import hashlib
import itertools as it
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from reversible_intervention_4103 import build as old
from implicit_operator_trajectory_3101.rules import reference
from topological_depth_3701.build import make_item, VERTICES, EDGES, FACES
from essential_depth_3600.build import valid

WORLD = 'r1603'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def save(name, records):
    with (HERE / name).open('w', encoding='utf8') as f:
        for row in records:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    return sha(HERE / name)


def prior_states():
    occupied, inputs = old.occupied_prior()
    for name in ('reversible_intervention_4103/topology_bank.jsonl',
                 'reversible_intervention_4103/hard_bank.jsonl',
                 'reversible_intervention_4103/clean_train.jsonl',
                 'topology_selective_intervention_4601/train_clean.jsonl',
                 'counterfactual_state_use_4301/bank_r1603.jsonl',
                 'conditional_branch_4501/bank_r1603.jsonl'):
        p = BASE / name
        inputs[name] = sha(p)
        for row in old.rows(p):
            root = tuple(row['root'])
            occupied.update((root, *reference(root, row['word'], WORLD)))
    return occupied, inputs


def topology(occupied):
    rng = random.Random(470120261110)
    seen = set()
    bank = []
    for dim, cells in enumerate((VERTICES, EDGES, FACES)):
        for cell in cells:
            for item in range(128):
                root, words, traces, _ = make_item(rng, WORLD, cell, dim, occupied, seen)
                for member, (word, trace) in enumerate(zip(words, traces)):
                    bank.append({'id': f'topo4701:{cell}:{item}:{member}',
                        'world': WORLD, 'root': root, 'word': word, 'cell': cell,
                        'dimension': dim, 'item': item, 'member': member,
                        'length': len(word), 'prompt': old.prompt(root, word),
                        'answer': old.answer(trace)})
    occupied.update(seen)
    assert len(bank) == 5760
    return save('topology_bank.jsonl', bank)


def short3(root):
    reached = {root}
    frontier = {root}
    for _ in range(3):
        new = {reference(s, op, WORLD)[0] for s in frontier for op in 'ABCDE'}
        frontier = new - reached
        reached.update(new)
    return reached


def four_bank(occupied):
    rng = random.Random(470120261111)
    cells = tuple(map(''.join, it.combinations('ABCDE', 4)))
    bank = []
    used_words = {c: set() for c in cells}
    for item in range(128):
        for attempt in range(100000):
            root = tuple(rng.randrange(8) for _ in range(8))
            if root in occupied:
                continue
            short = short3(root)
            records = []
            states = {root}
            for cell in cells:
                pair = []
                for trial in range(400):
                    word = ''.join(rng.choice(cell) for _ in range(8))
                    if set(word) != set(cell) or word in used_words[cell] or any(a == b for a,b in zip(word,word[1:])):
                        continue
                    if pair and (word == pair[0][0] or word[0] == pair[0][0][0]):
                        continue
                    trace = valid(root, word, WORLD)
                    if trace is None or trace[-1] in short or set(trace) & occupied:
                        continue
                    if pair and trace[-1] == pair[0][1][-1]:
                        continue
                    pair.append((word, trace))
                    if len(pair) == 2:
                        break
                if len(pair) != 2:
                    break
                for member, (word, trace) in enumerate(pair):
                    states.update(trace)
                    records.append({'id': f'four4701:{cell}:{item}:{member}',
                        'world': WORLD, 'root': root, 'word': word, 'cell': cell,
                        'item': item, 'pair_id': item, 'member': member, 'length': 8,
                        'prompt': old.prompt(root, word), 'answer': old.answer(trace)})
            if len(records) == 10:
                occupied.update(states)
                bank.extend(records)
                for row in records:
                    used_words[row['cell']].add(row['word'])
                break
        else:
            raise RuntimeError(('four-bank sampling',item))
    assert len(bank) == 1280
    return save('four_bank.jsonl', bank)


def draw(rng, occupied, word, ident):
    for _ in range(100000):
        root = tuple(rng.randrange(8) for _ in range(8))
        trace = reference(root, word, WORLD)
        if {root, *trace} & occupied:
            continue
        occupied.update((root, *trace))
        return {'id': ident, 'world': WORLD, 'root': root, 'word': word,
                'length': len(word), 'prompt': old.prompt(root, word),
                'answer': old.answer(trace)}
    raise RuntimeError('training sampling')


def training(occupied):
    rng = random.Random(470120261112)
    pool = {}
    for cell in FACES:
        pool[cell] = []
        for j in range(1008):
            length = 7 + j % 2
            while True:
                word = ''.join(rng.choice(cell) for _ in range(length))
                if set(word) == set(cell) and all(a != b for a,b in zip(word,word[1:])):
                    break
            pool[cell].append(draw(rng, occupied, word, f'pool4701:{cell}:{j}'))
    anchors = [draw(rng,occupied,rng.choice('ABCDE'),f'anchor4701:u:{j}') for j in range(536)]
    anchors += [draw(rng,occupied,''.join(rng.sample('ABCDE',2)),f'anchor4701:b:{j}') for j in range(536)]
    hashes = {}
    for target in FACES:
        bank = [dict(row) for row in pool[target]]
        for cell in FACES:
            if cell != target:
                bank.extend(dict(row) for row in pool[cell][:224])
        bank.extend(dict(row) for row in anchors)
        for length in (7,8):
            indices = [i for i,row in enumerate(bank) if i < 1008 and row['length'] == length]
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            donors = {shuffled[j]: shuffled[(j+1)%len(shuffled)] for j in range(len(shuffled))}
            original = [row['answer'] for row in bank]
            for i in indices:
                a = original[i].split(' | ')[:3]
                b = original[donors[i]].split(' | ')[3:]
                bank[i]['answer'] = ' | '.join(a+b)
                bank[i]['corrupted_target'] = target
                assert bank[i]['answer'] != original[i]
        assert len(bank) == 4096
        hashes[target] = save(f'train_{target}.jsonl',bank)
    save('training_pool.jsonl',[row for cell in FACES for row in pool[cell]]+anchors)
    return hashes


def main():
    assert not (HERE/'manifest.json').exists()
    occupied, inputs = prior_states()
    probes = {'topology':topology(occupied),'four':four_bank(occupied)}
    train = training(occupied)
    gate = {'status':'PASS_R4701_MODEL_BLIND_BANKS','inputs':inputs,
            'probes':probes,'training':train,'rows_per_arm':4096,
            'target_rows':1008,'other_face_rows':2016,'anchor_rows':1072}
    (HERE/'cpu_gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps({'status':gate['status'],'probes':probes,'training_arms':len(train)}))


if __name__ == '__main__':
    main()
