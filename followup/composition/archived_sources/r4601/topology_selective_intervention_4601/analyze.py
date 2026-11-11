"""Independently score the prospective r3803 state and transfer probes."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'topological_depth_3701'))
import topology as topo

THETAS = tuple(x / 100 for x in range(60, 91, 5))
STATE = re.compile(r'[0-7](?: [0-7]){7}\Z')


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


def score_predictions(path, bank, tok, kind):
    seen = set()
    item_members = defaultdict(dict)
    program_counts = defaultdict(Counter)
    for pred in rows(path):
        index = pred['source_index']
        row = bank[index]
        assert index not in seen and pred['id'] == row['id']
        seen.add(index)
        raw = pred['raw_generated_ids']
        stop = next((i for i, token in enumerate(raw)
                     if token in (151645, 151643)), None)
        eos = None if stop is None else raw[stop]
        cap = ((40, 55, 180)[row['dimension']] if kind == 'topology' else 180)
        assert pred['first_eos_index'] == stop and pred['terminator_id'] == eos
        assert pred['decode_cap_reached'] == (stop is None and len(raw) >= cap)
        content = raw if stop is None else raw[:stop]
        decoded = tok.decode(content, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
        assert decoded == pred['prediction']
        gold = row['answer'].split(' | ')
        actual = decoded.split(' | ')
        prefix = int(len(actual) >= row['length'] and actual[:row['length']] == gold)
        strict = int(eos == 151645 and decoded == row['answer'] and
                     not set(tok.all_special_ids).intersection(content))
        format_eos = int(eos == 151645 and all(STATE.fullmatch(x) for x in actual))
        group = str(row['dimension'] if kind == 'topology' else row['length'])
        program_counts[group]['total'] += 1
        program_counts[group]['prefix'] += prefix
        program_counts[group]['strict'] += strict
        program_counts[group]['format_eos'] += format_eos
        program_counts[group]['eos'] += int(eos == 151645)
        key = (row['cell'], row['item']) if kind == 'topology' else (row['length'], row['pair_id'])
        assert row['member'] not in item_members[key]
        item_members[key][row['member']] = {'prefix': prefix, 'strict': strict,
                                            'format_eos': format_eos}
    assert len(seen) == len(bank)
    if kind == 'topology':
        cell_flags = {}
        for cell in topo.CELLS:
            expected = {0} if len(cell) == 1 else {0, 1}
            cell_flags[cell] = []
            for item in range(128):
                assert set(item_members[cell, item]) == expected
                cell_flags[cell].append(int(all(v['prefix'] for v in item_members[cell, item].values())))
        scores = {cell: sum(flags) / 128 for cell, flags in cell_flags.items()}
        strict_scores = {}
        for cell in topo.CELLS:
            strict_scores[cell] = sum(int(all(v['strict'] for v in item_members[cell, item].values()))
                                      for item in range(128)) / 128
        return {'program_counts': {k: dict(v) for k, v in program_counts.items()},
                'scores': scores, 'strict_scores': strict_scores,
                'item_flags': cell_flags,
                'minimum_vertex': min(scores[c] for c in topo.VERTICES),
                'minimum_edge': min(scores[c] for c in topo.EDGES),
                'minimum_face': min(scores[c] for c in topo.FACES)}
    pair_flags = {}
    for length in (7, 8):
        pair_flags[str(length)] = []
        for item in range(256):
            assert set(item_members[length, item]) == {0, 1}
            pair_flags[str(length)].append(int(all(v['strict'] for v in item_members[length, item].values())))
    return {'program_counts': {k: dict(v) for k, v in program_counts.items()},
            'pair_flags': pair_flags,
            'strict_pairs': {length: sum(flags) for length, flags in pair_flags.items()}}


def grid(flags):
    return sum(flags) >= 5 and any(all(flags[i:i + 3]) for i in range(5))


def source_eligibility(topo_score):
    mask = topo.closed_mask(topo_score['scores'], .8)
    betti = [topo.chain_data(mask, k)[2] for k in range(3)]
    return {'pass': topo_score['minimum_vertex'] >= .95 and
                    topo_score['minimum_edge'] >= .85 and betti[2] == 0,
            'minimum_vertex': topo_score['minimum_vertex'],
            'minimum_edge': topo_score['minimum_edge'],
            'theta80_betti': betti, 'theta80_cells': mask.bit_count()}


def event(old, new):
    path = []
    for theta in THETAS:
        a = topo.closed_mask(old['scores'], theta)
        b = topo.closed_mask(new['scores'], theta)
        path.append({'theta': theta,
                     'source_betti': [topo.chain_data(a, k)[2] for k in range(3)],
                     'arm_betti': [topo.chain_data(b, k)[2] for k in range(3)],
                     'source_cells': a.bit_count(), 'arm_cells': b.bit_count(),
                     'degree2': topo.bridge(a, b, 2)})
    return {'thresholds': path,
            'H2_new_support_grid': grid([x['degree2']['new_checkpoint_supported'] > 0
                                         for x in path])}


def bootstrap_event(old, new, seed):
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(1000):
        indices = {cell: rng.integers(0, 128, size=128) for cell in topo.CELLS}
        old_scores = {cell: float(np.mean(np.array(old['item_flags'][cell])[indices[cell]]))
                      for cell in topo.CELLS}
        new_scores = {cell: float(np.mean(np.array(new['item_flags'][cell])[indices[cell]]))
                      for cell in topo.CELLS}
        flags = []
        for theta in THETAS:
            a = topo.closed_mask(old_scores, theta)
            b = topo.closed_mask(new_scores, theta)
            flags.append(topo.bridge(a, b, 2)['new_checkpoint_supported'] > 0)
        passes += grid(flags)
    return passes / 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('baseline', 'cycle', 'acyclic'), required=True)
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()
    assert topo.fixtures()
    frozen = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in frozen['files'].items():
        assert sha(HERE / name) == digest, name
    folder = HERE / ('probe_baseline' if args.arm == 'baseline'
                     else f'probe_{args.arm}_{args.step:04d}')
    runtime = json.loads((folder / 'runtime.json').read_text())
    assert runtime['status'] == ('COMPLETE_R3803_PROBE' if args.arm == 'baseline' else 'COMPLETE_R4601_PROBE')
    if args.arm == 'baseline':
        assert runtime['adapter_sha256'] == frozen['source_adapter_sha256']
    assert runtime['arm'] == args.arm and runtime['step'] == args.step
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct',
                                         local_files_only=True, trust_remote_code=False)
    topo_bank = list(rows(HERE / 'topology_bank.jsonl'))
    hard_bank = list(rows(HERE / 'hard_bank.jsonl'))
    top = score_predictions(folder / 'predictions_topology.jsonl', topo_bank, tok, 'topology')
    hard = score_predictions(folder / 'predictions_hard.jsonl', hard_bank, tok, 'hard')
    result = {'status': 'COMPLETE_R4601_INDEPENDENT_ANALYSIS',
              'arm': args.arm, 'step': args.step,
              'probe_sha256': {name: sha(folder / name) for name in
                               ('predictions_topology.jsonl', 'predictions_hard.jsonl', 'runtime.json')},
              'topology': top, 'hard': hard}
    if args.arm == 'baseline':
        mask = topo.closed_mask(top['scores'], .8)
        betti = [topo.chain_data(mask, k)[2] for k in range(3)]
        result['baseline_gate'] = {
            'pass': top['minimum_vertex'] >= .95 and
                    top['minimum_edge'] >= .85 and betti[2] > 0 and
                    hard['strict_pairs']['7'] >= 205 and
                    hard['strict_pairs']['8'] >= 205,
            'theta80_betti': betti,
            'minimum_vertex': top['minimum_vertex'],
            'minimum_edge': top['minimum_edge'],
            'strict_pairs': hard['strict_pairs']}
    else:
        source = json.loads((HERE / 'probe_baseline' / 'analysis.json').read_text())
        result['event'] = event(source['topology'], top)
        result['bootstrap_H2'] = bootstrap_event(source['topology'], top,
                                                 380320261113 + args.step +
                                                 (0 if args.arm == 'shell' else 1000))
    (folder / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'arm': args.arm, 'step': args.step,
        'baseline_gate': result.get('baseline_gate'),
        'minimum_vertex': top['minimum_vertex'],
        'minimum_edge': top['minimum_edge'],
        'minimum_face': top['minimum_face'],
        'hard_pairs': hard['strict_pairs'],
        'event': None if args.arm == 'baseline' else {
            'H2_grid': result['event']['H2_new_support_grid'],
            'bootstrap': result['bootstrap_H2'],
            'theta80': next(x for x in result['event']['thresholds'] if x['theta'] == .8)}}))


if __name__ == '__main__':
    main()
