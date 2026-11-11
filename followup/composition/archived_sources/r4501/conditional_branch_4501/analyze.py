"""Independently score conditional-branch counterfactual continuations."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
STATE = re.compile(r'[0-7](?: [0-7]){7}\Z')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def state_text(state):
    return ' '.join(map(str, state))


def score_world(world, tok):
    bank = rows(HERE / f'bank_{world}.jsonl')
    out = HERE / f'probe_{world}'
    runtime = json.loads((out / 'runtime.json').read_text())
    assert runtime['status'] == 'COMPLETE_R4501_BRANCH_GENERATION'
    assert runtime['manifest_sha256'] == sha(HERE / 'manifest.json')
    predictions = rows(out / 'predictions.jsonl')
    assert len(bank) == 256 and len(predictions) == 512
    seen = set()
    item_scores = {}
    for pred in predictions:
        index, condition = pred['source_index'], pred['condition']
        assert condition in ('true', 'edit')
        row = bank[index]
        assert pred['id'] == row['id']
        key = (index, condition)
        assert key not in seen
        seen.add(key)
        raw = pred['raw_generated_ids']
        stop = next((i for i, token in enumerate(raw)
                     if token in (151645, 151643)), None)
        eos = None if stop is None else raw[stop]
        content = raw if stop is None else raw[:stop]
        decoded = tok.decode(content, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
        assert decoded == pred['prediction']
        assert pred['first_eos_index'] == stop and pred['terminator_id'] == eos
        assert pred['decode_cap_reached'] == (stop is None and len(raw) >= 120)
        actual = decoded.split(' | ')
        expected = row[f'{condition}_suffix'].split(' | ')
        target = state_text(row[f'{condition}_next'])
        original = state_text(row['true_next'])
        assert expected[0] == target
        item_scores[key] = {
            'id': row['id'], 'world': world, 'length': row['length'],
            'k': row['k'], 'condition': condition,
            'branch_direction': row['branch_direction'],
            'first_correct': int(bool(actual) and actual[0] == target),
            'first_original': int(bool(actual) and actual[0] == original),
            'full_prefix': int(len(actual) >= len(expected) and
                               actual[:len(expected)] == expected),
            'strict_suffix': int(eos == 151645 and decoded == row[f'{condition}_suffix']
                                 and not set(tok.all_special_ids).intersection(content)),
            'format_eos': int(eos == 151645 and
                              all(STATE.fullmatch(x) for x in actual)),
            'extra_states': len(actual) - len(expected),
            'first_state': actual[0] if actual else '',
            'target_state': target, 'original_state': original,
        }
    assert len(seen) == 512
    scored = [item_scores[i, c] for i in range(256) for c in ('true', 'edit')]
    (out / 'analysis.jsonl').write_text(''.join(
        json.dumps(x, separators=(',', ':')) + '\n' for x in scored))
    summary = {'world': world,
               'probe_sha256': sha(out / 'predictions.jsonl'),
               'counts': {}}
    for depth in (7, 8, 'active_to_inactive', 'inactive_to_active', 'all'):
        chosen = [i for i in range(256)
                  if depth == 'all' or bank[i]['length'] == depth or
                     bank[i]['branch_direction'] == depth]
        true = [item_scores[i, 'true'] for i in chosen]
        edit = [item_scores[i, 'edit'] for i in chosen]
        delta = np.array([a['first_correct'] - a['first_original']
                          for a in edit], dtype=float)
        seed_offset = {7: 7, 8: 8, 'active_to_inactive': 100,
                       'inactive_to_active': 200, 'all': 0}[depth]
        rng = np.random.default_rng(450120261122 +
                                    (0 if world == 'r1603' else 1000) +
                                    seed_offset)
        samples = rng.integers(0, len(chosen), size=(10000, len(chosen)))
        summary['counts'][str(depth)] = {
            'n': len(chosen),
            'true_first_correct': sum(x['first_correct'] for x in true),
            'edit_first_correct': sum(x['first_correct'] for x in edit),
            'edit_first_original': sum(x['first_original'] for x in edit),
            'true_full_prefix': sum(x['full_prefix'] for x in true),
            'edit_full_prefix': sum(x['full_prefix'] for x in edit),
            'true_strict_suffix': sum(x['strict_suffix'] for x in true),
            'edit_strict_suffix': sum(x['strict_suffix'] for x in edit),
            'true_format_eos': sum(x['format_eos'] for x in true),
            'edit_format_eos': sum(x['format_eos'] for x in edit),
            'edited_minus_original_next': float(delta.mean()),
            'paired_lower_2p5': float(np.percentile(delta[samples].mean(axis=1), 2.5)),
        }
    overall = summary['counts']['all']
    summary['gates'] = {
        'unedited_successor': overall['true_first_correct'] / overall['n'] >= .90,
        'edited_successor': overall['edit_first_correct'] / overall['n'] >= .75,
        'edited_preference': overall['paired_lower_2p5'] > 0,
        'active_to_inactive': summary['counts']['active_to_inactive']['edit_first_correct'] /
                              summary['counts']['active_to_inactive']['n'] >= .70,
        'inactive_to_active': summary['counts']['inactive_to_active']['edit_first_correct'] /
                              summary['counts']['inactive_to_active']['n'] >= .70,
    }
    summary['joint_pass'] = all(summary['gates'].values())
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(HERE / name) == digest, name
    tok = AutoTokenizer.from_pretrained(BASE / 'implicit_operator_trajectory_3101',
                                         local_files_only=True, trust_remote_code=False)
    result = {world: score_world(world, tok) for world in ('r1603', 'r4004')}
    result['cross_world_pass'] = all(result[w]['joint_pass'] for w in
                                     ('r1603', 'r4004'))
    (HERE / 'AGGREGATE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
