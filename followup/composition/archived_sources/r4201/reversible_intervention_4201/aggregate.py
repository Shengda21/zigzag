"""Apply locked reversible-intervention gates to independently scored probes."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'topological_depth_3701'))
import topology as topo

THETAS = tuple(x / 100 for x in range(60, 91, 5))


def load(arm, step=0):
    name = 'probe_baseline' if arm == 'baseline' else f'probe_{arm}_{step:04d}'
    result = json.loads((HERE / name / 'analysis.json').read_text())
    assert result['status'] == 'COMPLETE_R3803_INDEPENDENT_ANALYSIS'
    return result


def grid(flags):
    return sum(flags) >= 5 and any(all(flags[i:i + 3]) for i in range(5))


def masks(old, new, mode):
    flags = []
    for theta in THETAS:
        a = topo.closed_mask(old, theta)
        b = topo.closed_mask(new, theta)
        if mode == 'loss':
            flags.append(topo.chain_data(a, 2)[2] > 0 and
                         topo.chain_data(b, 2)[2] == 0)
        else:
            flags.append(topo.bridge(a, b, 2)['new_checkpoint_supported'] > 0)
    return flags


def bootstrap(old, new, mode, seed):
    rng = np.random.default_rng(seed)
    passes = 0
    for _ in range(1000):
        indices = {cell: rng.integers(0, 128, 128) for cell in topo.CELLS}
        old_scores = {cell: float(np.mean(np.array(old['item_flags'][cell])[indices[cell]]))
                      for cell in topo.CELLS}
        new_scores = {cell: float(np.mean(np.array(new['item_flags'][cell])[indices[cell]]))
                      for cell in topo.CELLS}
        passes += grid(masks(old_scores, new_scores, mode))
    return passes / 1000


def paired(clean, corrupt, length, seed):
    a = np.array(clean['hard']['pair_flags'][str(length)], dtype=np.int8)
    b = np.array(corrupt['hard']['pair_flags'][str(length)], dtype=np.int8)
    assert a.shape == b.shape == (256,)
    delta = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 256, size=(10000, 256))
    replicates = delta[indices].mean(axis=1)
    return {'difference': float(delta.mean()),
            'bonferroni_lower': float(np.percentile(replicates, 2.5)),
            'clean_only': int(((a == 1) & (b == 0)).sum()),
            'corrupt_only': int(((a == 0) & (b == 1)).sum())}


def format_eos(result):
    blocks = (result['topology']['program_counts'], result['hard']['program_counts'])
    assert sum(g['total'] for block in blocks for g in block.values()) == 6784
    return sum(g['format_eos'] for block in blocks for g in block.values()) / 6784


def summary(arm, step, result):
    scores = result['topology']['scores']
    mask = topo.closed_mask(scores, .8)
    return {'arm': arm, 'step': step,
            'vertex_min': result['topology']['minimum_vertex'],
            'edge_min': result['topology']['minimum_edge'],
            'face_min': result['topology']['minimum_face'],
            'theta80_betti': [topo.chain_data(mask, k)[2] for k in range(3)],
            'strict_pairs_7': result['hard']['strict_pairs']['7'],
            'strict_pairs_8': result['hard']['strict_pairs']['8'],
            'format_eos': format_eos(result)}


def main():
    assert topo.fixtures()
    baseline = load('baseline')
    corrupt = {step: load('corrupt', step) for step in (32, 64, 96, 128)}
    recover = {step: load('recover', step) for step in (32, 64, 96, 128)}
    clean1, clean2 = load('clean1', 128), load('clean2', 128)
    loss_flags = masks(baseline['topology']['scores'], corrupt[128]['topology']['scores'], 'loss')
    gain_flags = masks(corrupt[128]['topology']['scores'], recover[128]['topology']['scores'], 'gain')
    loss_support = bootstrap(baseline['topology'], corrupt[128]['topology'],
                             'loss', 420120261122)
    gain_support = bootstrap(corrupt[128]['topology'], recover[128]['topology'],
                             'gain', 420120261123)
    pairwise = {str(length): paired(clean1, corrupt[128], length,
                                   420120261124 + length) for length in (7, 8)}
    table = [summary('baseline', 0, baseline)]
    table += [summary('corrupt', step, corrupt[step]) for step in (32, 64, 96, 128)]
    table += [summary('clean1', 128, clean1)]
    table += [summary('recover', step, recover[step]) for step in (32, 64, 96, 128)]
    table += [summary('clean2', 256, clean2)]
    gates = {
        'baseline_eligible': baseline['baseline_gate']['pass'],
        'H2_grid_loss': grid(loss_flags),
        'H2_loss_bootstrap': loss_support >= .95,
        'corrupt_transfer_below_205': min(corrupt[128]['hard']['strict_pairs'].values()) < 205,
        'paired_superiority_7': pairwise['7']['bonferroni_lower'] > 0,
        'paired_superiority_8': pairwise['8']['bonferroni_lower'] > 0,
        'corrupt_unary_retained': corrupt[128]['topology']['minimum_vertex'] >= .95,
        'corrupt_binary_retained': corrupt[128]['topology']['minimum_edge'] >= .85,
        'H2_grid_recovery': grid(gain_flags),
        'H2_recovery_bootstrap': gain_support >= .95,
        'recovery_transfer_7': recover[128]['hard']['strict_pairs']['7'] >= 205,
        'recovery_transfer_8': recover[128]['hard']['strict_pairs']['8'] >= 205,
        'clean1_retains_H2': topo.chain_data(topo.closed_mask(clean1['topology']['scores'], .8), 2)[2] > 0,
        'clean2_retains_H2': topo.chain_data(topo.closed_mask(clean2['topology']['scores'], .8), 2)[2] > 0,
        'clean1_transfer': min(clean1['hard']['strict_pairs'].values()) >= 205,
        'clean2_transfer': min(clean2['hard']['strict_pairs'].values()) >= 205,
    }
    result = {'status': 'COMPLETE_R4201_PREDECLARED_INTERVENTION_GATE',
              'table': table, 'loss_flags': loss_flags,
              'loss_bootstrap': loss_support, 'gain_flags': gain_flags,
              'gain_bootstrap': gain_support, 'paired': pairwise,
              'gates': gates, 'joint_pass': all(gates.values())}
    (HERE / 'AGGREGATE.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'joint_pass': result['joint_pass'],
                      'gates': gates, 'paired': pairwise,
                      'loss_bootstrap': loss_support,
                      'gain_bootstrap': gain_support, 'table': table}))


if __name__ == '__main__':
    main()
