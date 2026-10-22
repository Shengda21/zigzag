"""Zero-GPU counterexample (positioning review, experiment C): on the paper's
own L0 -> L1 -> L2 -> L3 sequence, union- and intersection-monotonised
ordinary persistence lose or fabricate an H1 class, while the zigzag barcode
reports the truth: two distinct H1 intervals [1,1] and [3,3].

  L0 = tree {vF,vD,vP, eFD, eDP}
  L1 = L0 + eFP                (cycle F-D-P born)
  L2 = L0                      (cycle killed by deletion)
  L3 = L0 + {vR, eFR, eRP}     (a different cycle F-D-P-R... born: F-D-P + P-R + R-F)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw_zigzag as CW   # noqa: E402


def mask(cx, names):
    m = 0
    for n in names:
        m |= 1 << cx.index(n)
    assert cx.is_closed(m), names
    return m


def ordinary_persistence_h1(cx, filtration):
    """Ordinary (nested) persistence over F2 via the same generalised-rank
    decomposition; all arrows forward. Returns intervals in checkpoint indices."""
    for a, b in zip(filtration, filtration[1:]):
        assert a & b == a, 'filtration must be nested'
    dims = [len(cx.homology(m, 1)[0]) for m in filtration]
    arrows = [(i, i + 1, cx.induced(filtration[i], filtration[i + 1], 1)) for i in range(len(filtration) - 1)]
    return sorted(CW.decompose(dims, arrows)), dims


def main():
    cx = CW.eleven_cell_complex()
    L0 = mask(cx, ['vF', 'vD', 'vP', 'eFD', 'eDP'])
    L1 = mask(cx, ['vF', 'vD', 'vP', 'eFD', 'eDP', 'eFP'])
    L2 = L0
    L3 = mask(cx, ['vF', 'vD', 'vP', 'vR', 'eFD', 'eDP', 'eFR', 'eRP'])
    seq = (L0, L1, L2, L3)
    truth_betti = [cx.betti(m)[1] for m in seq]
    zz = sorted(cx.barcode(seq, 1, 'union'))
    zz_ck = [[j // 2 for j in range(a, b + 1) if j % 2 == 0] for a, b in zz]
    union = []
    acc = 0
    for m in seq:
        acc |= m
        union.append(acc)
    inter = []
    acc = (1 << len(cx.names)) - 1
    for m in seq:
        acc &= m
        inter.append(acc)
    u_bars, u_dims = ordinary_persistence_h1(cx, union)
    i_bars, i_dims = ordinary_persistence_h1(cx, inter)
    result = dict(
        sequence=dict(L0=cx.cells_in(L0), L1=cx.cells_in(L1), L2=cx.cells_in(L2), L3=cx.cells_in(L3)),
        true_beta1_per_checkpoint=truth_betti,
        zigzag_H1_intervals_checkpoints=zz_ck,
        union_monotonised=dict(beta1=u_dims, H1_intervals=u_bars,
                               reading='the class born at 1 never dies (the deletion at 2 is erased) and the new class at 3 is reported next to a ghost'),
        intersection_monotonised=dict(beta1=i_dims, H1_intervals=i_bars, reading='both classes vanish: intersections never contain a cycle'),
        verdict='zigzag reports two distinct H1 intervals [1,1] and [3,3]; union monotonisation fabricates persistence across the deletion, intersection monotonisation loses both events',
    )
    assert truth_betti == [0, 1, 0, 1]
    assert zz_ck == [[1], [3]], zz_ck
    assert u_dims == [0, 1, 1, 2] and any(b == 3 for a, b in u_bars)
    assert i_dims == [0, 0, 0, 0]
    out = HERE.parent / 'MONOTONIZATION_COUNTEREXAMPLE.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 2.6))
        y = 0
        for label, bars, color in [('zigzag', [(a, b) for a, b in zz_ck and [(c[0], c[-1]) for c in zz_ck]], 'C0'),
                                   ('union-monotonised', u_bars, 'C3'), ('intersection-monotonised', i_bars, 'C2')]:
            if not bars:
                ax.text(0.05, y, f'{label}: no H1 interval', va='center', fontsize=8)
            for a, b in bars:
                ax.hlines(y, a - 0.15, b + 0.15, color=color, lw=6)
                ax.text(b + 0.25, y, label, va='center', fontsize=8)
                y += 1
            y += 1
        ax.set(xlabel='checkpoint (L0..L3)', yticks=[], xticks=range(4), xlim=(-0.5, 5.5), ylim=(-1, y))
        ax.set_title('H1 intervals: zigzag vs monotonised ordinary persistence', fontsize=9)
        fig.tight_layout()
        fig.savefig(HERE.parent / 'MONOTONIZATION_COUNTEREXAMPLE.png', dpi=160)
    except Exception as e:      # noqa: BLE001
        result['figure'] = f'skipped: {e}'
    print(json.dumps(dict(true_beta1=truth_betti, zigzag=zz_ck, union=u_bars, intersection=i_bars)))


if __name__ == '__main__':
    main()
