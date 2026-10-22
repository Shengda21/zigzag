"""Figures for the FORGET3 numerical example (paper §4.1).

  python fig_forget3.py --robust <dir with *.robust.json> --runs <run dir root> --arm world635_S1 --out <dir>

Fig. A (one arm): (a) cell scores of the original B edges, the replacement edges and fC123 with the
curriculum windows; (b) H1 barcode of K_t(0.8) cap K_B, union-bridged zigzag vs union-monotonised
ordinary persistence.  Fig. B (all arms): bootstrap support of each criterion and the S3 p-value.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

B_ORIG = ('B1B2', 'B2B3', 'B1B3', 'B3B4')
CRIT = [('c1_teach', '1 teach'), ('c2_death', '2 death'), ('c3_replacement', '3 replacement'), ('c4_recovery', '4 recovery'),
        ('c5_zigzag', '5 zigzag'), ('c7_placebo', '7 placebo'), ('c6_mechanism', 'R5 mechanism')]


def fig_arm(rob, out, F=1):
    st = rob['steps']
    sc = rob.get('scores_point') or {}
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
    a = ax[0]
    for lo, hi, col, lab in ((288 * F, 480 * F, '#f4d6d6', 'B edges retired / replaced'), (128 * F, 352 * F, '#dfe8f5', 'fC123 practised')):
        a.axvspan(lo, hi, color=col, alpha=0.7, lw=0, label=lab)
    for e in B_ORIG:
        a.plot(st, sc[e], color='#b22222', lw=1.1, alpha=0.8, label='original B edges' if e == B_ORIG[0] else None)
        a.plot(st, sc[e + 'r'], color='#1f5fa8', lw=1.1, ls='--', alpha=0.8, label='replacement edges' if e == B_ORIG[0] else None)
    a.plot(st, sc['fC123'], color='#2e8b57', lw=1.4, label='face fC123')
    a.axhline(0.8, color='k', lw=0.6, ls=':')
    a.set_ylabel('cell score $m_t(\\sigma)$')
    a.set_ylim(-0.03, 1.03)
    a.legend(fontsize=7, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    b = ax[1]
    zz = rob['detail']['zigzag']['0.8']

    def node_step(j):
        i = j // 2
        return st[i] if j % 2 == 0 else (st[i] + st[i + 1]) / 2
    for k, (bars, col, y0, lab) in enumerate(((zz['bars'], '#6a3d9a', 1.0, 'zigzag (union bridges)'), (zz['mono_bars'], '#999999', 0.0, 'monotonised $\\widetilde K_t$'))):
        for n, (s, e) in enumerate(sorted(bars)):
            y = y0 + 0.12 * n
            x0, x1 = node_step(s), node_step(e)
            b.plot([x0, x1 if x1 > x0 else x0 + 4 * F], [y, y], color=col, lw=3, solid_capstyle='butt', label=lab if n == 0 else None)
    b.set_yticks([])
    b.set_xlabel('training step')
    b.set_ylabel('$H_1$ bars of $K_t(0.8)\\cap K_B$')
    b.legend(fontsize=7, loc='upper left', frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f"forget3_{rob['run']}.{ext}", dpi=200)
    plt.close(fig)


def fig_support(robs, out, name):
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(robs) + 1.2))
    runs = [r['run'] for r in robs]
    M = [[r['support'][k] if r['point'][k] else 0.0 for k, _ in CRIT] for r in robs]
    im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    for i, r in enumerate(robs):
        for j, (k, _) in enumerate(CRIT):
            ax.text(j, i, ('%.2f' % r['support'][k]) if r['point'][k] else 'x', ha='center', va='center', fontsize=7,
                    color='w' if M[i][j] < 0.6 else 'k')
    ax.set_xticks(range(len(CRIT)), [c for _, c in CRIT], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(runs)), [f"{r}  (S3 p={rb['temporal_S3']['p_lower']:.3f})" if rb['temporal_S3']['p_lower'] is not None else r for r, rb in zip(runs, robs)], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label='bootstrap support (x = point estimate fails)')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--robust', required=True)
    p.add_argument('--arm', action='append', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--scale', type=int, default=1)
    p.add_argument('--name', default='forget3_support')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    robs = [json.loads(f.read_text(encoding='utf8')) for f in sorted(Path(a.robust).glob('*.robust.json'))]
    for r in robs:
        if r['run'] in a.arm:
            fig_arm(r, out, a.scale)
    fig_support([r for r in robs if 'point' in r], out, a.name)
    print('figures written to', out)


if __name__ == '__main__':
    main()
