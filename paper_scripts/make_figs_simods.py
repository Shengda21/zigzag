"""Figures for the SIMODS manuscript (English labels). Writes figures/*.pdf and *.png.

Inputs (read-only):
  zigzag_v5/cloud_return_round4/analysis_r4/forget3_confirm2_SFT/world637_S1.robust.json   scores + historical-union bars
  zigzag_v5/paper_sec4/mv_counts.json       union / intersection H1 bars (K_B, theta = 0.8), transition classes
  zigzag_v5/paper_sec4/theta_sweep.json     fine threshold sweep of the K_B union H1 barcode, robust counts
  SIMODS/forget3_summary.json               cross-arm robust-event counts
  zigzag_v5/cloud_return_20260923/analysis/crosslingual/world50*/analysis.json
  zigzag_v5/cloud_return_round4/analysis_r4/grokking_confirm_posthoc/*.robust.json
  Pythia: env ZZ_PYTHIA_ROBUST (robust.json) and ZZ_PYTHIA_H0_JSON (pythia_h0_bars json); defaults = round-4 files.

  python make_figs_simods.py
"""
from __future__ import annotations
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = Path(__file__).resolve().parent
V5 = HERE.parent          # repository root (mirrors the zigzag_v5 layout)
sys.path.insert(0, str(V5 / 'crosslingual' / 'code'))
import cw_zigzag as CW            # noqa: E402

OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'], 'font.size': 8,
                     'pdf.fonttype': 42, 'axes.titlesize': 8, 'legend.fontsize': 7})
A4 = V5 / 'cloud_return_round4' / 'analysis_r4'
PURPLE, TEAL, GREY, ORANGE, RED, BLUE, GREEN = '#6a3d9a', '#1b9e77', '#8c8c8c', '#e6550d', '#b22222', '#1f5fa8', '#2e8b57'
B_ORIG = ('B1B2', 'B2B3', 'B1B3', 'B3B4')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    plt.close(fig)


def node_pos(steps, j, log=False):
    i = j // 2
    if j % 2 == 0:
        return steps[i]
    return (steps[i] * steps[i + 1]) ** .5 if log else (steps[i] + steps[i + 1]) / 2


def draw_bars(ax, steps, bars, y0, colour, dy=0.16, lw=3.0, bridge_colour=None, log=False, label=None, min_len=5):
    """bars: list of (start_node, end_node). Bridge-only bars drawn as short ticks in bridge_colour."""
    for n, (s, e) in enumerate(sorted(bars)):
        x0, x1 = node_pos(steps, s, log), node_pos(steps, e, log)
        y = y0 + dy * n
        if s == e and s % 2 == 1 and bridge_colour:
            ax.plot([x0 - min_len / 2, x0 + min_len / 2], [y, y], color=bridge_colour, lw=lw, solid_capstyle='butt')
            ax.plot([x0], [y], marker='|', color=bridge_colour, ms=7, mew=1.5)
        else:
            ax.plot([x0, x1 if x1 > x0 else x0 + min_len], [y, y], color=colour, lw=lw, solid_capstyle='butt',
                    label=label if n == 0 else None)
    return y0 + dy * max(len(bars), 1)


# ------------------------------------------------------------------------------------ Figure: forgetting and recovery
def fig_forget(run='world637_S1'):
    r = json.loads((A4 / 'forget3_confirm2_SFT' / f'{run}.robust.json').read_text(encoding='utf8'))
    mv = json.loads((V5 / 'paper_sec4' / 'mv_counts.json').read_text(encoding='utf8'))
    arm = next(a for a in mv['arms'] if a['run'] == run)['thetas']['0.8']['K_B']
    st, sc = r['steps'], r['scores_point']
    fig, ax = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 0.9, 0.75]))
    a = ax[0]
    a.axvspan(288, 480, color='#f4d6d6', alpha=.85, lw=0, label='replacement transfers taught, originals withheld (288–480)')
    a.axvspan(128, 352, color='#dfe8f5', alpha=.85, lw=0, label='C-group two-cell taught (128–352)')
    for i, e in enumerate(B_ORIG):
        a.plot(st, sc[e], color=RED, lw=1.1, label='original B transfers (4 edges)' if i == 0 else None)
        a.plot(st, sc[e + 'r'], color=BLUE, lw=1.1, ls='--', label='replacement B transfers (4 edges)' if i == 0 else None)
    a.plot(st, sc['fC123'], color=GREEN, lw=1.5, label='C-group two-cell (path consistency)')
    a.axhline(.8, color='k', lw=.6, ls=':')
    a.set_ylabel('cell score $s_t(\\sigma)$')
    a.set_ylim(-.03, 1.03)
    a.legend(ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False)
    b = ax[1]
    y = draw_bars(b, st, arm['union_H1_bars'], 0.0, PURPLE, bridge_colour=ORANGE, label='union bridges')
    y = draw_bars(b, st, arm['intersection_H1_bars'], y + 0.25, TEAL, label='intersection bridges')
    bo = [x for x in arm['union_H1_bars'] if x[0] == x[1]]
    if bo:
        xb = node_pos(st, bo[0][0])
        b.annotate('transition class at the\nbridge 480/496 ($\\rho_1=1$)', xy=(xb, 0.16 * arm['union_H1_bars'].index(bo[0])),
                   xytext=(xb + 60, 0.9), fontsize=7, arrowprops=dict(arrowstyle='-', color=ORANGE, lw=.8), color=ORANGE)
    b.set_yticks([])
    b.set_ylabel('$\\mathrm{H}_1$ bars of $K_B$\n(zigzag, $\\theta=0.8$)', fontsize=7.5)
    b.legend(loc='upper left', frameon=False, ncol=2)
    c = ax[2]
    mono = r['detail']['zigzag']['0.8']['mono_bars']
    draw_bars(c, st, mono, 0.0, GREY, label='historical union $\\widetilde K_t$')
    c.text(10, 0.16 * 1.4, '$\\beta_1=5$: two supported classes\nand three artifact classes', fontsize=7, va='center')
    c.set_yticks([])
    c.set_ylabel('$\\mathrm{H}_1$ bars of $K_B$\n(historical union)', fontsize=7.5)
    c.set_xlabel('training step')
    c.legend(loc='upper left', frameon=False)
    save(fig, 'fig_forget')


# ------------------------------------------------------------------------------------ Figure: threshold sweep and robust counts
def fig_sweep(run='world637_S1'):
    sw = json.loads((V5 / 'paper_sec4' / 'theta_sweep.json').read_text(encoding='utf8'))
    arm = next(a for a in sw if a['run'] == run)
    st = arm['steps']
    summ = json.loads((HERE / 'forget3_summary.json').read_text(encoding='utf8'))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.9), gridspec_kw=dict(width_ratios=[1.45, 1]))
    a = ax[0]
    for th, bars in arm['sweep_KB_H1'].items():
        y = float(th)
        for s, e in bars:
            x0, x1 = node_pos(st, s), node_pos(st, e)
            if s == e and s % 2 == 1:
                a.plot([x0], [y], marker='|', color=ORANGE, ms=4, mew=1.2)
            else:
                a.plot([x0, x1 if x1 > x0 else x0 + 5], [y, y], color=PURPLE, lw=1.6, solid_capstyle='butt')
    for d, ls in ((0.05, '--'), (0.10, ':')):
        a.axhline(0.8 + d, color='k', lw=.5, ls=ls)
        a.axhline(0.8 - d, color='k', lw=.5, ls=ls)
    a.axhline(0.8, color='k', lw=.8)
    a.set_xlabel('training step')
    a.set_ylabel('threshold $\\theta$')
    a.set_ylim(0.49, 0.96)
    a.plot([], [], color=PURPLE, lw=1.6, label='$\\mathrm{H}_1$ bar of $K_B$ (union zigzag)')
    a.plot([], [], marker='|', color=ORANGE, ls='none', label='transition class')
    a.plot([], [], color='k', lw=.5, ls='--', label='$\\theta=0.8\\pm0.05$')
    a.plot([], [], color='k', lw=.5, ls=':', label='$\\theta=0.8\\pm0.10$')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=6.5, ncol=2)
    b = ax[1]
    names = [('death of the original loop (forgetting, 288-352)', 'death of original loop (288–352)', RED, '-'),
             ('death of the replacement loop (480-512)', 'death of replacement loop (480–512)', BLUE, '-'),
             ('birth of the recovered loop (>=496)', 'birth of recovered loop ($\\geq 496$)', RED, '--'),
             ('birth of the original loop (learning, <=288)', 'birth of original loop ($\\leq 288$)', RED, ':'),
             ('birth of the replacement loop (336-416)', 'birth of replacement loop (336–416)', BLUE, ':'),
             ('transition class at bridge ending 496', 'transition class (480/496)', ORANGE, '-')]
    deltas = [0.0, 0.02, 0.05, 0.1]
    for key, lab, col, ls in names:
        ev = summ['events'].get(key, {})
        ys = [len(ev.get(str(d), [])) for d in deltas]
        b.plot(deltas, ys, color=col, ls=ls, marker='o', ms=3.5, lw=1.2, label=lab)
    b.set_xlabel('window half-width $\\delta$')
    b.set_ylabel('arms (of 6) with a $\\delta$-robust event')
    b.set_xticks(deltas)
    b.set_yticks(range(0, 7))
    b.set_ylim(-0.3, 6.3)
    b.legend(frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)
    save(fig, 'fig_sweep')


# ------------------------------------------------------------------------------------ Figure: count versus organisation
def fig_count():
    cx = CW.crosslingual_complex()
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for f in sorted(glob.glob(str(V5 / 'cloud_return_20260923/analysis/crosslingual/world50*/analysis.json'))):
        a = json.loads(Path(f).read_text(encoding='utf8'))
        st, tr = a['steps'], a['trajectory']
        W, b0, b1 = [], [], []
        for i in range(len(st)):
            m = cx.mask_from_scores({c: tr[c][i] for c in cx.names}, .8)
            bb = (cx.betti(m) + [0, 0])[:2]
            W.append(len(cx.cells_in(m)))
            b0.append(bb[0])
            b1.append(bb[1])
        col = BLUE if 'MULTI' in f else '#d95f02'
        ax[0].plot(st, W, color=col, lw=1, alpha=.85)
        ax[1].step(st, b0, where='post', color=col, lw=1, alpha=.85)
        ax[1].step(st, np.array(b1) + .06, where='post', color=col, lw=1, alpha=.85, ls='--')
    ax[0].set_ylabel('number of cells $|K_t(0.8)|$')
    ax[1].set_ylabel('$\\beta_0$ (solid), $\\beta_1$ (dashed)')
    for x in ax:
        x.set_xlabel('training step')
    ax[0].plot([], [], color=BLUE, label='general data multilingual')
    ax[0].plot([], [], color='#d95f02', label='general data English only')
    ax[0].legend(frameon=False)
    save(fig, 'fig_count')


# ------------------------------------------------------------------------------------ Figure: grokking birth steps
def fig_grok():
    fam = {'P_S': GREEN, 'P_D': GREEN, 'Pn_D': GREEN, 'Ps_S': GREEN, 'P_M': '#7570b3', 'P_Q': '#7570b3', 'Pi_Q': '#7570b3'}
    PRIME = {'705': 101, '706': 109}
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    runs = sorted(glob.glob(str(A4 / 'grokking_confirm_posthoc' / '*.robust.json')))
    labels = []
    for y, f in enumerate(runs):
        r = json.loads(Path(f).read_text(encoding='utf8'))
        for k, (e, s) in enumerate(r['detail']['birth_step_08'].items()):
            ax.scatter(s, y + (k - 3) * .06, color=fam[e], s=14, zorder=3)
        stem = Path(f).stem.replace('.robust', '')
        w, seed = stem.replace('MODP_world', '').split('_seed')
        labels.append(f'$p={PRIME[w]}$, seed {seed}')
    ax.set_yticks(range(len(runs)), labels, fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('training step at which the held-out accuracy of the operation first stays above 0.8')
    for lab, col in (('additive family (4 operations)', GREEN), ('multiplicative family (3 operations)', '#7570b3')):
        ax.scatter([], [], color=col, s=14, label=lab)
    ax.set_xticks([3000, 5000, 10000, 20000, 30000], ['3k', '5k', '10k', '20k', '30k'])
    ax.set_xlim(2500, 40000)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    save(fig, 'fig_grok')


# ------------------------------------------------------------------------------------ Figure: Pythia public checkpoints
def fig_pythia():
    rob = Path(os.environ['ZZ_PYTHIA_ROBUST']) if os.environ.get('ZZ_PYTHIA_ROBUST') else A4 / 'pythia_runs' / 'pythia-1.4b.robust.json'
    hbp = Path(os.environ['ZZ_PYTHIA_H0_JSON']) if os.environ.get('ZZ_PYTHIA_H0_JSON') else V5 / 'paper_sec4' / 'pythia_h0_bars.json'
    r = json.loads(rob.read_text(encoding='utf8'))
    hb = json.loads(hbp.read_text(encoding='utf8'))
    st, sc = r['steps'], r['scores_point']
    assert hb['steps'] == st, 'robust.json and h0 bars must come from the same checkpoint grid'
    names = {'e_iso_us': 'date: 2021-03-05 $\\to$ 03/05/2021',
             'e_iso_text': 'date: 2021-03-05 $\\to$ March 5, 2021',
             'e_us_text': 'date: 03/05/2021 $\\to$ March 5, 2021',
             'face_iso_us_text': 'date: two-cell (path consistency)',
             'e_word_upper': 'word $\\to$ upper case',
             'e_caesar3_caesar3upper': 'Caesar-shifted word $\\to$ upper case',
             'e_reverse_upperreverse': 'reversed word $\\to$ upper case'}
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    a = ax[0]
    for c, lab in names.items():
        if c in sc:
            a.plot(st, sc[c], marker='o', ms=2.0, lw=1.0, ls='--' if c.startswith('face') else '-', label=lab)
    for th in (.6, .8):
        a.axhline(th, color='k', lw=.5, ls=':')
    a.set_xscale('log')
    a.set_ylabel('cell score')
    a.legend(ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0))
    b = ax[1]
    zz = sorted(hb['zigzag'], key=lambda d: (d['start_node'], d['end_node']))
    mo = sorted(hb['mono'], key=lambda d: (d['start_node'], d['end_node']))

    def draw(bar_list, y0, colour_fn):
        y = y0
        for d in bar_list:
            x0 = node_pos(st, d['start_node'], log=True)
            x1 = node_pos(st, d['end_node'], log=True) if d['end_node'] > d['start_node'] else x0 * 1.08
            b.plot([x0, x1], [y, y], color=colour_fn(d), lw=2.2, solid_capstyle='butt')
            y += 1
        return y
    y = draw(mo, 0.0, lambda d: GREY)
    sup = [d['support'] for d in zz if d['finite'] and not d['in_mono']]
    draw(zz, y + 1.5, lambda d: ORANGE if (d['finite'] and not d['in_mono']) else PURPLE)
    b.set_yticks([])
    b.set_xscale('log')
    b.set_xlabel('Pythia-1.4B pretraining step')
    b.set_ylabel('$\\mathrm{H}_0$ bars ($\\theta=0.6$)')
    nz, nm = hb['counts']['zigzag_finite'], hb['counts']['mono_finite']
    b.plot([], [], color=PURPLE, lw=2.2, label=f'union zigzag ({nz} finite bars)')
    if sup:
        b.plot([], [], color=ORANGE, lw=2.2, label=f'of which {len(sup)} from a score briefly below $\\theta$ (bootstrap support {min(sup):.2f}–{max(sup):.2f})')
    b.plot([], [], color=GREY, lw=2.2, label=f'historical union ({nm} finite bars)')
    b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=1)
    save(fig, 'fig_pythia')


if __name__ == '__main__':
    fig_forget()
    fig_sweep()
    fig_count()
    fig_grok()
    fig_pythia()
    print('figures ->', OUT)
