"""Figures for the paper's examples section, plain-language Chinese legends (v4).

  python make_paper_figures.py      (writes paper_sec4/figures/*.pdf and *.png)

Inputs (all local, already verified returns):
  cloud_return_round4/analysis_r4/forget3_confirm2_SFT/world637_S1.robust.json
  cloud_return_20260923/analysis/crosslingual/world50*/analysis.json
  cloud_return_round4/analysis_r4/grokking_confirm_posthoc/*.robust.json
  cloud_return_round4/analysis_r4/pythia_runs/pythia-1.4b.robust.json
  paper_sec4/pythia_h0_bars.json
"""
from __future__ import annotations
import glob
import os
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

V5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5 / 'crosslingual' / 'code'))
import cw_zigzag as CW            # noqa: E402
import zz_robust as R             # noqa: E402

OUT = V5 / 'paper_sec4' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'font.size': 9, 'pdf.fonttype': 42})
A4 = V5 / 'cloud_return_round4' / 'analysis_r4'
PRIME = {'705': 101, '706': 109}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    plt.close(fig)


def node_pos(steps, j, log=False):
    """position of zigzag node j: even = checkpoint, odd = bridge between two checkpoints"""
    i = j // 2
    if j % 2 == 0:
        return steps[i]
    return (steps[i] * steps[i + 1]) ** .5 if log else (steps[i] + steps[i + 1]) / 2


def fig_forget3(run='world637_S1'):
    r = json.loads((A4 / 'forget3_confirm2_SFT' / f'{run}.robust.json').read_text(encoding='utf8'))
    st, sc = r['steps'], r['scores_point']
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 4.9), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    a = ax[0]
    a.axvspan(288, 480, color='#f4d6d6', alpha=.8, lw=0, label='撤走原规则、改教新规则的时段')
    a.axvspan(128, 352, color='#dfe8f5', alpha=.8, lw=0, label='C 组一致性任务参与训练的时段')
    for i, e in enumerate(R.B_ORIG):
        a.plot(st, sc[e], color='#b22222', lw=1.1, label='B 组原规则（4 条）' if i == 0 else None)
        a.plot(st, sc[e + 'r'], color='#1f5fa8', lw=1.1, ls='--', label='B 组新规则（4 条）' if i == 0 else None)
    a.plot(st, sc['fC123'], color='#2e8b57', lw=1.5, label='C 组三条规则的一致性')
    a.axhline(.8, color='k', lw=.6, ls=':')
    a.set_ylabel('答对率')
    a.set_ylim(-.03, 1.03)
    a.legend(fontsize=7, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.0), frameon=False)
    b = ax[1]
    zz = r['detail']['zigzag']['0.8']
    for bars, col, y0, lab in ((zz['bars'], '#6a3d9a', 1.0, '本文方法'), (zz['mono_bars'], '#8c8c8c', 0.0, '只增不减的累计方式')):
        for n, (s, e) in enumerate(sorted(bars)):
            x0, x1 = node_pos(st, s), node_pos(st, e)
            b.plot([x0, x1 if x1 > x0 else x0 + 5], [y0 + .13 * n] * 2, color=col, lw=3.2, solid_capstyle='butt', label=lab if n == 0 else None)
    b.set_yticks([])
    b.set_xlabel('训练步')
    b.set_ylabel('B 组中每个环的存在时段', fontsize=8)
    b.legend(fontsize=7, loc='upper left', frameon=False)
    save(fig, 'fig_forget3')


def fig_crosslingual():
    cx = CW.crosslingual_complex()
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
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
        col = '#1f5fa8' if 'MULTI' in f else '#d95f02'
        ax[0].plot(st, W, color=col, lw=1, alpha=.8)
        ax[1].step(st, b0, where='post', color=col, lw=1, alpha=.8)
        ax[1].step(st, np.array(b1) + .06, where='post', color=col, lw=1, alpha=.8, ls='--')
    ax[0].set_ylabel('已掌握的项数 $W(t)$')
    ax[1].set_ylabel('块数（实线）、环数（虚线）')
    for x in ax:
        x.set_xlabel('训练步')
    ax[0].plot([], [], color='#1f5fa8', label='通用知识为多语言')
    ax[0].plot([], [], color='#d95f02', label='通用知识仅英文')
    ax[0].legend(fontsize=7, frameon=False)
    save(fig, 'fig_crosslingual')


def fig_grokking():
    fam = {'P_S': '#1b9e77', 'P_D': '#1b9e77', 'Pn_D': '#1b9e77', 'Ps_S': '#1b9e77',
           'P_M': '#7570b3', 'P_Q': '#7570b3', 'Pi_Q': '#7570b3'}
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    runs = sorted(glob.glob(str(A4 / 'grokking_confirm_posthoc' / '*.robust.json')))
    labels = []
    for y, f in enumerate(runs):
        r = json.loads(Path(f).read_text(encoding='utf8'))
        for k, (e, s) in enumerate(r['detail']['birth_step_08'].items()):
            ax.scatter(s, y + (k - 3) * .06, color=fam[e], s=14, zorder=3)
        stem = Path(f).stem.replace('.robust', '')
        w, seed = stem.replace('MODP_world', '').split('_seed')
        labels.append(f'p={PRIME[w]}，种子 {seed}')
    ax.set_yticks(range(len(runs)), labels, fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('该运算在训练时没见过的数对上答对率稳定超过 0.8 的训练步')
    for lab, col in (('加法类运算（4 种）', '#1b9e77'), ('乘法类运算（3 种）', '#7570b3')):
        ax.scatter([], [], color=col, s=14, label=lab)
    ax.set_xticks([3000, 5000, 10000, 20000, 30000], ['3k', '5k', '10k', '20k', '30k'])
    ax.set_xlim(2500, 40000)
    ax.legend(fontsize=7, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2)
    save(fig, 'fig_grokking')


def fig_pythia():
    _rob = Path(os.environ['ZZ_PYTHIA_ROBUST']) if os.environ.get('ZZ_PYTHIA_ROBUST') else A4 / 'pythia_runs' / 'pythia-1.4b.robust.json'
    r = json.loads(_rob.read_text(encoding='utf8'))   # env ZZ_PYTHIA_ROBUST overrides (round 5: 55 checkpoints)
    st, sc = r['steps'], r['scores_point']
    names = {'e_iso_us': '日期 2021-03-05 → 03/05/2021',
             'e_iso_text': '日期 2021-03-05 → March 5, 2021',
             'e_us_text': '日期 03/05/2021 → March 5, 2021',
             'face_iso_us_text': '日期：两条转换路径结果一致',
             'e_word_upper': '单词转大写',
             'e_caesar3_caesar3upper': '字母后移三位的单词转大写',
             'e_reverse_upperreverse': '字母倒序的单词转大写'}
    _hb = Path(os.environ['ZZ_PYTHIA_H0_JSON']) if os.environ.get('ZZ_PYTHIA_H0_JSON') else V5 / 'paper_sec4' / 'pythia_h0_bars.json'
    hb = json.loads(_hb.read_text(encoding='utf8'))
    assert hb['steps'] == st
    fig, ax = plt.subplots(2, 1, figsize=(6.3, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    a = ax[0]
    for c, lab in names.items():
        a.plot(st, sc[c], marker='o', ms=2.2, lw=1.1, ls='--' if c.startswith('face') else '-', label=lab)
    for th in (.6, .8):
        a.axhline(th, color='k', lw=.5, ls=':')
    a.set_xscale('log')
    a.set_ylabel('答对率')
    a.legend(fontsize=7, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.0))
    b = ax[1]
    zz = sorted(hb['zigzag'], key=lambda d: (d['start_node'], d['end_node']))
    mo = sorted(hb['mono'], key=lambda d: (d['start_node'], d['end_node']))

    def draw(bar_list, y0, colour_fn):
        y = y0
        for d in bar_list:
            x0 = node_pos(st, d['start_node'], log=True)
            x1 = node_pos(st, d['end_node'], log=True) if d['end_node'] > d['start_node'] else x0 * 1.08
            b.plot([x0, x1], [y, y], color=colour_fn(d), lw=2.4, solid_capstyle='butt')
            y += 1
        return y
    y = draw(mo, 0.0, lambda d: '#8c8c8c')
    sup = [d['support'] for d in zz if d['finite'] and not d['in_mono']]
    draw(zz, y + 1.5, lambda d: '#e6550d' if (d['finite'] and not d['in_mono']) else '#6a3d9a')
    b.set_yticks([])
    b.set_xscale('log')
    b.set_xlabel('Pythia-1.4B 预训练步')
    b.set_ylabel('每一块的存在时段（标准 0.6）', fontsize=8)
    nz, nm = hb['counts']['zigzag_finite'], hb['counts']['mono_finite']
    b.plot([], [], color='#6a3d9a', lw=2.4, label=f'本文方法（有限时段 {nz} 个）')
    b.plot([], [], color='#e6550d', lw=2.4, label=f'其中 {len(sup)} 个来自某项能力短暂跌破标准（重复抽样中 {round(100 * min(sup))}%–{round(100 * max(sup))}% 出现）')
    b.plot([], [], color='#8c8c8c', lw=2.4, label=f'只增不减的累计方式（有限时段 {nm} 个）')
    b.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    save(fig, 'fig_pythia')


if __name__ == '__main__':
    fig_forget3()
    fig_crosslingual()
    fig_grokking()
    fig_pythia()
    print('figures ->', OUT)
