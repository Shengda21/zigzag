"""Cell scores -> closed subcomplexes -> zigzag barcodes, event classification,
synchrony statistic, permutation null and independent-transfer windows.

Works for two pre-registered designs:
  * crosslingual : K* = G x [en,zh] (33 cells, cw_zigzag.crosslingual_complex);
                   probes in en/zh build the complex, ru/ja are transfer tasks.
  * stage_ab     : the 11-cell theory complex on the target codebook
                   (edges from ATOMIC_EVAL, faces from EVAL_TARGET joint tasks;
                   vertex read-out is not probed in Stage A/B, so vertices are
                   fixed present = 1.0 and the complex is effectively edge/face).

All numbers are development descriptions unless the confirmation design says
otherwise; thresholds, windows and the synchrony rule come from
PREREGISTRATION_CROSSLINGUAL_20260922.md and are inputs here, not tuned.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'stage_ab' / 'code'))
import cw_zigzag as CW        # noqa: E402
import zz_score as Z          # noqa: E402

THETA = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
EDGES = Z.ATOMIC_EDGES
VERTS = ('vF', 'vD', 'vP', 'vR')
FACES = ('fA', 'fB')
MAIN_TEMPLATES = dict(en=(0, 1), zh=(0, 1), ru=(0, 1), ja=(0, 1))
ZH_SET = tuple(f'{e}@zh' for e in EDGES) + tuple(f'{e}@I' for e in EDGES)   # the ten cells the synchrony rule counts
SYNC_MIN_CELLS, SYNC_MIN_THETAS, SYNC_TOL = 6, 5, 1


def rows(p):
    p = Path(p)
    text = gzip.open(p, 'rt', encoding='utf8').read() if p.suffix == '.gz' else p.read_text(encoding='utf8')
    return [json.loads(s) for s in text.splitlines() if s.strip()]


def load_run(run_dir):
    run_dir = Path(run_dir)
    out = {}
    files = sorted(list(run_dir.glob('eval_*.jsonl')) + list(run_dir.glob('eval_*.jsonl.gz')), key=lambda f: int(f.name.split('_')[1].split('.')[0]))
    stride = int(os.environ.get('ZZ_EVAL_STRIDE', '1'))          # coarser checkpoint grid for very long runs (zigzag decomposition is O(n^2))
    for i, f in enumerate(files):
        if i % stride == 0 or i == len(files) - 1:
            out[int(f.name.split('_')[1].split('.')[0])] = rows(f)
    return out


def acc(rs):
    return sum(r['correct'] for r in rs) / len(rs) if rs else 0.0


def core_id(r):
    return r['id'].replace(f"probe_{r['lang']}_{r['template']}_", '', 1)


# ------------------------------------------------------------ cell extractors
def cells_crosslingual(rs, templates=MAIN_TEMPLATES):
    spec = [r for r in rs if r['vocabulary'] == 'spec' and r['lang'] in ('en', 'zh') and r['template'] in templates[r['lang']]]
    by = collections.defaultdict(list)
    for r in spec:
        by[(r['task'], r['lang'])].append(r)
    score = {}
    for lang in ('en', 'zh'):
        for v in VERTS:
            score[f'{v}@{lang}'] = acc(by[(v, lang)])
        for e in EDGES:
            score[f'{e}@{lang}'] = acc(by[(e, lang)])
        for f in FACES:
            qs = by[(f, lang)]
            score[f'{f}@{lang}'] = min(acc([q for q in qs if q['override'] == o]) for o in (False, True)) if qs else 0.0
    # bridge cells: same item (root/task/override) correct in both languages, template index matched
    paired = collections.defaultdict(dict)
    for r in spec:
        paired[(r['task'], core_id(r), r['template'])][r['lang']] = r['correct']
    both = collections.defaultdict(list)
    for (task, cid, t), langs in paired.items():
        if 'en' in langs and 'zh' in langs:
            both[task].append((cid, int(langs['en'] and langs['zh'])))
    for c in VERTS + EDGES:
        vals = [v for _, v in both[c]]
        score[f'{c}@I'] = sum(vals) / len(vals) if vals else 0.0
    for f in FACES:
        per_override = collections.defaultdict(list)
        for cid, v in both[f]:
            per_override[cid.endswith('_1')].append(v)
        score[f'{f}@I'] = min((sum(v) / len(v) for v in per_override.values()), default=0.0) if per_override else 0.0
    return score


def cells_stage_ab(rs, templates=(0, 1)):
    score = {v: 1.0 for v in VERTS}
    base = [r for r in rs if r['vocabulary'] == 'target' and r['kind'] == 'base' and r['template'] in templates]
    for e in EDGES:
        score[e] = acc([r for r in base if r['task'] == e])
    joint = [r for r in rs if r['vocabulary'] == 'target' and r['kind'] == 'joint' and r['template'] in templates]
    for f, d in zip(FACES, 'AB'):
        qs = [r for r in joint if r['domain'] == d]
        score[f] = min(acc([q for q in qs if q['condition'] == c]) for c in ('DEPENDENT', 'INDEPENDENT')) if qs else 0.0
    return score


def transfer_crosslingual(rs):
    out = {}
    for lang in ('ru', 'ja', 'en', 'zh'):
        sp = [r for r in rs if r['vocabulary'] == 'spec' and r['lang'] == lang]
        out[f'{lang}_edges'] = acc([r for r in sp if r['kind'] == 'base' and r['task'] in EDGES])
        out[f'{lang}_joint'] = acc([r for r in sp if r['kind'] == 'joint'])
        out[f'{lang}_repair'] = acc([r for r in sp if r['kind'] == 'repair'])
        out[f'{lang}_parse'] = sum(r['valid_format'] for r in sp) / len(sp) if sp else 0.0
    for lang in ('en', 'zh', 'ru'):
        g = [r for r in rs if r['vocabulary'] == 'general' and r['lang'] == lang]
        out[f'general_{lang}'] = acc(g)
    return out


def transfer_stage_ab(rs):
    rep = [r for r in rs if r['vocabulary'] == 'target' and r['kind'] == 'repair']
    return dict(repair=acc(rep), repair_A=acc([r for r in rep if r['domain'] == 'A']), repair_B=acc([r for r in rep if r['domain'] == 'B']),
                template3_edges=acc([r for r in rs if r['vocabulary'] == 'target' and r['kind'] == 'base' and r['template'] == 3]))


DESIGNS = dict(
    crosslingual=dict(complex=CW.crosslingual_complex, cells=cells_crosslingual, transfer=transfer_crosslingual, sync_set=ZH_SET,
                      transfer_keys=('ru_edges', 'ja_edges', 'ru_joint', 'ja_joint'), baseline_edges=tuple(f'{e}@zh' for e in EDGES),
                      faces=('fA@zh', 'fB@zh', 'fA@en', 'fB@en')),
    stage_ab=dict(complex=CW.eleven_cell_complex, cells=cells_stage_ab, transfer=transfer_stage_ab, sync_set=EDGES + FACES,
                  transfer_keys=('repair', 'repair_A', 'repair_B'), baseline_edges=EDGES, faces=FACES),
)


def design_from_spec(spec_path):
    """Design built from a generated COMPLEX.json (complex_worlds/code/spec_world.py):
    vertex score = read-out accuracy, edge score = conversion accuracy, face score =
    min over override/no-override of joint accuracy; latent cells form the synchrony
    set, taught edges are the baseline, transfer tasks are chains and repairs."""
    spec = json.loads(Path(spec_path).read_text(encoding='utf8'))
    cells = [(n, d, list(b)) for n, d, b in spec['cells']]
    main_t = tuple(spec['main_templates'])
    vocab = spec.get('vocabulary', 'rcn')          # generated worlds tag their probe rows with this vocabulary

    routes = spec.get('face_routes', {}) if spec.get('face_readout') == 'routes' else {}

    def item_key(r):
        k = r.get('item', r.get('root'))
        return json.dumps(k, sort_keys=True) if isinstance(k, (list, dict)) else k

    def route_score(by, tasks):
        # every listed task strictly correct on the same item; an item counts once per (item) after pooling templates
        per = {}
        for t in tasks:
            ok = collections.defaultdict(lambda: True)
            seen = set()
            for r in by[t]:
                k = item_key(r)
                seen.add(k)
                ok[k] = ok[k] and bool(r['correct'])
            per[t] = (seen, ok)
        common = set.intersection(*[seen for seen, _ in per.values()]) if per else set()
        if not common:
            return 0.0
        return sum(all(per[t][1][k] for t in tasks) for k in common) / len(common)

    # vertex_present_mode == 'blanket': 0-cells are present by construction (representations are given), so the
    # closure never depends on a read-out probe; used for from-scratch grokking worlds whose copy probes are untrained.
    blanket = spec.get('vertex_present_mode') == 'blanket' or os.environ.get('ZZ_VERTICES_PRESENT') == '1'

    def cells_fn(rs):
        by = collections.defaultdict(list)
        for r in rs:
            if r.get('vocabulary') == vocab and r.get('template') in main_t:
                by[r['task']].append(r)
        score = {}
        for n, d, _ in cells:
            if d == 0:
                score[n] = 1.0 if blanket else acc(by['v' + n])
            elif d == 1:
                score[n] = acc(by[n])
            elif n in routes:
                score[n] = route_score(by, routes[n])
            else:
                qs = by[n]
                score[n] = min(acc([q for q in qs if q['override'] == o]) for o in (False, True)) if qs else 0.0
        return score

    def transfer_fn(rs):
        out = {t: acc([r for r in rs if r['task'] == t]) for t in spec['transfer_tasks']}
        out['taught_edges_template3'] = acc([r for r in rs if r.get('kind') == 'edge' and r.get('template') == 3 and r.get('cell_kind') == 'taught'])
        out['control_edges'] = acc([r for r in rs if r.get('cell_kind') == 'control'])
        return out
    n_lat = len(spec['latent_cells'])
    return dict(complex=lambda: CW.CWComplex(cells), cells=cells_fn, transfer=transfer_fn, sync_set=tuple(spec['latent_cells']),
                transfer_keys=tuple(spec['transfer_tasks']), baseline_edges=tuple(spec['taught_edges']), faces=tuple(spec['faces']),
                sync_min=max(2, (n_lat + 1) // 2), control_cells=tuple(spec.get('control_cells', ())), cell_kind=spec.get('cell_kind', {}))


def event_ledger(cx, per_theta, steps, thetas, cells, control):
    """First checkpoint at which each listed cell is present, per threshold, plus
    whether it stayed present for the following checkpoint (stability >= 2 checkpoints)
    and any control-cell violations."""
    ledger = {}
    for c in cells:
        i = cx.index(c)
        per = {}
        for th in thetas:
            masks = per_theta[th]['masks']
            first = next((k for k, m in enumerate(masks) if m >> i & 1), None)
            stable = first is not None and first + 1 < len(masks) and bool(masks[first + 1] >> i & 1)
            per[str(th)] = dict(first_step=steps[first] if first is not None else None, stable=stable)
        ledger[c] = per
    violations = {c: {th: v for th, v in ledger[c].items() if v['first_step'] is not None} for c in control if c in ledger}
    return dict(cells=ledger, control_violations={c: v for c, v in violations.items() if v})


# ------------------------------------------------------------------ analysis
def analyze_run(run_dir, design, thetas=THETA, n_perm=0, seed=0, window=1):
    D = DESIGNS[design] if design in DESIGNS else design_from_spec(design)
    cx = D['complex']()
    evals = load_run(run_dir)
    steps = sorted(evals)
    traj = {c: [] for c in cx.names}
    transfer = []
    for s in steps:
        sc = D['cells'](evals[s])
        for c in cx.names:
            traj[c].append(float(sc.get(c, 0.0)))
        transfer.append(dict(step=s, **D['transfer'](evals[s])))
    per_theta = {}
    entries = {}
    for th in thetas:
        masks = [cx.mask_from_scores({c: traj[c][i] for c in cx.names}, th) for i in range(len(steps))]
        betti = [(list(cx.betti(m)) + [0, 0, 0, 0])[:4] for m in masks]     # pad: betti() stops at the top dimension present
        bars = {}
        for k in (0, 1):
            for br in ('union', 'intersection'):
                bb = cx.barcode(tuple(masks), k, br)
                bars[f'H{k}_{br}'] = [dict(start_node=a, end_node=b, checkpoint_steps=[steps[j // 2] for j in range(a, b + 1) if j % 2 == 0],
                                           transition_only=all(j % 2 for j in range(a, b + 1))) for a, b in bb]
        trans = []
        ent = []
        for i in range(len(steps) - 1):
            h1 = CW.classify_h1_change(cx, masks[i], masks[i + 1])
            h0 = CW.classify_h0_change(cx, masks[i], masks[i + 1])
            added = set(h1['added'])
            n_sync = len(added & set(D['sync_set']))
            ent.append(n_sync)
            trans.append(dict(from_step=steps[i], to_step=steps[i + 1], h1=h1['category'], h1_delta=h1['h1_delta'], h0=h0.get('category'),
                              added=sorted(added), removed=sorted(h1['removed']), sync_entries=n_sync))
        per_theta[th] = dict(masks=masks, betti=betti, barcodes=bars, transitions=trans)
        entries[th] = ent
        if design == 'crosslingual':
            # secondary readout: G_en (+) G_zh without bridge/square/3-cells -> the "component merge" reading
            bridge = sum(1 << i for i, n in enumerate(cx.names) if n.endswith('@I'))
            bf = [m & ~bridge for m in masks]
            assert all(cx.is_closed(m) for m in bf)
            bf_trans = []
            for i in range(len(steps) - 1):
                h1 = CW.classify_h1_change(cx, bf[i], bf[i + 1])
                h0 = CW.classify_h0_change(cx, bf[i], bf[i + 1])
                bf_trans.append(dict(from_step=steps[i], to_step=steps[i + 1], h1=h1['category'], h1_delta=h1['h1_delta'], h0=h0.get('category'),
                                     added=sorted(h1['added']), removed=sorted(h1['removed'])))
            per_theta[th]['bridge_free'] = dict(betti=[(list(cx.betti(m)) + [0, 0, 0, 0])[:4] for m in bf],
                                                barcodes={f'H{k}_{br}': [dict(start_node=a, end_node=b, checkpoint_steps=[steps[j // 2] for j in range(a, b + 1) if j % 2 == 0])
                                                                         for a, b in cx.barcode(tuple(bf), k, br)] for k in (0, 1) for br in ('union', 'intersection')},
                                                transitions=bf_trans)
    # synchrony rule: a transition index t* with >= SYNC_MIN_CELLS entries at >= SYNC_MIN_THETAS thresholds within +-SYNC_TOL
    T = len(steps) - 1
    sync_min = D.get('sync_min', SYNC_MIN_CELLS)
    support = np.zeros(T, dtype=int)
    for th in thetas:
        e = np.array(entries[th])
        for t in range(T):
            lo, hi = max(0, t - SYNC_TOL), min(T, t + SYNC_TOL + 1)
            if e[lo:hi].max(initial=0) >= sync_min:
                support[t] += 1
    raw = np.zeros(T, dtype=int)
    for th in thetas:
        raw += np.array(entries[th], dtype=int)
    t_star = int(raw.argmax()) if T else -1          # transition with the most zh entries summed over thresholds
    spread = {}
    for th in thetas:
        first = {}
        for tr in per_theta[th]['transitions']:
            for c in tr['added']:
                if c in D['sync_set'] and c not in first:
                    first[c] = tr['to_step']
        spread[th] = dict(entered=len(first), range_steps=(max(first.values()) - min(first.values())) if first else None)
    sync = dict(synchronous=bool(T and support[t_star] >= SYNC_MIN_THETAS), t_star_transition=t_star,
                t_star_steps=[steps[t_star], steps[t_star + 1]] if T else None, thresholds_supporting=int(support[t_star]) if T else 0,
                rule=dict(min_cells=sync_min, min_thetas=SYNC_MIN_THETAS, tolerance_checkpoints=SYNC_TOL, cells=list(D['sync_set'])),
                entries_per_theta={str(th): entries[th] for th in thetas}, spread_per_theta={str(th): spread[th] for th in thetas})
    gradual = all(v['range_steps'] is not None and v['range_steps'] >= 4 * (steps[1] - steps[0]) for v in spread.values()) if T else False
    sync['verdict'] = 'synchronous' if sync['synchronous'] else ('gradual' if gradual else 'undetermined')
    null = None
    if n_perm:
        null = CW.permutation_null(cx, traj, list(thetas), n_perm, seed, k=1, bridge='union')
        null = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in null.items()}
    # transfer windows around t*: delta over [t*-window, t*+window] vs every other window of the same length
    tw = None
    if T and sync['synchronous']:
        keys = D['transfer_keys']
        def delta(a, b, k):
            return transfer[b][k] - transfer[a][k]
        lo, hi = max(0, t_star + 1 - window), min(len(steps) - 1, t_star + 1 + window)   # event checkpoint = t*+1, window +-w checkpoints
        event = {k: delta(lo, hi, k) for k in keys}
        controls = []
        for a in range(0, len(steps) - (hi - lo)):
            b = a + (hi - lo)
            if b < lo or a > hi:
                controls.append({k: delta(a, b, k) for k in keys})
        base = dict(min_zh_edge=[min(traj[e][i] for e in D['baseline_edges']) for i in range(len(steps))],
                    faces_passed=[sum(traj[f][i] >= 0.8 for f in D['faces']) for i in range(len(steps))],
                    mean_score=[float(np.mean([traj[c][i] for c in cx.names])) for i in range(len(steps))])
        tw = dict(window_steps=[steps[lo], steps[hi]], event_delta=event,
                  control_mean={k: float(np.mean([c[k] for c in controls])) if controls else None for k in keys},
                  control_n=len(controls), baselines_at_window=dict(min_zh_edge=base['min_zh_edge'][lo:hi + 1], faces_passed=base['faces_passed'][lo:hi + 1],
                                                                   mean_score=base['mean_score'][lo:hi + 1]))
    ledger = event_ledger(cx, per_theta, steps, thetas, list(D['sync_set']) + list(D.get('control_cells', ())), D.get('control_cells', ()))
    return dict(run=str(run_dir), design=design, steps=steps, trajectory=traj, transfer=transfer, thresholds=list(thetas), ledger=ledger,
                per_theta={str(th): dict(betti=v['betti'], barcodes=v['barcodes'], transitions=v['transitions'], bridge_free=v.get('bridge_free')) for th, v in per_theta.items()},
                synchrony=sync, null_model=null, transfer_windows=tw)


def write_outputs(res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'analysis.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf8')
    with (out_dir / 'states.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['complex', 'threshold', 'step', 'beta0', 'beta1', 'beta2'])
        for th, v in res['per_theta'].items():
            for s, b in zip(res['steps'], v['betti']):
                w.writerow(['primary', th, s] + list(b[:3]))
            if v.get('bridge_free'):
                for s, b in zip(res['steps'], v['bridge_free']['betti']):
                    w.writerow(['bridge_free', th, s] + list(b[:3]))
    with (out_dir / 'events.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['complex', 'threshold', 'from_step', 'to_step', 'h1', 'h1_delta', 'h0', 'sync_entries', 'added', 'removed'])
        for th, v in res['per_theta'].items():
            for t in v['transitions']:
                w.writerow(['primary', th, t['from_step'], t['to_step'], t['h1'], t['h1_delta'], t['h0'], t['sync_entries'], ' '.join(t['added']), ' '.join(t['removed'])])
            for t in (v.get('bridge_free') or {}).get('transitions', []):
                w.writerow(['bridge_free', th, t['from_step'], t['to_step'], t['h1'], t['h1_delta'], t['h0'], '', ' '.join(t['added']), ' '.join(t['removed'])])
    with (out_dir / 'barcodes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'homology', 'bridge', 'start_node', 'end_node', 'checkpoint_steps', 'transition_only'])
        for th, v in res['per_theta'].items():
            for name, bars in v['barcodes'].items():
                k, br = name.split('_')
                for b in bars:
                    w.writerow([th, k, br, b['start_node'], b['end_node'], ' '.join(map(str, b['checkpoint_steps'])), b['transition_only']])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        steps = res['steps']
        for c, ys in res['trajectory'].items():
            if c in (DESIGNS[res['design']] if res['design'] in DESIGNS else design_from_spec(res['design']))['sync_set']:
                axs[0].plot(steps, ys, label=c, lw=1)
        axs[0].set(title='synchrony cells', ylim=(-.02, 1.02), xlabel='step')
        axs[0].legend(fontsize=6, ncol=2)
        for th, v in res['per_theta'].items():
            axs[1].plot(steps, [b[1] for b in v['betti']], label=f'θ={th}', lw=1)
        axs[1].set(title='β1(K_t(θ))', xlabel='step')
        axs[1].legend(fontsize=6)
        for k in (DESIGNS[res['design']] if res['design'] in DESIGNS else design_from_spec(res['design']))['transfer_keys']:
            axs[2].plot(steps, [t[k] for t in res['transfer']], label=k, lw=1)
        axs[2].set(title='independent transfer', ylim=(-.02, 1.02), xlabel='step')
        axs[2].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / 'overview.png', dpi=150)
        plt.close(fig)
    except Exception as e:      # noqa: BLE001
        (out_dir / 'NO_FIGURE.txt').write_text(str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True, help='directory containing world*_ARM run folders')
    p.add_argument('--design', required=True, help="'crosslingual', 'stage_ab', or a path to a generated COMPLEX.json")
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--perm', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260922)
    a = p.parse_args()
    summary = {}
    for run in sorted(a.runs.glob('world*')):
        if not (run / 'COMPLETE.json').exists() and not list(run.glob('eval_*.jsonl')):
            continue
        res = analyze_run(run, a.design, n_perm=a.perm, seed=a.seed)
        write_outputs(res, a.out / run.name)
        summary[run.name] = dict(verdict=res['synchrony']['verdict'], t_star=res['synchrony']['t_star_steps'],
                                 thresholds_supporting=res['synchrony']['thresholds_supporting'],
                                 min_p=min(res['null_model']['p_values']) if res['null_model'] and 'p_values' in res['null_model'] else None,
                                 transfer_windows=res['transfer_windows'], ledger=res['ledger'])
        print(run.name, summary[run.name]['verdict'], summary[run.name]['t_star'])
    (a.out / 'SUMMARY.json').write_text(json.dumps(dict(design=a.design, scope='development description; no confirmatory inference', runs=summary),
                                                   ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
