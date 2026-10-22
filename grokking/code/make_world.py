"""Build TRAIN_<edge/face>.jsonl, PROBES.jsonl, COMPLEX.json, GROUND_TRUTH.json
and MANIFEST.json for one or more worlds of a modular-arithmetic spec
(specs/MODP.json, specs/MODP_LATENTSUB.json, specs/MODP_ADDONLY.json).

`build_data()` is the single source of truth for turning a spec dict into
in-memory rows (split, placebo table, training pools, probes); `build()`
wraps it to also compute/assert ground truth and write everything to disk.
run_grok.py imports `build_data` directly so a --frac hyperparameter run
(scan_grok.py) can regenerate an equivalent in-memory dataset without
touching the canonical on-disk artifacts of world701/702.

A spec may optionally declare `"cell_subset": {"vertices": [...], "edges":
[...], "faces": [...]}` to use only part of modp_common's global 11-vertex/
13-edge/3-face complex (MODP_ADDONLY does this: only P, S, X / P_S, P_X /
no faces). Specs without `cell_subset` (MODP, MODP_LATENTSUB) behave exactly
as before -- the *_names() helpers below fall back to the full global lists.

Usage: python make_world.py --spec MODP --worlds 701,702
       python make_world.py --spec MODP --worlds 701 --dryrun-p 13   (tests only)
"""
from __future__ import annotations
import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GROK_DIR = HERE.parent
ROOT = GROK_DIR.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import modp_common as M   # noqa: E402
import zz_score as Z      # noqa: E402
import cw_zigzag as CW    # noqa: E402


# --------------------------------------------------------------- cell subset
def spec_vertex_names(spec):
    return spec.get('cell_subset', {}).get('vertices', list(M.VERTICES))


def spec_edges(spec):
    names = spec.get('cell_subset', {}).get('edges', [e['name'] for e in M.EDGES])
    return [e for e in M.EDGES if e['name'] in names]


def spec_faces(spec):
    names = spec.get('cell_subset', {}).get('faces', [f['name'] for f in M.FACES])
    return [f for f in M.FACES if f['name'] in names]


def spec_transfer_tasks(spec):
    return spec.get('transfer_tasks', list(M.TRANSFER_TASKS))


def resolve_kinds(spec):
    eo = spec.get('edge_kind_overrides', {})
    fo = spec.get('face_kind_overrides', {})
    kind_e = {}
    for e in spec_edges(spec):
        default = 'control' if e.get('placebo') else 'taught'
        kind_e[e['name']] = eo.get(e['name'], default)
    kind_f = {f['name']: fo.get(f['name'], 'taught') for f in spec_faces(spec)}
    return kind_e, kind_f


# ------------------------------------------------------------------ topology
def cells_of(spec):
    """11 vertices in the full complex (P_X's destination X included -- P_X
    is a real P->X edge, never a self-loop: a regular CW complex forbids a
    single-vertex loop, and an (even spuriously present) loop would create a
    phantom H1 class the instant it entered the complex, which is exactly
    what a placebo control must never do). A spec's `cell_subset` can use
    only part of this complex (see module docstring)."""
    cells = [(v, 0, []) for v in spec_vertex_names(spec)]
    for e in spec_edges(spec):
        cells.append((e['name'], 1, [e['src'], e['dst']]))
    for f in spec_faces(spec):
        cells.append((f['name'], 2, list(f['cycle'])))
    return cells


def betti_of(cx, present):
    mask = 0
    for n in present:
        mask |= 1 << cx.index(n)
    assert cx.is_closed(mask), f'not closed: missing {sorted(n for n in cx.cells_in(cx.close(mask)) if n not in present)}'
    b = list(cx.betti(mask)) + [0, 0]
    return b[:2]


def ground_truth(spec, kind_e, kind_f):
    """Present sets are closure-driven by default (a vertex is present only
    if some present edge/face touches it), NOT "all vertices always
    present" -- in the 11-vertex MODP/MODP_LATENTSUB complex only 10
    vertices are ever touched by a taught/latent edge, so blanket-including
    every vertex would silently add X as a permanently disconnected
    component to every phase and inflate beta0 by 1. The one exception is
    the 'vertices_only' baseline, which is deliberately "every declared
    vertex, no edges".

    A spec can opt into `"vertex_present_mode": "blanket"` (MODP_ADDONLY
    does) to make every declared vertex present unconditionally instead --
    appropriate for a complex small enough that a control vertex's
    (dis)connection is itself the thing being measured (see
    specs/MODP_ADDONLY.json and README "最小规格与经典设置"). In blanket
    mode the placebo-invariance guarantee below does NOT generally hold (see
    the 'all_present' phase instead), so it is only computed/asserted in the
    default closure-driven mode.
    """
    cx = CW.CWComplex(cells_of(spec))
    vnames = spec_vertex_names(spec)
    blanket = spec.get('vertex_present_mode', 'closure') == 'blanket'

    def present(e_kinds, f_kinds):
        edges_in = [n for n, k in kind_e.items() if k in e_kinds]
        faces_in = [n for n, k in kind_f.items() if k in f_kinds]
        s = set(edges_in) | set(faces_in)
        if blanket:
            s |= set(vnames)
        for e in spec_edges(spec):
            if e['name'] in edges_in:
                s.add(e['src'])
                s.add(e['dst'])
        for f in spec_faces(spec):
            if f['name'] in faces_in:
                s.update(f['path'])
        return s
    phases = {
        'vertices_only': betti_of(cx, set(vnames)),
        'taught_no_faces': betti_of(cx, present(('taught',), ())),
        'taught_with_faces': betti_of(cx, present(('taught',), ('taught',))),
    }
    has_latent = any(k == 'latent' for k in kind_e.values()) or any(k == 'latent' for k in kind_f.values())
    if has_latent:
        phases['plus_latent_edge_only'] = betti_of(cx, present(('taught', 'latent'), ('taught',)))
        phases['plus_latent_edge_and_face'] = betti_of(cx, present(('taught', 'latent'), ('taught', 'latent')))
    if blanket:
        # In a complex this small, forcing every control cell in is itself a
        # meaningful (not invariant) topology change -- e.g. MODP_ADDONLY's
        # only control vertex X is otherwise isolated, so P_X connecting it
        # merges two components into one. This is reported as 'all_present',
        # NOT asserted equal to the taught baseline (see docstring).
        phases['all_present'] = betti_of(cx, present(('taught', 'latent', 'control'), ('taught', 'latent', 'control')))
    exp = spec['expected_topology']
    for k, v in phases.items():
        assert v == exp[k], (k, v, exp[k])
    assert set(phases) == set(exp), (set(phases), set(exp))

    out = dict(phases=phases,
               latent_edges=[n for n, k in kind_e.items() if k == 'latent'],
               latent_faces=[n for n, k in kind_f.items() if k == 'latent'],
               control_cells=[n for n, k in kind_e.items() if k == 'control'])

    if not blanket:
        # Placebo-invariance check (closure-driven complexes only, see
        # docstring): forcing the control edge P_X (and its destination
        # vertex X) into an otherwise-closed present set must NOT change
        # either Betti number. This holds because P->X is a pendant leaf
        # (X has no other edge touching it): adding a leaf never merges/
        # splits a component (beta0 unchanged) and never closes/opens a
        # cycle (beta1 unchanged). This is a structural guarantee about the
        # *complex*, not about training -- it's what makes P_X a valid
        # "must never move the topology" control cell in the first place.
        baseline_name = 'plus_latent_edge_and_face' if has_latent else 'taught_with_faces'
        baseline = present(('taught', 'latent'), ('taught', 'latent')) if has_latent else present(('taught',), ('taught',))
        with_placebo = baseline | {M.PLACEBO_EDGE, 'X'}
        b_before = betti_of(cx, baseline)
        b_after = betti_of(cx, with_placebo)
        assert b_before == b_after, ('placebo entering must not change Betti numbers', b_before, b_after)
        out['phases']['placebo_entered_no_betti_change'] = dict(before=b_before, after=b_after, baseline_phase=baseline_name)
        out['note'] = ('present sets never include control cells (P_X) or their destination vertex (X) except '
                        'in the explicit placebo-invariance check above; "latent" here can mean either a '
                        'structurally-untrained cell (MODP_LATENTSUB: P_D, f_sub) or, for MODP, the held-out '
                        'generalisation curve of an otherwise-taught cell (grokking) -- see README.')
    else:
        out['note'] = ('vertex_present_mode=blanket: every declared vertex is always present, so a control '
                        'vertex\'s (dis)connection is itself measured by the "all_present" phase rather than '
                        'asserted invariant -- see README "最小规格与经典设置".')
    return out


# --------------------------------------------------------------------- data
def filtered_pool(split, label, needs_bnz, needs_a_nz=False):
    pool = [pq for pq, lab in split.items() if lab == label]
    if needs_bnz:
        pool = [(a, b) for a, b in pool if b != 0]
    if needs_a_nz:
        pool = [(a, b) for a, b in pool if a != 0]
    return pool


def correct_and_wrong(f, a, b, p, rng):
    cmp = f['compare']
    if cmp == 'sub':
        correct = (-b) % p
    elif cmp == 'comm':
        correct = b % p
    elif cmp == 'div':
        correct = M.inv(b, p)
    else:
        raise KeyError(cmp)
    while True:
        wrong = rng.randrange(p)
        if wrong != correct:
            return correct, wrong


def build_data(spec, world, only_tasks=None):
    """Compute (deterministically, from spec+world alone) the split, the
    placebo lookup table, one training-row list per taught edge/face
    (train_rows, keyed by task name) and the full probe list. only_tasks,
    if given, restricts *training* pools to those task names (used by
    scan_grok.py to train a single edge without generating the rest)."""
    name = spec['name']
    p = spec['primes'][str(world)]
    kind_e, kind_f = resolve_kinds(spec)
    edges, faces, vertices = spec_edges(spec), spec_faces(spec), spec_vertex_names(spec)
    templates = spec['main_templates']
    frac = spec['split_frac']
    split = M.build_split(p, spec['split_seed_offset'] + world, frac)
    n_train = sum(1 for v in split.values() if v == 'train')
    n_held = sum(1 for v in split.values() if v == 'heldout')

    prng = random.Random(spec.get('placebo_seed_offset', 950000) + world)
    placebo_table = [prng.randrange(p) for _ in range(p * p)]

    rng = random.Random(f'{name}:{world}:gen:frac{frac}')

    train_rows = {}
    for e in edges:
        if e.get('placebo') or kind_e[e['name']] != 'taught':
            continue
        if only_tasks and e['name'] not in only_tasks:
            continue
        pool = filtered_pool(split, 'train', e.get('needs_bnz', False))
        rows = []
        for i, (a, b) in enumerate(pool):
            t = rng.choice(templates)
            rows.append(M.edge_row(e['name'], a, b, p, t, f'tr_{e["name"]}_{i}', 'train', cell_kind='taught'))
        train_rows[e['name']] = rows
    if M.PLACEBO_EDGE in kind_e and kind_e[M.PLACEBO_EDGE] in ('taught', 'control') and \
            (not only_tasks or M.PLACEBO_EDGE in only_tasks):
        e = M.EDGE_BY_NAME[M.PLACEBO_EDGE]
        pool = filtered_pool(split, 'train', False)
        rows = []
        for i, (a, b) in enumerate(pool):
            t = rng.choice(templates)
            rows.append(M.edge_row(e['name'], a, b, p, t, f'tr_{e["name"]}_{i}', 'train',
                                    placebo_table=placebo_table, cell_kind='control'))
        train_rows[e['name']] = rows

    for f in faces:
        if kind_f[f['name']] != 'taught':
            continue
        if only_tasks and f['name'] not in only_tasks:
            continue
        needs_a = f['compare'] == 'div'
        pool = filtered_pool(split, 'train', f.get('needs_bnz', False), needs_a_nz=needs_a)
        rows = []
        for i, (a, b) in enumerate(pool):
            correct, wrong = correct_and_wrong(f, a, b, p, rng)
            for j, (other, override) in enumerate(((correct, False), (wrong, True))):
                t = rng.choice(templates)
                rows.append(M.face_row(f['name'], a, b, other, p, t, f'tr_{f["name"]}_{i}_{j}', 'train', override))
        train_rows[f['name']] = rows

    # ---- probes (all edges/faces/vertices regardless of kind, + transfer)
    P = spec['probe']
    probes = []
    probe_counts = {}
    for e in edges:
        pool = filtered_pool(split, 'heldout', e.get('needs_bnz', False))
        roots = rng.sample(pool, min(P['edge_roots'], len(pool)))
        for t in templates:
            for i, (a, b) in enumerate(roots):
                if e.get('placebo'):
                    probes.append(M.edge_row(e['name'], a, b, p, t, f'pr_{e["name"]}_{t}_{i}', 'heldout',
                                              placebo_table=placebo_table, cell_kind='control'))
                else:
                    probes.append(M.edge_row(e['name'], a, b, p, t, f'pr_{e["name"]}_{t}_{i}', 'heldout',
                                              cell_kind=kind_e[e['name']]))
        probe_counts[e['name']] = len(roots) * len(templates)

    for v in vertices:
        needs_bnz = v in M.BNZ_VERTICES
        pool = filtered_pool(split, 'heldout', needs_bnz)
        roots = rng.sample(pool, min(P['vertex_roots'], len(pool)))
        for t in templates:
            for i, (a, b) in enumerate(roots):
                if v == 'X':
                    probes.append(M.vertex_row(v, a, b, p, t, f'pr_v{v}_{t}_{i}', 'heldout',
                                                placebo_table=placebo_table))
                else:
                    probes.append(M.vertex_row(v, a, b, p, t, f'pr_v{v}_{t}_{i}', 'heldout'))
        probe_counts['v' + v] = len(roots) * len(templates)

    for f in faces:
        needs_a = f['compare'] == 'div'
        pool = filtered_pool(split, 'heldout', f.get('needs_bnz', False), needs_a_nz=needs_a)
        roots = rng.sample(pool, min(P['face_roots'], len(pool)))
        n = 0
        for t in templates:
            for i, (a, b) in enumerate(roots):
                correct, wrong = correct_and_wrong(f, a, b, p, rng)
                for j, (other, override) in enumerate(((correct, False), (wrong, True))):
                    probes.append(M.face_row(f['name'], a, b, other, p, t, f'pr_{f["name"]}_{t}_{i}_{j}',
                                              'heldout', override, cell_kind=kind_f[f['name']]))
                    n += 1
        probe_counts[f['name']] = n

    for task in spec_transfer_tasks(spec):
        k = P['transfer_roots']
        for i in range(k):
            a, b, c = rng.randrange(p), rng.randrange(p), rng.randrange(p)
            probes.append(M.transfer_row(task, a, b, c, p, f'pr_{task}_{i}'))
        probe_counts[task] = k

    assert all(Z.score(r, r['gold'])['correct'] == 1 for r in probes), 'a probe gold row failed its own scorer'
    for r in probes:
        if r['kind'] in ('edge', 'vertex', 'joint'):
            assert tuple(r['root'][:2]) in split and split[tuple(r['root'][:2])] == 'heldout', r['id']

    return dict(p=p, kind_e=kind_e, kind_f=kind_f, split=split, placebo_table=placebo_table,
                train_rows=train_rows, probes=probes, probe_counts=probe_counts,
                n_train_pairs=n_train, n_heldout_pairs=n_held)


def build(spec, world, out_root):
    name = spec['name']
    dat = build_data(spec, world)
    kind_e, kind_f = dat['kind_e'], dat['kind_f']
    d = out_root / f'world{world}'
    hashes = {}
    train_counts = {}
    for task, rows in dat['train_rows'].items():
        hashes[f'TRAIN_{task}.jsonl'] = M.jwrite(d / f'TRAIN_{task}.jsonl', rows)
        train_counts[task] = len(rows)
    hashes['PROBES.jsonl'] = M.jwrite(d / 'PROBES.jsonl', dat['probes'])

    # ---- topology
    gt = ground_truth(spec, kind_e, kind_f)
    cell_kind = {**{v: 'vertex' for v in spec_vertex_names(spec)}, **kind_e, **kind_f}
    latent_cells = [n for n, k in kind_e.items() if k in ('taught', 'latent')] + \
                   [n for n, k in kind_f.items() if k in ('taught', 'latent')]
    taught_edges = [n for n, k in kind_e.items() if k == 'taught']
    complex_def = dict(
        name=name, vocabulary=spec['vocabulary'], cells=cells_of(spec), cell_kind=cell_kind,
        latent_cells=latent_cells, control_cells=gt['control_cells'], taught_edges=taught_edges,
        faces=[f['name'] for f in spec_faces(spec)], transfer_tasks=list(spec_transfer_tasks(spec)),
        main_templates=spec['main_templates'],
        face_readout='routes',
        face_readout_note=(
            'PRIMARY face score = route consistency: every edge in face_routes[name] must be correct '
            'for the SAME original item (row["item"], the pre-transform (a,b) even for a route probe on '
            'a transformed input such as Pn_D on (a,-b)) -- see code/face_scores.py:route_consistency, '
            'merged into every eval summary as modp:face:task<name>. Reverted from the joint '
            'equality-comparison readout after GPU evidence (701/702, --preset classic, 100k steps): '
            'f_sub/f_comm/f_div crossed 0.9 held-out accuracy at 500-2750 steps, BEFORE any of the 7 '
            'arithmetic edges (2500-43000 steps) -- because "a-b ?= a+n" is true iff n=-b, answerable from '
            'negation alone (learned in 200-400 steps) without needing subtraction at all. The '
            'equality-comparison task (task=name, kind=joint, override True/False) is still trained and '
            'probed -- design_from_spec has a built-in dim==2 branch (min(acc(override=False), '
            'acc(override=True))) still scores it -- but it is now a SECONDARY readout only; see README '
            '"等式捷径" and PREREGISTRATION_GROKKING_DRAFT.md for the evidence and the confirmatory '
            'criteria this readout change feeds into.'
        ),
        face_routes={f['name']: list(f['cycle']) for f in spec_faces(spec)},
    )
    (d / 'COMPLEX.json').write_text(json.dumps(complex_def, ensure_ascii=False, indent=1), encoding='utf8')
    (d / 'GROUND_TRUTH.json').write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding='utf8')

    man = dict(version='MODP_GROK_20260922', spec=name, world=world, p=dat['p'], split_frac=spec['split_frac'],
               split_seed=spec['split_seed_offset'] + world, placebo_seed=spec.get('placebo_seed_offset', 950000) + world,
               placebo_table_len=len(dat['placebo_table']), n_pairs=dat['p'] * dat['p'],
               n_train_pairs=dat['n_train_pairs'], n_heldout_pairs=dat['n_heldout_pairs'],
               kind_edges=kind_e, kind_faces=kind_f, steps=spec['steps'], eval_every=spec['eval_every'],
               ckpt_every=spec['ckpt_every'], hashes=hashes, train_counts=train_counts,
               probe_counts=dat['probe_counts'],
               probe_total=dict(collections.Counter(r['kind'] for r in dat['probes'])),
               scope='Development world; controls and structurally-latent cells (MODP_LATENTSUB only) never '
                     'trained; probes only ever drawn from the held-out split.')
    (d / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf8')
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='MODP')
    ap.add_argument('--worlds', default='701,702')
    ap.add_argument('--dryrun-p', type=int, default=None, help='override every world prime with this small p (tests only)')
    a = ap.parse_args()
    spec = json.loads((GROK_DIR / 'specs' / f'{a.spec}.json').read_text(encoding='utf8'))
    worlds = [int(w) for w in a.worlds.split(',')]
    if a.dryrun_p:
        spec = json.loads(json.dumps(spec))
        spec['primes'] = {str(w): a.dryrun_p for w in worlds}
        spec['probe'] = dict(edge_roots=32, vertex_roots=16, face_roots=16, transfer_roots=16)
    out = GROK_DIR / 'data' / a.spec
    mans = [build(spec, w, out) for w in worlds]
    (out / 'SPEC_MANIFEST.json').write_text(json.dumps(dict(
        status='PREPARED_LOCAL_NOT_TRAINED', spec=a.spec,
        spec_sha256=M.sha256_file(GROK_DIR / 'specs' / f'{a.spec}.json'),
        worlds=[m['world'] for m in mans], manifests=mans), ensure_ascii=False, indent=1), encoding='utf8')
    for m in mans:
        print(m['world'], m['p'], m['n_train_pairs'], m['n_heldout_pairs'], m['train_counts'], m['probe_counts'])


if __name__ == '__main__':
    main()
