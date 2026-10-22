"""Generate a training stream, probe set, ground-truth topology and cw_zigzag
complex definition from a declared graph specification (specs/*.json).

Semantics. Every vertex X is a representation of an underlying index 0..31:
X-code = X + three digits taken from a digit table T_X (vertices in the same
`digit_group` share a table). An edge X->Y carries a permutation sigma on the
index: X-code of i  ->  Y-code of sigma(i).
  rule=codebook   : sigma = identity (consistent) or a random derangement (twist)
  rule=relabel    : shared digit table and sigma = identity, i.e. "keep the digits,
                    swap the prefix" -- the only rule a model can generalise to
                    an untaught pair
  rule=composition: sigma = composition along `via` (so the direct edge agrees
                    with that route)
Cell kinds: taught (in the stream from introduce_step), latent (probed only),
control (untaught, underivable placebo), taught_inconsistent (joint task is
trained with flag 0; never a 2-cell).

A 2-cell is declared only for consistent cycles; the ground-truth topology is
computed with cw_zigzag and asserted against `expected_topology`.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CW_DIR = HERE.parent
ROOT = CW_DIR.parent
sys.path.insert(0, str(ROOT / 'stage_ab' / 'code'))
sys.path.insert(0, str(ROOT / 'crosslingual' / 'code'))
import zz_score as Z          # noqa: E402
import cw_zigzag as CW        # noqa: E402

SUFFIX = {0: '只给规定格式的答案，不要解释。', 1: '请仅填写答案；字段用竖线分隔，序列内部用逗号分隔。',
          2: '完成任务，禁止附加说明，只填写所要求的字段。', 3: '依字段先后作答，使用竖线和逗号分隔，不添加解释。'}
TRAIN_TEMPLATES = (0, 1, 2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jwrite(p, rs):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rs), encoding='utf8')
    return sha(p)


class World:
    def __init__(self, spec, world, seed):
        self.spec = spec
        self.world = world
        self.rng = random.Random(seed)
        n = spec['alphabet']
        groups = {v: v for v in spec['vertices']}
        for g in spec['digit_groups']:
            for v in g:
                groups[v] = g[0]
        tables = {}
        for v in spec['vertices']:
            key = groups[v]
            if key not in tables:
                tables[key] = self.rng.sample(range(100, 1000), n)
            self.__dict__.setdefault('T', {})[v] = tables[key]
        self.inv = {v: {c: i for i, c in enumerate(t)} for v, t in self.T.items()}
        self.edges = {e['name']: e for e in spec['edges']}
        self.sigma = {}
        for e in spec['edges']:
            if e['rule'] == 'composition':
                continue
            if e.get('twist'):
                while True:
                    p = list(range(n))
                    self.rng.shuffle(p)
                    if all(p[i] != i for i in range(n)):
                        break
                self.sigma[e['name']] = p
            else:
                self.sigma[e['name']] = list(range(n))
        for e in spec['edges']:
            if e['rule'] == 'composition':
                p = list(range(n))
                for step in e['via']:
                    p = [self.sigma[step][i] for i in p]
                self.sigma[e['name']] = p
        for e in spec['edges']:
            if e['rule'] == 'relabel':
                assert self.T[e['src']] == self.T[e['dst']], f'relabel edge {e["name"]} needs a shared digit table'
        self.faces = {f['name']: f for f in spec['faces']}
        for f in spec['faces']:
            f['consistent'] = self.consistent(f)
            if f['kind'] == 'taught_inconsistent':
                assert not f['consistent'], f['name']
            elif f['kind'] in ('taught', 'latent'):
                assert f['consistent'], f['name']

    # ---- code arithmetic
    def code(self, v, i):
        return f'{v}{self.T[v][i]:03d}'

    def view(self, v, idx):
        return ','.join(self.code(v, i) for i in idx)

    def apply(self, edge, idx):
        return tuple(self.sigma[edge][i] for i in idx)

    def edge_between(self, a, b):
        for e in self.spec['edges']:
            if e['src'] == a and e['dst'] == b:
                return e['name']
        raise KeyError((a, b))

    def consistent(self, f):
        x, y, z = f['path']
        xy, yz, xz = self.edge_between(x, y), self.edge_between(y, z), self.edge_between(x, z)
        return all(self.sigma[yz][self.sigma[xy][i]] == self.sigma[xz][i] for i in range(self.spec['alphabet']))

    # ---- tasks
    def vertex_task(self, v, idx, t, rid, split):
        return dict(id=rid, task=f'v{v}', kind='vertex', vocabulary='rcn', template=t, root=list(idx), split=split,
                    prompt=f'读取代码序列 {self.view(v, idx)}，原顺序输出每个代码的三位数字编号，去掉字母。 {SUFFIX[t]}',
                    gold=','.join(f'{self.T[v][i]:03d}' for i in idx), schema=['digits'], field_length=len(idx))

    def edge_task(self, name, idx, t, rid, split):
        e = self.edges[name]
        return dict(id=rid, task=name, kind='edge', vocabulary='rcn', template=t, root=list(idx), split=split, cell_kind=e['kind'],
                    prompt=f"按已学习且固定的代码词典，把 {e['src']} 序列 {self.view(e['src'], idx)} 转换成 {e['dst']} 序列。 {SUFFIX[t]}",
                    gold=self.view(e['dst'], self.apply(name, idx)), schema=[e['dst']], field_length=len(idx))

    def joint_task(self, fname, idx, other, t, rid, override, split):
        f = self.faces[fname]
        x, y, z = f['path']
        xy, yz, xz = self.edge_between(x, y), self.edge_between(y, z), self.edge_between(x, z)
        middle_idx = self.apply(xy, other if override else idx)
        end_idx = self.apply(yz, middle_idx)
        direct_idx = self.apply(xz, idx)
        ins = f'将中间状态明确替换为 {self.view(y, self.apply(xy, other))}。' if override else '保持按词典得到的中间状态。'
        prompt = (f'从 {x} 序列 {self.view(x, idx)} 出发，比较 {x}->{z} 与 {x}->{y}->{z}。{ins}'
                  f'输出：直接{z}序列|实际中间{y}序列|沿该中间状态得到的{z}序列|两个终点是否相同(1或0)。 {SUFFIX[t]}')
        return dict(id=rid, task=fname, kind='joint', vocabulary='rcn', template=t, root=list(idx), other=list(other), override=override,
                    split=split, cell_kind=f['kind'], consistent=f['consistent'], prompt=prompt,
                    gold='|'.join([self.view(z, direct_idx), self.view(y, middle_idx), self.view(z, end_idx), str(int(direct_idx == end_idx))]),
                    schema=[z, y, z, 'binary'], field_length=len(idx))

    def repair_task(self, fname, idx, other, bad, t, rid, error, split):
        f = self.faces[fname]
        x, y, z = f['path']
        xy, yz = self.edge_between(x, y), self.edge_between(y, z)
        mid = self.apply(xy, idx)
        end = self.apply(yz, mid)
        om = self.apply(xy, bad) if error == 1 else mid
        oe = self.apply(yz, self.apply(xy, bad)) if error in (1, 2) else end
        prompt = (f'从 {x} 序列 {self.view(x, idx)} 出发执行 {x}->{y}->{z}。保持按词典得到的中间状态。'
                  f'观察中间状态 {self.view(y, om)}，终点 {self.view(z, oe)}。输出：应有中间{y}序列|应有{z}终点|最早错误(无错0/中间1/终点2)。 {SUFFIX[t]}')
        return dict(id=rid, task='repair_' + fname, kind='repair', vocabulary='rcn', template=t, root=list(idx), wrong_root=list(bad), error=error,
                    split=split, prompt=prompt, gold='|'.join([self.view(y, mid), self.view(z, end), str(error)]), schema=[y, z, 'step'], field_length=len(idx))

    def chain_task(self, chain, idx, t, rid, split):
        names = [self.edge_between(a, b) for a, b in zip(chain, chain[1:])]
        cur = idx
        outs = []
        for e in names:
            cur = self.apply(e, cur)
            outs.append(cur)
        arrow = '->'.join(chain)
        prompt = (f'按固定词典依次执行 {arrow}，从 {chain[0]} 序列 {self.view(chain[0], idx)} 出发。'
                  f'输出：{"|".join(f"{v}序列" for v in chain[1:])}。 {SUFFIX[t]}')
        return dict(id=rid, task='chain_' + '_'.join(chain), kind='transfer', vocabulary='rcn', template=t, root=list(idx), split=split,
                    requires=names, prompt=prompt, gold='|'.join(self.view(v, o) for v, o in zip(chain[1:], outs)), schema=chain[1:], field_length=len(idx))

    def format_task(self, idx, t, rid):
        a, b = idx
        v1, v2 = self.rng.sample(self.spec['vertices'], 2)
        fields = [self.view(v1, idx), self.view(v2, (a, b)), self.view(v2, (b, a)), str(int(a == b))]
        prompt = (f'只做字段复制，不做词典推理。请按固定槽位输出：槽A={v1}序列，槽B={v2}序列，槽C={v2}序列，槽E=相等标志。'
                  f'输入字段依次为：{"|".join(fields)}。只能输出四个字段，用竖线分隔，不能解释。 {SUFFIX[t]}')
        return dict(id=rid, kind='format', vocabulary='rcn', template=t, root=list(idx), prompt=prompt, gold='|'.join(fields),
                    schema=[v1, v2, v2, 'binary'], field_length=2)


def active_at(cell, step):
    """A taught cell is in the training stream at `step` iff introduced and not inside its retire window."""
    if cell.get('introduce_step', 0) > step:
        return False
    r, back = cell.get('retire_step'), cell.get('reintroduce_step')
    if r is not None and step >= r and (back is None or step < back):
        return False
    return True


def route_task(W, fname, idx, t, rid, split):
    """Second leg of the face route on the same item: the Y-code of sigma_xy(idx) -> Z-code."""
    f = W.faces[fname]
    x, y, z = f['path']
    xy, yz = W.edge_between(x, y), W.edge_between(y, z)
    mid = W.apply(xy, idx)
    return dict(id=rid, task=f'r_{fname}_{y}{z}', kind='route', vocabulary='rcn', template=t, root=list(idx), item=list(idx), split=split,
                face=fname, prompt=f"按已学习且固定的代码词典，把 {y} 序列 {W.view(y, mid)} 转换成 {z} 序列。 {SUFFIX[t]}",
                gold=W.view(z, W.apply(yz, mid)), schema=[z], field_length=len(idx))


# ------------------------------------------------------------------ topology
def cells_of(spec, include=('taught', 'latent', 'control', 'taught_inconsistent')):
    cells = [(v, 0, []) for v in spec['vertices']]
    for e in spec['edges']:
        if e['kind'] in include:
            cells.append((e['name'], 1, [e['src'], e['dst']]))
    for f in spec['faces']:
        if f['kind'] in include and f.get('consistent', True) and f['kind'] != 'taught_inconsistent':
            cells.append((f['name'], 2, list(f['cycle'])))
    return cells


def betti_of(spec, present):
    cx = CW.CWComplex(cells_of(spec))
    mask = 0
    for n in present:
        mask |= 1 << cx.index(n)
    assert cx.is_closed(mask), sorted(set(cx.names) - set(present))
    b = list(cx.betti(mask)) + [0, 0]
    return b[:2]


def ground_truth(spec):
    verts = set(spec['vertices'])
    taught_e = {e['name']: e['introduce_step'] for e in spec['edges'] if e['kind'] == 'taught'}
    taught_f = {f['name']: f['introduce_step'] for f in spec['faces'] if f['kind'] == 'taught'}
    latent_e = [e['name'] for e in spec['edges'] if e['kind'] == 'latent']
    latent_f = [f['name'] for f in spec['faces'] if f['kind'] == 'latent']
    control = [e['name'] for e in spec['edges'] if e['kind'] == 'control']

    cell_by = {**{e['name']: e for e in spec['edges']}, **{f['name']: f for f in spec['faces']}}

    def present(step, latent=False):
        s = set(verts) | {n for n in taught_e if active_at(cell_by[n], step)} | {n for n in taught_f if active_at(cell_by[n], step)}
        if latent:
            s |= set(latent_e) | set(latent_f)
        return s
    phases = {}
    for name, step in [('taught_before_64', 63), ('taught_before_128', 127), ('taught_before_256', 255), ('taught_all', 10 ** 9)]:
        phases[name] = betti_of(spec, present(step))
    phases['taught_plus_latent'] = betti_of(spec, present(10 ** 9, latent=True))
    exp = spec['expected_topology']
    for k, v in phases.items():
        if k in exp:
            assert v == exp[k], (k, v, exp[k])
    # optional step-indexed phase table (curricula with retire windows): {"<step>": [b0, b1]} checked at that step
    step_phases = {}
    for st, b in spec.get('expected_phases', {}).items():
        got = betti_of(spec, present(int(st)))
        assert got == b, ('expected_phases', st, got, b)
        step_phases[st] = got
    taught_windows = {n: dict(introduce=c.get('introduce_step', 0), retire=c.get('retire_step'), reintroduce=c.get('reintroduce_step'))
                      for n, c in cell_by.items() if c['kind'] in ('taught', 'taught_inconsistent')}
    # earliest possible step of each latent cell = max introduce_step over prerequisites
    edge_by = {e['name']: e for e in spec['edges']}
    earliest = {}
    for e in spec['edges']:
        if e['kind'] == 'latent':
            if e['rule'] == 'composition':
                earliest[e['name']] = max(taught_e[v] for v in e['via'])
            elif e['rule'] == 'relabel':
                earliest[e['name']] = min(t for n, t in taught_e.items() if edge_by[n]['rule'] == 'relabel')
    for f in spec['faces']:
        if f['kind'] == 'latent':
            earliest[f['name']] = max(earliest.get(c, taught_e.get(c, 0)) for c in f['cycle'])
    # bridge sequence
    seq = {}
    base = present(10 ** 9)
    b0 = betti_of(spec, base)
    bridges = [n for n in latent_e if edge_by[n]['rule'] == 'relabel']
    for i in range(1, len(bridges) + 1):
        seq['+'.join(bridges[:i])] = betti_of(spec, base | set(bridges[:i]))
    return dict(phases=phases, step_phases=step_phases, taught_windows=taught_windows, earliest_possible_step=earliest,
                control_cells=control, bridge_sequence=dict(taught_all=b0, **seq),
                permanent_holes=['fB123 cycle (twisted B1B3)', '11-edge cross-domain cycle A3B1..C4A1..A2A3 (no 2-cell)'])


# ------------------------------------------------------------------- world build
def build(spec, world, out):
    W = World(spec, world, 700000 + world)
    rng = W.rng
    n = spec['alphabet']
    pairs = [(a, b) for a in range(n) for b in range(n)]
    rng.shuffle(pairs)
    split = dict(train=pairs[:960], structure=pairs[960:992], independent=pairs[992:])
    P = spec['probe']
    ps = spec['per_step']
    taught_e = [e for e in spec['edges'] if e['kind'] == 'taught']
    taught_f = [f for f in spec['faces'] if f['kind'] in ('taught', 'taught_inconsistent')]
    stream = []
    for s in range(spec['steps']):
        avail_e = [e['name'] for e in taught_e if active_at(e, s)]
        avail_f = [f['name'] for f in taught_f if active_at(f, s)]
        rep = [f for f in spec['repair_faces'] if active_at(W.faces[f], s)] if s >= spec['repair_introduce_step'] else []
        block = []
        for j in range(ps['vertex']):
            v = rng.choice(spec['vertices'])
            idx = (rng.randrange(n),) if j % 2 == 0 else rng.choice(split['train'])
            block.append(W.vertex_task(v, idx, rng.choice(TRAIN_TEMPLATES), f'tr_v_{s}_{j}', 'train'))
        n_joint = ps['joint'] if avail_f else 0
        n_rep = ps['repair'] if rep else 0
        n_edge = ps['edge'] + (ps['joint'] - n_joint) + (ps['repair'] - n_rep)
        for j in range(n_edge):
            idx = (rng.randrange(n),) if j % 2 == 0 else rng.choice(split['train'])
            block.append(W.edge_task(rng.choice(avail_e), idx, rng.choice(TRAIN_TEMPLATES), f'tr_e_{s}_{j}', 'train'))
        for j in range(n_joint):
            idx, other = rng.sample(split['train'], 2)
            # DEFECT_20260923: earlier versions passed (override, rid) in the wrong order, so every training
            # joint row was the override variant. Keyword arguments make the intent explicit.
            block.append(W.joint_task(rng.choice(avail_f), idx, other, rng.choice(TRAIN_TEMPLATES), rid=f'tr_j_{s}_{j}', override=bool(j % 2), split='train'))
        for j in range(n_rep):
            idx, other, bad = rng.sample(split['train'], 3)
            block.append(W.repair_task(rng.choice(rep), idx, other, bad, rng.choice(TRAIN_TEMPLATES), f'tr_r_{s}_{j}', j % 3, 'train'))
        for j in range(ps['format']):
            block.append(W.format_task(rng.choice(split['train']), rng.choice(TRAIN_TEMPLATES), f'tr_f_{s}_{j}'))
        chains = [c for c in spec.get('train_chains', []) if all(active_at(W.edges[W.edge_between(a, b)], s) for a, b in zip(c, c[1:]))]
        n_chain = ps.get('chain', 0) if chains else 0
        for j in range(n_chain):
            block.append(W.chain_task(rng.choice(chains), rng.choice(split['train']), rng.choice(TRAIN_TEMPLATES), f'tr_c_{s}_{j}', 'train'))
        for j in range(ps.get('chain', 0) - n_chain):          # chain format not yet teachable: fill with edges
            idx = rng.choice(split['train'])
            block.append(W.edge_task(rng.choice(avail_e), idx, rng.choice(TRAIN_TEMPLATES), f'tr_e2_{s}_{j}', 'train'))
        assert len(block) == sum(ps.values()), (s, len(block))
        rng.shuffle(block)
        stream.extend(block)
    # probes on held-out roots
    probes = []
    st, ind = split['structure'], split['independent']
    for v in spec['vertices']:
        for t in P['vertex_templates']:
            for j, idx in enumerate(st[:P['vertex_roots']]):
                probes.append(W.vertex_task(v, idx, t, f'pr_v{v}_{t}_{j}', 'structure'))
    for e in spec['edges']:
        for t in P['edge_templates']:
            for j, idx in enumerate(st[:P['edge_roots']]):
                probes.append(W.edge_task(e['name'], idx, t, f'pr_{e["name"]}_{t}_{j}', 'structure'))
    for f in spec['faces']:
        for t in P['face_templates']:
            for j, idx in enumerate(st[:P['face_roots']]):
                other = st[(j + 1) % len(st)]
                for over in (False, True):
                    probes.append(W.joint_task(f['name'], idx, other, t, f'pr_{f["name"]}_{t}_{j}_{int(over)}', over, 'structure'))
    for f in [f for f in spec['faces'] if f['kind'] in ('taught', 'latent')]:      # routes only for declared 2-cells
        for t in P.get('route_templates', P['main_templates']):
            for j, idx in enumerate(st[:P['face_roots']]):
                probes.append(route_task(W, f['name'], idx, t, f'pr_route_{f["name"]}_{t}_{j}', 'structure'))
    for r in probes:
        r.setdefault('item', list(r['root']))
    for chain in spec['transfer_chains']:
        for j, idx in enumerate(ind[:P['chain_roots']]):
            probes.append(W.chain_task(chain, idx, 0, f'pr_chain_{"".join(chain)}_{j}', 'independent'))
    for fname in spec['repair_faces']:
        for j, idx in enumerate(ind[:P['repair_roots']]):
            bad = ind[(j + 2) % len(ind)]
            for err in (0, 1, 2):
                probes.append(W.repair_task(fname, idx, ind[(j + 1) % len(ind)], bad, 0, f'pr_rep_{fname}_{j}_{err}', err, 'independent'))
    held = set(st) | set(ind)
    for r in stream:
        if len(r['root']) == 2:
            assert tuple(r['root']) not in held, r['id']
        assert r.get('cell_kind', 'taught') in ('taught', 'taught_inconsistent'), r['id']
        if r['kind'] == 'joint':
            assert isinstance(r['override'], bool) and isinstance(r['id'], str), r
    jo = [r['override'] for r in stream if r['kind'] == 'joint']
    assert not jo or 0.4 <= sum(jo) / len(jo) <= 0.6, 'joint override variants must be balanced in the stream'
    assert len({r['id'] for r in stream}) == len(stream), 'training ids must be unique'
    assert all(Z.score(r, r['gold'])['correct'] for r in stream + probes)
    d = out / f'world{world}'
    hashes = {name: jwrite(d / name, rs) for name, rs in [('TRAIN_STREAM.jsonl', stream), ('PROBES.jsonl', probes)]}
    gt = ground_truth(spec)
    cx_cells = cells_of(spec)
    complex_def = dict(name=spec['name'], cells=cx_cells,
                       cell_kind={**{v: 'vertex' for v in spec['vertices']}, **{e['name']: e['kind'] for e in spec['edges']},
                                  **{f['name']: f['kind'] for f in spec['faces']}},
                       latent_cells=[e['name'] for e in spec['edges'] if e['kind'] == 'latent'] + [f['name'] for f in spec['faces'] if f['kind'] == 'latent'],
                       control_cells=gt['control_cells'], taught_edges=[e['name'] for e in spec['edges'] if e['kind'] == 'taught'],
                       faces=[f['name'] for f in spec['faces'] if f['kind'] in ('taught', 'latent')],
                       transfer_tasks=sorted({r['task'] for r in probes if r['kind'] in ('transfer', 'repair')}),
                       main_templates=P['main_templates'], robustness_template=3, vocabulary='rcn',
                       face_readout=spec.get('face_readout', 'joint'),
                       face_routes={f['name']: [W.edge_between(*f['path'][:2]), f"r_{f['name']}_{f['path'][1]}{f['path'][2]}", W.edge_between(f['path'][0], f['path'][2])]
                                    for f in spec['faces'] if f['kind'] in ('taught', 'latent')},
                       taught_windows={n: w for n, w in gt['taught_windows'].items()})
    (d / 'COMPLEX.json').write_text(json.dumps(complex_def, ensure_ascii=False, indent=1), encoding='utf8')
    (d / 'GROUND_TRUTH.json').write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding='utf8')
    man = dict(version='COMPLEX_20260922', spec=spec['name'], world=world, seed=700000 + world, steps=spec['steps'], per_step=ps,
               batch=sum(ps.values()), digit_tables=W.T, permutations={k: v for k, v in W.sigma.items()}, split=split, hashes=hashes,
               probe_counts=dict(collections.Counter(r['kind'] for r in probes)), train_counts=dict(collections.Counter(r['kind'] for r in stream)),
               checkpoint_every=16, seeds=[1, 2], scope='Development worlds from a declared graph spec; controls and latent cells never trained')
    (d / 'MANIFEST.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf8')
    return man


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', default='RCN12')
    p.add_argument('--worlds', default='601,602,603,604')
    a = p.parse_args()
    spec = json.loads((CW_DIR / 'specs' / f'{a.spec}.json').read_text(encoding='utf8'))
    out = CW_DIR / 'data' / a.spec
    mans = [build(json.loads(json.dumps(spec)), int(w), out) for w in a.worlds.split(',')]
    (out / 'SPEC_MANIFEST.json').write_text(json.dumps(dict(status='PREPARED_LOCAL_NOT_TRAINED', spec=a.spec, spec_sha256=sha(CW_DIR / 'specs' / f'{a.spec}.json'),
                                                            worlds=[m['world'] for m in mans], manifests=mans), ensure_ascii=False, indent=1), encoding='utf8')
    for m in mans:
        print(m['world'], m['train_counts'], m['probe_counts'], {k: v[:10] for k, v in m['hashes'].items()})


if __name__ == '__main__':
    main()
