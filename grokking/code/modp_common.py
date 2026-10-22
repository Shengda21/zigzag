"""Shared arithmetic, tokenisation and row-building code for the modular-
arithmetic grokking line (方案 A). Used by make_world.py (data generation),
run_grok.py / tiny_transformer.py (training + eval rendering) and
face_scores.py / test_grokking.py (secondary checks).

Complex (see specs/MODP.json, specs/MODP_LATENTSUB.json):
  11 vertices  P=(a,b) S=a+b D=a-b M=a*b N=-b Pn=(a,-b) Ps=(b,a)
               I=b^-1 (b!=0) Pi=(a,b^-1) Q=a*b^-1, X=a fixed i.i.d. Uniform(0,p)
               lookup table, independent of every real operation.
  13 edges     12 real conversions + 1 placebo P_X: P->X, a genuine 2-cell
               boundary {P, X} (NOT a self-loop -- a regular CW complex
               forbids a single-vertex loop, and a loop would spuriously
               create an H1 class the moment it entered the complex, which
               would break the placebo's "must never move the topology"
               role). X is a control vertex: it never appears in any
               "present" set used for a real Betti computation, so it is a
               pendant leaf whenever it *is* forced in for the explicit
               "placebo entering changes nothing" check in
               make_world.ground_truth().
  3 faces      f_sub (a-b = a+(-b)), f_comm (a+b = b+a), f_div (a/b = a*b^-1),
               each an equality-COMPARISON task (not a joint multi-field
               readout, see PREREGISTRATION_GROKKING_DRAFT.md rationale),
               plus a secondary "both routes correct" cross-check in
               face_scores.py.

Every vertex is a view: a tuple of 1 or 2 residues mod p. Every edge/face
task is a fixed token sequence built from that view plus an operator symbol;
the gold answer is always the *literal* numeric view of the destination
(edges) or a single 0/1 bit (faces) -- never derived by "chaining" routes,
so the generator itself never has to know how a model would compose edges.
"""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------- spec
VERTICES = ('P', 'S', 'D', 'M', 'N', 'Pn', 'Ps', 'I', 'Pi', 'Q', 'X')
# vertices whose view is a *pair* of two residues (all others are single numbers)
PAIR_VERTICES = ('P', 'Pn', 'Ps', 'Pi')
# vertices (and the edges that produce/consume them) that require b != 0
BNZ_VERTICES = ('I', 'Pi', 'Q')

EDGES = [
    dict(name='P_S', src='P', dst='S', op='+'),
    dict(name='P_D', src='P', dst='D', op='-'),
    dict(name='P_M', src='P', dst='M', op='*'),
    dict(name='P_N', src='P', dst='N', op='NEG'),
    dict(name='P_Pn', src='P', dst='Pn', op='MKNEG'),
    dict(name='Pn_D', src='Pn', dst='D', op='+'),
    dict(name='P_Ps', src='P', dst='Ps', op='MKSWAP'),
    dict(name='Ps_S', src='Ps', dst='S', op='+'),
    dict(name='P_I', src='P', dst='I', op='INV', needs_bnz=True),
    dict(name='P_Pi', src='P', dst='Pi', op='MKINV', needs_bnz=True),
    dict(name='Pi_Q', src='Pi', dst='Q', op='*', needs_bnz=True),
    dict(name='P_Q', src='P', dst='Q', op='/', needs_bnz=True),
    dict(name='P_X', src='P', dst='X', op='X', placebo=True),
]
EDGE_BY_NAME = {e['name']: e for e in EDGES}
REAL_EDGES = [e['name'] for e in EDGES if not e.get('placebo')]
PLACEBO_EDGE = 'P_X'

FACES = [
    dict(name='f_sub', cycle=['P_D', 'P_Pn', 'Pn_D'], path=['P', 'D', 'Pn'], compare='sub'),
    dict(name='f_comm', cycle=['P_S', 'P_Ps', 'Ps_S'], path=['P', 'S', 'Ps'], compare='comm'),
    dict(name='f_div', cycle=['P_Q', 'P_Pi', 'Pi_Q'], path=['P', 'Q', 'Pi'], compare='div', needs_bnz=True),
]
FACE_BY_NAME = {f['name']: f for f in FACES}

TRANSFER_TASKS = ('xfer_addsub3', 'xfer_muladd3')

# ----------------------------------------------------------------- symbols
SYMBOLS = ('+', '-', '*', '/', 'NEG', 'INV', 'MKNEG', 'MKSWAP', 'MKINV', 'X', '?=', '=')
VTAGS = tuple(f'[v{v}]' for v in VERTICES)
PAD = '<pad>'


def build_vocab(p: int) -> Dict[str, int]:
    """Deterministic token->id map: numbers 0..p-1, then symbols, then vertex
    tags, then pad. Order is fixed so it only depends on p (stable across a
    world's runs, reproducible from the spec + p alone)."""
    toks = [str(i) for i in range(p)] + list(SYMBOLS) + list(VTAGS) + [PAD]
    return {t: i for i, t in enumerate(toks)}


# --------------------------------------------------------------- arithmetic
def inv(b: int, p: int) -> int:
    assert b % p != 0, 'inverse of 0 is undefined'
    return pow(b, p - 2, p)


def vertex_view(name: str, a: int, b: int, p: int) -> Tuple[int, ...]:
    """Numeric view (1 or 2 residues) of vertex `name` on the *original*
    index pair (a, b). This is the single source of truth for every edge's
    gold answer: gold(edge) = vertex_view(edge.dst, a, b, p)."""
    a, b = a % p, b % p
    if name == 'P':
        return (a, b)
    if name == 'S':
        return ((a + b) % p,)
    if name == 'D':
        return ((a - b) % p,)
    if name == 'M':
        return ((a * b) % p,)
    if name == 'N':
        return ((-b) % p,)
    if name == 'Pn':
        return (a, (-b) % p)
    if name == 'Ps':
        return (b, a)
    if name == 'I':
        return (inv(b, p),)
    if name == 'Pi':
        return (a, inv(b, p))
    if name == 'Q':
        return ((a * inv(b, p)) % p,)
    raise KeyError(name)


def placebo_view(table: List[int], a: int, b: int, p: int) -> Tuple[int]:
    """P_X's label: a fixed lookup table, one i.i.d. Uniform(0, p) draw per
    (a, b), independent of every real operation. `table` has length p^2
    (index a*p+b), fixed per world and stored in MANIFEST.json. Because the
    label is independent random noise, any fixed model function matches it
    with expected probability exactly 1/p on held-out pairs -- this is the
    negative control for the grokking signature itself (a real edge should
    depart from 1/p; the placebo should not, however long training runs)."""
    idx = (a % p) * p + (b % p)
    return (table[idx],)


def fmt_view(view: Tuple[int, ...]) -> str:
    return ','.join(str(v) for v in view)


# ---------------------------------------------------------------- rendering
def edge_prompt_tokens(edge: dict, src_view: Tuple[int, ...], template: int) -> List[str]:
    x1, x2 = src_view
    op = edge['op']
    if template == 0:
        return [str(x1), op, str(x2), '=']
    return [op, str(x1), str(x2), '=']


def vertex_prompt_tokens(name: str, view: Tuple[int, ...], template: int) -> List[str]:
    vals = [str(v) for v in view]
    tag = f'[v{name}]'
    if template == 0:
        return [tag] + vals + ['=']
    return vals + [tag, '=']


def face_prompt_tokens(face: dict, a: int, b: int, other: int, p: int, template: int) -> Tuple[List[str], str]:
    """Returns (tokens, gold_bit) for the equality-comparison face task.
    `other` is the right-hand-side free value: for f_sub it's `n` compared
    against -b, for f_comm it's `b'` compared against b, for f_div it's `m`
    compared against b^-1. override=True (mismatched `other`) => gold '0'."""
    cmp = face['compare']
    a, b, other = a % p, b % p, other % p
    if cmp == 'sub':
        left_op, right_op = '-', '+'
        left = (a, b)
        right = (a, other)
        gold = int(((a - b) % p) == ((a + other) % p))
    elif cmp == 'comm':
        left_op, right_op = '+', '+'
        left = (a, b)
        right = (other, a)
        gold = int(((a + b) % p) == ((other + a) % p))
    elif cmp == 'div':
        left_op, right_op = '/', '*'
        left = (a, b)
        right = (a, other)
        gold = int(((a * inv(b, p)) % p) == ((a * other) % p))
    else:
        raise KeyError(cmp)
    l1, l2 = left
    r1, r2 = right
    if template == 0:
        toks = [str(l1), left_op, str(l2), '?=', str(r1), right_op, str(r2), '=']
    else:
        toks = ['?=', left_op, str(l1), str(l2), right_op, str(r1), str(r2), '=']
    return toks, str(gold)


MAX_LEN = 10   # longest prompt (face template1, 8 tokens) + longest answer (2) = 10


# --------------------------------------------------------------------- rows
def edge_row(edge_name: str, a: int, b: int, p: int, template: int, rid: str, split: str,
             item=None, placebo_table=None, cell_kind=None) -> dict:
    edge = EDGE_BY_NAME[edge_name]
    src_view = vertex_view(edge['src'], a, b, p)
    if edge.get('placebo'):
        dst_view = placebo_view(placebo_table, a, b, p)
    else:
        dst_view = vertex_view(edge['dst'], a, b, p)
    toks = edge_prompt_tokens(edge, src_view, template)
    ck = cell_kind if cell_kind is not None else ('control' if edge.get('placebo') else 'taught')
    return dict(id=rid, task=edge_name, kind='edge', vocabulary='modp', template=template,
                cell_kind=ck,
                root=[a, b], item=list(item if item is not None else (a, b)), split=split,
                prompt=' '.join(toks), gold=fmt_view(dst_view), schema=['exact'], field_length=1)


def vertex_row(name: str, a: int, b: int, p: int, template: int, rid: str, split: str,
               placebo_table=None) -> dict:
    view = placebo_view(placebo_table, a, b, p) if name == 'X' else vertex_view(name, a, b, p)
    toks = vertex_prompt_tokens(name, view, template)
    return dict(id=rid, task='v' + name, kind='vertex', vocabulary='modp', template=template,
                cell_kind='control' if name == 'X' else 'vertex',
                root=[a, b], item=[a, b], split=split,
                prompt=' '.join(toks), gold=fmt_view(view), schema=['exact'], field_length=1)


def face_row(face_name: str, a: int, b: int, other: int, p: int, template: int, rid: str,
             split: str, override: bool, cell_kind='taught') -> dict:
    face = FACE_BY_NAME[face_name]
    toks, gold = face_prompt_tokens(face, a, b, other, p, template)
    return dict(id=rid, task=face_name, kind='joint', vocabulary='modp', template=template,
                cell_kind=cell_kind, root=[a, b], item=[a, b], split=split, override=override,
                prompt=' '.join(toks), gold=gold, schema=['binary'], field_length=1)


def transfer_row(task: str, a: int, b: int, c: int, p: int, rid: str) -> dict:
    if task == 'xfer_addsub3':
        toks = [str(a), '+', str(b), '-', str(c), '=']
        gold = (a + b - c) % p
    elif task == 'xfer_muladd3':
        toks = [str(a), '*', str(b), '+', str(c), '=']
        gold = (a * b + c) % p
    else:
        raise KeyError(task)
    return dict(id=rid, task=task, kind='transfer', vocabulary='modp', template=0,
                root=[a, b, c], split='independent',
                prompt=' '.join(toks), gold=str(gold), schema=['exact'], field_length=1)


# ------------------------------------------------------------------- split
def build_split(p: int, seed: int, frac: float) -> Dict[Tuple[int, int], str]:
    """Partition Zp x Zp into train/heldout by whole orbits under the group
    generated by (a,b)->(a,-b), (a,b)->(b,a), (a,b)->(a,b^-1) [b=0 fixed].
    Assigning whole orbits guarantees: (a,b) heldout => (a,-b), (b,a),
    (a,b^-1) all heldout too (and symmetrically for train), which is exactly
    what stops Pn_D / Ps_S / Pi_Q from leaking through a differently-routed
    view of a pair the model has already seen via P_S / P_Ps / P_Pi."""
    def neg(pair):
        a, b = pair
        return (a, (-b) % p)

    def swap(pair):
        a, b = pair
        return (b, a)

    def invp(pair):
        a, b = pair
        return (a, inv(b, p)) if b % p != 0 else (a, b)

    seen = {}
    orbits: List[List[Tuple[int, int]]] = []
    for a in range(p):
        for b in range(p):
            if (a, b) in seen:
                continue
            stack = [(a, b)]
            members = set()
            while stack:
                x = stack.pop()
                if x in members:
                    continue
                members.add(x)
                for f in (neg, swap, invp):
                    stack.append(f(x))
            orb = sorted(members)
            for m in orb:
                seen[m] = len(orbits)
            orbits.append(orb)
    rng = random.Random(seed)
    order = list(range(len(orbits)))
    rng.shuffle(order)
    target = int(round(frac * p * p))
    split: Dict[Tuple[int, int], str] = {}
    n_train = 0
    for oi in order:
        orb = orbits[oi]
        label = 'train' if n_train < target else 'heldout'
        for m in orb:
            split[m] = label
        if label == 'train':
            n_train += len(orb)
    return split


def encode_row(row: dict, vocab: Dict[str, int], max_len: int = MAX_LEN):
    """Tokenise a data row into a fixed-length, right-padded id sequence for
    the trainer/evaluator: prompt tokens (row['prompt'], space-separated)
    followed by answer tokens (row['gold'], comma-separated -- 1 token for a
    single-number/bit answer, 2 for a pair-valued vertex/edge). Padding is
    always at the end, so causal attention alone keeps it from influencing
    any real position; no separate padding mask is needed."""
    ptoks = row['prompt'].split(' ')
    atoks = row['gold'].split(',')
    ids = [vocab[t] for t in ptoks] + [vocab[t] for t in atoks]
    prompt_len, answer_len = len(ptoks), len(atoks)
    assert prompt_len + answer_len <= max_len, (row['id'], prompt_len, answer_len)
    ids = ids + [vocab[PAD]] * (max_len - len(ids))
    return ids, prompt_len, answer_len


def render_prediction(answer_ids, id_to_tok) -> str:
    """Inverse of encode_row's answer half: token ids -> the comma-joined
    string format `gold` uses, so zz_score.score can compare directly."""
    return ','.join(id_to_tok[i] for i in answer_ids)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jwrite(path: Path, rows) -> str:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows), encoding='utf8')
    return sha256_file(path)
