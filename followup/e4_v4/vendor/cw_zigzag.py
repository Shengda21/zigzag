"""General finite CW-complex F2 homology and zigzag barcodes.

This module generalizes the exact-F2, cell-indexed algorithms that were
hard-coded to a specific eleven-cell complex in
``reproduction/analysis_dependencies/full_topology.py`` (homology, induced
maps, augmented union/intersection zigzag diagrams, and the generalized-rank
interval decomposition of a type-A quiver) so that they work on *any* finite
CW complex built from a plain list of named cells with named boundaries.

Nothing about the underlying algorithms has changed relative to
``full_topology.py`` -- every cell index there (``0..10``) is simply replaced
by a lookup into a cell list supplied by the caller, and the algebra
(F2 rank / basis / span / nullspace / generalized rank / interval
decomposition) is untouched and kept at module scope exactly as it was
already complex-agnostic in the original file.

Cells and chains
-----------------
A cell is identified by a string name. A `CWComplex` is built from
``(name, dim, boundary_names)`` triples: ``dim`` is a non-negative integer
and ``boundary_names`` lists the (unsigned, F2) codimension-1 faces of the
cell, each of which must itself be a cell of the complex with dimension
``dim - 1``. Internally every cell gets an integer index (its position in
the input list) and all chains/subcomplexes are represented as Python
integers used as bitmasks over these indices -- exactly as in
``full_topology.py``, just with ``n`` cells instead of a hard-coded 11.

A "mask" is always a bitmask over cell indices. A mask is a *closed
subcomplex* iff, for every cell it contains, it also contains that cell's
entire topological closure (itself plus all of its faces, recursively).

Zigzag node/bridge index convention
------------------------------------
Given a tuple of ``T`` checkpoint masks, the augmented zigzag diagram used by
`CWComplex.diagram`, `CWComplex.barcode` and `CWComplex.endpoints` has
``2*T - 1`` nodes: even node ``2*i`` is checkpoint ``i`` and odd node
``2*i+1`` is the union/intersection *bridge* between checkpoint ``i`` and
checkpoint ``i+1``. Interval endpoints returned by `barcode` are indices
into this ``0 .. 2*T-2`` node range.

See ``README_cw_zigzag.md`` in this directory for a full description of the
API, the node-index convention, the event-classification rules, and the
permutation null model (including its relationship to, and limitations
against, Definition 3.3 of the source paper).
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'CWComplex', 'rank', 'basis', 'span', 'nullspace', 'generalized_rank',
    'decompose', 'classify_h1_change', 'classify_h0_change',
    'eleven_cell_complex', 'theory_graph_cells', 'product_with_interval',
    'crosslingual_complex', 'permutation_null',
]


# --------------------------------------------------------------------------
# Module-level F2 linear algebra. These are already complex-agnostic in
# full_topology.py (they operate on bare bitmask vectors / abstract quiver
# dimensions and arrows), so they are ported verbatim, only documented.
# --------------------------------------------------------------------------

def rank(vectors: Iterable[int]) -> int:
    """F2 rank of a list of bitmask vectors, via Gaussian elimination."""
    piv: Dict[int, int] = {}
    for x in vectors:
        x = int(x)
        while x:
            j = x.bit_length() - 1
            if j in piv:
                x ^= piv[j]
            else:
                piv[j] = x
                break
    return len(piv)


def basis(vectors: Iterable[int]) -> List[int]:
    """A maximal F2-independent subset of `vectors`, kept in input order."""
    selected: List[int] = []
    for x in vectors:
        x = int(x)
        if rank(selected + [x]) > len(selected):
            selected.append(x)
    return selected


def span(b: Sequence[int]) -> List[int]:
    """All ``2**len(b)`` F2 linear combinations of the vectors in `b`."""
    result = [0]
    for x in b:
        result += [y ^ x for y in result]
    return result


def nullspace(equations: Iterable[int], n: int) -> List[int]:
    """A basis of the F2 nullspace of `equations` (bitmask rows) in n unknowns."""
    rows = [int(x) for x in equations if x]
    pivots: List[int] = []
    pos = 0
    for j in range(n):
        chosen = next((i for i in range(pos, len(rows)) if rows[i] >> j & 1), None)
        if chosen is None:
            continue
        rows[pos], rows[chosen] = rows[chosen], rows[pos]
        for i in range(len(rows)):
            if i != pos and rows[i] >> j & 1:
                rows[i] ^= rows[pos]
        pivots.append(j)
        pos += 1
    free = [j for j in range(n) if j not in pivots]
    out = []
    for j in free:
        x = 1 << j
        for row, p in zip(rows, pivots):
            if row >> j & 1:
                x |= 1 << p
        out.append(x)
    return out


def generalized_rank(dims: Sequence[int], arrows: Sequence[Tuple[int, int, Sequence[int]]],
                      a: int, b: int) -> int:
    """Rank of the limit -> colimit map of a type-A quiver restricted to ``[a, b]``.

    `dims` gives the vector-space dimension at each of the quiver's nodes;
    `arrows` is a list of ``(source, target, matrix_columns)`` triples where
    ``matrix_columns[j]`` is the (bitmask) image, in the target's basis, of
    the ``j``-th basis vector of the source.
    """
    offsets: Dict[int, int] = {}
    n = 0
    for i in range(a, b + 1):
        offsets[i] = n
        n += dims[i]
    if n == 0:
        return 0
    equations: List[int] = []
    relations: List[int] = []
    for s, t, cols in arrows:
        if not (a <= s <= b and a <= t <= b):
            continue
        for j, col in enumerate(cols):
            relations.append((1 << (offsets[s] + j)) ^ (int(col) << offsets[t]))
        for j in range(dims[t]):
            row = 1 << (offsets[t] + j)
            for l, col in enumerate(cols):
                if col >> j & 1:
                    row ^= 1 << (offsets[s] + l)
            equations.append(row)
    kernel = nullspace(equations, n)
    images = [x & ((1 << dims[a]) - 1) for x in kernel]
    return rank(relations + images) - rank(relations)


def decompose(dims: Sequence[int], arrows: Sequence[Tuple[int, int, Sequence[int]]]
              ) -> List[Tuple[int, int]]:
    """Interval decomposition ``[(a, b), ...]`` of a type-A zigzag quiver.

    Uses the standard inclusion-exclusion of generalized ranks over all
    sub-intervals; asserts internally that the resulting bar multiset
    reproduces the input dimension sequence.
    """
    n = len(dims)
    r: Dict[Tuple[int, int], int] = {}
    for a in range(n):
        for b in range(a, n):
            r[a, b] = generalized_rank(dims, arrows, a, b)
    bars: List[Tuple[int, int]] = []
    for a in range(n):
        for b in range(a, n):
            count = r[a, b] - r.get((a - 1, b), 0) - r.get((a, b + 1), 0) + r.get((a - 1, b + 1), 0)
            if count < 0:
                raise AssertionError(f'negative interval multiplicity at ({a},{b}): {count}')
            bars.extend([(a, b)] * count)
    if [sum(a <= i <= b for a, b in bars) for i in range(n)] != list(dims):
        raise AssertionError('interval decomposition does not reproduce input dimensions')
    return bars


# --------------------------------------------------------------------------
# CWComplex
# --------------------------------------------------------------------------

class CWComplex:
    """A finite CW complex with F2 (unsigned) boundary maps.

    Parameters
    ----------
    cells : list of (name, dim, boundary_names)
        `name` is a unique string identifying the cell. `dim` is its
        dimension (a non-negative int). `boundary_names` lists the names of
        its codimension-1 faces (each must already be a cell in the list,
        with dimension ``dim - 1``); the boundary map is unsigned (F2), so
        `boundary_names` need not repeat and order does not matter.

    Attributes
    ----------
    names : tuple[str, ...]
        Cell names in input order; a cell's *index* is its position here.
    dims : tuple[int, ...]
        Dimension of each cell, aligned with `names`.
    bound : tuple[int, ...]
        For each cell, the bitmask of its immediate (codimension-1) faces.
    closure : tuple[int, ...]
        For each cell, the bitmask of its full topological closure (itself
        plus all faces, recursively).
    """

    def __init__(self, cells: Sequence[Tuple[str, int, Sequence[str]]]):
        cells = list(cells)
        names = [str(c[0]) for c in cells]
        if len(set(names)) != len(names):
            dupes = sorted({nm for nm in names if names.count(nm) > 1})
            raise ValueError(f'duplicate cell names: {dupes}')
        self.names: Tuple[str, ...] = tuple(names)
        self.n: int = len(cells)
        self._index: Dict[str, int] = {nm: i for i, nm in enumerate(self.names)}
        self.dims: Tuple[int, ...] = tuple(int(c[1]) for c in cells)

        bound = [0] * self.n
        for i, (name, dim, bd) in enumerate(cells):
            m = 0
            for bname in bd:
                if bname not in self._index:
                    raise ValueError(f'cell {name!r} references unknown boundary cell {bname!r}')
                j = self._index[bname]
                if self.dims[j] != dim - 1:
                    raise ValueError(
                        f'cell {name!r} (dim {dim}) has boundary cell {bname!r} '
                        f'with dim {self.dims[j]}, expected dim {dim - 1}')
                m |= 1 << j
            bound[i] = m
        self.bound: Tuple[int, ...] = tuple(bound)

        # Closure(i) = {i} u closure(j) for every immediate face j of i.
        # Recursion always terminates because dim strictly decreases along
        # every boundary edge (enforced above), so there can be no cycles.
        closure: List[Optional[int]] = [None] * self.n

        def _closure_of(i: int, _stack: Tuple[int, ...] = ()) -> int:
            if closure[i] is not None:
                return closure[i]
            if i in _stack:
                raise ValueError('cyclic boundary relation detected')
            c = 1 << i
            b = self.bound[i]
            j = 0
            while b:
                if b & 1:
                    c |= _closure_of(j, _stack + (i,))
                b >>= 1
                j += 1
            closure[i] = c
            return c

        for i in range(self.n):
            _closure_of(i)
        self.closure: Tuple[int, ...] = tuple(closure)  # type: ignore[arg-type]

    # -- basic queries -----------------------------------------------------

    def index(self, name: str) -> int:
        """Integer index of a cell name."""
        return self._index[name]

    def cells_in(self, mask: int) -> List[str]:
        """Names of the cells set in `mask`, in index order."""
        mask = int(mask)
        return [self.names[i] for i in range(self.n) if mask >> i & 1]

    def diff(self, a: int, b: int) -> Dict[str, List[str]]:
        """Cell names added/removed going from mask `a` to mask `b`."""
        a, b = int(a), int(b)
        added = [self.names[i] for i in range(self.n) if (b >> i & 1) and not (a >> i & 1)]
        removed = [self.names[i] for i in range(self.n) if (a >> i & 1) and not (b >> i & 1)]
        return {'added': added, 'removed': removed}

    def is_closed(self, mask: int) -> bool:
        """True iff `mask` contains the full closure of every cell it contains."""
        mask = int(mask)
        if not (0 <= mask < (1 << self.n)):
            return False
        return all(not (mask >> i & 1) or mask & c == c for i, c in enumerate(self.closure))

    def close(self, mask: int) -> int:
        """Topological closure of `mask`: the smallest closed subcomplex containing it."""
        mask = int(mask)
        out = 0
        i = 0
        while mask:
            if mask & 1:
                out |= self.closure[i]
            mask >>= 1
            i += 1
        return out

    def mask_from_scores(self, scores: Mapping[str, float], threshold: float) -> int:
        """Threshold a per-cell score dict into a closed subcomplex mask.

        Matches `local_topology.mask_from_scores`: a cell enters the
        complex iff *every* cell in its closure (itself and all its faces,
        recursively) has a score ``>= threshold``. This guarantees the
        result is always a closed subcomplex.
        """
        if not (0 <= threshold <= 1):
            raise ValueError('threshold must lie in [0, 1]')
        vals = []
        for name in self.names:
            if name not in scores:
                raise ValueError(f'missing score for cell {name!r}')
            v = float(scores[name])
            if not math.isfinite(v) or not (0 <= v <= 1):
                raise ValueError(f'score for {name!r} must be a finite value in [0, 1]')
            vals.append(v)
        raw = sum((1 << i) for i, x in enumerate(vals) if x >= threshold)
        return sum((1 << i) for i, c in enumerate(self.closure) if raw & c == c)

    # -- boundary / homology -------------------------------------------------

    def _boundary(self, chain: int) -> int:
        ans = 0
        for i, b in enumerate(self.bound):
            if chain >> i & 1:
                ans ^= b
        return ans

    @lru_cache(maxsize=None)
    def homology(self, mask: int, k: int) -> Tuple[Tuple[int, ...], Dict[int, int]]:
        """F2 homology in dimension `k` of the closed subcomplex `mask`.

        Returns ``(reps, coord)``: `reps` is a tuple of representative
        k-chains (bitmasks) spanning a complement of the boundary group
        inside the cycle group (so ``len(reps)`` is the Betti number);
        `coord` maps every k-cycle chain (bitmask) to its homology class
        index (``0 .. len(reps)-1``) under this choice of representatives.
        """
        mask = int(mask)
        if not self.is_closed(mask):
            raise ValueError(f'mask {mask} is not a closed subcomplex')
        chains = [1 << i for i in range(self.n) if self.dims[i] == k and mask >> i & 1]
        bd_gens = [self.bound[i] for i in range(self.n) if self.dims[i] == k + 1 and mask >> i & 1]
        bd = basis(bd_gens)
        cycles = [x for x in span(chains) if self._boundary(x) == 0]
        reps: List[int] = []
        combined = bd[:]
        for x in cycles:
            if rank(combined + [x]) > len(combined):
                reps.append(x)
                combined.append(x)
        coord: Dict[int, int] = {}
        for h, z in enumerate(span(reps)):
            for b in span(bd):
                coord[z ^ b] = h
        assert len(cycles) == len(coord)
        return tuple(reps), coord

    def betti(self, mask: int) -> List[int]:
        """Betti numbers ``[b0, b1, ..., b_D]`` of `mask`, D = max active cell dim."""
        mask = int(mask)
        if not self.is_closed(mask):
            raise ValueError(f'mask {mask} is not a closed subcomplex')
        active_dims = [self.dims[i] for i in range(self.n) if mask >> i & 1]
        max_d = max(active_dims) if active_dims else 0
        return [len(self.homology(mask, k)[0]) for k in range(max_d + 1)]

    @lru_cache(maxsize=None)
    def induced(self, source: int, target: int, k: int) -> Tuple[int, ...]:
        """Matrix (as a tuple of target-basis columns) of ``H_k(source) -> H_k(target)``."""
        source, target = int(source), int(target)
        if source & target != source:
            raise ValueError('source must be a subcomplex of target')
        reps, _ = self.homology(source, k)
        _, coord = self.homology(target, k)
        return tuple(coord[r] for r in reps)

    # -- zigzag diagrams / barcodes -----------------------------------------

    def augmented(self, masks: Sequence[int], bridge: str = 'union') -> List[int]:
        """Interleave checkpoint masks with union/intersection bridge masks."""
        masks = [int(m) for m in masks]
        out: List[int] = []
        for i, m in enumerate(masks):
            if i:
                prev = masks[i - 1]
                out.append((prev | m) if bridge == 'union' else (prev & m))
            out.append(m)
        return out

    def diagram(self, masks: Sequence[int], k: int = 1, bridge: str = 'union'
                ) -> Tuple[List[int], List[Tuple[int, int, Tuple[int, ...]]]]:
        """Build the ``(dims, arrows)`` type-A quiver for a checkpoint sequence."""
        nodes = self.augmented(masks, bridge)
        dims = [len(self.homology(m, k)[0]) for m in nodes]
        arrows: List[Tuple[int, int, Tuple[int, ...]]] = []
        for i in range(len(nodes) - 1):
            forward = (i % 2 == 0) == (bridge == 'union')
            s, t = (i, i + 1) if forward else (i + 1, i)
            arrows.append((s, t, self.induced(nodes[s], nodes[t], k)))
        return dims, arrows

    def barcode(self, masks: Sequence[int], k: int = 1, bridge: str = 'union'
                ) -> Tuple[Tuple[int, int], ...]:
        """Interval decomposition of the augmented zigzag diagram.

        Node indices are as described in the module docstring: even node
        ``2*i`` is checkpoint ``i``, odd node ``2*i+1`` is the bridge
        between checkpoints ``i`` and ``i+1``.
        """
        masks_t = tuple(int(m) for m in masks)
        return self._barcode_cached(masks_t, k, bridge)

    @lru_cache(maxsize=50000)
    def _barcode_cached(self, masks: Tuple[int, ...], k: int, bridge: str
                         ) -> Tuple[Tuple[int, int], ...]:
        return tuple(decompose(*self.diagram(masks, k, bridge)))

    def endpoints(self, masks: Sequence[int], k: int = 1, bridge: str = 'union',
                  exclude_boundary: bool = False) -> np.ndarray:
        """Per-transition count of barcode endpoints (closed-form, no `decompose`).

        Returns an array of length ``len(masks) - 1``; entry ``i`` counts
        interval endpoints landing on nodes ``{2i, 2i+1, 2i+2}`` (checkpoint
        i, the bridge, checkpoint i+1). Provably equal to
        ``endpoints_from_barcode`` below; both are exercised in the tests.
        """
        masks = list(masks)
        dims, arrows = self.diagram(masks, k, bridge)
        ends = np.array(dims, dtype=int) * 2
        for s, t, cols in arrows:
            r = rank(cols)
            ends[s] -= r
            ends[t] -= r
        if exclude_boundary:
            ends[0] = 0
            ends[-1] = 0
        return np.array([sum(ends[2 * i:2 * i + 3]) for i in range(len(masks) - 1)])


# --------------------------------------------------------------------------
# Event classification
# --------------------------------------------------------------------------

def classify_h1_change(cx: CWComplex, prev_mask: int, next_mask: int) -> Dict[str, Any]:
    """Classify a beta_1 change between two closed subcomplexes of `cx`.

    When beta_1 *decreases* (``next`` has fewer independent 1-cycles than
    ``prev``):
      - ``'edge_loss'``   if any 0- or 1-cell was removed;
      - ``'clean_fill'``  else if a 2-cell was added (a cycle got capped);
      - ``'bridge_only'`` otherwise (the drop is purely a union/intersection
        bridge effect, with no local cell change explaining it).

    When beta_1 does not decrease: ``'h1_increase'`` or ``'no_change'``.

    Returns a dict with keys ``h1_delta``, ``category``, ``added``, ``removed``.
    """
    prev_mask, next_mask = int(prev_mask), int(next_mask)
    b1_prev = len(cx.homology(prev_mask, 1)[0])
    b1_next = len(cx.homology(next_mask, 1)[0])
    delta = b1_next - b1_prev
    d = cx.diff(prev_mask, next_mask)
    added, removed = d['added'], d['removed']
    if delta < 0:
        removed_low = [nm for nm in removed if cx.dims[cx.index(nm)] in (0, 1)]
        added_2d = [nm for nm in added if cx.dims[cx.index(nm)] == 2]
        if removed_low:
            category = 'edge_loss'
        elif added_2d:
            category = 'clean_fill'
        else:
            category = 'bridge_only'
    elif delta > 0:
        category = 'h1_increase'
    else:
        category = 'no_change'
    return {'h1_delta': delta, 'category': category, 'added': added, 'removed': removed}


def classify_h0_change(cx: CWComplex, prev_mask: int, next_mask: int) -> Dict[str, Any]:
    """Classify a beta_0 change between two closed subcomplexes of `cx`.

    When beta_0 *decreases* (components merge): ``'merge'`` if a 1-cell was
    added and no 0-cell was removed, else ``'vertex_loss'``.

    When beta_0 *increases* (components split off): ``'vertex_birth'`` if an
    isolated 0-cell was added, else ``'split'`` if a 1-cell was removed,
    else ``'other_h0_increase'``.

    Otherwise: ``'no_change'``.

    Returns a dict with keys ``h0_delta``, ``category``, ``added``, ``removed``.
    """
    prev_mask, next_mask = int(prev_mask), int(next_mask)
    b0_prev = len(cx.homology(prev_mask, 0)[0])
    b0_next = len(cx.homology(next_mask, 0)[0])
    delta = b0_next - b0_prev
    d = cx.diff(prev_mask, next_mask)
    added, removed = d['added'], d['removed']
    added_0d = [nm for nm in added if cx.dims[cx.index(nm)] == 0]
    added_1d = [nm for nm in added if cx.dims[cx.index(nm)] == 1]
    removed_0d = [nm for nm in removed if cx.dims[cx.index(nm)] == 0]
    removed_1d = [nm for nm in removed if cx.dims[cx.index(nm)] == 1]
    if delta < 0:
        category = 'merge' if (added_1d and not removed_0d) else 'vertex_loss'
    elif delta > 0:
        if added_0d:
            category = 'vertex_birth'
        elif removed_1d:
            category = 'split'
        else:
            category = 'other_h0_increase'
    else:
        category = 'no_change'
    return {'h0_delta': delta, 'category': category, 'added': added, 'removed': removed}


# --------------------------------------------------------------------------
# The original eleven-cell theory graph, and its product with an interval.
# --------------------------------------------------------------------------

def theory_graph_cells() -> List[Tuple[str, int, List[str]]]:
    """The eleven-cell complex from full_topology.py / local_topology.py.

    Cell order and names are exactly those of ``CELLS``/``DIMS``/``BOUND``
    in ``reproduction/analysis_dependencies/full_topology.py``: four
    vertices ``vF vD vP vR``, five edges ``eFD eDP eFP eFR eRP``, and two
    triangles ``fA`` (= F-D-P) and ``fB`` (= F-P-R) sharing edge ``eFP``.
    """
    return [
        ('vF', 0, []),
        ('vD', 0, []),
        ('vP', 0, []),
        ('vR', 0, []),
        ('eFD', 1, ['vF', 'vD']),
        ('eDP', 1, ['vD', 'vP']),
        ('eFP', 1, ['vF', 'vP']),
        ('eFR', 1, ['vF', 'vR']),
        ('eRP', 1, ['vR', 'vP']),
        ('fA', 2, ['eFD', 'eDP', 'eFP']),
        ('fB', 2, ['eFP', 'eFR', 'eRP']),
    ]


def eleven_cell_complex() -> CWComplex:
    """Reconstruct the original eleven-cell complex as a `CWComplex`."""
    return CWComplex(theory_graph_cells())


def product_with_interval(cells: Sequence[Tuple[str, int, Sequence[str]]],
                           ends: Tuple[str, str] = ('en', 'zh'),
                           bridge_tag: str = 'I') -> List[Tuple[str, int, List[str]]]:
    """Cylinder/prism product ``K* = G x I`` of a CW complex G with an interval.

    For every base cell ``sigma`` (dimension d, boundary ``{tau_1,...,tau_m}``)
    this produces three kinds of cells (F2: unsigned boundaries throughout):

      - ``sigma@end`` (dimension d) for each label in `ends`, with boundary
        ``{tau_1@end, ..., tau_m@end}``  -- a copy of G at each end;
      - ``sigma@bridge_tag`` (dimension d+1), with boundary the union of
        ``{tau_1@bridge_tag, ..., tau_m@bridge_tag}`` (the "side wall" swept
        out by sigma's own boundary) and ``{sigma@end for end in ends}``
        (the two "end caps").

    `ends` must contain exactly two labels (the two ends of the interval).
    Returns a list of ``(name, dim, boundary_names)`` triples, ready to hand
    to `CWComplex`.
    """
    if len(ends) != 2:
        raise ValueError('product_with_interval requires exactly two interval ends')
    out: List[Tuple[str, int, List[str]]] = []
    for name, dim, bd in cells:
        for end in ends:
            out.append((f'{name}@{end}', dim, [f'{b}@{end}' for b in bd]))
        side = [f'{b}@{bridge_tag}' for b in bd] + [f'{name}@{end}' for end in ends]
        out.append((f'{name}@{bridge_tag}', dim + 1, side))
    return out


def crosslingual_complex() -> CWComplex:
    """``theory_graph_cells() x I`` with ends ``('en', 'zh')``: 33 cells."""
    return CWComplex(product_with_interval(theory_graph_cells()))


# --------------------------------------------------------------------------
# Permutation null model
# --------------------------------------------------------------------------

def permutation_null(cx: CWComplex, score_matrix: Mapping[str, Sequence[float]],
                      thresholds: Sequence[float], n_perm: int, seed: int,
                      k: int = 1, bridge: str = 'union',
                      block: Optional[Any] = None, alpha: float = 0.05) -> Dict[str, Any]:
    """Pre-registerable permutation null model for zigzag endpoint bursts.

    NOTE ON SIGNATURE: the deliverable's pseudocode omits the target
    `CWComplex`, but computing masks and endpoints is impossible without
    one, so `cx` is the required first argument here.

    Statistic
    ---------
    For a threshold grid ``theta_1..theta_M`` and a checkpoint sequence of
    length ``T`` (one score per cell per checkpoint in `score_matrix`), each
    threshold gives one mask per checkpoint via ``cx.mask_from_scores`` and
    hence one ``endpoints()`` array of length ``T-1``. The observed
    statistic is the threshold-averaged endpoint count per transition,
    ``T_t = (1/M) * sum_theta E_{k,t,theta}`` for transition index
    ``t = 0 .. T-2`` (transition ``t`` links checkpoint ``t`` and checkpoint
    ``t+1``; see the module docstring for the node-index convention that
    `endpoints` itself is built on).

    Null model
    ----------
    For each permutation replicate, walk the transition boundaries
    ``t = 0 .. T-2`` in order; at each boundary, for every group of cells
    (see `block`), flip one fair coin and, on heads, swap that boundary's
    scores for every cell in the group (``score[t] <-> score[t+1]``),
    applied in place and in order. Every individual cell's time series ends
    up as some permutation of its own original values (marginals are
    preserved), while cells in the same block always move together.
    `n_perm` independent replicates give the null distribution of ``T_t``.

    Parameters
    ----------
    block : None, or dict[str, Any], or iterable of iterables of str
        ``None`` (default): every cell is permuted independently.
        A dict maps a cell name to a group id; cells sharing a group id are
        swapped synchronously. An iterable of iterables of cell names is
        also accepted as an explicit list of groups (cells not mentioned
        stay in their own singleton group).
    alpha : float
        Family-wise significance level for the Bonferroni correction across
        the ``T-1`` transitions tested (default 0.05).

    Returns
    -------
    dict with keys:
      observed          {t: float}      -- T_t
      p_values           {t: float}      -- (1 + #{T_perm[t] >= T_obs[t]}) / (n_perm + 1)
      null_quantiles      {t: {'q05','q50','q95': float}}
      n_perm, thresholds, k, bridge, n_transitions, alpha
      bonferroni_alpha    float          -- alpha / n_transitions
      significant_bonferroni {t: bool}   -- p_values[t] <= bonferroni_alpha
    """
    names = sorted(score_matrix.keys())
    if not names:
        raise ValueError('score_matrix must not be empty')
    T = len(score_matrix[names[0]])
    for name in names:
        if len(score_matrix[name]) != T:
            raise ValueError(f'score series for {name!r} has length {len(score_matrix[name])}, expected {T}')
    if T < 2:
        raise ValueError('need at least two checkpoints')
    thresholds = list(thresholds)
    M = len(thresholds)
    if M == 0:
        raise ValueError('thresholds must be non-empty')
    n_transitions = T - 1

    # Normalize `block` into a name -> group-id map, then into id -> members.
    if block is None:
        group_of = {name: name for name in names}
    elif isinstance(block, Mapping):
        group_of = {name: block.get(name, name) for name in names}
    else:
        group_of = {name: name for name in names}
        for gi, group in enumerate(block):
            for name in group:
                group_of[name] = ('__block__', gi)
    groups: Dict[Any, List[str]] = {}
    for name in names:
        groups.setdefault(group_of[name], []).append(name)

    def statistic(series: Mapping[str, Sequence[float]]) -> np.ndarray:
        acc = None
        for theta in thresholds:
            masks = [cx.mask_from_scores({nm: series[nm][t] for nm in names}, theta)
                     for t in range(T)]
            e = cx.endpoints(masks, k=k, bridge=bridge).astype(float)
            acc = e if acc is None else acc + e
        return acc / M

    t_obs = statistic(score_matrix)

    rng = random.Random(seed)
    null = np.zeros((n_perm, n_transitions), dtype=float)
    for b in range(n_perm):
        working = {name: list(score_matrix[name]) for name in names}
        for t in range(n_transitions):
            for members in groups.values():
                if rng.random() < 0.5:
                    for name in members:
                        working[name][t], working[name][t + 1] = working[name][t + 1], working[name][t]
        null[b] = statistic(working)

    observed: Dict[int, float] = {}
    p_values: Dict[int, float] = {}
    null_quantiles: Dict[int, Dict[str, float]] = {}
    for t in range(n_transitions):
        obs = float(t_obs[t])
        col = null[:, t]
        p = (1 + int(np.sum(col >= obs))) / (n_perm + 1)
        observed[t] = obs
        p_values[t] = float(p)
        null_quantiles[t] = {
            'q05': float(np.quantile(col, 0.05)),
            'q50': float(np.quantile(col, 0.50)),
            'q95': float(np.quantile(col, 0.95)),
        }
    bonferroni_alpha = alpha / n_transitions
    return {
        'observed': observed,
        'p_values': p_values,
        'null_quantiles': null_quantiles,
        'n_perm': n_perm,
        'thresholds': thresholds,
        'k': k,
        'bridge': bridge,
        'n_transitions': n_transitions,
        'alpha': alpha,
        'bonferroni_alpha': bonferroni_alpha,
        'significant_bonferroni': {t: p_values[t] <= bonferroni_alpha for t in range(n_transitions)},
    }
