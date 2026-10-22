#!/usr/bin/env python3
"""Small exact F2 algebra checks. NumPy only; NOT a production zigzag implementation.

The generalized-rank computation is intentionally transparent and slow.
The examples use explicitly specified induced homology maps.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def rref(a):
    a = np.asarray(a, dtype=np.uint8).copy() % 2
    row, pivots = 0, []
    for col in range(a.shape[1]):
        hits = np.flatnonzero(a[row:, col])
        if len(hits) == 0:
            continue
        hit = row + int(hits[0])
        a[[row, hit]] = a[[hit, row]]
        for rr in range(a.shape[0]):
            if rr != row and a[rr, col]:
                a[rr] ^= a[row]
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots

def rank(a):
    return len(rref(a)[1])

def kernel(a):
    reduced, pivots = rref(a)
    n = reduced.shape[1]
    free = [j for j in range(n) if j not in pivots]
    basis = np.zeros((n, len(free)), dtype=np.uint8)
    for j, col in enumerate(free):
        basis[col, j] = 1
        for rr, pivot in enumerate(pivots):
            basis[pivot, j] = reduced[rr, col]
    return basis

def generalized_rank(dims, arrows, lo, hi):
    """Rank of limit -> colimit on a connected subinterval of a type-A quiver."""
    offsets = {}
    total = 0
    for i in range(lo, hi + 1):
        offsets[i] = total
        total += dims[i]
    constraints, relations = [], []
    for src, dst, matrix in arrows:
        if not (lo <= src <= hi and lo <= dst <= hi):
            continue
        matrix = np.asarray(matrix, dtype=np.uint8).reshape(dims[dst], dims[src])
        c = np.zeros((dims[dst], total), dtype=np.uint8)
        c[:, offsets[dst]:offsets[dst] + dims[dst]] ^= np.eye(dims[dst], dtype=np.uint8)
        c[:, offsets[src]:offsets[src] + dims[src]] ^= matrix
        constraints.append(c)
        r = np.zeros((total, dims[src]), dtype=np.uint8)
        r[offsets[src]:offsets[src] + dims[src], :] ^= np.eye(dims[src], dtype=np.uint8)
        r[offsets[dst]:offsets[dst] + dims[dst], :] ^= matrix
        relations.append(r)
    c = np.concatenate(constraints, axis=0) if constraints else np.zeros((0, total), dtype=np.uint8)
    r = np.concatenate(relations, axis=1) if relations else np.zeros((total, 0), dtype=np.uint8)
    lim = kernel(c)
    # Any one node gives the same class in the colimit; do not sum them in F2.
    projection = np.zeros_like(lim)
    projection[:dims[lo], :] = lim[:dims[lo], :]
    return rank(np.concatenate([r, projection], axis=1)) - rank(r)

def barcode(dims, arrows):
    n = len(dims)
    ranks = {(i, j): generalized_rank(dims, arrows, i, j)
             for i in range(n) for j in range(i, n)}
    bars = []
    for i in range(n):
        for j in range(i, n):
            count = (ranks[(i, j)] - ranks.get((i - 1, j), 0)
                     - ranks.get((i, j + 1), 0) + ranks.get((i - 1, j + 1), 0))
            if count < 0:
                raise AssertionError("Negative interval multiplicity.")
            bars.extend([[i, j]] * count)
    assert [sum(a <= t <= b for a, b in bars) for t in range(n)] == dims
    return bars

def betti(vertices, edges, faces):
    vertices = list(vertices)
    edges = [tuple(sorted(e)) for e in edges]
    faces = [tuple(sorted(f)) for f in faces]
    d1 = np.zeros((len(vertices), len(edges)), dtype=np.uint8)
    d2 = np.zeros((len(edges), len(faces)), dtype=np.uint8)
    for j, e in enumerate(edges):
        for v in e:
            d1[vertices.index(v), j] = 1
    for j, f in enumerate(faces):
        for i in range(len(f)):
            edge = tuple(v for k, v in enumerate(f) if k != i)
            if edge not in edges:
                raise ValueError("Face closure violated.")
            d2[edges.index(edge), j] = 1
    assert not np.any((d1 @ d2) % 2)
    return [len(vertices) - rank(d1), len(edges) - rank(d1) - rank(d2)]

def run():
    one = np.ones((1, 1), dtype=np.uint8)
    cases = {
        "same_cycle": ([1, 1, 1], [(0, 1, one), (2, 1, one)], [[0, 2]]),
        "cycle_replacement": (
            [1, 2, 1], [(0, 1, [[1], [0]]), (2, 1, [[0], [1]])],
            [[0, 1], [1, 2]]),
        "union_bridge_only": (
            [0, 1, 0], [(0, 1, np.zeros((1, 0))), (2, 1, np.zeros((1, 0)))],
            [[1, 1]]),
        "fill_face_union": (
            [1, 0, 0], [(0, 1, np.zeros((0, 1))), (2, 1, np.zeros((0, 0)))],
            [[0, 0]]),
        "delete_edge_union": (
            [1, 1, 0], [(0, 1, one), (2, 1, np.zeros((1, 0)))],
            [[0, 1]]),
    }
    result = {}
    for name, (dims, arrows, expected) in cases.items():
        observed = barcode(dims, arrows)
        assert observed == expected, (name, observed, expected)
        result[name] = {"H1_dimensions": dims, "augmented_node_intervals": observed}
    v = [0, 1, 2, 3]
    e = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3)]
    assert betti(v, e, []) == [1, 2]
    assert betti(v, e, [(0, 1, 2)]) == [1, 1]
    assert betti(v, e, [(0, 1, 2), (0, 2, 3)]) == [1, 0]
    # Missing shared diagonal can still leave an outer quadrilateral H1 class.
    assert betti(v, [x for x in e if x != (0, 2)], []) == [1, 1]
    result["11_cell_complex_checks"] = "4 Betti/boundary checks passed."
    result["scope"] = "Exact algebra checks only; no model measurements or statistical calibration."
    return result

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("toy_check_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
