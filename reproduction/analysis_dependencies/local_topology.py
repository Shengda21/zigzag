#!/usr/bin/env python3
"""Exact F2, H1 local-span diagnostics on the registered eleven-cell complex.
This is not a general CW package and does not replace full zigzag barcodes.
All chain vectors are embedded in the same five-edge ambient chain group.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import numpy as np
from algebra_reference import rank, kernel

CELLS = ('vF','vD','vP','vR','eFD','eDP','eFP','eFR','eRP','fA','fB')
SIMPLICES = ((0,),(1,),(2,),(3,),(0,1),(1,2),(0,2),(0,3),(2,3),(0,1,2),(0,2,3))
EDGES = SIMPLICES[4:9]
CLOSURE = tuple(tuple(j for j,s in enumerate(SIMPLICES) if set(s) <= set(x)) for x in SIMPLICES)

def mask_from_scores(scores, threshold):
    a = np.asarray(scores, dtype=float)
    if a.shape != (11,) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError('scores must contain eleven finite values in [0,1]')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must lie in [0,1]')
    return sum((1 << i) for i in range(11) if min(a[list(CLOSURE[i])]) >= threshold)

def mask_from_cells(cells):
    unknown = set(cells)-set(CELLS)
    if unknown:
        raise ValueError(f'Unknown cells: {unknown}')
    m = sum(1 << CELLS.index(c) for c in set(cells))
    if not is_closed(m):
        raise ValueError('Input cells are not a subcomplex')
    return m

def is_closed(mask):
    return 0 <= mask < (1<<11) and all(not(mask>>i&1) or all(mask>>j&1 for j in CLOSURE[i]) for i in range(11))

@lru_cache(maxsize=2048)
def chains(mask):
    if not is_closed(mask):
        raise ValueError('Mask violates face closure')
    active_e = [j for j in range(5) if mask>>(4+j)&1]
    d1 = np.zeros((4,len(active_e)), dtype=np.uint8)
    for j,e in enumerate(active_e):
        d1[list(EDGES[e]), j] = 1
    z_local = kernel(d1)
    z = np.zeros((5,z_local.shape[1]), dtype=np.uint8)
    z[active_e,:] = z_local
    active_f = [j for j in (9,10) if mask>>j&1]
    b = np.zeros((5,len(active_f)), dtype=np.uint8)
    for j,f in enumerate(active_f):
        for edge in combinations(SIMPLICES[f],2):
            b[EDGES.index(tuple(sorted(edge))),j] = 1
    beta0 = sum(mask>>j&1 for j in range(4)) - rank(d1)
    beta1 = z.shape[1] - rank(b)
    return z,b,int(beta0),int(beta1)

def cat(*a):
    return np.concatenate(a,axis=1)

@lru_cache(maxsize=65536)
def local_span(left, right, bridge='union'):
    zl,bl,b0l,b1l = chains(int(left))
    zr,br,b0r,b1r = chains(int(right))
    if bridge == 'union':
        mid = left | right
        zu,bu,_,b1u = chains(mid)
        rb = rank(bu)
        rl = rank(cat(bu,zl))-rb
        rr = rank(cat(bu,zr))-rb
        rjoint = rank(cat(bu,zl,zr))-rb
        common = rl+rr-rjoint
        bridge_only = b1u-rjoint
    elif bridge == 'intersection':
        mid = left & right
        zi,bi,_,b1u = chains(mid)
        rl = rank(cat(bl,zi))-rank(bl)
        rr = rank(cat(br,zi))-rank(br)
        boundaries = np.zeros((10,bl.shape[1]+br.shape[1]),dtype=np.uint8)
        boundaries[:5,:bl.shape[1]] = bl
        boundaries[5:,bl.shape[1]:] = br
        simultaneous = np.concatenate([zi,zi],axis=0)
        rjoint = rank(cat(boundaries,simultaneous))-rank(boundaries)
        common = rl+rr-rjoint
        bridge_only = b1u-rjoint
    else:
        raise ValueError('bridge must be union or intersection')
    defect = b1l+b1r-2*common
    assert common >= 0 and common <= min(b1l,b1r)
    assert defect >= 0 and bridge_only >= 0
    return {'left_beta0':b0l,'right_beta0':b0r,'left_beta1':b1l,'right_beta1':b1r,
            'common_rank':int(common),'D_checkpoint':int(defect),
            'bridge_only':int(bridge_only),
            'nonnested':bool(left&~right and right&~left)}

def self_test():
    base = ['vF','vD','vP','vR','eFD','eFP','eFR']
    l = mask_from_cells(base+['eDP'])
    r = mask_from_cells(base+['eRP'])
    cases = {'identical':(l,l,0,0), 'replacement':(l,r,2,0),
             'fill':(l,l|(1<<9),1,0),
             'delete':(l,l&~(1<<5),1,0)}
    x = mask_from_cells(['vF','vD','vP','eFD','eDP'])
    y = mask_from_cells(['vF','vD','vP','eDP','eFP'])
    cases['union_bridge_only'] = (x,y,0,1)
    out = {}
    for name,(a,b,d,u) in cases.items():
        val = local_span(a,b,'union')
        assert (val['D_checkpoint'],val['bridge_only']) == (d,u), (name,val)
        out[name] = val
    legal = [m for m in range(1<<11) if is_closed(m)]
    # Exhaustively check identity and symmetry on every legal pair of states.
    for a in legal:
        for bridge in ('union','intersection'):
            assert local_span(a,a,bridge)['D_checkpoint'] == 0
        for b in legal:
            for bridge in ('union','intersection'):
                u,v = local_span(a,b,bridge),local_span(b,a,bridge)
                assert u['D_checkpoint'] == v['D_checkpoint']
                assert u['bridge_only'] == v['bridge_only']
    return {'cases':out,'legal_subcomplexes':len(legal),'all_pair_checks':len(legal)**2*2}

if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False,indent=2))
