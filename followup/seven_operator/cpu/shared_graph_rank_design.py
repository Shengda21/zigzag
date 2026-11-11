"""Finite intervention designs with shared private vertices; no neural data.

The original registered seven-operator experiment is unchanged.  Packing
and homology are checked by separate integer incidence/boundary routines.
"""
import itertools as it
import json
import math
import time
from collections import Counter
from pathlib import Path
from verify_balanced_rank_family import insert, member

HERE = Path(__file__).resolve().parent


def private_size(r):
    p = 2
    while math.comb(p, 2) < r:
        p += 1
    return p


def sufficient_pool(r, p):
    x = max(math.ceil((48*r)**(1/3)), math.ceil(math.sqrt(32*(p-1)))) + 1
    assert x**3 > 48*r and x*x > 32*(p-1)
    return x + 2


def pack(r, q):
    p = private_size(r)
    edges = list(it.combinations(range(p), 2))[:r]
    used_triangles = set()
    used_pairs = [set() for _ in range(p)]
    blocks = []
    for a, f in edges:
        found = False
        for b, c, e, g in it.permutations(range(q), 4):
            first, second = tuple(sorted((b,c,g))), tuple(sorted((b,e,g)))
            if first in used_triangles or second in used_triangles:
                continue
            at_a = {tuple(sorted((b,e))), tuple(sorted((b,c)))}
            at_f = {tuple(sorted((e,g))), tuple(sorted((c,g)))}
            if at_a & used_pairs[a] or at_f & used_pairs[f]:
                continue
            used_triangles.update((first,second))
            used_pairs[a].update(at_a)
            used_pairs[f].update(at_f)
            blocks.append((a,b+p,c+p,e+p,f,g+p))
            found = True
            break
        if not found:
            return None
    assert len(used_triangles) == 2*r
    return edges, blocks


def construction(r):
    p = private_size(r)
    for q in range(4, sufficient_pool(r,p)+1):
        packed = pack(r,q)
        if packed is not None:
            break
    else:
        raise AssertionError('Constructive counting bound failed')
    selected_edges, blocks = packed
    n = p+q+1
    d = n-1
    ta, tb, restored = set(), set(), set()
    face = lambda *v: tuple(sorted(v))
    gadgets = []
    for a,b,c,e,f,g in blocks:
        aa = {face(a,b,e),face(a,c,f),face(b,c,g),face(e,f,g)}
        bb = {face(a,b,c),face(a,e,f),face(b,e,g),face(c,f,g)}
        assert not (ta|tb)&(aa|bb)
        ta.update(aa)
        tb.update(bb)
        restored.update((face(a,c,f),face(a,e,f)))
        gadgets.append((a,b,c,d,e,f,g))
    incident = {face(*edge,v) for edge in selected_edges for v in range(n) if v not in edge}
    common = incident-restored
    assert len(ta)==len(tb)==4*r and len(restored)==2*r
    assert not (ta&tb or common&ta or common&tb)
    degree = Counter(v for edge in selected_edges for v in edge)
    triangle_count = sum(all(edge in selected_edges for edge in it.combinations(t,2)) for t in it.combinations(range(p),3))
    assert len(common)==r*(n-4)-sum(math.comb(degree[v],2) for v in range(p))+triangle_count
    return {'n':n,'r':r,'p':p,'q':q,'deleted_a':ta|common,'deleted_b':tb|common,'common':common,'gadgets':gadgets,'private_edges':selected_edges,'sufficient_q':sufficient_pool(r,p)}


def verify(design):
    tick=time.monotonic()
    n,r=design['n'],design['r']
    faces=list(it.combinations(range(n),3));fi={f:i for i,f in enumerate(faces)}
    edges=list(it.combinations(range(n),2));ei={e:i for i,e in enumerate(edges)}
    columns={f:sum(1<<ei[e] for e in it.combinations(f,2)) for f in faces}
    def boundary(chain):
        result=0
        while chain:
            bit=chain&-chain;chain^=bit;result^=columns[faces[bit.bit_length()-1]]
        return result
    def algebra(fs):
        pivot={};cycles=[]
        for face in sorted(fs):
            col=columns[face];chain=1<<fi[face]
            while col:
                p=col.bit_length()-1
                if p not in pivot:
                    pivot[p]=(col,chain);break
                old,w=pivot[p];col^=old;chain^=w
            if not col:cycles.append(chain)
        return {'betti':[1,len(edges)-n+1-len(pivot),len(cycles)],'cycles':cycles,'boundaries':{p:v[0] for p,v in pivot.items()}}
    da,db=design['deleted_a'],design['deleted_b'];universe=set(faces);a=universe-da;b=universe-db
    for k in (1,2):
        assert Counter(e for f in da for e in it.combinations(f,k))==Counter(e for f in db for e in it.combinations(f,k))
    actual={name:algebra(fs) for name,fs in (('A',a),('B',b),('intersection',a&b),('union',a|b))}
    h=math.comb(n-1,3);m=len(design['common'])
    expected={'A':[1,0,h-m-4*r],'B':[1,0,h-m-4*r],'intersection':[1,r,h-m-7*r],'union':[1,0,h-m]}
    assert {name:v['betti'] for name,v in actual.items()}==expected
    span={}
    for chain in actual['A']['cycles']+actual['B']['cycles']:insert(span,chain)
    assert len(span)==h-m-r
    extended=dict(span);delta_span=dict(actual['intersection']['boundaries'])
    for j,(aa,bb,cc,dd,ee,ff,gg) in enumerate(design['gadgets']):
        parts=set(it.combinations(sorted((aa,cc,ee,ff)),3));q=sum(1<<fi[f] for f in parts)
        split_a=sum(1<<fi[f] for f in parts if f in a);delta=boundary(split_a)
        assert boundary(q)==0 and parts<=a|b and not parts<=a and not parts<=b
        assert not member(span,q) and insert(extended,q) and insert(delta_span,delta)
        assert [delta>>ei[e]&1 for e in design['private_edges']]==[int(i==j) for i in range(r)]
        first=set(it.combinations(sorted((aa,bb,cc,dd)),3));second=set(it.combinations(sorted((aa,bb,dd,ee)),3))
        assert first<=a and not first<=b and second<=b and not second<=a
    assert len(extended)-len(span)==len(delta_span)-len(actual['intersection']['boundaries'])==r
    return {'n':n,'r':r,'private_vertices':design['p'],'pool_vertices':design['q'],'common_deleted_faces':m,'deleted_faces_each':len(da),'betti':expected,'endpoint_image_span':len(span),'connecting_rank':r,'verified_independent_canonical_cycles_and_images':r,'verified_crossover_pairs':r,'exact_vertex_pair_balance':True,'seconds':time.monotonic()-tick}


def main():
    tick=time.monotonic();results=[]
    for r in [*range(1,17),24,32,48,64]:
        result=verify(construction(r));results.append(result);print(json.dumps(result),flush=True)
    out={'status':'PASS_SHARED_PRIVATE_GRAPH_EXACT_BOUNDARIES','scope':'CPU finite-complex design checks only. No additional neural worlds, training seeds or gates. Original seven-operator protocol unchanged. Classical Pasch trade attribution remains necessary; no first-priority claim established.','operator_bound':'n=sqrt(2r)+O(r^(1/3)); rho2<=C(n-1,2) gives the same leading lower bound.','sufficient_pool_bound':'x=q-2 with x^3>48r and x^2>32(p-1); ordered quadruple exclusion count proves a legal block remains.','cases':results,'seconds':time.monotonic()-tick}
    (HERE/'SHARED_GRAPH_RANK_DESIGN.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'status':out['status'],'cases':len(results),'seconds':out['seconds']}),flush=True)


if __name__=='__main__':main()
