"""Finite-complex design verification only; no neural experimental evidence."""
import itertools as it,json,math,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent


def insert(basis,value):
    while value:
        k=value.bit_length()-1
        if k not in basis:basis[k]=value;return True
        value^=basis[k]
    return False


def member(basis,value):
    while value:
        k=value.bit_length()-1
        if k not in basis:return False
        value^=basis[k]
    return True


def construction(n,r):
    assert r>=1 and n>=6*r+1
    unique_a=set();unique_b=set();shared=set();gadgets=[];d=n-1
    def face(*v):return tuple(sorted(v))
    for j in range(r):
        a,b,c,e,f,g=range(6*j,6*j+6)
        unique_a.update((face(a,b,e),face(a,c,f),face(b,c,g),face(e,f,g)))
        unique_b.update((face(a,b,c),face(a,e,f),face(b,e,g),face(c,f,g)))
        shared.update(face(a,f,v) for v in range(n) if v not in (a,c,e,f))
        gadgets.append((a,b,c,d,e,f,g))
    assert not(unique_a&unique_b or shared&unique_a or shared&unique_b)
    return unique_a|shared,unique_b|shared,gadgets


def verify(n,r):
    tick=time.monotonic();all_faces=list(it.combinations(range(n),3));edges=list(it.combinations(range(n),2));fi={f:i for i,f in enumerate(all_faces)};ei={e:i for i,e in enumerate(edges)}
    def boundary(value):
        out=0
        while value:
            bit=value&-value;face=all_faces[bit.bit_length()-1];value^=bit
            for e in it.combinations(face,2):out^=1<<ei[e]
        return out
    def algebra(faces):
        pivots={};cycles=[]
        for face in sorted(faces):
            v=1<<fi[face];col=boundary(v)
            while col:
                k=col.bit_length()-1
                if k not in pivots:pivots[k]=(col,v);break
                old,chain=pivots[k];col^=old;v^=chain
            if not col:cycles.append(v)
        return {'betti':[1,len(edges)-n+1-len(pivots),len(cycles)],'cycles':cycles,'boundaries':{k:v[0] for k,v in pivots.items()}}
    deleted_a,deleted_b,gadgets=construction(n,r);faces=set(all_faces);a=faces-deleted_a;b=faces-deleted_b
    for size in (1,2):assert Counter(sub for f in deleted_a for sub in it.combinations(f,size))==Counter(sub for f in deleted_b for sub in it.combinations(f,size))
    actual={name:algebra(f) for name,f in (('A',a),('B',b),('intersection',a&b),('union',a|b))};h=math.comb(n-1,3);expected={'A':[1,0,h-r*n],'B':[1,0,h-r*n],'intersection':[1,r,h-r*n-3*r],'union':[1,0,h-r*n+4*r]};assert all(actual[k]['betti']==v for k,v in expected.items());span={}
    for v in actual['A']['cycles']+actual['B']['cycles']:insert(span,v)
    assert actual['union']['betti'][2]-len(span)==r;quotient=dict(span);delta_span=dict(actual['intersection']['boundaries']);witnesses=[]
    for aa,bb,cc,dd,ee,ff,gg in gadgets:
        q=tuple(sorted((aa,cc,ee,ff)));parts={f for f in it.combinations(q,3)};canonical=sum(1<<fi[f] for f in parts);split_a=sum(1<<fi[f] for f in parts if f in a);split_b=canonical^split_a;delta=boundary(split_a)
        assert boundary(canonical)==0 and not member(span,canonical) and insert(quotient,canonical);assert boundary(split_b)==delta and not member(actual['intersection']['boundaries'],delta) and insert(delta_span,delta)
        assert all(f in a|b for f in parts) and not parts<=a and not parts<=b
        qabcd=set(it.combinations(sorted((aa,bb,cc,dd)),3));qabde=set(it.combinations(sorted((aa,bb,dd,ee)),3));assert qabcd<=a and not qabcd<=b and qabde<=b and not qabde<=a
        witnesses.append({'ACEF_in_union_not_endpoints':True,'canonical_cycle_outside_endpoint_span':True,'independent_connecting_image':True,'ABCD_ABDE_boundary_crossover':True})
    assert len(quotient)-len(span)==len(delta_span)-len(actual['intersection']['boundaries'])==r
    return {'n':n,'r':r,'deleted_faces_each':len(deleted_a),'exact_vertex_pair_balance':True,'actual_betti':{k:v['betti'] for k,v in actual.items()},'endpoint_span_rank':len(span),'connecting_rank':r,'independent_canonical_witnesses':witnesses,'seconds':time.monotonic()-tick}


def main():
    configs=[(n,1) for n in range(7,13)]+[(13,2),(14,2),(19,3),(25,4)];results=[verify(n,r) for n,r in configs];out={'status':'PASS_FINITE_COMPLEX_FAMILY','scope':'A construction from classical pair-balanced Pasch trades and shared missing edge fans. Formula/finite-complex checks, not neural results and not an established novelty claim.','cases':results};(HERE/'BALANCED_RANK_FAMILY.json').write_text(json.dumps(out,indent=2));print(json.dumps({'status':out['status'],'cases':len(results),'max_n':max(n for n,r in configs),'max_r':max(r for n,r in configs),'seconds':sum(v['seconds'] for v in results)},indent=2))


if __name__=='__main__':main()
