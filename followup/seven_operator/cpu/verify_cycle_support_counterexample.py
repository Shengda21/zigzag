"""A union boundary can be supported by a sum of endpoint cycles.

This exact chain calculation is not an additional neural experiment.
"""
import itertools as it,json
from pathlib import Path
from verify_balanced_rank_family import insert,member
HERE=Path(__file__).resolve().parent
def main():
    vertices='ABCDEFG';faces=list(map(''.join,it.combinations(vertices,3)));edges=list(map(''.join,it.combinations(vertices,2)));fi={f:i for i,f in enumerate(faces)};ei={e:i for i,e in enumerate(edges)}
    def chain(items):return sum(1<<fi[f] for f in items)
    def boundary(value):
        out=0
        for f,i in fi.items():
            if value>>i&1:
                for e in it.combinations(f,2):out^=1<<ei[''.join(e)]
        return out
    def algebra(fs):
        pivots={};cycles=[]
        for f in sorted(fs):
            c=1<<fi[f];v=boundary(c)
            while v:
                k=v.bit_length()-1
                if k not in pivots:pivots[k]=(v,c);break
                w,d=pivots[k];v^=w;c^=d
            if not v:cycles.append(c)
        return {'betti':[1,15-len(pivots),len(cycles)],'cycles':cycles,'boundary_basis':{k:v[0] for k,v in pivots.items()}}
    ta={'ABE','ACF','BCG','EFG'};tb={'ABC','AEF','BEG','CFG'};allfaces=set(faces);q=chain(('ACE','ACF','AEF','CEF'));qa=chain(('ACE','AEF','CEF'));qb=chain(('ACF',));t=chain(('ACD','ADF','CDF'));delta=boundary(qa)
    assert boundary(q)==0 and qa^qb==q and boundary(qb)==boundary(t)==delta
    results=[]
    for name,common in (('bare_trade',set()),('registered_common_fan',{'ABF','ADF','AFG'})):
        a=allfaces-ta-common;b=allfaces-tb-common;sets={'A':a,'B':b,'intersection':a&b,'union':a|b};z={k:algebra(v) for k,v in sets.items()};span={}
        for v in z['A']['cycles']+z['B']['cycles']:insert(span,v)
        rho=z['union']['betti'][2]-len(span);outside=not member(span,q);nonbounding=not member(z['intersection']['boundary_basis'],delta)
        assert not {'ACE','ACF','AEF','CEF'}<=a and not {'ACE','ACF','AEF','CEF'}<=b
        assert rho==(0 if not common else 1) and outside==bool(common) and nonbounding==bool(common)
        if not common:
            assert {'ACD','ADF','CDF'}<=a&b and boundary(qa^t)==boundary(qb^t)==0 and (qa^t)^(qb^t)==q
        results.append({'design':name,'betti':{k:v['betti'] for k,v in z.items()},'canonical_boundary_complete_at_neither_endpoint':True,'endpoint_span_rank':len(span),'connecting_rank':rho,'canonical_cycle_outside_endpoint_span':outside,'connecting_chain_nonbounding_in_intersection':nonbounding})
    result={'status':'PASS_EXACT_CYCLE_SUPPORT_COUNTEREXAMPLE','scope':'Finite chain example only; no model fitting, evaluation or additional neural replication.','explicit_identity_over_F2':'q=(ACE+AEF+CEF+t)+(ACF+t), t=ACD+ADF+CDF','cases':results};(HERE/'CYCLE_SUPPORT_COUNTEREXAMPLE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
