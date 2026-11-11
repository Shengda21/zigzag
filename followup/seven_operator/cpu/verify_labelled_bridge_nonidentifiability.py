"""An explicit finite counterexample to identification by unlabelled zigzags.

No fitted model, numerical scores or neural predictions enter this check.
"""
import itertools as it,json
from collections import Counter
from pathlib import Path
from measure_labelled_zigzags import spaces,barcode
from verify_balanced_rank_family import insert,member
HERE=Path(__file__).resolve().parent
V='ABCDEFG';E=tuple(map(''.join,it.combinations(V,2)));F=set(map(''.join,it.combinations(V,3)));LOW=set(V)|set(E);EI={e:i for i,e in enumerate(E)}
COMMON=set('ADG AFG BCD BCE BCF BDG BFG CDF CFG DEG'.split())
EXTRA={k:set(v.split()) for k,v in {'A':'ABG ACD ACF BDE BEF CEG','B':'ABD ACF ACG BEF BEG CDE','C':'ABF ACD ACG BDE BEG CEF'}.items()}
def edge_boundary(faces):
    value=0
    for f in faces:
        for e in it.combinations(f,2):value^=1<<EI[''.join(e)]
    return value
def algebra(faces):
    pivots={}
    for f in sorted(faces):insert(pivots,edge_boundary((f,)))
    return [1,15-len(pivots),len(faces)-len(pivots)],pivots
def enc_edges(v):return [e for e,i in EI.items() if v>>i&1]
def all_bars(a,b):
    checkpoints=[LOW,LOW|F,LOW|a,LOW|b];out={}
    for bridge in ('union','intersection'):
        nodes=[]
        for i,k in enumerate(checkpoints):
            nodes.append(k)
            if i+1<len(checkpoints):nodes.append(k|checkpoints[i+1] if bridge=='union' else k&checkpoints[i+1])
        s=[spaces({f for f in k if len(f)==3}) for k in nodes];out[bridge]={str(i):barcode([v[i] for v in s],n) for i,n in ((0,1),(1,15),(2,35))}
    return out
def main():
    deleted={k:COMMON|v for k,v in EXTRA.items()};assert all(len(v)==16 for v in deleted.values())
    for size in (1,2):
        counts=[Counter(t for f in v for t in it.combinations(f,size)) for v in deleted.values()];assert counts[0]==counts[1]==counts[2]
    keep={k:F-v for k,v in deleted.items()};q=set(map(''.join,it.combinations('ABCG',3)));rows=[];full_bars=[]
    for k in ('B','C'):
        a,b=keep['A'],keep[k];I,U=a&b,a|b;ba,_=algebra(a);bb,_=algebra(b);bi,pivots=algebra(I);bu,_=algebra(U);rho=bu[2]-ba[2]-bb[2]+bi[2];assert ba==bb==[1,0,4] and bi==[1,1,1] and bu==[1,0,8] and rho==1
        assert q-a=={'ABG'} and q-b=={'ACG'} and q<=U;delta=edge_boundary(q&a);assert enc_edges(delta)==['AB','AG','BG'];outside=not member(pivots,delta);assert outside==(k=='B')
        if outside:
            # Gauge zero on the star at A. Find a dual cocycle that detects delta.
            rest=[e for e in E if not e.startswith('A')];witness=None
            for size in range(1,len(rest)+1):
                for edge_list in it.combinations(rest,size):
                    w=sum(1<<EI[e] for e in edge_list)
                    if (w&delta).bit_count()%2 and all(not((w&edge_boundary((f,))).bit_count()%2) for f in I):witness=list(edge_list);break
                if witness:break
            assert witness
            certificate={'cocycle_with_odd_pairing':witness}
        else:
            filling=next(list(t) for n in range(1,len(I)+1) for t in it.combinations(sorted(I),n) if edge_boundary(t)==delta)
            assert set(filling)<=I and edge_boundary((q&a)^set(filling))==edge_boundary((q-a)^set(filling))==0
            certificate={'intersection_filling':filling,'endpoint_cycles':[sorted((q&a)^set(filling)),sorted((q-a)^set(filling))]}
        full_bars.append(all_bars(a,b));rows.append({'pair':'A'+k,'betti':{'endpoints':[ba,bb],'intersection':bi,'union':bu},'rho2':rho,'q_outside_endpoint_supported_span':outside,'q_face_support':[sorted(q&a),sorted(q&b)],'connecting_chain':enc_edges(delta),**certificate})
    assert full_bars[0]==full_bars[1] and rows[0]['q_face_support']==rows[1]['q_face_support']
    out={'status':'PASS_EXACT_LABELLED_BRIDGE_NONIDENTIFIABILITY','scope':'Explicit finite complexes only, not additional model data. Both complete four-checkpoint diagrams have the same unlabelled union and intersection barcodes in degrees0--2; the named ABCG class has different endpoint-span membership.','common_deletions':sorted(COMMON),'additional_deletions':{k:sorted(v) for k,v in EXTRA.items()},'vertex_pair_counts_equal_for_all_three_endpoints':True,'named_boundary':'ABCG','pairs':rows,'equal_barcodes':full_bars[0]};(HERE/'LABELLED_BRIDGE_NONIDENTIFIABILITY.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='equal_barcodes'},indent=2))
if __name__=='__main__':main()
