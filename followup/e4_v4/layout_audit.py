"""CPU semantic oracle and output layout audit, no model execution."""
from generate import HERE,CONFIG,rows,dump,single,spec,specify_destinations
from spec_world import World
import zz_score as Z

def main():
    checks=[]
    for wid in CONFIG['pilot']:
        d=HERE/'data/pilot'/f'world{wid}';w=World(spec(),wid,wid)
        count=0;external_count=0;rehearsal_count=0
        for file in ['edge.jsonl','joint.jsonl','PROBES.jsonl','EXTERNAL.jsonl']:
            for r in rows(d/file):
                fields=r['gold'].split('|')
                assert len(fields)==4 and fields[-1]=='1' and r['schema'][-1]=='binary'
                assert all(Z.field_valid(v,k,2) for v,k in zip(fields,r['schema']))
                assert all(ord(c)<128 for c in r['prompt'])
                if r['kind']=='edge':
                    rr=[tuple(x) for x in r['field_roots']]; assert len(set(rr))==3
                    e=r['task'];ed=w.edges[e]
                    expected=[w.view(ed['dst'],w.apply(e,x)) for x in rr]
                    if r['split']=='train':
                        oracle=specify_destinations(single(w,e,tuple(r['root']),r['id'],tuple(r['other']),tuple(r['third']),'train'))
                        assert r['prompt']==oracle['prompt'];rehearsal_count+=1
                elif r['kind'] in ('mixed','reference'):
                    rr=[tuple(x) for x in r['field_roots']];assert len(set(rr))==(2 if r['kind']=='reference' else 3)
                    if r['kind']=='reference':
                        if r['reference_position']==2:
                            assert r['rules']==['AB','BC','AD'] and rr[0]==rr[1]!=rr[2]
                            assert 'Slot 2: rule BC; operand output 1.' in r['prompt']
                        else:
                            assert r['reference_position']==3 and r['rules']==['AB','BC','CD'] and rr[0]!=rr[1]==rr[2]
                            assert 'Slot 3: rule CD; operand output 2.' in r['prompt']
                        assert r['prompt'].count('operand output ')==1
                    expected=[w.view(e[-1],w.apply(e,x)) for e,x in zip(r['rules'],rr)]
                elif r['split']=='external':
                    cur=tuple(r['root']);expected=[]
                    for e in ['AB','BC','CD']:
                        cur=w.apply(e,cur);expected.append(w.view(e[-1],cur))
                    assert r['schema']==['B','C','D','binary'];external_count+=1
                else:
                    a,b,c=w.faces[r.get('face',r['task'])]['path'];root=tuple(r['root'])
                    mid=w.apply(a+b,root)
                    second=tuple(r['other']) if r['kind']=='edge_bundle' else mid
                    expected=[w.view(b,mid),w.view(c,w.apply(b+c,second)),w.view(c,w.apply(a+c,tuple(r['third'])))]
                assert fields[:3]==expected
                assert not Z.score(r,'|'.join(fields[:3]))['correct']
                assert not Z.score(r,r['gold']+'|1')['correct']
                assert not Z.score(r,'|'.join(fields[:3]+['0']))['correct']
                count+=1
        checks.append(dict(world=wid,rows=count,external_rows=external_count,shared_rehearsal_rows=rehearsal_count,all_pass=True))
    dump(HERE/'OUTPUT_LAYOUT_AUDIT.json',dict(checks=checks,meaning='Independent CPU map recomputation and strict format rejection; no LM results'))
    print(checks)
if __name__=='__main__':main()
