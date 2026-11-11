"""New word-pattern banks, generated before any model query."""
import itertools as it,json,random
from collections import defaultdict,Counter
from pathlib import Path
import iid_probes as iid
from rules import OPS,active_faces,coefficients,execute,reference,primitive
HERE=Path(__file__).resolve().parent
def pattern(w):
    order={}
    return ''.join(str(order.setdefault(g,len(order))) for g in w)
def pools(k):
    symbols='ABCDE'[:k];faces=list(map(''.join,it.combinations(symbols,3)));full=(1<<len(faces))-1
    excluded={pattern('ABCDABCD')} if k==4 else {pattern(w) for w in ('BACDBAEC','BDECBADE')}
    buckets=defaultdict(list)
    for chars in it.product(symbols,repeat=8):
        if any(a==b for a,b in zip(chars,chars[1:])) or len(set(chars))!=k:continue
        w=''.join(chars)
        if pattern(w) in excluded:continue
        mask=sum(1<<faces.index(f) for f in active_faces(w))
        if k==4 and mask!=full:continue
        buckets[mask].append(w)
    compatible={m:[v for v in buckets if m|v==full] for m in buckets}
    first=[(w,m) for m,words in buckets.items() if compatible[m] for w in words]
    assert len(first)==(888 if k==4 else 25680)
    return symbols,buckets,compatible,first,excluded
def build():
    cfg=json.loads((HERE/'case.json').read_text());parent=HERE.parent/cfg['primary_directory'];blocked=set();counts={}
    for role in ('primitive','identity','three'):
        for kind in ('train','test'):
            rows=[json.loads(l) for l in (parent/f'{kind}_{role}.jsonl').read_text().splitlines()]
            blocked.update(tuple(r['state']) for r in rows);counts[f'{kind}_{role}']=len(rows)
    pools_by_k={k:pools(k) for k in (4,5)}
    def programs(rng):
        out=[]
        for kind,cells,k in (('four',('ABCD','ABDE','ACEF'),4),('five',('ABCDE','ACDEG','BCDEF'),5)):
            symbols,buckets,compatible,first,excluded=pools_by_k[k]
            for cell in cells:
                w,mask=rng.choice(first);masks=compatible[mask];total=sum(len(buckets[m]) for m in masks);index=rng.randrange(total)
                for m in masks:
                    if index<len(buckets[m]):z=buckets[m][index];break
                    index-=len(buckets[m])
                mapping=dict(zip(symbols,cell));words=[''.join(mapping[g] for g in v) for v in (w,z)]
                assert all(pattern(v) not in excluded for v in words)
                assert set.union(*(active_faces(v) for v in words))==set(map(''.join,it.combinations(cell,3)))
                out.extend((kind,cell,j,v) for j,v in enumerate(words))
        return out
    iid.programs=programs
    rows,audit=iid.generate(cfg['world'],cfg['evaluation_seed'],128,blocked,include_primitive_halves=True)
    coef=coefficients(cfg['world']);families=[coef,*({f:tuple(a^b for a,b in zip(v,iid.DELTA)) if f in bad else v for f,v in coef.items()} for bad in (iid.A,iid.B))]
    patterns=Counter();paths=0
    for kind in ('four','five'):
        assert len(rows[kind])==768
        for r in rows[kind]:
            assert r['answer']==' | '.join(' '.join(map(str,s)) for s in reference(r['root'],r['word'],coef,cfg['world']))
            assert all(a!=b for a,b in zip(r['word'],r['word'][1:]));patterns[(kind,pattern(r['word']))]+=1
            for family in families:
                trace=reference(r['root'],r['word'],family,cfg['world']);assert trace==execute(r['root'],r['word'],family,cfg['world'])
                assert all(tuple(x) not in blocked and primitive(x,g,cfg['world']) not in blocked for x,g in zip([r['root'],*trace[:-1]],r['word']));paths+=1
        (HERE/f'{kind}_bank.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows[kind]))
    audit.update(input_file_counts=counts,independent_reference_paths_checked=paths,pattern_counts={k:sum(1 for kind,p in patterns if kind==k) for k in ('four','five')},pool_sizes={str(k):{'all_candidate_words':sum(map(len,p[1].values())),'eligible_first_words':len(p[3]),'excluded_patterns':sorted(p[4])} for k,p in pools_by_k.items()},old_templates_excluded_under_all_relabellings=True,all_numeric_training_and_component_test_inputs_excluded=True)
    (HERE/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':build()
