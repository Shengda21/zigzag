"""Exact outcomes and paired variety contrasts on common roots."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent


def main():
    allresults={}
    for world in ('r1603','r4004'):
        bank=[json.loads(l) for l in (HERE/f'bank_{world}.jsonl').read_text().splitlines()]
        result={}
        for route in ('lower','trained'):
            folder=HERE/f'probe_{world}_{route}';flags=defaultdict(dict);counts=defaultdict(lambda:defaultdict(int));first_errors=defaultdict(lambda:defaultdict(int));seen=set()
            for line in (folder/'predictions.jsonl').read_text().splitlines():
                p=json.loads(line);i=p['source_index'];r=bank[i];assert i not in seen and p['id']==r['id'];seen.add(i)
                raw=p['raw_generated_ids'];stop=next((j for j,t in enumerate(raw) if t in (151645,151643)),None);assert p['eos']==(None if stop is None else raw[stop])
                gold=r['answer'].split(' | ');actual=p['prediction'].split(' | ');L=r['length'];key=f'{L}:{r["variety"]}'
                strict=int(p['eos']==151645 and p['prediction']==r['answer']);prefix=int(actual[:L]==gold);final=int(len(actual)>=L and actual[L-1]==gold[-1]);flags[key,r['item']][r['member']]=strict
                for metric,v in [('strict',strict),('state_prefix',prefix),('final',final),('n',1)]:counts[key][metric]+=v
                err=next((j+1 for j in range(L) if j>=len(actual) or actual[j]!=gold[j]),L+1);first_errors[key][str(err)]+=1
            assert len(seen)==4096
            pairs={key:[int(flags[key,j]=={0:1,1:1}) for j in range(256)] for key in counts}
            result[route]={'counts':dict(counts),'strict_pairs':{k:sum(v) for k,v in pairs.items()},'pair_flags':pairs,'first_wrong_step':dict(first_errors)}
        rng=np.random.default_rng(490120261113);ix=rng.integers(0,256,(10000,256));contrasts={}
        for L in (5,8):
            for k in (3,4,5):
                d=np.array(result['trained']['pair_flags'][f'{L}:{k}'])-np.array(result['trained']['pair_flags'][f'{L}:2']);dist=d[ix].mean(1);lo,hi=np.quantile(dist,[.05/(2*6),1-.05/(2*6)])
                contrasts[f'{L}:{k}-2']={'difference':float(d.mean()),'simultaneous95_interval':[float(lo),float(hi)],'equivalent_pm05':bool(lo>-.05 and hi<.05)}
        result['variety_contrasts']=contrasts;result['joint_equivalence']=all(v['equivalent_pm05'] for v in contrasts.values());allresults[world]=result
    (HERE/'AGGREGATE.json').write_text(json.dumps(allresults,indent=2));print(json.dumps({w:{r:allresults[w][r]['strict_pairs'] for r in ('lower','trained')} for w in allresults}))


if __name__=='__main__':main()
