import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(HERE))
from topology_math import betti


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--higher-only',action='store_true');a=p.parse_args();runtime=json.loads((HERE/f'probe_{a.tag}/runtime.json').read_text());first=runtime['item_start'];last=runtime['item_stop'];result={'item_start':first,'item_stop':last}
    for kind in (('four','five') if a.higher_only else ('topology','four','five')):
        path=HERE/f'{kind}_bank.jsonl';bank=[json.loads(l) for l in path.read_text().splitlines()];bank=[r for r in bank if first<=r['item']<last];groups=defaultdict(dict);statuses=Counter();seen=set()
        for line in (HERE/f'probe_{a.tag}'/f'predictions_{kind}.jsonl').read_text().splitlines():
            z=json.loads(line);i=z['source_index'];r=bank[i];assert i not in seen and r['id']==z['id'];seen.add(i);gold=[list(map(int,s.split())) for s in r['answer'].split(' | ')];prefix=z['states'][:r['length']]==gold;strict=prefix and z['history']==r['word'] and len(z['states'])==r['length'] and z['status']=='self_stopped';groups[r['cell'],r['item']][r['member']]={'prefix':int(prefix),'strict':int(strict)};statuses[z['status']]+=1
        assert len(seen)==len(bank);cells={};flags={}
        for c in sorted({r['cell'] for r in bank}):
            n=last-first;pf=[];sf=[]
            for item in range(first,last):
                d=groups[c,item];assert set(d)==({0} if len(c)==1 else {0,1});pf.append(int(all(v['prefix'] for v in d.values())));sf.append(int(all(v['strict'] for v in d.values())))
            cells[c]={'items':n,'paired_prefix':sum(pf),'strict_pairs':sum(sf),'prefix_score':sum(pf)/n,'strict_score':sum(sf)/n};flags[c]={'prefix':pf,'strict':sf}
        result[kind]={'cells':cells,'flags':flags,'statuses':dict(statuses)}
    if 'topology' in result:
        scores={c:v['prefix_score'] for c,v in result['topology']['cells'].items()};result.update(scores=scores,betti80=betti(scores))
    (HERE/f'probe_{a.tag}/analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({'tag':a.tag,'betti80':result.get('betti80'),'higher_strict':{k:{c:v['strict_score'] for c,v in result[k]['cells'].items()} for k in ('four','five')}}),flush=True)


if __name__=='__main__':main()
