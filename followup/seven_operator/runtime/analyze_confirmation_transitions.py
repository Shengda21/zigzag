"""Descriptive own-state decomposition; no training, queries, or changed gates."""
import argparse,importlib.util,json
from collections import Counter,defaultdict
from pathlib import Path

TAGS=('source','two','three','exchange1','clean1','exchange2','clean2')
A={'ABE','ACF','BCG','EFG','ABF','ADF','AFG'}
B={'ABC','AEF','BEG','CFG','ABF','ADF','AFG'}
DELTA=(1,3,5,7,1,3,5,7)


def parse(value):
    if value['terminator_id']!=151645:return None
    digits=value['text'].split(' ')
    if len(digits)!=8 or any(len(d)!=1 or d not in '01234567' for d in digits):return None
    return tuple(map(int,digits))


def analyze(root):
    root=Path(root);cfg=json.loads((root/'case.json').read_text());final=json.loads((root/'FINAL_RESULT.json').read_text());spec=importlib.util.spec_from_file_location('functional_rules',root/'rules.py');rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules);coef=rules.coefficients(cfg['world']);reports={}
    for tag in TAGS:
        cells=defaultdict(Counter);operations=defaultdict(Counter);formats=Counter();programs=0
        changed=(A if cfg['order'][int(tag[-1])-1]=='A' else B) if tag.startswith('exchange') else set()
        for look in range(1,final['final_look']+1):
            folder=root/f'probe_{tag}_L{look}';runtime=json.loads((folder/'runtime.json').read_text())
            for kind in ('topology','four','five'):
                bank=[json.loads(line) for line in (root/f'{kind}_bank.jsonl').read_text().splitlines()];bank=[r for r in bank if runtime['item_start']<=r['item']<runtime['item_stop']]
                with (folder/f'predictions_{kind}.jsonl').open() as source:
                    for line in source:
                        r=json.loads(line);item=bank[r['source_index']];assert item['id']==r['id'];x=tuple(item['root']);history='';pi=ri=0;programs+=1
                        for choice in r['policy_outputs']:
                            g=choice['text']
                            if g=='STOP':break
                            assert len(g)==1 and g in 'ABCDEFG';p=r['primitive_outputs'][pi];pi+=1;z=parse(p);formats['primitive_requests']+=1
                            if z is None:formats['primitive_invalid']+=1;break
                            y=parse(r['residual_outputs'][ri]);ri+=1;formats['residual_requests']+=1
                            if y is None:formats['residual_invalid']+=1;break
                            f=''.join(sorted(history[-2:]+g)) if len(history)>=2 and len(set(history[-2:]+g))==3 else 'inactive';base=rules.primitive(x,g,cfg['world']);old=coef[f] if f!='inactive' else (0,)*8
                            expected=(0,)*8 if tag in ('source','two') else tuple(a^d for a,d in zip(old,DELTA)) if f in changed else old
                            effective=tuple(a^b for a,b in zip(y,base));residual=tuple(a^b for a,b in zip(y,z));old_next=tuple(a^b for a,b in zip(base,old));expected_next=tuple(a^b for a,b in zip(base,expected))
                            group='changed' if f in changed else 'inactive' if f=='inactive' else 'retained_active'
                            for key in (f'{kind}:{f}',f'{kind}:group:{group}','all:group:'+group):
                                c=cells[key];c['n']+=1;c['primitive_correct']+=int(z==base);c['residual_correction_correct']+=int(residual==expected);c['effective_correction_correct']+=int(effective==expected);c['coherent_transition_correct']+=int(y==expected_next);c['original_rule_transition_correct']+=int(y==old_next)
                            c=operations[g];c['n']+=1;c['primitive_correct']+=int(z==base);x=y;history+=g
                        assert pi==len(r['primitive_outputs']) and ri==len(r['residual_outputs'])
        reports[tag]={'programs':programs,'formats':dict(formats),'by_operator':{k:dict(v) for k,v in operations.items()},'groups':{k:dict(v) for k,v in cells.items()},'functional_reference':'identity correction' if tag in ('source','two') else 'original correction with the prescribed phase replacements'}
    out={'case':cfg['case'],'status':'DESCRIPTIVE_OWN_STATE_TRANSITIONS_RECONSTRUCTED','endpoints':reports,'scope':'Steps and reused queries are correlated observations. No step-level confidence interval, new scientific gate, new query or relabelling of the original capability scores is introduced.'};(root/'FUNCTIONAL_TRANSITIONS.json').write_text(json.dumps(out,indent=2));return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);args=ap.parse_args();out=analyze(args.study);print(json.dumps({'case':out['case'],'status':out['status'],'endpoints':len(out['endpoints'])}),flush=True)


if __name__=='__main__':main()
