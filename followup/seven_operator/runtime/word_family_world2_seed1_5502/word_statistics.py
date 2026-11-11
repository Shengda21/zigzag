"""One frozen look, separate simultaneous family for six inherited sources."""
import itertools as it,json,math
from pathlib import Path
from scipy.stats import beta
from iid_probes import A,B
TAGS=('source','two','three','exchange1','clean1','exchange2','clean2')
HIGH=('ABCD','ABDE','ACEF','ABCDE','ACDEG','BCDEF')
def analyze(root):
    cfg=json.loads((root/'case.json').read_text());out={};gates={};contrasts={};tail=.025/(2*252)
    for tag in TAGS:
        v=json.loads((root/f'probe_{tag}/analysis.json').read_text());r={};bad=set()
        if tag.startswith('exchange'):bad=A if cfg['order'][int(tag[-1])-1]=='A' else B
        for c in HIGH:
            z=v['four' if len(c)==4 else 'five'];flags=z['flags'][c]['strict'];assert len(flags)==128
            k=sum(flags);bounds=[0. if not k else float(beta.ppf(tail,k,129-k)),1. if k==128 else float(beta.ppf(1-tail,k+1,128-k))]
            positive=tag not in ('source','two') and not(set(map(''.join,it.combinations(c,3)))&bad)
            gates[tag+':'+c]=bounds[0]>=.8 if positive else bounds[1]<=.2
            r[c]={'strict_pairs':k,'n':128,'score':k/128,'bounds':bounds,'predicted_positive':positive,'flags':flags}
        out[tag]=r
    def contrast(name,values):
        mean=sum(values)/128;lower=mean-math.sqrt(2*math.log(48/.025)/128);contrasts[name]={'mean':mean,'lower':lower};gates[name]=lower>.3
    for c in HIGH:contrast('three-minus-two:'+c,[a-b for a,b in zip(out['three'][c]['flags'],out['two'][c]['flags'])])
    for phase,letter in enumerate(cfg['order'],1):
        high,low=('ABCD','ABDE') if letter=='A' else ('ABDE','ABCD');r=out[f'exchange{phase}'];contrast(f'crossover:{phase}',[a-b for a,b in zip(r[high]['flags'],r[low]['flags'])])
    result={'case':cfg['case'],'world':cfg['world'],'order':cfg['order'],'n':128,'joint_word_family_case_pass':all(gates.values()),'gates':gates,'failed_gates':[k for k,v in gates.items() if not v],'endpoints':out,'paired_contrasts':contrasts,'uncertainty':{'scores':252,'contrasts':48,'score_family_error':.025,'contrast_family_error':.025,'looks':1,'scope':'Separate simultaneous 95% family for this evaluation, conditional on inherited fixed models and stated generator. No seed-population inference or joint 95% claim across assessments.'}}
    (root/'RESULT.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':print(json.dumps(analyze(Path(__file__).resolve().parent)),flush=True)
