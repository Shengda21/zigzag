"""Pre-specified conditional simultaneous bounds; no confirmation data here."""
import itertools as it,json,math
from pathlib import Path
from scipy.stats import beta
from topology_math import betti
from iid_probes import A,B
from chain_witness import witness,cycles2,insert,member,F
SCORES=69*7*6;CONTRASTS=8*6;ETA=.05/4
TAGS=('source','two','three','exchange1','clean1','exchange2','clean2')
HIGH=('ABCD','ABDE','ACEF','ABCDE','ACDEG','BCDEF')


def interval(k,n):
    tail=ETA/(2*SCORES)
    return (0. if not k else float(beta.ppf(tail,k,n-k+1)),1. if k==n else float(beta.ppf(1-tail,k+1,n-k)))


def paired_lower(d):return sum(d)/len(d)-math.sqrt(2*math.log(CONTRASTS/ETA)/len(d))


def closed(scores,theta):
    return {c for c in scores if len(c)<=3 and all(scores[''.join(f)]>=theta for k in range(1,len(c)+1) for f in it.combinations(c,k))}


def kappa_pair(first,second,high,low):
    hi=closed(first,high)|closed(second,high);lo=[closed(s,low) for s in (first,second)];cyc=cycles2({c for c in hi if len(c)==3});end=[cycles2({c for c in cs if len(c)==3}) for cs in lo];out={};both=None
    for name,vectors in [('left',end[0]),('right',end[1]),('both',end[0]+end[1])]:
        base={}
        for v in vectors:insert(base,v)
        augmented=dict(base)
        for v in cyc:insert(augmented,v)
        out[name]=len(augmented)-len(base)
        if name=='both':both=base
    target_faces=set(map(''.join,it.combinations('ACEF',3)));target=sum(1<<F.index(f) for f in target_faces);out['ACEF_high_cycle_outside_low_endpoint_span']=target_faces<=hi and not member(both,target)
    return out


def robust_certificates(reports):
    theta=.8;delta=.01;eps=max(max(v['scores'][c]-lo,hi-v['scores'][c]) for v in reports.values() for c,(lo,hi) in v['bounds'].items() if len(c)<=3);determined=all(hi<theta-delta or lo>=theta+delta for v in reports.values() for c,(lo,hi) in v['bounds'].items() if len(c)<=3);widened=delta+eps;admissible=theta-widened>=0 and theta+widened<=1;actual=kappa_pair(reports['exchange1']['scores'],reports['exchange2']['scores'],theta,theta);learning=kappa_pair(reports['source']['scores'],reports['three']['scores'],theta,theta);profile={'exchange_birth2':actual['left']-actual['both'],'exchange_death2':actual['right']-actual['both'],'exchange_transition2':actual['both'],'learning_birth2':learning['left']-learning['both'],'learning_transition2':learning['both']}
    path=['source','three','exchange1','exchange2'];history=set();supported={}
    for tag in path:
        cs=closed(reports[tag]['scores'],theta);history|=cs
        for v in cycles2({c for c in cs if len(c)==3}):insert(supported,v)
    profile['full_history_alpha2']=len(cycles2({c for c in history if len(c)==3}))-len(supported)
    widened_exchange=kappa_pair(reports['exchange1']['scores'],reports['exchange2']['scores'],theta+widened,theta-widened) if admissible else None;widened_learning=kappa_pair(reports['source']['scores'],reports['three']['scores'],theta+widened,theta-widened) if admissible else None
    return {'center':theta,'target_half_window':delta,'all_cell_statuses_determined_by_simultaneous_bands_throughout_window':determined,'exact_robust_counts_if_determined':profile if determined else None,'uniform_cell_score_error_from_same_bands':eps,'estimated_half_window_for_stability_lower_bound':widened,'widened_window_admissible':admissible,'exchange_kappa_lower_bounds_via_stability':widened_exchange,'learning_kappa_lower_bounds_via_stability':widened_learning,'scope':'Exact derived birth/death counts use the fully determined diagram, not subtraction of two lower confidence bounds. The historical path excludes separate matched-control branches.'}


def combine(root,tag,look):
    parts=[json.loads((root/f'probe_{tag}_L{k}/analysis.json').read_text()) for k in range(1,look+1)];flags={};scores={};bounds={}
    for kind in ('topology','four','five'):
        for cell in parts[0][kind]['flags']:
            field='prefix' if kind=='topology' else 'strict';f=[x for p in parts for x in p[kind]['flags'][cell][field]];assert len(f)==256*look;flags[cell]=f;scores[cell]=sum(f)/len(f);bounds[cell]=interval(sum(f),len(f))
    assert len(scores)==69
    return {'scores':scores,'bounds':bounds,'flags':flags}


def analyze(root,look):
    assert look in (1,2);cfg=json.loads((root/'case.json').read_text());r={tag:combine(root,tag,look) for tag in TAGS};gates={};point_gates={};expectations={};contrasts={};failures=[]
    def require(name,value,point=None):gates[name]=bool(value);point_gates[name]=bool(value if point is None else point);failures.extend([] if value else [name])
    for tag,v in r.items():
        require(tag+':vertices',all(v['bounds'][c][0]>=.95 for c in v['scores'] if len(c)==1),all(v['scores'][c]>=.95 for c in v['scores'] if len(c)==1));require(tag+':edges',all(v['bounds'][c][0]>=.85 for c in v['scores'] if len(c)==2),all(v['scores'][c]>=.85 for c in v['scores'] if len(c)==2))
        bad=set()
        if tag.startswith('exchange'):bad=A if cfg['order'][int(tag[-1])-1]=='A' else B
        for cell in v['scores']:
            if len(cell)==3:
                absent=tag in ('source','two') or cell in bad;ceiling=.79 if tag in ('source','two') else .6;require(tag+':face:'+cell,(v['bounds'][cell][1]<ceiling) if absent else v['bounds'][cell][0]>=.81,v['scores'][cell]<ceiling if absent else v['scores'][cell]>=.81)
        predictions={c:tag not in ('source','two') for c in HIGH}
        if bad:
            for cell in HIGH:predictions[cell]=not bool(set(map(''.join,it.combinations(cell,3)))&bad)
        expectations[tag]=predictions
        for cell,positive in predictions.items():require(tag+':higher:'+cell,v['bounds'][cell][0]>=.8 if positive else v['bounds'][cell][1]<=.2,v['scores'][cell]>=.8 if positive else v['scores'][cell]<=.2)
    for cell in HIGH:
        d=[x-y for x,y in zip(r['three']['flags'][cell],r['two']['flags'][cell])];lo=paired_lower(d);contrasts['three-minus-two:'+cell]={'mean':sum(d)/len(d),'lower':lo};require('gain:'+cell,lo>.3,sum(d)/len(d)>.3)
    for j,letter in enumerate(cfg['order'],1):
        high,low=('ABCD','ABDE') if letter=='A' else ('ABDE','ABCD');f=r[f'exchange{j}']['flags'];d=[x-y for x,y in zip(f[high],f[low])];lo=paired_lower(d);contrasts[f'exchange{j}:{high}-minus-{low}']={'mean':sum(d)/len(d),'lower':lo};require(f'crossover:{j}',lo>.3,sum(d)/len(d)>.3)
    cells={c for c in r['three']['scores'] if len(c)<=3};complexes=[]
    for tag in ('exchange1','exchange2'):
        s=r[tag]['scores'];complexes.append(closed(s,.8))
    def b(complex_cells):return betti({c:float(c in complex_cells) for c in cells})
    lower={c for c in cells if len(c)<=2};full_lower=all(lower<=c for c in complexes);actual={'endpoint1':b(complexes[0]),'endpoint2':b(complexes[1]),'intersection':b(complexes[0]&complexes[1]),'union':b(complexes[0]|complexes[1]),'valid_full_lower_skeleton':full_lower};require('actual_algebra',full_lower and actual['endpoint1']==actual['endpoint2']==[1,0,13] and actual['intersection']==[1,1,10] and actual['union']==[1,0,17]);states=[{c for c in cs if len(c)==3} for cs in complexes];chains=witness(states[cfg['order'].index('A')],states[cfg['order'].index('B')]) if full_lower else {'all_witness_checks_pass':False,'reason':'Measured lower skeleton is incomplete.'};require('actual_labelled_chain_witness',chains['all_witness_checks_pass'])
    out={'case':cfg['case'],'world':cfg['world'],'order':cfg['order'],'look':look,'n':256*look,'joint_confirmation_case_pass':all(gates.values()),'gates':gates,'failed_gates':failures,'endpoints':r,'predicted_higher_support':expectations,'paired_contrasts':contrasts,'actual_algebra':actual,'labelled_chain_witness':chains,'uncertainty':{'scores':SCORES,'contrasts':CONTRASTS,'per_family_per_look_error':ETA,'total_error_for_two_families_two_looks':.05,'score_interval':'simultaneous two-sided Clopper-Pearson','paired_interval':'one-sided Hoeffding range[-1,1]','population':'conditional on fitted models and frozen sampling generator; no seed-population claim'}}
    out['point_estimate_gates_pass']=all(point_gates.values());out['point_estimate_gates']=point_gates
    out['robust_window_certificates']=robust_certificates(r)
    (root/f'CONFIRMATION_LOOK{look}.json').write_text(json.dumps(out,indent=2));return out
