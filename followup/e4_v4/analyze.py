"""Strong labeled-score controls and independent external behavior, world splits."""
import argparse,collections,itertools,json,sys
from generate import HERE,CONFIG,rows,dump,sha,spec
sys.path.insert(0,str(HERE/'vendor'))
import cw_zigzag as CW
import zz_score as Z
def mean(a):return sum(a)/len(a)
def extract(rs,previous=None):
    tasks=sorted([e['name'] for e in spec()['edges']]+[f['name'] for f in spec()['faces']])
    g=collections.defaultdict(list);items=collections.defaultdict(dict);mapitems=collections.defaultdict(dict);operands=collections.defaultdict(dict);external=[];external_scores=[]
    for r in rs:
        s=Z.score(r,r['prediction'])
        if r['split']=='external':external.append(s['correct']);external_scores.append(s);continue
        g[r['task']].append(s)
        items[r['task']][r['trial']]=s['correct']
        parts=Z.parse(r['prediction'])
        mapitems[r['task']][r['trial']]=[int(j<len(parts) and Z.field_valid(parts[j],r['schema'][j],2) and bool(s['field_correct'][j])) for j in range(3)]
    assert len(external)==512 and all(len(g[t])==512 for t in tasks+['mixed_BCD','reference_twohop'])
    scores={t:mean([r['correct'] for r in g[t]]) for t in tasks}
    raw=list(itertools.chain.from_iterable([mean([r['correct'] for r in g[t]]),mean([r['valid_format'] for r in g[t]])]+[mean([r['field_correct'][j] for r in g[t]]) for j in range(4)] for t in tasks+['mixed_BCD','reference_twohop']))
    es=[scores[e['name']] for e in spec()['edges']]
    conjunction=[];strict=[]
    for f in spec()['faces']:
        # World has identity index permutations; the same root indexes the
        # actual intermediate for the second leg. Never multiply marginals.
        ab,bc,ac=f['cycle']
        conjunction.append(mean([int(mapitems[ab][trial][0] and mapitems[bc][trial][0] and mapitems[ac][trial][1]) for trial in items[f['name']]]))
        strict.append(mean([int(items[ab][trial] and items[bc][trial] and items[ac][trial]) for trial in items[f['name']]]))
    cells=[(v,0,[]) for v in 'ABCD']+[(e['name'],1,[e['src'],e['dst']]) for e in spec()['edges']]+[(f['name'],2,f['cycle']) for f in spec()['faces']]
    cx=CW.CWComplex(cells);s={**{v:1. for v in 'ABCD'},**scores}
    betti=[];masks=[]
    for theta in [.6,.7,.8,.9]:
        mask=cx.mask_from_scores(s,theta);assert cx.is_closed(mask)
        betti.extend((cx.betti(mask)+[0,0,0])[:3]);masks.append(mask)
    old=previous or dict(raw=raw,masks=masks,betti=betti)
    # Temporal changes use only earlier/current checkpoints, never future target.
    changes=[]
    for a,b in zip(old['masks'],masks):
        changes.extend([(b&~a).bit_count(),(a&~b).bit_count()])
    feat=dict(raw=raw+conjunction,raw_temporal=raw+conjunction+[x-y for x,y in zip(raw,old['raw'])],
              edge=es+[min(es),max(es),mean(es)],betti=betti,
              temporal=betti+[x-y for x,y in zip(betti,old['betti'])]+changes,conjunction=es+conjunction)
    return dict(features=feat,target=mean(external),raw=raw,betti=betti,masks=masks,scores=scores,
                conjunction={f['name']:v for f,v in zip(spec()['faces'],conjunction)},
                diagnostics={t:dict(exact=mean([q['correct'] for q in gg]),strict_format=mean([q['valid_format'] for q in gg]),structure=mean([q['four_field_structure'] for q in gg]),syntax=mean([q['code_syntax'] for q in gg]),sentinel_one=mean([q['sentinel_one'] for q in gg]),destination_signatures=dict(collections.Counter(q['destination_signature'] for q in gg))) for t,gg in g.items()},
                external_format=mean([q['valid_format'] for q in external_scores]),
                external_diagnostics=dict(structure=mean([q['four_field_structure'] for q in external_scores]),syntax=mean([q['code_syntax'] for q in external_scores]),sentinel_one=mean([q['sentinel_one'] for q in external_scores]),destination_valid=[mean([q['destination_valid'][j] for q in external_scores]) for j in range(3)],destination_signatures=dict(collections.Counter(q['destination_signature'] for q in external_scores))),
                strict_prompt_conjunction={f['name']:v for f,v in zip(spec()['faces'],strict)},
                joint_minus_conjunction={f['name']:scores[f['name']]-v for f,v in zip(spec()['faces'],conjunction)})
def collect(phase):
    out=[]
    for w in CONFIG[phase]:
        for arm in ['edge','joint']:
            run=HERE/'runs'/phase/f'world{w}_{arm}';assert (run/'COMPLETE.json').exists(),run
            prev=None
            for step in range(0,513,32):
                p=run/f'eval_{step:04d}.jsonl';q=extract(rows(p),prev);prev=q
                out.append(dict(world=w,arm=arm,step=step,source_sha256=sha(p),**q))
    return out
def fit_predict(train,test,key,candidate):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor
    model=make_pipeline(StandardScaler(),Ridge(alpha=[.01,1,100][candidate])) if candidate<3 else ExtraTreesRegressor(n_estimators=200,max_depth=3 if candidate==3 else None,min_samples_leaf=2,random_state=42,n_jobs=1)
    model.fit([r['features'][key] for r in train],[r['target'] for r in train])
    return np.clip(model.predict([r['features'][key] for r in test]),0,1).tolist()
def paired(final,phase):
    import numpy as np
    by={(r['world'],r['arm']):r for r in final if r['step']==512}
    diffs=[];ed=[]
    for w in CONFIG[phase]:
        a,b=by[w,'edge'],by[w,'joint'];diffs.append(b['target']-a['target'])
        ed.append([b['scores'][e['name']]-a['scores'][e['name']] for e in spec()['edges']])
    rng=np.random.default_rng(42);ix=rng.integers(0,len(diffs),(10000,len(diffs)))
    d=np.array(diffs)[ix].mean(1);e=np.array(ed)[ix].mean(1)
    ci=np.quantile(e,[.05/6,1-.05/6],axis=0).tolist()
    eq=all(-.05<ci[0][j] and ci[1][j]<.05 for j in range(6))
    bci=np.quantile(d,[.025,.975]).tolist()
    return dict(external_paired_difference=float(np.mean(diffs)),external_world_bootstrap_95=bci,
                per_world_differences=diffs,edge_difference_simultaneous_90_bonferroni=ci,edge_equivalence_margin=.05,
                edge_equivalence_pass=eq,external_effect_pass=bool(bci[0]>.05),
                interpretation='Small number of worlds: bootstrap intervals descriptive. Claim at most intervention effect conditional on edge equivalence; no mechanism identification.')
def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','confirm'],required=True);a=p.parse_args()
    pilot=collect('pilot');current=pilot if a.phase=='pilot' else collect('confirm')
    keys=list(pilot[0]['features']);out=HERE/'analysis'/a.phase;out.mkdir(parents=True,exist_ok=True)
    if a.phase=='pilot':
        selected={}
        for key in keys:
            losses=[]
            for c in range(5):
                error=[]
                for w in CONFIG['pilot']:
                    tr=[r for r in pilot if r['world']!=w];te=[r for r in pilot if r['world']==w]
                    error.extend([(x-r['target'])**2 for x,r in zip(fit_predict(tr,te,key,c),te)])
                losses.append(mean(error))
            selected[key]=dict(candidate=min(range(5),key=lambda j:losses[j]),pilot_group_cv_mse=losses)
        dump(out/'SELECTION.json',selected)
    else:selected=json.loads((HERE/'analysis/pilot/SELECTION.json').read_text())
    reports={}
    for key in keys:
        pred=fit_predict(pilot,current,key,selected[key]['candidate'])
        reports[key]=dict(mse=mean([(x-r['target'])**2 for x,r in zip(pred,current)]),mae=mean([abs(x-r['target']) for x,r in zip(pred,current)]),
                          predictions=[dict(world=r['world'],arm=r['arm'],step=r['step'],observed=r['target'],predicted=x) for x,r in zip(pred,current)])
    comparisons={}
    for key in ['edge','betti','temporal']:
        per_world=[]
        for w in CONFIG[a.phase]:
            ids=[j for j,r in enumerate(current) if r['world']==w]
            per_world.append(mean([(reports['raw_temporal']['predictions'][j]['predicted']-current[j]['target'])**2-(reports[key]['predictions'][j]['predicted']-current[j]['target'])**2 for j in ids]))
        rng=np.random.default_rng(42);v=np.array(per_world);b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
        comparisons[key]=dict(mse_gain_vs_raw_temporal=float(v.mean()),world_bootstrap_95=np.quantile(b,[.025,.975]).tolist(),per_world=per_world)
    if a.phase=='pilot':
        gate_rows=[]
        for r in current:
            if r['step']!=512:continue
            ed=[r['diagnostics'][e['name']] for e in spec()['edges']]
            mixed=r['diagnostics']['mixed_BCD'];ref=r['diagnostics']['reference_twohop']
            ok=all(q['strict_format']>=.90 and q['exact']>=.10 for q in ed) and mean([q['exact'] for q in ed])>=.25 and mixed['strict_format']>=.90 and mixed['exact']>=.10 and ref['strict_format']>=.90 and ref['exact']>=.10 and r['external_format']>=.90
            audit=json.loads((HERE/'data/pilot'/f"world{r['world']}"/'TOKEN_AUDIT.json').read_text())
            ok=ok and audit['per_record_token_mismatches']==0
            gate_rows.append(dict(world=r['world'],arm=r['arm'],passed=bool(ok),edges=ed,mixed=mixed,reference=ref,external_format=r['external_format'],external_exact=r['target']))
        nondegenerate=any(.01<r['external_exact']<.99 for r in gate_rows) and len({r['target'] for r in current})>=2
        dump(out/'FEASIBILITY.json',dict(passed=all(r['passed'] for r in gate_rows) and nondegenerate,rows=gate_rows,external_nondegenerate=nondegenerate,primary_step=512))
    dump(out/'FEATURES.json',current)
    dump(out/'REPORT.json',dict(phase=a.phase,pairing=paired(current,a.phase),baselines=reports,comparisons=comparisons,
         note='Pilot in-sample report is development only; confirmation models fit pilot only. 17 checkpoints are not independent replicates.'))
if __name__=='__main__':main()
