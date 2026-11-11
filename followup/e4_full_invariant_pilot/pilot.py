"""Exploratory reuse only. No frozen analysis or manuscript is changed."""
import json,sys,itertools,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'followup/e4_v4'
sys.path.insert(0,str(SRC))
from analyze import fit_predict
from generate import spec
import cw_zigzag as C
OUT=Path(__file__).resolve().parent
sp=spec()
cx=C.CWComplex([(v,0,[]) for v in 'ABCD']+[(e['name'],1,[e['src'],e['dst']]) for e in sp['edges']]+[(f['name'],2,f['cycle']) for f in sp['faces']])
def load(p):return json.loads(p.read_text(encoding='utf8'))
def kappa(a,b,ap,bp,k,side):
    # H(union at strict threshold) -> H(union at permissive threshold),
    # modulo images of selected endpoint(s) at permissive threshold.
    u=a|b;up=ap|bp
    image=list(cx.induced(u,up,k))
    neighbors=[]
    if side in ('left','both'):neighbors+=list(cx.induced(ap,up,k))
    if side in ('right','both'):neighbors+=list(cx.induced(bp,up,k))
    return C.rank(image+neighbors)-C.rank(neighbors)
def enrich(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['world'],r['arm']].append(r)
    audit=[]
    for key,rr in groups.items():
        rr.sort(key=lambda r:r['step']);hist=[[] for _ in range(4)]
        for r in rr:
            s={**dict.fromkeys('ABCD',1.),**r['scores']};inv=[];barsfeat=[];details=[]
            for j,theta in enumerate([.6,.7,.8,.9]):
                m=cx.mask_from_scores(s,theta);assert m==r['masks'][j]
                prev=hist[j][-1] if hist[j] else m
                hist[j].append(m);union=0
                for x in hist[j]:union|=x
                values=[]
                for k in range(3):
                    supported=[v for x in hist[j] for v in cx.induced(x,union,k)]
                    alpha=len(cx.homology(union,k)[0])-C.rank(supported)
                    rho=kappa(prev,m,prev,m,k,'both')
                    win=[]
                    for delta in [0.,.025,.05]:
                        old=rr[max(0,r['step']//32-1)]
                        os={**dict.fromkeys('ABCD',1.),**old['scores']}
                        a,b=[cx.mask_from_scores(x,theta+delta) for x in (os,s)]
                        ap,bp=[cx.mask_from_scores(x,theta-delta) for x in (os,s)]
                        vals=[kappa(a,b,ap,bp,k,side) for side in ['left','right','both']]
                        if delta==0:assert vals[2]==rho
                        win+=vals
                    inv += [alpha,rho]+win
                    bars=cx.barcode(hist[j],k,'union');end=2*(len(hist[j])-1)
                    live=[end-a+1 for a,b in bars if b==end]
                    dead=[b-a+1 for a,b in bars if b<end]
                    # Prefix-only summaries: never use later barcode endpoints.
                    barsfeat += [len(bars),len(live),sum(live),max(live,default=0),len(dead),sum(dead),max(dead,default=0),sum(a==b and a%2==1 for a,b in bars)]
                    values.append(dict(k=k,alpha=alpha,rho=rho,window_counts=win,bars=bars))
                details.append(dict(theta=theta,degrees=values))
            r['features']['raw_temporal_invariants']=r['features']['raw_temporal']+inv
            r['features']['raw_temporal_prefix_barcodes']=r['features']['raw_temporal']+barsfeat
            audit.append(dict(world=r['world'],arm=r['arm'],step=r['step'],invariants=inv,prefix_barcode_summaries=barsfeat,details=details))
    return audit
def main():
    pilot=load(SRC/'analysis/pilot/FEATURES.json');test=load(SRC/'results/FEATURES.json')
    audit=enrich(pilot)+enrich(test)
    (OUT/'INVARIANTS.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    frozen=load(SRC/'analysis/pilot/SELECTION.json')['raw_temporal']['candidate']
    keys=['raw_temporal','raw_temporal_invariants','raw_temporal_prefix_barcodes'];reports={}
    for key in keys:
        losses=[]
        for c in range(5):
            errs=[]
            for w in [961,962]:
                tr=[r for r in pilot if r['world']!=w];te=[r for r in pilot if r['world']==w]
                errs += [(p-r['target'])**2 for p,r in zip(fit_predict(tr,te,key,c),te)]
            losses.append(float(np.mean(errs)))
        c=frozen if key=='raw_temporal' else int(np.argmin(losses))
        pred=fit_predict(pilot,test,key,c)
        reports[key]=dict(candidate=c,pilot_leave_world_out_mse=losses,mse=float(np.mean([(p-r['target'])**2 for p,r in zip(pred,test)])),predictions=pred)
    base=reports['raw_temporal'];comparison={}
    for key in keys[1:]:
        gains=[]
        for w in range(971,977):
            ids=[i for i,r in enumerate(test) if r['world']==w]
            gains.append(float(np.mean([(base['predictions'][i]-test[i]['target'])**2-(reports[key]['predictions'][i]-test[i]['target'])**2 for i in ids])))
        v=np.array(gains);rng=np.random.default_rng(42);boot=v[rng.integers(0,6,(10000,6))].mean(1)
        comparison[key]=dict(mean_gain=float(v.mean()),per_world=gains,descriptive_world_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),positive_worlds=int(sum(v>0)))
    result=dict(status='Exploratory reused confirmation worlds, not new confirmation',train_worlds=[961,962],reused_evaluation_worlds=list(range(971,977)),reports=reports,comparisons=comparison)
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:{x:y for x,y in v.items() if x!='predictions'} for k,v in reports.items()},indent=2));print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
