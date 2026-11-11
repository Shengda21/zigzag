"""Frozen exploratory bridge tau/kappa 2-cell ablation. Run with Python/NumPy."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1] if (HERE.parents[1]/'complex_worlds').exists() else HERE.parents[1]/'capability-zigzag-repro'
sys.path[:0]=[str(REPO/'crosslingual/code'),str(REPO/'paper_sec4')]
import cw_zigzag as CW
import zz_analyze as ZA
import zz_robust as R
from theta_sweep import kappa

def counts(cx,lo,hi,i):
    a,b=lo[i:i+2];c,d=hi[i:i+2]
    left=kappa(cx,c|d,a|b,[a],1);right=kappa(cx,c|d,a|b,[b],1);tau=kappa(cx,c|d,a|b,[a,b],1)
    return {'tau':tau,'left':left,'right':right,'birth':left-tau,'death':right-tau}

def main():
    spec=json.loads((REPO/'complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json').read_text(encoding='utf-8'))
    cx=CW.CWComplex([(n,d,list(b)) for n,d,b in spec['cells']]);retain=sum(1<<i for i,c in enumerate(spec['cells']) if c[1]<2)
    out={'status':'exploratory exact ablation, not independent confirmation','theta':.8,'deltas':[0,.05],'independent_worlds':3,'arms':[],'tau_mismatches':[],'differences':[]}
    for w in (635,636,637):
        for seed in (1,2):
            name=f'world{w}_S{seed}';ev=ZA.load_run(REPO/'cloud_return_round4/complex_worlds/runs/RCN12_FORGET3'/name);sm=R.ScoreModel(spec,ev);scores=sm.scores(sm.weights(0,0))
            arr=np.stack([scores[n][0] for n in cx.names]);mk=R.Masker(cx)
            masks={th:mk.masks(arr,th) for th in (.75,.8,.85)};deleted={th:[m&retain for m in ms] for th,ms in masks.items()};rows=[]
            for delta,lo,hi in ((0,.8,.8),(.05,.75,.85)):
                for i in range(len(sm.steps)-1):
                    a=counts(cx,masks[lo],masks[hi],i);b=counts(cx,deleted[lo],deleted[hi],i)
                    row={'steps':sm.steps[i:i+2],'delta':delta,'original':a,'deleted':b};rows.append(row)
                    if a['tau']!=b['tau']:out['tau_mismatches'].append({'run':name,**row})
                    if a!=b:out['differences'].append({'run':name,**row})
            out['arms'].append({'run':name,'rows':rows});print(name,'done',flush=True)
    out['bridge_scale_comparisons']=sum(len(a['rows']) for a in out['arms'])
    out['difference_counts']={str(d):{key:sum(r['delta']==d and r['original'][key]!=r['deleted'][key] for r in out['differences']) for key in ('tau','left','right','birth','death')} for d in (0,.05)}
    (HERE/'tau_cell_deletion.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    proof='Let Z+=Z_k(w_high), W=Z_k(A_low)+Z_k(B_low), and D=B_k(w_low), all embedded in C_k(w_low). Every (k+1)-cell in w_low belongs to A_low or B_low, so its boundary belongs to the corresponding cycle space, giving D subset W. Under q:Z_k(w_low)->H_k(w_low), I=q(Z+) and J=q(W)=W/D. Therefore dim(I+J)-dim J=dim(Z+ + W)-dim W. Removing (k+1)-cells changes neither Z+ nor W, proving tau invariance. The argument applies to any degree over the fixed field, with subcomplex inclusions and cellular chains in one ambient complex.'
    distinction='For a one-sided source A_low, D need not be contained in Z_k(A_low): boundaries contributed by B_low can be absent from A_low. Instead kappa_left=dim(Z+ + Z_k(A_low)+D)-dim(Z_k(A_low)+D). Thus one-sided kappa can depend on (k+1)-cells. Since tau is invariant, b=kappa_left-tau and d=kappa_right-tau can also depend on them. A valid counterexample is A a path with two triangle edges and B the filled triangle: original H1(A union B)=0, while deleting B face gives H1=1 and H1(A)=0. Left kappa and birth change 0 to 1, tau stays 0 because B after deletion contains the boundary cycle. Swapping A/B gives the right/death example. Assign all included cells score 1 and absent cells score 0: this works at theta=.8 for both delta=0 and .05.'
    # Validate the explicit counterexample using the same exact inclusion routines.
    tri=CW.CWComplex([('a',0,[]),('b',0,[]),('c',0,[]),('ab',1,['a','b']),('bc',1,['b','c']),('ac',1,['a','c']),('f',2,['ab','bc','ac'])])
    path=(1<<5)-1;filled=(1<<7)-1;boundary=(1<<6)-1
    ex={'original':counts(tri,[path,filled],[path,filled],0),'deleted':counts(tri,[path,boundary],[path,boundary],0)}
    assert ex['original']['left']==0 and ex['deleted']['left']==1 and ex['original']['tau']==ex['deleted']['tau']==0
    out['one_sided_counterexample']=ex
    (HERE/'tau_cell_deletion.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    text=['# Tau and one-sided kappa: exploratory 2-cell deletion audit','',proof,'',f"All six trajectories and 288 original bridges at each frozen scale delta=0/.05: {len(out['tau_mismatches'])}/{out['bridge_scale_comparisons']} tau mismatches.",'','Observed difference counts by scale:',json.dumps(out['difference_counts'],indent=2),'',distinction,'','Validated triangle counterexample:',json.dumps(ex,indent=2),'','All bridge-scale values and every difference are in tau_cell_deletion.json. No empirical prediction, coverage, mechanism or novelty claim follows. No GPU/cloud or manuscript edits. Run python tau_cell_deletion.py.']
    (HERE/'TAU_CELL_DELETION_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    assert not out['tau_mismatches'] and out['bridge_scale_comparisons']==576
if __name__=='__main__':main()
