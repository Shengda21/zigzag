"""Read-only exploratory alpha1 ablation; source and reproduction-package compatible.
Run: python two_cell_deletion.py. Writes only sibling two_cell_deletion outputs.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if (HERE.parents[1]/'complex_worlds').exists() else HERE.parents[1]/'capability-zigzag-repro'
sys.path[:0] = [str(REPO/'crosslingual/code'),str(REPO/'paper_sec4')]
import cw_zigzag as CW
import zz_analyze as ZA
import zz_robust as R
from mv_counts import spurious_history

def decomposition(cx,masks,t):
    prev=0
    for m in masks[:t]: prev |= m
    cur=prev|masks[t]
    j=list(cx.induced(prev,cur,1)); new=list(cx.induced(masks[t],cur,1))
    old=[int(v) for m in masks[:t] for v in cx.induced(m,prev,1)]
    oldcur=[int(v) for m in masks[:t] for v in cx.induced(m,cur,1)]
    killed=(len(cx.homology(prev,1)[0])-CW.rank(old))-(CW.rank(j)-CW.rank(oldcur))
    supported=(CW.rank(j)+CW.rank(new)-CW.rank(j+new))-(CW.rank(oldcur)+CW.rank(new)-CW.rank(oldcur+new))
    return {'loss_of_unsupported_directions':killed,'previous_artifact_becomes_supported':supported}

def main():
    spec=json.loads((REPO/'complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json').read_text(encoding='utf-8'))
    cx=CW.CWComplex([(n,d,list(b)) for n,d,b in spec['cells']])
    retain=sum(1<<i for i,c in enumerate(spec['cells']) if c[1]!=2)
    assert max(c[1] for c in spec['cells'])==2
    out={'status':'exploratory mathematical/computational ablation; not confirmation','theta':.8,
         'ablation':'delete every 2-cell from every checkpoint; retain all vertices/edges and original scores',
         'independent_worlds':3,'arms':[],'alpha_mismatch_count':0,'checkpoint_count':0}
    for w in (635,636,637):
        for seed in (1,2):
            name=f'world{w}_S{seed}'
            ev=ZA.load_run(REPO/'cloud_return_round4/complex_worlds/runs/RCN12_FORGET3'/name)
            sm=R.ScoreModel(spec,ev); scores=sm.scores(sm.weights(0,0))
            masks=R.Masker(cx).masks(np.stack([scores[n][0] for n in cx.names]),.8)
            stripped=[m&retain for m in masks]
            original=spurious_history(cx,masks,1); deleted=spurious_history(cx,stripped,1)
            rows=[{'step':step,'alpha_original':a['spurious'],'alpha_deleted':b['spurious'],
                'beta_original':a['beta'],'beta_deleted':b['beta'],'beta_deleted_minus_original':b['beta']-a['beta'],
                'historical_rho_original':a['connecting_rank_prev_new'],'historical_rho_deleted':b['connecting_rank_prev_new']}
                for step,a,b in zip(sm.steps,original,deleted)]
            mismatch=[r['step'] for r in rows if r['alpha_original']!=r['alpha_deleted']]
            rec={'run':name,'checkpoint_count':len(rows),'alpha_mismatch_steps':mismatch,
                 'beta_difference_steps':[r['step'] for r in rows if r['beta_deleted_minus_original']],
                 'final':rows[-1],'rows':rows}
            if name=='world636_S1':
                t=sm.steps.index(240)
                rec['step240_decomposition']={'original':decomposition(cx,masks,t),'deleted':decomposition(cx,stripped,t)}
                out['step240_decomposition']=rec['step240_decomposition']
            out['arms'].append(rec);out['alpha_mismatch_count']+=len(mismatch);out['checkpoint_count']+=len(rows)
            print(name,'mismatches',len(mismatch),'final',rec['final'],flush=True)
    out['identity']='W=sum Z_k(L_s) contains B_k(U), since the boundary of each (k+1)-cell lies in a checkpoint containing that cell. Hence S=W/B_k(U), H_k(U)/S is isomorphic to Z_k(U)/W, and alpha=dim Z_k(U)-dim W. Alpha_k depends only on checkpoint k-skeleta.'
    (HERE/'two_cell_deletion.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Two-cell deletion: exploratory alpha1 audit','',f"All six full-complex trajectories at theta=.8: {out['alpha_mismatch_count']}/{out['checkpoint_count']} alpha mismatches. Three independent worlds; checkpoints/seeds are repeated observations.",'',out['identity'],'', '| Run | Final alpha original/deleted | Final beta original/deleted | Checkpoints with beta difference |','|---|---:|---:|---:|']
    for a in out['arms']:
        r=a['final'];lines.append(f"| {a['run']} | {r['alpha_original']}/{r['alpha_deleted']} | {r['beta_original']}/{r['beta_deleted']} | {len(a['beta_difference_steps'])} |")
    lines+=['','At world636-S1 step240, original killed/becomes-supported = 1/0; after deletion = 0/1. These values are computed from the equality-condition subspace ranks. The original killing description is technically correct for those maps; it is not evidence that alpha detects fillings or that filling is necessary for its change.','',json.dumps(out['step240_decomposition'],indent=2),'','All 294 alpha/beta pairs and historical connecting ranks are in two_cell_deletion.json. No training, GPU, outcome fitting, manuscript edits or confirmation claims. The original audit script/results are unchanged.','', 'Run: python two_cell_deletion.py. Requires the existing repository routines and NumPy. Outputs contain no absolute paths, timestamps or random computations, allowing exact numeric comparison between both package locations.']
    (HERE/'TWO_CELL_DELETION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert out['alpha_mismatch_count']==0 and out['checkpoint_count']==294
    assert out['step240_decomposition']=={'original':{'loss_of_unsupported_directions':1,'previous_artifact_becomes_supported':0},'deleted':{'loss_of_unsupported_directions':0,'previous_artifact_becomes_supported':1}}

if __name__=='__main__':main()
