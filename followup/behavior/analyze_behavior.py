"""Read-only audit of C probes; all new outputs stay beside this script.

Post-hoc diagnostics. Bootstrap units are roots, never checkpoint rows. Seeds
within a world use the same weights; worlds are reported separately/fixed.
"""
from pathlib import Path
import collections
import gzip
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'capability-zigzag-repro'
RUNS = REPO / 'cloud_return_round4/complex_worlds/runs/RCN12_FORGET3'
B = 2000
SEED = 20261108
EDGES = ['C1C2', 'C2C3', 'C1C3', 'C3C4']
TARGETS = ['chain_C1_C2_C3_C4', 'repair_fC123']
sys.path.insert(0,str(REPO/'stage_ab/code'))
import zz_score as Z

def rows(p):
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p, 'rt', encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def ci(x):
    return [float(v) for v in np.quantile(x, [.025, .975])]

def weights(n, count, rng, block=1):
    if block == 1:
        return rng.multinomial(n, np.full(n, 1/n), size=count)
    starts = rng.integers(n, size=(count, int(np.ceil(n/block))))
    idx = ((starts[..., None] + np.arange(block)) % n).reshape(count, -1)[:, :n]
    return np.array([np.bincount(x, minlength=n) for x in idx])

def weighted(a, w):
    return np.einsum('...r,br->b...', a, w)/w.sum(axis=1).reshape((-1,)+(1,)*(a.ndim-1))

def load_arm(d):
    world = d.name.split('_')[0]
    data = REPO / 'complex_worlds/data/RCN12_FORGET3' / world
    manifest = json.loads((data/'MANIFEST.json').read_text(encoding='utf-8'))
    root = [tuple(r) for r in manifest['split']['structure']]
    ind = [tuple(r) for r in manifest['split']['independent']]
    assert len(root) == len(set(root)) == len(ind) == 32
    assert not (set(root) & set(ind))
    for e in EDGES:
        assert manifest['permutations'][e] == list(range(32))
    fs = sorted(d.glob('eval_*.jsonl.gz'), key=lambda p:int(p.name.split('_')[1].split('.')[0]))
    steps = [int(p.name.split('_')[1].split('.')[0]) for p in fs]
    edge = np.zeros((len(fs), 4, 3, 32)) # templates 0,1,3
    face = np.zeros((len(fs), 2, 2, 32)) # templates 0,3; keep/override
    valid = np.zeros_like(face)
    field = np.zeros(face.shape+(4,))
    independent = np.zeros((len(fs), 2, 32))
    failure = collections.Counter()
    scopes = collections.Counter()
    rescored = 0
    for ti, p in enumerate(fs):
        rs = rows(p)
        for q in rs:
            if q['task'] in EDGES+['fC123']+TARGETS:
                check=Z.score(q,q['prediction'])
                assert all(check[k]==q[k] for k in ('correct','valid_format','field_correct'))
                rescored+=1
        by = {(r['task'],tuple(r['root']),r['template'],r.get('override')):r for r in rs}
        for ei,e in enumerate(EDGES):
            for tj,t in enumerate((0,1,3)):
                for j,r in enumerate(root):
                    q=by[e,r,t,None]
                    edge[ti,ei,tj,j]=q['correct']
        for tj,t in enumerate((0,3)):
            for o in (0,1):
                for j,r in enumerate(root):
                    q=by['fC123',r,t,bool(o)]
                    # Actual gold field matching, not assumed marginal equivalence.
                    other=root[(j+1)%32]
                    assert tuple(q['other'])==other
                    direct=by['C1C3',r,t,None]
                    middle=by['C1C2',other if o else r,t,None]
                    terminal=by['C2C3',other if o else r,t,None]
                    gold=q['gold'].split('|')
                    assert gold==[direct['gold'],middle['gold'],terminal['gold'],str(1-o)]
                    face[ti,tj,o,j]=q['correct']; valid[ti,tj,o,j]=q['valid_format']
                    field[ti,tj,o,j]=q['field_correct']
                    if 128<=steps[ti]<=384 and t==0:
                        failure[str((o,tuple(i+1 for i,v in enumerate(q['field_correct']) if not v),q['valid_format']))]+=1
        for k,target in enumerate(TARGETS):
            for j,r in enumerate(ind):
                selected=[q for q in rs if q['task']==target and tuple(q['root'])==r]
                assert len(selected)==(3 if target.startswith('repair') else 1)
                assert all(q['split']=='independent' for q in selected)
                independent[ti,k,j]=np.mean([q['correct'] for q in selected])
        scopes.update((r['split'],r['kind'],r['task']) for r in rs)
    # Audit actual training file, not just the generator's assertions.
    train=rows(data/'TRAIN_STREAM.jsonl.gz')
    primary={tuple(q['root']) for q in train}
    secondary={tuple(q[k]) for q in train for k in ('other','bad') if k in q}
    audit={'n_files':len(fs),'roots':len(root),'independent_roots':len(ind),
           'raw_predictions_rescored':rescored,
           'structure_primary_overlap':len(set(root)&primary),
           'independent_primary_overlap':len(set(ind)&primary),
           'structure_secondary_overlap':len(set(root)&secondary),
           'independent_secondary_overlap':len(set(ind)&secondary),
           'C_identity_permutations_verified':True,'gold_field_matches_verified':True,
           'face_templates':[0,3],'edge_templates':[0,1,3],
           'C_joint_train_rows':sum(q.get('task')=='fC123' for q in train),
           'independent_tasks':[list(k)+[v] for k,v in scopes.items() if k[0]=='independent']}
    return dict(name=d.name,world=world,steps=np.array(steps),edge=edge,face=face,
                valid=valid,field=field,independent=independent,audit=audit,failure=dict(failure))

def diagnostics(a,w):
    stage=(a['steps']>=128)&(a['steps']<=384)
    out={}
    for tj,t in enumerate((0,3)):
        e=a['edge'][:,:,0 if t==0 else 2,:]
        for o in (0,1):
            # All three separately asked conversions on matching inputs.
            # For override, middle and downstream edge use the next heldout root.
            em=np.roll(e[:,0,:],-1,axis=-1) if o else e[:,0,:]
            et=np.roll(e[:,1,:],-1,axis=-1) if o else e[:,1,:]
            conj=em*et*e[:,2,:]
            minimal=et*e[:,2,:] if o else conj
            f=a['face'][:,tj,o,:]
            fields=a['field'][:,tj,o,:,:]
            # Per-root product uses measured template-0/1 edge success fractions.
            ep=a['edge'][:,:,:2,:].mean(axis=2)
            prod=ep[:,2,:]*(np.roll(ep[:,0,:],-1,axis=-1) if o else ep[:,0,:])*(np.roll(ep[:,1,:],-1,axis=-1) if o else ep[:,1,:])
            selected={}
            for label,mask in [('stage_128_384',stage),('full_trajectory',np.ones(len(stage),bool))]:
                fc=f[mask]; ec=conj[mask]
                gap=(ec-fc).mean(axis=0)
                boot=w@gap/32
                yes=ec==1
                item_fail=(yes*(1-fc)).mean(axis=0)
                item_yes=yes.mean(axis=0)
                den=w@item_yes
                conditional=np.divide(w@item_fail,den,out=np.zeros(len(w)),where=den>0)
                marg=np.prod(e[mask,:3,:].mean(axis=-1),axis=1).mean()
                selected[label]={
                    'face_accuracy':float(fc.mean()),'exact_matched_conjunction':float(ec.mean()),
                    'conjunction_minus_face':float(gap.mean()),'root_bootstrap_gap_CI':ci(boot),
                    'matched_template_fraction_product':float(prod[mask].mean()),
                    'marginal_product_reference':float(marg),
                    'n_edge_all_correct_face_wrong':int((yes*(1-fc)).sum()),
                    'n_face_correct_edge_conjunction_wrong':int((fc*(1-ec)).sum()),
                    'face_failure_given_conjunction_correct':float(item_fail.sum()/item_yes.sum()) if item_yes.sum() else None,
                    'root_bootstrap_conditional_failure_CI':ci(conditional),
                    'valid_format':float(a['valid'][mask,tj,o,:].mean()),
                    'field_accuracy':fields[mask].mean(axis=(0,1)).tolist(),
                    'field_errors_given_all_edges_correct':[(float(((1-fields[mask,:,:,][...,j])*yes).sum()/yes.sum()) if yes.sum() else None) for j in range(4)],
                    'face_wrong_with_all_fields_correct':int(((fc==0)&(fields[mask].min(axis=-1)==1)).sum()),
                    'n_observations':int(fc.size),'n_root_clusters':32}
                minimal_gap=(minimal[mask]-fc).mean(axis=0)
                selected[label].update({
                    'required_edge_conjunction_copy_override_middle':float(minimal[mask].mean()),
                    'required_conjunction_minus_face':float(minimal_gap.mean()),
                    'required_conjunction_gap_root_CI':ci(w@minimal_gap/32),
                    'conjunction_brier_against_face':float(((ec-fc)**2).mean()),
                    'template_fraction_product_brier_against_face':float(((prod[mask]-fc)**2).mean()),
                })
            # Per-step results retain trajectory without treating it as independent n.
            selected['series']=[dict(step=int(s),face=float(f[i].mean()),conjunction=float(conj[i].mean()),
                                     per_root_template_fraction_product=float(prod[i].mean()),
                                     fields=fields[i].mean(axis=0).tolist(),valid=float(a['valid'][i,tj,o].mean()))
                                for i,s in enumerate(a['steps'])]
            out[f'template_{t}_{"override" if o else "keep"}']=selected
    return out

def ridge_predict(trainx,trainy,testx):
    mu=trainx.mean(axis=0); sd=trainx.std(axis=0); sd[sd<1e-8]=1
    x=np.column_stack([np.ones(len(trainx)),(trainx-mu)/sd])
    z=np.column_stack([np.ones(len(testx)),(testx-mu)/sd])
    penalty=np.eye(x.shape[1]); penalty[0,0]=0
    coef=np.linalg.solve(x.T@x+penalty,x.T@trainy)
    return np.clip(z@coef,0,1)

def predictive(arms,sw,iw):
    # Independent ROOTS, not independent experimental worlds or task mechanisms.
    # Fixed ridge=1; leave both seeds of a world out together; no tuning.
    x=[];y=[];world=[]
    for a in arms:
        w=sw[a['world']]; v=iw[a['world']]
        edges=np.einsum('ter,r->te',a['edge'][:,:,:2,:].mean(axis=2),w)/w.sum()
        face=np.einsum('tor,r->to',a['face'][:,0,:,:],w)/w.sum()
        time=a['steps']/768
        phases=np.column_stack([a['steps']>=k for k in (128,288,352,480,544)])
        base=np.column_stack([time,time*time,phases,edges,edges[:,:3].min(axis=1),edges[:,:3].prod(axis=1)])
        x.append(np.column_stack([base,face,face.min(axis=1)]))
        y.append(np.einsum('tkr,r->tk',a['independent'],v)/v.sum())
        world.extend([a['world']]*len(time))
    x=np.concatenate(x);y=np.concatenate(y);world=np.array(world)
    p0=np.zeros_like(y);p1=np.zeros_like(y)
    for w in sorted(set(world)):
        tr=world!=w;te=~tr
        p0[te]=ridge_predict(x[tr,:-3],y[tr],x[te,:-3])
        p1[te]=ridge_predict(x[tr],y[tr],x[te])
    return dict(baseline_mse=((p0-y)**2).mean(axis=0).tolist(),
                plus_face_mse=((p1-y)**2).mean(axis=0).tolist(),
                improvement=(((p0-y)**2)-((p1-y)**2)).mean(axis=0).tolist(),
                by_world={w:dict(baseline_mse=((p0[world==w]-y[world==w])**2).mean(axis=0).tolist(),
                                plus_face_mse=((p1[world==w]-y[world==w])**2).mean(axis=0).tolist()) for w in sorted(set(world))})

def main():
    rng=np.random.default_rng(SEED)
    arms=[load_arm(d) for d in sorted(RUNS.iterdir()) if d.is_dir()]
    print('Loaded',len(arms),'arms',flush=True)
    ws={w:weights(32,B,rng) for w in sorted({a['world'] for a in arms})}
    results={'status':'post-hoc exploratory; no trained model changes','bootstrap':dict(B=B,seed=SEED), 'arms':[]}
    for a in arms:
        diag=diagnostics(a,ws[a['world']])
        # Override reuses next root. Circular blocks offer sensitivity to local
        # dependence along the root-pair cycle, not an iid confidence theorem.
        blocks={str(k):diagnostics(a,weights(32,B,rng,k)) for k in (4,8)}
        sensitivity={key:{k:blocks[k][key]['stage_128_384']['root_bootstrap_gap_CI'] for k in blocks} for key in diag}
        results['arms'].append(dict(name=a['name'],audit=a['audit'],diagnostics=diag,
                                    circular_block_gap_CI=sensitivity,failure_patterns_stage=a['failure']))
        print(a['name'],[(k,round(v['stage_128_384']['conjunction_minus_face'],4)) for k,v in diag.items()],flush=True)
    ones={w:np.ones(32) for w in ws}
    pred=predictive(arms,ones,ones)
    boot=[]
    for b in range(500):
        iw={w:weights(32,1,rng)[0] for w in ws}
        boot.append(predictive(arms,{w:ws[w][b] for w in ws},iw)['improvement'])
    pred['targets']=TARGETS
    pred['root_bootstrap_improvement_CI']=np.quantile(boot,[.025,.975],axis=0).T.tolist()
    pred['bootstrap_B']=500
    blockboot=[]
    for b in range(250):
        sw={w:weights(32,1,rng,8)[0] for w in ws}
        iw={w:weights(32,1,rng,8)[0] for w in ws}
        blockboot.append(predictive(arms,sw,iw)['improvement'])
    pred['circular_block8_improvement_CI']=np.quantile(blockboot,[.025,.975],axis=0).T.tolist()
    pred['circular_block8_B']=250
    pred['interpretation']='Independent target roots; post-hoc leave-world-out prediction, only three fixed worlds. No topology-versus-score utility claim.'
    results['predictive']=pred
    (HERE/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(pred,indent=2),flush=True)

if __name__=='__main__':main()
