"""Recheck persisted data and real archived resource observations, read-only."""
import collections,json,platform,statistics,time
from pathlib import Path
from generate import HERE,CONFIG,rows,sha,dump,record,spec,specify_destinations,neutral_prompt
from spec_world import World
import zz_score as Z
def main():
    tick=time.perf_counter();checks=[]
    for world in CONFIG['pilot']:
        d=HERE/'data/pilot'/f'world{world}';m=json.loads((d/'MANIFEST.json').read_text())
        for n,h in m['hashes'].items():assert sha(d/n)==h,n
        split=json.loads((d/'ROOT_SPLITS.json').read_text());tr={tuple(x) for x in split['train']};pr={tuple(x) for x in split['probe']};ex={tuple(x) for x in split['external']}
        assert not(tr&pr or tr&ex or pr&ex)
        streams=[rows(d/f'{arm}.jsonl') for arm in ['edge','joint']]
        for rs in streams:
            assert len(rs)==32768 and all(tuple(r['root']) in tr for r in rs)
            assert all(tuple(r.get('other',r['root'])) in tr for r in rs)
            assert all(tuple(r.get('third',r['root'])) in tr for r in rs)
            assert not any(r['task'] in CONFIG['heldout_faces'] for r in rs)
            assert all(Z.score(r,r['gold'])['correct'] for r in rs)
            fc={f:sum(r['task']==f for r in rs) for f in CONFIG['trained_faces']}
            assert set(fc.values())=={6144},fc
            exposure={}
            for r in rs:
                if r['task'] in CONFIG['trained_faces']:exposure.setdefault(tuple(r['root']),set()).add(r['task'])
            assert len(exposure)==2048 and all(len(v)==4 for v in exposure.values())
        w=World(spec(),world,world)
        for a,b in zip(*streams):
            assert 'Mode ' not in a['prompt'] and 'Mode ' not in b['prompt']
            if a['kind']=='edge_bundle':
                assert a['prompt']==specify_destinations(record(w,a['task'],tuple(a['root']),tuple(a['other']),'edge',a['id'],'train',tuple(a['third'])))['prompt']
                assert b['prompt']==specify_destinations(record(w,b['task'],tuple(b['root']),tuple(b['other']),'joint',b['id'],'train',tuple(b['third'])))['prompt']
                assert a['prompt'].split('Slot 2:')[0]==b['prompt'].split('Slot 2:')[0]
                assert a['prompt'].split('Slot 3:')[1]==b['prompt'].split('Slot 3:')[1]
            else: assert a==b
            assert a['root']==b['root'] and a['task']==b['task'] and len(a['prompt'])==len(b['prompt']) and len(a['gold'])==len(b['gold'])
        hist=[]
        for arm,rs in zip(['edge','joint'],streams):
            counts=collections.Counter()
            for r in rs:
                if r['kind'] in ('mixed','reference'):
                    for e,rr in zip(r['rules'],r['field_roots']):counts[e,tuple(rr)]+=1
                elif r['kind']!='edge':
                    a,b,c=r.get('face',r['task'])[1:]
                    counts[a+b,tuple(r['root'])]+=1
                    counts[b+c,tuple(r['root'] if arm=='joint' else r['other'])]+=1
                    counts[a+c,tuple(r['third'])]+=1
                else:
                    for rr in r['field_roots']:counts[r['task'],tuple(rr)]+=1
            hist.append(counts)
        assert hist[0]==hist[1],'Unequal labeled map/root marginal training exposures'
        for rs in streams:
            for r in rs:
                if r['kind'] in ('joint','edge_bundle'):
                    assert len({tuple(r['root']),tuple(r['other']),tuple(r['third'])})==3
                    assert r['gold'].split('|')[1]!=r['gold'].split('|')[2],'Trivial endpoint copy shortcut'
        ev=rows(d/'PROBES.jsonl');ext=rows(d/'EXTERNAL.jsonl')
        assert all(tuple(r['root']) in pr for r in ev) and all(tuple(r['root']) in ex for r in ext)
        assert collections.Counter(r['reference_position'] for r in ev if r['kind']=='reference')=={2:256,3:256}
        assert all(all(tuple(x) in pr for x in r.get('field_roots',[])) for r in ev)
        assert all('Mode ' not in r['prompt'] and r['prompt'].startswith('Apply the three map slots in order.') for r in ev+ext)
        assert all('operand output 1.' in r['prompt'] and 'operand output 2.' in r['prompt'] for r in ext)
        bytrial={}
        for r in ev:
            if r['kind']=='joint':
                triple=tuple(tuple(r[k]) for k in ['root','other','third'])
                assert len(set(triple))==3 and all(x in pr for x in triple)
                if r['trial'] in bytrial:assert bytrial[r['trial']]==triple
                bytrial[r['trial']]=triple
        assert len(bytrial)==512
        for r in ev:
            if r['kind']=='edge':
                root,other,third=bytrial[r['trial']]
                assert [tuple(x) for x in r['field_roots']]==[root,third,other]
        assert sum(r['kind']=='mixed' for r in streams[0])==2048
        assert sum(r['kind']=='reference' for r in streams[0])==2048
        refs=[r for r in streams[0] if r['kind']=='reference']
        assert collections.Counter(r['reference_position'] for r in refs)=={2:1024,3:1024}
        assert all(r['dependency_depth']==1 and r['prompt'].count('operand output ')==1 and tuple(r['root'])!=tuple(r['third']) and len({tuple(x) for x in r['field_roots']})==2 for r in refs)
        assert not any('operand output ' in r['prompt'] for r in streams[0] if r['kind']!='reference')
        assert sum(r['kind']=='edge' for r in streams[0])==4096
        assert all(len({tuple(r[k]) for k in ['root','other','third']})==3 for r in ev if r['kind']=='mixed')
        assert not({r['prompt'] for r in streams[0]+streams[1]}&{r['prompt'] for r in ev+ext})
        assert all(Z.score(r,r['gold'])['correct'] and not Z.score(r,'invalid')['correct'] for r in ev+ext)
        checks.append(dict(world=world,training_oracle_rows_checked=65536,evaluation_oracle_rows_checked=len(ev+ext),equal_per_map_root_training_exposures=True,all_pass=True))
    # Namespace and syntax are different diagnostics; strict scoring remains unchanged.
    sample=ext[0]; good=sample['gold']
    pp=good.split('|'); pp[0]=','.join('C'+x[1:] for x in pp[0].split(',')); wrong='|'.join(pp)
    assert Z.score(sample,good)['code_syntax']==1
    assert Z.score(sample,wrong)['code_syntax']==1 and Z.score(sample,wrong)['valid_format']==0
    assert Z.score(sample,wrong)['four_field_structure']==1
    assert Z.score(sample,'|'.join(good.split('|')[:3]))['four_field_structure']==0
    assert Z.score(sample,'|'.join(good.split('|')[:3]+['0']))['sentinel_one']==0
    from analyze import extract
    oracle=[dict(r,prediction=r['gold']) for r in ev+ext]
    perfect=extract(oracle);bad=extract([dict(r,prediction='invalid') for r in ev+ext],perfect)
    assert perfect['target']==1 and bad['target']==0 and perfect['betti']!=bad['betti']
    partial=extract([dict(r,prediction='|'.join(r['gold'].split('|')[:3])+ '|invalid') if r['kind']=='edge' else dict(r,prediction=r['gold']) for r in ev+ext])
    assert all(x==1 for x in partial['conjunction'].values()) and all(x==0 for x in partial['strict_prompt_conjunction'].values())
    from analyze import fit_predict
    fixtures=[dict(features=q['features'],target=q['target']) for q in [perfect,bad,perfect,bad]]
    for key in perfect['features']:
        for candidate in range(5):assert len(fit_predict(fixtures,fixtures,key,candidate))==4
    archived=HERE.parent.parent/'capability-zigzag-repro/cloud_return_round4/complex_worlds/runs/RCN12_FORGET3'
    resources=[]
    for p in sorted(archived.glob('world*/train.jsonl')):
        rr=rows(p);ss=[json.loads(q.read_text()) for q in p.parent.glob('eval_*.summary.json')]
        resources.append(dict(source=str(p.relative_to(HERE.parent.parent)),source_sha256=sha(p),steps=len(rr),
             train_step_seconds_median=statistics.median(r['seconds'] for r in rr),train_seconds_sum=sum(r['seconds'] for r in rr),
             max_peak_memory_gb=max(r.get('peak_memory_gb',0) for r in rr),eval_examples=sorted(set(r['n'] for r in ss)),
             eval_total_seconds=sum(r['seconds'] for r in ss),eval_seconds_per_example_median=statistics.median(r['seconds']/r['n'] for r in ss)))
    hub=Path.home()/'.cache/huggingface/hub';cache=[p.name for p in hub.glob('models--*')]
    import torch
    hw=dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,cuda_available=torch.cuda.is_available(),
            cached_model_directories=cache,target_1_5B_cache_present=any('Qwen2.5-1.5B' in p for p in cache))
    estimates=None
    if resources:
        ts=statistics.median(r['train_step_seconds_median'] for r in resources);es=statistics.median(r['eval_seconds_per_example_median'] for r in resources)
        estimates=dict(basis='Linear extrapolation from archived H800 logs; new prompts and concurrency differ. Not measured E4 runtime.',
             per_arm_training_seconds=512*ts,per_arm_eval_seconds=17*6656*es,
             pilot_four_arms_serial_hours=4*(512*ts+17*6656*es)/3600,
             confirm_twelve_arms_serial_hours=12*(512*ts+17*6656*es)/3600)
    dump(HERE/'LOCAL_AUDIT.json',dict(checks=checks,synthetic_feature_test=dict(perfect_target=1,invalid_target=0,meaning='Algorithm fixtures, not LM evidence'),
         local_hardware=hw,archived_h800_observations=resources,extrapolation=estimates,seconds=time.perf_counter()-tick,
         cloud_runs_executed=False,tokenizer_audit_pending=not all((HERE/"data/pilot"/f"world{w}"/"TOKEN_AUDIT.json").exists() for w in CONFIG["pilot"])))
    print(json.dumps(dict(checks=checks,hardware=hw,extrapolation=estimates),indent=2))
if __name__=='__main__':main()
