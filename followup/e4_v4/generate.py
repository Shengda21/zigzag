"""Install-free paired E4 generator. CPU audit is about data, never LM results."""
from pathlib import Path
import argparse, collections, hashlib, itertools, json, random, sys, time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from spec_world import World
import zz_score as Z

CONFIG=dict(version='E4-v4',review_status='reviewed-independent-pilot-candidate',model='Qwen/Qwen2.5-1.5B-Instruct',steps=512,batch=64,grid=32,
            pilot=[961,962],confirm=[971,972,973,974,975,976],seeds=[1],
            train_roots=2048,probe_roots=1024,external_roots=1024,alphabet=64,
            trained_faces=['fABC','fBCD','fABD','fACD'],heldout_faces=[],edge_margin=.05,
            external_min_effect=.05,primary_step=512)
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines()]
def write(p,rs):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf8',newline='\n') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def spec():
    es=[dict(name=a+b,src=a,dst=b,kind='taught',rule='codebook') for a,b in itertools.combinations('ABCD',2)]
    fs=[dict(name='f'+a+b+c,path=[a,b,c],cycle=[a+b,b+c,a+c],kind='taught' if a+b+c in ('ABC','BCD','ABD','ACD') else 'latent') for a,b,c in itertools.combinations('ABCD',3)]
    return dict(alphabet=64,vertices=list('ABCD'),digit_groups=[],edges=es,faces=fs)
def neutral_prompt(rules,operands):
    legend='Apply the three map slots in order. An operand is an explicit code sequence or output N from an earlier slot. '
    return legend+' '.join(f'Slot {i}: rule {e}; operand {x}.' for i,(e,x) in enumerate(zip(rules,operands),1))

def record(w,face,root,other,arm,rid,split,third=None):
    a,b,c=w.faces[face]['path']; ab,bc,ac=a+b,b+c,a+c
    assert third is not None,'Explicit third root required for split audit'
    mid=w.apply(ab,root); second=mid if arm=='joint' else other
    p=neutral_prompt([ab,bc,ac],[w.view(a,root),w.view(b,second),w.view(a,third)])
    gold='|'.join([w.view(b,mid),w.view(c,w.apply(bc,second)),w.view(c,w.apply(ac,third)),'1'])
    return dict(id=rid,task=face,kind='joint' if arm=='joint' else 'edge_bundle',vocabulary='e4',
                root=list(root),other=list(other),third=list(third),split=split,template=0,prompt=p,gold=gold,
                schema=[b,c,c,'binary'],field_length=2)
def single(w,e,root,rid,other,third,split='probe'):
    ed=w.edges[e]; field_roots=[root,third,other]
    assert len(set(field_roots))==3
    prompt=neutral_prompt([e]*3,[w.view(ed['src'],rr) for rr in field_roots])
    return dict(id=rid,task=e,kind='edge',vocabulary='e4',root=list(root),other=list(other),third=list(third),
                field_roots=[list(x) for x in field_roots],split=split,template=0,prompt=prompt,
                gold='|'.join([w.view(ed['dst'],w.apply(e,rr)) for rr in field_roots]+['1']),
                schema=[ed['dst']]*3+['binary'],field_length=2)

def mixed(w,root,other,third,rid,index,split):
    combinations=list(itertools.product(['AC','BC'],['AD','BD','CD']))
    qC,qD=combinations[index%6]; rules=['AB',qC,qD]; rr=[root,other,third]
    assert len(set(rr))==3
    prompt=neutral_prompt(rules,[w.view(e[0],x) for e,x in zip(rules,rr)])
    gold='|'.join([w.view(e[-1],w.apply(e,x)) for e,x in zip(rules,rr)]+['1'])
    return dict(id=rid,task='mixed_BCD',kind='mixed',vocabulary='e4',root=list(root),other=list(other),third=list(third),
                field_roots=[list(x) for x in rr],rules=rules,split=split,template=0,prompt=prompt,
                gold=gold,schema=['B','C','D','binary'],field_length=2)

def reference(w,root,other,third,rid,index,split):
    assert root!=third
    if index%2==0:
        rules=['AB','BC','AD']; rr=[root,root,third]
        operands=[w.view('A',root),'output 1',w.view('A',third)]
    else:
        rules=['AB','BC','CD']; rr=[third,root,root]
        operands=[w.view('A',third),w.view('B',root),'output 2']
    return dict(id=rid,task='reference_twohop',kind='reference',vocabulary='e4',root=list(root),other=list(other),third=list(third),
                field_roots=[list(x) for x in rr],rules=rules,reference_position=2 if index%2==0 else 3,dependency_depth=1,
                split=split,template=0,prompt=neutral_prompt(rules,operands),
                gold='|'.join([w.view(e[-1],w.apply(e,x)) for e,x in zip(rules,rr)]+['1']),schema=['B','C','D','binary'],field_length=2)

def specify_destinations(r):
    suffix='Output three comma-separated code sequences then literal 1, joined by |. No explanation.'
    # Single-edge legacy wording also normalizes to the common explicit suffix.
    legacy='Output the three resulting sequences then literal 1, joined by |. No explanation.'
    p=r['prompt'].replace(suffix,'').replace(legacy,'').rstrip()
    r['prompt']=p+' Output destinations: '+','.join(r['schema'][:3])+'. '+suffix
    return r

def external_record(w,root,rid):
    cur=root;fields=[]
    for e in ['AB','BC','CD']:
        cur=w.apply(e,cur); fields.append(w.view(e[-1],cur))
    return dict(id=rid,task='chain_A_B_C_D',kind='transfer',vocabulary='e4',template=0,root=list(root),split='external',
                requires=['AB','BC','CD'],prompt=neutral_prompt(['AB','BC','CD'],[w.view('A',root),'output 1','output 2']),
                gold='|'.join(fields+['1']),schema=['B','C','D','binary'],field_length=2)

def generate(world,phase,rebuild=False):
    out=HERE/'data'/phase/f'world{world}'
    if out.exists() and not rebuild:raise FileExistsError(f'Immutable data already exists: {out}')
    w=World(spec(),world,world);rng=random.Random(world)
    roots=list(itertools.product(range(64),repeat=2));rng.shuffle(roots)
    train,probe,external=roots[:2048],roots[2048:3072],roots[3072:]
    streams={a:[] for a in ('edge','joint')}; rehearsal_index=0
    for i in range(CONFIG['steps']*CONFIG['batch']):
        root=train[i%len(train)];other=train[(i+997)%len(train)]
        face=CONFIG['trained_faces'][(i+i//2048+(i//2048)//4)%4]
        is_rehearsal=(i//2+(i//2048)//2)%4==0
        for arm in streams:
            r=record(w,face,root,other,arm,f't{i}','train',train[(i+463)%len(train)])
            if is_rehearsal:
                e=list(w.edges)[(i//4)%6];r=single(w,e,root,f't{i}',other,train[(i+463)%len(train)],'train')
                r.update(split='train',template=0)
                if rehearsal_index%4==1: r=mixed(w,root,other,train[(i+463)%len(train)],f't{i}',rehearsal_index//4,'train')
                elif rehearsal_index%4==3: r=reference(w,root,other,train[(i+463)%len(train)],f't{i}',rehearsal_index//4,'train')
            streams[arm].append(specify_destinations(r))
        if is_rehearsal: rehearsal_index+=1
    ev=[]
    trials=[rng.sample(probe,3) for _ in range(512)]
    for j,(root,other,third) in enumerate(trials):
        for e in w.edges:
            r=single(w,e,root,f'edge_{e}_{j}',other,third);r['trial']=j
            ev.append(r)
        for f in w.faces:
            r=record(w,f,root,other,'joint',f'face_{f}_{j}','probe',third)
            r.update(kind='joint',trial=j);ev.append(r)
    for j,(root,other,third) in enumerate(trials):
        r=mixed(w,root,other,third,f'mixed_{j}',j,'probe');r['trial']=j;ev.append(r)
    for j,(root,other,third) in enumerate(trials):
        r=reference(w,root,other,third,f'reference_{j}',j,'probe');r['trial']=j;ev.append(r)
    ev=[specify_destinations(r) for r in ev]
    # Independent behavioral readout: unseen 3-hop route, distinct roots and
    # root population; common English/four-field format, absent from features.
    ext=[]
    for j in range(512):
        r=external_record(w,rng.choice(external),f'external_{j}');r['trial']=j;ext.append(specify_destinations(r))
    for arm,rs in streams.items():write(out/f'{arm}.jsonl',rs)
    write(out/'PROBES.jsonl',ev);write(out/'EXTERNAL.jsonl',ext)
    dump(out/'ROOT_SPLITS.json',dict(train=train,probe=probe,external=external,unused=[],sampling='512 iid diagnostic triples with replacement across trials; distinct members within each triple. 512 iid external roots with replacement. Shared fixed trial draws across arms/checkpoints.'))
    dump(out/'SPEC.json',w.spec)
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['MANIFEST.json','TOKEN_AUDIT.json']}
    audit=dict(world=world,phase=phase,training_records_per_arm=len(streams['edge']),probe_records=len(ev),external_records=len(ext),
               disjoint_roots=not(set(train)&set(probe) or set(train)&set(external) or set(probe)&set(external)),
               heldout_face_training_count=sum(r['task'] in CONFIG['heldout_faces'] for r in streams['joint']),
               character_budgets_equal=all(len(a['prompt'])==len(b['prompt']) and len(a['gold'])==len(b['gold']) for a,b in zip(*streams.values())),
               oracle_scoring_all_pass=all(Z.score(r,r['gold'])['correct'] for r in ev+ext+streams['edge'][:128]+streams['joint'][:128]),
               note='Root sequences held out; individual code symbols intentionally shared so dictionaries are learnable. No model inference.')
    assert audit['disjoint_roots'] and audit['oracle_scoring_all_pass'] and audit['character_budgets_equal'] and audit['heldout_face_training_count']==0
    dump(out/'MANIFEST.json',dict(config=CONFIG,hashes=hashes,audit=audit))
    return audit
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','confirm'],default='pilot');p.add_argument('--rebuild-unrun-pilot',action='store_true');a=p.parse_args()
    if a.rebuild_unrun_pilot:
        assert a.phase=='pilot' and not (HERE/'runs').exists() and not (HERE/'CONFIRM_LOCK.json').exists(),'Cannot overwrite scientific runs/frozen data'
    if a.phase=='confirm':assert (HERE/'CONFIRM_LOCK.json').exists(),'Freeze pilot first with freeze.py'
    t=time.perf_counter();audits=[generate(w,a.phase,a.rebuild_unrun_pilot) for w in CONFIG[a.phase]]
    dump(HERE/f'CPU_AUDIT_{a.phase}.json',dict(audits=audits,seconds=time.perf_counter()-t,meaning='Actual data generation and oracle-score audit, not an LM experiment'))
    dump(HERE/'CONFIG.json',CONFIG)
    print(json.dumps(audits,indent=2))
if __name__=='__main__':main()
