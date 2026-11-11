"""Create matched lower/three-variety continuations and fresh fixed probes."""
import hashlib,itertools as it,json,random,shutil,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
from counterfactual_state_use_4301.build import prior,reference,valid,reachable4,prompt,answer
from topological_depth_3701.build import face_word,VERTICES,EDGES,FACES


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def save(p,rows):
    with p.open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    return sha(p)


def row(world,root,w,ident,**kw):
    return {'id':ident,'world':world,'root':root,'word':w,'length':len(w),'prompt':prompt(root,w),'answer':answer(reference(root,w,world)),**kw}


def probes(folder,world,occupied,rng):
    topology=[];hard=[]
    for dim,cells in enumerate((VERTICES,EDGES,FACES)):
        for cell in cells:
            for item in range(128):
                for _ in range(100000):
                    root=tuple(rng.randrange(8) for _ in range(8))
                    if root in occupied:continue
                    if dim==0:words=[cell]
                    elif dim==1:words=[cell,cell[::-1]]
                    else:
                        L=rng.choice((7,8));words=[face_word(rng,cell,L),face_word(rng,cell,L)]
                        if words[0]==words[1] or words[0][0]==words[1][0]:continue
                    traces=[reference(root,w,world) if dim<2 else valid(root,w,world) for w in words]
                    if any(x is None for x in traces):continue
                    if dim==0 and traces[0][0]==root:continue
                    if dim==2 and traces[0][-1]==traces[1][-1]:continue
                    states={root,*[s for tr in traces for s in tr]}
                    if states&occupied:continue
                    occupied.update(states)
                    for member,w in enumerate(words):topology.append(row(world,root,w,f'topo5001:{world}:{cell}:{item}:{member}',cell=cell,dimension=dim,item=item,member=member))
                    break
                else:raise RuntimeError('topology sampling')
    used=set()
    for L in (7,8):
        for item in range(256):
            for _ in range(200000):
                words=[''.join(rng.choice('ABCDE') for _ in range(L)) for _ in range(2)]
                if any(set(w)!=set('ABCDE') or w in used or any(a==b for a,b in zip(w,w[1:])) for w in words) or words[0][0]==words[1][0]:continue
                root=tuple(rng.randrange(8) for _ in range(8))
                if root in occupied:continue
                traces=[valid(root,w,world) for w in words]
                if any(x is None for x in traces) or traces[0][-1]==traces[1][-1]:continue
                short=reachable4(root,world)
                if any(tr[-1] in short for tr in traces):continue
                states={root,*[s for tr in traces for s in tr]}
                if states&occupied:continue
                occupied.update(states);used.update(words)
                for member,w in enumerate(words):hard.append(row(world,root,w,f'hard5001:{world}:{L}:{item}:{member}',pair_id=item,member=member))
                break
            else:raise RuntimeError('hard sampling')
    assert len(topology)==5760 and len(hard)==1024
    return {'topology':save(folder/'topology_bank.jsonl',topology),'hard':save(folder/'hard_bank.jsonl',hard)}


def pattern(rng,L,k):
    while True:
        w=[rng.randrange(k) for _ in range(L)]
        if len(set(w))==k and all(a!=b for a,b in zip(w,w[1:])):return w


def continuation(folder,world,occupied,rng):
    pairs=[]
    for L in (7,8):
        for group in range(100):
            a=pattern(rng,L,2);b=pattern(rng,L,3)
            for mult in (1,2,3,4):
                for shift in range(5):
                    wa=''.join('ABCDE'[(mult*x+shift)%5] for x in a);wb=''.join('ABCDE'[(mult*x+shift)%5] for x in b);pairs.append((wa,wb))
    for j in range(48):
        w=rng.choice('ABCDE');pairs.append((w,w))
    for j in range(48):
        w=''.join(rng.sample('ABCDE',2));pairs.append((w,w))
    assert len(pairs)==4096
    rng.shuffle(pairs);banks={'two':[],'three':[]}
    for i,(wa,wb) in enumerate(pairs):
        for _ in range(100000):
            root=tuple(rng.randrange(8) for _ in range(8));traces=[reference(root,w,world) for w in (wa,wb)];states={root,*[s for tr in traces for s in tr]}
            if states&occupied:continue
            occupied.update(states)
            for arm,w in (('two',wa),('three',wb)):banks[arm].append(row(world,root,w,f'train5001:{world}:{i}'))
            break
        else:raise RuntimeError('continuation sampling')
    counts={arm:Counter(''.join(r['word'] for r in bank)) for arm,bank in banks.items()};assert counts['two']==counts['three']
    return {arm:{'sha256':save(folder/f'train_{arm}.jsonl',bank),'operator_tokens':dict(counts[arm]),'length_counts':dict(Counter(r['length'] for r in bank)),'variety_counts':dict(Counter(len(set(r['word'])) for r in bank))} for arm,bank in banks.items()}


def main():
    assert not (HERE/'data_manifest.json').exists()
    result={}
    for wi,world in enumerate(('r1603','r4004')):
        folder=HERE/world;folder.mkdir(exist_ok=True);occupied,_,inputs=prior(world)
        extra=[f'counterfactual_state_use_4301/bank_{world}.jsonl',f'conditional_branch_4501/bank_{world}.jsonl',f'order_length_factorial_4901/bank_{world}.jsonl']
        if world=='r1603':extra+=['topology_selective_intervention_4601/train_clean.jsonl','single_face_gate_4701/training_pool.jsonl','single_face_gate_4701/topology_bank.jsonl','single_face_gate_4701/four_bank.jsonl']
        for name in extra:
            for line in (BASE/name).read_text().splitlines():
                r=json.loads(line);root=tuple(r['root']);occupied.update((root,*reference(root,r['word'],world)))
        source=BASE/('prospective_support_3803/source_train.jsonl' if world=='r1603' else 'prospective_world_4004/source_train.jsonl');shutil.copyfile(source,folder/'source_train.jsonl')
        rng=random.Random(500120261110+wi)
        result[world]={'source_sha256':sha(folder/'source_train.jsonl'),'probes':probes(folder,world,occupied,rng),'continuations':continuation(folder,world,occupied,rng)}
    (HERE/'data_manifest.json').write_text(json.dumps({'status':'FROZEN_R5001_MODEL_BLIND_DATA','worlds':result},indent=2));print('FROZEN_R5001_MODEL_BLIND_DATA',sha(HERE/'data_manifest.json'))


if __name__=='__main__':main()
