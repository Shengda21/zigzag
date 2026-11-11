"""Independent shared-root probes, holding out every reference input state.

This generates data only. It makes no calls to a trained model.
"""
import itertools as it,json,random,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
from rules import OPS,coefficients,primitive,execute,answer,active_faces
A={'ABE','ACF','BCG','EFG','ABF','ADF','AFG'};B={'ABC','AEF','BEG','CFG','ABF','ADF','AFG'};DELTA=(1,3,5,7,1,3,5,7)


def programs(rng):
    result=[]
    for k in (1,2,3):
        for cell_tuple in it.combinations(OPS,k):
            cell=''.join(cell_tuple);perms=list(map(''.join,it.permutations(cell)));words=(cell,) if k==1 else tuple(rng.sample(perms,2))
            result.extend(('topology',cell,member,w) for member,w in enumerate(words))
    for cell in ('ABCD','ABDE','ACEF'):
        q=''.join(rng.sample(cell,4));z=q[0]+q[2]+q[1]+q[3]
        for member,w in enumerate((q+q,z+z)):
            assert active_faces(w)==set(map(''.join,it.combinations(cell,3)));result.append(('four',cell,member,w))
    for cell in ('ABCDE','ACDEG','BCDEF'):
        mapping=dict(zip('ABCDE',rng.sample(cell,5)));words=tuple(''.join(mapping[g] for g in w) for w in ('BACDBAEC','BDECBADE'));assert set.union(*(active_faces(w) for w in words))==set(map(''.join,it.combinations(cell,3)));result.extend(('five',cell,member,w) for member,w in enumerate(words))
    return result


def prepare_prefix_checks(words):
    prefixes=sorted({w[:j] for w in words for j in range(len(w))},key=lambda p:(len(p),p));active={p:active_faces(p) for p in prefixes};checks=set()
    for mode,bad in (('old',set()),('A',A),('B',B)):
        for p in prefixes:checks.add(('old' if not(active[p]&bad) else mode,p))
    return sorted(checks,key=lambda z:(len(z[1]),z[0],z[1])),active


def root_is_valid(root,checks,active,blocked,coefs,world,include_primitive_halves=False,words=()):
    states={('old',''):root}
    if root in blocked:return False
    for mode,prefix in checks:
        if not prefix:continue
        prev=prefix[:-1];bad=A if mode=='A' else B if mode=='B' else set();previous_mode='old' if not(active[prev]&bad) else mode;x=states[previous_mode,prev];g=prefix[-1];x=primitive(x,g,world)
        if include_primitive_halves and x in blocked:return False
        if len(prefix)>=3 and len(set(prefix[-3:]))==3:x=tuple(a^b for a,b in zip(x,coefs[mode][''.join(sorted(prefix[-3:]))]))
        if x in blocked:return False
        states[mode,prefix]=x
    if include_primitive_halves:
        for mode,bad in (('old',set()),('A',A),('B',B)):
            for w in words:
                p=w[:-1];m='old' if not(active[p]&bad) else mode
                if primitive(states[m,p],w[-1],world) in blocked:return False
    return True


def generate(world,seed,n,blocked,include_primitive_halves=False):
    rng=random.Random(seed);original=coefficients(world);coefs={'old':original,**{mode:{f:tuple(a^b for a,b in zip(v,DELTA)) if f in bad else v for f,v in original.items()} for mode,bad in (('A',A),('B',B))}};rows={'topology':[],'four':[],'five':[]};attempts=[];tick=time.monotonic()
    for item in range(n):
        specs=programs(rng);checks,active=prepare_prefix_checks([w for _,_,_,w in specs]);trials=0
        while True:
            root=tuple(rng.randrange(8) for _ in range(8));trials+=1
            if root_is_valid(root,checks,active,blocked,coefs,world,include_primitive_halves,[w for _,_,_,w in specs]):break
            if trials>=100000:raise RuntimeError('Specified held-out-input population is too sparse; no model query has been made.')
        attempts.append(trials)
        # Crucially, accepted roots and their states are NOT added to blocked.
        for kind,cell,member,w in specs:rows[kind].append({'id':f'iid:{world}:{seed}:{cell}:{item}:{member}','cell':cell,'item':item,'member':member,'root':root,'word':w,'length':len(w),'answer':answer(execute(root,w,original,world))})
    diagnostics={'world':world,'seed':seed,'items':n,'fixed_training_input_states':len(blocked),'total_root_draws':sum(attempts),'maximum_draws_per_item':max(attempts),'empirical_acceptance_fraction':n/sum(attempts),'seconds':time.monotonic()-tick,'sampling':'Each item independently draws its program set, then a uniform root conditional on every old/A/B reference numerical input avoiding the fixed training-input set. Accepted evaluation states never modify that set.','all_reference_intermediate_numeric_inputs_held_out':True,'primitive_half_step_inputs_also_held_out':include_primitive_halves,'model_queries':0};return rows,diagnostics
