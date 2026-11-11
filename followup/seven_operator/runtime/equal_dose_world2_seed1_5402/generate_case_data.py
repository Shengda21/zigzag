"""Build fresh matched low-order supervision and IID held-out program banks.

This module creates data only. It never imports a trained model or queries one.
"""
import itertools as it,json,random
from collections import Counter
from pathlib import Path


def write_rows(root,name,rows):
    with (root/name).open('w',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')


def build(root,config):
    from rules import OPS,primitive,execute,reference,coefficients,active_faces
    from iid_probes import generate,A,B,DELTA
    from prompts import primitive_prompt,residual_prompt
    world=config['world'];seed=config['data_seed'];rng=random.Random(seed);coef=coefficients(world);alternative={f:tuple(a^b for a,b in zip(v,DELTA)) for f,v in coef.items()};types=[('',g) for g in OPS]+[(a,g) for a in OPS for g in OPS if a!=g]+[(a+b,g) for a in OPS for b in OPS for g in OPS if a!=b and b!=g];assert len(types)==301
    def inverse(y,g):
        a=list(y)
        if g=='C':j=0 if world==1 else 4;a[j]=(a[j]-3)%8;return tuple(a)
        if g=='D':return primitive(y,g,world)
        perm=primitive(tuple(range(8)),g,world);return tuple(a[perm.index(j)] for j in range(8))
    banks={'primitive':[],'identity':[],'three':[]};blocked=set();marginals=[Counter(),Counter()];checked=changed=0
    for h,g in types:
        lo_h=g+h[1] if len(h)==2 and len(set(h+g))==3 else h
        for item in range(128):
            origin=tuple(rng.randrange(8) for _ in range(8));w=h+g;trace=execute(origin,w,coef,world);assert trace==reference(origin,w,coef,world);x=origin if not h else trace[-2];z=primitive(x,g,world);lo_origin=x
            for action in lo_h[::-1]:lo_origin=inverse(lo_origin,action)
            lo_word=lo_h+g;lo_trace=execute(lo_origin,lo_word,coef,world);assert len(set(lo_word))<=2 and not active_faces(lo_word) and lo_trace==reference(lo_origin,lo_word,coef,world)==execute(lo_origin,lo_word,alternative,world) and (lo_origin if not lo_h else lo_trace[-2])==x and lo_trace[-1]==z
            alternate=execute(origin,w,alternative,world);assert (origin if not h else alternate[-2])==x;changed+=int(alternate[-1]!=trace[-1]);checked+=1;blocked.update((tuple(x),tuple(z)));uid=f'{config["case"]}:{h}:{g}:{item}'
            banks['primitive'].append({'id':uid+':P','state':x,'action':g,'type':g,'prompt':primitive_prompt(x,g),'answer':' '.join(map(str,z))})
            banks['identity'].append({'id':uid+':I','state':z,'history':lo_h,'action':g,'type':lo_h+':'+g,'prompt':residual_prompt(z,lo_h,g),'answer':' '.join(map(str,z))})
            banks['three'].append({'id':uid+':T','state':z,'history':h,'action':g,'type':h+':'+g,'prompt':residual_prompt(z,h,g),'answer':' '.join(map(str,trace[-1]))})
            for j,hh in enumerate((lo_h,h)):marginals[j].update(('current:'+g,*(f'h{k}:'+v for k,v in enumerate(hh))))
    assert marginals[0]==marginals[1] and changed==26880
    order=list(range(38528));rng.shuffle(order)
    banks={role:[rows[i] for i in order] for role,rows in banks.items()}
    for role,rows in banks.items():write_rows(root,f'train_{role}.jsonl',rows)
    tests={r:[] for r in banks}
    for h,g in types:
        for item in range(128):
            while True:
                x=tuple(rng.randrange(8) for _ in range(8))
                if x not in blocked:break
            tests['primitive'].append({'id':f'component:{h}:{g}:{item}:P','state':x,'action':g,'type':g,'prompt':primitive_prompt(x,g),'answer':' '.join(map(str,primitive(x,g,world)))})
            while True:
                z=tuple(rng.randrange(8) for _ in range(8))
                if z not in blocked:break
            active=len(h)==2 and len(set(h+g))==3;y=tuple(a^b for a,b in zip(z,coef[''.join(sorted(h+g))])) if active else z
            record={'id':f'component:{h}:{g}:{item}:R','state':z,'history':h,'action':g,'type':h+':'+g,'prompt':residual_prompt(z,h,g),'answer':' '.join(map(str,y))};tests['three'].append(record)
            if not active:tests['identity'].append(record)
    for role,rows in tests.items():write_rows(root,f'test_{role}.jsonl',rows)
    training_input_count=len(blocked)
    blocked.update(tuple(r['state']) for rows in tests.values() for r in rows)
    for phase,letter in enumerate(config['order'],1):
        bad=A if letter=='A' else B
        for kind,rows in (('train',banks['three']),('component',tests['three'])):
            output=[]
            for r in rows:
                h=r['history'];g=r['action'];f=''.join(sorted(h+g)) if len(h)==2 and len(set(h+g))==3 else None;answer=r['answer'];target=' '.join(str(int(v)^d) for v,d in zip(answer.split(),DELTA)) if f in bad else answer;output.append({**r,'old_answer':answer,'answer':target,'changed':f in bad})
            assert sum(r['changed'] for r in output)==5376;write_rows(root,f'{kind}_{phase}.jsonl',output)
    rows,audit=generate(world,config['evaluation_seed'],512,blocked,include_primitive_halves=True);families=[coef,*({f:tuple(a^b for a,b in zip(v,DELTA)) if f in bad else v for f,v in coef.items()} for bad in (A,B))];paths=0
    for kind,bank in rows.items():
        for r in bank:
            assert r['answer']==' | '.join(' '.join(map(str,s)) for s in reference(r['root'],r['word'],coef,world))
            for family in families:
                trace=reference(r['root'],r['word'],family,world);states=[r['root'],*trace[:-1]];assert all(tuple(x) not in blocked and primitive(x,g,world) not in blocked for x,g in zip(states,r['word']));paths+=1
        write_rows(root,f'{kind}_bank.jsonl',bank)
    policy_rng=random.Random(config['policy_data_seed']);policy_train=[];blocked_words=set()
    for i in range(16384):
        length=policy_rng.randrange(1,9);symbols=policy_rng.sample(OPS,1 if length==1 or policy_rng.random()<.1 else 2)
        while True:
            word=''.join(policy_rng.choice(symbols) for _ in range(length))
            if set(word)==set(symbols):break
        k=policy_rng.randrange(length+1);blocked_words.add(word);policy_train.append({'id':f'policy:{i}','word':word,'history':word[:k],'answer':word[k] if k<length else 'STOP'})
    policy_test=[]
    for variety in (2,3,4,5,7):
        for i in range(1000):
            while True:
                symbols=policy_rng.sample(OPS,variety);chars=symbols+[policy_rng.choice(symbols) for _ in range(8-variety)];policy_rng.shuffle(chars);word=''.join(chars)
                if word not in blocked_words:break
            policy_test.append({'id':f'policy:L8:V{variety}:{i}','word':word,'variety':f'L8V{variety}'})
    for length in (1,2,3):
        for i in range(1000):policy_test.append({'id':f'policy:L{length}:{i}','word':''.join(policy_rng.sample(OPS,length)),'variety':f'L{length}V{length}'})
    write_rows(root,'policy_train.jsonl',policy_train);write_rows(root,'policy_test.jsonl',policy_test)
    audit.update({'status':'CONFIRMATION_DATA_ONLY_NO_MODEL_QUERIES','case':config['case'],'low_order_origins_dual_interpreter_checked':checked,'counterfactual_low_order_transcripts_unchanged':checked,'counterfactual_three_targets_changed':changed,'matched_history_position_marginals':True,'training_rows_per_numeric_role':38528,'distinct_numeric_training_inputs':training_input_count,'component_selection_inputs_also_excluded':True,'all_reference_paths_and_primitive_halves_checked':paths,'all512_roots_frozen_before_first_query':True,'policy_training_max_variety':2,'policy_numeric_labels':0,'higher_numeric_training_labels':0});(root/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2),newline='\n')
