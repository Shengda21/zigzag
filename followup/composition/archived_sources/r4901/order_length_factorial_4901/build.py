"""Freeze matched variety x length probes for fixed adapters."""
import hashlib,itertools as it,json,random,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
from counterfactual_state_use_4301.build import prior,reference,prompt,answer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (HERE/'manifest.json').exists()
    allstats={}
    for wi,world in enumerate(('r1603','r4004')):
        occupied,_,inputs=prior(world);rng=random.Random(490120261110+wi)
        names=[f'counterfactual_state_use_4301/bank_{world}.jsonl',f'conditional_branch_4501/bank_{world}.jsonl']
        if world=='r1603':names+=['topology_selective_intervention_4601/train_clean.jsonl','single_face_gate_4701/training_pool.jsonl','single_face_gate_4701/topology_bank.jsonl','single_face_gate_4701/four_bank.jsonl']
        for name in names:
            for line in (BASE/name).read_text().splitlines():
                row=json.loads(line);root=tuple(row['root']);occupied.update((root,*reference(root,row['word'],world)))
        bank=[]
        for length in (5,8):
            for item in range(256):
                words={}
                for k in (2,3,4,5):
                    subsets=list(map(''.join,it.combinations('ABCDE',k)));cell=subsets[(item+length)%len(subsets)]
                    pair=[]
                    while len(pair)<2:
                        w=''.join(rng.choice(cell) for _ in range(length))
                        if set(w)==set(cell) and all(a!=b for a,b in zip(w,w[1:])) and (not pair or w[0]!=pair[0][0]):pair.append(w)
                    words[k]=(cell,pair)
                for _ in range(100000):
                    root=tuple(rng.randrange(8) for _ in range(8));states={root};records=[]
                    for k,(cell,pair) in words.items():
                        for member,w in enumerate(pair):
                            tr=reference(root,w,world);states.update(tr)
                            records.append({'id':f'factorial4901:{world}:{length}:{k}:{item}:{member}','world':world,'root':root,'word':w,'length':length,'variety':k,'cell':cell,'item':item,'member':member,'prompt':prompt(root,w),'answer':answer(tr),'effective_steps':sum(a!=b for a,b in zip((root,*tr[:-1]),tr)),'distinct_states':len({root,*tr})})
                    if not states&occupied:
                        occupied.update(states);bank.extend(records);break
                else:raise RuntimeError('factorial sampling')
        assert len(bank)==4096
        with (HERE/f'bank_{world}.jsonl').open('w') as f:
            for row in bank:f.write(json.dumps(row,separators=(',',':'))+'\n')
        allstats[world]={'programs':len(bank),'effective_step_sums':{f'{L}:{k}':sum(r['effective_steps'] for r in bank if r['length']==L and r['variety']==k) for L in (5,8) for k in (2,3,4,5)}}
    training={}
    for name in ('prospective_support_3803/source_train.jsonl','prospective_support_3803/control_train.jsonl','prospective_world_4004/source_train.jsonl','prospective_world_4004/repair_train.jsonl','prospective_world_4004/control_train.jsonl','prospective_world_4004/extension_4006/train.jsonl','prospective_world_4004/stop_calibration_4007/train.jsonl'):
        path=BASE/name
        if path.exists():
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            training[name]={'examples':len(rows),'length_counts':dict(sorted(Counter(len(r['word']) for r in rows).items())),'distinct_operator_counts':dict(sorted(Counter(len(set(r['word'])) for r in rows).items())),'sha256':sha(path)}
    (HERE/'cpu_gate.json').write_text(json.dumps({'status':'PASS_MODEL_BLIND_R4901_BANKS','banks':allstats,'training_distributions':training},indent=2))
    names=('PROTOCOL.md','build.py','evaluate.py','analyze.py','cpu_gate.json','bank_r1603.jsonl','bank_r4004.jsonl')
    m={'status':'FROZEN_BEFORE_R4901_QUERY','files':{n:sha(HERE/n) for n in names}}
    (HERE/'manifest.json').write_text(json.dumps(m,indent=2));print(m['status'],sha(HERE/'manifest.json'))


if __name__=='__main__':main()
