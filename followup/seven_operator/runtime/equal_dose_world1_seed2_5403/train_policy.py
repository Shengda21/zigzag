import hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from policy import Policy,CLASSES

HERE=Path(__file__).resolve().parent


def main():
    manifest=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==sha for n,sha in manifest['files'].items())
    seed=json.loads((HERE/'case.json').read_text())['policy_training_seed'];random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);train=[json.loads(l) for l in (HERE/'policy_train.jsonl').read_text().splitlines()];values=torch.zeros(len(train),9,8,dtype=torch.float64);mask=torch.zeros(len(train),9,dtype=torch.bool);counts=[];labels=[]
    for i,r in enumerate(train):
        w=r['word'];assert len(set(w))<=2
        for j,g in enumerate(w):values[i,j,CLASSES.index(g)]=1
        values[i,len(w),7]=1;mask[i,:len(w)+1]=True;counts.append(len(r['history']));labels.append(CLASSES.index(r['answer']) if r['answer']!='STOP' else 7)
    counts=torch.tensor(counts);labels=torch.tensor(labels);weights=torch.nn.Parameter(.01*torch.randn(17,9,dtype=torch.float64));opt=torch.optim.Adam([weights],lr=.1);order=[];tick=time.monotonic()
    for epoch in range(2):
        ix=list(range(len(train)));random.Random(seed+epoch).shuffle(ix);order.extend(ix)
    assert len(order)==128*256
    with (HERE/'policy_training.jsonl').open('w',buffering=1) as log:
        for step in range(128):
            ix=order[step*256:(step+1)*256];a=torch.softmax(weights[counts[ix]].masked_fill(~mask[ix],-torch.inf),dim=1);probs=torch.einsum('bi,bij->bj',a,values[ix]);loss=-torch.log(probs[torch.arange(256),labels[ix]]).mean();opt.zero_grad();loss.backward();opt.step();log.write(json.dumps({'step':step+1,'loss':loss.item()})+'\n')
    np.save(HERE/'weights.npy',weights.detach().numpy());policy=Policy(HERE/'weights.npy');certificate=policy.certificate();test=[json.loads(l) for l in (HERE/'policy_test.jsonl').read_text().splitlines()];records=[{'id':r['id'],'history':'','outputs':[],'status':'active'} for r in test];calls=0
    for _ in range(16):
        active=[i for i,r in enumerate(records) if r['status']=='active']
        if not active:break
        outputs=policy.predict([test[i]['word'] for i in active],[records[i]['history'] for i in active]);calls+=len(active)
        for i,z in zip(active,outputs):
            r=records[i];r['outputs'].append(z)
            if z['text']=='STOP':r['status']='self_stopped'
            else:r['history']+=z['text']
    strata={}
    with (HERE/'policy_predictions.jsonl').open('w') as f:
        for i,(r,gold) in enumerate(zip(records,test)):
            if r['status']=='active':r['status']='iteration_cap_failure'
            s=strata.setdefault(gold['variety'],{'n':0,'strict':0});s['n']+=1;s['strict']+=int(r['status']=='self_stopped' and r['history']==gold['word']);r['source_index']=i;f.write(json.dumps(r,separators=(',',':'))+'\n')
    result={'categorical_policy_gate_pass':certificate['all_assignments_certificate_pass'] and all(v['strict']/v['n']>=.999 for v in strata.values()),'certificate':certificate,'by_stratum':strata,'classifier_evaluations':calls,'numeric_labels_used':0,'language_model_generated_tokens':0,'weights_sha256':hashlib.sha256((HERE/'weights.npy').read_bytes()).hexdigest(),'seconds':time.monotonic()-tick};(HERE/'POLICY_RESULT.json').write_text(json.dumps(result,indent=2));(HERE/'POLICY_COMPLETE').write_text('Inspect fixed endpoint RESULT.json.\n');print(json.dumps({k:v for k,v in result.items() if k!='certificate'},indent=2));print('CERTIFICATE_MIN_MASS',certificate['minimum_correct_position_mass'])


if __name__=='__main__':main()
