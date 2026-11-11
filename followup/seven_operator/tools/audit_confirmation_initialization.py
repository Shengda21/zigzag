"""Audit registered seeds using untouched policy rows; no training or LM queries."""
import argparse,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch


def audit(root):
    root=Path(root).resolve();cfg=json.loads((root/'case.json').read_text());w=np.load(root/'weights.npy');g=torch.Generator().manual_seed(cfg['policy_training_seed']);initial=(.01*torch.randn(17,9,dtype=torch.float64,generator=g)).numpy();assert np.array_equal(w[9:],initial[9:]) and not np.array_equal(w[:9],initial[:9]);logs=[json.loads(l) for l in (root/'policy_training.jsonl').read_text().splitlines()];assert len(logs)==128 and all(r['step']==i+1 and math.isfinite(r['loss']) for i,r in enumerate(logs));meta=json.loads((root/'POLICY_RESULT.json').read_text());assert hashlib.sha256((root/'weights.npy').read_bytes()).hexdigest()==meta['weights_sha256'];masses=[]
    for n in range(1,9):
        for k in range(n+1):
            a=np.exp(w[k,:n+1]-w[k,:n+1].max());a/=a.sum();masses.append(float(a[k]));assert a[k]>.5
    assert min(masses)==meta['certificate']['minimum_correct_position_mass'];tests=[json.loads(l) for l in (root/'policy_test.jsonl').read_text().splitlines()];strata=defaultdict(Counter);seen=set();calls=0
    with (root/'policy_predictions.jsonl').open() as file:
        for line in file:
            r=json.loads(line);i=r['source_index'];gold=tests[i];assert i not in seen and r['id']==gold['id'];seen.add(i);word=gold['word'];history='';stopped=False
            for j,z in enumerate(r['outputs']):
                a=np.exp(w[len(history),:len(word)+1]-w[len(history),:len(word)+1].max());a/=a.sum();p=np.zeros(8)
                for position,symbol in enumerate(word):p['ABCDEFG'.index(symbol)]+=a[position]
                p[7]+=a[len(word)];cat=int(p.argmax());text='ABCDEFG'[cat] if cat<7 else 'STOP';full=np.zeros(9);full[:len(a)]=a;assert z['category_id']==cat and z['text']==text and np.allclose(z['category_probabilities'],p,atol=1e-15,rtol=0) and np.allclose(z['position_attention'],full,atol=1e-15,rtol=0);assert 'terminator_id' not in z and 'raw_generated_ids' not in z;calls+=1
                if text=='STOP':assert j+1==len(r['outputs']);stopped=True;break
                history+=text
            assert history==r['history'] and r['status']==('self_stopped' if stopped else 'iteration_cap_failure');c=strata[gold['variety']];c['n']+=1;c['strict']+=int(stopped and history==word)
    assert len(seen)==len(tests)==8000 and calls==meta['classifier_evaluations'] and {k:dict(v) for k,v in strata.items()}==meta['by_stratum'];seeds={}
    for folder in root.glob('results_*'):
        path=folder/'adapter/runtime.json'
        if not path.exists():continue
        runtime=json.loads(path.read_text());args=runtime['arguments'];expected=cfg['training_seed']+(1000+100*args['phase']+args['chunk'] if 'phase' in args else {'primitive':0,'identity':100,'two':200,'three':200}[args['role']]+args['chunk']);assert runtime['seed']==expected;seeds[folder.name]=expected
    out={'status':'PASS_REGISTERED_INITIALIZATION_AND_POLICY_TESTS','case':cfg['case'],'untouched_policy_rows_9_to_16_match_registered_random_initialization_exactly':True,'trained_policy_rows_changed':True,'policy_updates':128,'policy_test_programs':len(tests),'policy_categories_reconstructed':calls,'minimum_universal_correct_position_mass':min(masses),'numeric_training_seeds':seeds,'scope':'No fit is rerun. Unused rows retain their exact seeded initialization because the frozen Adam policy trainer has zero gradient and zero weight decay on them.'};dest=root.parent/'confirmation_factored_5300/POLICY_SEED_AUDITS';dest.mkdir(parents=True,exist_ok=True);(dest/(cfg['case']+'.json')).write_text(json.dumps(out,indent=2));return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);args=ap.parse_args();out=audit(args.study);print(json.dumps({k:v for k,v in out.items() if k!='numeric_training_seeds'}),flush=True)
