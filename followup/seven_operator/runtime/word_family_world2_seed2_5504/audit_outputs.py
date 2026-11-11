import os
"""Reconstruct P/Phi/R rollouts from raw tokens and rebuild every score."""
import argparse, ast, hashlib, importlib.util, json, subprocess, sys
import numpy as np
from collections import Counter
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--policy',required=True);ap.add_argument('--tags',nargs='+',required=True);ap.add_argument('--aggregate',default='RESULT.json');ap.add_argument('--aggregate-script',default='aggregate.py');a=ap.parse_args();root=Path(a.study).resolve()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'),local_files_only=True)
    spec=importlib.util.spec_from_file_location('study_prompts',root/'prompts.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    weights=np.load(a.policy);policy_sha=hashlib.sha256(Path(a.policy).read_bytes()).hexdigest();assert weights.shape==(17,9)
    for n in range(1,9):
        for k in range(n+1):
            v=np.exp(weights[k,:n+1]-weights[k,:n+1].max());assert v[k]/v.sum()>.5
    tables={'policy':{},'primitive':{}};origin={}
    for role in tables:
        path=root/f'cache_origin_{role}.json'
        if path.exists():
            cache=json.loads(path.read_text());tables[role]=cache['outputs'];origin[role]={'weight':cache['weights_sha256'],'entries':len(tables[role])}
    decoded={};summaries={};total_programs=total_tokens=0

    def decode(z):
        raw=tuple(z['raw_generated_ids'])
        if raw not in decoded:
            stop=next((j for j,v in enumerate(raw) if v in (151645,151643)),None)
            term=None if stop is None else raw[stop]
            text=tok.decode(raw if stop is None else raw[:stop],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            decoded[raw]=(text,term,len(raw) if stop is None else stop+1)
        text,term,n=decoded[raw];assert text==z['text'] and term==z['terminator_id'];return text,term,n

    def state(text,term):
        digits=text.split(' ')
        if term!=151645 or len(digits)!=8 or any(len(v)!=1 or v not in '01234567' for v in digits):return None
        return list(map(int,digits))

    for tag in a.tags:
        folder=root/f'probe_{tag}';inherited=(folder/'PROVENANCE.json').exists();tables['residual']={};counts=Counter();new=Counter();tokens=0;programs=0;runtime=json.loads((folder/'runtime.json').read_text());assert runtime['weights_sha256']['policy']==policy_sha
        for role in origin:assert runtime['weights_sha256'][role]==origin[role]['weight']
        def visit(role,query,z):
            nonlocal tokens
            counts[role]+=1
            if role=='policy':
                word,history=query.split(':');v=weights[len(history),:len(word)+1];attention=np.exp(v-v.max());attention/=attention.sum();full=np.zeros(9);full[:len(attention)]=attention;prob=np.zeros(8)
                for j,g in enumerate(word):prob['ABCDEFG'.index(g)]+=attention[j]
                prob[7]+=attention[len(word)];category=int(prob.argmax());text='ABCDEFG'[category] if category<7 else 'STOP';term=None;n=0
                assert z['category_id']==category and z['text']==text and np.allclose(z['position_attention'],full,rtol=0,atol=1e-15) and np.allclose(z['category_probabilities'],prob,rtol=0,atol=1e-15)
                assert 'raw_generated_ids' not in z and 'terminator_id' not in z
            else:text,term,n=decode(z)
            if query in tables[role]:assert tables[role][query]==z
            else:tables[role][query]=z;new[role]+=1;tokens+=n
            return text,term
        for kind in ('four','five'):
            bank=[json.loads(l) for l in (root/f'{kind}_bank.jsonl').read_text().splitlines()];bank=[r for r in bank if runtime['item_start']<=r['item']<runtime['item_stop']];seen=set()
            with (folder/f'predictions_{kind}.jsonl').open() as f:
                for line in f:
                    r=json.loads(line);i=r['source_index'];assert i not in seen and r['id']==bank[i]['id'];seen.add(i);gold=bank[i];history='';current=gold['root'];states=[];pi=ri=0;status=None
                    for j,z in enumerate(r['policy_outputs']):
                        text,term=visit('policy',gold['word']+':'+history,z)
                        if text=='STOP':status='self_stopped';break
                        if len(text)!=1 or text not in 'ABCDEFG':status='policy_format_failure';break
                        g=text;old_history=history;history+=g
                        z=r['primitive_outputs'][pi];pi+=1;text,term=visit('primitive',p.primitive_prompt(current,g),z);middle=state(text,term)
                        if middle is None:status='primitive_format_failure';break
                        z=r['residual_outputs'][ri];ri+=1;text,term=visit('residual',p.residual_prompt(middle,old_history[-2:],g),z);next_state=state(text,term)
                        if next_state is None:status='residual_format_failure';break
                        current=next_state;states.append(current)
                    if status is None:assert len(r['policy_outputs'])==16;status='iteration_cap_failure'
                    assert j+1==len(r['policy_outputs']) and pi==len(r['primitive_outputs']) and ri==len(r['residual_outputs'])
                    assert history==r['history'] and states==r['states'] and current==r['state'] and status==r['status'];programs+=1
            assert len(seen)==len(bank)
        logical=counts['primitive']+counts['residual'];actual=new['primitive']+new['residual']
        assert runtime['logical_numeric_lm_requests']==logical and runtime['numeric_cache_hits']==logical-runtime['actual_numeric_lm_calls']
        assert runtime['logical_categorical_policy_requests']==counts['policy'] and runtime['categorical_cache_hits']==counts['policy']-runtime['actual_categorical_policy_evaluations'] and runtime['categorical_policy_generated_tokens']==0
        if not inherited:
            assert runtime['actual_numeric_lm_calls']==actual and runtime['actual_numeric_generated_tokens']==tokens and runtime['actual_categorical_policy_evaluations']==new['policy']
        before=json.loads((folder/'analysis.json').read_text());subprocess.run([sys.executable,str(root/'score.py'),'--tag',tag],cwd=root,check=True);assert json.loads((folder/'analysis.json').read_text())==before
        summaries[tag]={'programs':programs,'logical_neural_calls':dict(counts),'new_neural_calls_reconstructed':None if inherited else dict(new),'new_neural_tokens_reconstructed':None if inherited else tokens,'inherited_initial_output':inherited};total_programs+=programs;total_tokens+=tokens
    source=(root/'rollout.py').read_text();imports=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert set(imports)<=set(('os','argparse','hashlib','json','time','pathlib','prompts','policy','torch','transformers','peft'))
    assert all(s not in source for s in ("['answer']","['length']",'coefficients','execute(','reference('))
    if (root/a.aggregate).exists():
        before=json.loads((root/a.aggregate).read_text());subprocess.run([sys.executable,str(root/a.aggregate_script)],cwd=root,check=True);assert json.loads((root/a.aggregate).read_text())==before
    result={'status':'PASS_RAW_CATEGORICAL_POLICY_AND_LM_OUTPUTS','endpoints':summaries,'total_programs_reconstructed':total_programs,'scores_and_aggregate_rebuilt':True,'policy_all_assignment_certificate_verified':True,'categorical_probabilities_reconstructed_from_actual_weights':True,'no_synthetic_policy_EOS':True,'new_calls_and_tokens_verified_against_query_caches':True,'cache_origin':origin,'no_oracle_action_state_stop_in_rollout':True,'rollout_sha256':hashlib.sha256((root/'rollout.py').read_bytes()).hexdigest()}
    (root/'RAW_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
