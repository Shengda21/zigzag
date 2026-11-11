import os
"""Categorical neural policy + Qwen primitive + Qwen correction; no world code."""
import argparse,hashlib,json,time
from pathlib import Path
from policy import Policy
from prompts import primitive_prompt,residual_prompt
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;MODEL=os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def parse_state(z):
    parts=z['text'].split(' ')
    if z['terminator_id']!=151645 or len(parts)!=8 or any(len(v)!=1 or v not in '01234567' for v in parts):return None
    return list(map(int,parts))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primitive',required=True);ap.add_argument('--residual',required=True);ap.add_argument('--tag',required=True);ap.add_argument('--item-start',type=int,required=True);ap.add_argument('--item-stop',type=int,required=True);a=ap.parse_args();m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    paths={'policy':HERE/'weights.npy','primitive':Path(a.primitive)/'adapter_model.safetensors','residual':Path(a.residual)/'adapter_model.safetensors'};hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()};expected=json.loads((HERE/'ASSEMBLY_WEIGHTS.json').read_text())[a.tag];assert hashes==expected;policy=Policy(paths['policy']);assert policy.certificate()['all_assignments_certificate_pass'];out=HERE/f'probe_{a.tag}';out.mkdir(exist_ok=False)
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;tok.padding_side='left';base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True);model=PeftModel.from_pretrained(base,Path(a.primitive),adapter_name='primitive',is_trainable=False);model.load_adapter(Path(a.residual),adapter_name='residual',is_trainable=False);model.eval();cache={role:{} for role in paths}
    for role in ('policy','primitive'):
        file=HERE/f'cache_{role}.json'
        if file.exists():
            old=json.loads(file.read_text());assert old['weights_sha256']==hashes[role];cache[role]=old['outputs']
    calls=logical=tokens=policy_calls=policy_logical=0;tick=time.monotonic()
    def choose(words,histories):
        nonlocal policy_calls,policy_logical
        policy_logical+=len(words);queries=[w+':'+h for w,h in zip(words,histories)];specs=dict(zip(queries,zip(words,histories)));unseen=list(dict.fromkeys(q for q in queries if q not in cache['policy']))
        if unseen:
            predictions=policy.predict([specs[q][0] for q in unseen],[specs[q][1] for q in unseen]);cache['policy'].update(zip(unseen,predictions));policy_calls+=len(unseen)
        return [cache['policy'][q] for q in queries]
    def generate(prompts,role):
        nonlocal calls,logical,tokens
        logical+=len(prompts);table=cache[role];unseen=list(dict.fromkeys(p for p in prompts if p not in table))
        if unseen:model.set_adapter(role);model.eval()
        for start in range(0,len(unseen),2048):
            ps=unseen[start:start+2048];ss=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps];batch=tok(ss,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');width=batch['input_ids'].shape[1];gen=model.generate(**batch,max_new_tokens=32,do_sample=False,num_beams=1,eos_token_id=[151645,151643],pad_token_id=151645);calls+=len(ps)
            for prompt,z in zip(ps,gen[:,width:]):
                raw=z.tolist();stop=next((j for j,v in enumerate(raw) if v in (151645,151643)),None);term=None if stop is None else raw[stop];text=tok.decode(raw if stop is None else raw[:stop],skip_special_tokens=False,clean_up_tokenization_spaces=False);table[prompt]={'text':text,'terminator_id':term,'raw_generated_ids':raw};tokens+=len(raw) if stop is None else stop+1
        return [table[p] for p in prompts]
    with torch.inference_mode():
        for kind in ('topology','four','five'):
            bank=[json.loads(l) for l in (HERE/f'{kind}_bank.jsonl').read_text().splitlines()];bank=[r for r in bank if a.item_start<=r['item']<a.item_stop];assert bank;records=[{'id':r['id'],'source_index':i,'history':'','state':r['root'],'states':[],'policy_outputs':[],'primitive_outputs':[],'residual_outputs':[],'status':'active'} for i,r in enumerate(bank)]
            for iteration in range(16):
                active=[i for i,r in enumerate(records) if r['status']=='active']
                if not active:break
                outputs=choose([bank[i]['word'] for i in active],[records[i]['history'] for i in active]);moving=[]
                for i,z in zip(active,outputs):
                    r=records[i];r['policy_outputs'].append(z)
                    if z['category_id']==7:r['status']='self_stopped'
                    elif 0<=z['category_id']<7:g=z['text'];moving.append((i,g,r['history']));r['history']+=g
                    else:r['status']='policy_format_failure'
                outputs=generate([primitive_prompt(records[i]['state'],g) for i,g,h in moving],'primitive');correcting=[]
                for (i,g,h),z in zip(moving,outputs):
                    r=records[i];r['primitive_outputs'].append(z);value=parse_state(z)
                    if value is None:r['status']='primitive_format_failure'
                    else:correcting.append((i,g,h,value))
                outputs=generate([residual_prompt(value,h[-2:],g) for i,g,h,value in correcting],'residual')
                for (i,g,h,value),z in zip(correcting,outputs):
                    r=records[i];r['residual_outputs'].append(z);value=parse_state(z)
                    if value is None:r['status']='residual_format_failure'
                    else:r['state']=value;r['states'].append(value)
                print(json.dumps({'tag':a.tag,'kind':kind,'iteration':iteration+1,'active':len(active),'actual_numeric_lm_calls':calls,'seconds':time.monotonic()-tick}),flush=True)
            with (out/f'predictions_{kind}.jsonl').open('w') as f:
                for r in records:
                    if r['status']=='active':r['status']='iteration_cap_failure'
                    f.write(json.dumps(r,separators=(',',':'))+'\n')
    for role in ('policy','primitive'):(HERE/f'cache_{role}.json').write_text(json.dumps({'weights_sha256':hashes[role],'outputs':cache[role]},separators=(',',':')))
    runtime={'tag':a.tag,'item_start':a.item_start,'item_stop':a.item_stop,'weights_sha256':hashes,'logical_numeric_lm_requests':logical,'actual_numeric_lm_calls':calls,'numeric_cache_hits':logical-calls,'actual_numeric_generated_tokens':tokens,'logical_categorical_policy_requests':policy_logical,'actual_categorical_policy_evaluations':policy_calls,'categorical_cache_hits':policy_logical-policy_calls,'categorical_policy_generated_tokens':0,'seconds':time.monotonic()-tick};(out/'runtime.json').write_text(json.dumps(runtime,indent=2));print(json.dumps(runtime),flush=True)


if __name__=='__main__':main()
