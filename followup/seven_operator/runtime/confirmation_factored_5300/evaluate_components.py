import os
import argparse,hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent;MODEL=os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('primitive','identity','three'),required=True);p.add_argument('--chunk',type=int,required=True);a=p.parse_args();m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;tok.padding_side='left';base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True);adapter=HERE/f'results_{a.role}_{a.chunk}/adapter';model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval();bank=[json.loads(l) for l in (HERE/f'test_{a.role}.jsonl').read_text().splitlines()];out=HERE/f'component_{a.role}_{a.chunk}';out.mkdir(exist_ok=False);counts=defaultdict(Counter);tick=time.monotonic()
    with torch.inference_mode(),(out/'predictions.jsonl').open('w',buffering=1) as f:
        for start in range(0,len(bank),2048):
            rows=bank[start:start+2048];ss=[tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True) for r in rows];batch=tok(ss,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda');width=batch['input_ids'].shape[1];gen=model.generate(**batch,max_new_tokens=32,do_sample=False,num_beams=1,eos_token_id=[151645,151643],pad_token_id=151645)
            for j,r in enumerate(rows):
                raw=gen[j,width:].tolist();stop=next((q for q,v in enumerate(raw) if v in (151645,151643)),None);term=None if stop is None else raw[stop];text=tok.decode(raw if stop is None else raw[:stop],skip_special_tokens=False,clean_up_tokenization_spaces=False);c=counts[r['type']];c['n']+=1;c['strict']+=int(text==r['answer'] and term==151645);f.write(json.dumps({'id':r['id'],'source_index':start+j,'prediction':text,'terminator_id':term,'raw_generated_ids':raw},separators=(',',':'))+'\n')
            print(json.dumps({'role':a.role,'chunk':a.chunk,'completed':min(start+2048,len(bank)),'seconds':time.monotonic()-tick}),flush=True)
    mean=sum(c['strict'] for c in counts.values())/len(bank);minimum=min(c['strict']/c['n'] for c in counts.values());floor=.999 if a.role=='primitive' else .99;result={'role':a.role,'chunk':a.chunk,'mean_strict':mean,'minimum_type_strict':minimum,'component_gate_pass':mean>=.999 and minimum>=floor,'types':{k:dict(v) for k,v in counts.items()},'weights_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()};(out/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='types'}),flush=True)


if __name__=='__main__':main()
