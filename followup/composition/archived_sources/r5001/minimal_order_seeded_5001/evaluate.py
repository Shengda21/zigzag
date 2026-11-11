"""Evaluate every fixed source and endpoint; retain raw generated tokens."""
import argparse,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;MODEL=Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--world',choices=('r1603','r4004'),required=True);p.add_argument('--seed',type=int,choices=(1,2,3),required=True);p.add_argument('--stage',choices=('source','two','three'),required=True);a=p.parse_args()
    m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;tok.padding_side='left';base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True)
    folder=HERE/a.world;adapter=folder/f'results_s{a.seed}_{a.stage}/adapter';model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval();out=folder/f'probe_s{a.seed}_{a.stage}';out.mkdir(exist_ok=False);tick=time.monotonic()
    for kind in ('topology','hard'):
        bank=[json.loads(l) for l in (folder/f'{kind}_bank.jsonl').read_text().splitlines()]
        encoded=[tok.encode(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True),add_special_tokens=False) for r in bank]
        groups=[(d,[i for i,r in enumerate(bank) if r['dimension']==d],cap) for d,cap in ((0,40),(1,55),(2,180))] if kind=='topology' else [(L,[i for i,r in enumerate(bank) if r['length']==L],180) for L in (7,8)]
        with torch.inference_mode(),(out/f'predictions_{kind}.jsonl').open('w',buffering=1) as f:
            for group,indices,cap in groups:
                indices.sort(key=lambda i:len(encoded[i]))
                for start in range(0,len(indices),512):
                    chosen=indices[start:start+512];batch=tok.pad({'input_ids':[encoded[i] for i in chosen]},padding=True,return_tensors='pt').to('cuda');width=batch['input_ids'].shape[1];gen=model.generate(**batch,max_new_tokens=cap,do_sample=False,num_beams=1,repetition_penalty=1.,eos_token_id=[151645,151643],pad_token_id=151645)
                    for j,i in enumerate(chosen):
                        raw=gen[j,width:].tolist();stop=next((q for q,v in enumerate(raw) if v in (151645,151643)),None);content=raw if stop is None else raw[:stop]
                        f.write(json.dumps({'id':bank[i]['id'],'source_index':i,'raw_generated_ids':raw,'first_eos_index':stop,'terminator_id':None if stop is None else raw[stop],'decode_cap_reached':stop is None and len(raw)>=cap,'prediction':tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)})+'\n')
                    print(json.dumps({'world':a.world,'seed':a.seed,'stage':a.stage,'bank':kind,'group':group,'completed':min(start+512,len(indices)),'seconds':time.monotonic()-tick}),flush=True)
    (out/'runtime.json').write_text(json.dumps({'status':'COMPLETE_R5001_PROBE','world':a.world,'seed':a.seed,'stage':a.stage,'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),'seconds':time.monotonic()-tick},indent=2))


if __name__=='__main__':main()
