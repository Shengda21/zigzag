"""Greedy one-call factorial evaluation, retaining exact generated tokens."""
import argparse,hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
MODEL=Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')
ADAPTERS={'r1603':{'lower':ROOT/'prospective_support_3803/source_results/adapter_0512','trained':ROOT/'prospective_support_3803/results_control/adapter_0128'},'r4004':{'lower':ROOT/'prospective_world_4004/results_repair/adapter_0032','trained':ROOT/'prospective_world_4004/stop_calibration_4007/results_stop/adapter_0032'}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--world',choices=tuple(ADAPTERS),required=True);p.add_argument('--route',choices=('lower','trained'),required=True);a=p.parse_args()
    m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;tok.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True)
    adapter=ADAPTERS[a.world][a.route];model=PeftModel.from_pretrained(base,adapter,is_trainable=False);model.eval()
    bank=[json.loads(l) for l in (HERE/f'bank_{a.world}.jsonl').read_text().splitlines()]
    out=HERE/f'probe_{a.world}_{a.route}';out.mkdir(exist_ok=False);tick=time.monotonic()
    with torch.inference_mode(),(out/'predictions.jsonl').open('w',buffering=1) as f:
        for L in (5,8):
            indices=[i for i,r in enumerate(bank) if r['length']==L]
            for start in range(0,len(indices),512):
                ids=indices[start:start+512];enc=[tok.encode(tok.apply_chat_template([{'role':'user','content':bank[i]['prompt']}],tokenize=False,add_generation_prompt=True),add_special_tokens=False) for i in ids]
                batch=tok.pad({'input_ids':enc},padding=True,return_tensors='pt').to('cuda');width=batch['input_ids'].shape[1]
                gen=model.generate(**batch,max_new_tokens=180,do_sample=False,num_beams=1,repetition_penalty=1.,eos_token_id=[151645,151643],pad_token_id=151645)
                for j,i in enumerate(ids):
                    raw=gen[j,width:].tolist();stop=next((q for q,v in enumerate(raw) if v in (151645,151643)),None);content=raw if stop is None else raw[:stop]
                    f.write(json.dumps({'id':bank[i]['id'],'source_index':i,'raw_generated_ids':raw,'prediction':tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False),'eos':None if stop is None else raw[stop]})+'\n')
                print(json.dumps({'world':a.world,'route':a.route,'length':L,'completed':min(start+512,len(indices)),'seconds':time.monotonic()-tick}),flush=True)
    (out/'runtime.json').write_text(json.dumps({'status':'COMPLETE_R4901_PROBE','adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),'seconds':time.monotonic()-tick},indent=2))


if __name__=='__main__':main()
