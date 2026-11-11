"""Three independent source initializations, paired full trace continuations."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;MODEL=Path('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--world',choices=('r1603','r4004'),required=True);p.add_argument('--seed',type=int,choices=(1,2,3),required=True);p.add_argument('--stage',choices=('source','two','three'),required=True);a=p.parse_args()
    m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    tokenaudit=json.loads((HERE/'TOKEN_AUDIT.json').read_text());assert tokenaudit['status']=='PASS_EXACT_PAIRED_TOKEN_LENGTHS'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model,PeftModel
    torch.set_num_threads(8);seed=500120261110+100*a.seed+(0 if a.world=='r1603' else 10000);random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    folder=HERE/a.world;bank=folder/('source_train.jsonl' if a.stage=='source' else f'train_{a.stage}.jsonl');rows=[json.loads(l) for l in bank.read_text().splitlines()];steps=512 if a.stage=='source' else 128;assert len(rows)==steps*32
    order=list(range(len(rows)));random.Random(seed).shuffle(order);rows=[rows[i] for i in order]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;encoded=[]
    for row in rows:
        user={'role':'user','content':row['prompt']};assistant={'role':'assistant','content':row['answer']};prefix=tok.encode(tok.apply_chat_template([user],tokenize=False,add_generation_prompt=True),add_special_tokens=False);full=tok.encode(tok.apply_chat_template([user,assistant],tokenize=False),add_special_tokens=False);assert full[:len(prefix)]==prefix;encoded.append((full,[-100]*len(prefix)+full[len(prefix):]))
    base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True)
    source=folder/f'results_s{a.seed}_source/adapter'
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])) if a.stage=='source' else PeftModel.from_pretrained(base,source,is_trainable=True)
    model.train();model.config.use_cache=False;params=[v for v in model.parameters() if v.requires_grad];optim=torch.optim.AdamW(params,lr=1e-4 if a.stage=='source' else 1e-5,betas=(.9,.999),weight_decay=.01,fused=True)
    out=folder/f'results_s{a.seed}_{a.stage}';out.mkdir(exist_ok=False);tick=time.monotonic()
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(1,steps+1):
            block=encoded[(step-1)*32:step*32];width=max(len(x) for x,_ in block);ids=torch.full((32,width),151645,device='cuda',dtype=torch.long);mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
            for j,(x,y) in enumerate(block):ids[j,:len(x)]=torch.tensor(x,device='cuda');mask[j,:len(x)]=1;labels[j,:len(y)]=torch.tensor(y,device='cuda')
            optim.zero_grad(set_to_none=True);target=labels[:,1:];valid=target!=-100;counts=valid.sum(1);owner=torch.arange(32,device='cuda')[:,None].expand_as(target)[valid]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1];losses=F.cross_entropy(logits[valid].float(),target[valid],reduction='none');loss=(losses/counts[owner]).sum()/32
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            lr=(1e-5+9e-5*(1+math.cos(math.pi*max(0,step-16)/496))/2)*min(step/16,1) if a.stage=='source' else 3e-6+7e-6*(1+math.cos(math.pi*(step-1)/127))/2
            for group in optim.param_groups:group['lr']=lr
            optim.step();record={'step':step,'loss':float(loss),'grad_norm':float(norm),'lr':lr,'seconds':time.monotonic()-tick};log.write(json.dumps(record)+'\n')
            if step%32==0:print(json.dumps({'world':a.world,'seed':a.seed,'stage':a.stage,**record}),flush=True)
            del ids,mask,labels,target,valid,counts,owner,logits,losses,loss
    model.save_pretrained(out/'adapter');runtime={'status':'COMPLETE_R5001_TRAINING','world':a.world,'seed_index':a.seed,'training_seed':seed,'stage':a.stage,'updates':steps,'seconds':time.monotonic()-tick,'adapter_sha256':hashlib.sha256((out/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),'source_adapter_sha256':None if a.stage=='source' else hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2))


if __name__=='__main__':main()
