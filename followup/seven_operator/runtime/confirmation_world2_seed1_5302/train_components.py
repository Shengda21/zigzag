import os
import argparse,hashlib,json,math,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;MODEL=os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('primitive','identity','two','three'),required=True);p.add_argument('--chunk',type=int,required=True);p.add_argument('--source');a=p.parse_args();m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items());epochs=1 if a.role=='identity' else 2;steps=602*epochs;seed=json.loads((HERE/'case.json').read_text())['training_seed']+{'primitive':0,'identity':100,'two':200,'three':200}[a.role]+a.chunk
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel,LoraConfig,get_peft_model
    torch.set_num_threads(8);random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;encoded=[];name='train_identity.jsonl' if a.role=='two' else f'train_{a.role}.jsonl'
    for line in (HERE/name).read_text().splitlines():
        r=json.loads(line);u={'role':'user','content':r['prompt']};pr=tok.encode(tok.apply_chat_template([u],tokenize=False,add_generation_prompt=True),add_special_tokens=False);full=tok.encode(tok.apply_chat_template([u,{'role':'assistant','content':r['answer']}],tokenize=False),add_special_tokens=False);assert full[:len(pr)]==pr;encoded.append((full,[-100]*len(pr)+full[len(pr):]))
    assert len(encoded)==38528;lengths=[len(x) for x,y in encoded];maximum=max(lengths);aa=np.full((len(encoded),maximum),151645,dtype=np.int64);bb=np.zeros_like(aa);cc=np.full_like(aa,-100)
    for i,(x,y) in enumerate(encoded):aa[i,:len(x)]=x;bb[i,:len(x)]=1;cc[i,:len(y)]=y
    ids_cache=torch.from_numpy(aa).to('cuda');mask_cache=torch.from_numpy(bb).to('cuda');lab_cache=torch.from_numpy(cc).to('cuda');del encoded,aa,bb,cc;order=[]
    for epoch in range(epochs):
        ix=list(range(len(lengths)));random.Random(seed+10000*epoch).shuffle(ix);order.extend(ix)
    base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True);src=Path(a.source) if a.source else None;model=PeftModel.from_pretrained(base,src,is_trainable=True) if src else get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']));model.train();model.config.use_cache=False;params=[v for v in model.parameters() if v.requires_grad];peak=1e-4 if a.role=='identity' else 2e-4;optim=torch.optim.AdamW(params,lr=peak,betas=(.9,.999),weight_decay=.01,fused=True);out=HERE/f'results_{a.role}_{a.chunk}';out.mkdir(exist_ok=False);tick=time.monotonic()
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(1,steps+1):
            ix=order[(step-1)*64:step*64];width=max(lengths[i] for i in ix);idx=torch.tensor(ix,dtype=torch.long,device='cuda');ids=ids_cache[idx,:width];mask=mask_cache[idx,:width];labels=lab_cache[idx,:width];optim.zero_grad(set_to_none=True);target=labels[:,1:];valid=target!=-100;counts=valid.sum(1);owner=torch.arange(64,device='cuda')[:,None].expand_as(target)[valid]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1];losses=F.cross_entropy(logits[valid].float(),target[valid],reduction='none');loss=(losses/counts[owner]).sum()/64
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);lr=(3e-6+(peak-3e-6)*(1+math.cos(math.pi*max(0,step-16)/(steps-16)))/2)*min(step/16,1)
            for group in optim.param_groups:group['lr']=lr
            optim.step();record={'role':a.role,'chunk':a.chunk,'step':step,'loss':loss.detach().item(),'grad_norm':norm.item(),'lr':lr,'seconds':time.monotonic()-tick};log.write(json.dumps(record)+'\n')
            if step%56==0:print(json.dumps(record),flush=True)
            del idx,ids,mask,labels,target,valid,counts,owner,logits,losses,loss
    dest=out/'adapter';model.save_pretrained(dest);(dest/'runtime.json').write_text(json.dumps({'arguments':vars(a),'seed':seed,'updates':steps,'seconds':time.monotonic()-tick,'source_sha256':hashlib.sha256((src/'adapter_model.safetensors').read_bytes()).hexdigest() if src else None,'weights_sha256':hashlib.sha256((dest/'adapter_model.safetensors').read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
