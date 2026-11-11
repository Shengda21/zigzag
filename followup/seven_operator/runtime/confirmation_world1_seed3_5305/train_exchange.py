import os
import argparse,hashlib,json,math,random,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;MODEL=os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')


def main():
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=('exchange','clean'),required=True);p.add_argument('--phase',type=int,choices=(1,2),required=True);p.add_argument('--chunk',type=int,choices=(1,2),required=True);p.add_argument('--source',required=True);a=p.parse_args();m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    seed=json.loads((HERE/'case.json').read_text())['training_seed']+1000+a.phase*100+a.chunk;torch.set_num_threads(8);random.seed(seed);np.random.seed(seed%2**32);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token_id=151645;rows=[json.loads(l) for l in (HERE/f'train_{a.phase}.jsonl').read_text().splitlines()];random.Random(seed).shuffle(rows);encoded=[]
    for r in rows:
        u={'role':'user','content':r['prompt']};pr=tok.encode(tok.apply_chat_template([u],tokenize=False,add_generation_prompt=True),add_special_tokens=False);ans=r['answer'] if a.branch=='exchange' else r['old_answer'];full=tok.encode(tok.apply_chat_template([u,{'role':'assistant','content':ans}],tokenize=False),add_special_tokens=False);assert full[:len(pr)]==pr;encoded.append((full,[-100]*len(pr)+full[len(pr):]))
    assert len(encoded)==602*64;lengths=[len(x) for x,y in encoded];maximum=max(lengths);aa=np.full((len(encoded),maximum),151645,dtype=np.int64);bb=np.zeros_like(aa);cc=np.full_like(aa,-100)
    for i,(x,y) in enumerate(encoded):aa[i,:len(x)]=x;bb[i,:len(x)]=1;cc[i,:len(y)]=y
    ids_cache=torch.from_numpy(aa).to('cuda');mask_cache=torch.from_numpy(bb).to('cuda');lab_cache=torch.from_numpy(cc).to('cuda');del encoded,aa,bb,cc;src=Path(a.source);base=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'},local_files_only=True);model=PeftModel.from_pretrained(base,src,is_trainable=True);model.train();model.config.use_cache=False;params=[v for v in model.parameters() if v.requires_grad];optim=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True);out=HERE/f'results_{a.branch}_{a.phase}_{a.chunk}';out.mkdir(exist_ok=False);tick=time.monotonic()
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for step in range(1,603):
            sl=slice((step-1)*64,step*64);width=max(lengths[sl]);ids=ids_cache[sl,:width];mask=mask_cache[sl,:width];labels=lab_cache[sl,:width]
            optim.zero_grad(set_to_none=True);target=labels[:,1:];valid=target!=-100;counts=valid.sum(1);owner=torch.arange(64,device='cuda')[:,None].expand_as(target)[valid]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1];losses=F.cross_entropy(logits[valid].float(),target[valid],reduction='none');loss=(losses/counts[owner]).sum()/64
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);lr=(3e-6+97e-6*(1+math.cos(math.pi*max(0,step-16)/586))/2)*min(step/16,1)
            for group in optim.param_groups:group['lr']=lr
            optim.step();record={'branch':a.branch,'phase':a.phase,'chunk':a.chunk,'step':step,'loss':loss.detach().item(),'grad_norm':norm.item(),'lr':lr,'seconds':time.monotonic()-tick};log.write(json.dumps(record)+'\n')
            if step%56==0:print(json.dumps(record),flush=True)
            del ids,mask,labels,target,valid,counts,owner,logits,losses,loss
    dest=out/'adapter';model.save_pretrained(dest);(dest/'runtime.json').write_text(json.dumps({'arguments':vars(a),'seed':seed,'steps':602,'seconds':time.monotonic()-tick,'source_sha256':hashlib.sha256((src/'adapter_model.safetensors').read_bytes()).hexdigest(),'weights_sha256':hashlib.sha256((dest/'adapter_model.safetensors').read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
