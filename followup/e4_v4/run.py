"""Cloud-only paired runner. No server credentials, network calls or auto-download."""
import argparse,json,os,sys,time
from pathlib import Path
from generate import HERE,CONFIG,rows,sha,dump
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(HERE/'vendor'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','confirm'],required=True)
    p.add_argument('--world',type=int,required=True);p.add_argument('--arm',choices=['edge','joint'],required=True)
    p.add_argument('--model',required=True);p.add_argument('--micro',type=int,default=32);p.add_argument('--eval-batch',type=int,default=256)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    assert a.world in CONFIG[a.phase]
    package=json.loads((HERE/'PACKAGE_MANIFEST.json').read_text())
    for name,h in package['files'].items():assert sha(HERE/name)==h,('Package altered',name)
    if a.phase=='confirm':
        lock=json.loads((HERE/'CONFIRM_LOCK.json').read_text())
        for name,h in lock['files'].items():assert sha(HERE/name)==h,('Frozen file changed',name)
    import zz_train as T
    data=HERE/'data'/a.phase/f'world{a.world}'
    man=json.loads((data/'MANIFEST.json').read_text())
    for n,h in man['hashes'].items():assert sha(data/n)==h,n
    modelpath=Path(a.model);assert modelpath.is_dir()
    mc=json.loads((modelpath/'config.json').read_text())
    assert mc.get('hidden_size')==1536 and mc.get('num_hidden_layers')==28,'Require target Qwen2.5 1.5B architecture'
    weights=sorted(list(modelpath.glob('*.safetensors'))+list(modelpath.glob('*.bin')))
    assert weights,'No model weights found'
    identity={p.name:sha(p) for p in sorted(modelpath.iterdir()) if p.is_file()}
    assert identity,'Empty model identity'
    identity_path=HERE/'MODEL_IDENTITY.json'
    if identity_path.exists():assert json.loads(identity_path.read_text())==identity,'Base weights/tokenizer changed across arms'
    else:dump(identity_path,identity)
    # Same random initialization/dropout seed in each paired arm.
    T.seed(a.world*100+1);tok,model=T.load(str(modelpath),'cuda')
    pair=[T.encoded(tok,rows(data/f'{arm}.jsonl'),max_len=512,max_answer=96) for arm in ['edge','joint']]
    def budget(enc):return [(len(ids),sum(v!=-100 for v in lab)) for ids,lab in enc]
    b0,b1=map(budget,pair)
    mismatches=[i for i,(x,y) in enumerate(zip(b0,b1)) if x!=y]
    audit=dict(records=len(b0),per_record_token_mismatches=len(mismatches),first_mismatches=mismatches[:20],
               input_tokens=[sum(x[0] for x in b) for b in (b0,b1)],answer_tokens=[sum(x[1] for x in b) for b in (b0,b1)],
               padding_counted=False,model_config_sha256=sha(modelpath/'config.json'))
    dump(data/'TOKEN_AUDIT.json',audit)
    assert not mismatches,'Unequal useful input/answer token budgets; stop. Revise under a NEW version before confirmation.'
    if a.preflight_only:print(json.dumps(audit));return
    out=HERE/'runs'/a.phase/f'world{a.world}_{a.arm}'
    if (out/'COMPLETE.json').exists():raise FileExistsError(out)
    ev=rows(data/'PROBES.jsonl')+rows(data/'EXTERNAL.jsonl')
    enc=pair[0 if a.arm=='edge' else 1]
    # Stable paired within-step shuffle, exactly 64 examples per update.
    import random
    rng=random.Random(a.world*100+1)
    for i in range(0,len(enc),CONFIG['batch']):
        block=enc[i:i+CONFIG['batch']];rng.shuffle(block);enc[i:i+CONFIG['batch']]=block
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'RUN.json',dict(arguments=vars(a),config=CONFIG,data_sha256=sha(data/'MANIFEST.json'),token_audit=audit,
         model=str(modelpath),model_identity=identity,model_config_sha256=sha(modelpath/'config.json'),started=time.time(),
         dependencies={n:__import__(n).__version__ for n in ['torch','transformers','peft']},
         code={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}))
    final,stopped=T.train_loop(model,tok,out,enc,512,64,a.micro,'cuda',eval_rows=ev,eval_every=32,save_every=32,
          stop_flag=HERE/'STOP_COMPUTE',seed_value=a.world*100+1,eval_batch=a.eval_batch,max_new_tokens=96,keep_resume=True)
    if not stopped:
        adapters={str(p.relative_to(out)):sha(p) for p in out.glob('step_*/adapter_model.safetensors')}
        assert len(adapters)==17
        dump(out/'COMPLETE.json',dict(finished=time.time(),adapters=adapters,final_summary=final))
if __name__=='__main__':main()
