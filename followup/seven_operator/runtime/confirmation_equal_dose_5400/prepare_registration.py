"""Freeze the fixed-dose supplement without changing any primary input."""
import datetime,hashlib,itertools,json,shutil,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_rows(path,rows):
    with path.open('w',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
def main():
    assert not(HERE/'REGISTRATION.json').exists()
    primary_path=BASE/'confirmation_factored_5300/REGISTRATION.json';reg=json.loads(primary_path.read_text());template=BASE/reg['cases'][0]['directory'];sys.path.insert(0,str(template))
    from iid_probes import generate,A,B,DELTA
    from rules import coefficients,reference,primitive
    cases=[]
    for index,p in enumerate(reg['cases'],1):
        original=BASE/p['directory'];old=json.loads((original/'manifest.json').read_text());assert digest(original/'manifest.json')==p['manifest_sha256'];assert all(digest(original/n)==v for n,v in old['files'].items())
        cfg={**old['case'],'case':f'r{5400+index}','directory':p['directory'].replace('confirmation_','equal_dose_').replace(str(5300+index),str(5400+index)),'primary_case':p['case'],'primary_directory':p['directory'],'primary_manifest_sha256':p['manifest_sha256'],'evaluation_seed':p['evaluation_seed']+100_000_000_000_000,'fixed_exchange_chunks':2,'source_initializations_inherited':True}
        root=BASE/cfg['directory'];root.mkdir(exist_ok=False)
        for name in old['files']:
            if name not in ('case.json','PROTOCOL.md','DATA_AUDIT.json','topology_bank.jsonl','four_bank.jsonl','five_bank.jsonl'):shutil.copyfile(original/name,root/name)
        for name in ('prompts.py','DEPENDENCY_REPAIR.json'):shutil.copyfile(original/name,root/name)
        shutil.copyfile(HERE/'PROTOCOL.md',root/'PROTOCOL.md');shutil.copyfile(HERE/'run_equal_dose.py',root/'run_equal_dose.py');(root/'case.json').write_text(json.dumps(cfg,indent=2),newline='\n')
        blocked=set()
        for name in ('train_primitive.jsonl','train_identity.jsonl','train_three.jsonl','test_primitive.jsonl','test_identity.jsonl','test_three.jsonl'):
            with (root/name).open() as f:
                for line in f:blocked.add(tuple(json.loads(line)['state']))
        banks,audit=generate(cfg['world'],cfg['evaluation_seed'],512,blocked,include_primitive_halves=True);coef=coefficients(cfg['world']);families=[coef,*({f:tuple(a^b for a,b in zip(v,DELTA)) if f in bad else v for f,v in coef.items()} for bad in (A,B))];paths=0
        for kind,rows in banks.items():
            for row in rows:
                assert row['answer']==' | '.join(' '.join(map(str,s)) for s in reference(row['root'],row['word'],coef,cfg['world']))
                for family in families:
                    trace=reference(row['root'],row['word'],family,cfg['world']);states=[row['root'],*trace[:-1]];assert all(tuple(x) not in blocked and primitive(x,g,cfg['world']) not in blocked for x,g in zip(states,row['word']));paths+=1
            write_rows(root/f'{kind}_bank.jsonl',rows)
        audit.update(status='FROZEN_FRESH_FIXED_DOSE_BANK_NO_MODEL_QUERY',case=cfg['case'],primary_case=cfg['primary_case'],primary_data_audit_sha256=digest(original/'DATA_AUDIT.json'),inherited_training_data_unchanged=True,all_reference_paths_and_primitive_halves_checked=paths,model_queries=0);(root/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2))
        names=[str(path.relative_to(root)).replace('\\','/') for path in root.rglob('*') if path.is_file()];manifest={'status':'FROZEN_FIXED_DOSE_SUPPLEMENT_BEFORE_ANY_ADDITIONAL_TRAINING_OR_QUERY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':cfg,'files':{n:digest(root/n) for n in sorted(names)}};(root/'manifest.json').write_text(json.dumps(manifest,indent=2));cases.append({**cfg,'manifest_sha256':digest(root/'manifest.json')});print(json.dumps({'frozen':cfg['case'],'root_trials':audit['total_root_draws'],'paths_checked':paths}),flush=True)
    result={'status':'FROZEN_ALL_SIX_FIXED_DOSE_SUPPLEMENT_CASES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_registration_sha256':digest(primary_path),'reason':'Exact total phase-dose matching after primary r5301 selected one/two chunks. Original case1 outcome known before this supplement.','new_independent_task_initializations':0,'inherited_primary_initializations':6,'protocol_sha256':digest(HERE/'PROTOCOL.md'),'cases':cases,'max_case_processes':1,'files':{n:digest(HERE/n) for n in ('PROTOCOL.md','prepare_registration.py','run_equal_dose.py','run_suite.py','audit_equal_dose_case.py','postprocess_suite.py')}};(HERE/'REGISTRATION.json').write_text(json.dumps(result,indent=2))
    with tarfile.open(HERE/'fixed_dose_upload.tar.gz','w:gz',compresslevel=1) as tar:
        for cfg in cases:tar.add(BASE/cfg['directory'],arcname=cfg['directory'])
        for name in (*result['files'],'REGISTRATION.json'):tar.add(HERE/name,arcname=HERE.name+'/'+name)
    print(json.dumps({'registration_sha256':digest(HERE/'REGISTRATION.json'),'archive_bytes':(HERE/'fixed_dose_upload.tar.gz').stat().st_size}),flush=True)
if __name__=='__main__':main()
