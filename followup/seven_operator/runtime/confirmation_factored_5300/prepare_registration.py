"""Create all six banks only after an explicit, evidence-based method decision."""
import argparse,datetime,hashlib,json,shutil,tarfile
from pathlib import Path
from generate_case_data import build
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
FILES=('run_case.py','resources.py','train_components.py','train_exchange.py','evaluate_components.py','evaluate_exchange.py','train_policy.py','policy.py','rollout.py','score.py','confirm_statistics.py','chain_witness.py','topology_math.py','rules.py','iid_probes.py','generate_case_data.py','audit_outputs.py')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--create-registration',action='store_true',required=True);a=ap.parse_args();assert not (HERE/'REGISTRATION.json').exists();decision_path=HERE/'FINAL_METHOD_DECISION.json';decision=json.loads(decision_path.read_text());assert decision['chosen_architecture']=='learned_neural_interpreter' and decision['prior_failures_retained'] and decision['six_case_confirmation_standards_unchanged'];prerequisites={}
    for name,file,key in [('attention_assembly_5261','RESULT.json','development_first_acquisition_gate_pass'),('attention_exchange_5262','AGGREGATE.json','development_joint_gate_pass'),('attention_world2_5263','RESULT.json','development_first_acquisition_gate_pass'),('attention_exchange_world2_5264','AGGREGATE.json','development_joint_gate_pass'),('single_face_world1_5281','RESULT.json','single_face_joint_gate_pass'),('single_face_world2_5282','RESULT.json','single_face_joint_gate_pass')]:
        p=BASE/name/file;assert json.loads(p.read_text())[key];assert (BASE/name/'RAW_AUDIT.json').exists();prerequisites[name]={'report_sha256':digest(p),'raw_audit_sha256':digest(BASE/name/'RAW_AUDIT.json')}
    for name in ('key_switch_training_5290','dense_switch_training_5291','trained_content_probe_5292'):
        root=BASE/name;assert (root/'COMPLETE').exists() and (root/'RAW_TOKEN_AUDIT.json').exists();prerequisites[name]={'manifest_sha256':digest(root/'manifest.json'),'raw_audit_sha256':digest(root/'RAW_TOKEN_AUDIT.json')}
    bench=json.loads((HERE/'TRAINING_THROUGHPUT_RESULT.json').read_text());parallel=bool(bench['styles']['numeric']['recommend_parallel']);configs=[]
    for rep in (1,2,3):
        for world in (1,2):
            key=f'r{5300+len(configs)+1}';seed=53000020261110+world*1000000+rep*10000;order=('AB' if rep%2 else 'BA') if world==1 else ('BA' if rep%2 else 'AB');configs.append({'case':key,'directory':f'confirmation_world{world}_seed{rep}_{key[1:]}','world':world,'replicate':rep,'order':order,'data_seed':seed+1,'evaluation_seed':seed+2,'policy_data_seed':seed+3,'policy_training_seed':seed+4,'training_seed':seed+100,'look_sizes':[256,512],'component_caps':{'primitive':2,'identity':2,'three':3,'exchange_per_phase':2},'parallel_training_only':parallel})
    assert len(configs)==6 and all(not(BASE/c['directory']).exists() for c in configs)
    for cfg in configs:
        root=BASE/cfg['directory'];root.mkdir();(root/'case.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8',newline='\n')
        for name in FILES:shutil.copy2(HERE/name,root/name)
        shutil.copy2(HERE/'PROTOCOL.md',root/'PROTOCOL.md');shutil.copy2(BASE/'theory_experiment_closure_4800/BASE_MODEL_SNAPSHOT.json',root/'BASE_MODEL_SNAPSHOT.json');shutil.copy2(BASE/'theory_experiment_closure_4800/ENVIRONMENT_FACTORED_20261110.json',root/'ENVIRONMENT.json');build(root,cfg);names=sorted(p.name for p in root.iterdir() if p.is_file());m={'status':'FROZEN_INDEPENDENT_CONFIRMATION_BEFORE_ANY_CASE_QUERY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':cfg,'files':{n:digest(root/n) for n in names}};(root/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8',newline='\n');cfg['manifest_sha256']=digest(root/'manifest.json');print('FROZEN_CASE',cfg['case'],cfg['manifest_sha256'],flush=True)
    registration={'status':'FROZEN_ALL_SIX_CASES_BEFORE_CONFIRMATION_TRAINING_AND_QUERIES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method_decision_sha256':digest(decision_path),'prerequisites':prerequisites,'cases':configs,'total_error':.05,'score_family':2898,'contrast_family':48,'max_case_processes':2 if parallel else 1,'resource_mode':'parallel training with exclusive inference' if parallel else 'serial','no_failed_seed_replacement':True};(HERE/'REGISTRATION.json').write_text(json.dumps(registration,indent=2),encoding='utf-8',newline='\n')
    with tarfile.open(HERE/'confirmation_upload.tar.gz','w:gz') as tar:
        for name in ('REGISTRATION.json','FINAL_METHOD_DECISION.json','PROTOCOL.md','run_six.py'):tar.add(HERE/name,arcname=HERE.name+'/'+name)
        for cfg in configs:
            for path in sorted((BASE/cfg['directory']).iterdir()):tar.add(path,arcname=cfg['directory']+'/'+path.name)
    print('REGISTERED_ALL_SIX',digest(HERE/'REGISTRATION.json'),flush=True)


if __name__=='__main__':main()
