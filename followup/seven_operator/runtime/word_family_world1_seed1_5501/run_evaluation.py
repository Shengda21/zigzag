"""Evaluate exactly the inherited seven endpoints; no training or selection."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from word_statistics import analyze,TAGS
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=HERE,check=True)
def main():
    m=json.loads((HERE/'manifest.json').read_text());assert all(digest(HERE/n)==h for n,h in m['files'].items());cfg=m['case'];parent=HERE.parent/cfg['fixed_directory']
    assert digest(parent/'manifest.json')==cfg['fixed_manifest_sha256']
    raw=json.loads((parent/'CASE_RAW_AUDIT.json').read_text())
    if not raw.get('scientific_case_pass'):
        (HERE/'CASE_RAW_AUDIT.json').write_text(json.dumps({'case':cfg['case'],'status':'PASS_FROZEN_CASE_RAW_RECONSTRUCTION','scientific_case_pass':False,'dependency_failure':True,'whole_queries':0}));(HERE/'COMPLETE_FAILED_DEPENDENCY').write_text('Inherited fixed-dose dependency failed; no whole-program score imputed.');return
    selection=json.loads((parent/'SOURCE_SELECTION.json').read_text());shutil.copy2(parent/'weights.npy',HERE/'weights.npy');hashes={};paths=selection['residual_paths'];primitive=Path(selection['primitive'])
    for tag in TAGS:hashes[tag]={'policy':digest(HERE/'weights.npy'),'primitive':digest(primitive/'adapter_model.safetensors'),'residual':digest(Path(paths[tag])/'adapter_model.safetensors')}
    (HERE/'SOURCE_SELECTION.json').write_text(json.dumps({'inherited_fixed_case':cfg['fixed_case'],'parent_selection_sha256':digest(parent/'SOURCE_SELECTION.json'),'all_seven_weights_fixed_before_new_queries':True,'paths':paths,'primitive':str(primitive),'weights':hashes},indent=2));(HERE/'ASSEMBLY_WEIGHTS.json').write_text(json.dumps(hashes,indent=2))
    for tag in TAGS:
        run('rollout.py','--primitive',primitive,'--residual',paths[tag],'--tag',tag,'--item-start',0,'--item-stop',128);run('score.py','--tag',tag)
    result=analyze(HERE);run('audit_outputs.py','--study',HERE,'--policy',HERE/'weights.npy','--tags',*TAGS,'--aggregate','RESULT.json','--aggregate-script','word_statistics.py')
    (HERE/'CASE_RAW_AUDIT.json').write_text(json.dumps({'case':cfg['case'],'status':'PASS_FROZEN_CASE_RAW_RECONSTRUCTION','scientific_case_pass':result['joint_word_family_case_pass'],'manifest_verified':True,'raw_audit':'RAW_AUDIT.json','source_weights_archived_in':cfg['fixed_directory'],'evaluation_contains_no_training':True},indent=2));(HERE/'COMPLETE').write_text('New-pattern evaluation ended. Inspect result and raw audit.\n')
if __name__=='__main__':main()
