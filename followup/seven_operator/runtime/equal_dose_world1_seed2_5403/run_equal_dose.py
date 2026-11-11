"""Fixed two-chunk sequential exchange, with recipe-verified reuse only."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from confirm_statistics import analyze,TAGS
HERE=Path(__file__).resolve().parent
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=HERE,check=True)
def link_tree(source,dest):
    assert not dest.exists()
    shutil.copytree(source,dest,copy_function=os.link)
def fail(reason):
    (HERE/'CONFIRMATION_FAILURE.json').write_text(json.dumps({'joint_confirmation_case_pass':False,'reason':reason,'all_failed_components_retained':True},indent=2));(HERE/'COMPLETE_FAILED_COMPONENT').write_text(reason)
def main():
    m=json.loads((HERE/'manifest.json').read_text());assert all(digest(HERE/n)==v for n,v in m['files'].items());cfg=m['case'];parent=HERE.parent/cfg['primary_directory'];pm=parent/'manifest.json';assert digest(pm)==cfg['primary_manifest_sha256'];primary=json.loads(pm.read_text());assert all(digest(parent/n)==v for n,v in primary['files'].items())
    if (parent/'CONFIRMATION_FAILURE.json').exists():return fail('Original primary case failed; no replacement source or imputed whole-program scores.')
    final=json.loads((parent/'FINAL_RESULT.json').read_text());raw=json.loads((parent/'CASE_RAW_AUDIT.json').read_text())
    if not(final['joint_confirmation_case_pass'] and raw['status']=='PASS_FROZEN_CASE_RAW_RECONSTRUCTION' and raw['scientific_case_pass']):return fail('Original primary scientific/raw gate failed; retain inherited entry failure.')
    ps=json.loads((parent/'SOURCE_SELECTION.json').read_text());reused=[]
    for role in ('primitive','identity','three','two'):
        for src in sorted(parent.glob(f'results_{role}_*')):
            if src.is_dir():link_tree(src,HERE/src.name);reused.append({'kind':'inherited_source','source':str(src),'target':src.name})
        for src in sorted(parent.glob(f'component_{role}_*')):
            if src.is_dir():link_tree(src,HERE/src.name)
    for name in ('weights.npy','POLICY_RESULT.json','policy_training.jsonl','policy_predictions.jsonl'):os.link(parent/name,HERE/name)
    def local_adapter(path):return HERE/Path(path).parent.name/'adapter'
    primitive=local_adapter(ps['primitive']);models={tag:local_adapter(ps['residual_paths'][tag]) for tag in ('source','two','three')};source=models['three'];clean=source
    def chunk(branch,phase,index,src):
        name=f'results_{branch}_{phase}_{index}';old=parent/name;dest=HERE/name;seed=cfg['training_seed']+1000+phase*100+index;eligible=False
        if (old/'adapter/runtime.json').exists():
            runtime=json.loads((old/'adapter/runtime.json').read_text());a=runtime['arguments'];eligible=runtime['seed']==seed and runtime['steps']==602 and all(a[k]==v for k,v in {'branch':branch,'phase':phase,'chunk':index}.items()) and runtime['source_sha256']==digest(src/'adapter_model.safetensors') and runtime['weights_sha256']==digest(old/'adapter/adapter_model.safetensors') and digest(parent/f'train_{phase}.jsonl')==digest(HERE/f'train_{phase}.jsonl') and digest(parent/'train_exchange.py')==digest(HERE/'train_exchange.py')
        if eligible:
            link_tree(old,dest);reused.append({'kind':'exact_fixed_recipe_chunk','source':str(old),'target':name,'parent_sha256':runtime['source_sha256'],'seed':seed})
            if branch=='exchange' and (parent/f'component_{phase}_{index}').is_dir():link_tree(parent/f'component_{phase}_{index}',HERE/f'component_{phase}_{index}')
        else:run('train_exchange.py','--branch',branch,'--phase',phase,'--chunk',index,'--source',src)
        (HERE/'REUSED_COMPONENTS.json').write_text(json.dumps(reused,indent=2));return dest/'adapter'
    selection={k:ps[k] for k in ('primitive_chunks','identity_chunks','three_chunks')};selection['exchanges']={}
    for phase in (1,2):
        for index in (1,2):source=chunk('exchange',phase,index,source)
        if not(HERE/f'component_{phase}_2/analysis.json').exists():run('evaluate_exchange.py','--phase',phase,'--chunk',2)
        if not json.loads((HERE/f'component_{phase}_2/analysis.json').read_text())['component_gate_pass']:return fail(f'Fixed two-chunk phase{phase} failed original component gate; no more training.')
        models[f'exchange{phase}']=source;selection['exchanges'][str(phase)]={'chunks':2,'adapter':str(source)}
        for index in (1,2):clean=chunk('clean',phase,index,clean)
        models[f'clean{phase}']=clean
    selection.update(residual_paths={k:str(v) for k,v in models.items()},primitive=str(primitive),all_seven_weights_fixed_before_whole_queries=True,fixed_phase_chunks=2,inherited_primary_case=cfg['primary_case']);(HERE/'SOURCE_SELECTION.json').write_text(json.dumps(selection,indent=2));weights={}
    for tag,residual in models.items():
        hashes={'policy':digest(HERE/'weights.npy'),'primitive':digest(primitive/'adapter_model.safetensors'),'residual':digest(residual/'adapter_model.safetensors')}
        for look in (1,2):weights[f'{tag}_L{look}']=hashes
    (HERE/'ASSEMBLY_WEIGHTS.json').write_text(json.dumps(weights,indent=2))
    for look in (1,2):
        for tag in TAGS:
            name=f'{tag}_L{look}';run('rollout.py','--primitive',primitive,'--residual',models[tag],'--tag',name,'--item-start',256*(look-1),'--item-stop',256*look);run('score.py','--tag',name)
        result=analyze(HERE,look);print(json.dumps({'case':result['case'],'look':look,'pass':result['joint_confirmation_case_pass'],'failed_gates':result['failed_gates']}),flush=True)
        if result['joint_confirmation_case_pass'] or not result['point_estimate_gates_pass']:break
    (HERE/'FINAL_RESULT.json').write_text(json.dumps({'case':result['case'],'final_look':look,'joint_confirmation_case_pass':result['joint_confirmation_case_pass'],'report':f'CONFIRMATION_LOOK{look}.json'},indent=2));(HERE/'COMPLETE').write_text('Fixed-dose completion. Inspect the scientific result and raw audit.\n')
if __name__=='__main__':main()
