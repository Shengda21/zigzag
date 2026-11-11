"""Draft confirmation runner: all weights fixed before any whole-program query."""
import hashlib,json,subprocess,sys,os
from pathlib import Path
from confirm_statistics import analyze,TAGS
from resources import gpu_access
HERE=Path(__file__).resolve().parent


def run(*args):
    with gpu_access(str(args[0])):subprocess.run([sys.executable,*map(str,args)],cwd=HERE,check=True)


def acquire(role,cap):
    source=None
    for chunk in range(1,cap+1):
        command=['train_components.py','--role',role,'--chunk',chunk]
        if source:command+=['--source',source]
        run(*command);source=HERE/f'results_{role}_{chunk}/adapter';run('evaluate_components.py','--role',role,'--chunk',chunk)
        if json.loads((HERE/f'component_{role}_{chunk}/analysis.json').read_text())['component_gate_pass']:return source,chunk
    return None,None


def fail(reason):
    (HERE/'CONFIRMATION_FAILURE.json').write_text(json.dumps({'joint_confirmation_case_pass':False,'reason':reason,'all_failed_components_retained':True},indent=2));(HERE/'COMPLETE_FAILED_COMPONENT').write_text(reason)


def main():
    m=json.loads((HERE/'manifest.json').read_text());assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items());snapshot=json.loads((HERE/'BASE_MODEL_SNAPSHOT.json').read_text());model_root=Path(os.environ.get('SIMODS_BASE_MODEL','/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'))
    for name,identity in snapshot['files'].items():
        p=model_root/name;assert p.stat().st_size==identity['bytes'];h=hashlib.sha256()
        with p.open('rb') as f:
            for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
        assert h.hexdigest()==identity['sha256']
    run('train_policy.py')
    if not json.loads((HERE/'POLICY_RESULT.json').read_text())['categorical_policy_gate_pass']:return fail('Fixed categorical-policy endpoint failed.')
    primitive,pc=acquire('primitive',2)
    if primitive is None:return fail('Primitive fixed cap failed.')
    identity,ic=acquire('identity',2)
    if identity is None:return fail('Identity fixed cap failed.')
    source=identity;tc=None
    for chunk in (1,2,3):
        run('train_components.py','--role','three','--chunk',chunk,'--source',source);source=HERE/f'results_three_{chunk}/adapter';run('evaluate_components.py','--role','three','--chunk',chunk)
        if json.loads((HERE/f'component_three_{chunk}/analysis.json').read_text())['component_gate_pass']:tc=chunk;break
    if tc is None:return fail('Three-rule fixed cap failed.')
    two=identity
    for chunk in range(1,tc+1):run('train_components.py','--role','two','--chunk',chunk,'--source',two);two=HERE/f'results_two_{chunk}/adapter'
    models={'source':identity,'two':two,'three':source};selection={'primitive_chunks':pc,'identity_chunks':ic,'three_chunks':tc,'exchanges':{}};clean=source
    for phase in (1,2):
        chosen=None
        for chunk in (1,2):
            run('train_exchange.py','--branch','exchange','--phase',phase,'--chunk',chunk,'--source',source);source=HERE/f'results_exchange_{phase}_{chunk}/adapter';run('evaluate_exchange.py','--phase',phase,'--chunk',chunk)
            if json.loads((HERE/f'component_{phase}_{chunk}/analysis.json').read_text())['component_gate_pass']:chosen=chunk;break
        if chosen is None:return fail(f'Exchange phase{phase} fixed cap failed.')
        models[f'exchange{phase}']=source;selection['exchanges'][str(phase)]={'chunks':chosen,'adapter':str(source)}
        for chunk in range(1,chosen+1):run('train_exchange.py','--branch','clean','--phase',phase,'--chunk',chunk,'--source',clean);clean=HERE/f'results_clean_{phase}_{chunk}/adapter'
        models[f'clean{phase}']=clean
    selection['residual_paths']={k:str(v) for k,v in models.items()};selection['primitive']=str(primitive);selection['all_seven_weights_fixed_before_whole_queries']=True;(HERE/'SOURCE_SELECTION.json').write_text(json.dumps(selection,indent=2));weights={}
    for tag,residual in models.items():
        paths={'policy':HERE/'weights.npy','primitive':primitive/'adapter_model.safetensors','residual':residual/'adapter_model.safetensors'};hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
        for look in (1,2):weights[f'{tag}_L{look}']=hashes
    (HERE/'ASSEMBLY_WEIGHTS.json').write_text(json.dumps(weights,indent=2))
    for look in (1,2):
        for tag in TAGS:
            name=f'{tag}_L{look}';run('rollout.py','--primitive',primitive,'--residual',models[tag],'--tag',name,'--item-start',256*(look-1),'--item-stop',256*look);run('score.py','--tag',name)
        result=analyze(HERE,look)
        print(json.dumps({'case':result['case'],'look':look,'n':result['n'],'pass':result['joint_confirmation_case_pass'],'failed_gates':result['failed_gates']}),flush=True)
        if result['joint_confirmation_case_pass'] or not result['point_estimate_gates_pass']:break
    (HERE/'FINAL_RESULT.json').write_text(json.dumps({'case':result['case'],'final_look':look,'joint_confirmation_case_pass':result['joint_confirmation_case_pass'],'report':f'CONFIRMATION_LOOK{look}.json'},indent=2));(HERE/'COMPLETE').write_text('Completion is not a passing scientific gate; inspect FINAL_RESULT.json.\n')


if __name__=='__main__':main()
