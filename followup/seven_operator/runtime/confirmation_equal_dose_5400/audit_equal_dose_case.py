"""Reconstruct the fixed-dose supplement, explicitly allowing its inherited roots."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;GLOBAL=HERE.parent;sys.path.insert(0,str(GLOBAL))
from audit_component_outputs import audit as audit_components
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args();root=Path(a.study).resolve();m=json.loads((root/'manifest.json').read_text());cfg=m['case'];parent=root.parent/cfg['primary_directory'];assert digest(parent/'manifest.json')==cfg['primary_manifest_sha256'];assert all(digest(root/n)==v for n,v in m['files'].items());assert list(root.glob('COMPLETE*'));components=audit_components(root,standalone=False);training={}
    for folder in sorted(root.glob('results_*')):
        runtime=json.loads((folder/'adapter/runtime.json').read_text());assert digest(folder/'adapter/adapter_model.safetensors')==runtime['weights_sha256'];src=runtime['arguments'].get('source')
        if src:
            p=Path(src).resolve();assert any(p.is_relative_to(r) for r in (root,parent)) and digest(p/'adapter_model.safetensors')==runtime['source_sha256']
        else:assert runtime['arguments']['role'] in ('primitive','identity') and runtime['source_sha256'] is None
        logs=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()];assert len(logs)==runtime.get('updates',runtime.get('steps')) and all(r['step']==i+1 for i,r in enumerate(logs));training[folder.name]={'updates':len(logs),'parent':src,'weights_sha256':runtime['weights_sha256']}
    reused=json.loads((root/'REUSED_COMPONENTS.json').read_text()) if (root/'REUSED_COMPONENTS.json').exists() else []
    for r in reused:
        src=Path(r['source']).resolve();assert src.is_relative_to(parent);dest=root/r['target'];assert digest(src/'adapter/adapter_model.safetensors')==digest(dest/'adapter/adapter_model.safetensors') and digest(src/'training.jsonl')==digest(dest/'training.jsonl') and digest(src/'adapter/runtime.json')==digest(dest/'adapter/runtime.json')
    science=False;whole=None;fixed=None
    if (root/'FINAL_RESULT.json').exists():
        original=json.loads((parent/'CASE_RAW_AUDIT.json').read_text());assert original['status']=='PASS_FROZEN_CASE_RAW_RECONSTRUCTION' and original['scientific_case_pass'];selection=json.loads((root/'SOURCE_SELECTION.json').read_text());assert selection['fixed_phase_chunks']==2;doses=[]
        for branch in ('exchange','clean'):
            previous=Path(selection['residual_paths']['three'])
            for phase in (1,2):
                for index in (1,2):
                    folder=root/f'results_{branch}_{phase}_{index}';r=json.loads((folder/'adapter/runtime.json').read_text());args=r['arguments'];assert args['branch']==branch and args['phase']==phase and args['chunk']==index and r['steps']==602 and r['seed']==cfg['training_seed']+1000+100*phase+index and r['source_sha256']==digest(previous/'adapter_model.safetensors');previous=folder/'adapter'
                tag=f'exchange{phase}' if branch=='exchange' else f'clean{phase}';assert digest(previous/'adapter_model.safetensors')==digest(Path(selection['residual_paths'][tag])/'adapter_model.safetensors');doses.append({'branch':branch,'phase':phase,'chunks':2,'updates':1204})
        exposures=[]
        for phase in (1,2):
            counts=Counter()
            for line in (root/f'train_{phase}.jsonl').read_text().splitlines():
                r=json.loads(line)
                if not r['changed']:continue
                face=''.join(sorted(r['history']+r['action']))
                for g in face:counts[g]+=2
                for i,g in enumerate(face):
                    for h in face[i+1:]:counts[g+h]+=2
            exposures.append(dict(counts))
        assert exposures[0]==exposures[1];fixed={'status':'PASS_FIXED_TWO_CHUNK_SEQUENTIAL_LINEAGE_AND_PHASE_EXPOSURE','doses':doses,'changed_label_operator_pair_counts_each_phase':exposures[0],'independent_new_source_initializations':0}
        for tag,residual in selection['residual_paths'].items():
            expected={'policy':digest(root/'weights.npy'),'primitive':digest(Path(selection['primitive'])/'adapter_model.safetensors'),'residual':digest(Path(residual)/'adapter_model.safetensors')};assembly=json.loads((root/'ASSEMBLY_WEIGHTS.json').read_text())
            for look in (1,2):assert assembly[f'{tag}_L{look}']==expected
        final=json.loads((root/'FINAL_RESULT.json').read_text());tags=[f'{tag}_L{look}' for look in range(1,final['final_look']+1) for tag in ('source','two','three','exchange1','clean1','exchange2','clean2')];subprocess.run([sys.executable,str(root/'audit_outputs.py'),'--study',str(root),'--policy',str(root/'weights.npy'),'--tags',*tags,'--aggregate','NO_AGGREGATE_FILE.json'],cwd=root,check=True);whole=json.loads((root/'RAW_AUDIT.json').read_text());sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('fixed_dose_statistics',root/'confirm_statistics.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
        for look in range(1,final['final_look']+1):
            path=root/f'CONFIRMATION_LOOK{look}.json';before=path.read_bytes();after=stats.analyze(root,look);assert json.loads(before)==json.loads(json.dumps(after)) and path.read_bytes()==before
        assert after['joint_confirmation_case_pass']==final['joint_confirmation_case_pass'];science=final['joint_confirmation_case_pass'];subprocess.run([sys.executable,str(GLOBAL/'analyze_confirmation_transitions.py'),'--study',str(root)],check=True)
    else:assert (root/'COMPLETE_FAILED_COMPONENT').exists() or (root/'COMPLETE_EXECUTION_ERROR').exists()
    result={'status':'PASS_FROZEN_CASE_RAW_RECONSTRUCTION','case':cfg['case'],'scientific_case_pass':science,'completed_component_evaluations':len(components['components']),'training_lineage_and_update_counts':training,'whole_programs_reconstructed':0 if whole is None else whole['total_programs_reconstructed'],'simultaneous_statistics_rebuilt':whole is not None,'fixed_dose_audit':fixed,'source_initializations_inherited_from':cfg['primary_case'],'scope':'This inherited fixed-dose verification does not add independent task seeds or replace the primary result.'};(root/'CASE_RAW_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('training_lineage_and_update_counts','fixed_dose_audit')}))
if __name__=='__main__':main()
