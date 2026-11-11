"""Independent raw reconstruction of a frozen case; no training or new queries."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from audit_component_outputs import audit as audit_components


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args();root=Path(a.study).resolve();m=json.loads((root/'manifest.json').read_text());assert all(digest(root/n)==v for n,v in m['files'].items());assert list(root.glob('COMPLETE*'));component=audit_components(root,standalone=True);training={}
    for folder in sorted(root.glob('results_*')):
        path=folder/'adapter/runtime.json'
        if not path.exists():continue
        runtime=json.loads(path.read_text());assert digest(folder/'adapter/adapter_model.safetensors')==runtime['weights_sha256'];args=runtime['arguments'];parent=args.get('source')
        if parent:assert Path(parent).resolve().is_relative_to(root) and digest(Path(parent)/'adapter_model.safetensors')==runtime['source_sha256']
        else:assert args.get('role') in ('primitive','identity') and args['chunk']==1 and runtime['source_sha256'] is None
        count=0
        with (folder/'training.jsonl').open() as file:
            for line in file:
                r=json.loads(line);count+=1;assert r['step']==count
        assert count==runtime.get('updates',runtime.get('steps'));training[folder.name]={'updates':count,'parent':parent,'weights_sha256':runtime['weights_sha256']}
    final_path=root/'FINAL_RESULT.json';whole=None;science=False
    if final_path.exists():
        repair=json.loads((root/'DEPENDENCY_REPAIR.json').read_text());assert digest(root/'prompts.py')==repair['sha256']=='9092adc3253c18a0a7e0274eb6ffe39d257450efef68b945070813fa0a31a262'
        final=json.loads(final_path.read_text());selection=json.loads((root/'SOURCE_SELECTION.json').read_text());assembly=json.loads((root/'ASSEMBLY_WEIGHTS.json').read_text());p=Path(selection['primitive']);assert p.resolve().is_relative_to(root)
        for tag,path in selection['residual_paths'].items():
            path=Path(path);assert path.resolve().is_relative_to(root);expected={'policy':digest(root/'weights.npy'),'primitive':digest(p/'adapter_model.safetensors'),'residual':digest(path/'adapter_model.safetensors')}
            for look in (1,2):assert assembly[f'{tag}_L{look}']==expected
        tags=[f'{tag}_L{look}' for look in range(1,final['final_look']+1) for tag in ('source','two','three','exchange1','clean1','exchange2','clean2')];subprocess.run([sys.executable,str(root/'audit_outputs.py'),'--study',str(root),'--policy',str(root/'weights.npy'),'--tags',*tags,'--aggregate','NO_AGGREGATE_FILE.json'],cwd=root,check=True);whole=json.loads((root/'RAW_AUDIT.json').read_text());sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('case_confirm_statistics',root/'confirm_statistics.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
        for look in range(1,final['final_look']+1):
            path=root/f'CONFIRMATION_LOOK{look}.json';before_bytes=path.read_bytes();before=json.loads(before_bytes);after=stats.analyze(root,look);assert before==json.loads(json.dumps(after)) and path.read_bytes()==before_bytes
        assert after['joint_confirmation_case_pass']==final['joint_confirmation_case_pass'];science=final['joint_confirmation_case_pass']
    else:assert (root/'COMPLETE_FAILED_COMPONENT').exists() or (root/'COMPLETE_EXECUTION_ERROR').exists()
    if whole is not None:subprocess.run([sys.executable,str(Path(__file__).parent/'analyze_confirmation_transitions.py'),'--study',str(root)],check=True)
    if whole is not None:subprocess.run([sys.executable,str(Path(__file__).parent/'audit_confirmation_initialization.py'),'--study',str(root)],check=True)
    out={'status':'PASS_FROZEN_CASE_RAW_RECONSTRUCTION','case':m['case']['case'],'scientific_case_pass':science,'completed_component_evaluations':len(component['components']),'training_lineage_and_update_counts':training,'whole_programs_reconstructed':0 if whole is None else whole['total_programs_reconstructed'],'simultaneous_statistics_rebuilt':whole is not None,'scope':'A raw reconstruction pass does not convert a failed scientific case into a success.'};(root/'CASE_RAW_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='training_lineage_and_update_counts'}),flush=True)


if __name__=='__main__':main()
