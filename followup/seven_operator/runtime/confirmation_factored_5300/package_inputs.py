"""Package original frozen inputs plus the documented prompt dependency repair."""
import hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    reg=json.loads((HERE/'REGISTRATION.json').read_text());sources={};original=0
    for cfg in reg['cases']:
        root=BASE/cfg['directory'];manifest=root/'manifest.json';assert digest(manifest)==cfg['manifest_sha256']
        data=json.loads(manifest.read_text());names=set(data['files'])|{'manifest.json','prompts.py','DEPENDENCY_REPAIR.json'}
        for name,sha in data['files'].items():assert digest(root/name)==sha;original+=1
        assert digest(root/'prompts.py')=='9092adc3253c18a0a7e0274eb6ffe39d257450efef68b945070813fa0a31a262'
        for name in sorted(names):sources[str(Path(cfg['directory'])/name).replace('\\','/')]=root/name
    common=('REGISTRATION.json','PROTOCOL.md','run_six.py','REPRODUCE.md','BASE_PUBLIC_REVISION_CHECK.json','TECHNICAL_DEPENDENCY_REPAIR.json','FINAL_METHOD_DECISION.json','DESCRIPTIVE_READOUT_PLAN.json','collect_results.py','package_inputs.py')
    for name in common:sources['confirmation_factored_5300/'+name]=HERE/name
    for name in ('audit_confirmation_case.py','audit_component_outputs.py','audit_confirmation_initialization.py','analyze_confirmation_transitions.py','stage_terminal_archives.py','postprocess_confirmation_cases.py'):
        sources[name]=BASE/'cloud_tools'/name
    expected={name:digest(path) for name,path in sources.items()}
    dest=HERE/'reproducibility_inputs_with_dependency.tar.gz'
    with tarfile.open(dest,'w:gz',compresslevel=1) as tar:
        for name,path in sorted(sources.items()):tar.add(path,arcname=name,recursive=False)
    actual={}
    with tarfile.open(dest,'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            actual[member.name]=h.hexdigest()
    assert actual==expected
    out={'status':'PASS_INPUT_ONLY_ARCHIVE_CONTENTS_VERIFIED','file':dest.name,'bytes':dest.stat().st_size,'sha256':digest(dest),'registered_cases':len(reg['cases']),'original_frozen_case_files_unchanged':original,'files':expected,'scope':'Original registration and case manifests remain unchanged. The exact preexisting prompts.py dependency and repair records are included separately. No outputs, selected weights or terminal flags are included.'}
    (HERE/'INPUT_ARCHIVE_VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='files'}))
if __name__=='__main__':main()
