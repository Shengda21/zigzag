"""Cross-check new inference weights against the archived fixed-dose endpoints.

This audit does not query, fit, select, or alter a scientific criterion.
"""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
TAGS=('source','two','three','exchange1','clean1','exchange2','clean2')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());cfg=manifest['case'];parent=BASE/cfg['fixed_directory']
    assert all(digest(root/n)==h for n,h in manifest['files'].items())
    raw=json.loads((root/'CASE_RAW_AUDIT.json').read_text())
    if raw.get('dependency_failure'):
        assert raw['whole_queries']==0 and not list(root.glob('probe_*'))
        out={'case':cfg['case'],'status':'PASS_EXPLICIT_NO_QUERY_DEPENDENCY_FAILURE','scientific_case_pass':False}
    else:
        assert digest(parent/'manifest.json')==cfg['fixed_manifest_sha256'];parent_weights=json.loads((parent/'ASSEMBLY_WEIGHTS.json').read_text());selection=json.loads((parent/'SOURCE_SELECTION.json').read_text());own=json.loads((root/'SOURCE_SELECTION.json').read_text());assembled=json.loads((root/'ASSEMBLY_WEIGHTS.json').read_text())
        assert own['parent_selection_sha256']==digest(parent/'SOURCE_SELECTION.json') and own['paths']==selection['residual_paths'] and own['primitive']==selection['primitive'] and own['all_seven_weights_fixed_before_new_queries']
        actual={}
        for tag in TAGS:
            expected=parent_weights[tag+'_L1'];assert own['weights'][tag]==assembled[tag]==expected
            runtime=json.loads((root/f'probe_{tag}/runtime.json').read_text());assert runtime['weights_sha256']==expected and runtime['item_start']==0 and runtime['item_stop']==128
            assert digest(root/'weights.npy')==expected['policy'];actual[tag]=expected
        parent_archive=json.loads((parent/'ARCHIVE_VERIFICATION.json').read_text());assert parent_archive['weights_retained_losslessly_in_archive']>=8
        out={'case':cfg['case'],'status':'PASS_EXACT_SEVEN_INHERITED_ENDPOINTS','parent_case':cfg['fixed_case'],'parent_archive_sha256':parent_archive['sha256'],'all_seven_runtime_hashes_equal_archived_fixed_dose_assembly':True,'frozen_new_input_bytes_verified':True,'weights':actual,'scope':'Post-return raw provenance audit only. No new inference, fitting, selection or changed statistical gate.'}
    (root/'INHERITANCE_AUDIT.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
    reg=json.loads((BASE/'word_family_evaluation_5500/REGISTRATION.json').read_text())
    for cfg in reg['cases']:
        root=BASE/cfg['directory']
        if (root/'CASE_RAW_AUDIT.json').exists() and (root/'ARCHIVE_VERIFICATION.json').exists():print(json.dumps(audit(root)))
