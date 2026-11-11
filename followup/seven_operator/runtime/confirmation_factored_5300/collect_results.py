"""Collect all registered cases, retaining missing/failed outcomes explicitly."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
TAGS=('source','two','three','exchange1','clean1','exchange2','clean2')
HIGH=('ABCD','ABDE','ACEF','ABCDE','ACDEG','BCDEF')


def collect():
    registration=json.loads((HERE/'REGISTRATION.json').read_text());cases=[];rows=[]
    for cfg in registration['cases']:
        root=BASE/cfg['directory'];record={'case':cfg['case'],'world':cfg['world'],'replicate':cfg['replicate'],'order':cfg['order'],'status':'PENDING_LOCAL_RETURN','scientific_case_pass':None,'raw_audit_pass':False}
        final_path=root/'FINAL_RESULT.json';failed_path=root/'CONFIRMATION_FAILURE.json'
        if final_path.exists():
            final=json.loads(final_path.read_text());report=json.loads((root/final['report']).read_text());assert report['case']==cfg['case'] and report['joint_confirmation_case_pass']==final['joint_confirmation_case_pass'];record.update(status='COMPLETED_WHOLE_PROGRAM_EVALUATION',scientific_case_pass=final['joint_confirmation_case_pass'],n=report['n'],look=report['look'],failed_gates=report['failed_gates'],actual_algebra=report['actual_algebra'],paired_contrasts=report['paired_contrasts'],robust_window_certificates=report['robust_window_certificates'])
            phase_faces=[[v for c,v in report['endpoints'][tag]['scores'].items() if len(c)==3] for tag in ('exchange1','exchange2')];record['observed_phase_scalar_matching']={'face_score_multisets_equal':sorted(phase_faces[0])==sorted(phase_faces[1]),'face_means':[sum(v)/len(v) for v in phase_faces],'face_minima':[min(v) for v in phase_faces],'threshold_face_counts':{str(t):[sum(s>=t for s in v) for v in phase_faces] for t in (.6,.65,.7,.75,.8,.85,.9)},'role':'Descriptive invariant check, not an additional pass gate or a claim of equal latent population means.'}
            selection=json.loads((root/'SOURCE_SELECTION.json').read_text());record['selected_training_chunks']={k:selection[k] for k in ('primitive_chunks','identity_chunks','three_chunks','exchanges')};record['deployment_repair_resume']=(root/'WHOLE_EVALUATION_RESUMED.json').exists()
            for tag in TAGS:
                phase=cfg['order'][int(tag[-1])-1] if tag.startswith('exchange') else tag
                for cell in HIGH:
                    e=report['endpoints'][tag];flags=e['flags'][cell];rows.append({'case':cfg['case'],'world':cfg['world'],'replicate':cfg['replicate'],'order':cfg['order'],'endpoint':tag,'semantic_phase':phase,'task':cell,'correct':sum(flags),'n':len(flags),'rate':e['scores'][cell],'lower':e['bounds'][cell][0],'upper':e['bounds'][cell][1],'predicted_high':report['predicted_higher_support'][tag][cell]})
        elif failed_path.exists():
            failure=json.loads(failed_path.read_text());assert failure['joint_confirmation_case_pass'] is False;record.update(status='FAILED_FIXED_COMPONENT_CAP',scientific_case_pass=False,reason=failure['reason'],higher_outcomes='Not queried; no imputed task scores.')
        elif (root/'WHOLE_EVALUATION_RESUMED.json').exists():record.update(status='RESUMING_WHOLE_AFTER_TECHNICAL_DEPENDENCY_REPAIR',scientific_case_pass=None)
        elif (root/'COMPLETE_EXECUTION_ERROR').exists():record.update(status='EXECUTION_ERROR_UNRESOLVED',scientific_case_pass=False,execution_error=json.loads((root/'COMPLETE_EXECUTION_ERROR').read_text()))
        if (root/'CASE_RAW_AUDIT.json').exists():
            audit=json.loads((root/'CASE_RAW_AUDIT.json').read_text());assert audit['case']==cfg['case'];record['raw_audit_pass']=audit['status']=='PASS_FROZEN_CASE_RAW_RECONSTRUCTION';assert record['scientific_case_pass']==audit['scientific_case_pass'];record['whole_programs_reconstructed']=audit['whole_programs_reconstructed']
        if (root/'ARCHIVE_VERIFICATION.json').exists():record['two_archive_copies_verified']=True
        cases.append(record)
    whole=[c for c in cases if c['status']=='COMPLETED_WHOLE_PROGRAM_EVALUATION'];audited=[c for c in cases if c['raw_audit_pass']];out={'registered_cases':len(cases),'completed_whole_cases':len(whole),'raw_audited_cases':len(audited),'all_six_scientifically_and_computationally_pass':len(audited)==len(cases)==6 and all(c['scientific_case_pass'] for c in cases),'cases':cases,'scope':'No failed case is replaced or excluded. Intervals are conditional on fitted models and the specified root generator; no training-seed population interval is claimed.'}
    (HERE/'SUMMARY.json').write_text(json.dumps(out,indent=2));fields=('case','world','replicate','order','endpoint','semantic_phase','task','correct','n','rate','lower','upper','predicted_high')
    with (HERE/'HIGHER_RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return out


if __name__=='__main__':
    result=collect();print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
