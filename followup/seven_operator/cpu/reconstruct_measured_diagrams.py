"""Reconstruct measured barcodes from restored case reports, on CPU."""
import argparse,json
from pathlib import Path
from measure_labelled_zigzags import analyze
def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cases=[]
    for suite in ('confirmation_factored_5300','confirmation_equal_dose_5400'):
        reg=json.loads((a.study_root/suite/'REGISTRATION.json').read_text())
        for cfg in reg['cases']:
            root=a.study_root/cfg['directory'];final=root/'FINAL_RESULT.json'
            if not final.exists():cases.append({'case':cfg['case'],'assessment':suite,'status':'NO_COMPLETE_WHOLE_PROGRAM_RESULT'});continue
            f=json.loads(final.read_text());r=analyze(json.loads((root/f['report']).read_text()));r['assessment']=suite;cases.append(r)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'Postprocessing of actual measured diagrams, no fitting or queries.','cases':cases},indent=2));print(json.dumps({'cases':len(cases),'exact_barcode_cases':sum(c.get('checkpoint_restricted_barcodes_equal',False) for c in cases)}))
if __name__=='__main__':main()
