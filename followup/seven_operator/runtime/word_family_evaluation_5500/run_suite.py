"""Append the frozen evaluation after the existing fixed-dose suite."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def main():
    reg=json.loads((HERE/'REGISTRATION.json').read_text());progress=[]
    while not(BASE/'confirmation_equal_dose_5400/COMPLETE').exists():time.sleep(20)
    for cfg in reg['cases']:
        parent=BASE/cfg['fixed_directory']
        while not(parent/'CASE_RAW_AUDIT.json').exists():time.sleep(15)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
        root=BASE/cfg['directory'];assert not list(root.glob('COMPLETE*'));env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        with (root/'run.log').open('ab',buffering=0) as f:code=subprocess.run([sys.executable,'-u','run_evaluation.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:(root/'COMPLETE_EXECUTION_ERROR').write_text(json.dumps({'exit':code,'log':'run.log'}));raise RuntimeError('Evaluation error: preserve originals, do not continue silently.')
        subprocess.run([sys.executable,str(BASE/'stage_terminal_archives.py'),'--case',cfg['case'],cfg['directory']],check=True)
        progress.append({'case':cfg['case'],'exit':code});(HERE/'PROGRESS.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress[-1]),flush=True)
    (HERE/'COMPLETE').write_text('All six new-pattern evaluations completed; inspect every scientific and raw result.\n')
if __name__=='__main__':main()
