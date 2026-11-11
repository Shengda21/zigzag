"""Wait for all primary cases, then run the registered fixed-dose supplement."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def main():
    while not(BASE/'confirmation_factored_5300/COMPLETE').exists():time.sleep(30)
    reg=json.loads((HERE/'REGISTRATION.json').read_text());finished=[]
    for cfg in reg['cases']:
        parent=BASE/cfg['primary_directory']
        while not(parent/'CASE_RAW_AUDIT.json').exists():time.sleep(15)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
        root=BASE/cfg['directory'];assert not list(root.glob('COMPLETE*'));env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        with (root/'run.log').open('ab',buffering=0) as log:code=subprocess.run([sys.executable,'-u','run_equal_dose.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:(root/'COMPLETE_EXECUTION_ERROR').write_text(json.dumps({'exit':code,'log':'run.log'}))
        finished.append({'case':cfg['case'],'exit':code});(HERE/'PROGRESS.json').write_text(json.dumps({'finished':finished},indent=2));print(json.dumps(finished[-1]),flush=True)
    (HERE/'COMPLETE').write_text('All inherited fixed-dose trajectories terminated; inspect all science and raw audits.\n')
if __name__=='__main__':main()
