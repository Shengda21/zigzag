import json,os,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent


def main():
    registration=json.loads((HERE/'REGISTRATION.json').read_text());assert registration['status']=='FROZEN_ALL_SIX_CASES_BEFORE_CONFIRMATION_TRAINING_AND_QUERIES'
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
    env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    if registration['max_case_processes']==2:env['SIMODS_GPU_GATE']=str(HERE/'GPU.lock')
    pending=list(registration['cases']);running=[];finished=[]
    while pending or running:
        while pending and len(running)<registration['max_case_processes']:
            cfg=pending.pop(0);root=BASE/cfg['directory'];assert not list(root.glob('COMPLETE*'));log=open(root/'run.log','ab',buffering=0);p=subprocess.Popen([sys.executable,'-u','run_case.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);running.append((p,cfg,log));print(json.dumps({'started':cfg['case'],'pid':p.pid}),flush=True)
        for entry in list(running):
            p,cfg,log=entry;code=p.poll()
            if code is None:continue
            log.close();running.remove(entry);root=BASE/cfg['directory']
            if code!=0:(root/'COMPLETE_EXECUTION_ERROR').write_text(json.dumps({'exit':code,'log':'run.log'}))
            record={'case':cfg['case'],'exit':code,'markers':[q.name for q in root.glob('COMPLETE*')]};finished.append(record);(HERE/'PROGRESS.json').write_text(json.dumps({'finished':finished,'running':[c['case'] for p,c,l in running],'pending':[c['case'] for c in pending]},indent=2));print(json.dumps(record),flush=True)
        time.sleep(2)
    (HERE/'COMPLETE').write_text('All cases terminated. Scientific success requires the six final reports and independent raw audits.\n')


if __name__=='__main__':main()
