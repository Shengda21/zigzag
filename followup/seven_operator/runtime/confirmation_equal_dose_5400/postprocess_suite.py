"""Audit and archive all fixed-dose outcomes, including failed entries."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def main():
    reg=json.loads((HERE/'REGISTRATION.json').read_text())
    for cfg in reg['cases']:
        root=BASE/cfg['directory']
        while not list(root.glob('COMPLETE*')):time.sleep(15)
        subprocess.run([sys.executable,str(HERE/'audit_equal_dose_case.py'),'--study',str(root)],check=True)
        subprocess.run([sys.executable,str(BASE/'stage_terminal_archives.py'),'--case',cfg['case'],cfg['directory']],check=True)
        print('FIXED_DOSE_AUDITED_ARCHIVED',cfg['case'],flush=True)
if __name__=='__main__':main()
