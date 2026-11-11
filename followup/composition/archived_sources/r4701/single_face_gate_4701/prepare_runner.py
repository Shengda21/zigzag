"""Seal the R4701 runner, criterion and data before querying any checkpoint."""
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
FACES=('ABC','ABD','ABE','ACD','ACE','ADE','BCD','BCE','BDE','CDE')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()


def main():
    if (HERE/'manifest.json').exists() and not (HERE/'manifest_initial.json').exists():
        shutil.copyfile(HERE/'manifest.json',HERE/'manifest_initial.json')
    shutil.copyfile(BASE/'reversible_intervention_4103/analyze.py',HERE/'independent_scoring.py')
    text=(BASE/'topology_selective_intervention_4601/train.py').read_text()
    old="choices=('cycle', 'acyclic')"
    assert text.count(old)==1
    text=text.replace(old,'choices='+repr(FACES)).replace('460120261114','470120261114').replace('R4601','R4701')
    (HERE/'train.py').write_text(text)
    text=(BASE/'topology_selective_intervention_4601/evaluate.py').read_text()
    old="choices=('baseline', 'cycle', 'acyclic')"
    assert text.count(old)==1
    text=text.replace(old,'choices='+repr(('baseline',*FACES)))
    text=text.replace("    args = parser.parse_args()", "    parser.add_argument('--bank', choices=('topology','four'), required=True)\n    args = parser.parse_args()")
    old="    out = HERE / (f'probe_{args.arm}' if args.arm == 'baseline'\n                  else f'probe_{args.arm}_{args.step:04d}')"
    new="    out = HERE / (f'probe_baseline_{args.bank}' if args.arm == 'baseline' else f'probe_{args.arm}_{args.step:04d}_{args.bank}')"
    assert text.count(old)==1
    text=text.replace(old,new)
    text=text.replace("for bank_name, expected in (('topology', 5760), ('hard', 1024)):","for bank_name, expected in ((args.bank, 5760 if args.bank == 'topology' else 1280),):")
    text=text.replace("for length in (7, 8)]","for length in (8,)]")
    text=text.replace('R4601','R4701')
    (HERE/'evaluate.py').write_text(text)
    baseline=json.loads((BASE/'reversible_intervention_4103/probe_baseline/runtime.json').read_text())
    repair={'reason':'The scorer import referenced a module absent from the cloud directory. Bundle the identical existing independent scorer locally. Baseline diagnostic generations already exist; no intervention training/query has occurred. Data, treatment, selection rule and outcome are unchanged.',
            'initial_manifest_sha256':sha(HERE/'manifest_initial.json')}
    (HERE/'analysis_import_repair.json').write_text(json.dumps(repair,indent=2))
    names=('PROTOCOL.md','build.py','prepare_runner.py','cpu_gate.json','train.py','evaluate.py','score.py','independent_scoring.py','choose_checkpoint.py','run.sh','run_resume.sh','manifest_initial.json','analysis_import_repair.json','topology_bank.jsonl','four_bank.jsonl',*(f'train_{f}.jsonl' for f in FACES))
    m={'status':'FROZEN_BEFORE_R4701_INTERVENTION_QUERY_AFTER_IMPORT_REPAIR','source_adapter_sha256':baseline['adapter_sha256'],
       'files':{n:sha(HERE/n) for n in names},'faces':FACES}
    (HERE/'manifest.json').write_text(json.dumps(m,indent=2))
    print(json.dumps({'status':m['status'],'manifest_sha256':sha(HERE/'manifest.json'),'files':len(names)}))


if __name__=='__main__':main()
