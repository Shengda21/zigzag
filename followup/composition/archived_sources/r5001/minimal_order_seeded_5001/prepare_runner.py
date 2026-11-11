"""Freeze complete code and data before any new source model is trained."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent


def main():
    shutil.copyfile(BASE/'reversible_intervention_4103/analyze.py',HERE/'independent_scoring.py')
    names=['PROTOCOL.md','build.py','train.py','evaluate.py','score.py','aggregate.py','token_audit.py','prepare_runner.py','independent_scoring.py','data_manifest.json','run.sh']
    names += [f'{w}/{n}' for w in ('r1603','r4004') for n in ('source_train.jsonl','train_two.jsonl','train_three.jsonl','topology_bank.jsonl','hard_bank.jsonl')]
    m={'status':'FROZEN_BEFORE_R5001_NEW_SOURCE_TRAINING','files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}}
    (HERE/'manifest.json').write_text(json.dumps(m,indent=2));print(m['status'],hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest())


if __name__=='__main__':main()
