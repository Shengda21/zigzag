"""Choose on diagnostics only; never inspect four-operator labels."""
import argparse
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args()
    records=[];selected=None
    for step in (32,64,96,128):
        x=json.loads((HERE/f'probe_{a.arm}_{step:04d}_topology/analysis.json').read_text())
        s=x['scores'];other=min(v for c,v in s.items() if len(c)==3 and c!=a.arm)
        good=s[a.arm]<.60 and other>=.80 and x['minimum_vertex']>=.95 and x['minimum_edge']>=.85 and x['format_eos']>=.98
        records.append({'step':step,'target':s[a.arm],'minimum_other_face':other,'pass':good})
        if good and selected is None:selected=step
    result={'target':a.arm,'manipulation_pass':selected is not None,'selected_step':selected,'evaluation_step':selected or 128,'checkpoints':records}
    (HERE/f'selection_{a.arm}.json').write_text(json.dumps(result,indent=2))
    print(result['evaluation_step'])


if __name__=='__main__':main()
