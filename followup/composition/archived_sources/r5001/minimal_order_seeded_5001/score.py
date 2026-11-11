"""Independent per-item score reconstruction for each fixed endpoint."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'topological_depth_3701'))
import topology as t
from independent_scoring import score_predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--world',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct',local_files_only=True);folder=HERE/a.world;out=folder/f'probe_s{a.seed}_{a.stage}';banks={k:[json.loads(l) for l in (folder/f'{k}_bank.jsonl').read_text().splitlines()] for k in ('topology','hard')}
    top=score_predictions(out/'predictions_topology.jsonl',banks['topology'],tok,'topology');hard=score_predictions(out/'predictions_hard.jsonl',banks['hard'],tok,'hard');counts=list(top['program_counts'].values())+list(hard['program_counts'].values());mask=t.closed_mask(top['scores'],.8)
    result={'world':a.world,'seed':a.seed,'stage':a.stage,'topology':top,'hard':hard,'betti80':[t.chain_data(mask,k)[2] for k in range(3)],'format_eos':sum(c['format_eos'] for c in counts)/6784}
    result['endpoint_gate']=hard['strict_pairs']['7']>=205 and hard['strict_pairs']['8']>=205 and top['minimum_vertex']>=.95 and top['minimum_edge']>=.85 and result['format_eos']>=.98
    (out/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('topology','hard')})+' '+json.dumps(hard['strict_pairs']))


if __name__=='__main__':main()
