"""Independent token/EOS scoring for the diagnostic and four-subset banks."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from independent_scoring import score_predictions
sys.path.insert(0,str(BASE/'topological_depth_3701'))
import topology as topo


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--step',type=int,default=0);p.add_argument('--bank',choices=('topology','four'),required=True);a=p.parse_args()
    m=json.loads((HERE/'manifest.json').read_text())
    assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==v for n,v in m['files'].items())
    out=HERE/(f'probe_baseline_{a.bank}' if a.arm=='baseline' else f'probe_{a.arm}_{a.step:04d}_{a.bank}')
    runtime=json.loads((out/'runtime.json').read_text());assert runtime['status']=='COMPLETE_R4701_PROBE'
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct',local_files_only=True)
    bank=[json.loads(l) for l in (HERE/f'{a.bank}_bank.jsonl').read_text().splitlines()]
    if a.bank=='topology':
        result=score_predictions(out/'predictions_topology.jsonl',bank,tok,'topology')
        mask=topo.closed_mask(result['scores'],.8)
        result['betti80']=[topo.chain_data(mask,k)[2] for k in range(3)]
        counts=result['program_counts'];result['format_eos']=sum(v['format_eos'] for v in counts.values())/5760
        if a.arm=='baseline':
            assert result['minimum_face']>=.80 and result['minimum_vertex']>=.95 and result['minimum_edge']>=.85, 'R4701 source entry failed'
    else:
        seen=set();flags=defaultdict(dict);program=[]
        for line in (out/'predictions_four.jsonl').read_text().splitlines():
            pred=json.loads(line);i=pred['source_index'];row=bank[i]
            assert i not in seen and row['id']==pred['id'];seen.add(i)
            raw=pred['raw_generated_ids'];stop=next((j for j,t in enumerate(raw) if t in (151645,151643)),None)
            assert stop==pred['first_eos_index']
            ids=raw if stop is None else raw[:stop]
            decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert decoded==pred['prediction']
            strict=int(stop is not None and raw[stop]==151645 and decoded==row['answer'] and not set(tok.all_special_ids)&set(ids))
            prefix=int(decoded.split(' | ')[:8]==row['answer'].split(' | '))
            flags[(row['cell'],row['item'])][row['member']]=strict
            program.append({'source_index':i,'cell':row['cell'],'item':row['item'],'member':row['member'],'strict':strict,'state_prefix':prefix})
        assert len(seen)==1280
        cells=sorted(set(r['cell'] for r in bank))
        paired={c:[int(flags[c,i]=={0:1,1:1}) for i in range(128)] for c in cells}
        result={'pair_flags':paired,'strict_pairs':{c:sum(v) for c,v in paired.items()},'programs':program}
    result.update(arm=a.arm,step=a.step,bank=a.bank,adapter_sha256=runtime['adapter_sha256'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    if a.bank=='topology':
        print(json.dumps({'arm':a.arm,'step':a.step,'betti80':result['betti80'],'scores':result['scores'],'format_eos':result['format_eos']}))
    else:print(json.dumps({'arm':a.arm,'step':a.step,'four_strict_pairs':result['strict_pairs']}))


if __name__=='__main__':main()
