"""Fixed six-run constructive sufficiency and paired contrast analysis."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent


def main():
    rows=[];contrasts=[];rng=np.random.default_rng(500120261113);missing=[]
    for world in ('r1603','r4004'):
        for seed in (1,2,3):
            scores={}
            for stage in ('source','two','three'):
                path=HERE/world/f'probe_s{seed}_{stage}/analysis.json'
                if not path.exists():missing.append(str(path.relative_to(HERE)));continue
                a=json.loads(path.read_text());scores[stage]=a
                rows.append({'world':world,'seed':seed,'stage':stage,'strict7':a['hard']['strict_pairs']['7'],'strict8':a['hard']['strict_pairs']['8'],'minimum_unary':a['topology']['minimum_vertex'],'minimum_binary':a['topology']['minimum_edge'],'minimum_face':a['topology']['minimum_face'],'betti80':a['betti80'],'format_eos':a['format_eos'],'endpoint_gate':a['endpoint_gate']})
            if not {'two','three'}<=scores.keys():continue
            ix=rng.integers(0,256,(10000,256))
            for depth in ('7','8'):
                d=np.array(scores['three']['hard']['pair_flags'][depth])-np.array(scores['two']['hard']['pair_flags'][depth]);dist=d[ix].mean(axis=1)
                contrasts.append({'world':world,'seed':seed,'depth':int(depth),'three_minus_two':float(d.mean()),'paired95':np.quantile(dist,[.025,.975]).tolist(),'bonferroni_one_sided_lower_12':float(np.quantile(dist,.05/12)),'bonferroni_one_sided_upper_12':float(np.quantile(dist,1-.05/12)),'two_noninferior_pm05_conditional_on_model':bool(np.quantile(dist,1-.05/12)<.05)})
    complete=not missing;tworows=[r for r in rows if r['stage']=='two'];threerows=[r for r in rows if r['stage']=='three']
    result={'status':'COMPLETE_R5001_SIX_SEED_PAIRED_TEST' if complete else 'INCOMPLETE_R5001','missing':missing,'rows':rows,'contrasts':contrasts,'two_operator_constructive_sufficiency':bool(complete and len(tworows)==6 and all(r['endpoint_gate'] for r in tworows)),'three_operator_all_endpoints_pass':bool(complete and len(threerows)==6 and all(r['endpoint_gate'] for r in threerows)),'training_seed_inference':'Six world/seed outcomes are retained; item-bootstrap contrasts condition on each trained model and are not a population training-seed equivalence test.'}
    (HERE/'AGGREGATE.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('contrasts','missing')},indent=2))


if __name__=='__main__':main()
