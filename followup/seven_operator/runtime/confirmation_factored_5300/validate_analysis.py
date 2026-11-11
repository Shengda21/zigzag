"""Synthetic software preflight only; temporary fixtures are not experiments."""
import itertools as it,json,tempfile
from pathlib import Path
from confirm_statistics import analyze,TAGS,HIGH
from iid_probes import A,B
HERE=Path(__file__).resolve().parent


def fixture(root,order):
    (root/'case.json').write_text(json.dumps({'case':'SYNTHETIC_SOFTWARE_FIXTURE','world':1,'order':order}));cells=[*'ABCDEFG',*map(''.join,it.combinations('ABCDEFG',2)),*map(''.join,it.combinations('ABCDEFG',3))]
    for look in (1,2):
        for tag in TAGS:
            bad=set()
            if tag.startswith('exchange'):bad=A if order[int(tag[-1])-1]=='A' else B
            d={'item_start':256*(look-1),'item_stop':256*look,'topology':{'flags':{}},'four':{'flags':{}},'five':{'flags':{}}}
            for c in cells:
                value=len(c)<3 or (tag not in ('source','two') and c not in bad);d['topology']['flags'][c]={'prefix':[int(value)]*256}
            for c in HIGH:
                value=tag not in ('source','two') and not bool(set(map(''.join,it.combinations(c,3)))&bad);d['four' if len(c)==4 else 'five']['flags'][c]={'strict':[int(value)]*256}
            dest=root/f'probe_{tag}_L{look}';dest.mkdir();(dest/'analysis.json').write_text(json.dumps(d))


def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='simods_software_not_experiment_') as name:
        root=Path(name)
        for order in ('AB','BA'):
            path=root/order;path.mkdir();fixture(path,order);r=analyze(path,1);assert r['joint_confirmation_case_pass'] and r['n']==256 and r['labelled_chain_witness']['connecting_rank']==1;cert=r['robust_window_certificates'];assert cert['exchange_kappa_lower_bounds_via_stability']['both']==1 and cert['exact_robust_counts_if_determined']=={'exchange_birth2':3,'exchange_death2':3,'exchange_transition2':1,'learning_birth2':20,'learning_transition2':0,'full_history_alpha2':0};r=analyze(path,2);assert r['joint_confirmation_case_pass'] and r['n']==512;checks[order+'_canonical_phase_mapping_and_look_concatenation']=True
        path=root/'wrong_crossover';path.mkdir();fixture(path,'AB');q=path/'probe_exchange1_L1/analysis.json';d=json.loads(q.read_text());d['four']['flags']['ABCD']['strict']=[0]*256;q.write_text(json.dumps(d));r=analyze(path,1);assert not r['joint_confirmation_case_pass'] and not r['point_estimate_gates_pass'] and 'crossover:1' in r['failed_gates'];checks['incorrect_behavior_rejected_without_spurious_pass']=True
        path=root/'missing_vertex';path.mkdir();fixture(path,'AB');q=path/'probe_exchange1_L1/analysis.json';d=json.loads(q.read_text());d['topology']['flags']['A']['prefix']=[0]*256;q.write_text(json.dumps(d));r=analyze(path,1);assert not r['joint_confirmation_case_pass'] and not r['actual_algebra']['valid_full_lower_skeleton'] and not r['labelled_chain_witness']['all_witness_checks_pass'];checks['actual_closure_used_when_lower_skeleton_fails']=True
    out={'status':'PASS_CPU_SOFTWARE_PREFLIGHT_NOT_MODEL_RESULTS','checks':checks,'fixtures':'Pure all-zero/all-one symbolic flags in deleted temporary directories; no model outputs fabricated or saved as evidence.','confirmation_model_queries':0,'confirmation_training_runs':0};(HERE/'ANALYSIS_SOFTWARE_PREFLIGHT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
