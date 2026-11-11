"""Rescore all archived E4-v4 predictions with the frozen evaluator, on CPU."""
import argparse,gzip,hashlib,json,sys,tarfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent;CODE=HERE.parent
LOCK='98a95beb672d40cb9c4fdac6420ef1b8e9ee1d5eecb22c34f2d38d333f326c2f'
ANALYZER='171127bf254ab9aa21a9c2647370cdc94bb0f1e8da5e562ed2be2d9461439b67'
SELECTION='74080cf7b4fcccb2e309602d31f889115e1db2f3caab1055fa8436ddc5ea57c7'
def digest(data):return hashlib.sha256(data).hexdigest()
def require(ok,detail):
    if not ok:raise ValueError(detail)
class HashReader:
    def __init__(self,f):self.f=f;self.hash=hashlib.sha256()
    def read(self,n=-1):
        data=self.f.read(n);self.hash.update(data);return data
def analyzer():
    lock=(CODE/'CONFIRM_LOCK.json').read_bytes();require(digest(lock)==LOCK,'Frozen confirmation lock');files=json.loads(lock)['files']
    require(files['analyze.py']==ANALYZER,'Frozen analyzer identity')
    for name in ('analyze.py','generate.py','vendor/cw_zigzag.py','vendor/zz_score.py','vendor/spec_world.py'):
        require(digest((CODE/name).read_bytes())==files[name],name)
    sys.path.insert(0,str(CODE));import analyze
    return analyze
def rebuild_world(path,record,members):
    frozen=analyzer();lookup={m['member']:m for m in members};require(len(lookup)==34,record['archive']);data={}
    with Path(path).open('rb') as f:
        source=HashReader(f)
        with tarfile.open(fileobj=source,mode='r|gz') as tar:
            for item in tar:
                if item.name not in lookup:continue
                m=lookup[item.name];key=(m['arm'],m['step']);require(item.isfile() and key not in data,item.name);stored=tar.extractfile(item).read();require(digest(stored)==m['stored_sha256'],item.name);data[key]=(stored,m)
        while source.read(8*1024**2):pass
    require(source.hash.hexdigest()==record['sha256'],record['archive']);require(set(data)=={(a,s) for a in ('edge','joint') for s in range(0,513,32)},record['archive'])
    rows=[];total=0
    for arm in ('edge','joint'):
        prev=None
        for step in range(0,513,32):
            stored,m=data.pop((arm,step));raw=gzip.decompress(stored) if m['member'].endswith('.gz') else stored;require(digest(raw)==m['source_sha256'],m['member']);items=[json.loads(line) for line in raw.splitlines()];total+=len(items)
            q=frozen.extract(items,prev);prev=q;rows.append(dict(world=record['world'],arm=arm,step=step,source_sha256=m['source_sha256'],**q))
    return {'archive':record['archive'],'world':record['world'],'phase':record['phase'],'raw_files':len(rows),'rescored_model_predictions':total,'rows':rows}
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();require(1<=a.workers<=8,'Worker count');frozen=analyzer();manifest=json.loads((HERE/'DATA_MANIFEST.json').read_text());plans=[]
    for r in manifest['archives']:
        paths=[q for q in (a.archive_root/r['archive'],a.archive_root/'e4_v4_confirmation'/r['archive']) if q.is_file()];require(len(paths)==1,r['archive']);q=paths[0];require(q.stat().st_size==r['bytes'],r['archive']);plans.append((str(q),r,[m for m in manifest['raw_eval_files'] if m['archive']==r['archive']]))
    require(len(plans)==8,'Eight archived worlds');out=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(rebuild_world,*plan) for plan in plans]
        for job in as_completed(jobs):
            r=job.result();out.append(r);print(json.dumps({k:v for k,v in r.items() if k!='rows'}),flush=True)
    features={phase:[row for w in sorted(out,key=lambda r:r['world']) if w['phase']==phase for row in w['rows']] for phase in ('pilot','confirm')};reference={'pilot':CODE/'analysis/pilot/FEATURES.json','confirm':HERE/'FEATURES.json'};comp={}
    for phase,pth in reference.items():
        old=json.loads(pth.read_text());new=features[phase];require(len(new)==len(old),(phase,'row count'));diff=[(r['world'],r['arm'],r['step']) for r,q in zip(new,old) if r!=q];require(not diff,(phase,'feature mismatches',diff[:8]));comp[phase]={'rows':len(new),'all_feature_fields_exact':True,'frozen_features_sha256':digest(pth.read_bytes())}
    selection=(CODE/'analysis/pilot/SELECTION.json').read_bytes();require(digest(selection)==SELECTION,'Pilot-selected recipes');selected=json.loads(selection);refit={}
    import numpy,sklearn
    for phase,pth in (('pilot',CODE/'analysis/pilot/REPORT.json'),('confirm',HERE/'REPORT.json')):
        report=json.loads(pth.read_text());current=features[phase];differences={}
        for key,v in report['baselines'].items():
            pred=frozen.fit_predict(features['pilot'],current,key,selected[key]['candidate']);target=v['predictions'];require(len(pred)==len(target),(phase,key));require(all((r['world'],r['arm'],r['step'],r['target'])==(q['world'],q['arm'],q['step'],q['observed']) for r,q in zip(current,target)),(phase,key,'row alignment'));differences[key]=max(abs(u-q['predicted']) for u,q in zip(pred,target))
        paired=frozen.paired(current,phase);expected=report['pairing'];require(all(paired[k]==expected[k] for k in ('edge_equivalence_pass','external_effect_pass','per_world_differences')),(phase,'frozen pairing'));refit[phase]={'maximum_absolute_prediction_difference_by_family':differences,'fixed_candidate_predictions_match_at_1e_10':max(differences.values())<=1e-10,'paired_effects_recomputed':paired,'frozen_report_sha256':digest(pth.read_bytes())}
    result={'status':'PASS_RAW_TO_FROZEN_FEATURES_EXACT','frozen_analyzer_sha256':ANALYZER,'lock_sha256':LOCK,'selection_sha256':SELECTION,'archives':[{k:v for k,v in r.items() if k!='rows'} for r in sorted(out,key=lambda r:r['world'])],'raw_files':sum(r['raw_files'] for r in out),'rescored_model_predictions':sum(r['rescored_model_predictions'] for r in out),'feature_reconstruction':comp,'fixed_recipe_refit':refit,'local_analysis_versions':{'numpy':numpy.__version__,'sklearn':sklearn.__version__},'scope':'Archived predictions are rescored with the frozen parser and feature extractor. Fixed regression recipes are refitted without selection, and compared with the original reports. No language-model inference/training, thresholds, scientific endpoints or archived outputs are changed.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ('status','raw_files','rescored_model_predictions','feature_reconstruction','fixed_recipe_refit')}),flush=True)
if __name__=='__main__':main()
