"""Report every new-pattern case, including incomplete and failed evaluations."""
import csv,json
from pathlib import Path
from audit_word_family_inheritance import audit
BASE=Path(__file__).resolve().parent.parent;ROOT=BASE/'word_family_evaluation_5500'
def main():
    reg=json.loads((ROOT/'REGISTRATION.json').read_text());rows=[];lines=['# Disjoint program-pattern evaluation','', 'Same six inherited sources, no new training. One separate simultaneous 95% family; missing outcomes are not zero.','', '| Case | World / source | Status | Raw audit | Two archives |','| --- | --- | --- | --- | --- |'];reports=[];csv_rows=[]
    for cfg in reg['cases']:
        folder=BASE/cfg['directory'];result=json.loads((folder/'RESULT.json').read_text()) if (folder/'RESULT.json').exists() else None;raw=json.loads((folder/'CASE_RAW_AUDIT.json').read_text()) if (folder/'CASE_RAW_AUDIT.json').exists() else None;archive=json.loads((folder/'ARCHIVE_VERIFICATION.json').read_text()) if (folder/'ARCHIVE_VERIFICATION.json').exists() else None
        complete=result is not None;passed=result['joint_word_family_case_pass'] if complete else None;audited=bool(raw and raw['status']=='PASS_FROZEN_CASE_RAW_RECONSTRUCTION');archived=bool(archive and archive['status']=='VERIFIED_COMPLETE_'+cfg['case'].upper()+'_ARCHIVE')
        if audited and archived:audit(folder)
        status=('PASS' if passed else 'FAILED') if complete else ('DEPENDENCY FAILED' if raw and raw.get('dependency_failure') else 'PENDING')
        rows.append({'case':cfg['case'],'complete_evaluation':complete,'scientific_case_pass':passed,'raw_audit_pass':audited,'two_archives_verified':archived,'status':status});lines.append(f'| {cfg["case"]} | {cfg["world"]} / {cfg["replicate"]} | {status} | {audited} | {archived} |')
        if complete:
            reports.append(result)
            for tag,cells in result['endpoints'].items():
                for cell,v in cells.items():csv_rows.append({'case':cfg['case'],'world':cfg['world'],'replicate':cfg['replicate'],'order':cfg['order'],'endpoint':tag,'cell':cell,'strict_pairs':v['strict_pairs'],'n':v['n'],'score':v['score'],'lower':v['bounds'][0],'upper':v['bounds'][1],'predicted_positive':v['predicted_positive']})
    out={'new_training_initializations':0,'inherited_sources':6,'completed_cases':len(reports),'all_six_pass_and_archived':all(r['scientific_case_pass'] and r['raw_audit_pass'] and r['two_archives_verified'] for r in rows),'cases':rows};(ROOT/'SUMMARY.json').write_text(json.dumps(out,indent=2));(ROOT/'RETURNED_EVIDENCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (ROOT/'HIGHER_RESULTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('case','world','replicate','order','endpoint','cell','strict_pairs','n','score','lower','upper','predicted_positive'));writer.writeheader();writer.writerows(csv_rows)
    # Make the full roster available after all terminal raw records/archives return,
    # irrespective of whether scientific criteria succeeded.
    if all(r['raw_audit_pass'] and r['two_archives_verified'] for r in rows):
        tex=['% Generated from all six terminal raw-audited, doubly archived cases.','\\begin{table}[tbhp]','\\centering','\\caption{Disjoint word-pattern evaluation at $128$ independent root clusters per case. Entries are strict paired success counts. Learned minimum covers acquisition and both clean endpoints; low maximum covers source and matched two-operator endpoints. Phase quadruples give $ABCD/ABDE/ACEF/ABCDE$; controls give the minimum across both phases and $ACDEG,BCDEF$. LB is the minimum of eight simultaneous paired-contrast lower bounds. A star marks a failed joint criterion. All sources are inherited.}','\\label{supp:tab:word-patterns}','\\begin{tabular}{lrrrrrr}','\\toprule','Source & Learned min. & Low max. & $A$ & $B$ & Controls & LB\\\\','\\midrule']
        for cfg,row in zip(reg['cases'],rows):
            folder=BASE/cfg['directory'];label=f'W{cfg["world"]}S{cfg["replicate"]}'+('' if row['scientific_case_pass'] else '$^*$')
            if not row['complete_evaluation']:tex.append(label+' & \\multicolumn{6}{c}{No complete whole-program result}\\\\');continue
            r=json.loads((folder/'RESULT.json').read_text());e=r['endpoints'];mn=min(v['strict_pairs'] for tag in ('three','clean1','clean2') for v in e[tag].values());mx=max(v['strict_pairs'] for tag in ('source','two') for v in e[tag].values());triples=[]
            for phase in ('A','B'):
                z=e['exchange'+str(r['order'].index(phase)+1)];triples.append('/'.join(str(z[c]['strict_pairs']) for c in ('ABCD','ABDE','ACEF','ABCDE')))
            control=min(e[t][c]['strict_pairs'] for t in ('exchange1','exchange2') for c in ('ACDEG','BCDEF'));lower=min(v['lower'] for v in r['paired_contrasts'].values())
            tex.append(f'{label} & {mn} & {mx} & '+ ' & '.join(triples)+f' & {control} & {lower:.3f}'+'\\\\')
        tex+=['\\bottomrule','\\end{tabular}','\\end{table}'];(ROOT/'ALL_CASE_RESULTS_TABLE.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
