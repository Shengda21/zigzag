"""Report returned evidence only; never substitute predictions for missing runs."""
import importlib.util,json
from pathlib import Path
from confirmation_case_tables import write_case_tables
BASE=Path(__file__).resolve().parent.parent
def main():
    spec=importlib.util.spec_from_file_location('collect_registered',BASE/'confirmation_factored_5300/collect_results.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);summaries={};lines=['# Seven-operator confirmation: returned evidence','', 'Six original task initializations are shared by the primary and additional fixed-dose assessments. Missing results are not zeros. Confidence bands are simultaneous within each assessment separately, conditional on fitted models and its registered generator.','']
    all_reports={}
    for title,folder in (('Primary','confirmation_factored_5300'),('Fixed dose','confirmation_equal_dose_5400')):
        module.HERE=BASE/folder;summary=module.collect();summaries[title]=summary;lines += ['## '+title,'','| Case | World / source | Order | Status | n | Exchange chunks | Two archives |','| --- | --- | --- | --- | ---: | --- | --- |'];reports=[]
        for c in summary['cases']:
            chunks=c.get('selected_training_chunks',{}).get('exchanges',{});dose='/'.join(str(chunks[str(i)]['chunks']) for i in (1,2)) if chunks else 'pending';status=('PASS' if c['scientific_case_pass'] else 'FAILED') if c['scientific_case_pass'] is not None else c['status'];lines.append(f"| {c['case']} | {c['world']} / {c['replicate']} | {c['order']} | {status} | {c.get('n','pending')} | {dose} | {'verified' if c.get('two_archive_copies_verified') else 'pending'} |")
            root=BASE/next(x['directory'] for x in json.loads((BASE/folder/'REGISTRATION.json').read_text())['cases'] if x['case']==c['case'])
            if (root/'FINAL_RESULT.json').exists():
                f=json.loads((root/'FINAL_RESULT.json').read_text());reports.append(json.loads((root/f['report']).read_text()))
        lines.append('');all_reports[title]=reports
        write_case_tables(folder,title,summary,reports)
        if reports:
            minimum_vertex=min(v[0] for r in reports for e in r['endpoints'].values() for cell,v in e['bounds'].items() if len(cell)==1);minimum_contrast=min(v['lower'] for r in reports for v in r['paired_contrasts'].values());lines += [f'Returned whole-program cases: {len(reports)}/6. Minimum simultaneous vertex lower bound: {minimum_vertex:.9f}. Minimum paired-contrast lower bound: {minimum_contrast:.9f}.','']
    complete=all(s['all_six_scientifically_and_computationally_pass'] and all(c.get('two_archive_copies_verified',False) for c in s['cases']) for s in summaries.values());out={'independent_original_task_initializations':6,'assessment_count':2,'all_twelve_trajectories_pass_and_archived':complete,'assessment_summaries':summaries,'scope':'Six inherited source pairs, not twelve independent seeds. Passing evidence does not itself decide scientific novelty or journal readiness.'};dest=BASE/'confirmation_equal_dose_5400';(dest/'ALL_ASSESSMENTS.json').write_text(json.dumps(out,indent=2));(dest/'RETURNED_EVIDENCE.md').write_text('\n'.join(lines),encoding='utf8')
    if complete:
        tex=['% Generated from completed, raw-audited and doubly archived reports only.','\\begin{table}[tbhp]','\\centering','\\caption{Fixed-dose exchange across six inherited sources (two worlds, three sources each). Entries are the range of strict paired rates; brackets contain the envelope of simultaneous conditional confidence intervals. They are not training-seed population intervals.}','\\label{tab:labelled-confirmed}','\\begin{tabular}{lccc}','\\toprule','Phase & $ABCD$ & $ABDE$ & $ACEF$\\\\','\\midrule']
        for phase in ('A','B'):
            values=[]
            for cell in ('ABCD','ABDE','ACEF'):
                rates=[];bounds=[]
                for r in all_reports['Fixed dose']:
                    tag='exchange'+str(r['order'].index(phase)+1);e=r['endpoints'][tag];rates.append(e['scores'][cell]);bounds.append(e['bounds'][cell])
                lo,hi=min(rates),max(rates);point=f'{lo:.3f}' if lo==hi else f'{lo:.3f}--{hi:.3f}';values.append(f'{point} [{min(b[0] for b in bounds):.3f}, {max(b[1] for b in bounds):.3f}]')
            tex.append(phase+' & '+' & '.join(values)+'\\\\')
        tex += ['\\bottomrule','\\end{tabular}','\\end{table}'];(dest/'CONFIRMED_RESULTS_TABLE.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps({'returned_cases':{k:s['completed_whole_cases'] for k,s in summaries.items()},'all_pass_and_archived':complete,'publication_table_written':complete}))
if __name__=='__main__':main()
