"""Complete-roster tables, retaining scientific failures and missing whole results."""
from pathlib import Path

BASE=Path(__file__).resolve().parent.parent
CELLS=('ABCD','ABDE','ACEF','ABCDE','ACDEG','BCDEF')

def case_label(case):
    return str(case['world'])+'/'+str(case['replicate'])+('$^*$' if case['scientific_case_pass'] is False else '')


def write_case_tables(folder,title,summary,reports):
    if not all(c.get('raw_audit_pass') and c.get('two_archive_copies_verified') for c in summary['cases']):
        return False
    by_case={r['case']:r for r in reports}
    key='primary' if title=='Primary' else 'fixed-dose'
    tex=['% Actual returned data for all six terminal cases, including failures.',
         '\\begin{table}[htbp]','\\centering',
         '\\caption{'+title+' assessment: all registered sources. An asterisk marks a failed registered case. Counts are out of the row-specific number of root clusters. Source/two columns give the maximum across six external tasks; acquired and clean columns give their minimum. The last column is the minimum of eight simultaneous paired-contrast lower bounds for that case. No seed pooling is used.}',
         '\\label{supp:tab:'+key+'-sources}',
         '\\begin{tabular}{lrrrrrrr}','\\toprule',
         'World/seed & $n$ & Source & Two & Three & Clean 1 & Clean 2 & Contrast LB\\\\','\\midrule']
    for c in summary['cases']:
        label=case_label(c);r=by_case.get(c['case'])
        if r is None:
            tex.append(label+' & \\multicolumn{7}{l}{No complete whole-program result}\\\\')
            continue
        count=lambda tag,cell:sum(r['endpoints'][tag]['flags'][cell])
        counts=[max(count(tag,cell) for cell in CELLS) for tag in ('source','two')]
        counts += [min(count(tag,cell) for cell in CELLS) for tag in ('three','clean1','clean2')]
        lower=min(x['lower'] for x in r['paired_contrasts'].values())
        tex.append(label+' & '+str(r['n'])+' & '+' & '.join(map(str,counts))+f' & {lower:.3f}'+r'\\')
    tex += ['\\bottomrule','\\end{tabular}','\\end{table}',
            '\\begin{table}[htbp]','\\centering',
            '\\caption{'+title+' assessment: labelled exchange for every source. Ordered pairs list strict counts at semantic phases (A,B), irrespective of training order. The final column is the minimum across both phases and both retained five-operator controls ($ACDEG$, $BCDEF$). Denominators are the row-specific $n$ above.}',
            '\\label{supp:tab:'+key+'-exchange}',
            '\\begin{tabular}{llrrrrr}','\\toprule',
            'World/seed & Order & $ABCD$ & $ABDE$ & $ACEF$ & $ABCDE$ & Controls\\\\','\\midrule']
    for c in summary['cases']:
        label=case_label(c);r=by_case.get(c['case'])
        if r is None:
            tex.append(label+' & '+c['order']+' & \\multicolumn{5}{l}{No complete whole-program result}\\\\')
            continue
        tags=['exchange'+str(r['order'].index(phase)+1) for phase in 'AB']
        values=['$('+','.join(str(sum(r['endpoints'][tag]['flags'][cell])) for tag in tags)+')$' for cell in CELLS[:4]]
        control=min(sum(r['endpoints'][tag]['flags'][cell]) for tag in tags for cell in CELLS[4:])
        tex.append(label+' & '+c['order']+' & '+' & '.join(values)+' & '+str(control)+r'\\')
    tex += ['\\bottomrule','\\end{tabular}','\\end{table}']
    (BASE/folder/'ALL_CASE_RESULTS_TABLES.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
    return True
