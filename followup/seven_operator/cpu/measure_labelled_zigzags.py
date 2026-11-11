"""Exact barcodes of returned seven-operator diagrams; no model queries.

Every analysed node has the same complete graph and no three-cells.
H2 therefore embeds in common C2. With a fixed spanning-tree gauge,
H1 duals embed in the common dual of Z1; duality reverses arrows and
preserves interval supports. For either resulting inclusion zigzag,
the generalized rank on a contiguous node window is the dimension of
the intersection of its node subspaces. Möbius inversion gives bars.
"""
import itertools as it,json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
OPS='ABCDEFG';FACES=list(map(''.join,it.combinations(OPS,3)));FI={f:i for i,f in enumerate(FACES)};EDGES=list(map(''.join,it.combinations(OPS[1:],2)));EI={e:i for i,e in enumerate(EDGES)}
def insert(basis,v):
    while v:
        p=v.bit_length()-1
        if p not in basis:basis[p]=v;return True
        v^=basis[p]
    return False
def member(basis,v):
    while v:
        p=v.bit_length()-1
        if p not in basis:return False
        v^=basis[p]
    return True
def nullspace(rows,width):
    basis={}
    for r in rows:insert(basis,r)
    for p in sorted(basis):
        for q in sorted(basis):
            if q>p and basis[q]>>p&1:basis[q]^=basis[p]
    result=[]
    for j in range(width):
        if j in basis:continue
        v=1<<j
        for p,row in basis.items():
            if row>>j&1:v|=1<<p
        assert all((row&v).bit_count()%2==0 for row in rows);result.append(v)
    assert len(result)==width-len(basis);return result
def face_boundary(face):return sum(1<<EI[''.join(e)] for e in it.combinations(face,2) if 'A' not in e)
def spaces(faces):
    pivot={};cycles=[];boundaries=[]
    for face in sorted(faces):
        v=1<<FI[face];column=face_boundary(face);boundaries.append(column)
        while column:
            p=column.bit_length()-1
            if p not in pivot:pivot[p]=(column,v);break
            c,w=pivot[p];column^=c;v^=w
        if not column:cycles.append(v)
    return {0:[1],1:nullspace(boundaries,15),2:cycles}
def barcode(node_spaces,width):
    annihilators=[nullspace(s,width) for s in node_spaces];g={};length=len(node_spaces)
    for a in range(length):
        joined={}
        for b in range(a,length):
            for v in annihilators[b]:insert(joined,v)
            g[a,b]=width-len(joined)
    bars=[]
    for a in range(length):
        for b in range(a,length):
            n=g[a,b]-g.get((a-1,b),0)-g.get((a,b+1),0)+g.get((a-1,b+1),0);assert n>=0
            if n:bars.append({'start':a,'end':b,'multiplicity':n})
    for i,s in enumerate(node_spaces):assert sum(b['multiplicity'] for b in bars if b['start']<=i<=b['end'])==len(s)
    return bars
def restriction(bars):
    result=Counter()
    for b in bars:
        supported=[i//2 for i in range(0,7,2) if b['start']<=i<=b['end']]
        if supported:result[min(supported),max(supported)]+=b['multiplicity']
    return result
def analyze(report):
    tags=('source','three','exchange1','exchange2');scores=[report['endpoints'][t]['scores'] for t in tags];theta=.8
    if not all(v>=theta for s in scores for c,v in s.items() if len(c)<=2):return {'status':'COMMON_COMPLETE_GRAPH_PRECONDITION_NOT_MET','case':report['case'],'science_pass':report['joint_confirmation_case_pass']}
    checkpoints=[{f for f in FACES if s[f]>=theta} for s in scores];out={'status':'PASS_EXACT_MEASURED_ZIGZAG_BARCODES','case':report['case'],'world':report['world'],'order':report['order'],'science_pass':report['joint_confirmation_case_pass'],'n':report['n'],'threshold':theta,'true_window_diagram_certified':report['robust_window_certificates']['all_cell_statuses_determined_by_simultaneous_bands_throughout_window'],'checkpoint_labels':['Source','Learned',*report['order']],'checkpoint_face_counts':[len(f) for f in checkpoints]};named={}
    for bridge in ('union','intersection'):
        nodes=[]
        for i,c in enumerate(checkpoints):
            nodes.append(c)
            if i<3:nodes.append(c|checkpoints[i+1] if bridge=='union' else c&checkpoints[i+1])
        node_spaces=[spaces(c) for c in nodes];bars={str(k):barcode([s[k] for s in node_spaces],width) for k,width in ((0,1),(1,15),(2,35))};out[bridge]={'bars':bars,'node_betti':[[len(s[k]) for k in (0,1,2)] for s in node_spaces]}
        for quad in ('ABCD','ABDE','ACEF'):
            q=sum(1<<FI[''.join(f)] for f in it.combinations(quad,3));profile=[]
            for s in node_spaces:
                basis={}
                for v in s[2]:insert(basis,v)
                profile.append(member(basis,q))
            named.setdefault(quad,{})[bridge+'_node_support']=profile
    for k in ('0','1','2'):assert restriction(out['union']['bars'][k])==restriction(out['intersection']['bars'][k])
    for quad,z in named.items():z['strict_successes']=[sum(report['endpoints'][t]['flags'][quad]) for t in tags];z['checkpoint_support']=z['union_node_support'][::2]
    out['named_boundaries']=named;out['checkpoint_restricted_barcodes_equal']=True;out['bridge_only_degree2_union']=sum(b['multiplicity'] for b in out['union']['bars']['2'] if b['start']==b['end']==5);out['bridge_only_degree1_intersection']=sum(b['multiplicity'] for b in out['intersection']['bars']['1'] if b['start']==b['end']==5)
    return out
def plot(case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.5),gridspec_kw={'height_ratios':[1.25,1.]});colors={(1,3):'#2e6eaa',(1,2):'#338765',(1,1):'#6f7780',(3,3):'#c17e25',(0,0):'#6f7780'}
    for ax,(bridge,degree,title) in zip(axes.flat[:3],(('union','2',r'(a) Union bridges, $H_2$'),('intersection','2',r'(b) Intersection bridges, $H_2$'),('intersection','1',r'(c) Intersection bridges, $H_1$'))):
        bars=case[bridge]['bars'][degree]
        def key(b):
            support=[i//2 for i in (0,2,4,6) if b['start']<=i<=b['end']];return (-len(support),support[0] if support else 9,b['start'],b['end'])
        bars=sorted(bars,key=key)
        for x in (1,3,5):ax.axvspan(x-.11,x+.11,facecolor='#ececf0',edgecolor='none',linewidth=0,zorder=0)
        for row,b in enumerate(bars):
            support=[i//2 for i in (0,2,4,6) if b['start']<=i<=b['end']];color=colors.get((min(support),max(support)) if support else (), '#754296');a,z=b['start'],b['end'];ax.plot([a,z],[row,row],color=color,lw=3,solid_capstyle='round');ax.plot([a,z],[row,row],'.',color=color,ms=6,markeredgewidth=0);label=r'$\times'+str(b['multiplicity'])+'$'
            if not support and a==5:label+=(r' $q$' if degree=='2' else r' $\delta q$')
            ax.text(z+.14,row,label,ha='left',va='center',fontsize=9)
        ax.set_xlim(-.25,7.05);ax.set_ylim(max(len(bars)-.35,.8),-.65);ax.set_xticks((0,2,4,6),case['checkpoint_labels'],fontsize=9);ax.set_yticks([]);ax.set_title(title,loc='left',fontsize=10,pad=8);ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#a6a6a6');ax.spines['bottom'].set_linewidth(1.5);ax.tick_params(axis='x',length=0)
    ax=axes[1,1];rows=[case['named_boundaries'][q] for q in ('ABCD','ABDE','ACEF')];matrix=np.array([r['checkpoint_support'] for r in rows],dtype=int)
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='#a8c8e1' if value else '#f1f1f1',edgecolor='none'))
    ax.set_xlim(-.5,3.5);ax.set_ylim(2.5,-.5)
    for i,r in enumerate(rows):
        for j,number in enumerate(r['strict_successes']):ax.text(j,i,f'{number}/{case["n"]}',ha='center',va='center',fontsize=8.5)
    ax.set_xticks(range(4),case['checkpoint_labels'],fontsize=9);ax.set_yticks(range(3),('ABCD','ABDE','ACEF'),fontsize=9);ax.set_title('(d) Labelled support and execution',loc='left',fontsize=10,pad=8);ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    ax.text(.5,-.29,'Blue: supported boundary; entries: strict paired successes',transform=ax.transAxes,ha='center',fontsize=8)
    fig.subplots_adjust(left=.055,right=.98,top=.94,bottom=.12,hspace=.55,wspace=.28);dest=BASE.parent/'SIMODS20261022/figures'
    for ext in ('pdf','svg','png'):fig.savefig(dest/f'fig_labelled_zigzag.{ext}',dpi=220,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
def main():
    results=[]
    for folder in ('confirmation_factored_5300','confirmation_equal_dose_5400'):
        reg=json.loads((BASE/folder/'REGISTRATION.json').read_text())
        for cfg in reg['cases']:
            root=BASE/cfg['directory']
            if not(root/'FINAL_RESULT.json').exists():continue
            f=json.loads((root/'FINAL_RESULT.json').read_text());report=json.loads((root/f['report']).read_text());value=analyze(report);value['assessment']=folder;results.append(value)
    eligible=[r for r in results if r['status']=='PASS_EXACT_MEASURED_ZIGZAG_BARCODES'];preferred=[r for r in eligible if r['assessment']=='confirmation_equal_dose_5400'];representative=(preferred or eligible)[0] if eligible else None
    if representative:plot(representative)
    out={'scope':'Postprocessing of returned measured diagrams. No new training/query, independent replicate, statistical gate or novelty claim. Failed/common-graph-incompatible results remain listed.','cases':results,'plotted_case':None if representative is None else representative['case']};(HERE/'MEASURED_ZIGZAG_BARCODES.json').write_text(json.dumps(out,indent=2));print(json.dumps({'measured_cases':len(results),'plotted_case':out['plotted_case'],'restricted_equality_cases':sum(r.get('checkpoint_restricted_barcodes_equal',False) for r in results)}))
if __name__=='__main__':main()
