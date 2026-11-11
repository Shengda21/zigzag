"""Exploratory frozen CPU audit: theta=.8, delta=.05, all six FORGET3 arms.
No outcome prediction, model fitting, GPU, bootstrap, or confirmation claim.
Independent check uses chain-space kernels/quotients, never CW induced maps.
"""
import json, sys, time
from pathlib import Path
from functools import lru_cache
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if (HERE.parents[1] / 'complex_worlds').exists() else HERE.parents[1] / 'capability-zigzag-repro'
sys.path[:0] = [str(REPO/'crosslingual/code'), str(REPO/'paper_sec4')]
import cw_zigzag as CW
import zz_analyze as ZA
import zz_robust as R
from theta_sweep import kappa
from mv_counts import spurious_history, connecting_rank

def rank(vectors):
    """Independent dense uint8 Gaussian elimination over F2."""
    vv = list(vectors)
    if not vv: return 0
    n = max(vv).bit_length()
    a = np.array([[(v >> j) & 1 for j in range(n)] for v in vv], dtype=np.uint8)
    r = 0
    for j in range(n):
        p = np.flatnonzero(a[r:,j])
        if not len(p): continue
        p = r + p[0]; a[[r,p]] = a[[p,r]]
        for q in range(r+1,len(a)):
            if a[q,j]: a[q] ^= a[r]
        r += 1
        if r == len(a): break
    return r

def independent(spec):
    cells = spec['cells']; index = {c[0]:i for i,c in enumerate(cells)}
    boundaries = [sum(1 << index[n] for n in c[2]) for c in cells]
    @lru_cache(None)
    def chains(mask):
        piv = {}; cycles = []
        for i,c in enumerate(cells):
            if c[1] != 1 or not mask >> i & 1: continue
            v, chain = boundaries[i], 1 << i
            while v:
                j = v.bit_length()-1
                if j not in piv: piv[j] = (v,chain); break
                p,z = piv[j]; v ^= p; chain ^= z
            if not v: cycles.append(chain)
        bound = [boundaries[i] for i,c in enumerate(cells) if c[1] == 2 and mask >> i & 1]
        return cycles,bound
    def kap(high,low,sources):
        z,_ = chains(high); _,b = chains(low)
        j = [v for s in sources for v in chains(s)[0]]
        return rank(b+j+z)-rank(b+j)
    def alpha(prefix):
        u = 0
        for m in prefix: u |= m
        z,b = chains(u)
        support = [v for m in prefix for v in chains(m)[0]]
        return len(z)-rank(b+support)
    return kap,alpha

def main():
    started = time.time()
    spec = json.loads((REPO/'complex_worlds/data/RCN12_FORGET3/world635/COMPLEX.json').read_text(encoding='utf-8'))
    runs = REPO/'cloud_return_round4/complex_worlds/runs/RCN12_FORGET3'
    names = [f'world{w}_S{s}' for w in (635,636,637) for s in (1,2)]
    out = {'status':'exploratory; not untouched confirmation', 'theta':.8,'delta':.05,
           'scope':'full complex; every original checkpoint and adjacent bridge',
           'independent_units':3,'arms':[], 'failures':[],
           'source_sets':{'left':'{2i}','right':'{2i+2}','both':'{2i,2i+2}'},
           'checks':'κ: induced-map quotient vs independent chain-space quotient; α: historical induced support vs independent chain cycles; rho: H0 MV kernel vs H1 quotient. Delta monotonicity at 0,.05; no b/d monotonicity asserted.'}
    def check(ok,label,context):
        if not ok: out['failures'].append({'check':label,**context})
    for name in names:
        ev = ZA.load_run(runs/name); sm = R.ScoreModel(spec,ev)
        cx = CW.CWComplex([(n,d,list(b)) for n,d,b in spec['cells']])
        scores = sm.scores(sm.weights(0,0))
        arr = np.stack([scores[n][0] for n in cx.names])
        mk = R.Masker(cx)
        masks = {th:mk.masks(arr,th) for th in (.75,.8,.85)}
        center = masks[.8]; hist = spurious_history(cx,center,1)
        ik,ia = independent(spec)
        rec = {'run':name,'steps':sm.steps,'history':hist,'bridges':[]}
        prev_alpha = 0
        for t,h in enumerate(hist):
            ctx = {'run':name,'step':sm.steps[t]}
            check(h['spurious']==ia(center[:t+1]),'alpha independent equality',ctx)
            # Re-running prefix-only helper verifies no future checkpoint enters alpha.
            check(h==spurious_history(cx,center[:t+1],1)[-1],'alpha prefix equality',ctx)
            rr = h['connecting_rank_prev_new']
            check(rr <= h['spurious'] <= prev_alpha+rr,'historical step bound',ctx)
            prev_alpha = h['spurious']
        for i in range(len(center)-1):
            a,b = center[i:i+2]; rho = connecting_rank(cx,a,b,1)
            row = {'index':i,'steps':sm.steps[i:i+2],'rho1':rho,'kappa_delta0':{},'kappa_delta005':{}}
            for delta,lo,hi in ((0,.8,.8),(.05,.75,.85)):
                la,lb = masks[lo][i:i+2]; ha,hb = masks[hi][i:i+2]
                for key,sources in [('left',[la]),('right',[lb]),('both',[la,lb])]:
                    value = kappa(cx,ha|hb,la|lb,sources,1)
                    row['kappa_delta0' if delta==0 else 'kappa_delta005'][key] = value
                    check(value==ik(ha|hb,la|lb,sources),'kappa independent equality',{'run':name,'bridge':i,'delta':delta,'sources':key})
            check(row['kappa_delta0']['both']==rho,'kappa0 equals rho',{'run':name,'bridge':i})
            for key in ('left','right','both'):
                check(0<=row['kappa_delta005'][key]<=row['kappa_delta0'][key],'kappa window monotonicity',{'run':name,'bridge':i,'sources':key})
            row['birth1_delta005'] = row['kappa_delta005']['left']-row['kappa_delta005']['both']
            row['death1_delta005'] = row['kappa_delta005']['right']-row['kappa_delta005']['both']
            rec['bridges'].append(row)
        rec['summary'] = {'alpha_final':hist[-1]['spurious'], 'alpha_zero_steps':[sm.steps[i] for i,h in enumerate(hist) if h['spurious']==0],
            'rho_nonzero_bridges':[r['steps'] for r in rec['bridges'] if r['rho1']],
            'kappa005_nonzero':{k:[r['steps'] for r in rec['bridges'] if r['kappa_delta005'][k]] for k in ('left','right','both')},
            'zero_counts':{'rho':sum(r['rho1']==0 for r in rec['bridges']),**{f'kappa005_{k}':sum(r['kappa_delta005'][k]==0 for r in rec['bridges']) for k in ('left','right','both')}}}
        out['arms'].append(rec); print(name,rec['summary'],flush=True)
    out['elapsed_seconds'] = time.time()-started
    (HERE/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines = ['# E5 exploratory CPU audit','', 'Frozen θ=.8, δ=.05. All six original FORGET3 trajectories, full complex, all 49 checkpoints and 48 adjacent bridges per trajectory. Three independent worlds; seeds/checkpoints are repeated observations. No outcome fitting, GPU work or new confirmation evidence.','',f"Failures: {len(out['failures'])}. Runtime: {out['elapsed_seconds']:.1f} seconds.",'',out['checks'],'','Full per-checkpoint α, historical connecting ranks, every adjacent ρ, all three actual κ source-set values at δ=0/.05, zero locations and failures are in results.json. A zero is a substantive observed value, not omitted or replaced.','']
    for a in out['arms']: lines += [f"## {a['run']}",'',json.dumps(a['summary'],ensure_ascii=False),'']
    lines += ['## Limits','', 'This checks computation and finite trajectory illustrations. It does not test predictive utility, establish a mechanism, validate sampling coverage, or resolve mathematical originality. The independent chain implementation shares only the complex specification and threshold masks; it does not call CW homology, induced maps or rank. Threshold-mask extraction itself is not independently audited. No perturbation coverage claim is tested. Only κ monotonicity between the two frozen scales is checked; derived birth/death differences need not be monotone.','', 'Run locally with Python and NumPy: python audit.py. Existing repository routines are imported read-only.']
    (HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('failures',len(out['failures']),flush=True)

if __name__=='__main__': main()
