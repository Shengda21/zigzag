"""Independent numerical verification of returned binomial bounds and contrasts."""
import argparse
import functools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PHASES = ('source', 'two', 'three', 'exchange1', 'clean1', 'exchange2', 'clean2')
HIGH = ('ABCD', 'ABDE', 'ACEF', 'ABCDE', 'ACDEG', 'BCDEF')

@functools.cache
def tail(n, k, p, upper):
    terms = range(k+1) if upper else range(k,n+1)
    return math.fsum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in terms)

def verify(report, word):
    n = report['n']
    score_error, score_total = (.025, 252) if word else (.0125, 2898)
    contrast_error = .025 if word else .0125
    allocated = score_error / (2*score_total)
    by_phase = {}
    error = 0.
    total = 0
    for name in PHASES:
        endpoint = report['endpoints'][name]
        if word:
            entries = {cell:(e['flags'],e['score'],e['bounds']) for cell,e in endpoint.items()}
        else:
            entries = {cell:(endpoint['flags'][cell],score,endpoint['bounds'][cell])
                       for cell,score in endpoint['scores'].items()}
        by_phase[name] = {}
        for cell,(flags,point,(lo,hi)) in entries.items():
            assert len(flags)==n and set(flags)<={0,1}
            k = sum(flags)
            assert k/n == point and 0 <= lo <= point <= hi <= 1
            if k == 0:
                assert lo == 0
            else:
                error = max(error, abs(tail(n,k,lo,False)-allocated))
            if k == n:
                assert hi == 1
            else:
                error = max(error, abs(tail(n,k,hi,True)-allocated))
            by_phase[name][cell] = flags
            total += 1
    assert total == score_total//6 and error < 2e-12
    def differences(phase1,task1,phase2,task2):
        return [a-b for a,b in zip(by_phase[phase1][task1],by_phase[phase2][task2])]
    expected = {f'three-minus-two:{c}':differences('three',c,'two',c) for c in HIGH}
    for i,letter in enumerate(report['order'],1):
        high,low = ('ABCD','ABDE') if letter=='A' else ('ABDE','ABCD')
        tag = f'exchange{i}'
        key = f'crossover:{i}' if word else f'{tag}:{high}-minus-{low}'
        expected[key] = differences(tag,high,tag,low)
    assert set(expected) == set(report['paired_contrasts'])
    width = math.sqrt(2*math.log(48/contrast_error)/n)
    max_contrast_error = 0.
    for key,d in expected.items():
        observed=report['paired_contrasts'][key]
        mean=math.fsum(d)/n
        max_contrast_error=max(max_contrast_error,abs(mean-observed['mean']),
                               abs(mean-width-observed['lower']))
    assert max_contrast_error < 1e-14
    return {'case':report['case'], 'look':report.get('look',1), 'n':n,
            'score_bounds_verified':total, 'paired_contrasts_verified':len(expected),
            'max_abs_binomial_tail_probability_error':error,
            'max_abs_contrast_error':max_contrast_error}

def main():
    global BASE
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-root', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    BASE = args.study_root.resolve()
    suites=('confirmation_factored_5300','confirmation_equal_dose_5400','word_family_evaluation_5500')
    audited=[]
    for suite in suites:
        word=suite.startswith('word_family_')
        reg=json.loads((BASE/suite/'REGISTRATION.json').read_text())
        for case in reg['cases']:
            root=BASE/case['directory']
            if not (root/'CASE_RAW_AUDIT.json').exists() or not (root/'ARCHIVE_VERIFICATION.json').exists():
                continue
            reports = [root/'RESULT.json'] if word else sorted(root.glob('CONFIRMATION_LOOK*.json'))
            for path in reports:
                if path.exists():
                    report=json.loads(path.read_text())
                    assert report['case']==case['case'] and report['order']==case['order']
                    audited.append({'suite':suite,**verify(report,word)})
    result={'scope':'Independent finite binomial-tail summation and paired-contrast arithmetic from audited returned flags, no SciPy interval calls or model inference.',
            'case_looks':len(audited),'individual_score_bounds':sum(x['score_bounds_verified'] for x in audited),
            'paired_contrasts':sum(x['paired_contrasts_verified'] for x in audited),'cases':audited}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':
    main()
