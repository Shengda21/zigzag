"""Summarize returned labelled-support evidence; never install a draft result."""
import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SUITES = (
    ('Primary', 'confirmation_factored_5300'),
    ('Fixed dose', 'confirmation_equal_dose_5400'),
    ('New patterns', 'word_family_evaluation_5500'),
)
TASKS = {'ABCD', 'ABDE', 'ACEF', 'ABCDE', 'ACDEG', 'BCDEF'}
PHASES = {'source', 'two', 'three', 'clean1', 'clean2', 'A', 'B'}
RETAINED = {'ACDEG', 'BCDEF'}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def signed_bound(value, upper=False):
    return str(Decimal(str(value)).quantize(
        Decimal('0.001'), rounding=ROUND_CEILING if upper else ROUND_FLOOR
    ))

def predicted_high(phase, task):
    if phase in {'three', 'clean1', 'clean2'}:
        return True
    if phase in {'source', 'two'}:
        return False
    return task in RETAINED or task == {'A': 'ABCD', 'B': 'ABDE'}[phase]

def summarize(folder):
    root = BASE / folder
    reg = {c['case']: c for c in read_json(root/'REGISTRATION.json')['cases']}
    outcomes = {c['case']: c for c in read_json(root/'SUMMARY.json')['cases']}
    assert len(reg) == 6 and set(reg) == set(outcomes)
    is_word = folder == SUITES[-1][1]
    terminal = {}
    for case, rec in outcomes.items():
        archived = rec.get('two_archives_verified') if is_word else rec.get('two_archive_copies_verified')
        if rec.get('raw_audit_pass') and archived:
            assert rec['scientific_case_pass'] in (True, False, None)
            terminal[case] = rec
    tables = defaultdict(dict)
    with (root/'HIGHER_RESULTS.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            case = row['case']
            assert case in outcomes
            if case not in terminal:
                continue
            task = row['cell' if is_word else 'task']
            if is_word:
                phase = row['endpoint']
                if phase.startswith('exchange'):
                    phase = reg[case]['order'][int(phase[-1])-1]
            else:
                phase = row['semantic_phase']
            assert phase in PHASES and task in TASKS and (phase, task) not in tables[case]
            n = int(row['n'])
            successes = int(row['strict_pairs' if is_word else 'correct'])
            lo, hi = float(row['lower']), float(row['upper'])
            assert 0 <= successes <= n and 0 <= lo <= successes/n <= hi <= 1
            assert abs(successes/n-float(row['score' if is_word else 'rate'])) < 1e-12
            pred = row['predicted_positive' if is_word else 'predicted_high']
            assert pred in {'True', 'False'} and (pred == 'True') == predicted_high(phase, task)
            tables[case][phase, task] = (n, lo, hi, successes/n)
    scores = defaultdict(list)
    contrast_lower, contrast_effect, n_values = [], [], []
    for case, rec in terminal.items():
        entries = tables.get(case)
        if not entries:
            assert rec['scientific_case_pass'] is not True
            continue
        assert len(entries) == 42
        assert len({v[0] for v in entries.values()}) == 1
        n_values.append(next(iter(entries.values()))[0])
        case_result = read_json(BASE/reg[case]['directory']/'RESULT.json') if is_word else rec
        contrasts = case_result['paired_contrasts']
        assert len(contrasts) == 8
        for value in contrasts.values():
            contrast_lower.append(value['lower'])
            contrast_effect.append(value['mean'])
        for (phase, task), (_, lo, hi, point) in entries.items():
            if phase == 'two':
                group, bound = 'two', hi
            elif phase == 'three':
                group, bound = 'three', lo
            elif phase in {'A', 'B'}:
                group, bound = ('supported', lo) if predicted_high(phase, task) else ('unsupported', hi)
            else:
                continue
            scores[group].append((bound, point))
    groups = ('two','three','supported','unsupported')
    extremum = dict(zip(groups, (max,min,min,max)))
    bounds = {g: extremum[g](x[0] for x in scores[g]) if scores[g] else None for g in groups}
    points = {g: extremum[g](x[1] for x in scores[g]) if scores[g] else None for g in groups}
    return {
        'suite': folder, 'registered': len(reg), 'terminal_returned': len(terminal),
        'scored_cases': len(n_values),
        'passed': sum(c['scientific_case_pass'] is True for c in terminal.values()),
        'bounds': bounds, 'point_extrema': points,
        'n': sorted(set(n_values)),
        'min_paired_lower': min(contrast_lower, default=None),
        'min_paired_effect': min(contrast_effect, default=None),
    }

def main_table(rows):
    tex = [
        '% Generated from returned complete rosters and raw-audited case scores.',
        r'\begin{table}[!tbp]', r'\centering', r'\small',
        r'\caption{Three assessments of the same six sources. The denominator',
        r'$n$ counts root clusters per case. Pass counts cases meeting every',
        r'registered criterion. UB and LB are upper and lower bounds. Each',
        r'column gives the worst simultaneous conditional bound over all',
        r'scored cases and the relevant tasks or paired contrasts.}',
        r'\label{tab:labelled-confirmed}',
        r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'Assessment & Pass & $n$ & Two UB & Three LB & Support LB & Absent UB & Contrast LB' + chr(92)*2,
        r'\midrule',
    ]
    for (name, _), row in zip(SUITES, rows):
        bounds = [row['bounds'][g] for g in ('two','three','supported','unsupported')] + [row['min_paired_lower']]
        shown = [signed_bound(b, i in (0,3)) if b is not None else '--' for i,b in enumerate(bounds)]
        sizes = '/'.join(map(str, row['n'])) or '--'
        tex.append(' & '.join([name, f'{row["passed"]}/6', sizes, *shown]) + chr(92)*2)
    return '\n'.join(tex + [r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])

def main():
    global BASE
    p = argparse.ArgumentParser()
    p.add_argument('--study-root', type=Path, default=BASE)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    BASE = args.study_root.resolve()
    rows = [summarize(folder) for _,folder in SUITES]
    complete = all(r['terminal_returned'] == r['registered'] == 6 for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'all_terminal_and_archived': complete, 'assessments': rows}
    (args.output/'JOINED_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    if complete:
        (args.output/'JOINED_MAIN_TABLE.tex').write_text(main_table(rows), encoding='utf-8')
        paths = [BASE/folder/('ALL_CASE_RESULTS_TABLE.tex' if label=='New patterns' else 'ALL_CASE_RESULTS_TABLES.tex')
                 for label,folder in SUITES]
        (args.output/'JOINED_SUPPLEMENT_TABLES.tex').write_text(
            '\n'.join(p.read_text(encoding='utf-8') for p in paths), encoding='utf-8'
        )
    print(json.dumps(summary))
if __name__ == '__main__':
    main()
