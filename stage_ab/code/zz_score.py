"""Pure-Python scoring, grouping and pre-registered gates for Stage A/B and the
cross-lingual worlds. No torch; runs identically on the GPU box and on the local
CPU audit so that every prediction can be re-scored independently.

Row schema kinds: code letters ('F','D','P','R') with `field_length` codes of the
form <letter><3 digits>; 'digits' (3-digit numbers); 'binary' (0/1); 'step'
(0/1/2); 'letter' (A-D multiple choice); 'text' (normalised exact match);
'exact' (case-sensitive match after stripping outer whitespace).
"""
from __future__ import annotations
import collections, gzip, itertools, json, re, unicodedata
from pathlib import Path

ATOMIC_EDGES = ('eFD', 'eDP', 'eFP', 'eFR', 'eRP')
GROUP_ATTRS = ('task', 'template', 'condition', 'domain', 'lang')


def _norm_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s).lower()
    s = ''.join(ch for ch in s if not unicodedata.category(ch).startswith('P'))
    return re.sub(r'\s+', '', s)


def parse(prediction: str):
    """Split a prediction into pipe-separated fields; comma spacing normalised."""
    return [','.join(s.strip() for s in q.strip().split(',')) for q in prediction.strip().split('|')]


def field_valid(part: str, kind: str, field_length: int) -> bool:
    if kind == 'binary':
        return part in ('0', '1')
    if kind == 'step':
        return part in ('0', '1', '2')
    if kind == 'letter':
        return part in ('A', 'B', 'C', 'D')
    if kind == 'text' or kind == 'exact':
        return len(part) > 0
    items = part.split(',')
    if len(items) != field_length:
        return False
    pat = r'\d{3}' if kind == 'digits' else re.escape(kind) + r'\d{3}'
    return all(re.fullmatch(pat, a) is not None for a in items)


def field_equal(pred: str, gold: str, kind: str) -> bool:
    if kind == 'text':
        return _norm_text(pred) == _norm_text(gold)
    if kind == 'exact':                       # case-sensitive, whitespace-stripped (case-transform tasks)
        return pred.strip() == gold.strip()
    return pred == gold


def score(row: dict, prediction: str) -> dict:
    parts = parse(prediction)
    gold = row['gold'].split('|')
    schema = row['schema']
    n = len(schema)
    fl = row.get('field_length', 1)
    valid = len(parts) == n and all(field_valid(p, k, fl) for p, k in zip(parts, schema))
    fc = [int(i < len(parts) and field_equal(parts[i], g, k)) for i, (g, k) in enumerate(zip(gold, schema))]
    correct = int(valid and all(fc))
    return dict(correct=correct, valid_format=int(valid), field_correct=fc,
                missing_fields=max(0, n - len(parts)), extra_fields=max(0, len(parts) - n))


def group_keys(row: dict):
    """All group keys a row contributes to: vocab:kind[:attr value]* for up to two attrs."""
    base = f"{row.get('vocabulary', '?')}:{row.get('kind', '?')}"
    attrs = [(a, row[a]) for a in GROUP_ATTRS if a in row and row[a] is not None]
    keys = [base + ':all']
    for r in (1, 2):
        for combo in itertools.combinations(attrs, r):
            keys.append(base + ''.join(f':{a}{v}' for a, v in combo))
    return keys


def summarize(records) -> dict:
    """records: iterable of rows that already carry score() fields."""
    g = collections.defaultdict(lambda: dict(n=0, correct=0, valid=0, missing=0, extra=0, fields=None))
    for r in records:
        for k in group_keys(r):
            q = g[k]
            q['n'] += 1
            q['correct'] += r['correct']
            q['valid'] += r['valid_format']
            q['missing'] += int(r['missing_fields'] > 0)
            q['extra'] += int(r['extra_fields'] > 0)
            if q['fields'] is None:
                q['fields'] = [0] * len(r['field_correct'])
            q['fields'] = [a + b for a, b in zip(q['fields'], r['field_correct'])]
    out = {}
    for k, q in g.items():
        n = q['n']
        out[k] = dict(n=n, accuracy=q['correct'] / n, valid=q['valid'] / n,
                      missing_rate=q['missing'] / n, extra_rate=q['extra'] / n,
                      fields=[x / n for x in (q['fields'] or [])])
    return out


def score_file(rows_path):
    """Re-score an eval_*.jsonl written by the runner (rows carry 'prediction')."""
    p = Path(rows_path)
    text = gzip.open(p, 'rt', encoding='utf8').read() if p.suffix == '.gz' else p.read_text(encoding='utf8')
    rows = [json.loads(s) for s in text.splitlines() if s.strip()]
    recs = [dict(r, **score(r, r['prediction'])) for r in rows]
    return recs, summarize(recs)


# ----------------------------------------------------------------------------
# Pre-registered gates (STAGE_AB_RUN_PLAN). Thresholds are inputs, not tuned.
# ----------------------------------------------------------------------------
def _acc(summary, key, default=0.0):
    return summary.get(key, {}).get('accuracy', default)


def _valid(summary, key, default=0.0):
    return summary.get(key, {}).get('valid', default)


def atomic_gate(summary, vocab='target', templates=(0, 1, 3), threshold=0.9):
    vals = {}
    for t in templates:
        for e in ATOMIC_EDGES:
            vals[f'{vocab}:template{t}:{e}'] = _acc(summary, f'{vocab}:base:task{e}:template{t}')
    return dict(passed=min(vals.values()) >= threshold, threshold=threshold, min=min(vals.values()), scores=vals)


def stage_a_gate(summary, atomic_summary, templates=(0, 1, 2), parse_min=0.95, strict_min=0.9, missing_extra_max=0.05):
    """Format calibration: per-template parse rate, overall strict copy accuracy,
    missing/extra field rate; atomic edges taken from the COMMON stage summary."""
    parse = {f'template{t}': _valid(summary, f'aux:format:template{t}') for t in templates}
    strict = _acc(summary, 'aux:format:all')
    q = summary.get('aux:format:all', {})
    me = q.get('missing_rate', 1.0) + q.get('extra_rate', 1.0)
    atomic = atomic_gate(atomic_summary, 'target', (0, 1, 3), strict_min)
    checks = dict(parse_each_template=min(parse.values()) >= parse_min, strict_copy=strict >= strict_min,
                  missing_or_extra=me <= missing_extra_max, atomic_edge_strict=atomic['passed'])
    return dict(passed=all(checks.values()), checks=checks, parse=parse, strict=strict,
                missing_or_extra=me, atomic=atomic,
                thresholds=dict(parse_min=parse_min, strict_min=strict_min, missing_extra_max=missing_extra_max))


def stage_b_gate(summary, templates=(0, 1, 3), edge_min=0.9, joint_min=0.8, repair_min=0.8, parse_min=0.95):
    """Development gate, judged separately per template, per normal(DEPENDENT)/
    intervention(INDEPENDENT) joint condition and per repair domain A/B."""
    atomic = atomic_gate(summary, 'target', templates, edge_min)
    joint = {f'template{t}:{c}': _acc(summary, f'target:joint:template{t}:condition{c}')
             for t in templates for c in ('DEPENDENT', 'INDEPENDENT')}
    repair = {f'template{t}:{d}': _acc(summary, f'target:repair:template{t}:domain{d}')
              for t in templates for d in ('A', 'B')}
    parse = dict(joint=_valid(summary, 'target:joint:all'), repair=_valid(summary, 'target:repair:all'),
                 format=_valid(summary, 'aux:format:all'))
    checks = dict(atomic=atomic['passed'], joint=min(joint.values()) >= joint_min,
                  repair=min(repair.values()) >= repair_min, parse=min(parse.values()) >= parse_min)
    return dict(passed=all(checks.values()), checks=checks, atomic=atomic, joint=joint, repair=repair, parse=parse,
                thresholds=dict(edge_min=edge_min, joint_min=joint_min, repair_min=repair_min, parse_min=parse_min))


if __name__ == '__main__':
    import sys
    recs, summ = score_file(sys.argv[1])
    print(json.dumps({k: v for k, v in summ.items() if k.endswith(':all')}, ensure_ascii=False, indent=1))
