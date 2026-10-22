"""Route-consistency face score for the 3 faces -- as of the GPU-informed
revision below, this is the PRIMARY face readout (COMPLEX.json's
face_readout="routes"), not a secondary cross-check.

History: the equality-comparison joint task (task=face_name, kind='joint',
override True/False) was originally the primary readout, because
design_from_spec's built-in dim==2 branch (min(acc(override=False),
acc(override=True))) scores it with no analyzer change needed. Real GPU runs
(701/702, --preset classic, 100k steps) showed why that is the wrong primary
signal: f_sub/f_comm/f_div crossed 0.9 held-out accuracy at 500-2750 steps,
BEFORE any of the 7 arithmetic edges (2500-43000 steps) -- because
"a-b ?= a+n" is true iff n=-b, answerable from negation alone (learned in
200-400 steps) without the model ever needing subtraction. The equality task
is still trained/probed unchanged and is still scored by
design_from_spec's dim==2 branch, but that score is now a SECONDARY,
descriptive number only -- see README "等式捷径" and
PREREGISTRATION_GROKKING_DRAFT.md for the evidence and the confirmatory
criteria this feeds into.

This module computes the PRIMARY signal instead: "do all the edges on a
face's route land on the correct value for the same item" (e.g. f_sub: are
P_D(a,b), P_Pn(a,b) AND Pn_D(P_Pn(a,b)) all correct for the same original
(a,b))? Because every edge already carries item=[a,b] (the *original* pair,
even for a route probe whose numeric input is transformed), this needs no
joint task at all -- just matching rows by (task, item). It is merged into
each eval's summary.json under `modp:face:task<name>`; a confirmatory run
must read THIS key for face scores, not design_from_spec's own (secondary)
dim==2 computation, which zz_analyze.py has not been changed to skip.
"""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modp_common as M   # noqa: E402


def route_consistency(rows, route_tasks, vocab='modp'):
    """Fraction of items where every task in `route_tasks` is correct (all
    rows for that (task, item) pair correct, across whatever templates are
    present) on the same original item. Returns (fraction, n_items) or
    (None, 0) if no item is covered by every task in the route."""
    by_task_item = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get('vocabulary') != vocab or r.get('task') not in route_tasks:
            continue
        item = tuple(r.get('item', r.get('root', [])))
        by_task_item[r['task']][item].append(r['correct'])
    common = None
    for t in route_tasks:
        items = set(by_task_item[t].keys())
        common = items if common is None else (common & items)
        if not common:
            return None, 0
    if not common:
        return None, 0
    n_ok = 0
    for item in common:
        if all(all(v == 1 for v in by_task_item[t][item]) for t in route_tasks):
            n_ok += 1
    return n_ok / len(common), len(common)


def merge_face_cross_check(summary, scored_rows, vocab='modp'):
    """Add `<vocab>:face:task<face_name>` entries to an existing zz_score.summarize()
    dict (same {n, accuracy, ...}-shaped value so downstream code doesn't need
    to special-case it) and return it."""
    out = dict(summary)
    for f in M.FACES:
        frac, n = route_consistency(scored_rows, f['cycle'], vocab=vocab)
        out[f'{vocab}:face:task{f["name"]}'] = dict(
            n=n, accuracy=frac if frac is not None else 0.0, valid=1.0 if n else 0.0,
            missing_rate=0.0, extra_rate=0.0, fields=[],
            route=list(f['cycle']),
            note='PRIMARY face score (route consistency: all edges on the route correct for the same '
                 'original item); the equality-comparison joint task design_from_spec scores under '
                 'this same face name (dim==2 branch) is now a secondary, descriptive number only -- '
                 'see README "等式捷径".')
    return out


def main():
    """python face_scores.py eval_XXXXX.jsonl [eval_XXXXX.summary.json]
    Recomputes the route cross-check for an existing eval file and prints it
    (or merges into the summary file in place if given)."""
    rows_path = Path(sys.argv[1])
    rows = [json.loads(s) for s in rows_path.read_text(encoding='utf8').splitlines() if s.strip()]
    result = {f['name']: route_consistency(rows, f['cycle']) for f in M.FACES}
    if len(sys.argv) > 2:
        summary_path = Path(sys.argv[2])
        summary = json.loads(summary_path.read_text(encoding='utf8'))
        summary = merge_face_cross_check(summary, rows)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf8')
        print(f'merged into {summary_path}')
    print(json.dumps({k: v[0] for k, v in result.items()}, indent=1))


if __name__ == '__main__':
    main()
