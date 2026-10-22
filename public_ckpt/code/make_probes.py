"""Generate ``public_ckpt/data/PROBES.jsonl`` from the atomic-capability
complex defined in ``oracle.py``.

For every edge / vertex / face / transfer task in the complex this builds
few-shot completion prompts (no chat template -- these are base models) and
computes the gold answer with the pure-Python oracle. Every row is written
in the exact schema expected by ``stage_ab/code/zz_score.py::score`` (id,
task, kind, vocabulary, template, prompt, gold, schema, field_length), and
is re-scored against its own gold before being written out, so a broken
oracle or template can never silently produce an unscoreable row.

Row counts (see spec/ATOMIC_TASKS.md for the full complex):

  * edge        : 23 edges   x 32 items x 3 templates = 2208
  * vertex      : 17 vertices x 16 items x 2 templates =  544
  * composition :  9 faces   x  2 paths x 32 items x 2 templates = 1152
  * transfer    :  3 tasks   x 32 items x 1 template   =   96
  * total                                              = 4000
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_CODE_DIR = Path(__file__).resolve().parent
_ZZ_ROOT = _CODE_DIR.parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
_STAGE_AB_CODE = _ZZ_ROOT / 'stage_ab' / 'code'
if str(_STAGE_AB_CODE) not in sys.path:
    sys.path.insert(0, str(_STAGE_AB_CODE))

import oracle  # noqa: E402
import zz_score  # noqa: E402

DATA_DIR = _CODE_DIR.parents[0] / 'data'
PROBES_PATH = DATA_DIR / 'PROBES.jsonl'
MANIFEST_PATH = DATA_DIR / 'PROBES_MANIFEST.json'

N_DEMO = 3

EDGE_ITEMS, EDGE_TEMPLATES = 32, 3
VERTEX_ITEMS, VERTEX_TEMPLATES = 16, 2
FACE_ITEMS, FACE_TEMPLATES = 32, 2
TRANSFER_ITEMS, TRANSFER_TEMPLATES = 32, 1


# ---------------------------------------------------------------------------
# Few-shot templates -- plain completion style, no chat markup.
# ---------------------------------------------------------------------------

def template_qa(header: str, demos: Sequence[Tuple[str, str]], test_input: str) -> str:
    """``Q: ... / A: ...`` style, blank line between examples."""
    lines = [header, '']
    for din, dout in demos:
        lines.append(f'Q: {din}')
        lines.append(f'A: {dout}')
        lines.append('')
    lines.append(f'Q: {test_input}')
    lines.append('A:')
    return '\n'.join(lines)


def template_arrow(header: str, demos: Sequence[Tuple[str, str]], test_input: str) -> str:
    """``input -> output`` style, one example per line."""
    lines = [header]
    for din, dout in demos:
        lines.append(f'{din} -> {dout}')
    lines.append(f'{test_input} ->')
    return '\n'.join(lines)


def template_iolabel(header: str, demos: Sequence[Tuple[str, str]], test_input: str) -> str:
    """``input: ... | output: ...`` style, one example per line."""
    lines = [header]
    for din, dout in demos:
        lines.append(f'input: {din} | output: {dout}')
    lines.append(f'input: {test_input} | output:')
    return '\n'.join(lines)


TEMPLATES: List[Callable[[str, Sequence[Tuple[str, str]], str], str]] = [
    template_qa, template_arrow, template_iolabel,
]


def pick(pool: Sequence, k: int, key: str) -> List:
    """Deterministic (sha256-seeded) sample of `k` distinct items from `pool`."""
    rng = random.Random(oracle._seed_for(key))
    return rng.sample(list(pool), k)


def make_row(row_id: str, task: str, kind: str, domain: str, template_idx: int, prompt: str,
             gold: str, item, face: Optional[str] = None, path_index: Optional[int] = None) -> dict:
    """Build one probe row in the exact schema ``zz_score.score`` expects,
    plus bookkeeping fields (`domain`, `item`, `face`, `path_index`) used
    only by our own analysis scripts."""
    return dict(id=row_id, task=task, kind=kind, vocabulary='public', template=template_idx,
                prompt=prompt, gold=gold, schema=[oracle.schema_for_domain(domain)], field_length=1,
                domain=domain, item=item, face=face, path_index=path_index)


def gen_task_rows(kind: str, name: str, n_items: int, n_templates: int, rows_out: List[dict],
                   face: Optional[str] = None, path_index: Optional[int] = None,
                   shared_items: Optional[List] = None) -> List:
    """Generate `n_items` x `n_templates` rows for one edge/vertex/composition/
    transfer task, appending them to `rows_out`. Returns the item list used
    (so callers can reuse it, e.g. to share items across a face's two paths).
    """
    domain = oracle.domain_of(kind, name)
    header = oracle.header_of(kind, name)
    items = shared_items if shared_items is not None else pick(oracle.TEST_ITEMS[domain], n_items,
                                                                 f'{kind}:{name}:items')
    for t in range(n_templates):
        demo_items = pick(oracle.DEMO_ITEMS[domain], N_DEMO, f'{kind}:{name}:tpl{t}:demo')
        demo_pairs = [oracle.compute(kind, name, d) for d in demo_items]
        for idx, item in enumerate(items):
            test_input, gold = oracle.compute(kind, name, item)
            prompt = TEMPLATES[t](header, demo_pairs, test_input)
            row_id = f'{kind}__{name}__tpl{t}__it{idx}'
            if face is not None:
                row_id = f'{face}__{row_id}'
            rows_out.append(make_row(row_id, name, kind, domain, t, prompt, gold, item, face, path_index))
    return items


def build_all_rows() -> List[dict]:
    """Generate every probe row for the full complex + transfer tasks."""
    rows: List[dict] = []

    for name in sorted(oracle.EDGES):
        gen_task_rows('edge', name, EDGE_ITEMS, EDGE_TEMPLATES, rows)

    for name in sorted(oracle.VERTEX_DOMAIN):
        gen_task_rows('vertex', name, VERTEX_ITEMS, VERTEX_TEMPLATES, rows)

    for face_name in sorted(oracle.FACES):
        face = oracle.FACES[face_name]
        task_a, task_b = face['composition_tasks']
        domain = oracle.COMPOSITION_TASKS[task_a]['domain']
        shared_items = pick(oracle.TEST_ITEMS[domain], FACE_ITEMS, f'face:{face_name}:items')
        gen_task_rows('composition', task_a, FACE_ITEMS, FACE_TEMPLATES, rows,
                      face=face_name, path_index=0, shared_items=shared_items)
        gen_task_rows('composition', task_b, FACE_ITEMS, FACE_TEMPLATES, rows,
                      face=face_name, path_index=1, shared_items=shared_items)

    for name in sorted(oracle.TRANSFER_TASKS):
        gen_task_rows('transfer', name, TRANSFER_ITEMS, TRANSFER_TEMPLATES, rows)

    return rows


def self_check(rows: Sequence[dict]) -> None:
    """Assert every row scores ``correct == 1`` against its own gold, i.e.
    the row is well-formed and scoreable by ``zz_score.score`` unmodified."""
    ids_seen = set()
    for row in rows:
        if row['id'] in ids_seen:
            raise AssertionError(f'duplicate row id: {row["id"]!r}')
        ids_seen.add(row['id'])
        result = zz_score.score(row, row['gold'])
        if result['correct'] != 1:
            raise AssertionError(f'row {row["id"]!r} does not self-score correct: {result}')


def self_check_case_sensitivity(rows: Sequence[dict]) -> int:
    """Assert that, for every string-domain edge whose oracle function is a
    *pure* case transform (``oracle.PURE_CASE_EDGES``: the edges that apply
    ``upper``/``alternate``), predicting the **input copied verbatim**
    (i.e. no case change at all) scores ``correct == 0``.

    This is the regression check for the bug this schema switch fixes: under
    the old ``schema=['text']`` (case-insensitive) a verbatim copy of the
    input used to score ``correct == 1`` on these edges, making 'did the
    model actually change the case' indistinguishable from 'did it just copy
    the word'. Under the new domain-appropriate schema (``'exact'`` for the
    string domain, see ``oracle.schema_for_domain``) a verbatim copy must now
    score 0. Returns the number of rows checked (raises if none were, since
    that would silently make this check vacuous).
    """
    n_checked = 0
    for row in rows:
        if row['kind'] != 'edge' or row['task'] not in oracle.PURE_CASE_EDGES:
            continue
        test_input, gold = oracle.compute('edge', row['task'], row['item'])
        if gold != row['gold']:
            raise AssertionError(f'row {row["id"]!r} gold mismatch with oracle.compute: {gold!r} != {row["gold"]!r}')
        result = zz_score.score(row, test_input)  # prediction = verbatim copy of the shown input
        if result['correct'] != 0:
            raise AssertionError(
                f'row {row["id"]!r} (task={row["task"]!r}) should score correct=0 when the '
                f'prediction is just a verbatim copy of the input (case unchanged), got {result}')
        n_checked += 1
    if n_checked == 0:
        raise AssertionError('self_check_case_sensitivity checked 0 rows -- oracle.PURE_CASE_EDGES empty or unmatched?')
    return n_checked


def write_probes(rows: Sequence[dict]) -> str:
    """Write rows as JSON lines to PROBES_PATH; return its sha256 hex digest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROBES_PATH, 'w', encoding='utf8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return hashlib.sha256(PROBES_PATH.read_bytes()).hexdigest()


def build_manifest(rows: Sequence[dict], sha256: str) -> dict:
    """Per-kind / per-domain / per-task / per-face row counts, for a quick
    sanity read without re-parsing the whole jsonl file."""
    def count_by(field):
        c: Dict[str, int] = {}
        for r in rows:
            v = r.get(field)
            if v is None:
                continue
            c[v] = c.get(v, 0) + 1
        return dict(sorted(c.items()))

    return dict(
        n_rows=len(rows),
        sha256=sha256,
        by_kind=count_by('kind'),
        by_domain=count_by('domain'),
        by_task=count_by('task'),
        by_face=count_by('face'),
        n_vertices=len(oracle.VERTEX_DOMAIN),
        n_edges=len(oracle.EDGES),
        n_faces=len(oracle.FACES),
        n_transfer_tasks=len(oracle.TRANSFER_TASKS),
    )


def main() -> None:
    rows = build_all_rows()
    self_check(rows)
    n_case_checked = self_check_case_sensitivity(rows)
    sha256 = write_probes(rows)
    manifest = build_manifest(rows, sha256)
    manifest['case_sensitivity_check_rows'] = n_case_checked
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'wrote {PROBES_PATH} ({manifest["n_rows"]} rows, sha256={sha256[:12]}...)')
    print(f'wrote {MANIFEST_PATH}')
    print('by_kind:', manifest['by_kind'])
    print('by_domain:', manifest['by_domain'])
    print(f'case-sensitivity self-check: {n_case_checked} pure-case-transform rows all correctly '
          f'score 0 for a verbatim-copy prediction (edges: {oracle.PURE_CASE_EDGES})')


if __name__ == '__main__':
    main()
