"""Turn a directory of ``eval_<step>.jsonl`` files (written by
``run_public.py``) into per-checkpoint cell scores for the atomic-capability
complex defined in ``oracle.py`` / ``spec/COMPLEX.json``.

Score definitions (see ``spec/ATOMIC_TASKS.md``):

  * **vertex** / **edge** cell score = accuracy (mean ``correct``) over all
    probe rows for that vertex/edge's task, across every template and item.
  * **face** cell score = the fraction of (item, template) pairs for which
    *both* of the face's two composition paths were answered correctly --
    this is the "two paths agree with the model, and both are right"
    consistency rate the paper's 2-cells are meant to capture.
  * **transfer** task score (not a complex cell) = accuracy over its rows.

Usage::

    python cell_scores.py <runs_dir> [--out_dir <dir>]

Writes ``cell_scores.json`` (``{step_label: {cell_name: score}}``) and
``transfer_scores.json`` (``{step_label: {transfer_task: score}}``) into
`out_dir` (default: `runs_dir` itself).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List


def _label_of(path: Path) -> str:
    """The ``<label>`` in ``eval_<label>.jsonl``."""
    m = re.match(r'eval_(.+)\.jsonl$', path.name)
    if not m:
        raise ValueError(f'unexpected eval filename: {path.name}')
    return m.group(1)


def load_eval_rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def score_vertex_edge(rows: List[dict]) -> Dict[str, float]:
    """``{task_name: accuracy}`` for kind in {'vertex', 'edge'}."""
    n = collections.defaultdict(int)
    c = collections.defaultdict(int)
    for r in rows:
        if r['kind'] in ('vertex', 'edge'):
            n[r['task']] += 1
            c[r['task']] += r['correct']
    return {task: c[task] / n[task] for task in n}


def score_faces(rows: List[dict]) -> Dict[str, float]:
    """``{face_name: consistency_rate}``: fraction of (item, template) pairs
    where both composition paths of the face were answered correctly."""
    # face -> (item, template) -> {path_index: correct}
    groups: Dict[str, Dict[tuple, Dict[int, int]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r['kind'] != 'composition':
            continue
        face = r['face']
        key = (r['item'], r['template'])
        groups[face][key][r['path_index']] = r['correct']

    out: Dict[str, float] = {}
    for face, keyed in groups.items():
        both_correct = 0
        total = 0
        for key, paths in keyed.items():
            if 0 not in paths or 1 not in paths:
                continue  # incomplete pair (e.g. --limit truncated one side); skip rather than mis-score
            total += 1
            both_correct += int(paths[0] == 1 and paths[1] == 1)
        if total:
            out[face] = both_correct / total
    return out


def score_transfer(rows: List[dict]) -> Dict[str, float]:
    """``{transfer_task_name: accuracy}``."""
    n = collections.defaultdict(int)
    c = collections.defaultdict(int)
    for r in rows:
        if r['kind'] == 'transfer':
            n[r['task']] += 1
            c[r['task']] += r['correct']
    return {task: c[task] / n[task] for task in n}


def build_scores(runs_dir: Path):
    """Return ``(cell_scores, transfer_scores)``, each ``{label: {...}}``,
    over every ``eval_*.jsonl`` file found directly in `runs_dir`."""
    cell_scores: Dict[str, Dict[str, float]] = {}
    transfer_scores: Dict[str, Dict[str, float]] = {}
    for path in sorted(runs_dir.glob('eval_*.jsonl')):
        label = _label_of(path)
        rows = load_eval_rows(path)
        cells = {}
        cells.update(score_vertex_edge(rows))
        cells.update(score_faces(rows))
        cell_scores[label] = cells
        transfer_scores[label] = score_transfer(rows)
    return cell_scores, transfer_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs_dir', help='directory containing eval_<step>.jsonl files')
    ap.add_argument('--out_dir', default=None, help='where to write cell_scores.json / transfer_scores.json (default: runs_dir)')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_scores, transfer_scores = build_scores(runs_dir)

    (out_dir / 'cell_scores.json').write_text(
        json.dumps(cell_scores, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (out_dir / 'transfer_scores.json').write_text(
        json.dumps(transfer_scores, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

    n_labels = len(cell_scores)
    n_cells = len(next(iter(cell_scores.values()))) if cell_scores else 0
    print(f'wrote cell_scores.json ({n_labels} checkpoints, {n_cells} cells at the first checkpoint) to {out_dir}')
    print(f'wrote transfer_scores.json to {out_dir}')


if __name__ == '__main__':
    main()
