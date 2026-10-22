"""Cross-hardware replication check for Pythia-1.4B checkpoint 76000 (H800 session, 2026-09-25, vs RTX 3090 session, 2026-10-21).
Prints the agreement statistics quoted in the paper's appendix."""
from __future__ import annotations
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A = ROOT / "cloud_return_round4/public_ckpt/runs/pythia-1.4b/eval_76000.jsonl.gz"
B = ROOT / "cloud_return_round5/public_ckpt/runs/pythia-1.4b/eval_76000.jsonl.gz"


def rows(p):
    with gzip.open(p, "rt", encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


ra, rb = rows(A), rows(B)
assert [r["id"] for r in ra] == [r["id"] for r in rb]
same_pred = sum(x["prediction"] == y["prediction"] for x, y in zip(ra, rb))
same_corr = sum(x["correct"] == y["correct"] for x, y in zip(ra, rb))
acc = {}
for r in ra + rb:
    key = (r["kind"], r["task"] if r["kind"] != "composition" else f"{r['face']}/{r['path_index']}")
    acc.setdefault(key, [[0, 0], [0, 0]])
for i, rs in enumerate((ra, rb)):
    for r in rs:
        key = (r["kind"], r["task"] if r["kind"] != "composition" else f"{r['face']}/{r['path_index']}")
        acc[key][i][0] += r["correct"]
        acc[key][i][1] += 1
diffs = sorted(((abs(a[0] / a[1] - b[0] / b[1]), k) for k, (a, b) in acc.items()), reverse=True)
print(f"rows: {len(ra)}; identical greedy completions: {same_pred}; identical correctness: {same_corr}")
print(f"overall accuracy H800 = {sum(r['correct'] for r in ra) / len(ra):.4f}, RTX 3090 = {sum(r['correct'] for r in rb) / len(rb):.4f}")
print(f"largest per-cell accuracy difference: {diffs[0][0]:.4f} ({diffs[0][1]})")
