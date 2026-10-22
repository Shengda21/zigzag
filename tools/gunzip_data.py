"""Decompress the gzipped training streams of the task worlds in place (needed only to re-run training).
Evaluation files (eval_*.jsonl.gz) are read compressed by the analysis code and need no decompression.

  python tools/gunzip_data.py            # complex_worlds/data, grokking/data, crosslingual/data
"""
import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for base in ("complex_worlds/data", "grokking/data", "crosslingual/data"):
    for gz in sorted((ROOT / base).rglob("*.jsonl.gz")):
        out = gz.with_suffix("")
        if out.exists():
            continue
        with gzip.open(gz, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("decompressed", out.relative_to(ROOT))
