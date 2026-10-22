#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_general.py

Idempotent downloader + verifier for two public multilingual evaluation
datasets used in a cross-lingual experiment:

  1. Belebele  (facebook/belebele)  -> English / Simplified Chinese / Russian
  2. XQuAD     (google/xquad)       -> English / Chinese / Russian

This script ONLY downloads raw upstream files and writes a manifest
(raw/RAW_MANIFEST.json) describing them. It does NOT build any training
data, does NOT merge/transform records beyond the minimal parquet->jsonl
conversion needed because XQuAD is only published as parquet on the Hub.

Idempotency: for every target file, if it already exists on disk AND its
sha256 matches the value recorded in a previous run's RAW_MANIFEST.json,
the download is skipped. Otherwise it is (re)downloaded via
huggingface_hub.hf_hub_download, falling back to a direct `requests` GET
against the resolve/main URL (retried up to 3 times) if hf_hub_download
fails.

Usage:
    python download_general.py

Requires: huggingface_hub, requests, (pyarrow -- only needed for the
XQuAD parquet -> jsonl conversion; the script will try to `pip install
pyarrow` automatically if it is missing).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
import requests

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
BELEBELE_DIR = RAW_DIR / "belebele"
XQUAD_DIR = RAW_DIR / "xquad"
MANIFEST_PATH = RAW_DIR / "RAW_MANIFEST.json"

BELEBELE_REPO = "facebook/belebele"
BELEBELE_FILES = [
    "data/eng_Latn.jsonl",
    "data/zho_Hans.jsonl",
    "data/rus_Cyrl.jsonl",
    "README.md",
]
BELEBELE_LANG_MAP = {
    "eng_Latn": "data/eng_Latn.jsonl",
    "zho_Hans": "data/zho_Hans.jsonl",
    "rus_Cyrl": "data/rus_Cyrl.jsonl",
}

XQUAD_REPO = "google/xquad"
# As of this writing google/xquad on the HF Hub only publishes auto-converted
# parquet shards (one subdir per language config), there is no longer a raw
# xquad.<lang>.json at the repo root. We download the parquet (kept as the
# authoritative raw artifact) and additionally materialize a .jsonl view of
# it for convenience.
XQUAD_LANG_FILES = {
    "en": "xquad.en/validation-00000-of-00001.parquet",
    "zh": "xquad.zh/validation-00000-of-00001.parquet",
    "ru": "xquad.ru/validation-00000-of-00001.parquet",
}
XQUAD_README = "README.md"

RETRIES = 3
RETRY_SLEEP_SECONDS = 3

DOWNLOAD_TIMESTAMP_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_previous_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def previous_sha256_for(manifest: dict, rel_path: str) -> str | None:
    for section in ("belebele", "xquad"):
        files = manifest.get(section, {}).get("files", {})
        entry = files.get(rel_path)
        if entry:
            return entry.get("sha256")
    return None


def already_up_to_date(local_path: Path, expected_sha256: str | None) -> bool:
    if not local_path.exists() or local_path.stat().st_size == 0:
        return False
    if expected_sha256 is None:
        # No prior record -- can't confirm identity without re-hashing
        # against something; treat as needing (re)download to be safe,
        # UNLESS the file is present with nonzero size AND we have no
        # manifest at all (fresh raw/ dir wasn't touched). To stay strictly
        # correct we only skip when we have a previous recorded hash that
        # matches, per the task's idempotency rule.
        return False
    return sha256_file(local_path) == expected_sha256


def hf_download_with_fallback(repo_id: str, filename: str, local_dir: Path) -> Path:
    """Download `filename` from `repo_id` (dataset repo) into local_dir,
    preserving the repo-relative sub-path. Tries hf_hub_download first
    (retried), then falls back to a direct requests GET against
    resolve/main (also retried)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    target_path = local_dir / filename

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(local_dir),
            )
            return Path(downloaded)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    [hf_hub_download] attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    print(f"    hf_hub_download exhausted retries, falling back to requests: {last_err}")

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            tmp_path = target_path.with_suffix(target_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            tmp_path.replace(target_path)
            return target_path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    [requests] attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Failed to download {repo_id}/{filename} after all fallbacks: {last_err}")


def ensure_file(repo_id: str, filename: str, local_dir: Path, prev_manifest: dict) -> Path:
    """Idempotently ensure `filename` from `repo_id` exists under local_dir.
    Skips network access entirely if the file is already present on disk
    and its sha256 matches the value recorded in the previous manifest."""
    target_path = local_dir / filename
    prev_hash = previous_sha256_for(prev_manifest, str(target_path.relative_to(RAW_DIR)).replace("\\", "/"))

    if already_up_to_date(target_path, prev_hash):
        print(f"  [skip, up to date] {target_path.relative_to(RAW_DIR)}")
        return target_path

    print(f"  [download] {repo_id}::{filename} -> {target_path.relative_to(RAW_DIR)}")
    return hf_download_with_fallback(repo_id, filename, local_dir)


def ensure_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return
    except ImportError:
        print("  pyarrow not found, attempting `pip install pyarrow` ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])


def parquet_to_jsonl(parquet_path: Path, jsonl_path: Path) -> int:
    """Convert a parquet file to a .jsonl file (one JSON object per line).
    Returns the number of records written."""
    ensure_pyarrow()
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def extract_license_sentence(readme_path: Path) -> str:
    """Best-effort extraction of a single license-related sentence from a
    HF dataset card README.md.

    Preference order:
      1. Prose found under a '... License(s|ing Information) ...' heading,
         UNLESS it is just the generic HF template placeholder
         "[More Information Needed]" (not real content).
      2. The YAML front-matter `license:` tag (inline value, or the first
         `- <value>` list item on the following line for multi-license
         cards), formatted as `license: <value>`.
      3. Any other line in the document mentioning "cc-by"/"cc by".
    """
    if not readme_path.exists():
        return ""
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    def is_placeholder(s: str) -> bool:
        return "more information needed" in s.lower()

    # 1) find a heading containing "license"/"licensing" (case-insensitive);
    # only look within that section (stop at the next heading of any level)
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip().lower()
        if line.strip().startswith("#") and "licens" in stripped:
            for j in range(i + 1, len(lines)):
                candidate = lines[j].strip()
                if candidate.startswith("#"):
                    break  # left the license section
                if candidate and not is_placeholder(candidate) and not candidate.startswith("```"):
                    return candidate

    # 2) YAML front-matter `license:` tag
    for i, line in enumerate(lines):
        if line.strip().lower() == "license:":
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if nxt.startswith("-"):
                    return f"license: {nxt.lstrip('-').strip()}"
                break
        m = line.strip().lower()
        if m.startswith("license:") and m != "license:":
            return line.strip()

    # 3) fallback: any line mentioning cc-by
    for line in lines:
        low = line.lower()
        if "cc-by" in low or "cc by" in low:
            candidate = line.strip()
            if candidate:
                return candidate
    return ""


# --------------------------------------------------------------------------
# Belebele
# --------------------------------------------------------------------------

def download_belebele(prev_manifest: dict) -> dict:
    print("== Belebele ==")
    file_records = {}
    for filename in BELEBELE_FILES:
        path = ensure_file(BELEBELE_REPO, filename, BELEBELE_DIR, prev_manifest)
        rel = str(path.relative_to(RAW_DIR)).replace("\\", "/")
        file_records[rel] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return file_records


def verify_belebele(file_records: dict) -> dict:
    print("== Verifying Belebele ==")
    per_lang = {}
    id_sets = {}
    field_names_by_lang = {}

    for lang, relname in BELEBELE_LANG_MAP.items():
        path = BELEBELE_DIR / relname
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        per_lang[lang] = len(records)
        field_names_by_lang[lang] = sorted(records[0].keys()) if records else []

        # Identify the question-identity field: Belebele uses `link` +
        # `question_number` together as the stable cross-lingual key
        # (the same FLORES passage / question numbered identically in
        # every language variant).
        id_set = set()
        for r in records:
            key = (r.get("link"), r.get("question_number"))
            id_set.add(key)
        id_sets[lang] = id_set

    langs = list(BELEBELE_LANG_MAP.keys())
    common_ids = set.intersection(*id_sets.values()) if id_sets else set()
    all_equal = all(id_sets[l] == id_sets[langs[0]] for l in langs)

    result = {
        "num_records_per_language": per_lang,
        "field_names_per_language": field_names_by_lang,
        "id_field_used": "(link, question_number)",
        "id_sets_identical_across_languages": all_equal,
        "aligned_question_count": len(common_ids),
    }
    print(f"  records per language: {per_lang}")
    print(f"  (link, question_number) sets identical across languages: {all_equal}")
    print(f"  aligned question count: {len(common_ids)}")
    return result


# --------------------------------------------------------------------------
# XQuAD
# --------------------------------------------------------------------------

def download_xquad(prev_manifest: dict) -> dict:
    print("== XQuAD ==")
    file_records = {}

    # README
    readme_path = ensure_file(XQUAD_REPO, XQUAD_README, XQUAD_DIR, prev_manifest)
    rel = str(readme_path.relative_to(RAW_DIR)).replace("\\", "/")
    file_records[rel] = {"bytes": readme_path.stat().st_size, "sha256": sha256_file(readme_path)}

    # Parquet shards (raw, authoritative) + jsonl conversion (derived, for convenience)
    for lang, filename in XQUAD_LANG_FILES.items():
        parquet_path = ensure_file(XQUAD_REPO, filename, XQUAD_DIR, prev_manifest)
        rel = str(parquet_path.relative_to(RAW_DIR)).replace("\\", "/")
        file_records[rel] = {"bytes": parquet_path.stat().st_size, "sha256": sha256_file(parquet_path)}

        jsonl_path = XQUAD_DIR / f"xquad.{lang}.jsonl"
        print(f"  [convert] {parquet_path.name} -> {jsonl_path.relative_to(RAW_DIR)}")
        n = parquet_to_jsonl(parquet_path, jsonl_path)
        rel_jsonl = str(jsonl_path.relative_to(RAW_DIR)).replace("\\", "/")
        file_records[rel_jsonl] = {
            "bytes": jsonl_path.stat().st_size,
            "sha256": sha256_file(jsonl_path),
            "derived_from": rel,
            "note": "materialized from parquet; parquet is the authoritative raw artifact",
            "num_records": n,
        }

    return file_records


def verify_xquad(file_records: dict) -> dict:
    print("== Verifying XQuAD ==")
    per_lang_paragraphs = {}
    per_lang_questions = {}
    id_sets = {}
    field_names_by_lang = {}

    for lang in XQUAD_LANG_FILES.keys():
        jsonl_path = XQUAD_DIR / f"xquad.{lang}.jsonl"
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        per_lang_questions[lang] = len(records)
        field_names_by_lang[lang] = sorted(records[0].keys()) if records else []

        contexts = set(r.get("context") for r in records)
        per_lang_paragraphs[lang] = len(contexts)

        id_sets[lang] = set(r.get("id") for r in records)

    langs = list(XQUAD_LANG_FILES.keys())
    all_equal = all(id_sets[l] == id_sets[langs[0]] for l in langs)
    common_ids = set.intersection(*id_sets.values()) if id_sets else set()

    result = {
        "num_paragraphs_per_language": per_lang_paragraphs,
        "num_questions_per_language": per_lang_questions,
        "field_names_per_language": field_names_by_lang,
        "id_sets_identical_across_languages": all_equal,
        "aligned_question_count": len(common_ids),
    }
    print(f"  paragraphs per language: {per_lang_paragraphs}")
    print(f"  questions per language: {per_lang_questions}")
    print(f"  question id sets identical across languages: {all_equal}")
    print(f"  aligned question id count: {len(common_ids)}")
    return result


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def build_manifest(belebele_files, belebele_verify, xquad_files, xquad_verify) -> dict:
    belebele_license = extract_license_sentence(BELEBELE_DIR / "README.md")
    xquad_license = extract_license_sentence(XQUAD_DIR / "README.md")

    manifest = {
        "download_timestamp_utc": DOWNLOAD_TIMESTAMP_UTC,
        "belebele": {
            "source": "https://huggingface.co/datasets/facebook/belebele",
            "repo_id": BELEBELE_REPO,
            "license_excerpt_from_readme": belebele_license,
            "files": belebele_files,
            "verification": belebele_verify,
        },
        "xquad": {
            "source": "https://huggingface.co/datasets/google/xquad",
            "repo_id": XQUAD_REPO,
            "license_excerpt_from_readme": xquad_license,
            "note": (
                "google/xquad no longer hosts raw xquad.<lang>.json files at the repo "
                "root; only per-language auto-converted parquet shards "
                "(xquad.<lang>/validation-00000-of-00001.parquet) are published. "
                "Parquet files were downloaded as the authoritative raw artifact and "
                "additionally converted losslessly to xquad.<lang>.jsonl for convenience."
            ),
            "files": xquad_files,
            "verification": xquad_verify,
        },
    }
    return manifest


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prev_manifest = load_previous_manifest()

    try:
        belebele_files = download_belebele(prev_manifest)
        belebele_verify = verify_belebele(belebele_files)
    except Exception:
        print("ERROR while processing Belebele:")
        traceback.print_exc()
        belebele_files, belebele_verify = {}, {"error": "failed, see log"}

    try:
        xquad_files = download_xquad(prev_manifest)
        xquad_verify = verify_xquad(xquad_files)
    except Exception:
        print("ERROR while processing XQuAD:")
        traceback.print_exc()
        xquad_files, xquad_verify = {}, {"error": "failed, see log"}

    manifest = build_manifest(belebele_files, belebele_verify, xquad_files, xquad_verify)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
