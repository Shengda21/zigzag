"""Run the PROBES.jsonl probe set against a sequence of HuggingFace revisions
of a public pretraining checkpoint (e.g. ``EleutherAI/pythia-410m``), purely
by inference -- no gradient updates, no fine-tuning.

For each ``--revisions`` entry (an HF `revision` string, typically
``stepNNNN``) this:

  1. loads the tokenizer + causal LM at that revision,
  2. greedily decodes a completion for every probe prompt (batched, left
     padding, ``max_new_tokens=24``, truncated at the first newline),
  3. re-scores every row with ``zz_score.score`` and writes
     ``eval_<step>.jsonl`` (probe row + prediction + score fields) and
     ``eval_<step>.summary.json`` (``zz_score.summarize`` output + run
     metadata),
  4. frees the model (``del`` + ``gc.collect()``) before moving to the next
     revision, so peak memory is one model at a time.

Idempotent: a revision whose ``eval_<step>.summary.json`` already exists is
skipped without loading a model.

Example::

    python run_public.py --model EleutherAI/pythia-70m \\
        --revisions step1000,step32000,step143000 --device cpu --limit 12 \\
        --out ../runs/pythia-70m
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_CODE_DIR = Path(__file__).resolve().parent
_ZZ_ROOT = _CODE_DIR.parents[1]
_STAGE_AB_CODE = _ZZ_ROOT / 'stage_ab' / 'code'
if str(_STAGE_AB_CODE) not in sys.path:
    sys.path.insert(0, str(_STAGE_AB_CODE))

import zz_score  # noqa: E402

DEFAULT_PROBES_PATH = _CODE_DIR.parents[0] / 'data' / 'PROBES.jsonl'
DEFAULT_RUNS_DIR = _CODE_DIR.parents[0] / 'runs'


def parse_step(revision: str) -> Optional[int]:
    """Extract the integer step number from a revision string like
    ``'step32000'``; returns ``None`` if no ``stepNNN`` pattern is found."""
    m = re.search(r'step(\d+)', revision)
    return int(m.group(1)) if m else None


def step_label(revision: str) -> str:
    """Filesystem-safe label for a revision: the parsed step number if
    present, else the revision string with non-alnum characters replaced."""
    step = parse_step(revision)
    return str(step) if step is not None else re.sub(r'[^A-Za-z0-9_.-]', '_', revision)


def _stride_sample(items: List, n: int) -> List:
    """Evenly-strided sample of `n` items from `items` (n <= len(items))."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _stratified_composition_sample(rows: List[dict], n: int) -> List[dict]:
    """Sample `n` 'composition' rows biased towards *complete face pairs*: a
    ``(face, item, template)`` key whose both ``path_index`` 0 and 1 rows are
    present. Plain even striding over all composition rows (they're stored
    as one contiguous 64-row block per path per face) will usually miss
    pairing entirely at small `n`, and ``cell_scores.py::score_faces`` needs
    a matched pair to produce a non-empty face score -- so a small smoke
    test would otherwise never exercise face scoring. Guarantees at least
    one complete pair whenever ``n >= 2``.
    """
    if n >= len(rows):
        return list(rows)
    groups: Dict[tuple, Dict[int, dict]] = {}
    for r in rows:
        groups.setdefault((r['face'], r['item'], r['template']), {})[r['path_index']] = r
    complete_pairs = [g for g in groups.values() if 0 in g and 1 in g]
    n_pairs = min(n // 2, len(complete_pairs))
    out: List[dict] = []
    for g in _stride_sample(complete_pairs, n_pairs):
        out.append(g[0])
        out.append(g[1])
    if len(out) < n:
        chosen_ids = {r['id'] for r in out}
        remaining = [r for r in rows if r['id'] not in chosen_ids]
        out.extend(_stride_sample(remaining, n - len(out)))
    return out[:n]


def load_probes(path: Path, limit: Optional[int]) -> List[dict]:
    """Load probe rows from `path`. If `limit` is set and smaller than the
    full set, sample **stratified by kind**: 'edge' / 'vertex' /
    'composition' / 'transfer' each get an even share of `limit` (remainder
    distributed to the first kinds in sorted order), so a small smoke test
    touches every kind instead of whichever kind happens to occupy the
    globally-strided indices (with 4000 rows dominated by 'edge'/'composition',
    plain striding at e.g. --limit 12 could sample zero 'transfer' rows and
    no complete face pairs -- this is exactly what was observed in an earlier
    run). Within 'edge'/'vertex'/'transfer' each kind's own slice is
    evenly strided for task diversity; within 'composition' see
    :func:`_stratified_composition_sample`.
    """
    rows = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    if limit is None or limit >= len(rows):
        return rows
    by_kind: Dict[str, List[dict]] = {}
    for r in rows:
        by_kind.setdefault(r['kind'], []).append(r)
    kinds = sorted(by_kind)
    base, rem = divmod(limit, len(kinds))
    out: List[dict] = []
    for i, kind in enumerate(kinds):
        n_this = min(base + (1 if i < rem else 0), len(by_kind[kind]))
        if n_this <= 0:
            continue
        if kind == 'composition':
            out.extend(_stratified_composition_sample(by_kind[kind], n_this))
        else:
            out.extend(_stride_sample(by_kind[kind], n_this))
    return out


def truncate_at_newline(text: str) -> str:
    """First line of a generated continuation (our stop condition)."""
    return text.split('\n', 1)[0]



def curl_download(model_name: str, revision: str, dest: Path) -> Path:
    """Resumable download of one revision with curl (ZZ_CURL_DL=1): every file is fetched to
    <name>.part with `-C -`, mostly from hf-mirror.com (no proxy), every third attempt from huggingface.co
    through the proxy in the environment; a stall below 100 kB/s for
    60 s aborts the attempt so that the next one resumes."""
    import subprocess
    dest.mkdir(parents=True, exist_ok=True)
    for fn in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'pytorch_model.bin'):
        out, part = dest / fn, dest / (fn + '.part')
        if out.exists():
            continue
        for attempt in range(30):
            mirror = attempt % 3 != 2          # hf-mirror.com (no proxy) first; every third attempt through the proxy to huggingface.co
            base = 'https://hf-mirror.com' if mirror else 'https://huggingface.co'
            cmd = ['curl', '-sSL', '--fail', '-C', '-', '--connect-timeout', '20', '--speed-limit', '100000', '--speed-time', '60',
                   '-o', str(part), f'{base}/{model_name}/resolve/{revision}/{fn}']
            if mirror:
                cmd[1:1] = ['--noproxy', '*']
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                os.replace(part, out)
                break
            if r.returncode == 22 and fn != 'pytorch_model.bin':       # optional tokenizer file absent in this revision
                part.unlink(missing_ok=True)
                break
            print(f'[{revision}] curl {fn} attempt {attempt + 1} ({base}) exit {r.returncode}: {r.stderr.strip()[:120]}', flush=True)
            time.sleep(5)
        else:
            raise RuntimeError(f'curl download of {fn} failed 30 times')
    return dest

def run_revision(model_name: str, revision: str, rows: List[dict], device: str, batch_size: int,
                  max_new_tokens: int, out_dir: Path) -> Optional[dict]:
    """Run one revision end to end; returns the summary dict, or ``None`` if
    it was skipped because results already exist."""
    label = step_label(revision)
    jsonl_path = out_dir / f'eval_{label}.jsonl'
    summary_path = out_dir / f'eval_{label}.summary.json'
    if summary_path.exists():
        print(f'[{revision}] summary already exists at {summary_path}, skipping')
        return None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f'[{revision}] loading tokenizer + model ({model_name})...')
    # Stream: download this revision into its own temp cache, load from there, delete afterwards
    # (24 Pythia-410m revisions would not fit on the rented disk). HF_ENDPOINT may point to a mirror.
    tmp = Path(os.environ.get('ZZ_HF_TMP', str(Path.home() / '.cache' / 'zz_hf'))) / label
    from huggingface_hub import snapshot_download
    local = str(curl_download(model_name, revision, tmp)) if os.environ.get('ZZ_CURL_DL') == '1' else None
    for attempt in range(0 if local else 8):
        try:
            # Pythia revisions carry both an fp32 safetensors pair and an fp16 pytorch_model.bin: fetch only one weight format
            # (ZZ_HF_WEIGHTS=bin|safetensors, default bin when the revision has it) to halve the download.
            weights = ['pytorch_model*.bin'] if os.environ.get('ZZ_HF_WEIGHTS', 'bin') == 'bin' else ['*.safetensors']   # fallback when listing fails
            try:
                from huggingface_hub import HfApi
                names = [x.rfilename for x in HfApi().model_info(model_name, revision=revision).siblings]
                pref = os.environ.get('ZZ_HF_WEIGHTS', 'bin')
                if pref == 'bin' and any(n.endswith('.bin') for n in names):
                    weights = ['pytorch_model*.bin']
                elif any(n.endswith('.safetensors') for n in names):
                    weights = ['*.safetensors']
            except Exception as exc:      # noqa: BLE001 - listing is an optimisation only
                print(f'[{revision}] file listing failed ({type(exc).__name__}); assuming {weights}')
            ignore = ['*.safetensors', '*.safetensors.index.json'] if weights == ['pytorch_model*.bin'] else ['*.bin', 'pytorch_model.bin.index.json']
            local = snapshot_download(model_name, revision=revision, cache_dir=str(tmp),
                                      allow_patterns=['*.json', '*.txt', '*.model'] + weights, ignore_patterns=ignore)
            break
        except Exception as exc:      # broken mirror connection: clean up and retry
            print(f'[{revision}] download attempt {attempt + 1} failed: {type(exc).__name__}: {str(exc)[:160]}')
            if attempt == 7:          # keep the partial blobs between attempts: huggingface_hub resumes *.incomplete files
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            time.sleep(15)
    tokenizer = AutoTokenizer.from_pretrained(local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    load_kw = dict(torch_dtype=torch.bfloat16) if device == 'cuda' else {}
    model = AutoModelForCausalLM.from_pretrained(local, **load_kw)
    gpu_lock = None
    if device == 'cuda' and os.name != 'nt':      # several download workers share the GPU with training: one model on the GPU at a time
        import fcntl
        gpu_lock = open(os.environ.get('ZZ_GPU_LOCK', '/tmp/zz_public_gpu.lock'), 'a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    model.to(device)
    model.eval()
    t_load = time.time() - t0
    print(f'[{revision}] loaded in {t_load:.1f}s, running {len(rows)} probes (batch_size={batch_size})...')

    predictions: List[str] = []
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            prompts = [r['prompt'] for r in batch]
            enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
            gen_only = out[:, enc['input_ids'].shape[1]:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            predictions.extend(truncate_at_newline(t) for t in texts)
    t_infer = time.time() - t1
    print(f'[{revision}] inference done in {t_infer:.1f}s ({t_infer / max(1, len(rows)):.3f}s/row)')

    eval_rows = []
    for row, pred in zip(rows, predictions):
        scored = zz_score.score(row, pred)
        eval_rows.append(dict(row, prediction=pred, **scored))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf8', newline='\n') as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = zz_score.summarize(eval_rows)
    meta = dict(model=model_name, revision=revision, step=parse_step(revision), device=device,
                n_rows=len(rows), load_seconds=round(t_load, 2), infer_seconds=round(t_infer, 2),
                overall_accuracy_all_rows=sum(r['correct'] for r in eval_rows) / len(eval_rows) if eval_rows else 0.0,
                groups=summary)
    summary_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f'[{revision}] wrote {jsonl_path.name} and {summary_path.name} '
          f'(overall accuracy over sampled rows = {meta["overall_accuracy_all_rows"]:.3f})')

    del model, tokenizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    if gpu_lock is not None:
        gpu_lock.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='HF model id, e.g. EleutherAI/pythia-410m')
    ap.add_argument('--revisions', required=True, help='comma-separated HF revision names, e.g. step1000,step2000')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--limit', type=int, default=None, help='use only this many probe rows (debug/smoke test)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=24)
    ap.add_argument('--probes', default=str(DEFAULT_PROBES_PATH), help='path to PROBES.jsonl')
    ap.add_argument('--out', default=None, help='output directory (default: runs/<model-basename>/)')
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else DEFAULT_RUNS_DIR / args.model.replace('/', '__')
    rows = load_probes(Path(args.probes), args.limit)
    revisions = [r.strip() for r in args.revisions.split(',') if r.strip()]

    print(f'model={args.model} device={args.device} n_rows={len(rows)} out={out_dir}')
    results = []
    for rev in revisions:
        try:
            meta = run_revision(args.model, rev, rows, args.device, args.batch_size,
                                 args.max_new_tokens, out_dir)
        except Exception as exc:  # report and continue with the next revision
            shutil.rmtree(Path(os.environ.get('ZZ_HF_TMP', str(Path.home() / '.cache' / 'zz_hf'))) / step_label(rev), ignore_errors=True)
            print(f'[{rev}] FAILED: {type(exc).__name__}: {exc}')
            meta = dict(model=args.model, revision=rev, error=f'{type(exc).__name__}: {exc}')
            (out_dir / f'eval_{step_label(rev)}.error.json').parent.mkdir(parents=True, exist_ok=True)
            (out_dir / f'eval_{step_label(rev)}.error.json').write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        results.append(meta)
    print('done.')


if __name__ == '__main__':
    main()
