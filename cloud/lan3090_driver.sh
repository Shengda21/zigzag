#!/usr/bin/env bash
# Sequential driver for the LAN 3090 box: for each revision, download the fp16 checkpoint with aria2c
# (aifasthub.com first, hf-mirror.com as fallback), verify the byte size against the server-reported
# x-linked-size, then run the unchanged run_public.py (which finds the files already present under
# ZZ_HF_TMP/<label>/ and deletes them after scoring).  Idempotent: finished revisions are skipped.
#
#   MODEL=EleutherAI/pythia-1.4b REVS="step79000 step82000 ..." bash driver.sh
#   optional: LIMIT=40 (smoke), BATCH=64, OUT=<runs dir>
set -u
ROOT=${ROOT:-$HOME/zz_pythia}
PY=$ROOT/venv/bin/python
CODE=$ROOT/zz/public_ckpt/code
MODEL=${MODEL:-EleutherAI/pythia-1.4b}
REVS=${REVS:?need REVS}
BATCH=${BATCH:-64}
LIMIT=${LIMIT:-}
OUT=${OUT:-$ROOT/zz/public_ckpt/runs/$(basename "$MODEL")}
HF_TMP=$ROOT/hf_tmp
BASES=${BASES:-"https://aifasthub.com https://hf-mirror.com"}
CONN=${CONN:-16}
mkdir -p "$HF_TMP" "$OUT" "$ROOT/logs"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZZ_CURL_DL=1 ZZ_HF_TMP=$HF_TMP HF_HUB_OFFLINE=1

log() { echo "$(date '+%F %T') $*"; }

fetch_rev() {   # $1 = revision ; returns 0 when $HF_TMP/<label> holds all five files with the right sizes
  local rev=$1 label=${1#step} dl="$HF_TMP/${1#step}.dl" final="$HF_TMP/${1#step}"
  [ -f "$final/pytorch_model.bin" ] && [ -f "$final/config.json" ] && return 0
  mkdir -p "$dl"
  for base in $BASES; do
    local ok=1
    for fn in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
      [ -s "$dl/$fn" ] && continue
      curl -sS -L --fail --connect-timeout 15 -m 120 -o "$dl/$fn" "$base/$MODEL/resolve/$rev/$fn" || { log "[$rev] $fn failed from $base"; ok=0; break; }
    done
    [ $ok = 1 ] || continue
    local want
    want=$(curl -sS -L -I --connect-timeout 15 -m 60 "$base/$MODEL/resolve/$rev/pytorch_model.bin" 2>/dev/null | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{s=$2} tolower($1)=="content-length:"{c=$2} END{print (s!=""?s:c)}')
    log "[$rev] expected pytorch_model.bin bytes: ${want:-unknown} (via $base)"
    aria2c -q -x "$CONN" -s "$CONN" -k 4M -c --file-allocation=none --max-tries=30 --retry-wait=5 --timeout=60 \
           --lowest-speed-limit=20K --summary-interval=0 -d "$dl" -o pytorch_model.bin "$base/$MODEL/resolve/$rev/pytorch_model.bin"
    local got; got=$(stat -c %s "$dl/pytorch_model.bin" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$want" != "unknown" ] && [ "$got" != "$want" ]; then
      log "[$rev] size mismatch got=$got want=$want from $base; retrying with next base"; continue
    fi
    [ "$got" -gt 1000000 ] || { log "[$rev] bin too small ($got)"; continue; }
    rm -f "$dl/pytorch_model.bin.aria2"
    mv "$dl" "$final"
    return 0
  done
  return 1
}

for rev in $REVS; do
  label=${rev#step}
  if [ -f "$OUT/eval_$label.summary.json" ]; then log "[$rev] already done, skipping"; continue; fi
  t0=$(date +%s)
  if ! fetch_rev "$rev"; then log "[$rev] DOWNLOAD FAILED after all bases"; echo "$rev" >> "$ROOT/logs/failed_revs.txt"; continue; fi
  t1=$(date +%s)
  log "[$rev] download done in $((t1-t0)) s; starting inference"
  lim=""; [ -n "$LIMIT" ] && lim="--limit $LIMIT"
  ( cd "$CODE" && nice -n 10 "$PY" -u run_public.py --model "$MODEL" --revisions "$rev" --device cuda --batch_size "$BATCH" $lim --out "$OUT" )
  rc=$?
  t2=$(date +%s)
  log "[$rev] inference rc=$rc in $((t2-t1)) s; total $((t2-t0)) s"
  rm -rf "$HF_TMP/$label" "$HF_TMP/$label.dl"
  df -h "$ROOT" | tail -1
done
log "ALL_DONE"
