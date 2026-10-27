#!/usr/bin/env bash
# Run pipeline stages in order: bash run.sh prepare retrieve features train predict
# BER_SMOKE=1 runs on a 2% sample (separate work/output dirs). Logs + .ok/.fail markers in ~/ber/logs.
set -uo pipefail
ROOT="$HOME/ber"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"  # this repo's src/; data, work and logs stay under $ROOT
PY="$ROOT/.venv/bin/python"
export HF_HOME="$ROOT/hf_cache" TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1
TAG=$([ "${BER_SMOKE:-0}" = "1" ] && echo smoke || echo full)${BER_LOGTAG:+_$BER_LOGTAG}
mkdir -p "$ROOT/logs"
for st in "$@"; do
  LOG="$ROOT/logs/${TAG}_${st}.log"
  rm -f "$ROOT/logs/${TAG}_${st}.ok" "$ROOT/logs/${TAG}_${st}.fail"
  echo "== $st ($TAG) start $(date)" > "$LOG"
  if "$PY" "$SRC/$st.py" >> "$LOG" 2>&1; then
    echo "== $st ($TAG) done $(date)" >> "$LOG"
    touch "$ROOT/logs/${TAG}_${st}.ok"
  else
    echo "== $st ($TAG) FAILED $(date)" >> "$LOG"
    touch "$ROOT/logs/${TAG}_${st}.fail"
    exit 1
  fi
done
