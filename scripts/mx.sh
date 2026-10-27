#!/usr/bin/env bash
# Matcher experiments on validation (US/India route), run automatically after the second Qwen run:
#   a: e5 + bge + qwen + qwen2
#   b: a + fuzzy-namesake features (namefeat.py)
#   c: b + bagged trees (3 per boosting round)
# Each result is kept as decision/xgb_stage*_mx<a|b|c>.json; the #7 working copy is restored at the end.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/MX_DONE" "$L/MX_FAILED"
while [ ! -f "$L/full_qwen2_qwen.ok" ] && [ ! -f "$L/full_qwen2_qwen.fail" ]; do sleep 30; done
[ -f "$L/full_qwen2_qwen.ok" ] || { echo "qwen2 failed" > "$L/MX_FAILED"; exit 1; }
while [ ! -f "$L/full_nf_namefeat.ok" ] && [ ! -f "$L/full_nf_namefeat.fail" ]; do sleep 30; done
cd "$R"
run() {  # $1 = suffix, then environment assignments
  local sfx=$1; shift
  env "$@" BER_XENC_EXTRA=_bge,_qwen,_qwen2 BER_LOGTAG="mx$sfx" bash scripts/run.sh train || { echo "train $sfx failed" > "$L/MX_FAILED"; return 1; }
  for f in decision xgb_stage1 xgb_stage2; do cp "$W/$f.json" "$W/${f}_mx$sfx.json"; done
}
run a BER_NAMEFEAT=0 BER_XGB_TREES=1
if [ -f "$L/full_nf_namefeat.ok" ]; then
  run b BER_NAMEFEAT=1 BER_XGB_TREES=1
  run c BER_NAMEFEAT=1 BER_XGB_TREES=3
else
  run c BER_NAMEFEAT=0 BER_XGB_TREES=3
fi
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
touch "$L/MX_DONE"
