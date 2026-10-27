#!/usr/bin/env bash
# Final submission in one go (usage: bash fin.sh <mx variant: a|b|c>):
#  1. reproduce #19 through final.py -> output_repro19 (proof that the packaged code regenerates an upload)
#  2. refit the chosen US/India matcher on B + V and score the test pairs
#  3. final.py: US/India from that matcher; France from the self-trained route + filler/swap rule + decoy cap
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/FIN_DONE" "$L/FIN_FAILED"
fail() { echo "failed at: $1" > "$L/FIN_FAILED"; exit 1; }
VAR="${1:-b}"
case "$VAR" in a) NF=0; TR=1;; b) NF=1; TR=1;; c) NF=1; TR=3;; *) fail "unknown variant $VAR";; esac
export BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0
cd "$R"
BER_OUT="$R/output_repro19" BER_SCORED_SEEN="$W/test_scored_sub11.parquet" BER_DECISION_SEEN="$W/decision_sub11.json" \
  BER_LOGTAG=repro19 bash scripts/run.sh final || fail repro19
BER_FINAL_BV=1 BER_NAMEFEAT=$NF BER_XGB_TREES=$TR BER_XENC_EXTRA=_bge,_qwen,_qwen2 BER_LOGTAG="bv$VAR" \
  bash scripts/run.sh train predict || fail train_predict
cp "$W/test_scored.parquet" "$W/test_scored_final.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/$f.json" "$W/${f}_final.json"; cp "$W/${f}_sub7.json" "$W/$f.json"; done
BER_OUT="$R/output_final" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_LOGTAG=final bash scripts/run.sh final || fail final
touch "$L/FIN_DONE"
