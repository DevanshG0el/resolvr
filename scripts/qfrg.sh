#!/usr/bin/env bash
# #26b = #26 (France: sub11 matcher with the France-adapted Qwen) + rename guard: invented-name records at the S1
# address may not score higher than with the run-1 (US-trained) Qwen (test_scored_qus). US/India as #25.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/QFRG_DONE" "$L/QFRG_FAILED"
cd "$R"
BER_OUT="$R/output_qfrg" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_qfr.parquet" BER_DECISION_UNSEEN="$W/decision_sub11.json" \
  BER_GUARD_SCORED="$W/test_scored_qus.parquet" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=qfrgF bash scripts/run.sh final && touch "$L/QFRG_DONE" || touch "$L/QFRG_FAILED"
