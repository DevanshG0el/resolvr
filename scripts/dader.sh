#!/usr/bin/env bash
# #23: France route with domain-adversarial alignment (dader.py = round-1 French self-training + gradient reversal).
# Phase 1 (GPU) may run while norm.sh runs: it reads the record texts from work/prenorm and writes only
# xenc*_test_dader.parquet. Phase 2 waits for NORM_DONE (both scripts swap the same matcher / feature files), then
# the #7 matcher re-predicts France with the new French scores on the normalised features, and final.py builds the
# full submission with the normalised US/India scores (as #22) and #20's filler/swap settings -> output_dader.
# #23 therefore differs from #22 only in the French cross-encoder scores.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/DADER_DONE" "$L/DADER_FAILED"
fail() { echo "failed at: $1" > "$L/DADER_FAILED"; exit 1; }
cd "$R"
until [ -f "$L/full_norm_prepare.log" ] || [ -f "$L/NORM_FAILED" ]; do sleep 30; done  # norm.sh backups complete
[ -f "$W/prenorm/test_recs.parquet" ] || fail "no work/prenorm backup"
BER_LOGTAG=dader bash scripts/run.sh dader || fail dader
until [ -f "$L/NORM_DONE" ] || [ -f "$L/NORM_FAILED" ]; do sleep 60; done
[ -f "$L/NORM_DONE" ] || fail "norm.sh failed; phase 2 not run"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
cp "$W/xenc_test_dader.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_dader.parquet" "$W/xenc_bge_test.parquet"
BER_OUT="$R/output_daderP" BER_LOGTAG=daderP bash scripts/run.sh predict; rc=$?
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
[ $rc -eq 0 ] || fail predict
cp "$W/test_scored.parquet" "$W/test_scored_dader.parquet"
BER_OUT="$R/output_dader" BER_SCORED_SEEN="$W/test_scored_normA.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_dader.parquet" BER_DECISION_UNSEEN="$W/decision_sub7.json" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=daderF bash scripts/run.sh final || fail final
touch "$L/DADER_DONE"
