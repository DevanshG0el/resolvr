#!/usr/bin/env bash
# #26: Qwen for France.
#  1. labels = #25's French decisions after the filler/swap rule and decoy cap (fillers on test_scored_sub25)
#  2. qwen_fr.py: the run-1 Qwen LoRA continued on French pseudo-labels (cross-fitted over French halves), scores the
#     French band; the untouched run-1 models score the same pairs (control)
#  3. the sub11 matcher (e5 + bge + Qwen; validation 0.98899 vs 0.9884 for the #7 matcher) re-predicts France with the
#     round-6 French e5/bge scores and the adapted Qwen scores -> test_scored_qfr; with the run-1 Qwen scores ->
#     test_scored_qus (control)
#  4. final.py: US/India as #25; France from each route + #20's filler/swap settings -> output_qfr, output_qus
# The working files are restored afterwards (xenc*_test = round-1 French rows, decision/xgb = sub7).
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/QFR_DONE" "$L/QFR_FAILED"
fail() { echo "failed at: $1" > "$L/QFR_FAILED"; exit 1; }
cd "$R"
export BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0
BER_SCORED="$W/test_scored_sub25.parquet" BER_VARIANT_NAME=sub25f BER_CANDIDATES="$R/output_r6/candidate_pairs.tsv" \
  BER_LOGTAG=fil25 bash scripts/run.sh fillers || fail fillers
BER_PSEUDO_SRC=test_scored_fillers_sub25f.parquet BER_PSEUDO_SELECTED="$R/output/fillers_sub25f/matching_results.tsv" \
  BER_PSEUDO_NEG=0.02 BER_QWEN_FR_SCORED="$W/test_scored_sub25.parquet" BER_LOGTAG=qfr bash scripts/run.sh qwen_fr || fail qwen_fr
[ -f "$W/xenc_qwen_test_orig.parquet" ] || cp "$W/xenc_qwen_test.parquet" "$W/xenc_qwen_test_orig.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub11.json" "$W/$f.json"; done
cp "$W/xenc_test_r6.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_r6.parquet" "$W/xenc_bge_test.parquet"
rc=0
cp "$W/xenc_qwen_test_fr.parquet" "$W/xenc_qwen_test.parquet"
BER_OUT="$R/output_qfrP" BER_LOGTAG=qfrP bash scripts/run.sh predict && cp "$W/test_scored.parquet" "$W/test_scored_qfr.parquet" || rc=1
cp "$W/xenc_qwen_test_frus.parquet" "$W/xenc_qwen_test.parquet"
BER_OUT="$R/output_qusP" BER_LOGTAG=qusP bash scripts/run.sh predict && cp "$W/test_scored.parquet" "$W/test_scored_qus.parquet" || rc=1
cp "$W/xenc_qwen_test_orig.parquet" "$W/xenc_qwen_test.parquet"
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
[ $rc -eq 0 ] || fail predict
for v in qfr qus; do
  BER_OUT="$R/output_$v" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
    BER_SCORED_UNSEEN="$W/test_scored_$v.parquet" BER_DECISION_UNSEEN="$W/decision_sub11.json" \
    BER_LOGTAG="${v}F" bash scripts/run.sh final || fail "final $v"
done
touch "$L/QFR_DONE"
