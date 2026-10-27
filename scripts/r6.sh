#!/usr/bin/env bash
# #25: French self-training round 6 = the round-5 recipe (restart from the round-1 French checkpoints, positives =
# selected French pairs with p >= 0.8 after the rule, negatives p <= 0.02) with #21's French decisions as labels
# instead of #20's. Runs on the pre-normalisation work files (restored from work/prenorm; the normalised ones are
# moved to work/postnorm), so #25 differs from #21 only in the French cross-encoders. US/India as in #20/#21.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"; BK="$W/prenorm"; PN="$W/postnorm"
rm -f "$L/R6_DONE" "$L/R6_FAILED"
fail() { echo "failed at: $1" > "$L/R6_FAILED"; exit 1; }
cd "$R"
mkdir -p "$PN"
for f in train_recs.parquet test_recs.parquet train_pairs.parquet feat_B.parquet feat_V.parquet feat_test.parquet \
         namefeat_B.parquet namefeat_V.parquet namefeat_test.parquet dicts.json; do
  [ -f "$PN/$f" ] || mv "$W/$f" "$PN/" || fail "save normalised $f"
  cp "$BK/$f" "$W/$f" || fail "restore $f"
done
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
BER_PSEUDO_ROUND=6 BER_PSEUDO_SRC=test_scored_fillers_sub21f.parquet BER_PSEUDO_SELECTED="$R/output/fillers_sub21f/matching_results.tsv" \
  BER_PSEUDO_NEG=0.02 BER_OUT="$R/output_r6P" BER_LOGTAG=r6 bash scripts/run.sh pseudo predict; rc=$?
cp "$W/xenc_test.parquet" "$W/xenc_test_r6.parquet"; cp "$W/xenc_bge_test.parquet" "$W/xenc_bge_test_r6.parquet"
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
[ $rc -eq 0 ] || fail pseudo_predict
cp "$W/test_scored.parquet" "$W/test_scored_sub25.parquet"
BER_OUT="$R/output_r6" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_sub25.parquet" BER_DECISION_UNSEEN="$W/decision_sub7.json" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=r6F bash scripts/run.sh final || fail final
touch "$L/R6_DONE"
