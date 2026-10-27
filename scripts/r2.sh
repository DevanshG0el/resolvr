#!/usr/bin/env bash
# France round 2 (#10). Backs up #9, runs the LOCO self-training check, pseudo-label round 2 on the
# cross-encoders (-> #10a, #7 XGBoost), then XGBoost retrained with French pseudo rows (-> #10b).
# Afterwards the #7 XGBoost/decision are restored as the working copy; #10b's are kept as *_sub10b.json.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/R2_DONE" "$L/R2_FAILED"
fail() { echo "failed at: $1" > "$L/R2_FAILED"; exit 1; }
cd "$W" || fail cd
cp -n test_scored.parquet test_scored_sub9.parquet
cp -n xenc_test.parquet xenc_test_sub9.parquet
cp -n xenc_bge_test.parquet xenc_bge_test_sub9.parquet
cp -n pseudo_meta.json pseudo_meta_r1.json
mkdir -p "$R/output_sub9"; [ -f "$R/output_sub9/matching_results.tsv" ] || cp "$R"/output/*.tsv "$R/output_sub9/"
cd "$R"
BER_LOGTAG=ls bash scripts/run.sh loco_self || echo "loco_self failed (continuing)"
BER_PSEUDO_ROUND=2 BER_PSEUDO_SRC=test_scored_sub9.parquet BER_PSEUDO_XGB=1 BER_LOGTAG=r2 \
  bash scripts/run.sh pseudo predict || fail pseudo_predict
cp "$W/test_scored.parquet" "$W/test_scored_sub10a.parquet"
mkdir -p "$R/output_sub10a"; cp "$R"/output/*.tsv "$R/output_sub10a/"
BER_TRAIN_EXTRA=fr_pseudo_train.parquet BER_XENC_EXTRA=_bge BER_LOGTAG=r2x \
  bash scripts/run.sh train predict || fail train_predict
cp "$W/test_scored.parquet" "$W/test_scored_sub10b.parquet"
mkdir -p "$R/output_sub10b"; cp "$R"/output/*.tsv "$R/output_sub10b/"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/$f.json" "$W/${f}_sub10b.json"; cp "$W/${f}_sub7.json" "$W/$f.json"; done
touch "$L/R2_DONE"
