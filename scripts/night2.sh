#!/usr/bin/env bash
# #11 = #9 + Qwen3-Reranker-4B as a third cross-encoder feature. Restores #9's cross-encoder test scores
# (round-1 French self-training), runs qwen.py, retrains the matcher with e5 + bge + qwen, predicts.
# Outputs kept as output_sub11 / *_sub11; the #7 matcher is restored as the working copy afterwards.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/Q11_DONE" "$L/Q11_FAILED"
fail() { echo "failed at: $1" > "$L/Q11_FAILED"; exit 1; }
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet" || fail restore
cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet" || fail restore
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
cd "$R"
BER_QWEN_STEPS=${BER_QWEN_STEPS:-1500} BER_QWEN_CAP=${BER_QWEN_CAP:-1500000} BER_LOGTAG=qwen bash scripts/run.sh qwen || fail qwen
BER_XENC_EXTRA=_bge,_qwen BER_LOGTAG=q11 bash scripts/run.sh train predict || fail train_predict
cp "$W/test_scored.parquet" "$W/test_scored_sub11.parquet"
mkdir -p "$R/output_sub11"; cp "$R"/output/*.tsv "$R/output_sub11/"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/$f.json" "$W/${f}_sub11.json"; cp "$W/${f}_sub7.json" "$W/$f.json"; done
touch "$L/Q11_DONE"
