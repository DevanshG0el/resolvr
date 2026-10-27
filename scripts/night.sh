#!/usr/bin/env bash
# Overnight (#11): waits for r2.sh, then Qwen3-Reranker-4B as a third cross-encoder feature, then two matchers:
#   #11a  e5 + bge + qwen cross-encoders
#   #11b  same + French pseudo-labelled rows in XGBoost training (fr_pseudo_train.parquet from r2.sh)
# Each variant's outputs/models are kept as *_sub11a / *_sub11b; the #7 matcher is restored as the working copy.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/NIGHT_DONE" "$L/NIGHT_FAILED"
fail() { echo "failed at: $1" > "$L/NIGHT_FAILED"; exit 1; }
while [ ! -f "$L/R2_DONE" ] && [ ! -f "$L/R2_FAILED" ]; do sleep 60; done
cd "$R"
BER_QWEN_STEPS=${BER_QWEN_STEPS:-2000} BER_QWEN_CAP=${BER_QWEN_CAP:-2000000} BER_LOGTAG=qwen bash scripts/run.sh qwen || fail qwen
save() {  # $1 = suffix
  cp "$W/test_scored.parquet" "$W/test_scored_$1.parquet"
  mkdir -p "$R/output_$1"; cp "$R"/output/*.tsv "$R/output_$1/"
  for f in decision xgb_stage1 xgb_stage2; do cp "$W/$f.json" "$W/${f}_$1.json"; done
}
BER_XENC_EXTRA=_bge,_qwen BER_LOGTAG=q11a bash scripts/run.sh train predict || fail q11a
save sub11a
if [ -f "$W/fr_pseudo_train.parquet" ]; then
  BER_TRAIN_EXTRA=fr_pseudo_train.parquet BER_XENC_EXTRA=_bge,_qwen BER_LOGTAG=q11b bash scripts/run.sh train predict || fail q11b
  save sub11b
fi
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
touch "$L/NIGHT_DONE"
