#!/usr/bin/env bash
# Normalisation fixes (EI legal form, Frs -> Freres for France, Pas-de-Calais -> Hauts-de-France alias) on the
# existing candidates: re-normalise, recompute features + name features, re-predict with the saved matchers
# (final US/India matcher, #7 France-route matcher with round-1 French cross-encoder scores), final.py -> output_norm.
# The previous work files are backed up in work/prenorm/ first (restore them to return to the #20 state).
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"; BK="$W/prenorm"
rm -f "$L/NORM_DONE" "$L/NORM_FAILED"
fail() { echo "failed at: $1" > "$L/NORM_FAILED"; exit 1; }
mkdir -p "$BK"
for f in train_recs test_recs train_pairs feat_B feat_V feat_test namefeat_B namefeat_V namefeat_test; do
  [ -f "$BK/$f.parquet" ] || cp "$W/$f.parquet" "$BK/" || fail "backup $f"
done
[ -f "$BK/dicts.json" ] || cp "$W/dicts.json" "$BK/"
cd "$R"
BER_LOGTAG=norm bash scripts/run.sh prepare features namefeat || fail prepare_features
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_final.json" "$W/$f.json"; done
BER_LOGTAG=normA bash scripts/run.sh predict || fail predict_usin
cp "$W/test_scored.parquet" "$W/test_scored_normA.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
BER_LOGTAG=normB bash scripts/run.sh predict || fail predict_france
cp "$W/test_scored.parquet" "$W/test_scored_normB.parquet"
BER_OUT="$R/output_norm" BER_SCORED_SEEN="$W/test_scored_normA.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_normB.parquet" BER_DECISION_UNSEEN="$W/decision_sub7.json" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=normF bash scripts/run.sh final || fail final
touch "$L/NORM_DONE"
