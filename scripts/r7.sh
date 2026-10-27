#!/usr/bin/env bash
# #27: French self-training round 7 (e5/bge, from the round-1 checkpoints) on #26b's French decisions, then France
# through the #26b route: sub11 matcher (e5 + bge + Qwen) with the France-adapted Qwen scores, rename guard (reference:
# the same route with the run-1 Qwen scores), filler/swap rule and decoy cap. US/India as #25/#26b -> output_r7.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/R7_DONE" "$L/R7_FAILED"
fail() { echo "failed at: $1" > "$L/R7_FAILED"; exit 1; }
cd "$R"
PY="${BER_PYTHON:-}"; [ -n "$PY" ] || { [ -x "$R/.venv/bin/python" ] && PY="$R/.venv/bin/python" || PY=python3; }
export BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0
# 1. #26b's France scores after the guard, then after the rule + its decisions (labels for round 7)
"$PY" - > "$L/full_r7_guard.log" 2>&1 <<'EOF' || fail guard_scores
import os, sys
sys.path.insert(0, os.path.expanduser("~/ber/code/business_entity_resolution/src"))
import polars as pl
from config import WORK
import final
test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
seen = set(pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list())
ids = test.filter(~pl.col("country").is_in(list(seen)))["id"]
sc = pl.read_parquet(WORK / "test_scored_qfr.parquet").select("s1_id", "rec_id", "p")
final.guard_renames(sc, ids, str(WORK / "test_scored_qus.parquet")).write_parquet(WORK / "test_scored_qfrg.parquet")
EOF
BER_SCORED="$W/test_scored_qfrg.parquet" BER_DECISION="$W/decision_sub11.json" BER_VARIANT_NAME=sub26bf \
  BER_CANDIDATES="$R/output_qfrg/candidate_pairs.tsv" BER_LOGTAG=fil26b bash scripts/run.sh fillers || fail fillers
# 2. round 7 + the France route with the adapted Qwen (-> r7q) and with the run-1 Qwen (-> r7u, guard reference)
[ -f "$W/xenc_qwen_test_orig.parquet" ] || cp "$W/xenc_qwen_test.parquet" "$W/xenc_qwen_test_orig.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub11.json" "$W/$f.json"; done
cp "$W/xenc_qwen_test_fr.parquet" "$W/xenc_qwen_test.parquet"
rc=0
BER_PSEUDO_ROUND=7 BER_PSEUDO_SRC=test_scored_fillers_sub26bf.parquet BER_PSEUDO_SELECTED="$R/output/fillers_sub26bf/matching_results.tsv" \
  BER_PSEUDO_NEG=0.02 BER_OUT="$R/output_r7P" BER_LOGTAG=r7 bash scripts/run.sh pseudo predict && cp "$W/test_scored.parquet" "$W/test_scored_r7q.parquet" || rc=1
cp "$W/xenc_test.parquet" "$W/xenc_test_r7.parquet"; cp "$W/xenc_bge_test.parquet" "$W/xenc_bge_test_r7.parquet"
cp "$W/xenc_qwen_test_frus.parquet" "$W/xenc_qwen_test.parquet"
[ $rc -eq 0 ] && { BER_OUT="$R/output_r7uP" BER_LOGTAG=r7u bash scripts/run.sh predict && cp "$W/test_scored.parquet" "$W/test_scored_r7u.parquet" || rc=1; }
cp "$W/xenc_qwen_test_orig.parquet" "$W/xenc_qwen_test.parquet"
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
[ $rc -eq 0 ] || fail pseudo_predict
# 3. final: US/India as before; France from the round-7 route with the guard, rule and cap
BER_OUT="$R/output_r7" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_r7q.parquet" BER_DECISION_UNSEEN="$W/decision_sub11.json" \
  BER_GUARD_SCORED="$W/test_scored_r7u.parquet" BER_LOGTAG=r7F bash scripts/run.sh final || fail final
touch "$L/R7_DONE"
