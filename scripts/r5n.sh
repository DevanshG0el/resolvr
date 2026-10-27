#!/usr/bin/env bash
# #24 = #21 (French round-5 self-training) + normalisation fixes (#22): re-score France with the saved round-5
# checkpoints, the #7 matcher re-predicts on the normalised features, final.py takes US/India from the normalised
# final matcher (as #22) and applies #20's filler/swap settings -> output_r5n. Also checks the re-scored route
# against #21's saved scores (same models, pre-normalisation features: expect nearly identical France scores).
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/R5N_DONE" "$L/R5N_FAILED"
fail() { echo "failed at: $1" > "$L/R5N_FAILED"; exit 1; }
cd "$R"
BER_RESCORE_SFX=_r5 BER_LOGTAG=r5n bash scripts/run.sh rescore_fr || fail rescore
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
cp "$W/xenc_test_r5.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_r5.parquet" "$W/xenc_bge_test.parquet"
BER_OUT="$R/output_r5nP" BER_LOGTAG=r5nP bash scripts/run.sh predict; rc=$?
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
[ $rc -eq 0 ] || fail predict
cp "$W/test_scored.parquet" "$W/test_scored_r5n.parquet"
PY="${BER_PYTHON:-}"; [ -n "$PY" ] || { [ -x "$R/.venv/bin/python" ] && PY="$R/.venv/bin/python" || PY=python3; }
"$PY" - > "$L/full_r5n_check.log" 2>&1 <<'EOF'
import polars as pl
W = __import__("os").path.expanduser("~/ber/work/")
a = pl.read_parquet(W + "test_scored_sub21.parquet").rename({"p": "p21"})
b = pl.read_parquet(W + "test_scored_r5n.parquet").rename({"p": "p24"})
t = pl.read_parquet(W + "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
fr = t.filter(pl.col("country") == "France")["id"]
m = a.join(b, on=["s1_id", "rec_id"]).filter(pl.col("s1_id").is_in(fr.implode()))
print(f"France pairs {m.height:,}; corr p21 vs p24 {m.select(pl.corr('p21', 'p24')).item():.5f}; "
      f"|diff| > 0.1: {(abs(m['p21'] - m['p24']) > 0.1).sum():,}; sides of 0.5 differ: {((m['p21'] >= 0.5) != (m['p24'] >= 0.5)).sum():,}")
EOF
BER_OUT="$R/output_r5n" BER_SCORED_SEEN="$W/test_scored_normA.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_r5n.parquet" BER_DECISION_UNSEEN="$W/decision_sub7.json" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=r5nF bash scripts/run.sh final || fail final
touch "$L/R5N_DONE"
