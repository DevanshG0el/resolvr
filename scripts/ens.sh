#!/usr/bin/env bash
# #28: France = average of the #26b route (round-6 e5/bge) and the #27 route (round-7 e5/bge), both with the
# France-adapted Qwen and the sub11 matcher; the guard reference is the average of their run-1-Qwen twins.
# Then rename guard, filler/swap rule and decoy cap. US/India as #26b -> output_ens.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/ENS_DONE" "$L/ENS_FAILED"
cd "$R"
PY="${BER_PYTHON:-}"; [ -n "$PY" ] || { [ -x "$R/.venv/bin/python" ] && PY="$R/.venv/bin/python" || PY=python3; }
"$PY" - > "$L/full_ens_avg.log" 2>&1 <<'EOF' || { touch "$L/ENS_FAILED"; exit 1; }
import os
import polars as pl
W = os.path.expanduser("~/ber/work/")
for a, b, out in (("test_scored_qfr", "test_scored_r7q", "test_scored_ens"), ("test_scored_qus", "test_scored_r7u", "test_scored_ensu")):
    x = pl.read_parquet(W + a + ".parquet").select("s1_id", "rec_id", pl.col("p").alias("pa"))
    y = pl.read_parquet(W + b + ".parquet").select("s1_id", "rec_id", pl.col("p").alias("pb"))
    m = x.join(y, on=["s1_id", "rec_id"])
    assert m.height == x.height == y.height
    m.select("s1_id", "rec_id", ((pl.col("pa") + pl.col("pb")) / 2).alias("p")).write_parquet(W + out + ".parquet")
    print(out, m.height, "pairs; corr", round(m.select(pl.corr("pa", "pb")).item(), 5))
EOF
BER_OUT="$R/output_ens" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
  BER_SCORED_UNSEEN="$W/test_scored_ens.parquet" BER_DECISION_UNSEEN="$W/decision_sub11.json" \
  BER_GUARD_SCORED="$W/test_scored_ensu.parquet" \
  BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_LOGTAG=ensF bash scripts/run.sh final && touch "$L/ENS_DONE" || touch "$L/ENS_FAILED"
