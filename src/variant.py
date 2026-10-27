"""Decision-rule variant from saved test probabilities (no re-scoring).

Countries that never occur in training (test: France) are the most exposed to domain shift, and
F0.5 punishes wrong merges more than misses. This variant shifts their calibrated logits by
BER_UNSEEN_BIAS (default -1.0) before the same expected-F0.5 / threshold selection.
Writes OUT/variant_<name>/matching_results.tsv (+ a copy of candidate_pairs.tsv) and validates it.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import polars as pl

import decide as D
from config import DATA, OUT, WORK
from predict import write_lists

SHIFT = float(os.environ.get("BER_UNSEEN_BIAS", -1.0))
NAME = os.environ.get("BER_VARIANT_NAME", f"unseen{SHIFT:+.1f}")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    dec = json.loads(open(os.environ.get("BER_DECISION", WORK / "decision.json")).read())
    cfg = dec["cfg"]
    sc = pl.read_parquet(os.environ.get("BER_SCORED", WORK / "test_scored.parquet"))
    cand_file = os.environ.get("BER_CANDIDATES", OUT / "candidate_pairs.tsv")
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
    seen = set(pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list())
    unseen = sorted(set(test["country"].unique().to_list()) - seen)
    log(f"countries absent from training: {unseen}; logit shift {SHIFT}")
    ex = D.exclusive(sc.select("s1_id", "rec_id", "p")).join(
        test.select(pl.col("id").alias("s1_id"), "country"), on="s1_id")
    shift = np.where(ex["country"].is_in(unseen).to_numpy(), SHIFT, 0.0)
    if cfg["rule"] == "threshold":
        p = D.calibrate(ex["p"].to_numpy(), 1.0, shift)
        sel = D.select_threshold(ex.with_columns(pl.Series("p", p)), "p", cfg["thr"])
    else:
        p = D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"] + shift * cfg["temp"])
        sel = D.select_expected_f(ex.with_columns(pl.Series("p", p)).select("s1_id", "rec_id", "p"), "p",
                                  empty_scale=cfg["empty_scale"])
    out = OUT / f"variant_{NAME}"
    out.mkdir(parents=True, exist_ok=True)
    write_lists(sel, test["id"], "matched_entity_ids", out / "matching_results.tsv")
    shutil.copy(cand_file, out / "candidate_pairs.tsv")
    per = test.select(pl.col("id").alias("s1_id"), "country").join(
        sel.group_by("s1_id").len(name="n_match"), on="s1_id", how="left").fill_null(0)
    print(per.group_by("country").agg(
        pl.len().alias("s1"), (pl.col("n_match") == 0).mean().round(4).alias("empty_rate"),
        pl.col("n_match").mean().round(3).alias("mean_matches")).sort("country"), flush=True)
    validator = DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"),
                        "--check-ids"], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1000:], flush=True)
    log(f"variant {NAME} written to {out}; validator exit {r.returncode}")


if __name__ == "__main__":
    main()
