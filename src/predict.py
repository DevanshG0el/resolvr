"""Stage E: score the test candidates, apply the tuned decision rule, write the submission files.

output/candidate_pairs.tsv  every candidate the matcher scored (the feat_test pairs)
output/matching_results.tsv final matches (subset of the candidates); one row per test S1
Then runs the official validator.
"""
import json
import subprocess
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import DATA, OUT, ROOT, WORK
from train import context, load_feats


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_lists(pairs: pl.DataFrame, s1_all: pl.Series, col: str, path):
    lists = pairs.group_by("s1_id").agg(pl.col("rec_id").unique().sort().str.join(",").alias(col))
    out = pl.DataFrame({"source1_entity_id": s1_all}).join(
        lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left"
    ).with_columns(pl.col(col).fill_null(""))
    assert out.height == s1_all.n_unique()
    out.write_csv(path, separator="\t", quote_style="never")
    log(f"wrote {path} ({out.height:,} rows, {(out[col] != '').sum():,} non-empty)")


def main():
    t0 = time.time()
    dec = json.loads((WORK / "decision.json").read_text())
    F = load_feats("test", dec.get("use_xenc", False), dec.get("xenc_tags", [""]), namefeat=dec.get("namefeat", False))
    m1 = xgb.Booster(model_file=str(WORK / "xgb_stage1.json"))
    X = F.select(dec["cols"]).to_numpy().astype(np.float32)
    p = m1.inplace_predict(X)
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / "xgb_stage2.json"))
        F2 = pl.concat([F, context(F, p)], how="horizontal")
        X2 = F2.select(dec["cols2"]).to_numpy().astype(np.float32)
        p = m2.inplace_predict(X2)
        if dec.get("cat_w", 0) > 0:  # CatBoost stage-2 blend (weight chosen on V by train.py)
            from catboost import CatBoostClassifier
            cm = CatBoostClassifier()
            cm.load_model(str(WORK / "cat_stage2.cbm"))
            p = (1 - dec["cat_w"]) * p + dec["cat_w"] * cm.predict_proba(X2)[:, 1]
            log(f"blended CatBoost stage 2 with weight {dec['cat_w']}")
    log(f"scored {F.height:,} test pairs")
    F.select("s1_id", "rec_id").with_columns(pl.Series("p", p)).write_parquet(WORK / "test_scored.parquet")

    cfg = dec["cfg"]
    ex = D.exclusive(F.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    if cfg["rule"] == "threshold":
        sel = D.select_threshold(ex, "p", cfg["thr"])
    else:
        ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
        sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"])

    recs = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"])
    s1 = recs.filter(pl.col("src") == 1)
    write_lists(F.select("s1_id", "rec_id"), s1["id"], "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    write_lists(sel, s1["id"], "matched_entity_ids", OUT / "matching_results.tsv")

    # sanity: matches per S1 by country (France has no labels, compare against US / India)
    per = s1.select(pl.col("id").alias("s1_id"), "country").join(
        sel.group_by("s1_id").len(name="n_match"), on="s1_id", how="left").fill_null(0)
    print(per.group_by("country").agg(
        pl.len().alias("s1"), (pl.col("n_match") == 0).mean().round(4).alias("empty_rate"),
        pl.col("n_match").mean().round(3).alias("mean_matches"),
    ).sort("country"), flush=True)
    print(per.group_by("country", "n_match").len(name="n_s1").sort("country", "n_match"), flush=True)

    validator = DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(OUT / "matching_results.tsv"),
                        "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"), "--check-ids"],
                       capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
    log(f"validator exit code {r.returncode}; stage E done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
