"""Blocking study: candidate-set size per S1 vs recall, for fixed rank cut-offs and a learned filter.

The learned filter ("blocker") is a small XGBoost on retrieval signals only (cosine, forward/reverse
rank, gaps to the S1's best candidate and to the record's best S1, top-1/top-2 margins) - no string
comparison - trained on split B and evaluated on split V. It is a blocking stage: it decides which
pairs the matcher ever sees. Saves WORK/xgb_blocker.json and WORK/blocking_study.csv.
"""
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import SEED, WORK

BASE_KF, BASE_KR = 20, 3
BLOCK_COLS = ["cos", "rank_fwd", "rank_rev", "gap_s1_top", "gap_rec_top", "s1_top_margin", "rec_top_margin",
              "s1_top1_cos", "rec_top1_cos", "is_s3"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def blocker_frame(cand: pl.DataFrame) -> pl.DataFrame:
    return cand.with_columns(
        (pl.col("s1_top1_cos") - pl.col("cos")).alias("gap_s1_top"),
        (pl.col("rec_top1_cos") - pl.col("cos")).fill_null(0).alias("gap_rec_top"),
        (pl.col("s1_top1_cos") - pl.col("s1_top2_cos")).alias("s1_top_margin"),
        (pl.col("rec_top1_cos") - pl.col("rec_top2_cos")).alias("rec_top_margin"),
        (pl.col("rec_id").str.slice(0, 2) == "S3").cast(pl.Int8).alias("is_s3"),
        pl.col("rank_fwd").cast(pl.Int16), pl.col("rank_rev").cast(pl.Int16),
    )


def main():
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    cand = pl.read_parquet(WORK / "cand_train.parquet").filter(
        (pl.col("rank_fwd") <= BASE_KF) | (pl.col("rank_rev") <= BASE_KR))
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    split = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split"]).filter(
        pl.col("src") == 1).select(pl.col("id").alias("s1_id"), "split")
    lab = pairs.select("s1_id", "rec_id", pl.lit(1, pl.Int8).alias("label"))
    d = blocker_frame(cand).join(split, on="s1_id").join(lab, on=["s1_id", "rec_id"], how="left").with_columns(
        pl.col("label").fill_null(0))
    n_true = pairs.group_by("split").len()
    tot = dict(zip(n_true["split"], n_true["len"]))
    B, V = d.filter(pl.col("split") == "B"), d.filter(pl.col("split") == "V")
    nV_s1 = V["s1_id"].n_unique()
    log(f"B {B.height:,} pairs, V {V.height:,} pairs ({V.height / nV_s1:.1f}/S1), V true pairs {tot['V']:,}")

    # optional: which V pairs does the current matcher accept (for 'kept matches' estimate)
    kept_ref = None
    vs = WORK / "valid_scored.parquet"
    if vs.exists():
        kept_ref = pl.read_parquet(vs, columns=["s1_id", "rec_id", "label", "p"]).filter(
            (pl.col("p") >= 0.5) & (pl.col("label") == 1)).select("s1_id", "rec_id")
        log(f"reference: {kept_ref.height:,} V true pairs currently matched (p>=0.5)")

    rows = []

    def evaluate(name, keep: pl.DataFrame):
        r = {"rule": name, "cands_per_s1": keep.height / nV_s1,
             "recall": int(keep["label"].sum()) / tot["V"]}
        if kept_ref is not None:
            r["matched_pairs_kept"] = keep.join(kept_ref, on=["s1_id", "rec_id"]).height / kept_ref.height
        rows.append(r)

    for kf in (3, 5, 8, 10, 15, 20):
        for kr in (1, 2, 3):
            evaluate(f"fwd<={kf} | rev<={kr}", V.filter((pl.col("rank_fwd") <= kf) | (pl.col("rank_rev") <= kr)))

    # learned blocker
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                  max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.9, min_child_weight=20, seed=SEED)
    XB = B.select(BLOCK_COLS).to_numpy().astype(np.float32)
    es = (np.random.default_rng(SEED).random(len(XB)) < 0.05)
    dtr = xgb.DMatrix(XB[~es], B["label"].to_numpy()[~es])
    dva = xgb.DMatrix(XB[es], B["label"].to_numpy()[es])
    bst = xgb.train(params, dtr, 600, evals=[(dva, "es")], early_stopping_rounds=40, verbose_eval=100)
    bst.save_model(WORK / "xgb_blocker.json")
    q = bst.predict(xgb.DMatrix(V.select(BLOCK_COLS).to_numpy().astype(np.float32)),
                    iteration_range=(0, bst.best_iteration + 1))
    Vq = V.with_columns(pl.Series("q", q)).with_columns(
        pl.col("q").rank("ordinal", descending=True).over("s1_id").alias("q_rank"))
    for t in (0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001):
        evaluate(f"blocker q>={t}", Vq.filter(pl.col("q") >= t))
        evaluate(f"blocker q>={t} | top-2 by q", Vq.filter((pl.col("q") >= t) | (pl.col("q_rank") <= 2)))
    out = pl.DataFrame(rows).sort("cands_per_s1")
    out.write_csv(WORK / "blocking_study.csv")
    print(out.with_columns(pl.col("cands_per_s1").round(2), pl.col("recall").round(5),
                           *([pl.col("matched_pairs_kept").round(5)] if kept_ref is not None else [])), flush=True)
    g = bst.get_score(importance_type="gain")
    log("blocker feature gain: " + ", ".join(f"{BLOCK_COLS[int(k[1:])]}={v:.0f}" for k, v in
                                               sorted(g.items(), key=lambda t: -t[1])))


if __name__ == "__main__":
    main()
