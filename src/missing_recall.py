"""Impact of missing data on the current matcher (split V, US/India): recall and precision by
record-address state (empty / no house number / present) x how many S1 entities share the record's core name.
Unique-name records with an empty address are 93-97% real matches in the training labels, so their recall
shows whether missing addresses are handled as well as the data allows. Model: BER_MR_SFX (default _sub11).
"""
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import WORK
from train import context, load_feats

SFX = os.environ.get("BER_MR_SFX", "_sub11")


def main():
    dec = json.loads((WORK / f"decision{SFX}.json").read_text())
    V = load_feats("V", dec.get("use_xenc", False), dec.get("xenc_tags", [""]), namefeat=dec.get("namefeat", False))
    m1 = xgb.Booster(model_file=str(WORK / f"xgb_stage1{SFX}.json"))
    p = m1.inplace_predict(V.select(dec["cols"]).to_numpy().astype(np.float32))
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / f"xgb_stage2{SFX}.json"))
        p = m2.inplace_predict(pl.concat([V, context(V, p)], how="horizontal").select(dec["cols2"]).to_numpy().astype(np.float32))
    cfg = dec["cfg"]
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"]).with_columns(pl.lit(1, pl.Int8).alias("sel"))
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id").with_columns(pl.lit(1, pl.Int8).alias("true"))
    F = V.select("s1_id", "rec_id", "rec_addr_empty", "num1_eq", "b_n_s1_same_name").with_columns(pl.Series("p", p))
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "country"])
    F = F.join(recs.rename({"id": "s1_id"}), on="s1_id")
    F = F.join(sel.select("s1_id", "rec_id", "sel"), on=["s1_id", "rec_id"], how="left").join(truth, on=["s1_id", "rec_id"], how="left")
    F = F.with_columns(pl.col("sel").fill_null(0), pl.col("true").fill_null(0))
    F = F.with_columns(
        pl.when(pl.col("rec_addr_empty") == 1).then(pl.lit("1 address empty"))
        .when(pl.col("num1_eq") == -1).then(pl.lit("2 no house number (one side)"))
        .otherwise(pl.lit("3 address with number")).alias("address"),
        pl.when(pl.col("b_n_s1_same_name") == 0).then(pl.lit("0 (no exact S1 name)"))
        .when(pl.col("b_n_s1_same_name") == 1).then(pl.lit("1 unique"))
        .when(pl.col("b_n_s1_same_name") == 2).then(pl.lit("2 namesakes"))
        .when(pl.col("b_n_s1_same_name") <= 5).then(pl.lit("3-5"))
        .otherwise(pl.lit("6+")).alias("name_shared_by"))
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    t = F.group_by("country", "address", "name_shared_by").agg(
        pl.col("true").sum().alias("true_pairs"),
        (pl.col("true") * pl.col("sel")).sum().alias("found"),
        pl.col("sel").sum().alias("selected"),
        ((pl.col("true") == 1) & (pl.col("p") < 0.5)).sum().alias("missed_p<0.5"),
    ).with_columns((pl.col("found") / pl.col("true_pairs")).round(4).alias("recall"),
                   (pl.col("found") / pl.col("selected")).round(4).alias("precision")).sort("country", "address", "name_shared_by")
    print("true pairs in the candidate set, by record-address state x S1 name ambiguity (split V):", flush=True)
    print(t, flush=True)


if __name__ == "__main__":
    main()
