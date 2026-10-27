"""Diagnostic: model confidence on test by country (incl. unlabeled France) vs the validation split.

If France is much less confident than US/India (fewer S1s with a clear top match, more pairs in the
uncertain band, more unknown-state pairs), the leaderboard gap is likely a France transfer problem.
"""
import json

import numpy as np
import polars as pl
import xgboost as xgb

from config import WORK
from train import context

COLS = ["s1_id", "rec_id", "state_eq", "ad_tset", "nm_tset", "cos", "rank_rev", "num_conflict"]


def main():
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    dec = json.loads((WORK / "decision.json").read_text())
    m1 = xgb.Booster(model_file=str(WORK / "xgb_stage1.json"))
    m2 = xgb.Booster(model_file=str(WORK / "xgb_stage2.json"))
    for name, split in (("V", "train"), ("test", "test")):
        F = pl.read_parquet(WORK / f"feat_{name}.parquet", columns=list(dict.fromkeys(COLS + dec["cols"])))
        p = m1.inplace_predict(F.select(dec["cols"]).to_numpy().astype(np.float32))
        if dec["stage"] == 2:
            F2 = pl.concat([F, context(F, p)], how="horizontal")
            p = m2.inplace_predict(F2.select(dec["cols2"]).to_numpy().astype(np.float32))
        country = pl.read_parquet(WORK / f"{split}_recs.parquet", columns=["id", "src", "country"]).filter(
            pl.col("src") == 1).select(pl.col("id").alias("s1_id"), "country")
        d = F.select(COLS).with_columns(pl.Series("p", p)).join(country, on="s1_id")
        del F
        per_s1 = d.group_by("s1_id", "country").agg(
            pl.col("p").max().alias("pmax"),
            ((pl.col("p") >= 0.1) & (pl.col("p") < 0.9)).sum().alias("n_uncertain"),
            (pl.col("p") >= 0.5).sum().alias("n_ge50"),
        )
        print(f"\n===== {name}: per-S1 confidence by country")
        print(per_s1.group_by("country").agg(
            pl.len().alias("s1"),
            (pl.col("pmax") >= 0.9).mean().round(4).alias("top>=0.9"),
            ((pl.col("pmax") >= 0.5) & (pl.col("pmax") < 0.9)).mean().round(4).alias("top0.5-0.9"),
            ((pl.col("pmax") >= 0.1) & (pl.col("pmax") < 0.5)).mean().round(4).alias("top0.1-0.5"),
            (pl.col("pmax") < 0.1).mean().round(4).alias("top<0.1"),
            pl.col("n_uncertain").mean().round(3).alias("uncertain_pairs/S1"),
            pl.col("n_ge50").mean().round(3).alias("pairs>=0.5/S1"),
        ).sort("country"), flush=True)
        top = d.sort("p", descending=True).group_by("s1_id", maintain_order=True).first()
        print(f"{name}: features of each S1's best pair")
        print(top.group_by("country").agg(
            (pl.col("state_eq") == -1).mean().round(4).alias("state_unknown"),
            (pl.col("state_eq") == 0).mean().round(4).alias("state_differs"),
            pl.col("ad_tset").median().alias("ad_tset_med"),
            pl.col("nm_tset").median().alias("nm_tset_med"),
            pl.col("cos").median().round(4).alias("cos_med"),
            (pl.col("rank_rev") == 1).mean().round(4).alias("rev_rank1"),
        ).sort("country"), flush=True)
        hi = d.filter(pl.col("p") >= 0.5)
        print(f"{name}: pairs with p>=0.5")
        print(hi.group_by("country").agg(
            pl.len().alias("n"), (pl.col("state_eq") == -1).mean().round(4).alias("state_unknown"),
            pl.col("ad_tset").mean().round(2).alias("ad_tset_mean"), pl.col("p").mean().round(4).alias("p_mean"),
        ).sort("country"), flush=True)


if __name__ == "__main__":
    main()
