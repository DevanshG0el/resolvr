"""Diagnostic: records with an EMPTY address whose name matches no S1 name exactly (split V, US/India).

Are the missed ones fixable? For each candidate pair of such a record:
  own similarity (fuzz.ratio of core names), how many retrieved S1 candidates have a name >= 80 similar
  (rec_n_s1_nm80, from namefeat.py), and whether this S1 is the record's closest name (rec_nm_rank == 1).
Reports recall/precision of the saved matcher per group and the real-match rate of the would-be rule
"closest name, fuzzy-unique, similarity >= t" (its precision if we forced it). Model: BER_EN_SFX (default _mxc).
"""
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import WORK
from train import context, load_feats

SFX = os.environ.get("BER_EN_SFX", "_mxc")


def main():
    dec = json.loads((WORK / f"decision{SFX}.json").read_text())
    V = load_feats("V", dec.get("use_xenc", False), dec.get("xenc_tags", [""]), namefeat=True)
    m1 = xgb.Booster(model_file=str(WORK / f"xgb_stage1{SFX}.json"))
    Xc = V.select(dec["cols"]).to_numpy().astype(np.float32)
    p = m1.inplace_predict(Xc)
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / f"xgb_stage2{SFX}.json"))
        p = m2.inplace_predict(pl.concat([V, context(V, p)], how="horizontal").select(dec["cols2"]).to_numpy().astype(np.float32))
    cfg = dec["cfg"]
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"]).with_columns(pl.lit(1, pl.Int8).alias("sel"))
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id").with_columns(pl.lit(1, pl.Int8).alias("true"))
    F = V.select("s1_id", "rec_id", "rec_addr_empty", "b_n_s1_same_name", "nm_ratio", "rec_n_s1_nm80", "rec_n_s1_nm90", "rec_nm_rank").with_columns(pl.Series("p", p))
    F = F.filter((pl.col("rec_addr_empty") == 1) & (pl.col("b_n_s1_same_name") == 0))
    F = F.join(sel.select("s1_id", "rec_id", "sel"), on=["s1_id", "rec_id"], how="left").join(truth, on=["s1_id", "rec_id"], how="left")
    F = F.with_columns(pl.col("sel").fill_null(0), pl.col("true").fill_null(0))
    F = F.with_columns(
        pl.when(pl.col("nm_ratio") >= 95).then(pl.lit("a >=95")).when(pl.col("nm_ratio") >= 90).then(pl.lit("b 90-95"))
        .when(pl.col("nm_ratio") >= 80).then(pl.lit("c 80-90")).otherwise(pl.lit("d <80")).alias("own_sim"),
        pl.when(pl.col("rec_n_s1_nm80") <= 1).then(pl.lit("1 fuzzy-unique")).when(pl.col("rec_n_s1_nm80") == 2).then(pl.lit("2"))
        .otherwise(pl.lit("3+")).alias("look_alikes"),
        (pl.col("rec_nm_rank") == 1).alias("closest"))
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    tp = F.filter(pl.col("true") == 1)
    print(f"cell: {F.height:,} candidate pairs, {tp.height:,} true pairs, recall {tp['sel'].mean():.4f}, "
          f"precision {F.filter(pl.col('sel') == 1)['true'].mean():.4f}", flush=True)
    print("\ntrue pairs by own name similarity x look-alike S1s x closest name:", flush=True)
    print(tp.group_by("own_sim", "look_alikes", "closest").agg(
        pl.len().alias("true_pairs"), pl.col("sel").mean().round(3).alias("recall"),
        (1 - pl.col("sel")).sum().alias("missed"), pl.col("p").median().round(3).alias("median_p")).sort("missed", descending=True), flush=True)
    print("\nwould-be rule 'closest name AND fuzzy-unique AND own similarity >= t' over ALL candidate pairs of the cell:", flush=True)
    for t in (95, 90, 85, 80):
        r = F.filter(pl.col("closest") & (pl.col("look_alikes") == "1 fuzzy-unique") & (pl.col("nm_ratio") >= t))
        print(f"  t={t}: {r.height:,} pairs, real-match rate {r['true'].mean():.4f}, already selected {r['sel'].mean():.4f}, "
              f"new true pairs if forced {int(((r['true'] == 1) & (r['sel'] == 0)).sum()):,}, "
              f"new false pairs if forced {int(((r['true'] == 0) & (r['sel'] == 0)).sum()):,}", flush=True)
    names = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "name"])
    miss = tp.filter(pl.col("sel") == 0).sample(min(12, tp.filter(pl.col("sel") == 0).height), seed=3)
    miss = miss.join(names.rename({"id": "s1_id", "name": "s1_name"}), on="s1_id").join(names.rename({"id": "rec_id", "name": "rec_name"}), on="rec_id")
    print("\nexamples of missed true pairs:", flush=True)
    for r in miss.iter_rows(named=True):
        print(f"  S1: {r['s1_name']}  <->  RC: {r['rec_name']}  (sim {r['nm_ratio']:.0f}, look-alikes {r['rec_n_s1_nm80']}, closest {r['closest']}, p {r['p']:.3f})", flush=True)


if __name__ == "__main__":
    main()
