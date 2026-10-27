"""Which data conditions drive the errors? Validation split V, current final model and decision rule.

Writes WORK/drivers_<table>.csv (small) and prints them:
  recall_<attr>  true pairs by data condition: share, recall, share of all misses
  fp_<attr>      false merges by condition
  s1_<attr>      per-S1 macro F0.5 by entity-level condition
"""
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import WORK
from train import context, load_feats

ATTRS = ["s1_id", "rec_id", "rec_addr_empty", "nm_tset", "num1_absdiff", "b_unseen_tok_share",
         "rec_native", "a_n_s1_same_name"]


def buckets(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("nm_tset").is_null()).then(pl.lit("not retrieved"))
        .when(pl.col("nm_tset") >= 100).then(pl.lit("1 identical"))
        .when(pl.col("nm_tset") >= 80).then(pl.lit("2 similar 80-99"))
        .when(pl.col("nm_tset") >= 50).then(pl.lit("3 partial 50-79"))
        .otherwise(pl.lit("4 different <50")).alias("name_similarity"),
        pl.when(pl.col("rec_addr_empty").is_null()).then(pl.lit("not retrieved"))
        .when(pl.col("rec_addr_empty") == 1).then(pl.lit("record address empty"))
        .otherwise(pl.lit("address present")).alias("address"),
        pl.when(pl.col("nm_tset").is_null()).then(pl.lit("not retrieved"))
        .when(pl.col("num1_absdiff").is_null()).then(pl.lit("number missing on a side"))
        .when(pl.col("num1_absdiff") == 0).then(pl.lit("1 equal"))
        .when(pl.col("num1_absdiff") <= 2).then(pl.lit("2 shifted by 1-2"))
        .otherwise(pl.lit("3 differs by >2")).alias("house_number"),
        pl.when(pl.col("nm_tset").is_null()).then(pl.lit("not retrieved"))
        .when((pl.col("b_unseen_tok_share") >= 0.99) & (pl.col("nm_tset") < 50)).then(pl.lit("gibberish name"))
        .otherwise(pl.lit("normal name")).alias("name_kind"),
        pl.when(pl.col("rec_native").is_null()).then(pl.lit("not retrieved"))
        .when(pl.col("rec_native") == 1).then(pl.lit("native script"))
        .otherwise(pl.lit("latin script")).alias("script"),
        pl.col("rec_id").str.slice(0, 2).alias("source"),
    )


def recall_table(tp: pl.DataFrame, col: str) -> pl.DataFrame:
    n_miss = int((tp["found"] == 0).sum())
    return (tp.group_by(col).agg(pl.len().alias("true_pairs"), pl.col("found").mean().alias("recall"),
                                 (pl.col("found") == 0).sum().alias("misses"))
            .with_columns((pl.col("true_pairs") / tp.height).alias("share_of_pairs"),
                          (pl.col("misses") / max(1, n_miss)).alias("share_of_misses"))
            .sort(col))


def country_check():
    """Exact check over ALL ground-truth pairs: does a true match ever cross countries?"""
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").select("s1_id", "rec_id")
    c = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "country"])
    j = (pairs.join(c.rename({"id": "s1_id", "country": "s1_country"}), on="s1_id", how="left")
         .join(c.rename({"id": "rec_id", "country": "rec_country"}), on="rec_id", how="left"))
    unresolved = j.filter(pl.col("s1_country").is_null() | pl.col("rec_country").is_null()).height
    t = j.group_by("s1_country", "rec_country").len(name="pairs").sort("s1_country", "rec_country")
    cross = int(t.filter(pl.col("s1_country") != pl.col("rec_country"))["pairs"].sum())
    print(f"\n== country check over ALL ground-truth pairs: {j.height:,} pairs, "
          f"{cross:,} cross-country, {unresolved:,} with an unknown id")
    print(t, flush=True)
    t.write_csv(WORK / "drivers_country_check.csv")


def main():
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(40)
    country_check()
    sfx = os.environ.get("BER_DRIVERS_SFX", "")  # e.g. "_sub11": analyse a saved model instead of the working copy
    dec = json.loads((WORK / f"decision{sfx}.json").read_text())
    V = load_feats("V", dec.get("use_xenc", False), dec.get("xenc_tags", [""]), namefeat=dec.get("namefeat", False))
    m1 = xgb.Booster(model_file=str(WORK / f"xgb_stage1{sfx}.json"))
    p = m1.inplace_predict(V.select(dec["cols"]).to_numpy().astype(np.float32))
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / f"xgb_stage2{sfx}.json"))
        p = m2.inplace_predict(pl.concat([V, context(V, p)], how="horizontal")
                               .select(dec["cols2"]).to_numpy().astype(np.float32))
    cfg = dec["cfg"]
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    if cfg["rule"] == "threshold":
        sel = D.select_threshold(ex, "p", cfg["thr"])
    else:
        ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
        sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"])
    sel = sel.with_columns(pl.lit(1, pl.Int8).alias("pred"))
    attrs = V.select(ATTRS)
    del V

    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    truth = pairs.filter(pl.col("split") == "V").select("s1_id", "rec_id")
    s1c = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"]).filter(
        (pl.col("src") == 1) & (pl.col("split") == "V")).select(pl.col("id").alias("s1_id"), "country")

    tp = buckets(truth.join(attrs, on=["s1_id", "rec_id"], how="left")
                 .join(sel, on=["s1_id", "rec_id"], how="left")
                 .with_columns(pl.col("pred").fill_null(0).alias("found")).join(s1c, on="s1_id"))
    for col in ("address", "name_similarity", "house_number", "name_kind", "script", "source", "country"):
        t = recall_table(tp, col)
        print(f"\n== true pairs by {col}")
        print(t, flush=True)
        t.write_csv(WORK / f"drivers_recall_{col}.csv")
    combo = recall_table(tp.with_columns((pl.col("address") + " | " + pl.col("name_similarity")).alias("addr_x_name")),
                         "addr_x_name")
    print("\n== true pairs by address x name similarity")
    print(combo, flush=True)
    combo.write_csv(WORK / "drivers_recall_addr_x_name.csv")

    owner = pairs.select("rec_id", pl.col("split").alias("owner_split"))
    fp = buckets(sel.join(truth, on=["s1_id", "rec_id"], how="anti").join(attrs, on=["s1_id", "rec_id"], how="left")
                 .join(owner, on="rec_id", how="left"))
    fp = fp.with_columns(pl.col("owner_split").fill_null("orphan (no S1)").alias("record_belongs_to"))
    for col in ("record_belongs_to", "name_similarity", "house_number", "address"):
        t = fp.group_by(col).len(name="false_merges").with_columns(
            (pl.col("false_merges") / fp.height).alias("share")).sort("false_merges", descending=True)
        print(f"\n== false merges by {col} (total {fp.height:,})")
        print(t, flush=True)
        t.write_csv(WORK / f"drivers_fp_{col}.csv")

    # entity-level: macro F0.5 by number of true matches and by name ambiguity
    n_true = truth.group_by("s1_id").len(name="n_true")
    amb = attrs.group_by("s1_id").agg(pl.col("a_n_s1_same_name").max().alias("same_name_s1"))
    ent = s1c.join(n_true, on="s1_id", how="left").join(amb, on="s1_id", how="left").with_columns(
        pl.col("n_true").fill_null(0),
        pl.when(pl.col("same_name_s1").fill_null(1) > 1).then(pl.lit("name shared with other S1"))
        .otherwise(pl.lit("unique name")).alias("name_ambiguity"),
    ).with_columns(pl.when(pl.col("n_true") >= 7).then(pl.lit("7+")).otherwise(pl.col("n_true").cast(pl.Utf8))
                   .alias("true_matches"))
    for col in ("true_matches", "name_ambiguity", "country"):
        rows = []
        for key in ent[col].unique().sort().to_list():
            ids = ent.filter(pl.col(col) == key)["s1_id"]
            m = D.macro_f05(sel.filter(pl.col("s1_id").is_in(ids.implode())).select("s1_id", "rec_id"),
                            truth.filter(pl.col("s1_id").is_in(ids.implode())), ids)
            rows.append({col: key, "s1": len(ids), "share_of_s1": len(ids) / ent.height, "f05": m["f05"],
                         "lost_f05_points": (1 - m["f05"]) * len(ids) / ent.height})
        t = pl.DataFrame(rows)
        print(f"\n== per-entity F0.5 by {col}")
        print(t, flush=True)
        t.write_csv(WORK / f"drivers_s1_{col}.csv")
    m = D.macro_f05(sel.select("s1_id", "rec_id"), truth, ent["s1_id"])
    print("\noverall:", m, flush=True)


if __name__ == "__main__":
    main()
