"""France vs US/India consistency check (no French labels needed).

The data generator is the same for every country (only the vocabulary changes), so within one pair type
(name relation x address similarity x house number) the true-match rate should be similar across
countries. Per bucket this compares
  V (US/India, labelled): true-match rate, and the rate the #7 matcher selects (with its precision/recall);
  France (test):          the rate #9 selects.
implied = France pairs x (France selected rate - V true rate): > 0 France over-matches this pair type
(likely false merges), < 0 France under-matches (likely misses).
Inputs: WORK/qwen_selection.parquet (p of the #7 matcher on every candidate pair; for test it carries #9's
French cross-encoder scores), feat_V / feat_test, decision_sub7.json. Output: WORK/fr_consistency.csv.
"""
import json

import polars as pl

import decide as D
from config import WORK

COLS = ["s1_id", "rec_id", "nm_exact", "nm_tset", "nm_ratio", "ad_tset", "rec_addr_empty", "num1_eq"]


def bucket(F: pl.DataFrame) -> pl.DataFrame:
    return F.with_columns(
        pl.when(pl.col("nm_exact") == 1).then(pl.lit("1 exact"))
        .when(pl.col("nm_tset") >= 100).then(pl.lit("2 subset"))
        .when(pl.col("nm_ratio") >= 90).then(pl.lit("3 r90"))
        .when(pl.col("nm_ratio") >= 75).then(pl.lit("4 r75")).otherwise(pl.lit("5 low")).alias("name"),
        pl.when(pl.col("rec_addr_empty") == 1).then(pl.lit("0 empty"))
        .when(pl.col("ad_tset") >= 95).then(pl.lit("1 a95"))
        .when(pl.col("ad_tset") >= 80).then(pl.lit("2 a80"))
        .when(pl.col("ad_tset") >= 60).then(pl.lit("3 a60")).otherwise(pl.lit("4 low")).alias("addr"),
        pl.when(pl.col("num1_eq") == 1).then(pl.lit("eq")).when(pl.col("num1_eq") == 0).then(pl.lit("diff"))
        .otherwise(pl.lit("na")).alias("num"),
    )


def selected(sc: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    ex = D.exclusive(sc.select("s1_id", "rec_id", "p"))
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    return D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"], device="cpu").select("s1_id", "rec_id").with_columns(
        pl.lit(1, pl.Int8).alias("sel"))


def main():
    cfg = json.loads((WORK / "decision_sub7.json").read_text())["cfg"]
    sel = pl.read_parquet(WORK / "qwen_selection.parquet")
    recs = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
    fr = recs.filter(pl.col("country") == "France")["id"]

    v = sel.filter(pl.col("split") == "V")
    v = v.join(selected(v, cfg), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("sel").fill_null(0))
    v = bucket(v.join(pl.read_parquet(WORK / "feat_V.parquet", columns=COLS), on=["s1_id", "rec_id"]))
    t = sel.filter((pl.col("split") == "test") & pl.col("s1_id").is_in(fr.implode()))
    t = t.join(selected(t, cfg), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("sel").fill_null(0))
    t = bucket(t.join(pl.read_parquet(WORK / "feat_test.parquet", columns=COLS), on=["s1_id", "rec_id"]))

    keys = ["name", "addr", "num"]
    gv = v.group_by(keys).agg(
        pl.len().alias("v_pairs"), pl.col("label").mean().alias("v_true"), pl.col("sel").cast(pl.Float64).mean().alias("v_sel"),
        (pl.col("label").cast(pl.Int32) * pl.col("sel")).sum().alias("_tp"), pl.col("sel").cast(pl.Int32).sum().alias("_ns"),
        pl.col("label").cast(pl.Int32).sum().alias("_np"))
    gv = gv.with_columns((pl.col("_tp") / pl.col("_ns")).alias("v_prec"), (pl.col("_tp") / pl.col("_np")).alias("v_rec")).drop("_tp", "_ns", "_np")
    gt = t.group_by(keys).agg(pl.len().alias("fr_pairs"), pl.col("sel").cast(pl.Float64).mean().alias("fr_sel"))
    g = gt.join(gv, on=keys, how="left").with_columns(
        (pl.col("fr_pairs") * (pl.col("fr_sel") - pl.col("v_true"))).round(0).alias("implied"))
    g.write_csv(WORK / "fr_consistency.csv")
    n_fr = t.height
    print(f"France: {n_fr:,} candidate pairs, {int(t['sel'].sum()):,} selected by #9 ({t['sel'].mean():.4f}); "
          f"V: {v.height:,} pairs, true rate {v['label'].mean():.4f}, selected {v['sel'].mean():.4f}")
    print(f"implied over-matching (sum of implied > 0): {g.filter(pl.col('implied') > 0)['implied'].sum():,.0f} pairs; "
          f"under-matching (sum < 0): {g.filter(pl.col('implied') < 0)['implied'].sum():,.0f} pairs")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    out = g.with_columns(pl.col("fr_sel", "v_true", "v_sel", "v_prec", "v_rec").round(3)).sort(
        pl.col("implied").abs(), descending=True)
    print(out.head(30))
    print("\nby name relation only:")
    print(t.group_by("name").agg(pl.len().alias("fr_pairs"), pl.col("sel").mean().round(3).alias("fr_sel")).join(
        v.group_by("name").agg(pl.len().alias("v_pairs"), pl.col("label").mean().round(3).alias("v_true"),
                               pl.col("sel").mean().round(3).alias("v_sel")), on="name").sort("name"))
    print("\nby address relation only:")
    print(t.group_by("addr").agg(pl.len().alias("fr_pairs"), pl.col("sel").mean().round(3).alias("fr_sel")).join(
        v.group_by("addr").agg(pl.len().alias("v_pairs"), pl.col("label").mean().round(3).alias("v_true"),
                               pl.col("sel").mean().round(3).alias("v_sel")), on="addr").sort("addr"))


if __name__ == "__main__":
    main()
