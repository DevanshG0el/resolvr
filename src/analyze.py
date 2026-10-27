"""Error analysis on validation split V with the saved models and decision rule.

Prints: error-type breakdown (misses: not retrieved / scored low / lost to exclusivity;
false merges: record of another S1 / orphan record / on a singleton S1), by country and source,
plus raw-text examples of each type. Writes WORK/valid_scored.parquet.
"""
import json

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import WORK
from train import context, load_feats


def main():
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_tbl_rows(40)
    dec = json.loads((WORK / "decision.json").read_text())
    V = load_feats("V", dec.get("use_xenc", False), dec.get("xenc_tags", [""]))
    m1 = xgb.Booster(model_file=str(WORK / "xgb_stage1.json"))
    p = m1.inplace_predict(V.select(dec["cols"]).to_numpy().astype(np.float32))
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / "xgb_stage2.json"))
        V2 = pl.concat([V, context(V, p)], how="horizontal")
        p = m2.inplace_predict(V2.select(dec["cols2"]).to_numpy().astype(np.float32))
    cfg = dec["cfg"]
    sc = V.select("s1_id", "rec_id", "label", "cos", "rank_fwd", "rank_rev", "nm_tset", "ad_tset").with_columns(pl.Series("p", p))
    sc.write_parquet(WORK / "valid_scored.parquet")
    ex = D.exclusive(sc.select("s1_id", "rec_id", "p"))
    if cfg["rule"] == "threshold":
        sel = D.select_threshold(ex, "p", cfg["thr"])
    else:
        ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
        sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"])
    sel = sel.with_columns(pl.lit(1).alias("pred"))

    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    truth = pairs.filter(pl.col("split") == "V").select("s1_id", "rec_id")
    owner = pairs.select(pl.col("rec_id"), pl.col("s1_id").alias("true_s1"), pl.col("split").alias("true_split"))
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "country", "name", "address", "name_native"])
    n_true = truth.group_by("s1_id").len(name="n_true")

    # misses
    fn = truth.join(sel, on=["s1_id", "rec_id"], how="anti")
    fn = fn.join(sc.select("s1_id", "rec_id", "p", "rank_fwd", "rank_rev"), on=["s1_id", "rec_id"], how="left")
    kept_for = ex.select("rec_id", pl.col("s1_id").alias("kept_s1"))
    fn = fn.join(kept_for, on="rec_id", how="left").with_columns(
        pl.when(pl.col("p").is_null()).then(pl.lit("not_retrieved"))
        .when(pl.col("kept_s1") != pl.col("s1_id")).then(pl.lit("lost_exclusivity"))
        .otherwise(pl.lit("scored_low")).alias("type"))
    # false merges
    fp = sel.join(truth, on=["s1_id", "rec_id"], how="anti").join(owner, on="rec_id", how="left")
    fp = fp.join(n_true, on="s1_id", how="left").with_columns(
        pl.when(pl.col("n_true").is_null()).then(pl.lit("on_singleton_s1"))
        .when(pl.col("true_s1").is_null()).then(pl.lit("orphan_record"))
        .otherwise(pl.concat_str(pl.lit("record_of_"), pl.col("true_split"))).alias("type"))
    fp = fp.join(sc.select("s1_id", "rec_id", "p"), on=["s1_id", "rec_id"], how="left")

    info = recs.select("id", "src", "country", "name_native")
    for tag, df in (("MISSES", fn), ("FALSE MERGES", fp)):
        d = df.join(info, left_on="rec_id", right_on="id", how="left")
        print(f"\n===== {tag}: {df.height:,}")
        print(d.group_by("type").len().sort("len", descending=True))
        print(d.group_by("country", "src", "name_native", "type").len().sort("len", descending=True).head(20))

    txt = recs.select("id", pl.concat_str([pl.col("name"), pl.lit(" | "), pl.col("address").fill_null("")]).alias("t"))
    show = lambda df, extra: (
        df.join(txt.rename({"id": "s1_id", "t": "s1"}), on="s1_id")
        .join(txt.rename({"id": "rec_id", "t": "rec"}), on="rec_id")
        .select(["type", "p"] + extra + ["s1", "rec"])
    )
    for t in fn["type"].unique().sort():
        print(f"\n--- miss examples: {t}")
        print(show(fn.filter(pl.col("type") == t).sample(min(15, fn.filter(pl.col("type") == t).height), seed=1),
                   ["rank_fwd", "rank_rev"]))
    for t in fp["type"].unique().sort():
        d = fp.filter(pl.col("type") == t).sample(min(15, fp.filter(pl.col("type") == t).height), seed=1)
        d = show(d, ["true_s1"]).join(txt.rename({"id": "true_s1", "t": "true_s1_text"}), on="true_s1", how="left")
        print(f"\n--- false-merge examples: {t}")
        print(d)
    print("\np distribution of misses scored low:", fn.filter(pl.col("type") == "scored_low")["p"].describe())


if __name__ == "__main__":
    main()
