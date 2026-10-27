"""How often do NON-matching look-alikes keep the house number? (labelled US/India validation pairs)

For candidate pairs on the same street (address token-set >= 90, non-conflicting legal form) whose record
name is the S1 name with words added ("add") or one word swapped ("swap"), counts by label x same/different
first house number. k_true = P(same number | real match), k_false = P(same number | not a match).
This is the constant the filler/swap rule needs to decide whether same-number swaps in an unseen country are
real matches (k_false small) or decoys (k_false large).
"""
import polars as pl

import fillers as FL
from config import WORK


def main():
    sel = pl.read_parquet(WORK / "qwen_selection.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id", "p", "label")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"])
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    names = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "name", "address"])
    for c in ("US", "India"):
        ids = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V") & (pl.col("country") == c))["id"]
        P = FL.pairs("train", sel, ids).filter(pl.col("w").is_not_null() & (pl.col("legal_eq") != 0) & (pl.col("ad_tset") >= 90)
                                              & pl.col("num1_eq").is_in([0, 1]))
        P = P.with_columns(pl.col("w").str.extract(r"^(\w+):", 1).alias("kind"),
                           pl.len().over("w").alias("w_count"))
        P = P.with_columns(pl.when(pl.col("w_count") >= 50).then(pl.lit("frequent word")).otherwise(pl.lit("rare word")).alias("wfreq"))
        g = P.group_by("kind", "wfreq", "label").agg(pl.len().alias("pairs"), (pl.col("num1_eq") == 1).mean().round(4).alias("share_same_number"))
        print(f"\n===== {c}: same-street add/swap pairs by label (share_same_number = k)", flush=True)
        print(g.sort("kind", "wfreq", "label"), flush=True)
        f = P.filter((pl.col("kind") == "swap") & (pl.col("label") == 0))
        ex = f.join(names.rename({"id": "s1_id", "name": "s_name", "address": "s_addr"}), on="s1_id").join(
            names.rename({"id": "rec_id", "name": "r_name", "address": "r_addr"}), on="rec_id")
        for num in (1, 0):
            e = ex.filter(pl.col("num1_eq") == num)
            print(f"--- {c}: NON-matching swaps, {'same' if num else 'different'} number ({e.height:,}); examples:", flush=True)
            for r in e.sample(min(6, e.height), seed=1).iter_rows(named=True):
                print(f"  S1: {r['s_name']} | {r['s_addr']}\n  RC: {r['r_name']} | {r['r_addr']}", flush=True)


if __name__ == "__main__":
    main()
