"""Write a score file with the rename guard of final.py applied to the countries absent from training.

Used to derive labels from a guarded France route (the fillers stage then applies the filler/swap rule and decides).
BER_GUARD_IN: scores to guard; BER_GUARD_SCORED: reference route; BER_GUARD_OUT: output (file names under WORK).
"""
import os

import polars as pl

import final
from config import WORK


def main():
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
    seen = set(pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list())
    ids = test.filter(~pl.col("country").is_in(list(seen)))["id"]
    sc = pl.read_parquet(WORK / os.environ["BER_GUARD_IN"]).select("s1_id", "rec_id", "p")
    final.guard_renames(sc, ids, str(WORK / os.environ["BER_GUARD_SCORED"])).write_parquet(WORK / os.environ["BER_GUARD_OUT"])


if __name__ == "__main__":
    main()
