"""Stage C5 (optional): record-side fuzzy-namesake features over ALL retrieved S1 candidates.

Why: a record with an empty address can only be placed by its name. The existing namesake counts use exact
core-name equality, which fails for noisy names ("Vision Partnres Corp"), so the matcher cannot tell a
fuzzy-unique name from one shared by several businesses (validation: empty address + no exact S1 name =
15.5k true pairs at 52% recall). For every record, among its retrieved S1 candidates (same rule as the
candidate set: forward rank <= 20 or reverse rank <= 3; all S1 entities, whatever their split, so train
and test values are computed the same way), with fuzz.ratio on core names:
  rec_n_s1_nm90 / rec_n_s1_nm80   number of candidate S1s whose name is >= 90 / >= 80 similar
  rec_nm_rank                     rank of this S1 by name similarity among the record's candidates
  rec_nm_gap                      this S1's similarity minus the record's best
Output: WORK/namefeat_{B,V,test}.parquet (s1_id, rec_id + the 4 features); used when BER_NAMEFEAT=1.
"""
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import WORK

N_CHUNKS = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def compute(recs_file: str, cand_file: str) -> pl.DataFrame:
    names = pl.read_parquet(WORK / recs_file, columns=["id", "name_core"]).with_columns(pl.col("name_core").fill_null(""))
    c = pl.read_parquet(WORK / cand_file, columns=["s1_id", "rec_id", "rank_fwd", "rank_rev"])
    c = c.filter((pl.col("rank_fwd").fill_null(10 ** 6) <= 20) | (pl.col("rank_rev").fill_null(10 ** 6) <= 3)).select("s1_id", "rec_id")
    c = c.with_columns((pl.col("rec_id").hash(seed=0) % N_CHUNKS).alias("_k"))
    parts = []
    for k in range(N_CHUNKS):  # chunked by record, so every record's candidates stay together
        x = c.filter(pl.col("_k") == k).drop("_k")
        x = x.join(names.rename({"id": "s1_id", "name_core": "a"}), on="s1_id").join(
            names.rename({"id": "rec_id", "name_core": "b"}), on="rec_id")
        sim = cpdist(x["a"].to_list(), x["b"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        x = x.select("s1_id", "rec_id").with_columns(pl.Series("nm", sim))
        x = x.with_columns(
            (pl.col("nm") >= 90).sum().over("rec_id").cast(pl.Int16).alias("rec_n_s1_nm90"),
            (pl.col("nm") >= 80).sum().over("rec_id").cast(pl.Int16).alias("rec_n_s1_nm80"),
            pl.col("nm").rank("min", descending=True).over("rec_id").cast(pl.Int16).alias("rec_nm_rank"),
            (pl.col("nm") - pl.col("nm").max().over("rec_id")).alias("rec_nm_gap"),
        ).drop("nm")
        parts.append(x)
        log(f"{cand_file}: chunk {k + 1}/{N_CHUNKS} done ({x.height:,} pairs)")
    return pl.concat(parts)


def main():
    t0 = time.time()
    for recs_file, cand_file, splits in (("train_recs.parquet", "cand_train.parquet", ("B", "V")),
                                         ("test_recs.parquet", "cand_test.parquet", ("test",))):
        allf = compute(recs_file, cand_file)
        for name in splits:
            F = pl.read_parquet(WORK / f"feat_{name}.parquet", columns=["s1_id", "rec_id"])
            out = F.join(allf, on=["s1_id", "rec_id"], how="left")
            out.write_parquet(WORK / f"namefeat_{name}.parquet")
            miss = out["rec_n_s1_nm90"].null_count()
            log(f"namefeat_{name}: {out.height:,} pairs ({miss:,} without a value); "
                f"share of records with >= 2 fuzzy namesakes: {(out['rec_n_s1_nm90'] >= 2).mean():.3f}")
    log(f"stage C5 done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
