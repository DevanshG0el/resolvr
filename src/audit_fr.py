"""French normalisation / decision audit on the test set (no labels: for manual inspection).

Prints samples of French (a) borderline pairs, (b) retrieval-confident pairs the matcher rejects,
(c) S1s with no predicted match and their top candidates, and (d) the most frequent address
components of French records without a recognised state. Uses test_scored_sub7.parquet (the final model).
"""
import polars as pl

from config import WORK


def main():
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(60)
    recs = pl.read_parquet(WORK / "test_recs.parquet", columns=[
        "id", "src", "country", "name", "address", "name_core", "legal", "addr_str", "state", "addr_clean"])
    fr = recs.filter(pl.col("country") == "France")
    sc = pl.read_parquet(WORK / "test_scored_sub7.parquet")
    cand = pl.read_parquet(WORK / "cand_test.parquet", columns=["s1_id", "rec_id", "cos", "rank_fwd", "rank_rev"])
    s1 = fr.filter(pl.col("src") == 1).select(pl.col("id").alias("s1_id"), pl.col("name").alias("s1_name"),
                                              pl.col("address").alias("s1_addr"), pl.col("state").alias("s1_state"))
    rc = fr.filter(pl.col("src") != 1).select(pl.col("id").alias("rec_id"), pl.col("name").alias("rec_name"),
                                              pl.col("address").alias("rec_addr"), pl.col("state").alias("rec_state"))
    d = sc.join(s1, on="s1_id").join(rc, on="rec_id").join(cand, on=["s1_id", "rec_id"], how="left")
    show = ["p", "cos", "rank_fwd", "rank_rev", "s1_name", "rec_name", "s1_addr", "rec_addr", "s1_state", "rec_state"]
    print(f"French scored pairs: {d.height:,}; p>=0.5: {(d['p'] >= 0.5).sum():,}; 0.2<=p<=0.8: "
          f"{((d['p'] >= 0.2) & (d['p'] <= 0.8)).sum():,}", flush=True)
    print("\n=== (a) borderline French pairs (0.2 <= p <= 0.8)")
    print(d.filter((pl.col("p") >= 0.2) & (pl.col("p") <= 0.8)).sample(40, seed=1).select(show).sort("p"))
    print("\n=== (b) retrieval-confident but rejected (cos >= 0.95, p < 0.3)")
    b = d.filter((pl.col("cos") >= 0.95) & (pl.col("p") < 0.3))
    print(f"count: {b.height:,}")
    print(b.sample(min(30, b.height), seed=2).select(show).sort("p"))
    print("\n=== (c) French S1 with no candidate p >= 0.5: top-3 candidates")
    best = d.group_by("s1_id").agg(pl.col("p").max().alias("pmax"))
    empty_ids = best.filter(pl.col("pmax") < 0.5).sample(15, seed=3)["s1_id"]
    print(d.filter(pl.col("s1_id").is_in(empty_ids.implode())).sort(["s1_id", "p"], descending=[False, True])
          .group_by("s1_id", maintain_order=True).head(3).select(["s1_id"] + show))
    print("\n=== (d) most frequent address components of French S2/S3 records without a recognised state")
    comp = (fr.filter((pl.col("src") != 1) & pl.col("state").is_null())
            .select(pl.col("addr_clean").str.split(", ").alias("c")).explode("c")
            .filter(~pl.col("c").str.contains(r"\d")).group_by("c").len().sort("len", descending=True).head(40))
    print(comp)
    print("\n=== (e) French S1 name tokens: most frequent (vocabulary size check)")
    tok = (fr.filter(pl.col("src") == 1).select(pl.col("name_core").str.split(" ").alias("t")).explode("t")
           .group_by("t").len().sort("len", descending=True))
    print(f"distinct core-name tokens in French S1: {tok.height:,}; top 30:")
    print(tok.head(30))
    print("\n=== (f) legal forms: French S1 vs S2/S3 records")
    print(fr.group_by("src", "legal").len().sort(["src", "len"], descending=[False, True]).group_by("src", maintain_order=True).head(8))


if __name__ == "__main__":
    main()
