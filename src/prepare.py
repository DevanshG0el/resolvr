"""Stage A: load raw TSVs, assign train splits, mine transliteration dictionaries, normalise, save parquet.

Outputs (in WORK):
  train_recs.parquet, test_recs.parquet  normalised records of all three sources
  train_pairs.parquet                    ground-truth (s1_id, rec_id, split) pairs
  dicts.json                             mined transliteration dictionaries
"""
import json
import time

import polars as pl

import normalize as N
from config import DATA, SMOKE, SMOKE_FRACTION, SPLIT_FRACTIONS, WORK

KW = dict(separator="\t", quote_char=None, infer_schema=False)
KEEP_COLS = [
    "id", "src", "country", "split", "name", "address", "name_native", "web_stem", "legal",
    "name_full", "name_core", "name_sorted", "name_nospace", "name_phon", "name_tok", "name_nums",
    "addr_clean", "addr_str", "addr_tok", "addr_loc_tok", "addr_nums", "addr_num1", "state",
    "addr_empty", "emb_text",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def id_num(col="id"):
    return pl.col(col).str.slice(3).cast(pl.Int64)


def split_bucket():
    """0..99, deterministic, independent of smoke_keep()."""
    return (id_num() * 2654435761 // 1048576) % 100


def smoke_keep():
    return (id_num() * 2246822519 // 65536) % 1000 < SMOKE_FRACTION * 1000


def read_source(split: str, s: int) -> pl.DataFrame:
    path = DATA / split / f"{split}_source{s}.tsv"
    df = pl.read_csv(path, **KW).rename(
        {"entity_id": "id", "business_name": "name", "business_address": "address"}
    )
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f) - 1
    assert df.height == n_lines, f"{path}: parsed {df.height} rows but file has {n_lines} lines"
    return df.with_columns(pl.lit(s, pl.Int8).alias("src"), pl.col("country").fill_null("UNKNOWN"))


def assign_split(df: pl.DataFrame) -> pl.DataFrame:
    b = split_bucket()
    expr, lo = None, 0
    for name, frac in SPLIT_FRACTIONS.items():
        hi = lo + round(frac * 100)
        cond = (b >= lo) & (b < hi)
        expr = pl.when(cond).then(pl.lit(name)) if expr is None else expr.when(cond).then(pl.lit(name))
        lo = hi
    return df.with_columns(expr.otherwise(pl.lit("V")).alias("split"))


def load_train():
    s1 = assign_split(read_source("train", 1))
    s2, s3 = read_source("train", 2), read_source("train", 3)
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", **KW)
    pairs = (
        gt.rename({"source1_entity_id": "s1_id"})
        .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids", empty_as_null=True).rename({"matched_entity_ids": "rec_id"})
        .filter(pl.col("rec_id") != "")
        .join(s1.select(pl.col("id").alias("s1_id"), "split"), on="s1_id")
    )
    if SMOKE:
        s1 = s1.filter(smoke_keep())
        pairs = pairs.filter(pl.col("s1_id").is_in(s1["id"].implode()))
        keep = lambda d: d.filter(pl.col("id").is_in(pairs["rec_id"].implode()) | smoke_keep())
        s2, s3 = keep(s2), keep(s3)
    recs = pl.concat([s1, s2.with_columns(pl.lit(None, pl.Utf8).alias("split")),
                      s3.with_columns(pl.lit(None, pl.Utf8).alias("split"))], how="diagonal_relaxed")
    return recs, pairs


def load_test():
    parts = [read_source("test", s) for s in (1, 2, 3)]
    if SMOKE:
        parts = [p.filter(smoke_keep()) for p in parts]
    return pl.concat(parts, how="diagonal_relaxed").with_columns(pl.lit("test").alias("split"))


def mine_dicts(recs: pl.DataFrame, pairs: pl.DataFrame) -> dict:
    """Transliteration dictionaries mined from split-A matches only."""
    a = pairs.filter(pl.col("split") == "A")
    s1 = recs.filter(pl.col("src") == 1)
    s1_state = N.normalize_addresses(s1.select("id", "address", "country"), {}).select(
        pl.col("id").alias("s1_id"),
        pl.struct("country", "state").map_elements(
            lambda r: N.STATE_NAME.get((r["country"], r["state"])), return_dtype=pl.Utf8).alias("s1_state_name"),
    )
    s1 = s1.select(pl.col("id").alias("s1_id"), pl.col("name").alias("s1_name"), pl.col("address").alias("s1_addr"))
    rc = recs.filter(pl.col("src") != 1).select(pl.col("id").alias("rec_id"), "name", "address")
    j = a.join(s1, on="s1_id").join(s1_state, on="s1_id").join(rc, on="rec_id")
    jn = j.filter(pl.col("name").str.contains(N.INDIC_PL))
    ja = j.filter(pl.col("address").str.contains(N.INDIC_PL))
    return {
        "name": N.mine_token_dict(jn.select(pl.col("name").alias("native_text"), pl.col("s1_name").alias("latin_text"))),
        "addr": N.mine_token_dict(ja.select(pl.col("address").alias("native_text"), pl.col("s1_addr").alias("latin_text"))),
        "addr_comp": N.mine_component_dict(ja.select(pl.col("address").alias("native_text"),
                                                     pl.col("s1_state_name").alias("latin_value"))),
    }


def report(recs: pl.DataFrame, pairs: pl.DataFrame, tag: str):
    log(f"-- {tag}: rows by src/country")
    print(recs.group_by("src", "country").agg(
        pl.len().alias("n"), pl.col("name_native").mean().round(3).alias("native"),
        pl.col("state").is_not_null().mean().round(3).alias("state_found"),
        pl.col("addr_empty").mean().round(3).alias("addr_empty"),
        (pl.col("name_core") == "").mean().round(4).alias("core_empty"),
    ).sort("src", "country"), flush=True)
    if pairs is None or pairs.height == 0:
        return
    cols = ["id", "country", "name_core", "name_sorted", "name_phon", "addr_num1", "state", "legal"]
    left = recs.filter(pl.col("src") == 1).select(cols).rename({c: "a_" + c for c in cols})
    right = recs.filter(pl.col("src") != 1).select(cols + ["src", "name_native"]).rename({c: "b_" + c for c in cols})
    j = pairs.join(left, left_on="s1_id", right_on="a_id").join(right, left_on="rec_id", right_on="b_id")
    both = lambda c: pl.col("a_" + c).is_not_null() & pl.col("b_" + c).is_not_null() & (pl.col("a_" + c) != "") & (pl.col("b_" + c) != "")
    agree = lambda c: (pl.col("a_" + c) == pl.col("b_" + c))
    log(f"-- {tag}: agreement between true pairs (cleaning quality)")
    print(j.group_by("a_country", "src", "name_native").agg(
        pl.len().alias("pairs"),
        agree("name_core").mean().round(3).alias("core_eq"),
        agree("name_sorted").mean().round(3).alias("sorted_eq"),
        agree("name_phon").mean().round(3).alias("phon_eq"),
        both("addr_num1").mean().round(3).alias("num_both"),
        agree("addr_num1").filter(both("addr_num1")).mean().round(3).alias("num_eq"),
        both("state").mean().round(3).alias("state_both"),
        agree("state").filter(both("state")).mean().round(3).alias("state_eq"),
        agree("legal").filter(both("legal")).mean().round(3).alias("legal_eq"),
    ).sort("a_country", "src", "name_native"), flush=True)


def main():
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    t0 = time.time()
    log("loading train", "(SMOKE)" if SMOKE else "")
    recs, pairs = load_train()
    log(f"train records {recs.height:,}, pairs {pairs.height:,}")
    dicts = mine_dicts(recs, pairs)
    for tag, d in dicts.items():
        log(f"mined {tag} dictionary: {len(d):,} entries; sample: {list(d.items())[:30]}")
    (WORK / "dicts.json").write_text(json.dumps(dicts, ensure_ascii=False), encoding="utf-8")

    recs = N.normalize(recs, dicts).select(KEEP_COLS)
    log("normalised train")
    recs.write_parquet(WORK / "train_recs.parquet")
    pairs.write_parquet(WORK / "train_pairs.parquet")
    report(recs, pairs.filter(pl.col("split") != "ghost"), "train")
    ex = recs.filter(pl.col("name_native")).select("name", "name_full", "address", "addr_clean").head(12)
    print(ex, flush=True)
    del recs

    log("loading test")
    test = load_test()
    test = N.normalize(test, dicts).select(KEEP_COLS)
    test.write_parquet(WORK / "test_recs.parquet")
    report(test, None, "test")
    print(test.filter(pl.col("country") == "France").select("name", "name_full", "legal", "address", "addr_clean", "state").head(12), flush=True)
    log(f"stage A done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
