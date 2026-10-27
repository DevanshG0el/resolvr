"""Exploratory data analysis and data-cleaning statistics for the report.

Reads the raw TSVs (student_resource/dataset) and the normalised parquet files of the pipeline (work/), writes
results/eda.json and results/eda_*.csv:
  - records per split / source / country, missing fields, non-Latin script share, text lengths
  - ground-truth structure: matches per S1 entity, singleton share, S2 vs S3 matches
  - the train split sizes (ghost / A / B / V)
  - cleaning effect: how often true pairs agree on name/number/state keys before vs after normalisation
Usage: python experiments/eda.py
"""
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import DATA, WORK  # noqa: E402

RES = ROOT / "results"
KW = dict(separator="\t", quote_char=None, infer_schema=False)
NON_LATIN = r"[\p{Devanagari}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}\p{Bengali}\p{Gujarati}\p{Gurmukhi}\p{Oriya}\p{Arabic}]"


def source_stats():
    rows = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            df = pl.read_csv(DATA / split / f"{split}_source{s}.tsv", **KW)
            g = df.group_by("country").agg(
                pl.len().alias("records"),
                pl.col("business_name").is_null().mean().alias("name_missing"),
                pl.col("business_address").is_null().mean().alias("address_missing"),
                pl.col("business_name").str.contains(NON_LATIN).fill_null(False).mean().alias("name_non_latin"),
                pl.col("business_address").str.contains(NON_LATIN).fill_null(False).mean().alias("address_non_latin"),
                pl.col("business_name").str.len_chars().mean().alias("name_len"),
                pl.col("business_address").str.len_chars().mean().alias("address_len"),
                pl.col("business_address").str.contains(r"\b\d{6}\b").fill_null(False).mean().alias("has_pin6"),
                pl.col("business_address").str.contains(r"(?i)\bnear\b|\bopp\b|\bbehind\b").fill_null(False).mean().alias("landmark"),
            )
            rows.append(g.with_columns(pl.lit(split).alias("split"), pl.lit(f"S{s}").alias("source")))
            print(split, s, df.height, flush=True)
    return pl.concat(rows).select("split", "source", pl.exclude("split", "source")).sort("split", "source", "country")


def truth_stats():
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", **KW)
    s1 = pl.read_csv(DATA / "train" / "train_source1.tsv", **KW).select(pl.col("entity_id").alias("source1_entity_id"), "country")
    m = gt.join(s1, on="source1_entity_id").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
    m = m.with_columns(pl.col("matched_entity_ids").list.eval(pl.element().filter(pl.element() != "")))
    m = m.with_columns(
        pl.col("matched_entity_ids").list.len().alias("n"),
        pl.col("matched_entity_ids").list.eval(pl.element().str.starts_with("S2")).list.sum().alias("n_s2"),
    ).with_columns((pl.col("n") - pl.col("n_s2")).alias("n_s3"))
    card = (m.with_columns(pl.when(pl.col("n") >= 6).then(pl.lit("6+")).otherwise(pl.col("n").cast(pl.Utf8)).alias("k"))
            .group_by("country", "k").len().sort("country", "k"))
    summary = m.group_by("country").agg(
        pl.len().alias("s1_entities"), (pl.col("n") == 0).mean().alias("singleton_share"),
        pl.col("n").mean().alias("mean_matches"), pl.col("n_s2").mean().alias("mean_s2"), pl.col("n_s3").mean().alias("mean_s3"),
        pl.col("n").max().alias("max_matches"),
    ).sort("country")
    return card, summary


def cleaning_effect():
    """Share of true (S1, record) pairs whose keys agree exactly, before vs after normalisation."""
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    recs = pl.scan_parquet(WORK / "train_recs.parquet").select("id", "src", "country", "name", "address", "name_core",
                                                               "name_sorted", "name_phon", "addr_num1", "state")
    ids = pl.concat([pairs["s1_id"], pairs["rec_id"]]).unique()
    r = recs.filter(pl.col("id").is_in(ids.implode())).collect()
    raw_name = pl.col("name").str.to_lowercase().str.strip_chars()
    r = r.with_columns(raw_name.alias("name_raw"),
                       pl.col("address").str.extract(r"(\d+)").alias("num_raw"))
    a = r.rename(lambda c: "a_" + c)
    b = r.rename(lambda c: "b_" + c)
    j = pairs.join(a, left_on="s1_id", right_on="a_id").join(b, left_on="rec_id", right_on="b_id")
    eq = lambda x, y: ((pl.col(x) == pl.col(y)) & pl.col(x).is_not_null()).mean()
    both = lambda c: pl.col("a_" + c).is_not_null() & pl.col("b_" + c).is_not_null()
    out = j.group_by("a_country").agg(
        pl.len().alias("pairs"),
        eq("a_name_raw", "b_name_raw").alias("name_exact_raw"),
        eq("a_name_core", "b_name_core").alias("name_exact_clean"),
        eq("a_name_sorted", "b_name_sorted").alias("name_sorted_clean"),
        eq("a_name_phon", "b_name_phon").alias("name_phonetic_clean"),
        (pl.col("a_num_raw") == pl.col("b_num_raw")).filter(both("num_raw")).mean().alias("number_eq_raw"),
        (pl.col("a_addr_num1") == pl.col("b_addr_num1")).filter(both("addr_num1")).mean().alias("number_eq_clean"),
        both("state").mean().alias("state_both_known"),
        (pl.col("a_state") == pl.col("b_state")).filter(both("state")).mean().alias("state_eq_clean"),
    ).sort("a_country")
    return out


def split_sizes():
    s1 = pl.scan_parquet(WORK / "train_recs.parquet").filter(pl.col("src") == 1).group_by("split").len().collect()
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").group_by("split").len(name="true_pairs")
    return s1.join(pairs, on="split", how="left").sort("split")


def main():
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    src = source_stats()
    card, summ = truth_stats()
    clean = cleaning_effect()
    splits = split_sizes()
    for name, df in (("sources", src), ("cardinality", card), ("truth", summ), ("cleaning", clean), ("splits", splits)):
        df.write_csv(RES / f"eda_{name}.csv")
        print(name, df, sep="\n", flush=True)
    (RES / "eda.json").write_text(json.dumps({
        "sources": src.to_dicts(), "cardinality": card.to_dicts(), "truth": summ.to_dicts(),
        "cleaning": clean.to_dicts(), "splits": splits.to_dicts(),
    }, indent=1, default=float))


if __name__ == "__main__":
    main()
