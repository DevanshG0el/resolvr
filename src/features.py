"""Stage C: pairwise features for every candidate pair (the matcher's input set).

Candidate set = retrieval pairs with rank_fwd <= KF or rank_rev <= KR (this is exactly what
candidate_pairs.tsv contains for test).

Feature groups
  retrieval : cosine, forward/reverse ranks, gaps to the best S1 / best record
  name      : rapidfuzz ratio / token_set / token_sort / partial / Jaro-Winkler on core, full,
              no-space and phonetic names; token Jaccard; rarest-token containment; web-stem match;
              legal-form agreement / conflict; numbers inside names
  address   : rapidfuzz on normalised address; house-number set overlap / conflict; first number;
              state agreement (-1 unknown); locality-token Jaccard; missing-address flags
  ambiguity : how many S1 / S2-S3 records share this exact core name (per state)
  group     : per-S1 max and rank of the main similarities (how this candidate compares to the
              S1's other candidates)
"""
import os
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from config import WORK

KF = int(os.environ.get("BER_KF", 20))
KR = int(os.environ.get("BER_KR", 3))
CHUNK = 4_000_000

REC_COLS = [
    "id", "src", "country", "state", "name_native", "web_stem", "legal", "name_full", "name_core",
    "name_sorted", "name_nospace", "name_phon", "name_tok", "name_nums", "addr_str", "addr_nums",
    "addr_num1", "addr_loc_tok", "addr_empty",
]
STR_SPECS = {
    "nm_ratio": ("name_core", fuzz.ratio),
    "nm_tset": ("name_core", fuzz.token_set_ratio),
    "nm_tsort": ("name_core", fuzz.token_sort_ratio),
    "nm_partial": ("name_core", fuzz.partial_ratio),
    "nm_jw": ("name_core", JaroWinkler.normalized_similarity),
    "nm_full_tset": ("name_full", fuzz.token_set_ratio),
    "nm_full_ratio": ("name_full", fuzz.ratio),
    "nm_nospace": ("name_nospace", fuzz.ratio),
    "nm_nospace_partial": ("name_nospace", fuzz.partial_ratio),
    "nm_phon": ("name_phon", fuzz.ratio),
    "ad_ratio": ("addr_str", fuzz.ratio),
    "ad_tset": ("addr_str", fuzz.token_set_ratio),
    "ad_tsort": ("addr_str", fuzz.token_sort_ratio),
    "ad_partial": ("addr_str", fuzz.partial_ratio),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def record_stats(recs: pl.DataFrame) -> pl.DataFrame:
    """Unsupervised per-record statistics: rarest name token and name-frequency counts."""
    toks = recs.select("id", pl.col("name_tok")).explode("name_tok").drop_nulls()
    df = toks.group_by("name_tok").len(name="df")
    rare = (
        toks.join(df, on="name_tok")
        .sort(["id", "df", "name_tok"])
        .group_by("id", maintain_order=True).agg(
            pl.col("name_tok").first().alias("rare_tok"),
            pl.col("df").first().log1p().alias("rare_logdf"),
            # most common token: tiny for gibberish names made only of never-seen words
            pl.col("df").max().log1p().alias("common_logdf"),
            (pl.col("df") <= 2).mean().alias("unseen_tok_share"),
        )
    )
    is1 = pl.col("src") == 1
    key = ["country", "name_sorted"]
    n_s1 = recs.filter(is1).group_by(key).len(name="n_s1_same_name")
    n_rc = recs.filter(~is1).group_by(key).len(name="n_rec_same_name")
    n_s1_st = recs.filter(is1).group_by(key + ["state"]).len(name="n_s1_same_name_state")
    out = (
        recs.select("id", *key, "state").join(rare, on="id", how="left")
        .join(n_s1, on=key, how="left").join(n_rc, on=key, how="left")
        .join(n_s1_st, on=key + ["state"], how="left")
        .with_columns(pl.col("n_s1_same_name", "n_rec_same_name", "n_s1_same_name_state").fill_null(0))
    )
    return out.drop(*key, "state")


def select_candidates(cand: pl.DataFrame) -> pl.DataFrame:
    """Retrieval cut (fwd <= KF or rev <= KR), then the learned blocking filter if block.json exists:
    keep pairs whose blocker probability q (retrieval signals only) is >= the chosen threshold."""
    cand = cand.filter((pl.col("rank_fwd") <= KF) | (pl.col("rank_rev") <= KR))
    cfg_path = WORK / "block.json"
    if os.environ.get("BER_BLOCK", "1") == "0" or not cfg_path.exists():
        return cand
    import json
    import xgboost as xgb
    from block import blocker_frame
    def q_of(model_file, cols):
        bst = xgb.Booster(model_file=str(WORK / model_file))
        best = bst.attr("best_iteration")  # use the same trees as the study that chose the threshold
        return bst.inplace_predict(fr.select(cols).to_numpy().astype(np.float32),
                                   iteration_range=(0, int(best) + 1) if best is not None else (0, 0))

    cfg = json.loads(cfg_path.read_text())
    fr = blocker_frame(cand)  # neighbourhood signals are computed on the full retrieval candidate list
    keep = q_of("xgb_blocker.json", cfg["cols"]) >= cfg["threshold"]
    n0 = cand.height
    log(f"learned blocking filter q >= {cfg['threshold']}: {n0:,} -> {int(keep.sum()):,} pairs")
    cfg2_path = WORK / "block2.json"
    if os.environ.get("BER_BLOCK2", "1") == "1" and cfg2_path.exists():
        cfg2 = json.loads(cfg2_path.read_text())
        keep &= q_of("xgb_blocker2.json", cfg2["cols"]) >= cfg2["threshold"]
        log(f"second pruning filter q2 >= {cfg2['threshold']:.4f}: -> {int(keep.sum()):,} pairs")
    return cand.filter(pl.Series(keep))


def pair_frame(cand: pl.DataFrame, recs: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    r = recs.join(stats, on="id", how="left")
    a = r.filter(pl.col("src") == 1).drop("src").rename(lambda c: "a_" + c)
    b = r.filter(pl.col("src") != 1).rename(lambda c: "b_" + c)
    return cand.join(a, left_on="s1_id", right_on="a_id").join(b, left_on="rec_id", right_on="b_id")


def jacc(x, y):
    inter = pl.col(x).list.set_intersection(pl.col(y)).list.len()
    union = pl.col(x).list.set_union(pl.col(y)).list.len()
    return inter, pl.when(union > 0).then(inter / union).otherwise(-1.0)


def number_proximity(df: pl.DataFrame) -> pl.DataFrame:
    """House-number closeness. True matches sometimes shift a number by 1-2; decoy businesses on
    the same street differ by more, so 'equal / not equal' alone cannot separate them."""
    to_i = lambda c: pl.col(c).cast(pl.Int64, strict=False)
    d1 = (to_i("a_addr_num1") - to_i("b_addr_num1")).abs()
    first = df.select(
        d1.clip(upper_bound=100000).alias("num1_absdiff"),
        (d1 / pl.max_horizontal(to_i("a_addr_num1"), to_i("b_addr_num1")).clip(lower_bound=1)).alias("num1_reldiff"),
    )
    ex = (
        df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"), "a_addr_nums", "b_addr_nums")
        .explode("a_addr_nums").explode("b_addr_nums")
        .with_columns((to_i("a_addr_nums") - to_i("b_addr_nums")).abs().alias("d"))
        .group_by("row").agg(
            pl.col("d").min().clip(upper_bound=100000).alias("nums_min_absdiff"),
            (pl.col("d") <= 2).sum().alias("nums_n_close"),
        )
    )
    allrows = pl.DataFrame({"row": pl.int_range(df.height, dtype=pl.UInt32, eager=True)})
    close = allrows.join(ex, on="row", how="left").sort("row").drop("row")
    return pl.concat([first, close], how="horizontal")


def features_chunk(df: pl.DataFrame) -> pl.DataFrame:
    out = {}
    for name, (col, scorer) in STR_SPECS.items():
        a = df["a_" + col].fill_null("").to_list()
        b = df["b_" + col].fill_null("").to_list()
        out[name] = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    web = cpdist(df["b_web_stem"].fill_null("").to_list(), df["a_name_nospace"].fill_null("").to_list(),
                 scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    feats = df.select("s1_id", "rec_id").with_columns([pl.Series(k, v) for k, v in out.items()])
    feats = feats.with_columns(pl.Series("web_ratio", web))

    tok_i, tok_j = jacc("a_name_tok", "b_name_tok")
    num_i, num_j = jacc("a_addr_nums", "b_addr_nums")
    loc_i, loc_j = jacc("a_addr_loc_tok", "b_addr_loc_tok")
    nn_i, _ = jacc("a_name_nums", "b_name_nums")
    both_nums = (pl.col("a_addr_nums").list.len() > 0) & (pl.col("b_addr_nums").list.len() > 0)
    both_legal = (pl.col("a_legal") != "") & (pl.col("b_legal") != "")
    legal_i = pl.col("a_legal").str.split(" ").list.set_intersection(pl.col("b_legal").str.split(" ")).list.len()
    sem = df.select(
        pl.col("cos"), pl.col("rank_fwd").cast(pl.Int16), pl.col("rank_rev").cast(pl.Int16),
        (pl.col("s1_top1_cos") - pl.col("cos")).alias("gap_s1_top"),
        (pl.col("rec_top1_cos") - pl.col("cos")).fill_null(0).alias("gap_rec_top"),
        (pl.col("s1_top1_cos") - pl.col("s1_top2_cos")).alias("s1_top_margin"),
        (pl.col("rec_top1_cos") - pl.col("rec_top2_cos")).alias("rec_top_margin"),
        pl.col("b_src").cast(pl.Int8).alias("src"),
        pl.col("b_name_native").cast(pl.Int8).alias("rec_native"),
        pl.col("b_web_stem").is_not_null().cast(pl.Int8).alias("rec_has_web"),
        pl.col("b_addr_empty").cast(pl.Int8).alias("rec_addr_empty"),
        (pl.col("a_name_core") == pl.col("b_name_core")).cast(pl.Int8).alias("nm_exact"),
        (pl.col("a_name_sorted") == pl.col("b_name_sorted")).cast(pl.Int8).alias("nm_sorted_exact"),
        tok_i.alias("tok_inter"), tok_j.alias("tok_jacc"),
        pl.col("b_name_tok").list.contains(pl.col("a_rare_tok")).fill_null(False).cast(pl.Int8).alias("a_rare_in_b"),
        pl.col("a_name_tok").list.contains(pl.col("b_rare_tok")).fill_null(False).cast(pl.Int8).alias("b_rare_in_a"),
        pl.col("a_rare_logdf"), pl.col("b_rare_logdf"),
        pl.col("a_name_tok").list.len().alias("a_ntok"), pl.col("b_name_tok").list.len().alias("b_ntok"),
        pl.col("a_name_core").str.len_chars().alias("a_len"), pl.col("b_name_core").str.len_chars().alias("b_len"),
        pl.when(both_legal).then((pl.col("a_legal") == pl.col("b_legal")).cast(pl.Int8)).otherwise(-1).alias("legal_eq"),
        pl.when(both_legal).then((legal_i == 0).cast(pl.Int8)).otherwise(-1).alias("legal_conflict"),
        (pl.col("a_legal") != "").cast(pl.Int8).alias("a_has_legal"),
        (pl.col("b_legal") != "").cast(pl.Int8).alias("b_has_legal"),
        pl.when((pl.col("a_name_nums").list.len() > 0) & (pl.col("b_name_nums").list.len() > 0))
        .then((nn_i > 0).cast(pl.Int8)).otherwise(-1).alias("name_num_match"),
        num_i.alias("num_inter"), num_j.alias("num_jacc"),
        pl.when(both_nums).then((num_i == 0).cast(pl.Int8)).otherwise(-1).alias("num_conflict"),
        pl.when(pl.col("a_addr_num1").is_not_null() & pl.col("b_addr_num1").is_not_null())
        .then((pl.col("a_addr_num1") == pl.col("b_addr_num1")).cast(pl.Int8)).otherwise(-1).alias("num1_eq"),
        pl.col("a_addr_nums").list.len().alias("a_nnums"), pl.col("b_addr_nums").list.len().alias("b_nnums"),
        pl.when(pl.col("a_state").is_not_null() & pl.col("b_state").is_not_null())
        .then((pl.col("a_state") == pl.col("b_state")).cast(pl.Int8)).otherwise(-1).alias("state_eq"),
        loc_i.alias("loc_inter"), loc_j.alias("loc_jacc"),
        pl.col("a_n_s1_same_name"), pl.col("a_n_rec_same_name"), pl.col("a_n_s1_same_name_state"),
        pl.col("b_n_s1_same_name"),
        pl.col("a_common_logdf"), pl.col("b_common_logdf"),
        pl.col("a_unseen_tok_share"), pl.col("b_unseen_tok_share"),
    )
    feats = pl.concat([feats, sem, number_proximity(df)], how="horizontal")
    empty_addr = (df["a_addr_str"] == "") | (df["b_addr_str"] == "")
    return feats.with_columns(
        [pl.when(pl.Series(empty_addr)).then(-1.0).otherwise(pl.col(c)).alias(c)
         for c in ("ad_ratio", "ad_tset", "ad_tsort", "ad_partial")]
        + [pl.when(pl.col("web_ratio") == 0).then(-1.0).otherwise(pl.col("web_ratio")).alias("web_ratio")]
    )


def group_features(f: pl.DataFrame) -> pl.DataFrame:
    """How each candidate compares with the other candidates of the same S1.

    Record-side comparisons are only taken from retrieval (gap_rec_top, rank_rev), which search the
    full S1 population; a window over this table would see only one split's S1s in training."""
    exprs = [pl.len().over("s1_id").alias("n_cand_s1")]
    for c in ("cos", "nm_tset", "nm_ratio", "ad_tset", "num_jacc", "nm_phon"):
        mx = pl.col(c).max().over("s1_id")
        exprs += [
            (pl.col(c) - mx).alias(f"{c}_gap_s1"),
            pl.col(c).rank("min", descending=True).over("s1_id").cast(pl.Int16).alias(f"{c}_rank_s1"),
        ]
    exprs.append(((pl.col("cos") >= pl.col("cos").max().over("s1_id") - 0.03).sum().over("s1_id"))
                 .alias("n_near_top_s1"))
    return f.with_columns(exprs)


ADMIN_MIN_STATELESS = 0.30
ADMIN_MIN_COUNT = 50
# Department -> region aliases the miner cannot find because the department name also occurs inside S1 addresses
# (e.g. "Rue du Ferry Nord-Pas-de-Calais"): standard administrative geography, like the US state table.
STATIC_ADMIN = [("France", "pas de calais", "HDF")]


def adapt_admin_levels(recs: pl.DataFrame, addr_clean: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Data-driven fix for sources that name a different administrative level than Source 1
    (test France: S1 gives the region, S2/S3 mostly give the department).

    Only for countries where > 30% of S2/S3 records have no recognised state. From confident pairs
    (mutual best retrieval match with above-median cosine), count record address components that
    never occur in any S1 address together with the matched S1's state; keep components seen >= 50
    times that point to one state with purity >= 0.95. Records containing such a component get that
    state, and the component's tokens are dropped from their address tokens/string."""
    r = recs.select("id", "src", "country", "state").join(addr_clean, on="id")
    rec = r.filter(pl.col("src") != 1)
    sl = rec.group_by("country").agg(pl.col("state").is_null().mean().alias("sl"))
    countries = sl.filter(pl.col("sl") > ADMIN_MIN_STATELESS)["country"].to_list()
    log(f"admin-level adaptation: stateless share by country {dict(sl.iter_rows())} -> active for {countries}")
    if not countries:
        return recs

    def comps(df):
        return (df.with_columns(pl.col("addr_clean").str.split(", ").alias("comp")).explode("comp")
                .filter(pl.col("comp").is_not_null() & (pl.col("comp") != "") & ~pl.col("comp").str.contains(r"\d")))

    s1 = r.filter((pl.col("src") == 1) & pl.col("country").is_in(countries))
    s1_comps = comps(s1).select("country", "comp").unique()
    rec_sl = rec.filter(pl.col("state").is_null() & pl.col("country").is_in(countries))
    mutual = cand.filter((pl.col("rank_fwd") == 1) & (pl.col("rank_rev") == 1))
    conf = mutual.filter(pl.col("cos") >= pl.col("cos").median()).select("s1_id", "rec_id")
    j = (conf.join(s1.select(pl.col("id").alias("s1_id"), pl.col("state").alias("s1_state")), on="s1_id")
         .join(rec_sl.select(pl.col("id").alias("rec_id"), "country", "addr_clean"), on="rec_id"))
    j = comps(j).join(s1_comps, on=["country", "comp"], how="anti").drop_nulls("s1_state")
    st = (j.group_by("country", "comp", "s1_state").len()
          .with_columns(pl.col("len").sum().over("country", "comp").alias("tot"))
          .with_columns((pl.col("len") / pl.col("tot")).alias("purity")))
    amap = (st.filter((pl.col("tot") >= ADMIN_MIN_COUNT) & (pl.col("purity") >= 0.95))
            .sort("len", descending=True).unique(["country", "comp"], keep="first")
            .select("country", "comp", pl.col("s1_state").alias("alias_state")))
    static = pl.DataFrame(STATIC_ADMIN, schema=["country", "comp", "alias_state"], orient="row").filter(pl.col("country").is_in(countries))
    if static.height:
        static = static.with_columns(pl.col(c).cast(amap.schema[c]) for c in amap.columns)
        amap = pl.concat([amap, static.join(amap.select("country", "comp"), on=["country", "comp"], how="anti")])
    log(f"mined {amap.height} admin aliases: {sorted(amap.select('comp', 'alias_state').rows())[:40]}")
    if amap.height == 0:
        return recs
    hit = (comps(rec_sl).join(amap, on=["country", "comp"])
           .unique("id", keep="first").select("id", "alias_state", pl.col("comp").alias("alias_comp")))
    has = pl.col("alias_comp").is_not_null()
    drop = pl.col("alias_comp").str.split(" ")
    new_str = pl.col("addr_str")
    for c in amap["comp"].unique().to_list():  # few aliases; str.replace needs a literal pattern
        new_str = pl.when(pl.col("alias_comp") == c).then(
            pl.col("addr_str").str.replace(c, "", literal=True).str.replace_all(r"\s+", " ").str.strip_chars()
        ).otherwise(new_str)
    recs = recs.join(hit, on="id", how="left").with_columns(
        pl.coalesce("state", "alias_state").alias("state"),
        pl.when(has).then(pl.col("addr_loc_tok").list.set_difference(drop)).otherwise(pl.col("addr_loc_tok")).alias("addr_loc_tok"),
        new_str.alias("addr_str"),
    )
    log(f"admin aliases applied to {hit.height:,} records")
    return recs.drop("alias_state", "alias_comp")


def build(split: str, s1_filter=None):
    t0 = time.time()
    recs = pl.read_parquet(WORK / f"{split}_recs.parquet", columns=REC_COLS + ["split", "addr_clean"])
    if split == "train":
        recs = recs.filter(pl.col("split").is_null() | (pl.col("split") != "ghost"))
    addr_clean = recs.select("id", "addr_clean")
    recs = recs.drop("split", "addr_clean")
    all_cand = pl.read_parquet(WORK / f"cand_{split}.parquet")
    recs = adapt_admin_levels(recs, addr_clean, all_cand)
    stats = record_stats(recs)
    cand = select_candidates(all_cand)
    if s1_filter is not None:
        cand = cand.filter(pl.col("s1_id").is_in(s1_filter.implode()))
    log(f"{split}: {cand.height:,} candidate pairs (fwd<={KF} | rev<={KR}) for {cand['s1_id'].n_unique():,} S1")
    cand = cand.sort("s1_id")
    parts = []
    for s in range(0, cand.height, CHUNK):
        pf = pair_frame(cand.slice(s, CHUNK), recs, stats)
        parts.append(features_chunk(pf))
        log(f"  features {min(s + CHUNK, cand.height):,}/{cand.height:,} ({time.time() - t0:.0f}s)")
    feats = group_features(pl.concat(parts, how="vertical_relaxed"))
    return feats


def main():
    pl.Config.set_tbl_cols(-1)
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    s1_split = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split"]).filter(pl.col("src") == 1)
    for sp in ("B", "V"):
        ids = s1_split.filter(pl.col("split") == sp)["id"]
        f = build("train", ids)
        lab = pairs.filter(pl.col("split") == sp).select(pl.col("s1_id"), pl.col("rec_id"), pl.lit(1, pl.Int8).alias("label"))
        f = f.join(lab, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("label").fill_null(0))
        f.write_parquet(WORK / f"feat_{sp}.parquet")
        n_pos = pairs.filter(pl.col("split") == sp).height
        log(f"feat_{sp}: {f.height:,} rows, positives {int(f['label'].sum()):,} of {n_pos:,} true pairs "
            f"(candidate recall {f['label'].sum() / n_pos:.4f})")
    f = build("test")
    f.write_parquet(WORK / "feat_test.parquet")
    log(f"feat_test: {f.height:,} rows")
    print(f.describe(), flush=True)


if __name__ == "__main__":
    main()
