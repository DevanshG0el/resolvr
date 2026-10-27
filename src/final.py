"""Stage F: final submission = per-country routing + the filler/swap rule for countries absent from training.

Countries seen in training (US, India) take the scores of the main matcher (BER_SCORED_SEEN, e.g. the B+V
matcher with e5 + bge + Qwen cross-encoder features) and its decision rule (BER_DECISION_SEEN).
Countries absent from training (test: France) take the scores of the matcher whose cross-encoders were adapted
to them by self-training (BER_SCORED_UNSEEN, BER_DECISION_UNSEEN), then the unsupervised filler/swap rule of
fillers.py (BER_FILLERS_SHARE, default 0.80), then that decision rule.
Writes OUT/matching_results.tsv and OUT/candidate_pairs.tsv (every scored pair = the matcher's input) and runs
the official validator.
"""
import json
import os
import subprocess
import sys
import time

import polars as pl

os.environ.setdefault("BER_FILLERS_SHARE", "0.80")
import fillers as FL  # noqa: E402
from config import DATA, OUT, WORK  # noqa: E402
from predict import write_lists  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guard_renames(sc: pl.DataFrame, ids: pl.Series, ref_path: str) -> pl.DataFrame:
    """Unseen countries: invented-name records at the S1 address (name token-set < 40, every record name token unseen
    among S1 names, address token-set >= 90) may not score higher than in the reference route. Counting on the test
    files shows these records behave differently in France (same : different house number 0.63 vs 3.1-4.5 in
    US/India, where 71% of the same-number ones are real), so the France-adapted Qwen is not trusted to raise them."""
    f = pl.read_parquet(WORK / "feat_test.parquet", columns=["s1_id", "rec_id", "nm_tset", "b_unseen_tok_share", "rec_addr_empty", "ad_tset"])
    flag = f.filter(pl.col("s1_id").is_in(ids.implode()) & (pl.col("nm_tset") < 40) & (pl.col("b_unseen_tok_share") >= 0.99)
                    & (pl.col("rec_addr_empty") == 0) & (pl.col("ad_tset") >= 90)).select("s1_id", "rec_id")
    ref = flag.join(pl.read_parquet(ref_path).select("s1_id", "rec_id", pl.col("p").alias("p_ref")), on=["s1_id", "rec_id"])
    out = sc.join(ref, on=["s1_id", "rec_id"], how="left").with_columns(
        pl.when(pl.col("p_ref").is_not_null()).then(pl.min_horizontal("p", "p_ref")).otherwise(pl.col("p")).alias("p_g"))
    log(f"rename guard: {flag.height:,} invented-name pairs at the S1 address; lowered {(out['p_g'] < out['p']).sum():,}; "
        f"p >= 0.5 before {out.filter(pl.col('p_ref').is_not_null() & (pl.col('p') >= 0.5)).height:,}, after "
        f"{out.filter(pl.col('p_ref').is_not_null() & (pl.col('p_g') >= 0.5)).height:,}")
    return out.select("s1_id", "rec_id", pl.col("p_g").alias("p"))


def load(env_scored, env_dec, default_scored, default_dec):
    sc = pl.read_parquet(os.environ.get(env_scored, WORK / default_scored)).select("s1_id", "rec_id", "p")
    cfg = json.loads(open(os.environ.get(env_dec, WORK / default_dec)).read())["cfg"]
    return sc, cfg


def main():
    t0 = time.time()
    seen_sc, seen_cfg = load("BER_SCORED_SEEN", "BER_DECISION_SEEN", "test_scored.parquet", "decision.json")
    un_sc, un_cfg = load("BER_SCORED_UNSEEN", "BER_DECISION_UNSEEN", "test_scored_sub9.parquet", "decision_sub7.json")
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
    seen = set(pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list())
    unseen = sorted(set(test["country"].unique().to_list()) - seen)
    un_ids = test.filter(pl.col("country").is_in(unseen))["id"]
    log(f"countries absent from training: {unseen} ({un_ids.len():,} S1); filler rule share >= {FL.SHARE_MIN}")
    assert seen_sc.select("s1_id", "rec_id").sort("s1_id", "rec_id").equals(
        un_sc.select("s1_id", "rec_id").sort("s1_id", "rec_id")), "both score files must cover the same candidate pairs"
    if os.environ.get("BER_GUARD_SCORED"):
        un_sc = guard_renames(un_sc, un_ids, os.environ["BER_GUARD_SCORED"])

    parts = [FL.decide(seen_sc.filter(~pl.col("s1_id").is_in(un_ids.implode())), seen_cfg, "p")]
    for c in unseen:
        ids = test.filter(pl.col("country") == c)["id"]
        P = FL.pairs("test", un_sc, ids)
        t, q = FL.word_table(P)
        R = FL.apply_rule(P, t)
        hits = R.filter(pl.col("rule"))
        log(f"{c}: coincidence rates {q}; rule pairs {hits.height:,}, raised {hits.filter(pl.col('p_new') > pl.col('p')).height:,}; "
            f"decoy candidates {R.filter(pl.col('decoy')).height:,}, with p >= 0.5 before the cap {R.filter(pl.col('decoy') & (pl.col('p') >= 0.5)).height:,}")
        sc = un_sc.filter(pl.col("s1_id").is_in(ids.implode())).join(
            R.select("s1_id", "rec_id", "p_new"), on=["s1_id", "rec_id"], how="left").with_columns(pl.col("p_new").fill_null(pl.col("p")))
        parts.append(FL.decide(sc, un_cfg, "p_new"))
    sel = pl.concat(parts)
    write_lists(seen_sc.select("s1_id", "rec_id"), test["id"], "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    write_lists(sel, test["id"], "matched_entity_ids", OUT / "matching_results.tsv")
    per = test.select(pl.col("id").alias("s1_id"), "country").join(sel.group_by("s1_id").len(name="n"), on="s1_id", how="left").fill_null(0)
    print(per.group_by("country").agg(pl.len().alias("s1"), pl.col("n").mean().round(4).alias("matches_per_s1"),
                                      (pl.col("n") == 0).mean().round(4).alias("empty_rate")).sort("country"), flush=True)
    validator = DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(OUT / "matching_results.tsv"),
                        "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"), "--check-ids"],
                       capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-800:], flush=True)
    log(f"validator exit code {r.returncode}; stage F done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
