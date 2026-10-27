"""Filler-word rule for countries absent from training (test: France), mined without labels.

Records in S2/S3 often carry a word the S1 name does not have. The generator uses such words in two ways:
cosmetic (same business, e.g. "Et Fils", "Groupe" with the same house number) and decoy (a different
business: the word plus a changed house number). The matcher learned the US/India vocabulary from labels but
has never seen the French one, so it rejects French cosmetic additions.

Counting, per country, over candidate pairs whose record name = S1 name + an added word set w (core names,
legal forms removed; at most 2 added tokens):
  N_eq(w)   pairs with the same first house number,   N_diff(w) pairs with a different one.
Decoys keep the house number only by coincidence, at a rate q estimated from the decoy-dominated word sets
(median N_eq/N_diff over sets with N_diff >= 1000). The expected real-match share among same-number pairs is
  share(w) = 1 - q * N_diff(w) / N_eq(w).
For an unseen country, a same-number, same-address pair (ad_tset >= ADDR_MIN) with share(w) >= SHARE_MIN gets
p = max(p, share(w)); the saved decision rule (exclusivity + expected-F0.5) then runs as usual.

Modes:
  BER_FILLERS_VALIDATE=1: runs the counting on labelled split V (US/India) and prints estimated shares next to
    the observed real-match rates, plus the F0.5 change if the rule were applied there.
  default: applies the rule to the test scores (BER_SCORED, default test_scored_sub9.parquet) and writes
    OUT/fillers_<BER_VARIANT_NAME>/ (matching_results.tsv, candidate_pairs.tsv), then validates.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import polars as pl

import decide as D
from config import DATA, OUT, WORK

SHARE_MIN = float(os.environ.get("BER_FILLERS_SHARE", 0.85))
ADDR_MIN = float(os.environ.get("BER_FILLERS_ADDR", 90))
MIN_EQ = 50
# Optional, for the "swap" structure (see swap_keep.py for the measured constant):
Q_SWAP = os.environ.get("BER_FILLERS_Q_SWAP", "")        # "add" (reuse the add-decoy rate) or a number
DECOY_CAP = os.environ.get("BER_FILLERS_DECOY_CAP", "")  # a number: cap p of same-number swaps judged decoys
DECOY_MAX = 0.2
DECOY_ADDR = float(os.environ.get("BER_FILLERS_DECOY_ADDR", os.environ.get("BER_FILLERS_ADDR", 90)))  # address bar for the cap
SWAP2 = os.environ.get("BER_FILLERS_SWAP2", "0") == "1"  # also: drop one word + add a two-word set


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def added_words(P: pl.DataFrame) -> pl.DataFrame:
    """P: s1_id, rec_id, a (S1 name_core), b (record name_core). Adds `w`, or null:
    "add:<words>"  the record name is the S1 name plus 1-2 words;
    "swap:<words>" the record drops one S1 word (of >= 2) and adds one unrelated word (not a typo of it), or a
                   two-word set such as "et fils" when BER_FILLERS_SWAP2=1."""
    from rapidfuzz import fuzz
    ta, tb = pl.col("a").str.split(" ").list.unique(), pl.col("b").str.split(" ").list.unique()
    P = P.with_columns(ta.alias("_ta"), tb.alias("_tb"))
    P = P.with_columns(pl.col("_tb").list.set_difference("_ta").list.sort().alias("_add"),
                       pl.col("_ta").list.set_difference("_tb").list.sort().alias("_remw"))
    P = P.with_columns(pl.col("_remw").list.len().alias("_rem"), pl.col("_ta").list.len().alias("_na"))
    x = P["_remw"].list.first().fill_null("").to_list()
    y = P["_add"].list.first().fill_null("").to_list()
    P = P.with_columns(pl.Series("_sim", [fuzz.ratio(u, v) if u and v else 100.0 for u, v in zip(x, y)]))
    add = (pl.col("_rem") == 0) & pl.col("_add").list.len().is_between(1, 2) & (pl.col("a") != "")
    swap = (pl.col("_rem") == 1) & (pl.col("_add").list.len() == 1) & (pl.col("_na") >= 2) & (pl.col("_sim") < 75)
    if SWAP2:
        swap = swap | ((pl.col("_rem") == 1) & (pl.col("_add").list.len() == 2) & (pl.col("_na") >= 2))
    return P.with_columns(
        pl.when(add).then(pl.lit("add:") + pl.col("_add").list.join("+"))
        .when(swap).then(pl.lit("swap:") + pl.col("_add").list.join("+")).otherwise(None).alias("w")
    ).drop("_ta", "_tb", "_add", "_remw", "_rem", "_na", "_sim")


def word_table(P: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Per added word set: N_eq, N_diff, share. P needs w and num1_eq (1 equal, 0 differs, -1 missing)."""
    t = P.filter(pl.col("w").is_not_null() & (pl.col("legal_eq") != 0)).group_by("w").agg(
        (pl.col("num1_eq") == 1).sum().alias("n_eq"), (pl.col("num1_eq") == 0).sum().alias("n_diff"))
    # coincidence rate estimated separately per structure ("add" / "swap"): the pairs that are not the
    # same business share the house number at different rates (decoys vs. neighbours in one building)
    t = t.with_columns(pl.col("w").str.extract(r"^(\w+):", 1).alias("kind"))
    qs = {}
    for k in t["kind"].unique().to_list():
        big = t.filter((pl.col("kind") == k) & (pl.col("n_diff") >= 1000))
        qs[k] = float((big["n_eq"] / big["n_diff"]).median()) if big.height >= 3 else 0.01
    if Q_SWAP and "swap" in qs:
        qs["swap"] = qs.get("add", 0.01) if Q_SWAP == "add" else float(Q_SWAP)
    t = t.with_columns(pl.col("kind").replace_strict(qs, return_dtype=pl.Float64).alias("q"))
    t = t.with_columns((1 - pl.col("q") * pl.col("n_diff") / pl.col("n_eq").clip(1)).clip(0, 1).alias("share"))
    q = ", ".join(f"{k} {v:.4f}" for k, v in sorted(qs.items()))
    t = t.with_columns(pl.when(pl.col("n_eq") >= MIN_EQ).then(pl.col("share")).otherwise(0.0).alias("share"),
                       (pl.col("n_eq") >= MIN_EQ).alias("frequent"))
    return t, q


def pairs(split: str, sc: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    recs = pl.read_parquet(WORK / f"{split}_recs.parquet", columns=["id", "name_core"])
    fname = "V" if split == "train" else "test"
    feat = pl.read_parquet(WORK / f"feat_{fname}.parquet", columns=["s1_id", "rec_id", "num1_eq", "ad_tset", "legal_eq"])
    P = sc.filter(pl.col("s1_id").is_in(s1_ids.implode())).join(feat, on=["s1_id", "rec_id"])
    P = P.join(recs.rename({"id": "s1_id", "name_core": "a"}), on="s1_id").join(
        recs.rename({"id": "rec_id", "name_core": "b"}), on="rec_id")
    return added_words(P.with_columns(pl.col("a").fill_null(""), pl.col("b").fill_null("")))


def apply_rule(P: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    P = P.join(t.select("w", "share", "frequent"), on="w", how="left").with_columns(
        pl.col("share").fill_null(0.0), pl.col("frequent").fill_null(False))
    base = (pl.col("num1_eq") == 1) & (pl.col("ad_tset") >= ADDR_MIN) & (pl.col("legal_eq") != 0)
    hit = base & (pl.col("share") >= SHARE_MIN)
    P = P.with_columns(hit.alias("rule"), pl.when(hit).then(pl.max_horizontal("p", "share")).otherwise(pl.col("p")).alias("p_new"))
    decoy = ((pl.col("num1_eq") == 1) & (pl.col("ad_tset") >= DECOY_ADDR) & (pl.col("legal_eq") != 0)
             & pl.col("frequent") & (pl.col("share") < DECOY_MAX) & pl.col("w").str.starts_with("swap:"))
    if DECOY_CAP:
        P = P.with_columns(pl.when(decoy).then(pl.min_horizontal("p_new", pl.lit(float(DECOY_CAP)))).otherwise(pl.col("p_new")).alias("p_new"))
    return P.with_columns(decoy.alias("decoy"))


def decide(sc: pl.DataFrame, cfg: dict, col: str) -> pl.DataFrame:
    ex = D.exclusive(sc.select("s1_id", "rec_id", pl.col(col).alias("p")))
    if cfg["rule"] == "threshold":
        return D.select_threshold(ex, "p", cfg["thr"])
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    return D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"], device=os.environ.get("BER_DEVICE", "cuda"))


def validate(cfg: dict):
    sel = pl.read_parquet(WORK / "qwen_selection.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id", "p", "label")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"])
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    for c in ("US", "India"):
        ids = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V") & (pl.col("country") == c))["id"]
        P = pairs("train", sel, ids)
        t, q = word_table(P)
        obs = P.filter((pl.col("num1_eq") == 1) & (pl.col("legal_eq") != 0)).group_by("w").agg(pl.col("label").mean().alias("true_rate_eq"))
        t = t.join(obs, on="w", how="left")
        log(f"{c}: coincidence rates q: {q}; added word sets, estimated share vs observed real-match rate (same number):")
        pl.Config.set_tbl_rows(40)
        print(t.filter(pl.col("n_eq") >= MIN_EQ).sort("n_eq", descending=True).head(40).with_columns(
            pl.col("share", "true_rate_eq").round(3)), flush=True)
        for kind in ("add:", "swap:"):
            h = apply_rule(P, t).filter(pl.col("rule") & pl.col("w").str.starts_with(kind))
            log(f"{c}: {kind} rule pairs {h.height:,}, real-match rate {h['label'].mean() if h.height else float('nan'):.4f}, "
                f"model p >= 0.5 already {(h['p'] >= 0.5).mean() if h.height else float('nan'):.4f}")
        R = apply_rule(P, t)
        hits = R.filter(pl.col("rule"))
        log(f"{c}: rule pairs {hits.height:,}, real-match rate among them {hits['label'].mean() if hits.height else float('nan'):.4f}, "
            f"already above 0.5 before: {(hits['p'] >= 0.5).mean() if hits.height else float('nan'):.4f}")
        V = sel.filter(pl.col("s1_id").is_in(ids.implode())).join(R.select("s1_id", "rec_id", "p_new"), on=["s1_id", "rec_id"], how="left") \
               .with_columns(pl.col("p_new").fill_null(pl.col("p")))
        tr = truth.filter(pl.col("s1_id").is_in(ids.implode()))
        f0 = D.macro_f05(decide(V, cfg, "p"), tr, ids)["f05"]
        f1 = D.macro_f05(decide(V, cfg, "p_new"), tr, ids)["f05"]
        log(f"{c}: V F0.5 without rule {f0:.5f}, with rule (as if {c} were unseen) {f1:.5f} ({f1 - f0:+.5f})")


def main():
    dec = json.loads(open(os.environ.get("BER_DECISION", WORK / "decision_sub7.json")).read())
    cfg = dec["cfg"]
    if os.environ.get("BER_FILLERS_VALIDATE", "0") == "1":
        return validate(cfg)
    name = os.environ.get("BER_VARIANT_NAME", "fillers")
    sc = pl.read_parquet(os.environ.get("BER_SCORED", WORK / "test_scored_sub9.parquet")).select("s1_id", "rec_id", "p")
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"]).filter(pl.col("src") == 1)
    seen = set(pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list())
    unseen = sorted(set(test["country"].unique().to_list()) - seen)
    new_p = []
    for c in unseen:
        ids = test.filter(pl.col("country") == c)["id"]
        P = pairs("test", sc, ids)
        t, q = word_table(P)
        log(f"{c}: coincidence rates q: {q}; added word sets (same-number share):")
        pl.Config.set_tbl_rows(45)
        print(t.filter(pl.col("n_eq") >= MIN_EQ).sort("n_eq", descending=True).head(45).with_columns(pl.col("share").round(3)), flush=True)
        R = apply_rule(P, t)
        hits = R.filter(pl.col("rule"))
        dec_ = R.filter(pl.col("decoy"))
        log(f"{c}: same-number low-share swaps (decoy candidates): {dec_.height:,}, with p >= 0.5: {(dec_['p'] >= 0.5).sum():,}"
            + (f"; capped at {DECOY_CAP}" if DECOY_CAP else " (not capped)"))
        log(f"{c}: rule raises {hits.filter(pl.col('p_new') > pl.col('p')).height:,} of {hits.height:,} rule pairs; "
            f"by word set: " + ", ".join(f"{w} {n}" for w, n in hits.group_by("w").len().sort("len", descending=True).head(12).iter_rows()))
        new_p.append(R.select("s1_id", "rec_id", "p_new"))
    upd = pl.concat(new_p) if new_p else pl.DataFrame(schema={"s1_id": pl.Utf8, "rec_id": pl.Utf8, "p_new": pl.Float64})
    sc = sc.join(upd, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("p_new").fill_null(pl.col("p")))
    sel0, sel1 = decide(sc, cfg, "p"), decide(sc, cfg, "p_new")
    per = lambda s: test.select(pl.col("id").alias("s1_id"), "country").join(s.group_by("s1_id").len(name="n"), on="s1_id", how="left") \
        .fill_null(0).group_by("country").agg(pl.col("n").mean().round(4).alias("matches_per_s1")).sort("country")
    log("matches per S1 before / after:")
    print(per(sel0).join(per(sel1), on="country", suffix="_after"), flush=True)
    out = OUT / f"fillers_{name}"
    out.mkdir(parents=True, exist_ok=True)
    from predict import write_lists
    write_lists(sel1, test["id"], "matched_entity_ids", out / "matching_results.tsv")
    shutil.copy(os.environ.get("BER_CANDIDATES", OUT / "candidate_pairs.tsv"), out / "candidate_pairs.tsv")
    sc.select("s1_id", "rec_id", pl.col("p_new").alias("p")).write_parquet(WORK / f"test_scored_fillers_{name}.parquet")
    validator = DATA.parent / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test"), "--check-ids"],
                       capture_output=True, text=True)
    print(r.stdout[-1200:], r.stderr[-800:], flush=True)
    log(f"written {out}; validator exit {r.returncode}")


if __name__ == "__main__":
    main()
