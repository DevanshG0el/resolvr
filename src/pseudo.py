"""France self-training: adapt both cross-encoder families to the unseen country with pseudo-labels.

Uses only unlabeled test inputs plus the model's own predictions (no ground truth, no external data):
1. Pseudo-labels from the final model's French test predictions (test_scored_sub7.parquet):
   positives  p >= POS_P, the pair is the record's best S1, and the record's runner-up S1 has p < 0.05;
   negatives  p <= NEG_P among the pairs the cross-encoders re-score.
2. French S1s are split into two halves. For each cross-encoder family (e5, bge) and each half, the
   existing checkpoint is fine-tuned further on that half's pseudo-pairs plus a replay sample of real
   split-B training pairs, then scores the OTHER half's French pairs (cross-fitting, so no French pair is
   scored by a model that trained on it).
3. The French rows of xenc_test.parquet / xenc_bge_test.parquet are replaced (originals kept as
   *_prefr.parquet). US/India rows are untouched. Run predict.py afterwards.
"""
import json
import os
import shutil
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import xenc as X
from config import SEED, WORK

POS_P = float(os.environ.get("BER_PSEUDO_POS", 0.98))
NEG_P = float(os.environ.get("BER_PSEUDO_NEG", 0.005))
REPLAY = int(os.environ.get("BER_PSEUDO_REPLAY", 150_000))
STEPS = int(os.environ.get("BER_PSEUDO_STEPS", 2000))
SRC = os.environ.get("BER_PSEUDO_SRC", "test_scored_sub7.parquet")  # predictions the pseudo-labels come from
ROUND = int(os.environ.get("BER_PSEUDO_ROUND", 1))  # 2: continue each half's round-1 French checkpoint
XGB_ROWS = os.environ.get("BER_PSEUDO_XGB", "0") == "1"  # also write French rows for XGBoost training
# optional final selection (a matching_results.tsv, e.g. after the filler/swap rule): its pairs with p >= SEL_MIN
# become confident positives, so knowledge added after the matcher reaches the cross-encoders
SELECTED = os.environ.get("BER_PSEUDO_SELECTED", "")
SEL_MIN = float(os.environ.get("BER_PSEUDO_SEL_MIN", 0.8))
FAMILIES = {  # tag: (checkpoint dir for half 0, max_len)
    "": ("xenc_h0" if (WORK / "xenc_h0" / "config.json").exists() else "xenc", 160),
    "_bge": ("xenc_bge_h0", 128),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fr_half(s1_ids: pl.Series) -> np.ndarray:
    return ((s1_ids.str.slice(3).cast(pl.Int64) * 2862933555 // 16384) % 2).to_numpy()


def selected_pairs(fr_s1: pl.Series) -> pl.DataFrame:
    m = pl.read_csv(SELECTED, separator="	", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.filter(pl.col("source1_entity_id").is_in(fr_s1.implode()))
            .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("rec_id"))
            .explode("rec_id").filter(pl.col("rec_id") != "").with_columns(pl.lit(True).alias("_sel")))


def pseudo_pairs(fr_s1: pl.Series) -> pl.DataFrame:
    sc = pl.read_parquet(WORK / SRC).filter(pl.col("s1_id").is_in(fr_s1.implode()))
    if SELECTED:
        sel = selected_pairs(fr_s1)
        sc = sc.join(sel, on=["s1_id", "rec_id"], how="left").with_columns(
            pl.when(pl.col("_sel").fill_null(False) & (pl.col("p") >= SEL_MIN)).then(pl.lit(0.999)).otherwise(pl.col("p")).alias("p")).drop("_sel")
        log(f"selection {SELECTED}: {sel.height:,} selected French pairs; {sc.filter(pl.col('p') == 0.999).height:,} "
            f"used as positives (p >= {SEL_MIN})")
    sc = sc.with_columns(
        pl.col("p").max().over("rec_id").alias("rec_best"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("rec_id").fill_null(0).alias("rec_second"),
    )
    pos = sc.filter((pl.col("p") >= POS_P) & (pl.col("p") == pl.col("rec_best")) & (pl.col("rec_second") < 0.05))
    scored = pl.read_parquet(WORK / "xenc_test.parquet", columns=["s1_id", "rec_id"])
    neg = sc.filter(pl.col("p") <= NEG_P).join(scored, on=["s1_id", "rec_id"])
    rng = np.random.default_rng(SEED)
    if neg.height > int(1.5 * pos.height):
        neg = neg.filter(pl.Series(rng.random(neg.height) < 1.5 * pos.height / neg.height))
    log(f"French pseudo-labels: {pos.height:,} positives (p >= {POS_P}), {neg.height:,} negatives (p <= {NEG_P}) "
        f"over {fr_s1.len():,} French S1")
    return pl.concat([pos.select("s1_id", "rec_id").with_columns(pl.lit(1.0).alias("label")),
                      neg.select("s1_id", "rec_id").with_columns(pl.lit(0.0).alias("label"))])


def replay_pairs(pairs_file=None) -> pl.DataFrame:
    """Real split-B training pairs so continued fine-tuning does not forget US/India."""
    pairs = pl.read_parquet(pairs_file or WORK / "train_pairs.parquet").filter(pl.col("split") == "B")
    rng = np.random.default_rng(SEED + 7)
    pos = pairs.select("s1_id", "rec_id").sample(REPLAY // 2, seed=SEED)
    cand = pl.read_parquet(WORK / "cand_train.parquet", columns=["s1_id", "rec_id", "rank_fwd"]).filter(
        pl.col("rank_fwd") <= 3).join(pairs.select("s1_id", "rec_id"), on=["s1_id", "rec_id"], how="anti")
    b_ids = pairs["s1_id"].unique()
    neg = cand.filter(pl.col("s1_id").is_in(b_ids.implode())).sample(REPLAY // 2, seed=SEED).select("s1_id", "rec_id")
    return pl.concat([pos.with_columns(pl.lit(1.0).alias("label")), neg.with_columns(pl.lit(0.0).alias("label"))])


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"])
    # entities of the countries absent from training (test: France); country is an open set, nothing is hard-coded
    trained = pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list()
    fr_s1 = test.filter((pl.col("src") == 1) & ~pl.col("country").is_in(trained))["id"]
    half = dict(zip(fr_s1.to_list(), fr_half(fr_s1).tolist()))
    pp = pseudo_pairs(fr_s1).with_columns(pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h"))
    rp = attach_text(replay_pairs(), "train")
    X.LR, X.MAX_STEPS = float(os.environ.get("BER_PSEUDO_LR", 1e-5)), STEPS

    for tag, (ckpt, max_len) in FAMILIES.items():
        f = WORK / f"xenc{tag}_test.parquet"
        backup = WORK / f"xenc{tag}_test_prefr.parquet"
        if not backup.exists():
            shutil.copy(f, backup)
        cur = pl.read_parquet(backup)
        col = f"xenc{tag}"
        fr_rows = cur.filter(pl.col("s1_id").is_in(fr_s1.implode())).with_columns(
            pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h"))
        new_parts = []
        X.MAX_LEN, X.TAG = max_len, tag
        for h in (0, 1):
            if ROUND == 1:
                X.BASE, out_dir = str(WORK / ckpt), WORK / f"xenc{tag}_fr{h}"
            else:  # continue the model that trained on this half in round 1 (keeps cross-fitting intact)
                X.BASE, out_dir = str(WORK / f"xenc{tag}_fr{h}"), WORK / f"xenc{tag}_fr{h}_r{ROUND}"
            tr = pl.concat([attach_text(pp.filter(pl.col("h") == h).drop("h"), "test"), rp], how="vertical_relaxed")
            tr = tr.sample(fraction=1.0, shuffle=True, seed=SEED + h)
            log(f"[{tag or 'e5'}] half {h}: fine-tuning {X.BASE} on {tr.height:,} pairs "
                f"({int((tr['label'] == 1).sum()):,} pos) for {STEPS} steps")
            model, tok = X.train_xenc(tr, out_dir)
            target = fr_rows.filter(pl.col("h") == 1 - h).select("s1_id", "rec_id")
            model.eval()
            x = X.score(model, tok, attach_text(target, "test"))
            new_parts.append(target.with_columns(pl.Series(col, x)))
            del model
            torch.cuda.empty_cache()
        new_fr = pl.concat(new_parts)
        old = fr_rows.select("s1_id", "rec_id", col).join(new_fr.rename({col: "new"}), on=["s1_id", "rec_id"])
        log(f"[{tag or 'e5'}] French {col}: old mean {old[col].mean():.3f} -> new {old['new'].mean():.3f}; "
            f"share > 0 old {(old[col] > 0).mean():.4f} -> new {(old['new'] > 0).mean():.4f}; "
            f"rank corr {old.select(pl.corr(col, 'new', method='spearman')).item():.4f}")
        rest = cur.filter(~pl.col("s1_id").is_in(fr_s1.implode()))
        pl.concat([rest, new_fr.select(rest.columns)]).write_parquet(f)
        log(f"[{tag or 'e5'}] wrote {f.name}: {rest.height:,} unchanged rows + {new_fr.height:,} French rows re-scored")
    if XGB_ROWS:
        french_xgb_rows(fr_s1)
    (WORK / "pseudo_meta.json").write_text(json.dumps({"pos_p": POS_P, "neg_p": NEG_P, "steps": STEPS, "src": SRC, "selected": SELECTED,
                                                        "round": ROUND, "replay": REPLAY, "families": list(FAMILIES)}))
    log(f"France self-training done in {time.time() - t0:.0f}s; now run predict.py")


def french_xgb_rows(fr_s1: pl.Series):
    """French training rows for XGBoost: only French S1s whose EVERY candidate pair is confidently
    labeled (p >= POS_P or p <= NEG_P) and whose positives are each record's best S1, so the per-S1
    context features stay complete. Features come from feat_test with the (new) cross-encoder scores."""
    from train import load_feats
    dec = json.loads((WORK / "decision.json").read_text())
    T = load_feats("test", True, dec.get("xenc_tags", [""]))
    sc = pl.read_parquet(WORK / SRC).filter(pl.col("s1_id").is_in(fr_s1.implode()))
    sc = sc.with_columns(pl.col("p").max().over("rec_id").alias("rec_best"))
    ok = sc.group_by("s1_id").agg(
        ((pl.col("p") >= POS_P) | (pl.col("p") <= NEG_P)).all().alias("clean"),
        ((pl.col("p") >= POS_P) & (pl.col("p") < pl.col("rec_best"))).any().alias("conflict"),
    ).filter(pl.col("clean") & ~pl.col("conflict")).select("s1_id")
    lab = sc.join(ok, on="s1_id").select("s1_id", "rec_id", (pl.col("p") >= POS_P).cast(pl.Int8).alias("label"))
    E = T.join(lab, on=["s1_id", "rec_id"])
    E.write_parquet(WORK / "fr_pseudo_train.parquet")
    log(f"French XGBoost rows: {E.height:,} pairs from {ok.height:,} of {fr_s1.len():,} French S1 "
        f"({int(E['label'].sum()):,} positives) -> fr_pseudo_train.parquet")


def attach_text(df, split):
    return X.attach_text(df, split)


if __name__ == "__main__":
    main()
