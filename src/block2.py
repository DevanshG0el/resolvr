"""Stage B3: second learned pruning step (supervised meta-blocking with S1-neighbourhood signals).

Runs after block.py's filter and can only REMOVE pairs, so every surviving pair keeps its cross-encoder
scores. Adds S1-side context computed on the full retrieval candidate list: how many records pick the S1
as their best / top-3 match, how crowded its top is, and where the pair ranks by the record's own
preference. Trained on split-B pairs that pass the first filter; evaluated on split V.

The threshold is chosen automatically as the smallest candidate set that keeps at least
BER_BLOCK2_TARGET_KEEP (default 0.997) of the pairs the reference matcher currently gets right
(valid_scored.parquet). Saves WORK/xgb_blocker2.json and WORK/block2.json (read by features.py).
"""
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

from block import BASE_KF, BASE_KR, BLOCK_COLS_V2, blocker_frame
from config import SEED, WORK

TARGET_KEEP = float(os.environ.get("BER_BLOCK2_TARGET_KEEP", 0.997))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def predict(bst, X):
    best = bst.attr("best_iteration")
    return bst.inplace_predict(X, iteration_range=(0, int(best) + 1) if best is not None else (0, 0))


def main():
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    cand = pl.read_parquet(WORK / "cand_train.parquet").filter(
        (pl.col("rank_fwd") <= BASE_KF) | (pl.col("rank_rev") <= BASE_KR))
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    split = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split"]).filter(
        pl.col("src") == 1).select(pl.col("id").alias("s1_id"), "split")
    lab = pairs.select("s1_id", "rec_id", pl.lit(1, pl.Int8).alias("label"))
    fr = blocker_frame(cand).join(split, on="s1_id").join(lab, on=["s1_id", "rec_id"], how="left").with_columns(
        pl.col("label").fill_null(0))
    tot_v = pairs.filter(pl.col("split") == "V").height

    cfg1 = json.loads((WORK / "block.json").read_text())
    bst1 = xgb.Booster(model_file=str(WORK / "xgb_blocker.json"))
    fr = fr.with_columns(pl.Series("q1", predict(bst1, fr.select(cfg1["cols"]).to_numpy().astype(np.float32))))
    fr = fr.filter(pl.col("q1") >= cfg1["threshold"])
    B, V = fr.filter(pl.col("split") == "B"), fr.filter(pl.col("split") == "V")
    nV_s1 = pl.read_parquet(WORK / "train_recs.parquet", columns=["src", "split"]).filter(
        (pl.col("src") == 1) & (pl.col("split") == "V")).height
    log(f"after first filter: B {B.height:,} pairs, V {V.height:,} pairs ({V.height / nV_s1:.2f}/S1), "
        f"V recall {int(V['label'].sum()) / tot_v:.5f}")

    kept_ref = pl.read_parquet(WORK / "valid_scored.parquet", columns=["s1_id", "rec_id", "label", "p"]).filter(
        (pl.col("p") >= 0.5) & (pl.col("label") == 1)).select("s1_id", "rec_id").with_columns(pl.lit(1).alias("_k"))

    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                  max_depth=7, eta=0.08, subsample=0.8, colsample_bytree=0.9, min_child_weight=20, seed=SEED)
    XB = B.select(BLOCK_COLS_V2).to_numpy().astype(np.float32)
    es = np.random.default_rng(SEED).random(len(XB)) < 0.05
    bst = xgb.train(params, xgb.DMatrix(XB[~es], B["label"].to_numpy()[~es]), 1000,
                    evals=[(xgb.DMatrix(XB[es], B["label"].to_numpy()[es]), "es")],
                    early_stopping_rounds=40, verbose_eval=200)
    bst.save_model(WORK / "xgb_blocker2.json")
    Vq = V.with_columns(pl.Series("q2", predict(bst, V.select(BLOCK_COLS_V2).to_numpy().astype(np.float32)))).join(
        kept_ref, on=["s1_id", "rec_id"], how="left")

    rows, best = [], None
    for t in np.geomspace(0.001, 0.5, 90):
        k = Vq.filter(pl.col("q2") >= t)
        kept = k["_k"].is_not_null().sum() / kept_ref.height
        rows.append({"threshold": float(t), "cands_per_s1": k.height / nV_s1,
                     "recall": int(k["label"].sum()) / tot_v, "matched_pairs_kept": kept})
        if kept >= TARGET_KEEP:
            best = float(t)
    curve = pl.DataFrame(rows)
    curve.write_csv(WORK / "blocking2_study.csv")
    print(curve.filter(pl.col("threshold").is_in(curve["threshold"].gather(list(range(0, 90, 6))).implode()))
          .with_columns(pl.col("cands_per_s1").round(3), pl.col("recall").round(5), pl.col("matched_pairs_kept").round(5)),
          flush=True)
    base_kept = Vq["_k"].is_not_null().sum() / kept_ref.height
    if best is None:
        log(f"no threshold reaches kept >= {TARGET_KEEP} (first filter alone keeps {base_kept:.5f}); not writing block2.json")
        return
    k = Vq.filter(pl.col("q2") >= best)
    res = {"threshold": best, "cols": BLOCK_COLS_V2, "target_keep": TARGET_KEEP,
           "valid_cands_per_s1": k.height / nV_s1, "valid_recall": int(k["label"].sum()) / tot_v,
           "valid_matched_pairs_kept": float(k["_k"].is_not_null().sum() / kept_ref.height),
           "first_filter_cands_per_s1": V.height / nV_s1, "first_filter_matched_pairs_kept": float(base_kept)}
    (WORK / "block2.json").write_text(json.dumps(res, indent=1))
    log(f"second filter q2 >= {best:.4f}: {V.height / nV_s1:.2f} -> {res['valid_cands_per_s1']:.2f} candidates/S1; "
        f"recall {res['valid_recall']:.5f}; matched pairs kept {res['valid_matched_pairs_kept']:.5f} "
        f"(first filter alone {base_kept:.5f})")
    g = bst.get_score(importance_type="gain")
    log("feature gain: " + ", ".join(f"{BLOCK_COLS_V2[int(k[1:])]}={v:.0f}"
                                    for k, v in sorted(g.items(), key=lambda t: -t[1])[:12]))


if __name__ == "__main__":
    main()
