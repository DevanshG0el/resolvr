"""Stage D: train the matcher on split B, tune the decision layer on split V.

Two XGBoost stages (GPU):
  stage 1: pair features -> p1 (out-of-fold on B, 2 folds grouped by S1)
  stage 2: pair features + per-S1 context of p1 (max, rank, gap, sum, count > 0.5) -> p2
Decision (tuned on V to maximise macro F0.5): exclusivity + calibrated expected-F0.5 selection,
compared against a plain probability threshold; the better rule is saved.
"""
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import SEED, WORK

ID_COLS = ["s1_id", "rec_id", "label"]
PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
    max_depth=8, eta=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    max_bin=256, reg_lambda=1.0, seed=SEED,
    num_parallel_tree=int(os.environ.get("BER_XGB_TREES", 1)),  # >1: a small bagged forest per boosting round
)
if os.environ.get("BER_XGB_PARAMS"):  # JSON file with overrides, e.g. hp_best.json from hp_search.py
    PARAMS.update(json.loads(open(os.environ["BER_XGB_PARAMS"]).read()))
ROUNDS = 3000
CAT = os.environ.get("BER_CAT", "0") == "1"  # also fit CatBoost at stage 2 and blend (weight chosen on V)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s1_fold(df: pl.DataFrame, k: int, mult: int = 2654435761) -> np.ndarray:
    """Deterministic S1-grouped bucket in [0, k). Use a different mult for independent splits."""
    return ((df["s1_id"].str.slice(3).cast(pl.Int64) * mult // 4096) % k).to_numpy()


def context(df: pl.DataFrame, p: np.ndarray) -> pl.DataFrame:
    """Per-S1 context of a first-stage probability.

    Includes per-source "slots": how many confident matches the S1 already has from the candidate's
    own source and from the other source (excluding the candidate). Entities usually have 1-2
    records per source, so an S1 with an open slot is the likelier owner of an ambiguous record."""
    d = df.select("s1_id", pl.col("rec_id").str.slice(0, 2).alias("_src")).with_columns(pl.Series("p1", p))
    mx = pl.col("p1").max().over("s1_id")
    conf = (pl.col("p1") > 0.5).cast(pl.Int32)
    same = conf.sum().over("s1_id", "_src") - conf
    return d.with_columns(
        (pl.col("p1") - mx).alias("p1_gap"),
        mx.alias("p1_max"),
        pl.col("p1").rank("min", descending=True).over("s1_id").alias("p1_rank"),
        pl.col("p1").sum().over("s1_id").alias("p1_sum"),
        (pl.col("p1") > 0.5).sum().over("s1_id").alias("p1_n50"),
        pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("s1_id").alias("p1_second"),
        same.alias("p1_n50_same_src"),
        (conf.sum().over("s1_id") - conf - same).alias("p1_n50_other_src"),
        (pl.col("p1").sum().over("s1_id", "_src") - pl.col("p1")).alias("p1_sum_same_src_others"),
        pl.col("p1").rank("min", descending=True).over("s1_id", "_src").alias("p1_rank_src"),
    ).drop("s1_id", "_src")


def fit(X, y, Xv=None, yv=None, rounds=ROUNDS):
    dtr = xgb.QuantileDMatrix(X, y, max_bin=PARAMS["max_bin"])
    evals = [(dtr, "train")]
    kw = {}
    if Xv is not None:
        dv = xgb.QuantileDMatrix(Xv, yv, ref=dtr)
        evals.append((dv, "valid"))
        kw = dict(early_stopping_rounds=100)
    return xgb.train(PARAMS, dtr, rounds, evals=evals, verbose_eval=250, **kw)


def predict(bst, X):
    return bst.inplace_predict(X, iteration_range=(0, bst.best_iteration + 1))


def stage_fit(F: pl.DataFrame, cols, tag):
    """2-fold OOF predictions on F plus a final model on all of F (rounds from the folds)."""
    X, y = F.select(cols).to_numpy().astype(np.float32), F["label"].to_numpy()
    fold = s1_fold(F, 2)
    oof = np.zeros(len(y), dtype=np.float32)
    its = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        # early stopping on a slice of the training fold (never on V)
        es = s1_fold(F.filter(pl.Series(tr)), 20, mult=2246822519) == 0
        assert 0 < es.sum() < len(es), "early-stopping slice must be a proper subset"
        Xt, yt = X[tr], y[tr]
        bst = fit(Xt[~es], yt[~es], Xt[es], yt[es])
        its.append(bst.best_iteration + 1)
        oof[te] = predict(bst, X[te])
        log(f"{tag} fold {k}: best iteration {its[-1]}")
    full = fit(X, y, rounds=int(np.mean(its) * 1.1))
    full.set_attr(best_iteration=str(full.num_boosted_rounds() - 1))
    return oof, full


def cat_fit(F: pl.DataFrame, cols, iters=None):
    """CatBoost on the same inputs. iters=None: early stopping on the same S1-grouped slice as XGBoost."""
    from catboost import CatBoostClassifier
    X, y = F.select(cols).to_numpy().astype(np.float32), F["label"].to_numpy()
    params = dict(loss_function="Logloss", depth=8, learning_rate=0.08, l2_leaf_reg=3.0, random_seed=SEED,
                  thread_count=-1, verbose=250, border_count=254)

    if iters is None:
        es = s1_fold(F, 20, mult=2246822519) == 0
        for task in ("GPU", "CPU"):
            try:
                m = CatBoostClassifier(iterations=4000, od_type="Iter", od_wait=100, task_type=task, **params)
                m.fit(X[~es], y[~es], eval_set=(X[es], y[es]), use_best_model=True)
                log(f"CatBoost ({task}) best iteration {m.get_best_iteration() + 1}")
                return m, m.get_best_iteration() + 1
            except Exception as e:  # noqa: BLE001
                log(f"CatBoost {task} failed: {str(e)[:200]}")
        raise RuntimeError("CatBoost could not be trained")
    for task in ("GPU", "CPU"):
        try:
            m = CatBoostClassifier(iterations=int(iters * 1.1), task_type=task, **params)
            m.fit(X, y)
            return m, iters
        except Exception as e:  # noqa: BLE001
            log(f"CatBoost {task} failed: {str(e)[:200]}")
    raise RuntimeError("CatBoost could not be trained")


def quick_f05(V: pl.DataFrame, p: np.ndarray, truth, s1v, cfg=None) -> float:
    cfg = cfg or {"temp": 1.0, "bias": 0.0, "empty_scale": 0.8}
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    return D.macro_f05(D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"]), truth, s1v)["f05"]


def importance(bst, cols, top=25):
    g = bst.get_score(importance_type="gain")
    items = sorted(((cols[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v) for k, v in g.items()),
                   key=lambda t: -t[1])[:top]
    log("top features by gain: " + ", ".join(f"{k}={v:.0f}" for k, v in items))


def tune(V: pl.DataFrame, p: np.ndarray, truth: pl.DataFrame, s1_all: pl.Series):
    base = V.select("s1_id", "rec_id").with_columns(pl.Series("p", p))
    ex = D.exclusive(base)
    results = []
    for thr in (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        m = D.macro_f05(D.select_threshold(ex, "p", thr), truth, s1_all)
        results.append(({"rule": "threshold", "thr": thr}, m))
    for temp in (0.7, 0.85, 1.0, 1.2):
        for bias in (-2.0, -1.5, -1.0, -0.5, 0.0):
            for es in (0.8, 1.0, 1.2):
                e2 = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), temp, bias)))
                m = D.macro_f05(D.select_expected_f(e2, "p", empty_scale=es), truth, s1_all)
                results.append(({"rule": "expected_f", "temp": temp, "bias": bias, "empty_scale": es}, m))
    results.sort(key=lambda t: -t[1]["f05"])
    for cfg, m in results[:8]:
        log(f"  {cfg} -> F0.5 {m['f05']:.5f} P {m['precision_micro']:.4f} R {m['recall_micro']:.4f} "
            f"single {m['singleton_acc']:.4f} empty {m['pred_empty_rate']:.4f}")
    return results[0]


def use_xenc() -> bool:
    return os.environ.get("BER_USE_XENC", "1") == "1" and (WORK / "xenc_B.parquet").exists()


def xenc_tags() -> list:
    """Cross-encoder features to use: the base one ('') plus extra tags from BER_XENC_EXTRA (e.g. '_bge')."""
    return [""] + [t for t in os.environ.get("BER_XENC_EXTRA", "").split(",") if t]


def use_namefeat() -> bool:
    return os.environ.get("BER_NAMEFEAT", "0") == "1" and (WORK / "namefeat_B.parquet").exists()


def load_feats(name: str, xenc: bool, tags=None, namefeat=None) -> pl.DataFrame:
    """Feature table, optionally with cross-encoder logits joined as extra features (null = pair not
    re-scored) and the record-side fuzzy-namesake features of namefeat.py (namefeat=None: BER_NAMEFEAT).
    If any cross-encoder was not cross-fitted, split B is restricted to the half it did not train on."""
    F = pl.read_parquet(WORK / f"feat_{name}.parquet")
    if use_namefeat() if namefeat is None else namefeat:
        F = F.join(pl.read_parquet(WORK / f"namefeat_{name}.parquet"), on=["s1_id", "rec_id"], how="left")
    if not xenc:
        return F
    tags = xenc_tags() if tags is None else tags

    def crossfit(t):
        meta = WORK / f"xenc{t}_meta.json"
        return meta.exists() and json.loads(meta.read_text()).get("crossfit", False)

    if name == "B" and not all(crossfit(t) for t in tags):
        from xenc import b_half
        F = F.filter(pl.Series(b_half(F["s1_id"]) == 1))
    for t in tags:
        F = F.join(pl.read_parquet(WORK / f"xenc{t}_{name}.parquet"), on=["s1_id", "rec_id"], how="left")
    return F


def main():
    t0 = time.time()
    xenc = use_xenc()
    tags = xenc_tags() if xenc else []
    nf = use_namefeat()
    B = load_feats("B", xenc, tags, namefeat=nf)
    V = load_feats("V", xenc, tags, namefeat=nf)
    extra = os.environ.get("BER_TRAIN_EXTRA")  # optional extra training rows, e.g. French pseudo-labels
    if extra:
        E = pl.read_parquet(WORK / extra)
        for c in [c for c in B.columns if c.startswith("xenc") and c not in E.columns]:  # e.g. a newer cross-encoder
            E = E.join(pl.read_parquet(WORK / f"{c}_test.parquet"), on=["s1_id", "rec_id"], how="left")
        B = pl.concat([B, E.select(B.columns)], how="vertical_relaxed")
        log(f"added {E.height:,} extra training rows from {extra} ({int(E['label'].sum()):,} positives)")
    cols = [c for c in B.columns if c not in ID_COLS]
    log(f"cross-encoder features: {tags if xenc else 'off'}")
    log(f"B {B.height:,} rows ({int(B['label'].sum()):,} pos), V {V.height:,} rows, {len(cols)} features")

    oof1, m1 = stage_fit(B, cols, "stage1")
    importance(m1, cols)
    Xv = V.select(cols).to_numpy().astype(np.float32)
    p1v = predict(m1, Xv)

    B2 = pl.concat([B, context(B, oof1)], how="horizontal")
    V2 = pl.concat([V, context(V, p1v)], how="horizontal")
    cols2 = cols + [c for c in B2.columns if c.startswith("p1")]
    _, m2 = stage_fit(B2, cols2, "stage2")
    importance(m2, cols2)
    p2v = predict(m2, V2.select(cols2).to_numpy().astype(np.float32))

    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    truth = pairs.filter(pl.col("split") == "V").select("s1_id", "rec_id")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"])
    s1v = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V"))

    cat_w, cat_model, cat_iters = 0.0, None, None
    if CAT:
        cat_model, cat_iters = cat_fit(B2, cols2)
        pcv = cat_model.predict_proba(V2.select(cols2).to_numpy().astype(np.float32))[:, 1]
        scores = {w: quick_f05(V, (1 - w) * p2v + w * pcv, truth, s1v["id"]) for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
        log("CatBoost blend on V (weight -> F0.5): " + ", ".join(f"{w}: {f:.5f}" for w, f in scores.items()))
        cat_w = max(scores, key=scores.get)
        if cat_w > 0:
            p2v = (1 - cat_w) * p2v + cat_w * pcv
        log(f"CatBoost blend weight {cat_w}")
    log(f"V: {s1v.height:,} S1, {truth.height:,} true pairs; stage1 vs stage2 decision tuning:")
    best1 = tune(V, p1v, truth, s1v["id"])
    best2 = tune(V, p2v, truth, s1v["id"])
    stage, (cfg, m) = (2, best2) if best2[1]["f05"] >= best1[1]["f05"] else (1, best1)
    log(f"BEST: stage {stage} {cfg} -> {m}")

    # per-country breakdown of the chosen configuration
    p = p2v if stage == 2 else p1v
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    if cfg["rule"] == "threshold":
        sel = D.select_threshold(ex, "p", cfg["thr"])
    else:
        ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
        sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"])
    for c in s1v["country"].unique().sort():
        ids = s1v.filter(pl.col("country") == c)["id"]
        mc = D.macro_f05(sel.filter(pl.col("s1_id").is_in(ids.implode())),
                         truth.filter(pl.col("s1_id").is_in(ids.implode())), ids)
        log(f"  {c}: F0.5 {mc['f05']:.5f} P {mc['precision_micro']:.4f} R {mc['recall_micro']:.4f} "
            f"singleton acc {mc['singleton_acc']} empty {mc['pred_empty_rate']:.4f}")

    if os.environ.get("BER_FINAL_BV", "0") == "1":
        # final models: the decision rule stays the one tuned on V above, the matcher is refit on B + V
        BV = pl.concat([B, V.select(B.columns)], how="vertical_relaxed")
        oof_bv, m1 = stage_fit(BV, cols, "final B+V stage1")
        BV2 = pl.concat([BV, context(BV, oof_bv)], how="horizontal")
        _, m2 = stage_fit(BV2, cols2, "final B+V stage2")
        if cat_w > 0:
            cat_model, _ = cat_fit(BV2, cols2, iters=cat_iters)
        log(f"final matcher refit on B + V ({BV.height:,} rows); decision rule kept from V tuning")

    m1.save_model(WORK / "xgb_stage1.json")
    m2.save_model(WORK / "xgb_stage2.json")
    if cat_w > 0:
        cat_model.save_model(str(WORK / "cat_stage2.cbm"))
    (WORK / "decision.json").write_text(json.dumps(
        {"stage": stage, "cfg": cfg, "cols": cols, "cols2": cols2, "use_xenc": xenc, "xenc_tags": tags, "namefeat": nf, "cat_w": cat_w, "valid": m}, default=float, indent=1))
    log(f"stage D done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
