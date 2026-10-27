"""Model comparison on the held-out validation split.

All models score the same candidate pairs (the output of blocking) with the same features:
  train  = split B (S1 entities disjoint from V)
  tune   = half of split V (picks each model's decision threshold)
  test   = the other half of split V (every number reported below)

Each model's pair scores go through the same decision step (every S2/S3 record is kept for its
best S1 only, then a global probability threshold tuned on V-tune), except the final pipeline,
which also uses the expected-F0.5 selection rule from src/decide.py.

Outputs: results/model_comparison.csv, results/per_country.csv, results/curves.npz,
         results/importance.csv
Usage:   python experiments/compare_models.py   (needs the work/ artefacts of the pipeline)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import decide as D  # noqa: E402
from config import SEED, WORK  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
FULL = json.loads((WORK / "decision_final.json").read_text())["cols"]  # what the final matcher uses
XENC = ["xenc", "xenc_bge", "xenc_qwen", "xenc_qwen2"]  # e5 / bge-reranker / Qwen3-Reranker cross-encoder scores
FEATS = [c for c in FULL if c not in XENC and not c.startswith("rec_n") and not c.startswith("rec_nm")]  # hand features


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    f = pl.read_parquet(WORK / f"feat_{split}.parquet")
    nf = pl.read_parquet(WORK / f"namefeat_{split}.parquet")
    f = f.drop("label", strict=False).join(nf.drop("label", strict=False), on=["s1_id", "rec_id"], how="left")
    for x in XENC:
        xe = pl.read_parquet(WORK / f"{x}_{split}.parquet")
        f = f.join(xe.select("s1_id", "rec_id", x), on=["s1_id", "rec_id"], how="left")
    lab = pairs.select("s1_id", "rec_id", pl.lit(1, pl.Int8).alias("label"))
    return f.join(lab, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("label").fill_null(0))


def matrix(df: pl.DataFrame, cols) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32).fill_null(-1).fill_nan(-1) for c in cols]).to_numpy()


def s1_half(ids: pl.Series) -> np.ndarray:
    return ((ids.str.slice(3).cast(pl.Int64) * 2246822519 // 8192) % 2 == 0).to_numpy()


def decide(df: pl.DataFrame, p: np.ndarray, thr: float) -> pl.DataFrame:
    d = df.select("s1_id", "rec_id").with_columns(pl.Series("p", p))
    return D.exclusive(d.filter(pl.col("p") >= thr)).select("s1_id", "rec_id")


def tune_threshold(df, p, truth, s1_ids):
    best = (-1, 0.5)
    for thr in np.r_[np.linspace(0.05, 0.95, 19), 0.97, 0.98, 0.99]:
        f = D.macro_f05(decide(df, p, thr), truth, s1_ids)["f05"]
        best = max(best, (f, thr))
    return best[1]


# ---------------------------------------------------------------- models
def fit_logreg(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_rf(X, y):
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(len(y), 1_000_000), replace=False)  # RAM/CPU budget
    m = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=5, n_jobs=8, random_state=SEED)
    m.fit(X[idx], y[idx])
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_mlp(X, y):
    import torch
    torch.manual_seed(SEED)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    dev = "cuda"
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 1)).to(dev)
    Xt = torch.tensor((X - mu) / sd, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    for epoch in range(6):
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(yt), 8192):
            b = perm[i:i + 8192]
            opt.zero_grad()
            lossf(net(Xt[b]).squeeze(1), yt[b]).backward()
            opt.step()
    net.eval()

    def pred(Z):
        with torch.no_grad():
            Zt = torch.tensor((Z - mu) / sd, device=dev)
            return torch.cat([torch.sigmoid(net(Zt[i:i + 65536]).squeeze(1)) for i in range(0, len(Zt), 65536)]).cpu().numpy()
    return pred


def fit_lgbm(X, y):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, n_jobs=8, random_state=SEED, verbose=-1)
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_cat(X, y):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.08, task_type="GPU", random_seed=SEED, verbose=0)
    m.fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def fit_xgb(X, y, rounds=800, keep=None):
    import xgboost as xgb
    params = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=8, eta=0.08,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=SEED)
    bst = xgb.train(params, xgb.DMatrix(X, label=y), rounds)
    if keep is not None:
        keep["bst"] = bst
    return lambda Z: bst.predict(xgb.DMatrix(Z))


def fit_two_stage(B, cols):
    """The project's matcher: XGBoost stage 1 (out-of-fold) -> per-S1 context -> XGBoost stage 2."""
    import train as T
    X, y = matrix(B, cols), B["label"].to_numpy()
    fold = T.s1_fold(B, 2)
    p1 = np.zeros(len(y), dtype=np.float32)
    for k in range(2):
        tr = fold != k
        p1[~tr] = fit_xgb(X[tr], y[tr])(X[~tr])
    s1 = fit_xgb(X, y)
    ctx_cols = None

    def ctx(df, p):
        c = T.context(df, p)
        return c.to_numpy().astype(np.float32), c.columns
    C, ctx_cols = ctx(B, p1)
    keep = {}
    s2 = fit_xgb(np.hstack([X, C]), y, keep=keep)

    def pred(df):
        Z = matrix(df, cols)
        Cz, _ = ctx(df, s1(Z))
        return s2(np.hstack([Z, Cz]))
    return pred, keep["bst"], cols + list(ctx_cols)


def main():
    t0 = time.time()
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    s1 = (pl.scan_parquet(WORK / "train_recs.parquet").filter(pl.col("src") == 1)
          .select(pl.col("id").alias("s1_id"), "split", "country").collect())
    B = load("B", pairs.filter(pl.col("split") == "B"))
    V = load("V", pairs.filter(pl.col("split") == "V"))
    log(f"B {B.height:,} pairs ({B['label'].sum():,} pos), V {V.height:,} pairs ({V['label'].sum():,} pos)")

    v_s1 = s1.filter(pl.col("split") == "V")
    tune_mask_s1 = s1_half(v_s1["s1_id"])
    s1_tune, s1_test = v_s1["s1_id"].filter(tune_mask_s1), v_s1["s1_id"].filter(~tune_mask_s1)
    truth = pairs.filter(pl.col("split") == "V").select("s1_id", "rec_id")
    truth_tune = truth.filter(pl.col("s1_id").is_in(s1_tune.implode()))
    truth_test = truth.filter(pl.col("s1_id").is_in(s1_test.implode()))
    in_tune = V["s1_id"].is_in(s1_tune.implode()).to_numpy()
    Vt, Ve = V.filter(pl.Series(in_tune)), V.filter(pl.Series(~in_tune))
    ye = Ve["label"].to_numpy()
    ctry = s1.select("s1_id", "country")
    ceiling = D.macro_f05(Ve.filter(pl.col("label") == 1).select("s1_id", "rec_id"), truth_test, s1_test)
    log(f"V-test: {len(s1_test):,} S1 entities, blocking recall ceiling F0.5 {ceiling['f05']:.4f}")

    Xb, yb = matrix(B, FEATS), B["label"].to_numpy()
    Xt, Xe = matrix(Vt, FEATS), matrix(Ve, FEATS)
    rows, per_country, curves, scores = [], [], {}, {}

    def evaluate(name, family, pt, pe, train_s, extra=None):
        t = time.time()
        thr = tune_threshold(Vt, pt, truth_tune, s1_tune)
        pred = decide(Ve, pe, thr)
        if extra is not None:
            pred = extra
        m = D.macro_f05(pred, truth_test, s1_test)
        fpr, tpr, _ = roc_curve(ye, pe)
        pr, rc, _ = precision_recall_curve(ye, pe)
        step = max(1, len(pr) // 2000)
        curves[name] = dict(fpr=fpr[::max(1, len(fpr) // 2000)], tpr=tpr[::max(1, len(tpr) // 2000)], pr=pr[::step], rc=rc[::step])
        yhat = (pe >= thr).astype(int)
        tp, fp, fn = int(((yhat == 1) & (ye == 1)).sum()), int(((yhat == 1) & (ye == 0)).sum()), int(((yhat == 0) & (ye == 1)).sum())
        tn = len(ye) - tp - fp - fn
        row = dict(model=name, family=family, roc_auc=roc_auc_score(ye, pe), pr_auc=average_precision_score(ye, pe),
                   threshold=thr, pair_precision=tp / max(1, tp + fp), pair_recall=tp / max(1, tp + fn),
                   pair_f1=2 * tp / max(1, 2 * tp + fp + fn), tp=tp, fp=fp, fn=fn, tn=tn,
                   entity_f05=m["f05"], entity_precision=m["precision_micro"], entity_recall=m["recall_micro"],
                   singleton_acc=m["singleton_acc"], train_s=train_s)
        rows.append(row)
        scores[name] = pe
        pc = pred.join(ctry, on="s1_id")
        for c in ("India", "US"):
            ids = s1_test.to_frame().join(ctry, on="s1_id").filter(pl.col("country") == c)["s1_id"]
            mc = D.macro_f05(pc.filter(pl.col("country") == c).select("s1_id", "rec_id"),
                             truth_test.filter(pl.col("s1_id").is_in(ids.implode())), ids)
            per_country.append(dict(model=name, country=c, f05=mc["f05"], precision=mc["precision_micro"], recall=mc["recall_micro"]))
        log(f"{name:38s} AUC {row['roc_auc']:.4f} PR-AUC {row['pr_auc']:.4f} F0.5 {m['f05']:.4f} "
            f"P {m['precision_micro']:.4f} R {m['recall_micro']:.4f} (train {train_s:.0f}s, eval {time.time() - t:.0f}s)")
        pl.DataFrame(rows).write_csv(RES / "model_comparison.csv")
        pl.DataFrame(per_country).write_csv(RES / "per_country.csv")

    # unsupervised scores
    col = lambda df, c: df[c].cast(pl.Float32).fill_null(0).to_numpy()
    evaluate("Fuzzy string rule (RapidFuzz)", "Rule-based", (col(Vt, "nm_tset") + col(Vt, "ad_tset")) / 200,
             (col(Ve, "nm_tset") + col(Ve, "ad_tset")) / 200, 0)
    evaluate("Bi-encoder cosine (fine-tuned e5)", "Transformer (bi-encoder)", col(Vt, "cos"), col(Ve, "cos"), 0)
    evaluate("Cross-encoder (fine-tuned e5)", "Transformer (cross-encoder)", col(Vt, "xenc"), col(Ve, "xenc"), 0)
    evaluate("BGE-reranker-v2-m3 (fine-tuned)", "Transformer (cross-encoder)", col(Vt, "xenc_bge"), col(Ve, "xenc_bge"), 0)

    for name, family, fit in (
        ("Logistic Regression", "Linear", fit_logreg),
        ("Random Forest", "Bagged trees", fit_rf),
        ("MLP (3-layer, PyTorch)", "Neural network", fit_mlp),
        ("LightGBM", "Gradient boosting", fit_lgbm),
        ("CatBoost", "Gradient boosting", fit_cat),
        ("XGBoost", "Gradient boosting", fit_xgb),
    ):
        t = time.time()
        f = fit(Xb, yb)
        tr = time.time() - t
        evaluate(name, family, f(Xt), f(Xe), tr)

    # the project's full matcher: extra features + two stages + expected-F0.5 decision
    cols = FULL
    t = time.time()
    f, bst, all_cols = fit_two_stage(B, cols)
    tr = time.time() - t
    pt, pe = f(Vt), f(Ve)
    evaluate("Ours: 2-stage XGBoost + cross-encoder", "Hybrid (ours)", pt, pe, tr)
    cfg = json.loads((WORK / "decision_final.json").read_text())["cfg"]
    d = Ve.select("s1_id", "rec_id").with_columns(pl.Series("p", D.calibrate(pe, cfg["temp"], cfg["bias"])))
    sel = D.select_expected_f(D.exclusive(d), empty_scale=cfg["empty_scale"]).select("s1_id", "rec_id")
    evaluate("Ours + expected-F0.5 decision", "Hybrid (ours)", pt, pe, tr, extra=sel)

    # hard pairs: the uncertain band the Qwen3-Reranker-4B (LoRA) scored; pair-level AUC of every score there
    hard = Ve["xenc_qwen"].is_not_null().to_numpy()
    yh = ye[hard]
    hard_rows = [dict(model=k, roc_auc=roc_auc_score(yh, v[hard]), pr_auc=average_precision_score(yh, v[hard]))
                 for k, v in scores.items()]
    hard_rows.append(dict(model="Qwen3-Reranker-4B (LoRA)", roc_auc=roc_auc_score(yh, col(Ve, "xenc_qwen")[hard]),
                          pr_auc=average_precision_score(yh, col(Ve, "xenc_qwen")[hard])))
    pl.DataFrame(hard_rows).with_columns(pl.lit(int(hard.sum())).alias("n_pairs"), pl.lit(int(yh.sum())).alias("n_pos")
                                         ).write_csv(RES / "hard_pairs.csv")
    log(f"hard pairs {hard.sum():,} ({yh.mean():.3f} positive)")
    imp = bst.get_score(importance_type="gain")
    pl.DataFrame({"feature": [all_cols[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k for k in imp],
                  "gain": list(imp.values())}).sort("gain", descending=True).write_csv(RES / "importance.csv")
    np.savez_compressed(RES / "curves.npz", **{f"{k}|{a}": v for k, c in curves.items() for a, v in c.items()})
    (RES / "ceiling.json").write_text(json.dumps(ceiling, indent=1))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
