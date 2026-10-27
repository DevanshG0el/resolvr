"""Random search over XGBoost hyperparameters for the US/India matcher (split B -> V, variant c inputs).

Each configuration trains stage 1 + stage 2 exactly as train.py does (S1-grouped 2-fold out-of-fold
context, early stopping on an S1-grouped slice of the training fold) and is scored on V with the decision
rule of the reference model (BER_HP_REF, default _mxc), so configurations are compared on the metric
itself. Output: WORK/hp_search.csv and WORK/hp_best.json (pass to train.py via BER_XGB_PARAMS).
"""
import json
import os
import random
import time

import numpy as np
import polars as pl

import decide as D
import train as T
from config import SEED, WORK

N_TRIALS = int(os.environ.get("BER_HP_TRIALS", 8))
REF = os.environ.get("BER_HP_REF", "_mxc")
SPACE = {
    "max_depth": [6, 7, 8, 9, 10],
    "eta": [0.04, 0.06, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def score(B, V, cols, cfg, truth, s1v):
    oof1, m1 = T.stage_fit(B, cols, "hp/stage1")
    p1v = T.predict(m1, V.select(cols).to_numpy().astype(np.float32))
    B2 = pl.concat([B, T.context(B, oof1)], how="horizontal")
    V2 = pl.concat([V, T.context(V, p1v)], how="horizontal")
    cols2 = cols + [c for c in B2.columns if c.startswith("p1")]
    _, m2 = T.stage_fit(B2, cols2, "hp/stage2")
    p = T.predict(m2, V2.select(cols2).to_numpy().astype(np.float32))
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
    return D.macro_f05(D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"]), truth, s1v)["f05"]


def main():
    ref = json.loads((WORK / f"decision{REF}.json").read_text())
    tags = ref["xenc_tags"]
    B = T.load_feats("B", True, tags, namefeat=ref.get("namefeat", False))
    V = T.load_feats("V", True, tags, namefeat=ref.get("namefeat", False))
    cols = [c for c in ref["cols"] if c in B.columns]
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split"])
    s1v = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V"))["id"]
    base = dict(T.PARAMS)
    rng = random.Random(SEED)
    trials = [{}]  # trial 0 = the current parameters (reference, same seed)
    seen = set()
    while len(trials) < N_TRIALS + 1:
        t = {k: rng.choice(v) for k, v in SPACE.items()}
        key = tuple(sorted(t.items()))
        if key not in seen:
            seen.add(key)
            trials.append(t)
    rows = []
    for i, t in enumerate(trials):
        T.PARAMS.clear()
        T.PARAMS.update(base)
        T.PARAMS.update(t)
        t0 = time.time()
        f = score(B, V, cols, ref["cfg"], truth, s1v)
        rows.append({"trial": i, **{k: T.PARAMS[k] for k in SPACE}, "f05": f, "seconds": round(time.time() - t0)})
        log(f"trial {i}: {t or 'current parameters'} -> V F0.5 {f:.5f} ({time.time() - t0:.0f}s)")
        pl.DataFrame(rows).write_csv(WORK / "hp_search.csv")
    res = pl.DataFrame(rows).sort("f05", descending=True)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    print(res.with_columns(pl.col("f05").round(5)), flush=True)
    best = res.row(0, named=True)
    (WORK / "hp_best.json").write_text(json.dumps({k: best[k] for k in SPACE}))
    log(f"best: {({k: best[k] for k in SPACE})} -> {best['f05']:.5f} (current parameters: "
        f"{res.filter(pl.col('trial') == 0)['f05'].item():.5f})")


if __name__ == "__main__":
    main()
