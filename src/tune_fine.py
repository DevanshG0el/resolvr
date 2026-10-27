"""Finer decision-rule search on split V for a saved matcher (BER_TUNE_SFX, e.g. "_sub11").

train.py's grid put empty_scale at its lowest value (0.8), so the optimum may lie outside it. This rescans
temperature x bias x empty_scale on the same V predictions, globally and per country, and prints the best
configurations with their F0.5 (no model is retrained). Output: WORK/tune_fine<sfx>.json with the best global cfg.
"""
import itertools
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

import decide as D
from config import WORK
from train import context, load_feats

SFX = os.environ.get("BER_TUNE_SFX", "_sub11")


def main():
    dec = json.loads((WORK / f"decision{SFX}.json").read_text())
    V = load_feats("V", dec.get("use_xenc", False), dec.get("xenc_tags", [""]), namefeat=dec.get("namefeat", False))
    m1 = xgb.Booster(model_file=str(WORK / f"xgb_stage1{SFX}.json"))
    p = m1.inplace_predict(V.select(dec["cols"]).to_numpy().astype(np.float32))
    if dec["stage"] == 2:
        m2 = xgb.Booster(model_file=str(WORK / f"xgb_stage2{SFX}.json"))
        p = m2.inplace_predict(pl.concat([V, context(V, p)], how="horizontal").select(dec["cols2"]).to_numpy().astype(np.float32))
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"])
    s1v = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V"))
    base = dec["cfg"]
    grid = list(itertools.product((0.85, 1.0, 1.15), (-0.5, -0.25, 0.0, 0.25, 0.5), (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
    res = {}
    for scope in ["all"] + sorted(s1v["country"].unique().to_list()):
        ids = s1v["id"] if scope == "all" else s1v.filter(pl.col("country") == scope)["id"]
        e = ex.filter(pl.col("s1_id").is_in(ids.implode()))
        t = truth.filter(pl.col("s1_id").is_in(ids.implode()))
        rows = []
        for temp, bias, es in grid:
            q = e.with_columns(pl.Series("p", D.calibrate(e["p"].to_numpy(), temp, bias)))
            f = D.macro_f05(D.select_expected_f(q, "p", empty_scale=es), t, ids)["f05"]
            rows.append({"temp": temp, "bias": bias, "empty_scale": es, "f05": f})
        r = pl.DataFrame(rows).sort("f05", descending=True)
        cur = r.filter((pl.col("temp") == base["temp"]) & (pl.col("bias") == base["bias"]) & (pl.col("empty_scale") == base["empty_scale"]))
        print(f"== {scope}: current cfg {base} -> {cur['f05'].item() if cur.height else float('nan'):.5f}; best:", flush=True)
        print(r.head(6).with_columns(pl.col("f05").round(5)), flush=True)
        res[scope] = r.row(0, named=True)
    (WORK / f"tune_fine{SFX}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
