"""Leave-one-country-out calibration of the unseen-country logit shift (teammate's proposal).

France has no labels, so the correction for an unseen country is measured on a country we can score:
train the matcher (stage 1 + 2, same recipe as train.py) on split-B pairs of the OTHER country only,
predict split V of the held-out country, and find the logit shift that maximises its macro F0.5 under
the same decision rule (exclusivity + expected-F0.5). Done for both held-out countries, with and without
the cross-encoder features (cross-encoders were trained on both countries, so the "with" variant is only
partially unseen). The in-domain model (trained on all of B) is scored the same way as a reference.

Output: WORK/loco_shift.csv and a printed summary.
"""
import json
import time

import numpy as np
import polars as pl

import decide as D
from config import WORK
from train import ID_COLS, context, load_feats, predict, stage_fit

SHIFTS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fit_predict(Btr: pl.DataFrame, Vte: pl.DataFrame, cols, tag):
    oof1, m1 = stage_fit(Btr, cols, f"{tag}/stage1")
    p1v = predict(m1, Vte.select(cols).to_numpy().astype(np.float32))
    B2 = pl.concat([Btr, context(Btr, oof1)], how="horizontal")
    V2 = pl.concat([Vte, context(Vte, p1v)], how="horizontal")
    cols2 = cols + [c for c in B2.columns if c.startswith("p1")]
    _, m2 = stage_fit(B2, cols2, f"{tag}/stage2")
    return predict(m2, V2.select(cols2).to_numpy().astype(np.float32))


def shift_curve(V: pl.DataFrame, p: np.ndarray, truth: pl.DataFrame, s1_ids: pl.Series, temp: float, es: float):
    ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
    out = {}
    for sh in SHIFTS:
        q = D.calibrate(ex["p"].to_numpy(), temp, sh * temp)  # logit shift of `sh` after calibration
        sel = D.select_expected_f(ex.with_columns(pl.Series("p", q)), "p", empty_scale=es)
        out[sh] = D.macro_f05(sel, truth, s1_ids)["f05"]
    return out


def main():
    dec = json.loads((WORK / "decision.json").read_text())
    cfg = dec["cfg"]
    temp, es = cfg.get("temp", 1.0), cfg.get("empty_scale", 1.0)
    tags = dec.get("xenc_tags", [""])
    B = load_feats("B", dec.get("use_xenc", False), tags)
    V = load_feats("V", dec.get("use_xenc", False), tags)
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country"]).filter(pl.col("src") == 1)
    country = recs.select(pl.col("id").alias("s1_id"), "country")
    B, V = B.join(country, on="s1_id"), V.join(country, on="s1_id")
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    cols_all = [c for c in dec["cols"] if c in B.columns]
    xcols = [c for c in cols_all if c.startswith("xenc")]
    variants = {"with_xenc": cols_all, "no_xenc": [c for c in cols_all if c not in xcols]}
    rows = []
    for vname, cols in variants.items():
        for held in ("India", "US"):
            s1_h = recs.filter((pl.col("split") == "V") & (pl.col("country") == held))["id"]
            truth_h = pairs.filter(pl.col("s1_id").is_in(s1_h.implode()))
            Vh = V.filter(pl.col("country") == held).drop("country")
            for mode, Btr in (("loco", B.filter(pl.col("country") != held)), ("in_domain", B)):
                t0 = time.time()
                p = fit_predict(Btr.drop("country"), Vh, cols, f"{vname}/{held}/{mode}")
                curve = shift_curve(Vh, p, truth_h, s1_h, temp, es)
                best = max(curve, key=curve.get)
                log(f"{vname:9s} held-out {held:5s} {mode:9s}: F0.5 at shift 0 = {curve[0.0]:.5f}, "
                    f"best shift {best:+.2f} -> {curve[best]:.5f} ({time.time() - t0:.0f}s)")
                rows += [{"variant": vname, "held_out": held, "mode": mode, "shift": s, "f05": f}
                         for s, f in curve.items()]
    res = pl.DataFrame(rows)
    res.write_csv(WORK / "loco_shift.csv")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(-1)
    print(res.pivot(on="shift", index=["variant", "held_out", "mode"], values="f05")
          .with_columns(pl.exclude("variant", "held_out", "mode").round(5)), flush=True)


if __name__ == "__main__":
    main()
