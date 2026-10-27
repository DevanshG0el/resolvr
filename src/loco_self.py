"""Does adding pseudo-labeled rows of an unseen country to the matcher's training help? (LOCO simulation)

Mirrors what BER_TRAIN_EXTRA=fr_pseudo_train.parquet does for France, on a country with labels:
1. train the matcher (stage 1 + 2) on split-B pairs of the OTHER country only, predict split V of the
   held-out country (as the real pipeline predicts France without French labels);
2. pseudo-label the held-out V pairs exactly as pseudo.french_xgb_rows does: keep only S1s whose every
   candidate is confident (p >= POS_P or p <= NEG_P) and whose positives are each record's best S1;
3. retrain on other-country B + those pseudo rows, predict the same held-out V again and compare macro F0.5
   (transductive, like the test set). Pseudo-label precision/recall against the true labels is logged.

Output: WORK/loco_self.csv and a printed summary.
"""
import json
import time

import numpy as np
import polars as pl

from config import WORK
from loco import SHIFTS, fit_predict, shift_curve
from pseudo import NEG_P, POS_P
from train import load_feats


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pseudo_rows(Vh: pl.DataFrame, p: np.ndarray, truth: pl.DataFrame):
    sc = Vh.select("s1_id", "rec_id").with_columns(pl.Series("p", p))
    sc = sc.with_columns(pl.col("p").max().over("rec_id").alias("rec_best"))
    ok = sc.group_by("s1_id").agg(
        ((pl.col("p") >= POS_P) | (pl.col("p") <= NEG_P)).all().alias("clean"),
        ((pl.col("p") >= POS_P) & (pl.col("p") < pl.col("rec_best"))).any().alias("conflict"),
    ).filter(pl.col("clean") & ~pl.col("conflict")).select("s1_id")
    lab = sc.join(ok, on="s1_id").select("s1_id", "rec_id", (pl.col("p") >= POS_P).cast(pl.Int8).alias("plabel"))
    chk = lab.join(truth.with_columns(pl.lit(1, pl.Int8).alias("true")), on=["s1_id", "rec_id"], how="left").fill_null(0)
    npos = int(chk["plabel"].sum())
    prec = float(((chk["plabel"] == 1) & (chk["true"] == 1)).sum() / max(npos, 1))
    missed = int(((chk["plabel"] == 0) & (chk["true"] == 1)).sum())
    n_s1 = Vh["s1_id"].n_unique()
    stats = {"clean_s1": ok.height, "clean_share": ok.height / n_s1, "pseudo_pos": npos, "pos_precision": prec,
             "true_pos_labeled_neg": missed}
    E = Vh.join(lab, on=["s1_id", "rec_id"]).drop("label").rename({"plabel": "label"})
    return E, stats


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
            t0 = time.time()
            s1_h = recs.filter((pl.col("split") == "V") & (pl.col("country") == held))["id"]
            truth_h = pairs.filter(pl.col("s1_id").is_in(s1_h.implode()))
            Vh = V.filter(pl.col("country") == held).drop("country")
            Btr = B.filter(pl.col("country") != held).drop("country")
            p0 = fit_predict(Btr, Vh, cols, f"{vname}/{held}/loco")
            c0 = shift_curve(Vh, p0, truth_h, s1_h, temp, es)
            E, st = pseudo_rows(Vh, p0, truth_h)
            p1 = fit_predict(pl.concat([Btr, E.select(Btr.columns)], how="vertical_relaxed"), Vh, cols,
                             f"{vname}/{held}/self")
            c1 = shift_curve(Vh, p1, truth_h, s1_h, temp, es)
            log(f"{vname:9s} held-out {held:5s}: pseudo {st['clean_s1']:,} clean S1 ({st['clean_share']:.1%}), "
                f"{st['pseudo_pos']:,} pos at precision {st['pos_precision']:.4f}, {st['true_pos_labeled_neg']:,} true "
                f"pairs labeled negative | F0.5 shift 0: loco {c0[0.0]:.5f} -> self {c1[0.0]:.5f} "
                f"({c1[0.0] - c0[0.0]:+.5f}); best shift loco {max(c0, key=c0.get):+.2f}, "
                f"self {max(c1, key=c1.get):+.2f} ({time.time() - t0:.0f}s)")
            for mode, curve in (("loco", c0), ("self", c1)):
                rows += [{"variant": vname, "held_out": held, "mode": mode, "shift": s, "f05": f, **st}
                         for s, f in curve.items()]
            pl.DataFrame(rows).write_csv(WORK / "loco_self.csv")
    res = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(-1)
    print(res.filter(pl.col("shift").is_in([-1.0, -0.5, 0.0, 0.5])).pivot(
        on="shift", index=["variant", "held_out", "mode"], values="f05").with_columns(
        pl.exclude("variant", "held_out", "mode").round(5)), flush=True)


if __name__ == "__main__":
    main()
