"""How much would more matcher training data help? (estimates the gain of refitting on B + V)

Trains the matcher (stage 1 + 2, same recipe and features as the saved decision) on 25%, 50% and 100% of
split B (whole S1 groups) and scores split V with the saved decision rule. The gain per doubling of B
approximates what doubling the data again (B + V, as BER_FINAL_BV=1 does) would give.
Output: printed table and WORK/learning_curve.csv.
"""
import json
import time

import numpy as np
import polars as pl

import decide as D
from config import WORK
from loco import fit_predict
from train import load_feats, s1_fold


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    dec = json.loads((WORK / "decision.json").read_text())
    cfg = dec["cfg"]
    tags = dec.get("xenc_tags", [""])
    B = load_feats("B", dec.get("use_xenc", False), tags)
    V = load_feats("V", dec.get("use_xenc", False), tags)
    cols = [c for c in dec["cols"] if c in B.columns]
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split"])
    s1v = recs.filter((pl.col("src") == 1) & (pl.col("split") == "V"))["id"]
    quarter = s1_fold(B, 4, mult=2654435789)
    rows = []
    for frac, keep in ((0.25, quarter == 0), (0.5, quarter <= 1), (1.0, np.ones(B.height, bool))):
        t0 = time.time()
        Bf = B.filter(pl.Series(keep))
        p = fit_predict(Bf, V, cols, f"B x{frac}")
        ex = D.exclusive(V.select("s1_id", "rec_id").with_columns(pl.Series("p", p)))
        if cfg["rule"] == "threshold":
            sel = D.select_threshold(ex, "p", cfg["thr"])
        else:
            ex = ex.with_columns(pl.Series("p", D.calibrate(ex["p"].to_numpy(), cfg["temp"], cfg["bias"])))
            sel = D.select_expected_f(ex, "p", empty_scale=cfg["empty_scale"])
        m = D.macro_f05(sel, pairs, s1v)
        rows.append({"b_fraction": frac, "rows": Bf.height, "f05": m["f05"], "precision": m["precision_micro"],
                     "recall": m["recall_micro"]})
        log(f"B x{frac}: {Bf.height:,} rows -> V F0.5 {m['f05']:.5f} ({time.time() - t0:.0f}s)")
    res = pl.DataFrame(rows)
    res.write_csv(WORK / "learning_curve.csv")
    g1 = rows[1]["f05"] - rows[0]["f05"]
    g2 = rows[2]["f05"] - rows[1]["f05"]
    print(res, flush=True)
    log(f"gain 25%->50%: {g1:+.5f}; 50%->100%: {g2:+.5f}; expected B -> B+V (one more doubling): about {g2:+.5f} "
        f"(or {g2 * g2 / g1 if g1 > 0 else g2:+.5f} if the gains keep shrinking at the same rate)")


if __name__ == "__main__":
    main()
