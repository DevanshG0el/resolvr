"""Train-vs-test feature drift per country (split V vs test US / India / France) for the matcher's inputs.

Population stability index (PSI) per feature: bins are the deciles of V (US+India); PSI < 0.1 stable,
0.1-0.2 moderate, > 0.2 shifted. Count/frequency features (namesake counts, word document frequencies) are
computed within each dataset's population, and test has fewer S1 entities than train (France only 259k), so
those are the ones to watch. Also: candidates per record. Output: WORK/drift.csv and a printed table.
"""
import numpy as np
import polars as pl

from config import WORK

FEATS = ["a_n_s1_same_name", "b_n_s1_same_name", "a_n_rec_same_name", "a_n_s1_same_name_state", "common_logdf",
         "rare_logdf", "unseen_tok_share", "n_cand_s1", "cos", "rank_fwd", "rank_rev", "rec_top1_cos",
         "nm_tset", "nm_ratio", "ad_tset", "num1_eq", "rec_addr_empty", "state_eq", "legal_eq", "a_len", "b_len"]


def psi(ref: np.ndarray, x: np.ndarray, bins: int = 10) -> float:
    ref, x = ref[~np.isnan(ref)], x[~np.isnan(x)]
    if len(ref) == 0 or len(x) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare category shares
        vals = np.unique(np.concatenate([ref, x]))
        r = np.array([(ref == v).mean() for v in vals]) + 1e-6
        q = np.array([(x == v).mean() for v in vals]) + 1e-6
        return float(np.sum((q - r) * np.log(q / r)))
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref) + 1e-6
    q = np.histogram(np.clip(x, edges[0], edges[-1]), edges)[0] / len(x) + 1e-6
    return float(np.sum((q - r) * np.log(q / r)))


def main():
    v = pl.read_parquet(WORK / "feat_V.parquet")
    t = pl.read_parquet(WORK / "feat_test.parquet")
    cols = [c for c in FEATS if c in v.columns and c in t.columns]
    rv = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "country"])
    rt = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "country"])
    v = v.join(rv.rename({"id": "s1_id"}), on="s1_id")
    t = t.join(rt.rename({"id": "s1_id"}), on="s1_id")
    v = v.with_columns(pl.len().over("rec_id").alias("cand_per_record"))
    t = t.with_columns(pl.len().over("rec_id").alias("cand_per_record"))
    cols = cols + ["cand_per_record"]
    groups = {"V US": v.filter(pl.col("country") == "US"), "V India": v.filter(pl.col("country") == "India"),
              "test US": t.filter(pl.col("country") == "US"), "test India": t.filter(pl.col("country") == "India"),
              "test France": t.filter(pl.col("country") == "France")}
    rows = []
    for c in cols:
        ref_all = v[c].cast(pl.Float64).to_numpy()
        row = {"feature": c}
        for g, d in groups.items():
            x = d[c].cast(pl.Float64).to_numpy()
            row[f"{g} median"] = float(np.nanmedian(x)) if len(x) else float("nan")
        row["PSI test US vs V US"] = psi(groups["V US"][c].cast(pl.Float64).to_numpy(), groups["test US"][c].cast(pl.Float64).to_numpy())
        row["PSI test India vs V India"] = psi(groups["V India"][c].cast(pl.Float64).to_numpy(), groups["test India"][c].cast(pl.Float64).to_numpy())
        row["PSI test France vs V"] = psi(ref_all, groups["test France"][c].cast(pl.Float64).to_numpy())
        rows.append(row)
    res = pl.DataFrame(rows).sort("PSI test France vs V", descending=True)
    res.write_csv(WORK / "drift.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    print(res, flush=True)


if __name__ == "__main__":
    main()
