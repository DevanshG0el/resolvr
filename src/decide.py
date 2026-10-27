"""Decision layer and metric.

- macro_f05: the competition metric (per-S1 F0.5, singletons count 1/0).
- exclusive: every S2/S3 record keeps only its best-scoring S1 (records match at most one S1).
- select_expected_f: per S1, keep the top-k candidates where k maximises expected F0.5 given
  calibrated probabilities (Poisson-binomial over the top-k and the rest; k = 0 means "predict
  empty", scored by the probability that none of the candidates is a match).
"""
import numpy as np
import polars as pl
import torch

BETA2 = 0.25


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred/truth: (s1_id, rec_id) pairs. s1_ids: every evaluated S1 (incl. singletons)."""
    base = pl.DataFrame({"s1_id": s1_ids.unique()})
    npred = pred.group_by("s1_id").len(name="np")
    ntrue = truth.group_by("s1_id").len(name="nt")
    tp = pred.join(truth, on=["s1_id", "rec_id"]).group_by("s1_id").len(name="tp")
    d = (
        base.join(npred, on="s1_id", how="left").join(ntrue, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left").fill_null(0)
        .with_columns(
            pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
            .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
            .otherwise(
                (1 + BETA2) * (pl.col("tp") / pl.col("np")) * (pl.col("tp") / pl.col("nt"))
                / (BETA2 * pl.col("tp") / pl.col("np") + pl.col("tp") / pl.col("nt"))
            ).alias("f")
        )
    )
    single = d.filter(pl.col("nt") == 0)
    return {
        "f05": d["f"].mean(),
        "n_s1": d.height,
        "precision_micro": d["tp"].sum() / max(1, d["np"].sum()),
        "recall_micro": d["tp"].sum() / max(1, d["nt"].sum()),
        "singleton_acc": single["f"].mean() if single.height else None,
        "nonsingleton_f": d.filter(pl.col("nt") > 0)["f"].mean(),
        "pred_empty_rate": (d["np"] == 0).mean(),
    }


def exclusive(df: pl.DataFrame, p: str = "p") -> pl.DataFrame:
    """Keep, for every record, only its highest-probability S1."""
    return df.filter(pl.col(p) == pl.col(p).max().over("rec_id")).unique("rec_id", keep="first")


def calibrate(p: np.ndarray, temp: float, bias: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (np.log(p / (1 - p)) + bias) / temp
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def select_expected_f(df: pl.DataFrame, p: str = "p", nmax: int = 12, empty_scale: float = 1.0,
                      device: str = "cuda") -> pl.DataFrame:
    """df: s1_id, rec_id, p (already exclusive). Returns the selected (s1_id, rec_id) rows."""
    if df.height == 0:
        return df.select("s1_id", "rec_id")
    d = (
        df.sort(["s1_id", p], descending=[False, True])
        .with_columns(pl.int_range(1, pl.len() + 1).over("s1_id").alias("r"))
        .filter(pl.col("r") <= nmax)
        .with_columns((pl.col("s1_id").rank("dense") - 1).cast(pl.Int64).alias("g"))
    )
    G, N = int(d["g"].max()) + 1, nmax
    dev = device if torch.cuda.is_available() else "cpu"
    P = torch.zeros((G, N), dtype=torch.float64, device=dev)
    P[torch.from_numpy(d["g"].to_numpy()).to(dev), torch.from_numpy(d["r"].to_numpy() - 1).to(dev)] = \
        torch.from_numpy(d[p].to_numpy().astype(np.float64)).to(dev)

    def shift(x):
        return torch.nn.functional.pad(x[:, :-1], (1, 0))

    A = torch.zeros((G, N + 1, N + 1), dtype=torch.float64, device=dev)  # positives among top-k
    A[:, 0, 0] = 1
    for k in range(1, N + 1):
        pk = P[:, k - 1:k]
        A[:, k] = A[:, k - 1] * (1 - pk) + shift(A[:, k - 1]) * pk
    B = torch.zeros((G, N + 1, N + 1), dtype=torch.float64, device=dev)  # positives among the rest
    B[:, N, 0] = 1
    for k in range(N - 1, -1, -1):
        pk = P[:, k:k + 1]
        B[:, k] = B[:, k + 1] * (1 - pk) + shift(B[:, k + 1]) * pk
    a = torch.arange(N + 1, dtype=torch.float64, device=dev)
    E = torch.empty((G, N + 1), dtype=torch.float64, device=dev)
    E[:, 0] = B[:, 0, 0] * empty_scale
    for k in range(1, N + 1):
        W = (1 + BETA2) * a[:, None] / (k + BETA2 * (a[:, None] + a[None, :]))  # [a, b]
        E[:, k] = torch.einsum("ga,gb,ab->g", A[:, k], B[:, k], W)
    kstar = E.argmax(1).cpu().numpy()
    return d.filter(pl.col("r") <= pl.Series(kstar[d["g"].to_numpy()])).select("s1_id", "rec_id")


def select_threshold(df: pl.DataFrame, p: str, thr: float) -> pl.DataFrame:
    return df.filter(pl.col(p) >= thr).select("s1_id", "rec_id")
