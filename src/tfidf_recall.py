"""Could a sparse TF-IDF candidate channel (Sparkly-style) recover the true pairs our blocking misses? (split V)

Missed true pairs of split V are of two kinds: never retrieved by the dense bi-encoder (not in the forward
top-20 / reverse top-3 base set) and retrieved but removed by the learned blocking filter. For each such record,
TF-IDF (hashed word unigrams of name + address, plus character 3-grams of the core name; binary tf, idf over the
country's S1 entities, L2-normalised) scores ALL S1 entities of its country; we report the rank of the true S1
(recall@k). CPU only. Output: WORK/tfidf_recall.csv and a printed table.
"""
import time
import zlib

import numpy as np
import polars as pl
import scipy.sparse as sp

from config import WORK

H = 1 << 20
KS = (1, 5, 10, 20, 50)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def feats(name: str, addr: str) -> list:
    toks = [f"w:{t}" for t in (name + " " + addr).split() if t]
    s = f" {name} "
    grams = [f"c:{s[i:i + 3]}" for i in range(len(s) - 2)]
    return list({zlib.crc32(t.encode()) % H for t in toks + grams})


def matrix(names, addrs) -> sp.csr_matrix:
    indptr, idx = [0], []
    for n, a in zip(names, addrs):
        f = feats(n or "", a or "")
        idx.extend(f)
        indptr.append(len(idx))
    data = np.ones(len(idx), dtype=np.float32)
    return sp.csr_matrix((data, np.array(idx, dtype=np.int64), np.array(indptr, dtype=np.int64)), shape=(len(indptr) - 1, H))


def l2(m: sp.csr_matrix) -> sp.csr_matrix:
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel()) + 1e-9
    return sp.diags(1 / norm).dot(m).tocsr()


def main():
    t0 = time.time()
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=["id", "src", "split", "country", "name_core", "addr_clean"])
    truth = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    base = pl.read_parquet(WORK / "cand_train.parquet", columns=["s1_id", "rec_id", "rank_fwd", "rank_rev"]).filter(
        (pl.col("rank_fwd") <= 20) | (pl.col("rank_rev") <= 3)).select("s1_id", "rec_id").with_columns(pl.lit(True).alias("in_base"))
    kept = pl.read_parquet(WORK / "feat_V.parquet", columns=["s1_id", "rec_id"]).with_columns(pl.lit(True).alias("in_cand"))
    t = truth.join(base, on=["s1_id", "rec_id"], how="left").join(kept, on=["s1_id", "rec_id"], how="left")
    t = t.with_columns(pl.when(pl.col("in_cand").fill_null(False)).then(pl.lit("found"))
                       .when(pl.col("in_base").fill_null(False)).then(pl.lit("removed by blocking filter"))
                       .otherwise(pl.lit("never retrieved")).alias("kind"))
    log("true V pairs: " + ", ".join(f"{k} {n:,}" for k, n in t.group_by("kind").len().iter_rows()))
    miss = t.filter(pl.col("kind") != "found")
    info = recs.select("id", "country", "name_core", "addr_clean")
    miss = miss.join(info.rename({"id": "s1_id"}).select("s1_id", "country"), on="s1_id").join(
        info.rename({"id": "rec_id", "name_core": "rn", "addr_clean": "ra"}).drop("country"), on="rec_id")
    miss = miss.with_columns((pl.col("ra").fill_null("") == "").alias("addr_empty"))
    rows = []
    for c in miss["country"].unique().sort().to_list():
        s1 = recs.filter((pl.col("src") == 1) & (pl.col("country") == c))
        S = matrix(s1["name_core"].to_list(), s1["addr_clean"].to_list())
        df = np.asarray((S > 0).sum(axis=0)).ravel()
        idf = np.log((1 + S.shape[0]) / (1 + df)).astype(np.float32) + 1
        D = sp.diags(idf)
        S = l2((S @ D).tocsr())
        ST = S.T.tocsr()  # transposed once: each query block is a CSR x CSR product
        pos = {k: i for i, k in enumerate(s1["id"].to_list())}
        m = miss.filter(pl.col("country") == c)
        Q = l2((matrix(m["rn"].to_list(), m["ra"].to_list()) @ D).tocsr())
        tgt = np.array([pos[x] for x in m["s1_id"].to_list()])
        ranks = np.empty(len(tgt), dtype=np.int64)
        for i in range(0, Q.shape[0], 256):
            block = (Q[i:i + 256] @ ST).toarray()
            true_s = block[np.arange(block.shape[0]), tgt[i:i + 256]]
            ranks[i:i + 256] = (block > true_s[:, None]).sum(axis=1) + 1
        rows.append(m.select("kind", "addr_empty", "country").with_columns(pl.Series("rank", ranks)))
        log(f"{c}: {S.shape[0]:,} S1 entities, {len(tgt):,} missed true pairs scored ({time.time() - t0:.0f}s)")
    r = pl.concat(rows)
    out = r.group_by("country", "kind", "addr_empty").agg(
        pl.len().alias("missed_true_pairs"), *[(pl.col("rank") <= k).mean().round(3).alias(f"recall@{k}") for k in KS]
    ).sort("country", "kind", "addr_empty")
    out.write_csv(WORK / "tfidf_recall.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(220)
    print(out, flush=True)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
