"""Recompute each indexed S2/S3 record's top-10 reverse neighbours in the sampled S1 pool, with their row ids
(rev_top.npy, rev_idx.npy). The live matcher needs the ids to drop the S1 a query is a variant of."""
import numpy as np
import polars as pl
import torch

from engine import DATA

recs = pl.read_parquet(DATA / "index.parquet", columns=["id", "src", "country"])
E = torch.from_numpy(np.load(DATA / "emb.npy")).cuda()
country, src = recs["country"].to_numpy(), recs["src"].to_numpy()
top = np.full((recs.height, 10), -1.0, dtype=np.float32)
idx = np.full((recs.height, 10), -1, dtype=np.int64)
for c in np.unique(country):
    q = np.where((country == c) & (src != 1))[0]
    p = np.where((country == c) & (src == 1))[0]
    P = E[torch.from_numpy(p).cuda()]
    for i in range(0, len(q), 8192):
        v, j = (E[torch.from_numpy(q[i:i + 8192]).cuda()] @ P.T).topk(min(10, len(p)), dim=1)
        top[q[i:i + 8192]] = v.float().cpu().numpy()
        idx[q[i:i + 8192]] = p[j.cpu().numpy()]
np.save(DATA / "rev_top.npy", top)
np.save(DATA / "rev_idx.npy", idx)
print("done", top.shape)
