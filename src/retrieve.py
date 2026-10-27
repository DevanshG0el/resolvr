"""Stage B: neural blocking.

1. Fine-tune multilingual-e5-base as a bi-encoder on split-A matches (symmetric InfoNCE,
   in-batch negatives; batches are grouped by country+state so negatives are hard).
2. Embed every record (train: non-ghost S1 + all S2/S3; test: everything).
3. Exact kNN on GPU within each country, both directions:
     forward  S1 -> S2/S3 top K_FWD      reverse  S2/S3 -> S1 top K_REV
   Candidates = union, with cosine and both ranks kept as features.

Tokenisation runs in a background thread so the CPU (8 cores) and GPU overlap.

Outputs (in WORK): retriever/ (model), cand_train.parquet (S1 in splits B, V), cand_test.parquet
"""
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import K_FWD, K_REV, MAX_LEN, MODEL_NAME, SEED, SMOKE, WORK

EPOCHS = int(os.environ.get("BER_RET_EPOCHS", 1))
MAX_STEPS = int(os.environ.get("BER_RET_MAX_STEPS", 1000))
BATCH = int(os.environ.get("BER_RET_BATCH", 512))
EMB_BATCH = int(os.environ.get("BER_EMB_BATCH", 4096))
LR = float(os.environ.get("BER_RET_LR", 3e-5))
TAU = 0.05
DEV = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mean_pool(out, mask):
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


def tok_cpu(tok, texts):
    return tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


def to_dev(enc):
    return {k: v.to(DEV, non_blocking=True) for k, v in enc.items()}


def prefetch(tok, text_batches, depth=6):
    """Yield tokenised batches; a single worker thread tokenises ahead of the GPU.
    (One thread only: concurrent calls on one fast tokenizer raise 'Already borrowed'.)"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        it = iter(text_batches)
        q = deque()
        for tb in it:
            q.append((tb, ex.submit(lambda t: [tok_cpu(tok, x) for x in t[1]], tb)))
            if len(q) >= depth:
                break
        while q:
            tb, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt, ex.submit(lambda t: [tok_cpu(tok, x) for x in t[1]], nxt)))
            yield tb[0], fut.result()


# ----------------------------------------------------------------------------- fine-tuning
def finetune(recs: pl.DataFrame, pairs: pl.DataFrame, out_dir):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEV)
    model.train()
    text = recs.select("id", "emb_text")
    s1 = recs.filter(pl.col("src") == 1).select(
        pl.col("id").alias("s1_id"), (pl.col("country") + "|" + pl.col("state").fill_null("?")).alias("grp")
    )
    a = (
        pairs.filter(pl.col("split") == "A").join(s1, on="s1_id")
        .join(text.rename({"id": "s1_id", "emb_text": "q"}), on="s1_id")
        .join(text.rename({"id": "rec_id", "emb_text": "p"}), on="rec_id")
    )
    n_s1 = a["s1_id"].n_unique()
    total = min(MAX_STEPS, max(1, n_s1 // BATCH) * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=True)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm))
    )
    log(f"fine-tune: {a.height:,} A pairs, {n_s1:,} S1, {total} steps, batch {BATCH}")
    step, t0, t_gpu = 0, time.time(), 0.0
    rng = np.random.default_rng(SEED)
    for ep in range(EPOCHS):
        if step >= total:
            break
        # one random positive per S1; order by (country|state, random) so a batch shares a region
        ex = (
            a.with_columns(pl.Series("r", rng.random(a.height)))
            .sort("r").unique("s1_id", keep="first")
            .with_columns(pl.Series("r2", rng.random(n_s1)))
            .sort(["grp", "r2"])
        )
        q_all, p_all = ex["q"].to_list(), ex["p"].to_list()
        bounds = [(i, min(i + BATCH, len(q_all))) for i in range(0, len(q_all), BATCH)]
        bounds = [b for b in bounds if b[1] - b[0] >= 8]
        rng.shuffle(bounds)
        bounds = bounds[: total - step]
        text_batches = [(None, (q_all[b0:b1], p_all[b0:b1])) for b0, b1 in bounds]
        for _, (qe, pe) in prefetch(tok, text_batches):
            tg = time.time()
            qe, pe = to_dev(qe), to_dev(pe)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = mean_pool(model(**qe).last_hidden_state, qe["attention_mask"])
                p = mean_pool(model(**pe).last_hidden_state, pe["attention_mask"])
            q, p = F.normalize(q.float(), dim=-1), F.normalize(p.float(), dim=-1)
            logits = q @ p.T / TAU
            lab = torch.arange(len(q), device=DEV)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                acc = (logits.argmax(1) == lab).float().mean().item()
                t_gpu += time.time() - tg
                log(f"  ep {ep} step {step}/{total} loss {loss.item():.4f} in-batch acc {acc:.3f} "
                    f"({(time.time() - t0) / step:.2f}s/step overall)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"fine-tune done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- embedding
@torch.no_grad()
def embed(texts, model, tok, bs=EMB_BATCH):
    """Returns L2-normalised fp16 embeddings on the GPU, row-aligned with texts."""
    model.eval()
    lens = np.fromiter((len(t) for t in texts), dtype=np.int32, count=len(texts))
    order = np.argsort(lens, kind="stable")
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16, device=DEV)
    batches = [(order[i:i + bs], ([texts[j] for j in order[i:i + bs]],)) for i in range(0, len(order), bs)]
    t0, done = time.time(), 0
    for bi, (idx, (enc,)) in enumerate(prefetch(tok, batches)):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        out[torch.from_numpy(idx).to(DEV)] = F.normalize(e.float(), dim=-1).half()
        done += len(idx)
        if bi % 500 == 0:
            torch.cuda.synchronize()
            log(f"  embedded {done:,}/{len(texts):,} ({done / max(1e-6, time.time() - t0):,.0f}/s)")
    torch.cuda.synchronize()
    log(f"  embedded {len(texts):,} in {time.time() - t0:.0f}s")
    return out


# ----------------------------------------------------------------------------- kNN
@torch.no_grad()
def knn(Q: torch.Tensor, C: torch.Tensor, k: int):
    """Exact inner-product top-k on GPU. Q, C: normalised fp16 GPU tensors. Returns numpy."""
    k = min(k, C.shape[0])
    chunk = max(256, int(30e9 // (2 * max(1, C.shape[0]))))
    vals, idxs = [], []
    for s in range(0, Q.shape[0], chunk):
        sc = Q[s:s + chunk] @ C.T
        v, i = sc.topk(k, dim=1)
        vals.append(v.float().cpu())
        idxs.append(i.int().cpu())
        del sc
    torch.cuda.empty_cache()
    return torch.cat(vals).numpy(), torch.cat(idxs).numpy()


def candidates(recs: pl.DataFrame, emb: torch.Tensor, query_mask: np.ndarray, s1_mask: np.ndarray):
    """recs row i <-> emb row i. query_mask: S1 rows used as forward queries.
    s1_mask: all S1 rows that exist (reverse-search corpus). Returns candidate pairs (row indices)."""
    country = recs["country"].to_numpy()
    src = recs["src"].to_numpy()
    T = lambda a: torch.from_numpy(a).to(DEV)
    out = []
    for c in np.unique(country):
        in_c = country == c
        q_rows = np.where(in_c & query_mask)[0]
        s1_rows = np.where(in_c & s1_mask)[0]
        r_rows = np.where(in_c & (src != 1))[0]
        if len(q_rows) == 0 or len(r_rows) == 0:
            continue
        t0 = time.time()
        v, i = knn(emb[T(q_rows)], emb[T(r_rows)], K_FWD)
        fwd = pl.DataFrame({
            "s1_row": np.repeat(q_rows, v.shape[1]), "rec_row": r_rows[i.ravel()],
            "cos_f": v.ravel(), "rank_fwd": np.tile(np.arange(1, v.shape[1] + 1, dtype=np.int16), len(q_rows)),
        })
        v, i = knn(emb[T(r_rows)], emb[T(s1_rows)], K_REV)
        rev = pl.DataFrame({
            "rec_row": np.repeat(r_rows, v.shape[1]), "s1_row": s1_rows[i.ravel()],
            "cos_r": v.ravel(), "rank_rev": np.tile(np.arange(1, v.shape[1] + 1, dtype=np.int16), len(r_rows)),
        })
        rec_top = rev.filter(pl.col("rank_rev") == 1).select("rec_row", pl.col("cos_r").alias("rec_top1_cos"))
        rec_top2 = rev.filter(pl.col("rank_rev") == 2).select("rec_row", pl.col("cos_r").alias("rec_top2_cos"))
        rev = rev.filter(pl.col("s1_row").is_in(q_rows))
        cand = (
            fwd.join(rev, on=["s1_row", "rec_row"], how="full", coalesce=True)
            .with_columns(
                pl.coalesce("cos_f", "cos_r").alias("cos"),
                pl.col("rank_fwd").fill_null(99), pl.col("rank_rev").fill_null(99),
            ).drop("cos_f", "cos_r")
            .join(rec_top, on="rec_row", how="left").join(rec_top2, on="rec_row", how="left")
        )
        s1_top = fwd.filter(pl.col("rank_fwd") == 1).select("s1_row", pl.col("cos_f").alias("s1_top1_cos"))
        s1_top2 = fwd.filter(pl.col("rank_fwd") == 2).select("s1_row", pl.col("cos_f").alias("s1_top2_cos"))
        cand = cand.join(s1_top, on="s1_row", how="left").join(s1_top2, on="s1_row", how="left")
        log(f"  {c}: {len(q_rows):,} queries x {len(r_rows):,} records -> {cand.height:,} candidates "
            f"({time.time() - t0:.0f}s)")
        out.append(cand)
    return pl.concat(out, how="vertical_relaxed")


def recall_report(cand: pl.DataFrame, pairs: pl.DataFrame, recs: pl.DataFrame):
    gt = pairs.join(recs.select(pl.col("id").alias("s1_id"), "country"), on="s1_id")
    gt = gt.join(cand.select("s1_id", "rec_id", "rank_fwd", "rank_rev"), on=["s1_id", "rec_id"], how="left")
    found = pl.col("rank_fwd").is_not_null()
    aggs = [pl.len().alias("gt_pairs"), found.mean().alias("recall_union")]
    for k in (1, 3, 5, 10, 20, 30, 40):
        aggs.append((pl.col("rank_fwd") <= k).fill_null(False).mean().alias(f"fwd@{k}"))
    for k in (1, 2, 3, 5):
        aggs.append((pl.col("rank_rev") <= k).fill_null(False).mean().alias(f"rev@{k}"))
    for kf, kr in ((10, 1), (10, 2), (20, 2), (20, 3), (30, 3)):
        aggs.append(((pl.col("rank_fwd") <= kf) | (pl.col("rank_rev") <= kr)).fill_null(False).mean()
                    .alias(f"f{kf}|r{kr}"))
    print(gt.group_by("split", "country").agg(aggs).sort("split", "country"), flush=True)
    for kf, kr in ((40, 5), (10, 1), (10, 2), (20, 2), (20, 3), (30, 3)):
        n = cand.filter((pl.col("rank_fwd") <= kf) | (pl.col("rank_rev") <= kr)).height
        log(f"candidate pairs with fwd<={kf} | rev<={kr}: {n:,} ({n / cand['s1_id'].n_unique():.1f} per S1)")


def run_split(name, recs, emb, query_mask, s1_mask):
    cand = candidates(recs, emb, query_mask, s1_mask)
    ids = recs["id"]
    cand = cand.with_columns(
        pl.Series("s1_id", ids.gather(cand["s1_row"].to_numpy())),
        pl.Series("rec_id", ids.gather(cand["rec_row"].to_numpy())),
    ).drop("s1_row", "rec_row")
    cand.write_parquet(WORK / f"cand_{name}.parquet")
    return cand


def main():
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(250)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    cols = ["id", "src", "country", "state", "split", "emb_text"]
    recs = pl.read_parquet(WORK / "train_recs.parquet", columns=cols)
    pairs = pl.read_parquet(WORK / "train_pairs.parquet")
    ret_dir = WORK / "retriever"
    if not (ret_dir / "config.json").exists():
        finetune(recs, pairs, ret_dir)
    else:
        log("retriever already trained, reusing", ret_dir)
    tok = AutoTokenizer.from_pretrained(ret_dir)
    model = AutoModel.from_pretrained(ret_dir, dtype=torch.bfloat16).to(DEV)

    # ---- train: keep non-ghost world; forward queries only for splits B and V
    recs = recs.filter(pl.col("split").is_null() | (pl.col("split") != "ghost"))
    log(f"embedding {recs.height:,} train records")
    emb = embed(recs["emb_text"].to_list(), model, tok)
    split = recs["split"].fill_null("").to_numpy()
    is_s1 = recs["src"].to_numpy() == 1
    cand = run_split("train", recs, emb, is_s1 & np.isin(split, ["B", "V"]), is_s1)
    recall_report(cand, pairs.filter(pl.col("split").is_in(["B", "V"])), recs)
    del emb, cand
    torch.cuda.empty_cache()

    # ---- test
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=cols)
    log(f"embedding {test.height:,} test records")
    emb = embed(test["emb_text"].to_list(), model, tok)
    is_s1 = test["src"].to_numpy() == 1
    cand = run_split("test", test, emb, is_s1, is_s1)
    per = cand.group_by("s1_id").len()
    log(f"test candidates {cand.height:,}; per S1 mean {per['len'].mean():.1f}; "
        f"S1 without candidates {int(is_s1.sum()) - per.height}")
    log("stage B done")


if __name__ == "__main__":
    main()
