"""Stage C2 (optional): cross-encoder re-scoring of the non-trivial candidate pairs.

A sequence-pair classifier initialised from the fine-tuned retriever (multilingual-e5-base,
XLM-R architecture, new 1-logit head) reads "S1 text </s></s> record text" jointly.
- Trained on split B1 (half of B's entities): all positives + stage-1 hard negatives + 5% easy negatives.
- Scores pairs with stage-1 probability >= XENC_MIN_P on B2 (the other half, used to train the
  stacker), V and test. The stacker therefore never sees scores produced on the cross-encoder's
  own training data.
Cross-fitting (default): a second model trained on the other half of B scores the first half, so the
stacker can use all of B; V and test get the mean of both models' logits.
Output: WORK/xenc_{B,V,test}.parquet (s1_id, rec_id, xenc), WORK/xenc_meta.json, WORK/xenc_h{0,1}/ (models).
"""
import json
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import xgboost as xgb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import SEED, WORK
from retrieve import to_dev

MIN_P = float(os.environ.get("BER_XENC_MIN_P", 0.002))
BATCH = int(os.environ.get("BER_XENC_BATCH", 256))
INF_BATCH = int(os.environ.get("BER_XENC_INF_BATCH", 1024))
LR = float(os.environ.get("BER_XENC_LR", 2e-5))
MAX_LEN = int(os.environ.get("BER_XENC_MAX_LEN", 160))
MAX_STEPS = int(os.environ.get("BER_XENC_MAX_STEPS", 0))  # 0 = one full pass
# A second, differently-initialised cross-encoder can be added as an extra feature: set BER_XENC_TAG
# (e.g. "_bge") and BER_XENC_BASE (e.g. BAAI/bge-reranker-v2-m3). Default: the fine-tuned retriever.
TAG = os.environ.get("BER_XENC_TAG", "")
BASE = os.environ.get("BER_XENC_BASE", "")
DEV = "cuda"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def b_half(s1_ids: pl.Series) -> np.ndarray:
    """0/1 half of split B, independent of the train.py folds."""
    return ((s1_ids.str.slice(3).cast(pl.Int64) * 3266489917 // 8192) % 2).to_numpy()


def tok_pairs(tok, a, b):
    return tok(a, b, padding=True, truncation="longest_first", max_length=MAX_LEN, return_tensors="pt")


def pair_prefetch(tok, batches, depth=6):
    """batches: list of (payload, (texts_a, texts_b)); tokenised ahead in one worker thread."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque

    with ThreadPoolExecutor(max_workers=1) as ex:
        it = iter(batches)
        q = deque()
        for b in it:
            q.append((b[0], ex.submit(tok_pairs, tok, *b[1])))
            if len(q) >= depth:
                break
        while q:
            payload, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt[0], ex.submit(tok_pairs, tok, *nxt[1])))
            yield payload, fut.result()


def stage1_p(name: str, dec: dict, m1) -> pl.DataFrame:
    F_ = pl.read_parquet(WORK / f"feat_{name}.parquet", columns=["s1_id", "rec_id"] + dec["cols"]
                         + (["label"] if name != "test" else []))
    p = m1.inplace_predict(F_.select(dec["cols"]).to_numpy().astype(np.float32))
    keep = ["s1_id", "rec_id"] + (["label"] if name != "test" else [])
    return F_.select(keep).with_columns(pl.Series("p1", p))


def attach_text(df: pl.DataFrame, split: str) -> pl.DataFrame:
    t = pl.read_parquet(WORK / f"{split}_recs.parquet", columns=["id", "emb_text"]).with_columns(
        pl.col("emb_text").str.replace(r"^query: ", "")
    )
    return df.join(t.rename({"id": "s1_id", "emb_text": "ta"}), on="s1_id").join(
        t.rename({"id": "rec_id", "emb_text": "tb"}), on="rec_id")


def train_xenc(tr: pl.DataFrame, out_dir):
    base = BASE or str(WORK / "retriever")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(DEV)
    model.train()
    rng = np.random.default_rng(SEED)
    tr = tr.sample(fraction=1.0, shuffle=True, seed=SEED)
    a, b, y = tr["ta"].to_list(), tr["tb"].to_list(), tr["label"].to_numpy().astype(np.float32)
    batches = [(y[i:i + BATCH], (a[i:i + BATCH], b[i:i + BATCH])) for i in range(0, len(a), BATCH)]
    if MAX_STEPS:
        batches = batches[:MAX_STEPS]
    total = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=True)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    log(f"cross-encoder training: {len(a):,} pairs ({y.mean():.3f} positive), {total} steps")
    t0 = time.time()
    for step, (yb, enc) in enumerate(pair_prefetch(tok, batches), 1):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(yb).to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            log(f"  step {step}/{total} loss {loss.item():.4f} ({(time.time() - t0) / step:.3f}s/step)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return model, tok


@torch.no_grad()
def score(model, tok, df: pl.DataFrame) -> np.ndarray:
    model.eval()
    lens = (df["ta"].str.len_chars() + df["tb"].str.len_chars()).to_numpy()
    order = np.argsort(lens, kind="stable")
    a, b = df["ta"].to_list(), df["tb"].to_list()
    batches = [(order[i:i + INF_BATCH], ([a[j] for j in order[i:i + INF_BATCH]], [b[j] for j in order[i:i + INF_BATCH]]))
               for i in range(0, len(order), INF_BATCH)]
    out = np.empty(len(order), dtype=np.float32)
    t0 = time.time()
    for bi, (idx, enc) in enumerate(pair_prefetch(tok, batches)):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if bi % 1000 == 0:
            log(f"  scored {min((bi + 1) * INF_BATCH, len(order)):,}/{len(order):,} "
                f"({(bi + 1) * INF_BATCH / max(1e-6, time.time() - t0):,.0f}/s)")
    return out


CROSSFIT = os.environ.get("BER_XENC_CROSSFIT", "1") == "1"
# stage-1 model used to pick which pairs to re-score: must be a model trained WITHOUT the xenc feature
SEL_MODEL = os.environ.get("BER_SEL_MODEL", str(WORK / "xgb_stage1.json"))
SEL_DECISION = os.environ.get("BER_SEL_DECISION", str(WORK / "decision.json"))


def load_or_train(h: int, B_h: pl.DataFrame):
    """Cross-encoder for half h of split B (trained on that half only). Reuses a saved model."""
    d = WORK / f"xenc{TAG}_h{h}"
    if not TAG and h == 0 and not (d / "config.json").exists() and (WORK / "xenc" / "config.json").exists():
        d = WORK / "xenc"  # model trained by the earlier single-half run
    if (d / "config.json").exists():
        log(f"half {h}: reusing cross-encoder {d}")
        tok = AutoTokenizer.from_pretrained(d)
        return AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.bfloat16).to(DEV), tok
    rng = np.random.default_rng(SEED + h)
    easy = (B_h["p1"] < MIN_P) & (B_h["label"] == 0)
    tr = attach_text(B_h.filter((~easy) | pl.Series(rng.random(B_h.height) < 0.05)), "train")
    return train_xenc(tr, d)


def scored(name: str, h: int, df: pl.DataFrame, split: str, model, tok) -> pl.DataFrame:
    """Cross-encoder logits of model h on the pairs of df with p1 >= MIN_P (cached per half)."""
    f = WORK / f"xenc{TAG}_{name}_h{h}.parquet"
    if f.exists():
        log(f"{name}: reusing cached half-{h} scores {f.name}")
        return pl.read_parquet(f)
    sel = df.filter(pl.col("p1") >= MIN_P)
    log(f"{name}: half-{h} model scoring {sel.height:,} of {df.height:,} pairs (p1 >= {MIN_P})"
        + (f"; positives kept {int(sel['label'].sum()):,}/{int(df['label'].sum()):,}" if "label" in df.columns else ""))
    sel = attach_text(sel.select("s1_id", "rec_id"), split)
    out = sel.select("s1_id", "rec_id").with_columns(pl.Series("xenc", score(model, tok, sel)))
    out.write_parquet(f)
    return out


def main():
    """Cross-fitting: model h is trained on half h of split B and scores the OTHER half of B, so the
    stacker can train on all of B without seeing in-sample scores. V and test get the mean logit of
    both models. With BER_XENC_CROSSFIT=0 only half 0 is used (the stacker then trains on half 1)."""
    t0 = time.time()
    torch.manual_seed(SEED)
    dec = json.loads(open(SEL_DECISION).read())
    assert "xenc" not in dec["cols"], "selection model must not use the xenc feature"
    m1 = xgb.Booster(model_file=SEL_MODEL)
    B = stage1_p("B", dec, m1)
    V = stage1_p("V", dec, m1)
    T = stage1_p("test", dec, m1)
    half = b_half(B["s1_id"])
    outB, outV, outT = [], [], []
    for h in ((0, 1) if CROSSFIT else (0,)):
        B_h = B.filter(pl.Series(half == h))
        model, tok = load_or_train(h, B_h)
        outB.append(scored("B", h, B.filter(pl.Series(half != h)), "train", model, tok))
        outV.append(scored("V", h, V, "train", model, tok))
        outT.append(scored("test", h, T, "test", model, tok))
        del model
        torch.cuda.empty_cache()
    col = f"xenc{TAG}"
    pl.concat(outB).rename({"xenc": col}).write_parquet(WORK / f"xenc{TAG}_B.parquet")

    def mean_of(parts):
        m = parts[0].rename({"xenc": "x0"})
        for i, q in enumerate(parts[1:], 1):
            m = m.join(q.rename({"xenc": f"x{i}"}), on=["s1_id", "rec_id"], how="full", coalesce=True)
        cols = [c for c in m.columns if c.startswith("x")]
        return m.select("s1_id", "rec_id", pl.mean_horizontal(cols).alias(col))

    mean_of(outV).write_parquet(WORK / f"xenc{TAG}_V.parquet")
    mean_of(outT).write_parquet(WORK / f"xenc{TAG}_test.parquet")
    (WORK / f"xenc{TAG}_meta.json").write_text(json.dumps(
        {"crossfit": CROSSFIT, "min_p": MIN_P, "base": BASE or "retriever", "max_len": MAX_LEN}))
    log(f"stage C2 done in {time.time() - t0:.0f}s (crossfit={CROSSFIT})")


if __name__ == "__main__":
    main()
