"""Stage C3 (optional): Qwen3-Reranker-4B as a third cross-encoder feature (xenc_qwen) on uncertain pairs.

Model: Qwen/Qwen3-Reranker-4B (Apache-2.0, 4.0B parameters), run locally. It reads its official reranking
prompt (instruction + Query + Document) and its relevance logit is logit("yes") - logit("no") at the last
position. Records are given as raw "name | address | country" (the other two cross-encoders read the
normalised text, so this one adds a different view).
Fine-tuning: LoRA adapters (rank 16) on every attention/MLP projection, base weights frozen, trained with
binary cross-entropy on split-B pairs only.
Which pairs: only the uncertain ones, where the #7 matcher's probability lies in a band [lo, hi]. For split B
that probability is out-of-fold (2-fold, grouped by S1), for V and test it is the saved #7 model, so the same
rule selects pairs everywhere. The widest band whose pair count fits BER_QWEN_CAP is used.
Cross-fitting: model h trains on the band pairs of B half h and scores every band pair whose S1 is in half
1-h (B, V and test alike), so each pair gets exactly one out-of-sample score from the same kind of model.
Output: WORK/xenc_qwen_{B,V,test}.parquet (s1_id, rec_id, xenc_qwen), xenc_qwen_meta.json, qwen_h{0,1}/lora.pt.
Then: BER_XENC_EXTRA=_bge,_qwen run.sh train predict. BER_QWEN_SMOKE=1: tiny mechanics/throughput test.
"""
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SEED, WORK
from retrieve import to_dev
from train import context, load_feats, stage_fit
from xenc import b_half

BASE = os.environ.get("BER_QWEN_BASE", "Qwen/Qwen3-Reranker-4B")
SMOKE = os.environ.get("BER_QWEN_SMOKE", "0") == "1"
STEPS = 30 if SMOKE else int(os.environ.get("BER_QWEN_STEPS", 3000))
BATCH = int(os.environ.get("BER_QWEN_BATCH", 64))
INF_BATCH = int(os.environ.get("BER_QWEN_INF_BATCH", 512))
LR = float(os.environ.get("BER_QWEN_LR", 1e-4))
RANK = int(os.environ.get("BER_QWEN_RANK", 16))
CAP = int(os.environ.get("BER_QWEN_CAP", 3_500_000))  # max pairs scored over B + V + test
SEL = os.environ.get("BER_QWEN_SEL", "_sub7")  # suffix of the decision/xgb files used to pick the band
BANDS = [(0.0005, 0.9995), (0.001, 0.999), (0.002, 0.998), (0.005, 0.995), (0.01, 0.99), (0.02, 0.98), (0.05, 0.95)]
NAME = "qwen_smoke" if SMOKE else os.environ.get("BER_QWEN_NAME", "qwen")  # e.g. "qwen2" for a second run
TAG = f"_{NAME}"
DEV = "cuda"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = ("Given a business listing (name | address | country), judge whether the Document is a listing of "
            "the same business entity at the same location as the Query")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale


def _swap(model, fn):
    for mod in list(model.modules()):
        for t in TARGETS:
            child = getattr(mod, t, None)
            new = fn(child) if child is not None else None
            if new is not None:
                setattr(mod, t, new)


def add_lora(model):
    _swap(model, lambda c: LoRALinear(c, RANK, 2 * RANK) if isinstance(c, nn.Linear) else None)


@torch.no_grad()
def merge_lora(model):
    def merge(c):
        if not isinstance(c, LoRALinear):
            return None
        c.base.weight += (c.B @ c.A * c.scale).to(c.base.weight.dtype)
        return c.base
    _swap(model, merge)


def load_model(lora_path=None):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(DEV)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    add_lora(model)
    if lora_path is not None:
        missing, unexpected = model.load_state_dict(torch.load(lora_path, map_location=DEV), strict=False)
        assert not unexpected, unexpected[:5]
    ids = [tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")]
    return model, tok, ids


def relevance(model, ids, enc) -> torch.Tensor:
    h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).last_hidden_state[:, -1, :]
    z = h @ model.get_output_embeddings().weight[ids].T.to(h.dtype)
    return (z[:, 1] - z[:, 0]).float()


# ---------------------------------------------------------------- text
def texts(split: str) -> pl.DataFrame:
    r = pl.read_parquet(WORK / f"{split}_recs.parquet", columns=["id", "name", "address", "country"])
    return r.select("id", (pl.col("name").fill_null("").str.slice(0, 150) + " | "
                           + pl.col("address").fill_null("").str.slice(0, 200) + " | "
                           + pl.col("country").fill_null("")).alias("t"))


def attach(df: pl.DataFrame, split: str) -> pl.DataFrame:
    t = texts(split)
    return df.join(t.rename({"id": "s1_id", "t": "ta"}), on="s1_id").join(t.rename({"id": "rec_id", "t": "tb"}), on="rec_id")


def prompts(df: pl.DataFrame) -> list:
    return [f"{PREFIX}<Instruct>: {INSTRUCT}\n<Query>: {a}\n<Document>: {b}{SUFFIX}"
            for a, b in zip(df["ta"].to_list(), df["tb"].to_list())]


def prefetch(tok, batches, depth=6):
    """batches: list of (payload, texts); tokenised ahead in one worker thread."""
    def enc(t):
        return tok(t, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with ThreadPoolExecutor(max_workers=1) as ex:
        it = iter(batches)
        q = deque()
        for b in it:
            q.append((b[0], ex.submit(enc, b[1])))
            if len(q) >= depth:
                break
        while q:
            payload, fut = q.popleft()
            nxt = next(it, None)
            if nxt is not None:
                q.append((nxt[0], ex.submit(enc, nxt[1])))
            yield payload, fut.result()


# ---------------------------------------------------------------- train / score
def train(tr: pl.DataFrame, out_path):
    model, tok, ids = load_model()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"LoRA rank {RANK}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters")
    P, y = prompts(tr), tr["label"].to_numpy().astype(np.float32)
    rng = np.random.default_rng(SEED)
    need = STEPS * BATCH
    idx = np.concatenate([rng.permutation(len(P)) for _ in range(need // len(P) + 1)])[:need]
    batches = [(y[idx[i:i + BATCH]], [P[j] for j in idx[i:i + BATCH]]) for i in range(0, need, BATCH)]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    warm = max(1, int(0.05 * STEPS))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (STEPS - s) / max(1, STEPS - warm)))
    log(f"Qwen LoRA training: {len(P):,} pairs ({y.mean():.3f} positive), {STEPS} steps x {BATCH} "
        f"({need / len(P):.2f} epochs)")
    model.train()
    t0, run = time.time(), deque(maxlen=100)
    for step, (yb, enc) in enumerate(prefetch(tok, batches), 1):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = relevance(model, ids, enc)
        loss = F.binary_cross_entropy_with_logits(z, torch.from_numpy(yb).to(DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 100 == 0 or step == 1:
            log(f"  step {step}/{STEPS} loss {np.mean(run):.4f} ({(time.time() - t0) / step:.3f}s/step, "
                f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.0f} GB)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}, out_path)
    return model, tok, ids


@torch.no_grad()
def score(model, tok, ids, df: pl.DataFrame) -> np.ndarray:
    model.eval()
    P = prompts(df)
    order = np.argsort(np.fromiter((len(p) for p in P), dtype=np.int64, count=len(P)), kind="stable")
    batches = [(order[i:i + INF_BATCH], [P[j] for j in order[i:i + INF_BATCH]]) for i in range(0, len(order), INF_BATCH)]
    out = np.empty(len(order), dtype=np.float32)
    t0 = time.time()
    for bi, (idx, enc) in enumerate(prefetch(tok, batches)):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = relevance(model, ids, enc).cpu().numpy()
        if bi % 200 == 0:
            log(f"  scored {min((bi + 1) * INF_BATCH, len(order)):,}/{len(order):,} "
                f"({min((bi + 1) * INF_BATCH, len(order)) / max(1e-6, time.time() - t0):,.0f}/s)")
    return out


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


# ---------------------------------------------------------------- selection
def selection() -> pl.DataFrame:
    """#7 matcher probability for every candidate pair: out-of-fold on B, saved model on V and test."""
    f = WORK / "qwen_selection.parquet"
    if f.exists():
        return pl.read_parquet(f)
    dec = json.loads((WORK / f"decision{SEL}.json").read_text())
    tags = dec["xenc_tags"]
    m1 = xgb.Booster(model_file=str(WORK / f"xgb_stage1{SEL}.json"))
    m2 = xgb.Booster(model_file=str(WORK / f"xgb_stage2{SEL}.json"))
    B = load_feats("B", True, tags)
    oof1, _ = stage_fit(B, dec["cols"], "select/stage1")
    oof2, _ = stage_fit(pl.concat([B, context(B, oof1)], how="horizontal"), dec["cols2"], "select/stage2")
    parts = [B.select("s1_id", "rec_id", pl.col("label").cast(pl.Int8)).with_columns(
        pl.Series("p", oof2), pl.lit("B").alias("split"))]
    for name in ("V", "test"):
        X = load_feats(name, True, tags)
        p1 = m1.inplace_predict(X.select(dec["cols"]).to_numpy().astype(np.float32))
        p2 = m2.inplace_predict(pl.concat([X, context(X, p1)], how="horizontal").select(dec["cols2"]).to_numpy().astype(np.float32))
        lab = pl.col("label").cast(pl.Int8) if "label" in X.columns else pl.lit(None, pl.Int8).alias("label")
        parts.append(X.select("s1_id", "rec_id", lab).with_columns(pl.Series("p", p2), pl.lit(name).alias("split")))
    sel = pl.concat(parts).with_columns(pl.Series("h", b_half(pl.concat([q["s1_id"] for q in parts]))).cast(pl.Int8))
    sel.write_parquet(f)
    return sel


def choose_band(sel: pl.DataFrame):
    err = ((pl.col("p") >= 0.5).cast(pl.Int8) != pl.col("label"))
    best = None
    for lo, hi in BANDS:
        inb = sel.filter(pl.col("p").is_between(lo, hi))
        n = inb.height
        v = sel.filter(pl.col("split") == "V")
        cov = v.filter(err & pl.col("p").is_between(lo, hi)).height / max(1, v.filter(err).height)
        log(f"band [{lo}, {hi}]: {n:,} pairs (" + ", ".join(
            f"{s} {inb.filter(pl.col('split') == s).height:,}" for s in ("B", "V", "test"))
            + f"); V errors at 0.5 inside band: {cov:.1%}")
        if best is None and n <= CAP:
            best = (lo, hi)
    return best or BANDS[-1]


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    if SMOKE:  # mechanics + throughput on a small random sample; no selection model needed
        B = pl.read_parquet(WORK / "feat_B.parquet", columns=["s1_id", "rec_id", "label"]).sample(12_000, seed=SEED)
        band = B.with_columns(pl.lit("B").alias("split"), pl.Series("h", b_half(B["s1_id"])).cast(pl.Int8))
    else:
        sel = selection()
        lo, hi = choose_band(sel)
        band = sel.filter(pl.col("p").is_between(lo, hi))
        log(f"using band [{lo}, {hi}]: {band.height:,} pairs")
        (WORK / f"xenc{TAG}_band.json").write_text(json.dumps({"lo": lo, "hi": hi, "n": band.height}))
    outs = {"B": [], "V": [], "test": []}
    for h in (0, 1):
        lp = WORK / f"{NAME}_h{h}" / "lora.pt"
        if lp.exists():
            log(f"half {h}: reusing {lp}")
            model, tok, ids = load_model(lp)
        else:
            tr = band.filter((pl.col("split") == "B") & (pl.col("h") == h))
            if not SMOKE:  # a few easy pairs from outside the band keep the scale anchored
                easy = sel.filter((pl.col("split") == "B") & (pl.col("h") == h) & ~pl.col("p").is_between(lo, hi))
                tr = pl.concat([tr, easy.sample(min(easy.height, tr.height // 10), seed=SEED + h)])
            if SMOKE and h == 0:  # zero-shot quality of the untouched reranker, for reference
                model, tok, ids = load_model()
                zs = attach(band.filter(pl.col("h") == 1).head(4000), "train")
                log(f"zero-shot AUC on {zs.height:,} B pairs: {auc(zs['label'].to_numpy(), score(model, tok, ids, zs)):.4f}")
                del model
                torch.cuda.empty_cache()
            model, tok, ids = train(attach(tr, "train"), lp)
        merge_lora(model)
        for split in ("B", "V", "test"):
            f = WORK / f"xenc{TAG}_{split}_h{h}.parquet"
            if f.exists():
                outs[split].append(pl.read_parquet(f))
                continue
            tgt = band.filter((pl.col("split") == split) & (pl.col("h") == 1 - h))
            if tgt.height == 0:
                continue
            tgt = attach(tgt.select("s1_id", "rec_id", "label"), "test" if split == "test" else "train")
            log(f"{split}: half-{h} model scoring {tgt.height:,} band pairs of the other half")
            x = score(model, tok, ids, tgt)
            if split != "test":
                log(f"{split}: AUC of half-{h} Qwen scores on these pairs {auc(tgt['label'].to_numpy(), x):.4f}")
            out = tgt.select("s1_id", "rec_id").with_columns(pl.Series(f"xenc{TAG}", x))
            out.write_parquet(f)
            outs[split].append(out)
        del model
        torch.cuda.empty_cache()
    for split, parts in outs.items():
        if parts:
            pl.concat(parts).write_parquet(WORK / f"xenc{TAG}_{split}.parquet")
    (WORK / f"xenc{TAG}_meta.json").write_text(json.dumps(
        {"crossfit": True, "base": BASE, "rank": RANK, "steps": STEPS, "batch": BATCH, "lr": LR}))
    log(f"stage C3 ({NAME}) done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
