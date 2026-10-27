"""France: DADER-style domain-adversarial alignment added to the French self-training (unsupervised domain adaptation).

Same recipe as round 1 of pseudo.py (same base checkpoints, same pseudo-labels from test_scored_sub7, same split-B
replay pairs, 2000 steps of 256 pairs, lr 1e-5, cross-fitted over two halves of the French S1 entities), plus one term:
  - a small domain classifier on the pair representation (<s> token of the last layer) is trained to tell
    split-B candidate pairs (source: US / India) from French candidate pairs (target), both unlabelled and chosen
    by the same stage-1 selection (p1 >= 0.002);
  - a gradient-reversal layer between the encoder and that classifier (weight ramped from 0 to BER_DADER_LAMBDA)
    pushes the encoder towards pair representations the classifier cannot tell apart, so what the match head
    learned on labelled data transfers better to French pairs.
The domain loss uses no match labels (only which country a pair comes from), so any French pair can be used.
Model h then scores the French pairs of the other half, as in round 1. Checks printed: domain-classifier accuracy
during training, AUC on a split-V sample (round-1 model vs this one: US/India discrimination must not degrade)
and agreement with the round-1 French scores.
Output: WORK/xenc{tag}_test_dader.parquet (US/India rows unchanged, French rows re-scored); the working
xenc{tag}_test.parquet files are not touched. References: Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation
by Backpropagation"; Tu et al. (2022), "Domain Adaptation for Deep Entity Resolution" (DADER).
"""
import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import xenc as X
from config import SEED, WORK
from pseudo import FAMILIES, fr_half, pseudo_pairs, replay_pairs
from retrieve import to_dev

STEPS = int(os.environ.get("BER_DADER_STEPS", 2000))
LAMBDA = float(os.environ.get("BER_DADER_LAMBDA", 0.1))
LR = float(os.environ.get("BER_DADER_LR", 1e-5))
HEAD_LR = float(os.environ.get("BER_DADER_HEAD_LR", 1e-3))
DOM_HALF = int(os.environ.get("BER_DADER_DOM_HALF", 64))  # unlabelled source + target pairs per step, each
DOM_POOL = 400_000  # pairs sampled per domain pool (2000 steps x 64 = 128k are drawn)
N_CHECK = 100_000  # split-V pairs for the AUC check
DEV = "cuda"
# record texts and train pairs: the copies taken before a normalisation re-run (work/prenorm) when present, so the
# inputs are exactly those of the round-1 run and of every other cross-encoder score
TXT = WORK / "prenorm" if (WORK / "prenorm" / "test_recs.parquet").exists() else WORK
TEXT = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def attach(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """xenc.attach_text on cached record texts from TXT."""
    if split not in TEXT:
        TEXT[split] = pl.read_parquet(TXT / f"{split}_recs.parquet", columns=["id", "emb_text"]).with_columns(
            pl.col("emb_text").str.replace(r"^query: ", ""))
    t = TEXT[split]
    return df.join(t.rename({"id": "s1_id", "emb_text": "ta"}), on="s1_id").join(
        t.rename({"id": "rec_id", "emb_text": "tb"}), on="rec_id")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def sample(df: pl.DataFrame, n: int) -> pl.DataFrame:
    return df.sample(min(n, df.height), seed=SEED)


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def train_dader(base: str, tr: pl.DataFrame, dom_src: pl.DataFrame, dom_tgt: pl.DataFrame, out_dir, h: int):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(DEV)
    dom = nn.Sequential(nn.Linear(model.config.hidden_size, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEV)
    model.train()
    dom.train()
    # labelled stream: exactly as xenc.train_xenc (the round-1 run)
    tr = tr.sample(fraction=1.0, shuffle=True, seed=SEED)
    a, b, y = tr["ta"].to_list(), tr["tb"].to_list(), tr["label"].to_numpy().astype(np.float32)
    batches = [(y[i:i + X.BATCH], (a[i:i + X.BATCH], b[i:i + X.BATCH])) for i in range(0, len(a), X.BATCH)][:STEPS]
    total = len(batches)
    # unlabelled domain stream: DOM_HALF source + DOM_HALF target pairs per step
    rng = np.random.default_rng(SEED + 100 + h)
    si, ti = rng.integers(0, dom_src.height, (total, DOM_HALF)), rng.integers(0, dom_tgt.height, (total, DOM_HALF))
    sa, sb, ta, tb = dom_src["ta"].to_list(), dom_src["tb"].to_list(), dom_tgt["ta"].to_list(), dom_tgt["tb"].to_list()
    dbatches = [(None, ([sa[j] for j in si[k]] + [ta[j] for j in ti[k]], [sb[j] for j in si[k]] + [tb[j] for j in ti[k]]))
                for k in range(total)]
    dl = torch.cat([torch.zeros(DOM_HALF), torch.ones(DOM_HALF)]).to(DEV)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": LR}, {"params": dom.parameters(), "lr": HEAD_LR}],
                            weight_decay=0.01, fused=True)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    log(f"  training: {len(a):,} labelled pairs ({y.mean():.3f} positive), {total} steps of {X.BATCH} labelled + "
        f"{2 * DOM_HALF} unlabelled domain pairs; gradient reversal up to {LAMBDA}")
    t0, hist = time.time(), []
    tok_d = AutoTokenizer.from_pretrained(base)  # second instance: a fast tokenizer must not be shared by two threads
    for step, ((yb, enc), (_, enc_d)) in enumerate(zip(X.pair_prefetch(tok, batches), X.pair_prefetch(tok_d, dbatches)), 1):
        lam = LAMBDA * (2.0 / (1.0 + np.exp(-10.0 * step / total)) - 1.0)
        enc, enc_d = to_dev(enc), to_dev(enc_d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
            hd = model.base_model(**enc_d).last_hidden_state[:, 0]
            d = dom(GradReverse.apply(hd, lam)).squeeze(-1)
        loss_m = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(yb).to(DEV))
        loss_d = F.binary_cross_entropy_with_logits(d.float(), dl)
        opt.zero_grad(set_to_none=True)
        (loss_m + loss_d).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dom.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append((loss_m.item(), loss_d.item(), ((d.detach() > 0).float() == dl).float().mean().item()))
        if step % 200 == 0 or step == 1:
            m = np.mean(hist[-200:], axis=0)
            log(f"  step {step}/{total} match loss {m[0]:.4f} domain loss {m[1]:.4f} domain accuracy {m[2]:.3f} "
                f"(reversal weight {lam:.3f}; {(time.time() - t0) / step:.3f}s/step)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return model, tok


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    log(f"record texts and train pairs from {TXT}")
    test = pl.read_parquet(TXT / "test_recs.parquet", columns=["id", "src", "country"])
    fr_s1 = test.filter((pl.col("src") == 1) & (pl.col("country") == "France"))["id"]
    half = dict(zip(fr_s1.to_list(), fr_half(fr_s1).tolist()))
    pp = pseudo_pairs(fr_s1).with_columns(pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h"))
    rp = attach(replay_pairs(TXT / "train_pairs.parquet"), "train")
    vpos = pl.read_parquet(TXT / "train_pairs.parquet").filter(pl.col("split") == "V").select("s1_id", "rec_id")
    X.LR = LR
    for tag, (ckpt, max_len) in FAMILIES.items():
        X.MAX_LEN, X.TAG = max_len, tag
        col = f"xenc{tag}"
        prefr = pl.read_parquet(WORK / f"xenc{tag}_test_prefr.parquet")
        fr_rows = prefr.filter(pl.col("s1_id").is_in(fr_s1.implode())).with_columns(
            pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h"))
        dom_tgt = attach(sample(fr_rows.select("s1_id", "rec_id"), DOM_POOL), "test")
        dom_src = attach(sample(pl.read_parquet(WORK / f"xenc{tag}_B.parquet", columns=["s1_id", "rec_id"]), DOM_POOL), "train")
        check = sample(pl.read_parquet(WORK / f"xenc{tag}_V.parquet", columns=["s1_id", "rec_id"]), N_CHECK)
        check = attach(check.join(vpos.with_columns(pl.lit(1).alias("label")), on=["s1_id", "rec_id"], how="left")
                       .with_columns(pl.col("label").fill_null(0)), "train")
        yv = check["label"].to_numpy()
        parts = []
        for h in (0, 1):
            tr = pl.concat([attach(pp.filter(pl.col("h") == h).drop("h"), "test"), rp], how="vertical_relaxed")
            tr = tr.sample(fraction=1.0, shuffle=True, seed=SEED + h)
            log(f"[{tag or 'e5'}] half {h}: from {ckpt}; {dom_src.height:,} source / {dom_tgt.height:,} French unlabelled pairs")
            model, tok = train_dader(str(WORK / ckpt), tr, dom_src, dom_tgt, WORK / f"xenc{tag}_fr{h}_dader", h)
            model.eval()
            a_new = auc(yv, X.score(model, tok, check))
            target = fr_rows.filter(pl.col("h") == 1 - h).select("s1_id", "rec_id")
            parts.append(target.with_columns(pl.Series(col, X.score(model, tok, attach(target, "test")))))
            del model
            torch.cuda.empty_cache()
            ref = AutoModelForSequenceClassification.from_pretrained(WORK / f"xenc{tag}_fr{h}", dtype=torch.bfloat16).to(DEV)
            a_ref = auc(yv, X.score(ref, tok, check))
            del ref
            torch.cuda.empty_cache()
            log(f"[{tag or 'e5'}] half {h}: AUC on {check.height:,} split-V pairs ({int(yv.sum()):,} matches): "
                f"round-1 model {a_ref:.5f}, with domain alignment {a_new:.5f}")
        new_fr = pl.concat(parts)
        cmp = pl.read_parquet(WORK / f"xenc{tag}_test_sub9.parquet").join(new_fr.rename({col: "new"}), on=["s1_id", "rec_id"])
        log(f"[{tag or 'e5'}] French {col} vs round 1: mean {cmp[col].mean():.3f} -> {cmp['new'].mean():.3f}; "
            f"share > 0 {(cmp[col] > 0).mean():.4f} -> {(cmp['new'] > 0).mean():.4f}; sign changes {((cmp[col] > 0) != (cmp['new'] > 0)).sum():,}; "
            f"rank corr {cmp.select(pl.corr(col, 'new', method='spearman')).item():.4f}")
        rest = prefr.filter(~pl.col("s1_id").is_in(fr_s1.implode()))
        pl.concat([rest, new_fr.select(rest.columns)]).write_parquet(WORK / f"xenc{tag}_test_dader.parquet")
        log(f"[{tag or 'e5'}] wrote xenc{tag}_test_dader.parquet: {rest.height:,} unchanged + {new_fr.height:,} French rows")
    log(f"DADER done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
