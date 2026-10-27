"""France: adapt the Qwen3-Reranker LoRA of run 1 to the unseen country, with the final French decisions as labels.

Labels: pseudo.pseudo_pairs with BER_PSEUDO_SRC / BER_PSEUDO_SELECTED (the French decisions after the filler/swap
rule and decoy cap; positives: selected pairs with p >= 0.8 that are their record's clear best; negatives p <= 0.02).
For each half h of the French S1 entities (pseudo.fr_half), the run-1 LoRA trained on split-B half h (qwen_h{h}) is
trained further on labelled French pairs of half h and a replay sample of real split-B band pairs of B half h, then
scores the French band pairs of the other half:
  - French training pairs: every labelled pair that the round-1 France route found uncertain (run-1 band of
    qwen_selection), i.e. the pairs where self-training and the rule changed the answer, plus as many other
    labelled pairs;
  - French band = pairs whose France-route probability before the rule (BER_QWEN_FR_SCORED) lies in the run-1 band,
    the same "uncertain for the e5 + bge matcher" definition the Qwen feature has on split B.
The untouched run-1 model h scores the same pairs, as a control. Checks: AUC on a split-V band sample (real labels)
before/after, and agreement with the other half's pseudo-labels before/after.
Output: WORK/xenc_qwen_test_fr.parquet (adapted) and WORK/xenc_qwen_test_frus.parquet (run-1 models): the non-French
rows of xenc_qwen_test.parquet plus the French band rows. LoRA weights: WORK/qwen_fr_h{h}/lora.pt.
"""
import json
import os
import time
from collections import deque

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

import qwen as Q
from config import SEED, WORK
from pseudo import fr_half, pseudo_pairs
from retrieve import to_dev
from xenc import b_half

STEPS = int(os.environ.get("BER_QWEN_FR_STEPS", 1500))
LR = float(os.environ.get("BER_QWEN_FR_LR", 5e-5))
REPLAY = int(os.environ.get("BER_QWEN_FR_REPLAY", 40_000))
SCORED = os.environ.get("BER_QWEN_FR_SCORED", str(WORK / "test_scored_sub25.parquet"))
N_CHECK = int(os.environ.get("BER_QWEN_FR_CHECK", 20_000))
COL = "xenc_qwen"
TEXT = {}
log = Q.log


def attach(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """qwen.attach on cached raw texts ("name | address | country")."""
    if split not in TEXT:
        TEXT[split] = Q.texts(split)
    t = TEXT[split]
    return df.join(t.rename({"id": "s1_id", "t": "ta"}), on="s1_id").join(t.rename({"id": "rec_id", "t": "tb"}), on="rec_id")


def sample(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    return df.sample(min(n, df.height), seed=seed)


def adapt(h: int, tr: pl.DataFrame, target: pl.DataFrame, check: pl.DataFrame):
    """Returns (run-1 scores, adapted scores) on target, and the adapted model's scores on check."""
    model, tok, ids = Q.load_model(WORK / f"qwen_h{h}" / "lora.pt")
    x_us = Q.score(model, tok, ids, target)
    log(f"half {h}: run-1 model scored {target.height:,} French band pairs of half {1 - h}")
    params = [p for p in model.parameters() if p.requires_grad]
    P, y = Q.prompts(tr), tr["label"].to_numpy().astype(np.float32)
    rng = np.random.default_rng(SEED + h)
    need = STEPS * Q.BATCH
    idx = np.concatenate([rng.permutation(len(P)) for _ in range(need // len(P) + 1)])[:need]
    batches = [(y[idx[i:i + Q.BATCH]], [P[j] for j in idx[i:i + Q.BATCH]]) for i in range(0, need, Q.BATCH)]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    warm = max(1, int(0.05 * STEPS))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (STEPS - s) / max(1, STEPS - warm)))
    log(f"half {h}: continued LoRA training on {len(P):,} pairs ({y.mean():.3f} positive), {STEPS} steps x {Q.BATCH}, lr {LR}")
    model.train()
    t0, run = time.time(), deque(maxlen=100)
    for step, (yb, enc) in enumerate(Q.prefetch(tok, batches), 1):
        enc = to_dev(enc)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = Q.relevance(model, ids, enc)
        loss = F.binary_cross_entropy_with_logits(z, torch.from_numpy(yb).to(Q.DEV))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 100 == 0 or step == 1:
            log(f"  step {step}/{STEPS} loss {np.mean(run):.4f} ({(time.time() - t0) / step:.3f}s/step, "
                f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.0f} GB)")
    out = WORK / f"qwen_fr_h{h}" / "lora.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}, out)
    Q.merge_lora(model)
    x_fr = Q.score(model, tok, ids, target)
    x_chk = Q.score(model, tok, ids, check)
    del model, opt
    torch.cuda.empty_cache()
    return x_us, x_fr, x_chk


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    band_cfg = json.loads((WORK / "xenc_qwen_band.json").read_text())
    lo, hi = band_cfg["lo"], band_cfg["hi"]
    test = pl.read_parquet(WORK / "test_recs.parquet", columns=["id", "src", "country"])
    # entities of the countries absent from training (test: France); country is an open set, nothing is hard-coded
    trained = pl.read_parquet(WORK / "train_recs.parquet", columns=["country"])["country"].unique().to_list()
    fr_s1 = test.filter((pl.col("src") == 1) & ~pl.col("country").is_in(trained))["id"]
    half = dict(zip(fr_s1.to_list(), fr_half(fr_s1).tolist()))
    hcol = pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h")
    # ---- data (all of it prepared before any model is loaded, so a data problem fails in the first minutes)
    pp = pseudo_pairs(fr_s1).with_columns(hcol)
    sel = pl.read_parquet(WORK / "qwen_selection.parquet")
    r1band = (sel.filter((pl.col("split") == "test") & pl.col("s1_id").is_in(fr_s1.implode()) & pl.col("p").is_between(lo, hi))
              .select("s1_id", "rec_id").with_columns(pl.lit(True).alias("r1band")))
    pp = pp.join(r1band, on=["s1_id", "rec_id"], how="left").with_columns(pl.col("r1band").fill_null(False))
    cur = pl.read_parquet(SCORED).filter(pl.col("s1_id").is_in(fr_s1.implode()))
    fband = cur.filter(pl.col("p").is_between(lo, hi)).select("s1_id", "rec_id").with_columns(hcol)
    log(f"band [{lo}, {hi}]; French pseudo-labelled pairs {pp.height:,} ({int(pp['label'].sum()):,} positive), of which in the "
        f"round-1 band {int(pp['r1band'].sum()):,}; French band now (France route before the rule) {fband.height:,} of {cur.height:,} pairs")
    vus = pl.read_parquet(WORK / "xenc_qwen_V.parquet")
    vband = sel.filter((pl.col("split") == "V") & pl.col("p").is_between(lo, hi)).select("s1_id", "rec_id", "label")
    vband = vband.with_columns(pl.Series("bh", b_half(vband["s1_id"])).cast(pl.Int8)).join(vus, on=["s1_id", "rec_id"])
    data = {}
    for h in (0, 1):
        mine = pp.filter(pl.col("h") == h)
        inb = mine.filter(pl.col("r1band"))
        fr_tr = pl.concat([inb, sample(mine.filter(~pl.col("r1band")), inb.height, SEED + h)]).select("s1_id", "rec_id", "label")
        rep = sample(sel.filter((pl.col("split") == "B") & (pl.col("h") == h) & pl.col("p").is_between(lo, hi)), REPLAY, SEED + h)
        rep = rep.select("s1_id", "rec_id", pl.col("label").cast(pl.Float64))
        tr = pl.concat([attach(fr_tr, "test"), attach(rep, "train")], how="vertical_relaxed").sample(fraction=1.0, shuffle=True, seed=SEED + h)
        target = attach(fband.filter(pl.col("h") == 1 - h).select("s1_id", "rec_id"), "test")
        check = attach(sample(vband.filter(pl.col("bh") == 1 - h), N_CHECK, SEED + h), "train")
        data[h] = (tr, target, check)
        log(f"half {h}: training {tr.height:,} pairs ({fr_tr.height:,} French: {inb.height:,} round-1 band + {fr_tr.height - inb.height:,} other; "
            f"{rep.height:,} split-B replay); scoring {target.height:,} French band pairs of half {1 - h}; check {check.height:,} split-V pairs")
    # ---- models
    parts_fr, parts_us = [], []
    for h in (0, 1):
        f = WORK / f"xenc_qwen_frtest_h{h}.parquet"
        tr, target, check = data[h]
        if f.exists():
            log(f"half {h}: reusing {f.name}")
            got = pl.read_parquet(f)
        else:
            x_us, x_fr, x_chk = adapt(h, tr, target, check)
            yv = check["label"].to_numpy()
            log(f"half {h}: split-V AUC on {check.height:,} band pairs: run-1 {Q.auc(yv, check[COL].to_numpy()):.4f}, adapted {Q.auc(yv, x_chk):.4f}")
            got = target.select("s1_id", "rec_id").with_columns(pl.Series("us", x_us), pl.Series("fr", x_fr))
            got.write_parquet(f)
        lab = got.join(pp.select("s1_id", "rec_id", "label"), on=["s1_id", "rec_id"])
        if lab.height and 0 < lab["label"].sum() < lab.height:
            yl = lab["label"].to_numpy()
            log(f"half {h}: agreement with the other half's pseudo-labels ({lab.height:,} labelled band pairs, "
                f"{yl.mean():.3f} positive): AUC run-1 {Q.auc(yl, lab['us'].to_numpy()):.4f}, adapted {Q.auc(yl, lab['fr'].to_numpy()):.4f}")
        log(f"half {h}: French band scores: run-1 mean {got['us'].mean():.3f} (share > 0 {(got['us'] > 0).mean():.3f}), "
            f"adapted mean {got['fr'].mean():.3f} (share > 0 {(got['fr'] > 0).mean():.3f})")
        parts_fr.append(got.select("s1_id", "rec_id", pl.col("fr").alias(COL)))
        parts_us.append(got.select("s1_id", "rec_id", pl.col("us").alias(COL)))
    orig = WORK / "xenc_qwen_test_orig.parquet"  # qfr.sh keeps the original here while it swaps files
    base = pl.read_parquet(orig if orig.exists() else WORK / "xenc_qwen_test.parquet")
    rest = base.filter(~pl.col("s1_id").is_in(fr_s1.implode()))
    for name, parts in (("fr", parts_fr), ("frus", parts_us)):
        new = pl.concat(parts).with_columns(pl.col(COL).cast(base.schema[COL]))
        pl.concat([rest, new.select(rest.columns)]).write_parquet(WORK / f"xenc_qwen_test_{name}.parquet")
        log(f"wrote xenc_qwen_test_{name}.parquet: {rest.height:,} US/India rows + {new.height:,} French band rows")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
