"""Re-score the French candidate pairs with saved French self-training checkpoints (cross-fitted halves).

For each cross-encoder family, model xenc{tag}_fr{h}{BER_RESCORE_SFX} scores the French pairs of the other half, as
in pseudo.py. Record texts come from work/prenorm when present (the texts the models were trained and scored on).
Output: WORK/xenc{tag}_test{BER_RESCORE_SFX}.parquet (US/India rows from the pre-France file, French rows re-scored).
Used to rebuild a France route whose test scores were not kept (round 5: BER_RESCORE_SFX=_r5).
"""
import os
import time

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import xenc as X
from config import WORK
from dader import TXT, attach, log
from pseudo import FAMILIES, fr_half

SFX = os.environ.get("BER_RESCORE_SFX", "_r5")


def main():
    t0 = time.time()
    log(f"record texts from {TXT}; checkpoints xenc*_fr{{0,1}}{SFX}")
    test = pl.read_parquet(TXT / "test_recs.parquet", columns=["id", "src", "country"])
    fr_s1 = test.filter((pl.col("src") == 1) & (pl.col("country") == "France"))["id"]
    half = dict(zip(fr_s1.to_list(), fr_half(fr_s1).tolist()))
    for tag, (_, max_len) in FAMILIES.items():
        X.MAX_LEN = max_len
        col = f"xenc{tag}"
        prefr = pl.read_parquet(WORK / f"xenc{tag}_test_prefr.parquet")
        fr_rows = prefr.filter(pl.col("s1_id").is_in(fr_s1.implode())).with_columns(
            pl.col("s1_id").replace_strict(half, return_dtype=pl.Int8).alias("h"))
        parts = []
        for h in (0, 1):
            d = WORK / f"xenc{tag}_fr{h}{SFX}"
            tok = AutoTokenizer.from_pretrained(d)
            model = AutoModelForSequenceClassification.from_pretrained(d).to("cuda")  # checkpoint dtype, as when trained
            target = fr_rows.filter(pl.col("h") == 1 - h).select("s1_id", "rec_id")
            parts.append(target.with_columns(pl.Series(col, X.score(model, tok, attach(target, "test")))))
            log(f"[{tag or 'e5'}] {d.name} scored {target.height:,} French pairs of half {1 - h}")
            del model
            torch.cuda.empty_cache()
        new_fr = pl.concat(parts)
        cmp = pl.read_parquet(WORK / f"xenc{tag}_test_sub9.parquet").join(new_fr.rename({col: "new"}), on=["s1_id", "rec_id"])
        log(f"[{tag or 'e5'}] French {col} vs round 1: mean {cmp[col].mean():.3f} -> {cmp['new'].mean():.3f}; "
            f"rank corr {cmp.select(pl.corr(col, 'new', method='spearman')).item():.4f}")
        rest = prefr.filter(~pl.col("s1_id").is_in(fr_s1.implode()))
        pl.concat([rest, new_fr.select(rest.columns)]).write_parquet(WORK / f"xenc{tag}_test{SFX}.parquet")
        log(f"[{tag or 'e5'}] wrote xenc{tag}_test{SFX}.parquet")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
