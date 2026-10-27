"""Build the demo app's data bundle (app/data/) from the pipeline's work/ and output/ artefacts.

  demo.sqlite      a stratified sample of test S1 entities (per country), their final-stage candidate
                   records and the submitted matches with scores (the "Explore" tab)
  index.parquet    normalised fields of every record the live matcher can retrieve
  emb.npy          their fine-tuned e5 retriever embeddings (fp16, L2-normalised)
  rev_top.npy      each record's top-10 cosine to the sampled S1 pool (reverse-rank lookup)
  stats.json       corpus statistics (name-token document frequency, same-name counts) from the full test set
  model.json       the live matcher: XGBoost on the 81 hand-crafted features, trained on split B
  meta.json        threshold, feature list, validation scores

Usage: python app/build_index.py [--per-country 20000]
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import WORK  # noqa: E402

BER = WORK.parent
DATA = ROOT / "app" / "data"
DATA.mkdir(parents=True, exist_ok=True)
FULL = json.loads((WORK / "decision_final.json").read_text())["cols"]
FEATS = [c for c in FULL if not c.startswith("xenc") and not c.startswith("rec_n")]
REC_COLS = ["id", "src", "country", "state", "name_native", "web_stem", "legal", "name_full", "name_core",
            "name_sorted", "name_nospace", "name_phon", "name_tok", "name_nums", "addr_str", "addr_nums",
            "addr_num1", "addr_loc_tok", "addr_empty", "emb_text", "name", "address"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sample_s1(per_country: int) -> pl.DataFrame:
    s1 = (pl.scan_parquet(WORK / "test_recs.parquet").filter(pl.col("src") == 1)
          .select("id", "country").collect())
    h = (s1["id"].str.slice(3).cast(pl.Int64) * 2654435761 // 4096) % 1_000_003
    s1 = s1.with_columns(pl.Series("h", h)).sort("h")
    return pl.concat([g.head(per_country) for _, g in s1.group_by("country", maintain_order=True)]).drop("h")


def read_lists(path: Path, col: str, ids: pl.Series) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (df.rename({"source1_entity_id": "s1_id", col: "rec_id"})
            .filter(pl.col("s1_id").is_in(ids.implode()))
            .with_columns(pl.col("rec_id").fill_null("").str.split(","))
            .explode("rec_id").filter(pl.col("rec_id") != ""))


def corpus_stats() -> dict:
    """Name-token document frequencies and same-name counts over the full test corpus (as record_stats)."""
    lf = pl.scan_parquet(WORK / "test_recs.parquet")
    df = (lf.select(pl.col("name_tok").list.unique()).explode("name_tok").drop_nulls()
          .group_by("name_tok").len().collect(engine="streaming"))
    key = ["country", "name_sorted"]
    s1 = lf.filter(pl.col("src") == 1).group_by(key).len().collect(engine="streaming")
    rc = lf.filter(pl.col("src") != 1).group_by(key).len().collect(engine="streaming")
    s1s = lf.filter(pl.col("src") == 1).group_by(key + ["state"]).len().collect(engine="streaming")
    k = lambda r: "|".join("" if v is None else str(v) for v in r[:-1])
    return {
        "tok_df": dict(zip(df["name_tok"].to_list(), df["len"].to_list())),
        "n_s1": {k(r): r[-1] for r in s1.iter_rows()},
        "n_rec": {k(r): r[-1] for r in rc.iter_rows()},
        "n_s1_state": {k(r): r[-1] for r in s1s.iter_rows()},
    }


def train_model():
    import xgboost as xgb
    pairs = pl.read_parquet(WORK / "train_pairs.parquet").filter(pl.col("split") == "B")
    B = (pl.read_parquet(WORK / "feat_B.parquet").drop("label", strict=False)
         .join(pairs.select("s1_id", "rec_id", pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "rec_id"], how="left"))
    X = B.select([pl.col(c).cast(pl.Float32).fill_null(-1).fill_nan(-1) for c in FEATS]).to_numpy()
    y = B["label"].fill_null(0).to_numpy()
    params = dict(objective="binary:logistic", tree_method="hist", device="cuda", max_depth=8, eta=0.08,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=42)
    bst = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=FEATS), 800)
    bst.save_model(DATA / "model.json")
    log("live matcher trained on", len(y), "pairs")


@torch.no_grad()
def embed_all(texts):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WORK / "retriever")
    model = AutoModel.from_pretrained(WORK / "retriever", dtype=torch.float16).cuda().eval()
    out = []
    for i in range(0, len(texts), 1024):
        enc = tok(texts[i:i + 1024], padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).half().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=20000)
    args = ap.parse_args()

    s1 = sample_s1(args.per_country)
    log("sampled S1:", s1.group_by("country").len().sort("country").rows())
    cands = read_lists(BER / "output_ens" / "candidate_pairs.tsv", "candidate_entity_ids", s1["id"])
    matches = read_lists(BER / "output_ens" / "matching_results.tsv", "matched_entity_ids", s1["id"])
    ctry = s1.rename({"id": "s1_id"})
    scored = pl.concat([
        pl.read_parquet(WORK / "test_scored_final.parquet").join(ctry.filter(pl.col("country") != "France"), on="s1_id"),
        pl.read_parquet(WORK / "test_scored_ens.parquet").join(ctry.filter(pl.col("country") == "France"), on="s1_id"),
    ]).select("s1_id", "rec_id", "p")
    pairs = (cands.join(scored, on=["s1_id", "rec_id"], how="left")
             .join(matches.with_columns(pl.lit(1).alias("matched")), on=["s1_id", "rec_id"], how="left")
             .with_columns(pl.col("matched").fill_null(0)))
    rec_ids = pairs["rec_id"].unique()
    want = pl.concat([s1["id"], rec_ids])
    recs = (pl.scan_parquet(WORK / "test_recs.parquet").filter(pl.col("id").is_in(want.implode()))
            .select(REC_COLS).collect())
    log(f"records: {recs.height:,} ({rec_ids.len():,} S2/S3), candidate pairs {pairs.height:,}, matches {matches.height:,}")

    db = DATA / "demo.sqlite"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE records (id TEXT PRIMARY KEY, src INT, country TEXT, name TEXT, address TEXT)")
    con.executemany("INSERT INTO records VALUES (?,?,?,?,?)", recs.select("id", "src", "country", "name", "address").iter_rows())
    con.execute("CREATE TABLE pairs (s1_id TEXT, rec_id TEXT, p REAL, matched INT)")
    con.executemany("INSERT INTO pairs VALUES (?,?,?,?)", pairs.select("s1_id", "rec_id", "p", "matched").iter_rows())
    con.execute("CREATE INDEX pairs_s1 ON pairs(s1_id)")
    con.execute("CREATE VIRTUAL TABLE s1_fts USING fts5(id UNINDEXED, name, address, country UNINDEXED)")
    con.executemany("INSERT INTO s1_fts VALUES (?,?,?,?)",
                    recs.filter(pl.col("src") == 1).select("id", "name", "address", "country").iter_rows())
    con.commit()
    con.close()
    log("wrote", db)

    recs.write_parquet(DATA / "index.parquet")
    emb = embed_all(recs["emb_text"].to_list())
    np.save(DATA / "emb.npy", emb.numpy())
    log("embedded index", emb.shape)
    # reverse lookup: each S2/S3 record's top-10 S1 neighbours (rev_top.npy, rev_idx.npy)
    import runpy
    runpy.run_path(str(ROOT / "app" / "rev_index.py"))

    (DATA / "stats.json").write_text(json.dumps(corpus_stats(), ensure_ascii=False))
    log("corpus stats written")
    train_model()
    comp = ROOT / "results" / "model_comparison.csv"
    thr = 0.5
    if comp.exists():
        r = pl.read_csv(comp).filter(pl.col("model") == "XGBoost")
        thr = float(r["threshold"][0]) if r.height else thr
    (DATA / "meta.json").write_text(json.dumps({"feats": FEATS, "threshold": thr,
                                                "countries": sorted(s1["country"].unique().to_list())}, indent=1))
    log("done")


if __name__ == "__main__":
    main()
