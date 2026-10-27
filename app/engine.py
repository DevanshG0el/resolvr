"""Live matcher behind the demo API: one query record -> ranked candidate matches with reasons.

Mirrors the batch pipeline on a single record, using the bundle written by build_index.py:
  normalise (src/normalize.py) -> embed with the fine-tuned e5 retriever -> forward kNN over the index of
  the same country + reverse rank against the S1 pool -> the 81 pairwise features (src/features.py) ->
  XGBoost probability -> expected-F0.5 selection (src/decide.py). Per-feature SHAP contributions explain
  each score.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import decide as D  # noqa: E402
import features as Fe  # noqa: E402
import normalize as N  # noqa: E402
from config import WORK  # noqa: E402

DATA = ROOT / "app" / "data"
K_FWD, KF, KR = 40, Fe.KF, Fe.KR

LABELS = {  # plain-English names for the explanation panel
    "nm_ratio": "Name similarity", "nm_tset": "Name token overlap", "nm_tsort": "Name (sorted tokens)",
    "nm_partial": "Name partial match", "nm_jw": "Name Jaro-Winkler", "nm_full_tset": "Full-name overlap",
    "nm_full_ratio": "Full-name similarity", "nm_nospace": "Name without spaces", "nm_phon": "Phonetic name",
    "nm_nospace_partial": "Name partial (no spaces)", "ad_ratio": "Address similarity", "ad_tset": "Address token overlap",
    "ad_tsort": "Address (sorted tokens)", "ad_partial": "Address partial match", "web_ratio": "Website stem",
    "cos": "Semantic similarity (e5)", "rank_fwd": "Retrieval rank", "rank_rev": "Reverse retrieval rank",
    "gap_s1_top": "Gap to best candidate", "gap_rec_top": "Gap to record's best S1", "num1_eq": "House number equal",
    "num_jacc": "Address numbers overlap", "num_conflict": "Address numbers conflict", "state_eq": "Same state/region",
    "legal_eq": "Same legal form", "legal_conflict": "Legal form conflict", "tok_jacc": "Name token Jaccard",
    "a_rare_in_b": "Rare name token shared", "b_rare_in_a": "Rare name token shared (reverse)",
    "loc_jacc": "Locality overlap", "nm_exact": "Exact core name", "nm_sorted_exact": "Exact name (any order)",
    "nums_min_absdiff": "House number distance", "num1_absdiff": "First number distance",
    "cos_gap_s1": "Semantic gap to best", "nm_tset_gap_s1": "Name gap to best", "ad_tset_gap_s1": "Address gap to best",
    "n_near_top_s1": "Near-tied candidates", "a_has_legal": "Query has legal form", "b_has_legal": "Record has legal form",
    "rec_top_margin": "Record's S1 margin", "s1_top_margin": "Query's top-2 margin", "n_cand_s1": "Number of candidates",
    "rec_addr_empty": "Record has no address", "b_nnums": "Numbers in record address", "rec_native": "Record in native script", "src": "Record source",
    "a_rare_logdf": "Query rarest-token frequency", "b_rare_logdf": "Record rarest-token frequency",
    "cos_rank_s1": "Semantic rank among candidates", "nm_tset_rank_s1": "Name rank among candidates", "a_n_s1_same_name": "Other businesses with this name",
}


def label(f: str) -> str:
    return LABELS.get(f, f.replace("_", " "))


class Engine:
    def __init__(self):
        t0 = time.time()
        self.meta = json.loads((DATA / "meta.json").read_text())
        self.feats = self.meta["feats"]
        self.bst = xgb.Booster(model_file=str(DATA / "model.json"))
        self.dicts = json.loads((WORK / "dicts.json").read_text(encoding="utf-8"))
        self.index = pl.read_parquet(DATA / "index.parquet")
        self.emb = torch.from_numpy(np.load(DATA / "emb.npy").astype(np.float32))
        self.rev = np.load(DATA / "rev_top.npy")
        self.rev_idx = np.load(DATA / "rev_idx.npy")
        st = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
        self.tok_df, self.n_s1, self.n_rec, self.n_s1_state = st["tok_df"], st["n_s1"], st["n_rec"], st["n_s1_state"]
        self.country = self.index["country"].to_numpy()
        self.is_rec = self.index["src"].to_numpy() != 1
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(WORK / "retriever")
        self.enc = AutoModel.from_pretrained(WORK / "retriever").eval()
        self.load_s = time.time() - t0

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        e = self.tok([text], truncation=True, max_length=96, return_tensors="pt")
        h = self.enc(**e).last_hidden_state
        v = h.mean(1)[0]
        return torch.nn.functional.normalize(v, dim=0)

    def stats(self, recs: pl.DataFrame) -> pl.DataFrame:
        """Per-record statistics as in features.record_stats, looked up in full-corpus tables."""
        rows = []
        for r in recs.select("id", "country", "name_sorted", "state", "name_tok").iter_rows(named=True):
            toks = r["name_tok"] or []
            dfs = sorted((self.tok_df.get(t, 0) + (1 if r["id"] == "S1-QUERY" else 0), t) for t in toks)
            key = f"{r['country']}|{r['name_sorted'] or ''}"
            rows.append(dict(
                id=r["id"],
                rare_tok=dfs[0][1] if dfs else None,
                rare_logdf=math.log1p(dfs[0][0]) if dfs else None,
                common_logdf=math.log1p(max(d for d, _ in dfs)) if dfs else None,
                unseen_tok_share=sum(d <= 2 for d, _ in dfs) / len(dfs) if dfs else None,
                n_s1_same_name=self.n_s1.get(key, 0) + (1 if r["id"] == "S1-QUERY" else 0),
                n_rec_same_name=self.n_rec.get(key, 0),
                n_s1_same_name_state=self.n_s1_state.get(f"{key}|{r['state'] or ''}", 0) + (1 if r["id"] == "S1-QUERY" else 0),
            ))
        return pl.DataFrame(rows, schema_overrides={"rare_logdf": pl.Float64, "common_logdf": pl.Float64,
                                                     "unseen_tok_share": pl.Float64, "rare_tok": pl.Utf8})

    def match(self, name: str, address: str, country: str, top: int = 10) -> dict:
        t0 = time.time()
        q = pl.DataFrame({"id": ["S1-QUERY"], "src": pl.Series([1], dtype=pl.Int8), "name": [name or ""],
                          "address": [address or ""], "country": [country]})
        q = N.normalize(q, self.dicts)
        qv = self.embed(q["emb_text"][0])
        mask = self.is_rec & (self.country == country)
        if not mask.any():  # unseen country label: search everything
            mask = self.is_rec
        rows = np.where(mask)[0]
        sims = self.emb[torch.from_numpy(rows)] @ qv
        k = min(K_FWD, len(rows))
        v, i = sims.topk(k)
        v, rr = v.numpy(), rows[i.numpy()]
        # the query stands in for its nearest reference (S1) entity when it is a variant of one, so that
        # entity is taken out of the reverse comparison instead of competing with its own copy
        s1_rows = np.where(~self.is_rec & (self.country == country))[0]
        replaced = None
        if len(s1_rows):
            s = self.emb[torch.from_numpy(s1_rows)] @ qv
            j = int(s.argmax())
            if float(s[j]) >= 0.85:
                replaced = (int(s1_rows[j]), float(s[j]))
        rev = self.rev[rr].copy()
        if replaced is not None:
            rev[self.rev_idx[rr] == replaced[0]] = -1.0
            rev = -np.sort(-rev, axis=1)
        rank_rev = 1 + (rev > v[:, None]).sum(1)
        rank_rev = np.where(rank_rev > 10, 99, rank_rev)
        top2 = np.sort(np.c_[rev[:, :2], v], axis=1)[:, ::-1]
        cand = pl.DataFrame({
            "s1_id": ["S1-QUERY"] * k, "rec_id": self.index["id"].gather(rr),
            "cos": v.astype(np.float32), "rank_fwd": np.arange(1, k + 1, dtype=np.int16),
            "rank_rev": rank_rev.astype(np.int16),
            "s1_top1_cos": np.float32(v[0]), "s1_top2_cos": np.float32(v[1] if k > 1 else v[0]),
            "rec_top1_cos": top2[:, 0].astype(np.float32), "rec_top2_cos": top2[:, 1].astype(np.float32),
        }).filter((pl.col("rank_fwd") <= KF) | (pl.col("rank_rev") <= KR))
        recs = pl.concat([q.select(Fe.REC_COLS),
                          self.index.filter(pl.col("id").is_in(cand["rec_id"].implode())).select(Fe.REC_COLS)],
                         how="vertical_relaxed")
        pf = Fe.pair_frame(cand, recs, self.stats(recs))
        f = Fe.group_features(Fe.features_chunk(pf))
        X = f.select([pl.col(c).cast(pl.Float32).fill_null(-1).fill_nan(-1) for c in self.feats]).to_numpy()
        dm = xgb.DMatrix(X, feature_names=self.feats)
        p = self.bst.predict(dm)
        contrib = self.bst.predict(dm, pred_contribs=True)[:, :-1]
        scored = f.select("s1_id", "rec_id").with_columns(pl.Series("p", p))
        sel = set(D.select_expected_f(scored, device="cpu")["rec_id"].to_list()) if len(p) else set()

        info = self.index.filter(pl.col("id").is_in(scored["rec_id"].implode())).select(
            "id", "src", "name", "address", "country")
        info = {r["id"]: r for r in info.iter_rows(named=True)}
        out = []
        for j in np.argsort(-p)[:top]:
            rid = scored["rec_id"][int(j)]
            c = contrib[j]
            order = np.argsort(-np.abs(c))[:6]
            out.append(dict(
                id=rid, source=f"S{info[rid]['src']}", name=info[rid]["name"], address=info[rid]["address"],
                p=float(p[j]), matched=rid in sel, cos=float(f["cos"][int(j)]),
                name_sim=float(f["nm_tset"][int(j)]), addr_sim=float(f["ad_tset"][int(j)]),
                reasons=[dict(feature=self.feats[o], label=label(self.feats[o]), value=float(X[j, o]),
                              impact=float(c[o])) for o in order],
            ))
        return dict(
            query=dict(name=name, address=address, country=country, name_clean=q["name_full"][0],
                       address_clean=q["addr_str"][0], legal=q["legal"][0], state=q["state"][0]),
            replaced=None if replaced is None else dict(id=self.index["id"][replaced[0]], name=self.index["name"][replaced[0]],
                                                        cos=round(replaced[1], 3)),
            candidates=out, n_retrieved=k, n_scored=len(p), n_matched=len(sel),
            ms=round((time.time() - t0) * 1000),
        )
