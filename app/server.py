"""Demo API for the business entity resolution project.

  GET  /api/health                 liveness + bundle info
  GET  /api/overview               dataset stats, model comparison, per-country, hard pairs, curves, importance
  GET  /api/search?q=&country=     search the sampled test S1 entities (SQLite FTS5)
  GET  /api/entity/{id}            one S1 entity with its final-stage candidates and submitted matches
  POST /api/match                  live matching of a typed record {name, address, country}
  POST /api/batch                  TSV upload (business_name, business_address, country) -> matches per row

Run: uvicorn server:app --app-dir app --port 7400
"""
import csv
import io
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import DATA, ROOT, Engine

RES = ROOT / "results"
app = FastAPI(title="Business Entity Resolution API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = Engine()


def db():
    con = sqlite3.connect(f"file:{DATA / 'demo.sqlite'}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


class Query(BaseModel):
    name: str = Field(..., max_length=300)
    address: str = Field("", max_length=500)
    country: str = Field(..., max_length=40)


@app.get("/api/health")
def health():
    return {"ok": True, "index_records": engine.index.height, "countries": engine.meta["countries"],
            "load_seconds": round(engine.load_s, 1)}


@lru_cache(1)
def overview_payload():
    rows = pl.read_csv(RES / "model_comparison.csv").to_dicts()
    curves = np.load(RES / "curves.npz")
    pr = {}
    for key in curves.files:
        model, arr = key.split("|")
        pr.setdefault(model, {})[arr] = curves[key][:: max(1, len(curves[key]) // 200)].round(4).tolist()
    eda = json.loads((RES / "eda.json").read_text()) if (RES / "eda.json").exists() else None
    return {
        "models": rows,
        "per_country": pl.read_csv(RES / "per_country.csv").to_dicts(),
        "hard_pairs": pl.read_csv(RES / "hard_pairs.csv").to_dicts(),
        "importance": pl.read_csv(RES / "importance.csv").head(15).to_dicts(),
        # the deployed single-stage matcher: gain of the 81 hand-crafted features (no stage-2 context dominating)
        "importance_live": sorted(({"feature": k, "gain": v} for k, v in engine.bst.get_score(importance_type="gain").items()),
                                  key=lambda r: -r["gain"])[:15],
        "ceiling": json.loads((RES / "ceiling.json").read_text()),
        "curves": pr,
        "eda": eda,
    }


@app.get("/api/overview")
def overview():
    return overview_payload()


@app.get("/api/search")
def search(q: str = "", country: str = "", limit: int = 20):
    con = db()
    limit = max(1, min(limit, 50))
    args, where = [], []
    if country:
        where.append("country = ?")
        args.append(country)
    if q.strip():
        terms = " ".join(f'"{t}"*' for t in q.replace('"', " ").split()[:8])
        sql = f"SELECT id, name, address, country FROM s1_fts WHERE s1_fts MATCH ? {'AND ' + where[0] if where else ''} " \
              f"ORDER BY rank LIMIT ?"
        rows = con.execute(sql, [terms, *args, limit]).fetchall()
    else:
        sql = f"SELECT id, name, address, country FROM s1_fts {'WHERE ' + where[0] if where else ''} " \
              f"ORDER BY random() LIMIT ?"
        rows = con.execute(sql, [*args, limit]).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/entity/{s1_id}")
def entity(s1_id: str):
    con = db()
    r = con.execute("SELECT * FROM records WHERE id = ?", [s1_id]).fetchone()
    if r is None or r["src"] != 1:
        raise HTTPException(404, "unknown S1 entity")
    cands = con.execute(
        "SELECT r.id, r.src, r.name, r.address, p.p, p.matched FROM pairs p JOIN records r ON r.id = p.rec_id "
        "WHERE p.s1_id = ? ORDER BY p.matched DESC, p.p DESC", [s1_id]).fetchall()
    return {"entity": dict(r), "candidates": [dict(c) for c in cands]}


@app.post("/api/match")
def match(q: Query):
    if not q.name.strip():
        raise HTTPException(422, "name is required")
    return engine.match(q.name, q.address, q.country)


@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    out = []
    for i, row in enumerate(reader):
        if i >= 50:
            break
        name = row.get("business_name") or row.get("name") or ""
        if not name.strip():
            continue
        res = engine.match(name, row.get("business_address") or row.get("address") or "", row.get("country") or "India", top=5)
        out.append({"row": i + 1, "name": name, "matches": [c["id"] for c in res["candidates"] if c["matched"]],
                    "best": res["candidates"][0] if res["candidates"] else None})
    return {"rows": out}
