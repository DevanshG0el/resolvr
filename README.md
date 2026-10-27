# Resolvr: Business Entity Resolution

Match noisy business records from three independent sources (ML Challenge 2026: US, India, and France at test time only). For every Source 1 entity, the pipeline finds all Source 2/3 records that describe the same business. It is scored by per-entity macro F0.5.

| | |
|---|---|
| Held-out macro F0.5 | **0.9891** (precision 0.9979, recall 0.9695) on 221,041 validation entities |
| Blocking | 6.8 candidates per entity, F0.5 ceiling 0.9967 |
| Models compared | 11 (rule, bi-encoder, 2 cross-encoders, LR, RF, MLP, LightGBM, CatBoost, XGBoost, ours) |
| Demo | Next.js + FastAPI, live matching with SHAP explanations, installable PWA |

## Pipeline

```
raw TSV -> normalise -> retrieve (fine-tuned multilingual-e5 kNN + learned blocker) -> 81 pair features
        -> cross-encoders (e5, BGE-reranker-v2-m3, Qwen3-Reranker-4B LoRA) -> 2-stage XGBoost
        -> exclusivity + expected-F0.5 selection -> matching_results.tsv, candidate_pairs.tsv
```

All models are MIT/Apache-2.0 licensed and at most 4B parameters.

## Repository layout

| Path | What it holds |
|---|---|
| `src/` | The pipeline. `config.py` (paths, splits), `normalize.py` (cleaning and transliteration), `prepare.py` (load, split, mine dictionaries), `retrieve.py` (bi-encoder fine-tuning and kNN), `block.py` (learned blocker), `features.py` / `namefeat.py` (pair features), `xenc.py` / `qwen.py` (cross-encoders), `train.py` (2-stage XGBoost), `decide.py` (metric and expected-F0.5 rule), `predict.py`, `final.py` (submission files). The remaining modules are the studies behind the design: blocking, drift, leave-one-country-out, France pseudo-labels and audits. |
| `scripts/` | `run.sh` runs stages in order. The other scripts are the dated experiment runs (#2 to #29) that produced the leaderboard submissions. |
| `experiments/` | `compare_models.py` (11-model comparison), `eda.py` (data and cleaning statistics), `figures.py` (report figures). |
| `results/` | Comparison, per-country, hard-pair, EDA and blocking CSVs, plus curves. |
| `app/` | `build_index.py` (demo bundle), `engine.py` (live matcher), `server.py` (FastAPI), `serve.sh`, `Dockerfile`. |
| `web/` | Next.js 15 frontend: Live match, Explore, Models, Pipeline. |
| `report/` | `Report_Business_Entity_Resolution.docx`, `build_report.js`, `figures/`. |

## Setup

```bash
python3.12 -m venv ~/ber/.venv && ~/ber/.venv/bin/pip install -r requirements.txt
# data: unzip the challenge's student_resource into ~/ber/student_resource (dataset/train, dataset/test)
```

Large artefacts stay outside the repo and are not committed. They live under `$BER_ROOT` (default `~/ber`): `work/` holds the Parquet files and models, `output/` the submission files.

## Reproduce

```bash
# 1. pipeline (GPU); logs in ~/ber/logs
bash scripts/run.sh prepare retrieve block features namefeat xenc
BER_XENC_TAG=_bge BER_XENC_BASE=BAAI/bge-reranker-v2-m3 bash scripts/run.sh xenc
bash scripts/run.sh qwen
BER_XENC_EXTRA=_bge,_qwen bash scripts/run.sh train predict final   # -> ~/ber/output/*.tsv
# the France-specific route (pseudo-labels, French cross-encoders, rename guard) is in scripts/r5.sh ... ens.sh

# 2. validate the submission
cd ~/ber/student_resource && python3 utils/validate_submission.py \
  --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test

# 3. analysis for the report
python experiments/compare_models.py && python experiments/eda.py && python experiments/figures.py
node report/build_report.js
```

`BER_SMOKE=1` runs every stage on a 2% sample.

## Demo app

```bash
python app/build_index.py          # once: 60k-entity sample, embeddings, live model -> app/data/
cd web && npm ci && npm run build && cd ..
bash app/serve.sh                  # API on :7400, web on :7410
~/opt/cloudflared/cloudflared tunnel --no-autoupdate --config /dev/null --url http://127.0.0.1:7410   # public HTTPS link
```

Or run `docker compose up --build`, which needs `app/data/` and the pipeline's `work/`.

API endpoints:
- `GET /api/health`
- `GET /api/overview`
- `GET /api/search?q=&country=`
- `GET /api/entity/{id}`
- `POST /api/match` with `{name, address, country}`
- `POST /api/batch` with a TSV upload

## Results (V-test, 221,041 entities)

| Model | Macro F0.5 | Precision | Recall |
|---|---|---|---|
| Fuzzy string rule | 0.7435 | 0.7823 | 0.8535 |
| Bi-encoder cosine (e5) | 0.7996 | 0.8452 | 0.8238 |
| Logistic Regression | 0.9515 | 0.9799 | 0.9112 |
| Random Forest | 0.9758 | 0.9924 | 0.9473 |
| Cross-encoder (e5) | 0.9784 | 0.9906 | 0.9564 |
| MLP | 0.9791 | 0.9932 | 0.9546 |
| BGE-reranker-v2-m3 | 0.9802 | 0.9935 | 0.9540 |
| CatBoost | 0.9830 | 0.9952 | 0.9608 |
| LightGBM | 0.9831 | 0.9953 | 0.9611 |
| XGBoost | 0.9840 | 0.9954 | 0.9630 |
| **Ours: 2-stage XGBoost + cross-encoders + expected-F0.5** | **0.9891** | **0.9979** | **0.9695** |
