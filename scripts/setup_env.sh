#!/usr/bin/env bash
# One-time environment + data setup for the BER pipeline. Safe to re-run.
# Never modifies the system torch / numpy / CUDA packages: new packages go into a
# venv that reuses system site-packages, and pip runs against a constraints file
# pinning every protected package to its currently installed version.
set -uo pipefail
ROOT="$HOME/ber"
LOG="$ROOT/logs"
mkdir -p "$LOG" "$ROOT/work" "$ROOT/output" "$ROOT/hf_cache"
exec > >(tee -a "$LOG/setup.log") 2>&1
rm -f "$LOG/SETUP_DONE" "$LOG/SETUP_FAILED"
fail() { echo "!! $1"; touch "$LOG/SETUP_FAILED"; exit 1; }
echo "== start $(date)"

echo "== zip check"
ls -l "$ROOT/raw/student_resource.zip" || fail "zip missing: SFTP path mapping differs from container home"

echo "== system versions (before)"
python3 -c "import sys,torch,numpy;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.is_available());print('numpy',numpy.__version__)" | tee "$LOG/versions_before.txt"
nvidia-smi --query-gpu=driver_version,name,memory.used,memory.total --format=csv,noheader
python3 -m venv -h >/dev/null 2>&1 && echo "venv module present"
python3 -m pip freeze > "$LOG/system_freeze.txt" 2>/dev/null
echo "system packages: $(wc -l < "$LOG/system_freeze.txt")"

echo "== extract dataset"
if [ ! -f "$ROOT/student_resource/dataset/test/test_source3.tsv" ]; then
  python3 - <<PY || fail "extract failed"
import zipfile
z = zipfile.ZipFile("$ROOT/raw/student_resource.zip")
members = [m for m in z.namelist() if m.startswith("student_resource/") and ".DS_Store" not in m]
z.extractall("$ROOT", members=members)
print("extracted", len(members), "entries")
PY
fi
wc -l "$ROOT"/student_resource/dataset/*/*.tsv

echo "== venv"
VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || {
    echo "venv with pip failed; retrying without pip (system pip module is reused)"
    rm -rf "$VENV"
    python3 -m venv --system-site-packages --without-pip "$VENV" || fail "cannot create venv"
  }
fi
PY="$VENV/bin/python"
$PY -m pip --version || fail "no pip available inside venv"

echo "== constraints (protected packages pinned to installed versions)"
grep -iE '^(torch|torchvision|torchaudio|numpy|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' "$LOG/system_freeze.txt" > "$ROOT/constraints.txt"
grep -qi '^torch==' "$ROOT/constraints.txt" || python3 -c "import torch;print('torch=='+torch.__version__)" >> "$ROOT/constraints.txt"
grep -qi '^numpy==' "$ROOT/constraints.txt" || python3 -c "import numpy;print('numpy=='+numpy.__version__)" >> "$ROOT/constraints.txt"
cat "$ROOT/constraints.txt"

echo "== install packages"
PKGS=("polars>=1.30" "rapidfuzz>=3.8" "xgboost-cu12>=3.0" "anyascii>=0.3" "indic-transliteration>=2.3" "scikit-learn>=1.4" "jellyfish>=1.0")
$PY -m pip install --upgrade-strategy only-if-needed -c "$ROOT/constraints.txt" "${PKGS[@]}" || fail "core package install failed"
NLP=("transformers>=4.51,<6" "huggingface-hub<2" "safetensors" "tokenizers")
$PY -m pip install --upgrade-strategy only-if-needed -c "$ROOT/constraints.txt" "${NLP[@]}" || echo "!! NLP package install failed (stage 2 only; continuing)"

echo "== verify"
$PY - <<'PY' || fail "verification failed"
import importlib
for m in ["polars", "rapidfuzz", "xgboost", "anyascii", "indic_transliteration", "sklearn", "jellyfish", "transformers", "torch", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print("OK  ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("FAIL", m, repr(e))
import numpy as np, xgboost as xgb, torch
from rapidfuzz.process import cpdist
from rapidfuzz import fuzz
print("cpdist", cpdist(["abc", "hello"], ["abd", "hallo"], scorer=fuzz.ratio, workers=-1))
X = np.random.rand(20000, 10).astype(np.float32); y = (X[:, 0] > .5).astype(np.float32)
bst = xgb.train({"device": "cuda", "tree_method": "hist", "objective": "binary:logistic"}, xgb.DMatrix(X, y), 10)
print("xgboost cuda train OK; USE_CUDA =", xgb.build_info().get("USE_CUDA"))
print("torch", torch.__version__, "cuda ok", torch.cuda.is_available(), "| numpy", np.__version__)
from anyascii import anyascii
print("translit:", anyascii("राम मार्केटिंग प्राइवेट लिमिटेड"))
PY

echo "== system versions (after) - must match before"
python3 -c "import sys,torch,numpy;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.is_available());print('numpy',numpy.__version__)" > "$LOG/versions_after.txt"
diff "$LOG/versions_before.txt" "$LOG/versions_after.txt" && echo "system versions unchanged"
$PY -m pip freeze > "$LOG/venv_freeze.txt"

echo "== download retriever model (for stage 1 blocking)"
HF_HOME="$ROOT/hf_cache" $PY - <<'PY' || echo "!! model download failed (can retry later)"
import time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download("intfloat/multilingual-e5-base",
                      allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "tokenizer*"])
print("model at", p, f"in {time.time() - t:.0f}s")
PY

touch "$LOG/SETUP_DONE"
echo "== done $(date)"
