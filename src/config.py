"""Paths and run-wide settings. Override locations with BER_* environment variables."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("BER_ROOT", Path.home() / "ber"))
DATA = Path(os.environ.get("BER_DATA", ROOT / "student_resource" / "dataset"))
SMOKE = os.environ.get("BER_SMOKE", "0") == "1"
WORK = Path(os.environ.get("BER_WORK", ROOT / ("work_smoke" if SMOKE else "work")))
OUT = Path(os.environ.get("BER_OUT", ROOT / ("output_smoke" if SMOKE else "output")))
MODEL_NAME = os.environ.get("BER_BASE_MODEL", "intfloat/multilingual-e5-base")

SEED = 42
# Train S1 entities are split by hash of their id:
#   ghost (S1 removed, its S2/S3 records stay as orphans), A (retriever fine-tuning + dictionaries),
#   B (matcher training), V (validation).
SPLIT_FRACTIONS = {"ghost": 0.20, "A": 0.40, "B": 0.20, "V": 0.20}
SMOKE_FRACTION = 0.02  # share of S1 entities kept in smoke mode

# Retrieval
K_FWD = int(os.environ.get("BER_K_FWD", 40))  # S1 -> S2/S3 neighbours per S1
K_REV = int(os.environ.get("BER_K_REV", 10))  # S2/S3 -> S1 neighbours per record
MAX_LEN = 96
EMB_DIM = 768

for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)
