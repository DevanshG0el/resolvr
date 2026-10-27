#!/usr/bin/env bash
# #29 probe: #27 with France's decision calibration nudged towards matching (logit bias +0.2 / +0.4 in the sub11
# decision rule for unseen countries only). Everything else as #27 -> output_b020, output_b040.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/BIAS_DONE" "$L/BIAS_FAILED"
cd "$R"
PY="${BER_PYTHON:-}"; [ -n "$PY" ] || { [ -x "$R/.venv/bin/python" ] && PY="$R/.venv/bin/python" || PY=python3; }
for b in 020 040; do
  "$PY" -c "import json,os; W=os.path.expanduser('~/ber/work/'); d=json.load(open(W+'decision_sub11.json')); d['cfg']['bias']=int('$b')/100; json.dump(d, open(W+'decision_sub11_b$b.json','w'))" || { touch "$L/BIAS_FAILED"; exit 1; }
  BER_OUT="$R/output_b$b" BER_SCORED_SEEN="$W/test_scored_final.parquet" BER_DECISION_SEEN="$W/decision_final.json" \
    BER_SCORED_UNSEEN="$W/test_scored_r7q.parquet" BER_DECISION_UNSEEN="$W/decision_sub11_b$b.json" \
    BER_GUARD_SCORED="$W/test_scored_r7u.parquet" \
    BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
    BER_LOGTAG="b${b}F" bash scripts/run.sh final || { touch "$L/BIAS_FAILED"; exit 1; }
done
touch "$L/BIAS_DONE"
