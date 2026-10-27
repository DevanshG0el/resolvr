#!/usr/bin/env bash
# France self-training round 5 on the final (#20) French labels (weak supervision: the filler/swap rule and the
# decoy cap act as labelling functions). Positives: #20's selected French pairs with p >= 0.8 after the rule;
# negatives: p <= 0.02, which includes the decoy-capped type-word swaps. Starts from the round-1 French checkpoints,
# cross-fitted as before, then the #7 matcher re-predicts France and the same rule is re-applied -> fillers_sub21f.
set -uo pipefail
R="$HOME/ber"; W="$R/work"; L="$R/logs"
rm -f "$L/R5_DONE" "$L/R5_FAILED"
fail() { echo "failed at: $1" > "$L/R5_FAILED"; exit 1; }
cd "$R"
for f in decision xgb_stage1 xgb_stage2; do cp "$W/${f}_sub7.json" "$W/$f.json"; done
BER_PSEUDO_ROUND=5 BER_PSEUDO_SRC=test_scored_fillers_sub9f8.parquet BER_PSEUDO_SELECTED="$R/output/fillers_sub9f8/matching_results.tsv" \
  BER_PSEUDO_NEG=0.02 BER_LOGTAG=r5 bash scripts/run.sh pseudo predict || fail pseudo_predict
cp "$W/test_scored.parquet" "$W/test_scored_sub21.parquet"
CUDA_VISIBLE_DEVICES= BER_DEVICE=cpu BER_FILLERS_SHARE=0.80 BER_FILLERS_SWAP2=1 BER_FILLERS_DECOY_CAP=0.02 BER_FILLERS_DECOY_ADDR=0 \
  BER_SCORED="$W/test_scored_sub21.parquet" BER_VARIANT_NAME=sub21f BER_CANDIDATES="$R/output_sub9/candidate_pairs.tsv" \
  BER_LOGTAG=fil21 bash scripts/run.sh fillers || fail fillers
cp "$W/xenc_test_sub9.parquet" "$W/xenc_test.parquet"; cp "$W/xenc_bge_test_sub9.parquet" "$W/xenc_bge_test.parquet"
touch "$L/R5_DONE"
