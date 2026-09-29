#!/usr/bin/env bash
# nanoproof's own MCTS harness on miniF2F (prover_eval.py via train_nanoproof.sh eval).
#   NP_CKPT=<model_NNNNNN.pt> scripts/bench_nanoproof.sh
# Knobs: BUDGET  simulations per theorem (default 512)
#        SPLIT   valid | test (default valid)
#        PILOT   first N theorems only, in file order (default: all)
#        NP_WORK_DIR, NP_NUM_SAMPLES, NP_FIRST_TOKEN_CAP, NP_DISABLE_SOLVERS as in train_nanoproof.sh
#        NP_LEAN_PROCS   Lean workers (default 24; capped to what the node's open-file
#                        table holds), NP_WARMUP_BATCH concurrent Mathlib imports (default 8)
# Output: <ckpt dir>/eval_<step>_minif2f[-test]_<budget>/{theorems.jsonl,summary.toml}
#         (printed at the end; pass it to scripts/compare.py). Runs on the GPU node.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NP_CKPT:?set NP_CKPT to the model_NNNNNN.pt checkpoint}"
BUDGET="${BUDGET:-512}"; SPLIT="${SPLIT:-valid}"

extra="--force"                                    # redo the dir if this budget ran before
[ -n "${PILOT:-}" ] && extra="--max-theorems $PILOT $extra"
trap 'scripts/train_nanoproof.sh stop' EXIT        # leanserver (24 Mathlib workers) is not left behind
NP_NUM_SIMULATIONS="$BUDGET" NP_SPLIT="$SPLIT" NP_EVAL_EXTRA_ARGS="$extra" \
    scripts/train_nanoproof.sh eval

step="$(basename "$NP_CKPT" .pt)"; step="${step#model_}"
suffix=""; [ "$SPLIT" = test ] && suffix="-test"
echo "results: $(dirname "$NP_CKPT")/eval_${step}_minif2f${suffix}_${BUDGET}"
