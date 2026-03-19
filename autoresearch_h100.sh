#!/bin/bash
# =============================================================================
# AUTORESEARCH LOOP FOR H100 - Parameter Golf Challenge
# =============================================================================
# Usage on RunPod:
#   cd /workspace/parameter-golf
#   chmod +x autoresearch_h100.sh
#   ./autoresearch_h100.sh
#
# This script systematically tests hyperparameter configurations,
# logs results, and identifies the best setup under the 16MB limit.
# =============================================================================

set -euo pipefail

RESULTS_FILE="autoresearch_h100_results.tsv"
BEST_FILE="autoresearch_best.txt"
MAX_SIZE=16000000  # 16MB limit (decimal bytes)
NPROC=${NPROC:-1}  # Set to 8 for 8xH100 final runs

# Current best config as baseline
BEST_BPB=99.0

# --- Logging ---
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Header ---
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "experiment\tval_bpb\tint8_bpb\tsize_bytes\tparams\tstatus\tconfig" > "$RESULTS_FILE"
fi

# --- Run one experiment ---
run_experiment() {
    local name="$1"
    shift
    local env_vars=("$@")

    log "========== EXPERIMENT: $name =========="
    log "Config: ${env_vars[*]}"

    # Build the env string
    local cmd="RUN_ID=${name}"
    for var in "${env_vars[@]}"; do
        cmd="$cmd $var"
    done
    cmd="$cmd torchrun --standalone --nproc_per_node=$NPROC train_gpt.py"

    log "Running: $cmd"

    # Capture output
    local logfile="logs/autoresearch_${name}.log"
    mkdir -p logs
    eval "$cmd" 2>&1 | tee "$logfile"

    # Parse results
    local val_bpb=$(grep "final_int8_zlib_roundtrip_exact" "$logfile" | grep -oP 'val_bpb:\K[0-9.]+' || echo "FAIL")
    local size=$(grep "Total submission size int8+zlib:" "$logfile" | grep -oP '[0-9]+(?= bytes)' || echo "0")
    local params=$(grep "model_params:" "$logfile" | grep -oP '[0-9]+' || echo "0")
    local pre_quant_bpb=$(grep -E "^step:.*/.*val_bpb:" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")

    # Determine status
    local status="FAIL"
    if [ "$val_bpb" != "FAIL" ] && [ "$size" -gt 0 ]; then
        if [ "$size" -le "$MAX_SIZE" ]; then
            # Compare using bc for float comparison
            if echo "$val_bpb < $BEST_BPB" | bc -l | grep -q 1; then
                status="NEW_BEST"
                BEST_BPB="$val_bpb"
                echo "$name $val_bpb $size" > "$BEST_FILE"
                log "!!! NEW BEST: $val_bpb BPB ($size bytes) !!!"
            else
                status="KEEP"
            fi
        else
            status="OVER_BUDGET"
        fi
    fi

    # Log to TSV
    local config_str=$(printf '%s ' "${env_vars[@]}")
    echo -e "${name}\t${pre_quant_bpb}\t${val_bpb}\t${size}\t${params}\t${status}\t${config_str}" >> "$RESULTS_FILE"

    log "Result: val_bpb=$val_bpb size=$size status=$status"
    log "========================================="
}

# =============================================================================
# EXPERIMENTS TO RUN
# =============================================================================
# Each experiment inherits the defaults from train_gpt.py (our current best:
#   dim=1152, heads=18, kv_heads=6, layers=5, loops=3, mlp_mult=2,
#   matrix_lr=0.05, scalar_lr=0.05, qat_start=0.7, tied_embed_lr=0.05)
#
# We override only what changes per experiment via env vars.
# =============================================================================

log "Starting autoresearch loop on $(hostname)"
log "GPUs: $NPROC, Max size: $MAX_SIZE bytes"

# --- 0. Baseline: current config with full training ---
run_experiment "baseline_current" \
    "MAX_WALLCLOCK_SECONDS=0"

# --- 1. Learning rate sweep (matrix_lr) ---
run_experiment "lr_matrix_0.04" \
    "MAX_WALLCLOCK_SECONDS=0" "MATRIX_LR=0.04"

run_experiment "lr_matrix_0.06" \
    "MAX_WALLCLOCK_SECONDS=0" "MATRIX_LR=0.06"

run_experiment "lr_matrix_0.07" \
    "MAX_WALLCLOCK_SECONDS=0" "MATRIX_LR=0.07"

# --- 2. QAT start fraction sweep ---
run_experiment "qat_0.5" \
    "MAX_WALLCLOCK_SECONDS=0" "QAT_START_FRAC=0.5"

run_experiment "qat_0.6" \
    "MAX_WALLCLOCK_SECONDS=0" "QAT_START_FRAC=0.6"

run_experiment "qat_0.8" \
    "MAX_WALLCLOCK_SECONDS=0" "QAT_START_FRAC=0.8"

# --- 3. KV heads sweep ---
run_experiment "kv_heads_3" \
    "MAX_WALLCLOCK_SECONDS=0" "NUM_KV_HEADS=3"

run_experiment "kv_heads_9" \
    "MAX_WALLCLOCK_SECONDS=0" "NUM_KV_HEADS=9"

# --- 4. Recurrence depth sweep ---
run_experiment "layers4_loops4" \
    "MAX_WALLCLOCK_SECONDS=0" "NUM_LAYERS=4" "NUM_RECURRENCE_LOOPS=4"

run_experiment "layers6_loops3" \
    "MAX_WALLCLOCK_SECONDS=0" "NUM_LAYERS=6" "NUM_RECURRENCE_LOOPS=3"

run_experiment "layers5_loops4" \
    "MAX_WALLCLOCK_SECONDS=0" "NUM_LAYERS=5" "NUM_RECURRENCE_LOOPS=4"

# --- 5. Width sweep (may need kv_heads adjustment for budget) ---
run_experiment "dim1088_kv8" \
    "MAX_WALLCLOCK_SECONDS=0" "MODEL_DIM=1088" "NUM_HEADS=17" "NUM_KV_HEADS=8"

run_experiment "dim1216_kv4" \
    "MAX_WALLCLOCK_SECONDS=0" "MODEL_DIM=1216" "NUM_HEADS=19" "NUM_KV_HEADS=4"

# --- 6. Warmdown iters sweep ---
run_experiment "warmdown_1600" \
    "MAX_WALLCLOCK_SECONDS=0" "WARMDOWN_ITERS=1600"

run_experiment "warmdown_2000" \
    "MAX_WALLCLOCK_SECONDS=0" "WARMDOWN_ITERS=2000"

# --- 7. Embed LR sweep ---
run_experiment "embed_lr_0.8" \
    "MAX_WALLCLOCK_SECONDS=0" "EMBED_LR=0.8"

run_experiment "embed_lr_0.4" \
    "MAX_WALLCLOCK_SECONDS=0" "EMBED_LR=0.4"

# --- 8. Combined best (after checking results above, manually update) ---
# run_experiment "combined_best" \
#     "MAX_WALLCLOCK_SECONDS=0" "MATRIX_LR=..." "QAT_START_FRAC=..." ...

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============ AUTORESEARCH COMPLETE ============"
log "Results saved to: $RESULTS_FILE"
if [ -f "$BEST_FILE" ]; then
    log "Best config: $(cat $BEST_FILE)"
fi
log ""
log "Full results:"
column -t -s $'\t' "$RESULTS_FILE"
log "================================================"
