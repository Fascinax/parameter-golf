#!/usr/bin/env python3
"""
RunPod Autoresearch Runner — Parallel GPU experiment loop.

Runs a queue of hyperparameter experiments, evaluates results, and keeps or
discards each change based on BPB improvement. Supports parallel execution
on multi-GPU pods.

Usage:
    python3 runpod_autoresearch.py                  # sequential (1 GPU)
    python3 runpod_autoresearch.py --parallel        # parallel (all GPUs)
    python3 runpod_autoresearch.py --parallel --max-gpus 4
    python3 runpod_autoresearch.py --experiments experiments.json
    python3 runpod_autoresearch.py --baseline-only   # just run baseline BPB
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


WORKDIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = WORKDIR / "train_gpt.py"
LOGS_DIR = WORKDIR / "logs"
RESULTS_FILE = WORKDIR / "runpod_results.tsv"

# Baseline env vars (same as our best known config)
BASELINE_ENV = {
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "524288",
    "VAL_BATCH_SIZE": "524288",
    "VAL_LOSS_EVERY": "200",
    "WARMUP_STEPS": "5",
    "MAX_WALLCLOCK_SECONDS": "0",
    "TRAIN_LOG_EVERY": "50",
    "MODEL_DIM": "1152",
    "NUM_HEADS": "18",
    "NUM_KV_HEADS": "6",
    "NUM_LAYERS": "5",
    "NUM_RECURRENCE_LOOPS": "3",
    "MATRIX_LR": "0.05",
    "SCALAR_LR": "0.05",
    "EMBED_LR": "0.4",
    "TIED_EMBED_LR": "0.05",
    "MUON_MOMENTUM": "0.95",
    "MUON_BACKEND_STEPS": "5",
    "QAT_START_FRAC": "0.7",
    "ROPE_BASE": "5000",
    "LOGIT_SOFTCAP": "30.0",
}

# Default experiment queue — each entry overrides one or more baseline params
DEFAULT_EXPERIMENTS = [
    {
        "name": "matrix_lr_0.06",
        "desc": "Higher matrix LR — previously gave best BPB but oversize pre-zstd",
        "overrides": {"MATRIX_LR": "0.06"},
    },
    {
        "name": "matrix_lr_0.07",
        "desc": "Even higher matrix LR",
        "overrides": {"MATRIX_LR": "0.07"},
    },
    {
        "name": "scalar_lr_0.07",
        "desc": "Higher scalar LR",
        "overrides": {"SCALAR_LR": "0.07"},
    },
    {
        "name": "embed_lr_0.6",
        "desc": "Higher embedding LR",
        "overrides": {"EMBED_LR": "0.6"},
    },
    {
        "name": "muon_backend_7",
        "desc": "More Muon backend steps (better optimizer but bigger model?)",
        "overrides": {"MUON_BACKEND_STEPS": "7"},
    },
    {
        "name": "dim_1280_heads_20",
        "desc": "Wider model — more params, uses zstd headroom",
        "overrides": {"MODEL_DIM": "1280", "NUM_HEADS": "20"},
    },
    {
        "name": "layers_6_loops_3",
        "desc": "Deeper model — 6 layers × 3 loops = 18 effective",
        "overrides": {"NUM_LAYERS": "6"},
    },
    {
        "name": "loops_4",
        "desc": "More recurrence loops — 5 layers × 4 = 20 effective",
        "overrides": {"NUM_RECURRENCE_LOOPS": "4"},
    },
    {
        "name": "qat_0.5",
        "desc": "Earlier QAT start — more quantization-aware steps",
        "overrides": {"QAT_START_FRAC": "0.5"},
    },
    {
        "name": "qat_0.6",
        "desc": "Slightly earlier QAT start",
        "overrides": {"QAT_START_FRAC": "0.6"},
    },
    {
        "name": "rope_10000",
        "desc": "Standard RoPE base",
        "overrides": {"ROPE_BASE": "10000"},
    },
    {
        "name": "warmdown_50",
        "desc": "Longer warmdown (25% of steps)",
        "overrides": {"WARMDOWN_ITERS": "50"},
    },
    {
        "name": "combo_best_lr",
        "desc": "matrix_lr=0.06 + scalar_lr=0.06 combo",
        "overrides": {"MATRIX_LR": "0.06", "SCALAR_LR": "0.06"},
    },
    {
        "name": "combo_wide_lr",
        "desc": "Wider model + higher lr",
        "overrides": {"MODEL_DIM": "1280", "NUM_HEADS": "20", "MATRIX_LR": "0.06"},
    },
]


@dataclass
class ExperimentResult:
    name: str
    desc: str
    overrides: dict
    gpu_id: int = 0
    val_bpb: float = 0.0
    roundtrip_bpb: float = 0.0
    artifact_bytes: int = 0
    train_loss: float = 0.0
    wall_time_s: float = 0.0
    status: str = "pending"  # pending | running | done | failed | over_budget
    log_file: str = ""


def get_gpu_count() -> int:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            capture_output=True, text=True, timeout=30,
        )
        return int(result.stdout.strip())
    except Exception:
        return 1


def parse_log(log_path: str) -> dict:
    """Extract key metrics from a training log file."""
    metrics = {
        "val_bpb": None,
        "roundtrip_bpb": None,
        "artifact_bytes": None,
        "train_loss": None,
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except FileNotFoundError:
        return metrics

    # Last step val_bpb
    for m in re.finditer(r"step:\d+/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)", text):
        metrics["val_bpb"] = float(m.group(1))

    # Roundtrip BPB
    m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)", text)
    if m:
        metrics["roundtrip_bpb"] = float(m.group(1))

    # Artifact size
    m = re.search(r"Total submission size int8\+zlib: (\d+) bytes", text)
    if m:
        metrics["artifact_bytes"] = int(m.group(1))

    # Last train_loss
    for m in re.finditer(r"train_loss:([\d.]+)", text):
        metrics["train_loss"] = float(m.group(1))

    return metrics


def run_experiment(experiment: dict, gpu_id: int = 0) -> ExperimentResult:
    """Run a single training experiment on a specific GPU."""
    name = experiment["name"]
    desc = experiment.get("desc", "")
    overrides = experiment.get("overrides", {})

    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"runpod_{name}_gpu{gpu_id}.txt"

    # Build environment
    env = os.environ.copy()
    env.update(BASELINE_ENV)
    env.update(overrides)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["RUN_ID"] = f"{name}_gpu{gpu_id}"

    result = ExperimentResult(
        name=name, desc=desc, overrides=overrides,
        gpu_id=gpu_id, log_file=str(log_file), status="running",
    )

    print(f"[GPU {gpu_id}] START: {name} — {desc}")
    print(f"[GPU {gpu_id}]   overrides: {overrides}")

    t0 = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT)],
                env=env, stdout=lf, stderr=subprocess.STDOUT,
                timeout=7200,  # 2h max per experiment
                cwd=str(WORKDIR),
            )

        result.wall_time_s = time.time() - t0

        if proc.returncode != 0:
            result.status = "failed"
            print(f"[GPU {gpu_id}] FAILED: {name} (exit code {proc.returncode})")
            return result

        metrics = parse_log(str(log_file))
        result.val_bpb = metrics["val_bpb"] or 0.0
        result.roundtrip_bpb = metrics["roundtrip_bpb"] or 0.0
        result.artifact_bytes = metrics["artifact_bytes"] or 0
        result.train_loss = metrics["train_loss"] or 0.0

        if result.artifact_bytes > 16_000_000:
            result.status = "over_budget"
        else:
            result.status = "done"

        bpb_str = f"{result.roundtrip_bpb:.4f}" if result.roundtrip_bpb else "N/A"
        size_mb = f"{result.artifact_bytes / 1e6:.2f}" if result.artifact_bytes else "N/A"
        print(f"[GPU {gpu_id}] DONE: {name} — BPB={bpb_str}, size={size_mb}MB, "
              f"time={result.wall_time_s:.0f}s")

    except subprocess.TimeoutExpired:
        result.wall_time_s = time.time() - t0
        result.status = "failed"
        print(f"[GPU {gpu_id}] TIMEOUT: {name}")
    except Exception as e:
        result.wall_time_s = time.time() - t0
        result.status = "failed"
        print(f"[GPU {gpu_id}] ERROR: {name} — {e}")

    return result


def _run_experiment_wrapper(args):
    """Wrapper for ProcessPoolExecutor."""
    experiment, gpu_id = args
    return run_experiment(experiment, gpu_id)


def run_sequential(experiments: list[dict]) -> list[ExperimentResult]:
    """Run experiments one at a time on GPU 0."""
    results = []
    for exp in experiments:
        result = run_experiment(exp, gpu_id=0)
        results.append(result)
        log_result(result)
    return results


def run_parallel(experiments: list[dict], max_gpus: int) -> list[ExperimentResult]:
    """Run experiments in parallel across GPUs. Each GPU runs one experiment at a time."""
    gpu_count = min(get_gpu_count(), max_gpus)
    if gpu_count <= 1:
        print("Only 1 GPU available, falling back to sequential mode.")
        return run_sequential(experiments)

    print(f"=== Parallel mode: {gpu_count} GPUs, {len(experiments)} experiments ===")

    results = []
    # Process experiments in batches of gpu_count
    for batch_start in range(0, len(experiments), gpu_count):
        batch = experiments[batch_start:batch_start + gpu_count]
        batch_args = [(exp, gpu_id) for gpu_id, exp in enumerate(batch)]

        print(f"\n--- Batch {batch_start // gpu_count + 1}: "
              f"experiments {batch_start + 1}-{batch_start + len(batch)} ---")

        # Use ProcessPoolExecutor for true parallelism (separate CUDA contexts)
        with ProcessPoolExecutor(max_workers=len(batch_args)) as executor:
            futures = {
                executor.submit(_run_experiment_wrapper, args): args
                for args in batch_args
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log_result(result)

    return results


def log_result(result: ExperimentResult):
    """Append result to TSV file."""
    header_needed = not RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a") as f:
        if header_needed:
            f.write("name\tstatus\troundtrip_bpb\tval_bpb\tartifact_MB\t"
                    "train_loss\twall_time_s\tgpu\toverrides\tdesc\n")
        artifact_mb = f"{result.artifact_bytes / 1e6:.2f}" if result.artifact_bytes else "-"
        bpb = f"{result.roundtrip_bpb:.4f}" if result.roundtrip_bpb else "-"
        val = f"{result.val_bpb:.4f}" if result.val_bpb else "-"
        f.write(
            f"{result.name}\t{result.status}\t{bpb}\t{val}\t{artifact_mb}\t"
            f"{result.train_loss:.4f}\t{result.wall_time_s:.0f}\t{result.gpu_id}\t"
            f"{json.dumps(result.overrides)}\t{result.desc}\n"
        )


def print_summary(results: list[ExperimentResult]):
    """Print a ranked summary of all results."""
    done = [r for r in results if r.status == "done" and r.roundtrip_bpb > 0]
    done.sort(key=lambda r: r.roundtrip_bpb)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (ranked by roundtrip BPB, lower is better)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Name':<25} {'BPB':<10} {'Size MB':<10} {'Status':<12} {'Time':<8}")
    print("-" * 80)

    for i, r in enumerate(done, 1):
        size = f"{r.artifact_bytes / 1e6:.2f}" if r.artifact_bytes else "-"
        time_str = f"{r.wall_time_s:.0f}s"
        print(f"{i:<5} {r.name:<25} {r.roundtrip_bpb:<10.4f} {size:<10} {r.status:<12} {time_str:<8}")

    failed = [r for r in results if r.status in ("failed", "over_budget")]
    if failed:
        print(f"\nFailed/Over-budget: {len(failed)}")
        for r in failed:
            print(f"  - {r.name}: {r.status}")

    if done:
        best = done[0]
        print(f"\n>>> BEST: {best.name} — BPB={best.roundtrip_bpb:.4f}, "
              f"size={best.artifact_bytes / 1e6:.2f}MB")

    print(f"\nFull results: {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="RunPod Autoresearch Runner")
    parser.add_argument("--parallel", action="store_true",
                        help="Run experiments in parallel across GPUs")
    parser.add_argument("--max-gpus", type=int, default=8,
                        help="Max GPUs to use in parallel mode")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Path to JSON file with experiment definitions")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run the baseline to get reference BPB")
    parser.add_argument("--filter", type=str, default=None,
                        help="Comma-separated experiment names to run (subset)")
    args = parser.parse_args()

    print("=" * 60)
    print("RunPod Autoresearch Runner")
    print("=" * 60)

    gpu_count = get_gpu_count()
    print(f"GPUs available: {gpu_count}")
    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")

    # Load experiments
    if args.baseline_only:
        experiments = [{"name": "baseline", "desc": "Reference baseline", "overrides": {}}]
    elif args.experiments:
        with open(args.experiments) as f:
            experiments = json.load(f)
    else:
        # Always run baseline first for reference
        experiments = [{"name": "baseline", "desc": "Reference baseline", "overrides": {}}]
        experiments += DEFAULT_EXPERIMENTS

    # Filter if requested
    if args.filter:
        names = set(args.filter.split(","))
        experiments = [e for e in experiments if e["name"] in names]

    print(f"Experiments queued: {len(experiments)}")
    for e in experiments:
        print(f"  - {e['name']}: {e.get('desc', '')}")
    print()

    # Run
    if args.parallel and gpu_count > 1:
        # Run baseline first (sequential), then rest in parallel
        baseline = [e for e in experiments if e["name"] == "baseline"]
        rest = [e for e in experiments if e["name"] != "baseline"]

        results = []
        if baseline:
            print("--- Running baseline first ---")
            results += run_sequential(baseline)

        if rest:
            results += run_parallel(rest, min(gpu_count, args.max_gpus))
    else:
        results = run_sequential(experiments)

    print_summary(results)


if __name__ == "__main__":
    main()
