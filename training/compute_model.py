"""First-principles FLOP, memory and GPU-hour budget model for BedirAI training phases.

Matches the Phase 0 / Phase 1 / Phase 2 figures published on https://bedirsavasi.com/training/
so infrastructure partners and compute-grant reviewers can inspect every assumption,
reproduce the 110,000 H100-hour Phase 1 budget and convert job tables across
NVIDIA CUDA (H100, H200) and AMD ROCm (MI250X, MI300X, MI355X) accelerators.

Usage:
    python3 -m training.compute_model
    python3 -m training.compute_model --phase phase1 --hardware MI355X
    python3 -m training.compute_model --phase phase1 --sensitivity --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareSpec:
    """Accelerator peak dense BF16 throughput and memory capacity."""

    name: str
    vendor: str
    stack: str
    peak_bf16_tflops: float
    hbm_gb: int
    h100_effective_ratio: float  # effective throughput relative to H100-SXM
    notes: str


HARDWARE: dict[str, HardwareSpec] = {
    "H100": HardwareSpec(
        name="NVIDIA H100 SXM",
        vendor="NVIDIA",
        stack="CUDA",
        peak_bf16_tflops=989.0,
        hbm_gb=80,
        h100_effective_ratio=1.0,
        notes="Reference unit for training budgets (dense BF16 peak).",
    ),
    "H200": HardwareSpec(
        name="NVIDIA H200 SXM",
        vendor="NVIDIA",
        stack="CUDA",
        peak_bf16_tflops=989.0,
        hbm_gb=141,
        h100_effective_ratio=1.15,
        notes="Same BF16 peak as H100; higher HBM3e capacity and bandwidth lift rollout decode MFU.",
    ),
    "MI250X": HardwareSpec(
        name="AMD Instinct MI250X (full module, 2 GCD)",
        vendor="AMD",
        stack="ROCm",
        peak_bf16_tflops=383.0,
        hbm_gb=128,
        h100_effective_ratio=0.385,
        notes="64 GB per GCD; Phase 0 pilot hardware for 8B runs and 4-to-32-GPU scaling tables.",
    ),
    "MI300X": HardwareSpec(
        name="AMD Instinct MI300X",
        vendor="AMD",
        stack="ROCm",
        peak_bf16_tflops=1307.0,
        hbm_gb=192,
        h100_effective_ratio=1.2,
        notes="192 GB HBM3 fits 32B weights + KV cache comfortably on single-node rollouts.",
    ),
    "MI355X": HardwareSpec(
        name="AMD Instinct MI355X",
        vendor="AMD",
        stack="ROCm",
        peak_bf16_tflops=2500.0,
        hbm_gb=288,
        h100_effective_ratio=1.95,
        notes="288 GB HBM3e, ~2.5 PFLOPS dense BF16 (~5 PFLOPS FP8); strong fit for Phase 1 32B RL.",
    ),
}


@dataclass(frozen=True)
class JobSpec:
    """One compute line item in a training phase."""

    phase: str
    stage: str
    work: str
    h100_hours: float
    recommended_gpus: int
    runs: int
    active_params_b: float
    formula: str


def flops_cpt_or_sft(active_params: float, tokens: float) -> float:
    """Standard 6 * N * D training FLOPs (forward + backward)."""
    return 6.0 * active_params * tokens


def flops_rl_train(active_params: float, trained_tokens: float) -> float:
    """RL policy update FLOPs: 6 * N * D (policy fwd+bwd) + 2 * N * D (ref/old logprob fwd) = 8 * N * D."""
    return 8.0 * active_params * trained_tokens


def flops_rollout(active_params: float, tokens: float) -> float:
    """Forward-only inference FLOPs (prefill or decode): 2 * N * D."""
    return 2.0 * active_params * tokens


def gpu_hours_from_flops(flops: float, hw: HardwareSpec, mfu: float) -> float:
    """Convert total FLOPs into accelerator-hours given effective MFU."""
    if not (0.0 < mfu <= 1.0):
        raise ValueError(f"mfu must be in (0, 1], got {mfu!r}")
    flops_per_gpu_hour = mfu * (hw.peak_bf16_tflops * 1e12) * 3600.0
    return flops / flops_per_gpu_hour


def estimate_rl_run_h100_hours(
    *,
    active_params_b: float,
    steps: int,
    prompts_per_step: int,
    group_size: int,
    decode_tokens_per_episode: int,
    prefill_tokens_per_episode: int,
    train_tokens_per_episode: int,
    train_mfu: float = 0.40,
    prefill_mfu: float = 0.40,
    decode_mfu: float = 0.15,
) -> dict[str, float]:
    """First-principles H100-hour breakdown for one multi-turn RL run."""
    n = active_params_b * 1e9
    episodes = steps * prompts_per_step * group_size
    hw = HARDWARE["H100"]

    decode_h = gpu_hours_from_flops(flops_rollout(n, episodes * decode_tokens_per_episode), hw, decode_mfu)
    prefill_h = gpu_hours_from_flops(flops_rollout(n, episodes * prefill_tokens_per_episode), hw, prefill_mfu)
    train_h = gpu_hours_from_flops(flops_rl_train(n, episodes * train_tokens_per_episode), hw, train_mfu)
    return {
        "episodes": float(episodes),
        "decode_h100_hours": round(decode_h, 1),
        "prefill_h100_hours": round(prefill_h, 1),
        "train_h100_hours": round(train_h, 1),
        "total_h100_hours": round(decode_h + prefill_h + train_h, 1),
    }


def estimate_memory_gb(
    params_b: float,
    *,
    dtype_bytes: int = 2,
    optimizer_bytes_per_param: int = 12,
    fsdp_world_size: int = 8,
    activation_overhead_gb: float = 12.0,
) -> dict[str, float]:
    """Approximate per-GPU memory for BF16 inference vs FSDP AdamW training."""
    weights_gb = (params_b * 1e9 * dtype_bytes) / (1024**3)
    # Weights (2B) + Gradients (2B) + FP32 master/Adam states (12B) = 16B/param sharded
    total_train_gb = (params_b * 1e9 * (dtype_bytes * 2 + optimizer_bytes_per_param)) / (1024**3)
    per_gpu_train_gb = (total_train_gb / max(1, fsdp_world_size)) + activation_overhead_gb
    return {
        "params_b": params_b,
        "bf16_weights_gb": round(weights_gb, 1),
        "fsdp_world_size": float(fsdp_world_size),
        "est_train_gb_per_gpu": round(per_gpu_train_gb, 1),
    }


PHASE_JOBS: dict[str, list[JobSpec]] = {
    "phase0": [
        JobSpec(
            phase="phase0",
            stage="P0-SFT",
            work="Hilal-8B and Mizan-8B SFT (0.9B tokens, steelman + quote-hash format)",
            h100_hours=35.0,
            recommended_gpus=8,
            runs=1,
            active_params_b=8.0,
            formula="6 * 8e9 * 0.9e9 / (0.40 * 989e12 * 3600)",
        ),
        JobSpec(
            phase="phase0",
            stage="P0-Mizan",
            work="Mizan-8B judge RL (GRPO w/o std-norm, Brier + position invariance, 2 runs)",
            h100_hours=310.0,
            recommended_gpus=8,
            runs=2,
            active_params_b=8.0,
            formula="40k prompts * 8 rollouts, 7.5k seq, decode + 8ND update",
        ),
        JobSpec(
            phase="phase0",
            stage="P0-BoN",
            work="Protocol gate: Best-of-N debate vs consultancy pre-test (no RL)",
            h100_hours=215.0,
            recommended_gpus=8,
            runs=1,
            active_params_b=8.0,
            formula="20k prompts * 16 debates * 12k tokens decode",
        ),
        JobSpec(
            phase="phase0",
            stage="P0-SelfPlay",
            work="Bedir-8B debate self-play RL + RLAIF baseline (2 main runs)",
            h100_hours=1230.0,
            recommended_gpus=16,
            runs=2,
            active_params_b=8.0,
            formula="2 * (500 steps * 64 prompts * 8 episodes, 30k train tok/ep)",
        ),
        JobSpec(
            phase="phase0",
            stage="P0-Scaling",
            work="4->8->16->32 GPU scaling benchmarks, ablations & Mizan-Bench v0",
            h100_hours=410.0,
            recommended_gpus=32,
            runs=4,
            active_params_b=8.0,
            formula="Strong/weak scaling grid + 1 ablation run + eval",
        ),
    ],
    "phase1": [
        JobSpec(
            phase="phase1",
            stage="P1-CPT",
            work="Continued pre-training, shared 32B base (~40B public legal/parliamentary tokens)",
            h100_hours=5400.0,
            recommended_gpus=64,
            runs=1,
            active_params_b=32.0,
            formula="6 * 32e9 * 40e9 / (0.40 * 989e12 * 3600)",
        ),
        JobSpec(
            phase="phase1",
            stage="P1-Mizan",
            work="Mizan-32B judge RL (outcome + Brier + position consistency) and hardening",
            h100_hours=32000.0,
            recommended_gpus=64,
            runs=4,
            active_params_b=32.0,
            formula="SFT + 300k transcripts x 3 epochs RL + hardening + weak-to-strong sweeps",
        ),
        JobSpec(
            phase="phase1",
            stage="P1-Bedir",
            work="Bedir-32B self-play debate RL (DAPO + SPIRAL RAE), baselines and ablations",
            h100_hours=55000.0,
            recommended_gpus=64,
            runs=3,
            active_params_b=32.0,
            formula="3 runs x (1000 steps x 128 prompts x 8 episodes; 12k ctx; 8B judge)",
        ),
        JobSpec(
            phase="phase1",
            stage="P1-Tutulma",
            work="Tutulma red-team adversarial RL rounds against frozen/iterated Mizan",
            h100_hours=15000.0,
            recommended_gpus=32,
            runs=3,
            active_params_b=32.0,
            formula="3 iterative red-team attack cycles feeding Mizan hardening set",
        ),
        JobSpec(
            phase="phase1",
            stage="P1-Eval",
            work="Mizan-Bench v1 evaluation, cross-lingual audits and scaling verification",
            h100_hours=3000.0,
            recommended_gpus=16,
            runs=6,
            active_params_b=32.0,
            formula="Time-locked TR/EN/DE evals + master-key & rhetoric perturbation suites",
        ),
    ],
    "phase2": [
        JobSpec(
            phase="phase2",
            stage="P2-CPT",
            work="Continued pre-training, MoE base (~20B active params, 50B multilingual tokens)",
            h100_hours=67000.0,
            recommended_gpus=128,
            runs=1,
            active_params_b=20.0,
            formula="6 * 20e9 * 50e9 / (0.25 * 989e12 * 3600) [MoE MFU]",
        ),
        JobSpec(
            phase="phase2",
            stage="P2-Debate",
            work="MoE debater GSPO self-play RL (2 main runs, 32k context + search)",
            h100_hours=274000.0,
            recommended_gpus=256,
            runs=2,
            active_params_b=20.0,
            formula="2 x (2000 steps x 256 prompts x 8 episodes; GSPO sequence-level)",
        ),
        JobSpec(
            phase="phase2",
            stage="P2-Mizan",
            work="MoE/32B judge RL, Tutulma rounds, ablations and live benchmark",
            h100_hours=140000.0,
            recommended_gpus=128,
            runs=5,
            active_params_b=20.0,
            formula="30k judge + 40k red-team + 50k ablations + 20k eval",
        ),
    ],
}


def build_job_table(phase: str, hardware_key: str = "H100") -> dict[str, Any]:
    """Generate a grant-ready job table for one phase converted to ``hardware_key``."""
    if phase not in PHASE_JOBS:
        raise ValueError(f"unknown phase {phase!r}; choose from {sorted(PHASE_JOBS)}")
    if hardware_key not in HARDWARE:
        raise ValueError(f"unknown hardware {hardware_key!r}; choose from {sorted(HARDWARE)}")
    hw = HARDWARE[hardware_key]
    rows: list[dict[str, Any]] = []
    total_h100 = 0.0
    total_hw = 0.0
    for job in PHASE_JOBS[phase]:
        hw_hours = round(job.h100_hours / hw.h100_effective_ratio, 1)
        wall_hours_per_run = round(hw_hours / (job.recommended_gpus * job.runs), 1)
        total_h100 += job.h100_hours
        total_hw += hw_hours
        rows.append(
            {
                "stage": job.stage,
                "work": job.work,
                "h100_hours": job.h100_hours,
                "hw_gpu_hours": hw_hours,
                "recommended_gpus": job.recommended_gpus,
                "runs": job.runs,
                "wall_hours_per_run": wall_hours_per_run,
                "formula": job.formula,
            }
        )
    for row in rows:
        row["share_pct"] = round(100.0 * row["h100_hours"] / total_h100, 1)
    return {
        "phase": phase,
        "hardware": asdict(hw),
        "total_h100_hours": round(total_h100, 1),
        "total_hw_gpu_hours": round(total_hw, 1),
        "rows": rows,
    }


def phase1_sensitivity() -> list[dict[str, Any]]:
    """Sensitivity analysis for a single 32B Bedir self-play RL main run."""
    scenarios = [
        ("baseline (12k ctx, train MFU 40%, decode MFU 15%)", 12000, 0.40, 0.15),
        ("short context (6k ctx, train MFU 40%, decode MFU 15%)", 6000, 0.40, 0.15),
        ("conservative decode (12k ctx, train MFU 35%, decode MFU 8%)", 12000, 0.35, 0.08),
        ("optimised rollout (12k ctx, train MFU 42%, decode MFU 20%)", 12000, 0.42, 0.20),
    ]
    out: list[dict[str, Any]] = []
    for label, ctx, t_mfu, d_mfu in scenarios:
        # 2 players, each trained on context + dialogue (~2x ctx + 24k)
        train_tok = 2 * (ctx + 12000)
        prefill_tok = int(ctx * 3.3)
        est = estimate_rl_run_h100_hours(
            active_params_b=32.0,
            steps=1000,
            prompts_per_step=128,
            group_size=8,
            decode_tokens_per_episode=20000,
            prefill_tokens_per_episode=prefill_tok,
            train_tokens_per_episode=train_tok,
            train_mfu=t_mfu,
            decode_mfu=d_mfu,
        )
        out.append({"scenario": label, **est})
    return out


def _print_table(report: dict[str, Any], include_sensitivity: bool) -> None:
    hw = report["hardware"]
    print(f"\nBedirAI Compute Job Table — {report['phase'].upper()} on {hw['name']} ({hw['stack']})")
    print(f"  Peak BF16: {hw['peak_bf16_tflops']:.0f} TFLOPS · HBM: {hw['hbm_gb']} GB · Ratio vs H100: {hw['h100_effective_ratio']:.2f}x\n")
    print(f"{'Stage':<12} {'Share':>6} {'H100-h':>10} {'Target-h':>10} {'GPUs':>6} {'Runs':>5} {'Wall/run':>9}  Work")
    print("-" * 108)
    for r in report["rows"]:
        print(
            f"{r['stage']:<12} {r['share_pct']:>5.1f}% {r['h100_hours']:>10,.0f} "
            f"{r['hw_gpu_hours']:>10,.0f} {r['recommended_gpus']:>6d} {r['runs']:>5d} "
            f"{r['wall_hours_per_run']:>8.1f}h  {r['work']}"
        )
    print("-" * 108)
    print(
        f"{'TOTAL':<12} {'100.0%':>6} {report['total_h100_hours']:>10,.0f} "
        f"{report['total_hw_gpu_hours']:>10,.0f}"
    )

    if include_sensitivity:
        print("\nPhase 1 single 32B self-play RL run — first-principles FLOP sensitivity (H100-hours):")
        for item in phase1_sensitivity():
            print(
                f"  • {item['scenario']:<62} -> total {item['total_h100_hours']:>8,.1f} h "
                f"(train {item['train_h100_hours']:,.1f}, decode {item['decode_h100_hours']:,.1f}, prefill {item['prefill_h100_hours']:,.1f})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bedir-compute",
        description="First-principles FLOP and GPU-hour job-table planner for BedirAI.",
    )
    parser.add_argument("--phase", choices=("phase0", "phase1", "phase2", "all"), default="phase1")
    parser.add_argument("--hardware", choices=tuple(HARDWARE.keys()), default="H100")
    parser.add_argument("--sensitivity", action="store_true", help="include Phase 1 FLOP/MFU sensitivity table")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args(argv)

    phases = ("phase0", "phase1", "phase2") if args.phase == "all" else (args.phase,)
    tables = [build_job_table(p, args.hardware) for p in phases]

    if args.json:
        payload: dict[str, Any] = {"tables": tables}
        if args.sensitivity:
            payload["phase1_sensitivity"] = phase1_sensitivity()
        print(json.dumps(payload, indent=2))
        return 0

    for t in tables:
        _print_table(t, include_sensitivity=(args.sensitivity and t["phase"] == "phase1"))
    if args.sensitivity and "phase1" not in phases:
        print("\nPhase 1 single 32B self-play RL run — first-principles FLOP sensitivity (H100-hours):")
        for item in phase1_sensitivity():
            print(f"  • {item['scenario']}: {item['total_h100_hours']:,.1f} H100-h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
