"""4-to-32-GPU scaling measurement analyser for compute-grant proposals.

Reads measured throughput logs (JSON) produced by Phase 0 pilot runs on
AMD ROCm (MI250X/MI300X/MI355X) or NVIDIA CUDA (H100/H200), computes strong
scaling efficiency and achieved MFU, and prints a Markdown table ready for
technical review.

Usage:
    python3 -m training.scaling.benchmark_harness training/scaling/schema_example.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from training.compute_model import HARDWARE


def analyse_scaling(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute per-GPU throughput, scaling efficiency and achieved MFU."""
    hw_key = str(payload["hardware"])
    if hw_key not in HARDWARE:
        raise ValueError(f"unknown hardware {hw_key!r}; expected one of {sorted(HARDWARE)}")
    hw = HARDWARE[hw_key]
    params_b = float(payload["active_params_b"])
    flops_per_token = float(payload.get("flops_per_token_multiplier", 8.0)) * (params_b * 1e9)

    points = sorted(payload["measurements"], key=lambda m: int(m["gpus"]))
    if not points:
        raise ValueError("measurements list must be non-empty")

    base = points[0]
    base_gpus = int(base["gpus"])
    base_tps_per_gpu = float(base["tokens_per_sec"]) / base_gpus

    rows: list[dict[str, Any]] = []
    for pt in points:
        gpus = int(pt["gpus"])
        tps = float(pt["tokens_per_sec"])
        tps_per_gpu = tps / gpus
        efficiency = tps_per_gpu / base_tps_per_gpu if base_tps_per_gpu > 0 else 0.0
        achieved_tflops_per_gpu = (tps_per_gpu * flops_per_token) / 1e12
        mfu = achieved_tflops_per_gpu / hw.peak_bf16_tflops
        rows.append(
            {
                "gpus": gpus,
                "nodes": int(pt.get("nodes", max(1, gpus // 4))),
                "tokens_per_sec": round(tps, 1),
                "tokens_per_sec_per_gpu": round(tps_per_gpu, 1),
                "scaling_efficiency_pct": round(efficiency * 100.0, 1),
                "achieved_mfu_pct": round(mfu * 100.0, 1),
                "step_seconds": float(pt["step_seconds"]),
                "peak_hbm_gb": float(pt["peak_hbm_gb"]),
            }
        )

    return {
        "run_name": str(payload["run_name"]),
        "hardware": hw.name,
        "stack": hw.stack,
        "active_params_b": params_b,
        "synthetic_schema_example": bool(payload.get("synthetic_schema_example", False)),
        "base_gpus": base_gpus,
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    if report["synthetic_schema_example"]:
        lines.append("> **Schema example only (synthetic placeholder numbers).** Replace with measured Phase 0 pilot logs before citing in a proposal.\n")
    lines.append(f"### Scaling table: {report['run_name']} ({report['hardware']}, {report['active_params_b']:.0f}B)\n")
    lines.append("| GPUs | Nodes | Total tok/s | tok/s / GPU | Scaling eff. | Achieved MFU | Step (s) | Peak HBM (GB) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in report["rows"]:
        lines.append(
            f"| {r['gpus']} | {r['nodes']} | {r['tokens_per_sec']:,.1f} | "
            f"{r['tokens_per_sec_per_gpu']:,.1f} | {r['scaling_efficiency_pct']:.1f}% | "
            f"{r['achieved_mfu_pct']:.1f}% | {r['step_seconds']:.1f} | {r['peak_hbm_gb']:.1f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse 4-to-32-GPU scaling logs and emit a Markdown table.")
    parser.add_argument("log_file", type=Path, help="path to scaling log JSON")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args(argv)

    payload = json.loads(args.log_file.read_text("utf-8"))
    report = analyse_scaling(payload)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
