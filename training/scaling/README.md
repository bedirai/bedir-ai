# 4-to-32-GPU Scaling Harness

Phase 0 produces empirical strong/weak scaling measurements (4, 8, 16, and 32 GPUs) on AMD ROCm and NVIDIA CUDA before committing to the Phase 1 (~110,000 H100-hour) run.

## Usage

Inspect the schema with the synthetic placeholder file:

```bash
python3 -m training.scaling.benchmark_harness training/scaling/schema_example.json
```

When real pilot runs complete, save the measured JSON under `training/scaling/results/<hardware>-<date>.json` with `"synthetic_schema_example": false` and regenerate the Markdown table.

## Measured fields required per point

| Field | Meaning |
|---|---|
| `gpus` | Full accelerator count (for MI250X, 1 GPU = 1 full module / 2 GCDs) |
| `nodes` | Physical nodes allocated |
| `tokens_per_sec` | End-to-end effective trained tokens per second across all GPUs |
| `step_seconds` | Wall-clock time per RL optimisation step (rollout + ref logprobs + update) |
| `peak_hbm_gb` | Peak HBM usage per GCD / GPU |
