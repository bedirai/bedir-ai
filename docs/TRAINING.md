# Training Program

Every BedirAI model learns by competing against the others in one adversarial loop: **Bedir** argues both sides, the deterministic **quote-hash gate** removes unverified claims, **Tutulma** attacks the judge, and **Mizan** learns calibrated verdicts from real outcomes.

This document mirrors [bedirsavasi.com/training](https://bedirsavasi.com/training/) and ties every figure to executable code in [`training/compute_model.py`](../training/compute_model.py) and [`training/configs/`](../training/configs).

## Three phases, gated by empirical proof points

| Phase | Scope | Compute | Status |
|---|---|---|---|
| **Phase 0 · Pilot** | Hilal-8B & Mizan-8B; 4→32-GPU scaling measurements on AMD MI250X; Mizan-Bench v0 | 8B scale (~2,200 H100-h eq.) | Pilot |
| **Phase 1 · Core models** | Shared 32B CPT; Mizan-32B RL & hardening; Bedir-32B self-play RL; Tutulma rounds | **≈ 110,000 H100-hours** (range 100k–130k) | 12 months (planned) |
| **Phase 2 · Scale** | Mixture-of-experts debater (~20B active), GSPO self-play RL, additional languages & domains | ≈ 500,000 H100-hours | From 2027 |

**Hard transition gate:** Phase 1 (~32B) starts **only after** Phase 0 proves that (1) multi-turn debate lifts Mizan-8B accuracy on time-locked real outcomes over single-player and judge-only baselines, and (2) 4-to-32-GPU scaling logs validate achieved MFU on target hardware.

## Six stages, each with a gate

1. **Stage A — Data & SHA-256 span index:** Public court decisions, legislation and CC BY 4.0 peer reviews, de-identified (KVKK/GDPR) and indexed span-by-span. Enforced programmatically by [`bedir/lineage.py`](../bedir/lineage.py).
2. **Stage B — Supervised fine-tuning (SFT):** Steelman writing, strict verbatim quoting rules and cross-examination format, using debates generated strictly by permissively licensed open-weight models (MIT / Apache-2.0).
3. **Stage C — Judge RL (`Mizan`):** GRPO **without std-normalisation** (`norm_adv_by_std_in_grpo: false`, per arXiv:2505.17989) using Brier score + position-swap consistency penalty + master-key negative probes (`training/configs/phase0_mizan_8b_grpo.yaml`).
4. **Stage D — Protocol pre-test (Best-of-N, no RL):** Compares single-critique, 2-turn debate, 3-turn debate, cross-examination and consultancy at Bo{1, 4, 16} to verify Truth Advantage before spending RL compute.
5. **Stage E — Debater self-play RL (`Bedir`):** DAPO clip-higher (`0.20 / 0.28`) + SPIRAL Role-Conditioned Advantage Estimation (RAE) + FP32 LM head + zero-variance prompt filtering + deterministic quote-hash gate + 150-word turn cap (`training/configs/phase0_bedir_8b_selfplay.yaml`, `phase1_bedir_32b_selfplay.yaml`). Phase 2 uses GSPO for MoE stability.
6. **Stage F — Red-team hardening loop (`Tutulma`):** Adversarial RL against a frozen Mizan checkpoint under the quote-hash gate (`training/configs/phase1_tutulma_redteam.yaml`). Every successful attack is harvested into Mizan's hardening set across 3 cycles.

## Phase 1 compute budget (run by run)

| Stage | Work | H100-hours | Share |
|---|---|---:|---:|
| `P1-CPT` | Continued pre-training, shared 32B base (~40B public tokens) | 5,400 | 5% |
| `P1-Mizan` | Mizan-32B judge RL (outcome + Brier + position invariance) & hardening | 32,000 | 29% |
| `P1-Bedir` | Bedir-32B self-play RL (DAPO + SPIRAL RAE), baselines & ablations | 55,000 | 50% |
| `P1-Tutulma` | Tutulma red-team adversarial RL rounds | 15,000 | 13% |
| `P1-Eval` | Evaluation, Mizan-Bench v1, cross-lingual & perturbation audits | 3,000 | 3% |
| **Total** | **Range: 100,000–130,000 H100-hours** | **≈ 110,000** | **100%** |

Reproduce the full FLOP breakdown, convert to AMD ROCm (`MI250X`, `MI300X`, `MI355X`) or NVIDIA (`H100`, `H200`), and run the MFU/context sensitivity sweep:

```bash
python3 -m training.compute_model --phase phase1 --hardware H100 --sensitivity
python3 -m training.compute_model --phase phase1 --hardware MI355X
```

**First-principles sizing:** `6·N·D` FLOPs for CPT/SFT, `8·N·D` for RL policy updates (policy fwd+bwd + reference logprob fwd) and `2·N·D` for prefill/decode rollouts, assuming 40% training MFU and 15% decode MFU on H100 (BF16). Phase 0 replaces assumptions with measured numbers via [`training/scaling/benchmark_harness.py`](../training/scaling/benchmark_harness.py).

## Hardware-agnostic stack

- **AMD ROCm:** MI250X (Phase 0 8B pilot & scaling grid), MI300X, MI355X (Phase 1 32B target)
- **NVIDIA CUDA:** H100 SXM, H200 SXM
- **RL & rollouts:** `verl` (FSDP) + `vLLM` rollouts
- **Live public Arena:** Cloudflare Workers AI

## Frontier model APIs: evaluation only, never training data

Frontier model APIs are seated across the table from Mizan strictly for oversight research and evaluation — **0% of frontier API outputs enter any training set**:

| Allocation | Share | Purpose |
|---|---:|---|
| Oversight experiments | 45% | Frontier models debate while smaller Mizan judges (weak-to-strong oversight test) |
| Benchmark baselines | 25% | Public Mizan-Bench leaderboard comparisons against frontier judges and debaters |
| Red-team evaluation | 20% | Frontier models attempt to fool hardened Mizan checkpoints |
| Engineering | 10% | Evaluation and tooling agents across the repository |

This separation is enforced programmatically in [`bedir/lineage.py`](../bedir/lineage.py): any record from a closed frontier provider fails `check_source_policy(..., split="train")`.
