# Roadmap & Open Science Releases

BedirAI ships in gated stages: larger training runs begin only after smaller runs meet empirical accuracy, calibration and scaling thresholds.

## 12-month execution plan

| Window | Milestone | Proof point / Gate |
|---|---|---|
| **Months 0–2** | **Phase 0 Pilot:** Hilal-8B SFT, Mizan-8B judge RL, Best-of-N protocol test, 4→32-GPU scaling table on AMD MI250X, `Mizan-Bench v0` | **Gate 1:** Debate lifts Mizan-8B accuracy on time-locked real outcomes over judge-only and single-player baselines; empirical MFU measured. |
| **Months 2–3** | **Phase 1 Launch:** Kick off shared 32B CPT and RL pipeline calibrated on measured Phase 0 throughput numbers | Reproducible multi-node `verl` + `vLLM` scaling logs published. |
| **Months 3–6** | **Core Judge & Design Partners:** Shared 32B CPT (~40B tokens); Mizan-32B outcome RL & position-invariance hardening; first legal and risk design partners | `ECE <= 0.05`, position flip rate `< 5%`, quote validity `>= 99.5%`. |
| **Months 6–10** | **Self-Play & Red-Team Loop:** Bedir-32B self-play RL (DAPO + SPIRAL RAE); 3 cycles of Tutulma adversarial hardening | **Gate 2:** The 8B recipe holds at 32B; PGR `>= 0.45`, Persuasion Shift `<= 0`, post-hardening hack-rate drop `<= 2 pp`. |
| **Months 10–12** | **Open Science Release:** Public release of `Hilal-8B` (Apache-2.0), `Mizan-8B` open weights, `Mizan-Bench v1` and full technical report | Complete scaling curves, hack-rate trajectories and hardware-tuned training recipes published. |

## Open-by-default deliverables

Every milestone contributes reusable public artifacts to the AI safety and open-weights community:

- **Hilal-8B weights** — Apache-2.0 compact debater/judge for single-server private deployment.
- **Mizan-8B weights** — Open-weight calibrated debate judge for scalable-oversight research.
- **Mizan-Bench (v0 & v1)** — Time-locked, multilingual (TR / EN / DE) debate-judge benchmark with verifiable outcomes.
- **Quote-hash verifier, RL rewards & evaluation harness** — Available now in this repository under Apache-2.0.
- **Technical report & hardware recipes** — Full scaling tables, reward-hacking curves and ROCm/CUDA `verl` configurations.

## For infrastructure & research partners

We share Phase 0 pilot logs, empirical scaling tables and technical briefings with compute partners, research groups and design partners. Contact: **merhaba@bedirsavasi.com**.
