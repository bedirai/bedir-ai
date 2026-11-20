# Model Cards

BedirAI is a four-model adversarial family where each model has a single role and learns against the others. The live public [Arena](https://bedirsavasi.com/arena/) currently executes the Bedir protocol on open-weight models via Cloudflare Workers AI; BedirAI's trained models replace them phase by phase.

---

## Bedir — Debater

| Field | Specification |
|---|---|
| **Meaning** | *Bedir*: full moon (complete only in 180° opposition to the sun) |
| **Role** | Two-sided adversarial debater (always argues both sides; no one-sided mode) |
| **Target size** | 8B pilot → ~32B dense (Phase 1) → MoE (Phase 2) |
| **Training** | Continued pre-training on clean public legal & parliamentary corpora; SFT on steelman writing & quoting rules; self-play RL (DAPO + SPIRAL RAE) |
| **Reward** | Zero-sum judge probability `2*(p_side - 0.5)`, hard-gated by verbatim SHA-256 quote verification, 150-word turn cap, counterfactual citation necessity (`bedir.rewards.debater_reward`) |
| **Release** | Hosted API; weights for on-premise enterprise deployment |
| **Status** | Phase 1 (planned; 8B pilot recipe ready) |

## Mizan — Judge

| Field | Specification |
|---|---|
| **Meaning** | *Mizan*: the scales |
| **Role** | Independent calibrated judge that sees **only** verified quotes in randomised order, never the raw document |
| **Target size** | 8B (Phase 0 pilot, intentionally weaker for weak-to-strong oversight) → 32B (Phase 1) |
| **Training** | Outcome-trained GRPO (without std-normalisation) on real appellate, constitutional and peer-review outcomes; hardened against Tutulma and master-key probes |
| **Reward** | Negative Brier score + position-swap invariance penalty (`bedir.rewards.mizan_reward`) |
| **Release** | **Mizan-8B open weights**; Mizan-32B via API and on-premise licence |
| **Status** | Pilot |

## Tutulma — Red Team

| Field | Specification |
|---|---|
| **Meaning** | *Tutulma*: eclipse |
| **Role** | Adversarial red team trained to expose blind spots in Mizan (anchoring, selective omission, length/jargon padding, master-key triggers) while still passing the quote-hash gate |
| **Target size** | 8B (Phase 0) → 32B (Phase 1) |
| **Training** | Adversarial RL against frozen Mizan checkpoints; successful attacks are folded back into Mizan's training set |
| **Reward** | Probability shift toward the wrong outcome (`bedir.rewards.tutulma_reward`) |
| **Release** | **Not released (private by safety policy).** Only aggregate hack-rate curves and hardened Mizan checkpoints are published. |
| **Status** | Phase 1 (planned) |

## Hilal — Compact Open Model

| Field | Specification |
|---|---|
| **Meaning** | *Hilal*: crescent |
| **Role** | Single-server debater/judge for private deployment when documents cannot leave customer infrastructure |
| **Target size** | 8B dense |
| **Training** | SFT + distillation from Bedir and Mizan on verified evidence traces; quantised for single-GPU / single-server serving |
| **Release** | **Open weights under Apache-2.0** |
| **Status** | Pilot |

---

## Safety & dual-use commitments

1. **Both sides, always.** There is no API mode or model mode that generates one-sided persuasion without its steelman counter-argument.
2. **Persuader models stay private.** Because RL-trained persuaders are dual-use (arXiv:2608.11624), Tutulma weights are never published.
3. **Not legal or financial advice.** Bedir surfaces arguments and verifiable verbatim evidence; it does not predict live court outcomes or substitute for qualified counsel.
