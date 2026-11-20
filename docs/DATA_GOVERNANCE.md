# Data Governance, Licensing & Privacy

BedirAI enforces clean data lineage in code ([`bedir/lineage.py`](../bedir/lineage.py)). Every corpus and synthetic generator carries a machine-checked licence tag, PII redaction flag and temporal split boundary.

## Core principles

1. **Zero frontier model distillation.** Outputs from closed frontier APIs (OpenAI, Anthropic, Google Gemini API, xAI) **never** enter training or validation sets. Frontier APIs are used exclusively for evaluation, benchmark baselines and weak-to-strong oversight experiments.
2. **Permissive open-weight synthetic lineage.** All synthetic debate transcripts are produced on self-hosted or rented GPU compute using exclusively **MIT** or **Apache-2.0** licensed open-weight models, with model ID, revision hash and licence recorded per batch.
3. **Verifiable real-world outcomes.** Judge rewards come from published official decisions and peer reviews, not another LLM's subjective preference.
4. **De-identification before cross-border transfer.** No unmasked personal data leaves its jurisdiction (KVKK art. 6 & art. 9; GDPR).

## Source allowlist & split policy

| Source category | Legal basis / licence | Allowed splits | Notes |
|---|---|---|---|
| **Turkish published court decisions & legislation** | Statutory official texts (FSEK art. 31) | `train`, `val`, `eval` | PII redaction pass mandatory before training or cross-border transfer (`pii_redacted=true`); criminal/sensitive texts masked before ingestion |
| **German published court decisions** | Official works (§ 5 UrhG) | `train`, `val`, `eval` | De-identified before training |
| **US appellate decisions (Caselaw Access Project)** | Public domain / unrestricted | `train`, `val`, `eval` | Verifiable `affirmed` / `reversed` appellate labels |
| **OpenReview peer reviews & meta-reviews** | CC BY 4.0 (attribution preserved) | `train`, `val`, `eval` | Claim → critique → rebuttal → accept/reject outside law |
| **Permissive open-weight synthetic debates** | Apache-2.0 / MIT weights run on own/rented GPUs | `train`, `val` | Model ID, commit SHA and licence logged in manifest |
| **European Court of Human Rights (HUDOC)** | `ECHR-Eval-Only` | `eval` only | Excluded from training until written commercial training permission is granted |
| **Closed frontier model APIs** | `Frontier-API-Output` | `eval`, `baseline` only | Hard-blocked in `bedir.lineage.check_source_policy(..., split="train")` |

## Programmatic enforcement

```python
from bedir.lineage import check_source_policy, redact_pii, verify_temporal_holdout

# Blocked automatically: frontier API outputs cannot enter training
decision = check_source_policy(
    {"id": "frontier-run-1", "license": "Frontier-API-Output", "provider": "anthropic-api", "pii_redacted": True},
    split="train",
)
assert not decision.allowed

# Approved for evaluation/baseline only
eval_decision = check_source_policy(
    {"id": "frontier-run-1", "license": "Frontier-API-Output", "provider": "anthropic-api", "pii_redacted": True},
    split="eval",
)
assert eval_decision.allowed
```

## Temporal hold-out & anti-contamination

Mizan-Bench enforces strict chronological separation via `verify_temporal_holdout()`:
- Every training record has `decision_date <= cutoff_date`.
- Every test record has `decision_date > cutoff_date`.
- Zero exact document SHA-256 overlap and bounded 8-gram overlap (`<= 5%`) between train and test splits.
