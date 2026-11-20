# Mizan-Bench (Schema Preview)

**Mizan-Bench** is a time-locked, multilingual (Turkish, English, German) benchmark for evaluating debate judges and adversarial reasoning models against real, checkable outcomes.

> **Status:** `mizan_bench_sample.jsonl` in this directory is a **9-item synthetic schema preview** built from the sample documents in [`examples/documents/`](../examples/documents). Its purpose is to make the schema, SHA-256 document manifest, PII check and evaluation CLI executable today. Real time-locked appellate and peer-review releases (`Mizan-Bench v0` and `v1`) ship during Phase 0 and Phase 1 (see [`docs/ROADMAP.md`](../docs/ROADMAP.md)).

## Why time-locked?

Base models can memorise published court rulings and peer reviews. Every real Mizan-Bench release enforces a strict **temporal hold-out**:

- Training splits contain only records dated on or before `cutoff_date`.
- Evaluation splits contain only decisions published strictly after `cutoff_date`, verified with `bedir.lineage.verify_temporal_holdout` for date ordering, exact SHA-256 non-overlap and 8-gram overlap thresholds.

## Record schema (JSONL)

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier (`<domain>-<lang>-<slug>`) |
| `case_group_id` | string | Shared identifier linking parallel language versions for cross-lingual consistency |
| `lang` | string | `en`, `tr` or `de` |
| `domain` | string | `legal`, `policy`, `risk`, `peer_review` |
| `question` | string | Binary yes-or-no question evaluated by the two debaters and Mizan |
| `document` | string | Plain-text source document (200–12,000 chars for Arena; longer in Phase 1) |
| `document_sha256` | string | UTF-8 SHA-256 hex digest of `document` |
| `decision_date` | string | ISO date (`YYYY-MM-DD`) of the real decision (synthetic date in preview) |
| `published_after_cutoff` | bool | True when strictly after the model training cutoff |
| `outcome` | `0 \| 1 \| null` | Real verifiable outcome (`1` = FOR / upheld / violation, `0` = AGAINST / reversed / no violation, `null` for synthetic debate cases without a ground-truth court ruling) |
| `outcome_type` | string | `substantive`, `procedural` or `synthetic_open` |
| `synthetic` | bool | Explicitly marks synthetic schema items |
| `license` | string | Canonical lineage tag checked by `bedir.lineage.check_source_policy` |

## Validate the benchmark file

```bash
python3 -m bedir.eval --validate-bench benchmarks/mizan_bench_sample.jsonl
```

The validator verifies required fields, unique IDs, document SHA-256 hashes, character bounds and zero residual PII (TCKN, IBAN, email, phone, IPv4).
