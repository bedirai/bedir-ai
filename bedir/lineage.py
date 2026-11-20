"""Data lineage, license gating, PII redaction (KVKK/GDPR) and temporal hold-out checks.

Enforces BedirAI's data governance guarantees in code:
- Frontier model API outputs never enter training splits (evaluation/baseline only).
- Restricted corpora (e.g., ECHR prior to written commercial permission) are blocked
  from training splits and allowed only in evaluation.
- Residual personal data (Turkish TCKN with checksum validation, IBAN, email, phone,
  IPv4) is detected and redacted before data leaves its jurisdiction.
- Time-locked evaluation splits verify strict cutoff separation and n-gram non-overlap.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

ALLOWED_TRAINING_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "CC-BY-4.0",
        "CC0-1.0",
        "Public-Domain",
        "Statutory-Official-Decision",
    }
)

DISALLOWED_TRAINING_PROVIDERS = frozenset(
    {
        "openai-api",
        "anthropic-api",
        "gemini-api",
        "xai-api",
        "frontier-closed-api",
    }
)

EVAL_ONLY_LICENSES = frozenset(
    {
        "ECHR-Eval-Only",
        "CC-BY-NC-4.0",
        "Research-Only",
        "Frontier-API-Output",
    }
)

_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
_RE_PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?(?:\(0?\d{3}\)|0?\d{3})|\(0\d{3}\)|0\d{3})[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")
_RE_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")
_RE_11_DIGITS = re.compile(r"(?<!\d)(\d{11})(?!\d)")


def is_valid_tckn(candidate: str) -> bool:
    """Validate an 11-digit Turkish national ID (TCKN) using its official checksum."""
    if len(candidate) != 11 or not candidate.isdigit() or candidate[0] == "0":
        return False
    d = [int(ch) for ch in candidate]
    odd_sum = d[0] + d[2] + d[4] + d[6] + d[8]
    even_sum = d[1] + d[3] + d[5] + d[7]
    d10 = ((odd_sum * 7) - even_sum) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


@dataclass(frozen=True)
class PolicyDecision:
    """Result of checking a dataset or model source against BedirAI's lineage gate."""

    allowed: bool
    source_id: str
    split: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_source_policy(source: Mapping[str, Any], split: str = "train") -> PolicyDecision:
    """Enforce licensing and zero-frontier-distillation policy on a lineage record.

    Expected keys in ``source``:
    - ``id``: unique identifier
    - ``license``: SPDX or BedirAI canonical tag
    - ``provider`` (optional): e.g. ``"open-weights-local"`` or ``"openai-api"``
    - ``pii_redacted`` (optional bool): required to be True for ``train``
    """
    if split not in ("train", "val", "eval", "baseline"):
        raise ValueError(f"unsupported split: {split!r}")

    source_id = str(source.get("id", "<unknown>"))
    lic = str(source.get("license", "")).strip()
    provider = str(source.get("provider", "")).strip().lower()
    pii_redacted = bool(source.get("pii_redacted", False))

    if split in ("train", "val"):
        if provider in DISALLOWED_TRAINING_PROVIDERS:
            return PolicyDecision(
                False,
                source_id,
                split,
                f"provider {provider!r} is forbidden in training splits (zero frontier distillation)",
            )
        if lic in EVAL_ONLY_LICENSES:
            return PolicyDecision(
                False,
                source_id,
                split,
                f"license {lic!r} is restricted to evaluation/baseline splits",
            )
        if lic not in ALLOWED_TRAINING_LICENSES:
            return PolicyDecision(
                False,
                source_id,
                split,
                f"license {lic!r} is not on the training allowlist",
            )
        if not pii_redacted:
            return PolicyDecision(
                False,
                source_id,
                split,
                "unmasked source blocked: pii_redacted must be true before training/transfer",
            )
        return PolicyDecision(True, source_id, split, "approved for training")

    # eval / baseline splits
    if lic in ALLOWED_TRAINING_LICENSES or lic in EVAL_ONLY_LICENSES or provider in DISALLOWED_TRAINING_PROVIDERS:
        return PolicyDecision(True, source_id, split, "approved for evaluation/baseline only")
    return PolicyDecision(False, source_id, split, f"unrecognised license {lic!r}")


@dataclass(frozen=True)
class PIIFinding:
    """One detected residual personal-data match."""

    kind: str
    start: int
    end: int
    preview: str


def scan_pii(text: str) -> list[PIIFinding]:
    """Scan text for residual personal data (TCKN, IBAN, email, phone, IPv4)."""
    findings: list[PIIFinding] = []
    for match in _RE_11_DIGITS.finditer(text):
        token = match.group(1)
        if is_valid_tckn(token):
            findings.append(PIIFinding("TCKN", match.start(1), match.end(1), token[:2] + "*******" + token[-2:]))
    for kind, pattern in (
        ("IBAN", _RE_IBAN),
        ("EMAIL", _RE_EMAIL),
        ("PHONE", _RE_PHONE),
        ("IPV4", _RE_IPV4),
    ):
        for match in pattern.finditer(text):
            val = match.group(0)
            # Avoid double-flagging 11-digit TCKNs as phone numbers.
            if kind == "PHONE" and any(f.kind == "TCKN" and f.start <= match.start() < f.end for f in findings):
                continue
            findings.append(PIIFinding(kind, match.start(), match.end(), val[:3] + "***"))
    findings.sort(key=lambda f: (f.start, f.end))
    return findings


def redact_pii(text: str) -> tuple[str, list[PIIFinding]]:
    """Replace detected residual PII spans with typed placeholders ``[REDACTED_<KIND>]``."""
    findings = scan_pii(text)
    if not findings:
        return text, []
    # Merge overlapping spans, keeping earliest match.
    non_overlapping: list[PIIFinding] = []
    cursor = -1
    for item in findings:
        if item.start >= cursor:
            non_overlapping.append(item)
            cursor = item.end

    pieces: list[str] = []
    pos = 0
    for item in non_overlapping:
        pieces.append(text[pos : item.start])
        pieces.append(f"[REDACTED_{item.kind}]")
        pos = item.end
    pieces.append(text[pos:])
    return "".join(pieces), non_overlapping


def _token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class HoldoutAudit:
    """Temporal hold-out and contamination audit report."""

    passed: bool
    cutoff_date: str
    train_count: int
    test_count: int
    temporal_violations: int
    exact_hash_overlaps: int
    max_ngram_overlap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_temporal_holdout(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    cutoff_date: str,
    *,
    ngram_n: int = 8,
    max_allowed_ngram_overlap: float = 0.05,
) -> HoldoutAudit:
    """Verify strict temporal separation and low n-gram overlap between train and test splits.

    Each record requires ``decision_date`` (ISO ``YYYY-MM-DD``) and ``document``.
    """
    cutoff = _parse_iso_date(cutoff_date)
    temporal_violations = 0

    train_hashes: set[str] = set()
    train_ngrams: set[tuple[str, ...]] = set()
    for rec in train_records:
        if _parse_iso_date(rec["decision_date"]) > cutoff:
            temporal_violations += 1
        doc = str(rec.get("document", ""))
        train_hashes.add(hashlib.sha256(doc.encode("utf-8")).hexdigest())
        train_ngrams.update(_token_ngrams(doc, ngram_n))

    exact_overlaps = 0
    max_overlap = 0.0
    for rec in test_records:
        if _parse_iso_date(rec["decision_date"]) <= cutoff:
            temporal_violations += 1
        doc = str(rec.get("document", ""))
        h = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if h in train_hashes:
            exact_overlaps += 1
        grams = _token_ngrams(doc, ngram_n)
        if grams:
            overlap = len(grams & train_ngrams) / len(grams)
            if overlap > max_overlap:
                max_overlap = overlap

    passed = (
        temporal_violations == 0
        and exact_overlaps == 0
        and max_overlap <= max_allowed_ngram_overlap
    )
    return HoldoutAudit(
        passed=passed,
        cutoff_date=cutoff.isoformat(),
        train_count=len(train_records),
        test_count=len(test_records),
        temporal_violations=temporal_violations,
        exact_hash_overlaps=exact_overlaps,
        max_ngram_overlap=round(max_overlap, 6),
    )


def manifest_summary(sources: Iterable[Mapping[str, Any]], split: str = "train") -> dict[str, Any]:
    """Audit an iterable of data-lineage records for a target split."""
    decisions = [check_source_policy(src, split=split) for src in sources]
    rejected = [d.to_dict() for d in decisions if not d.allowed]
    return {
        "split": split,
        "total": len(decisions),
        "approved": len(decisions) - len(rejected),
        "rejected": len(rejected),
        "all_approved": len(rejected) == 0,
        "rejections": rejected,
    }
