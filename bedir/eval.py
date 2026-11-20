"""Reproducible evaluation harness for Bedir debates and Mizan-Bench files.

Usage:
    python3 -m bedir.eval --debates examples/debates
    python3 -m bedir.eval --validate-bench benchmarks/mizan_bench_sample.jsonl
    python3 -m bedir.eval --predictions path/to/eval_predictions.jsonl --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .lineage import scan_pii
from .metrics import (
    brier_score,
    cross_lingual_consistency,
    expected_calibration_error,
    hack_rate,
    outcome_accuracy,
    position_flip_rate,
    quote_validity_audit,
    rhetoric_sensitivity,
    truth_advantage,
)


def _load_json_files(target: Path) -> list[dict[str, Any]]:
    paths = sorted(target.glob("*.json")) if target.is_dir() else [target]
    return [json.loads(p.read_text("utf-8")) for p in paths]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text("utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from None
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object")
        rows.append(obj)
    return rows


def audit_debates(target: Path) -> dict[str, Any]:
    """Audit recorded debate JSON transcripts for quote integrity and latency."""
    debates = _load_json_files(target)
    qa = quote_validity_audit(debates)
    latencies: list[float] = []
    langs: list[str] = []
    for rec in debates:
        langs.append(str(rec.get("lang", "unknown")))
        for ev in rec.get("events", []):
            if ev.get("type") == "done" and isinstance(ev.get("ms"), (int, float)):
                latencies.append(float(ev["ms"]) / 1000.0)
    return {
        "mode": "debates_audit",
        "target": str(target),
        "languages": sorted(set(langs)),
        "quote_audit": qa.to_dict(),
        "mean_seconds": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "passed": qa.local_failures == 0 and qa.debates > 0,
    }


def validate_bench(path: Path) -> dict[str, Any]:
    """Validate a Mizan-Bench JSONL file for schema, document hashes and zero residual PII."""
    required = {
        "id",
        "case_group_id",
        "lang",
        "domain",
        "question",
        "document",
        "document_sha256",
        "decision_date",
        "published_after_cutoff",
        "synthetic",
        "license",
    }
    rows = _load_jsonl(path)
    errors: list[str] = []
    seen_ids: set[str] = set()
    langs: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)

    for idx, row in enumerate(rows, start=1):
        missing = sorted(required - set(row.keys()))
        if missing:
            errors.append(f"row {idx}: missing keys {missing}")
            continue
        rid = str(row["id"])
        if rid in seen_ids:
            errors.append(f"row {idx}: duplicate id {rid!r}")
        seen_ids.add(rid)
        doc = str(row["document"])
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest != row["document_sha256"]:
            errors.append(f"row {idx} ({rid}): document_sha256 mismatch")
        if len(doc) < 200 or len(doc) > 12000:
            errors.append(f"row {idx} ({rid}): document length {len(doc)} outside [200, 12000]")
        pii = scan_pii(doc)
        if pii:
            errors.append(f"row {idx} ({rid}): residual PII detected ({[f.kind for f in pii]})")
        if row.get("outcome") not in (None, 0, 1):
            errors.append(f"row {idx} ({rid}): outcome must be 0, 1 or null")
        lang = str(row["lang"])
        langs[lang] += 1
        groups[str(row["case_group_id"])].add(lang)

    return {
        "mode": "bench_validation",
        "path": str(path),
        "rows": len(rows),
        "case_groups": len(groups),
        "languages": dict(sorted(langs.items())),
        "synthetic_rows": sum(1 for r in rows if r.get("synthetic")),
        "errors": errors,
        "passed": len(rows) > 0 and len(errors) == 0,
    }


def evaluate_predictions(path: Path) -> dict[str, Any]:
    """Compute Mizan-Bench metrics over a JSONL prediction file.

    Required keys per row: ``p_for`` (0..1 or 0..100) and ``outcome`` (0 or 1).
    Optional keys enable extra diagnostics:
    - ``p_swapped``: position flip rate
    - ``p_perturbed``: rhetoric sensitivity
    - ``p_attacked``: Tutulma hack rate
    - ``case_group_id``: cross-lingual verdict consistency
    """
    rows = _load_jsonl(path)
    probs = [r["p_for"] for r in rows]
    outcomes = [int(r["outcome"]) for r in rows]

    report: dict[str, Any] = {
        "mode": "predictions_eval",
        "path": str(path),
        "count": len(rows),
        "outcome_accuracy": outcome_accuracy(probs, outcomes),
        "brier_score": brier_score(probs, outcomes),
        "ece_10bin": expected_calibration_error(probs, outcomes, n_bins=10),
        "truth_advantage": truth_advantage(rows),
    }

    swapped_pairs = [(r["p_for"], r["p_swapped"]) for r in rows if r.get("p_swapped") is not None]
    if swapped_pairs:
        report["position_flip_rate"] = position_flip_rate(
            [a for a, _ in swapped_pairs],
            [b for _, b in swapped_pairs],
        )

    perturbed_pairs = [(r["p_for"], r["p_perturbed"]) for r in rows if r.get("p_perturbed") is not None]
    if perturbed_pairs:
        report["rhetoric_sensitivity"] = rhetoric_sensitivity(
            [a for a, _ in perturbed_pairs],
            [b for _, b in perturbed_pairs],
        ).to_dict()

    attacked = [(r["p_for"], r["p_attacked"], int(r["outcome"])) for r in rows if r.get("p_attacked") is not None]
    if attacked:
        report["hack_rate"] = hack_rate(
            [a for a, _, _ in attacked],
            [b for _, b, _ in attacked],
            [y for _, _, y in attacked],
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("case_group_id"):
            grouped[str(r["case_group_id"])].append(float(r["p_for"]))
    cl = cross_lingual_consistency(grouped)
    if cl.groups > 0:
        report["cross_lingual"] = cl.to_dict()

    report["passed"] = True
    return report


def _print_human(result: dict[str, Any]) -> None:
    mode = result.get("mode")
    if mode == "debates_audit":
        qa = result["quote_audit"]
        print(f"Bedir debate audit: {result['target']}")
        print(f"  debates:           {qa['debates']} ({', '.join(result['languages'])})")
        print(f"  claims checked:    {qa['total_claims']}")
        print(f"  quote validity:    {qa['quote_validity_rate'] * 100:.1f}% ({qa['server_verified']}/{qa['total_claims']})")
        print(f"  local hash checks: {qa['local_matches']} passed, {qa['local_failures']} failed")
        if result["mean_seconds"] is not None:
            print(f"  mean latency:      {result['mean_seconds']:.2f} s")
    elif mode == "bench_validation":
        print(f"Mizan-Bench validation: {result['path']}")
        print(f"  rows:        {result['rows']} ({result['case_groups']} case groups)")
        print(f"  languages:   {result['languages']}")
        print(f"  synthetic:   {result['synthetic_rows']}/{result['rows']}")
        print(f"  status:      {'PASS' if result['passed'] else 'FAIL'}")
        for err in result["errors"]:
            print(f"  ERROR: {err}")
    else:
        print(f"Mizan-Bench evaluation: {result['path']} (n={result['count']})")
        for key in (
            "outcome_accuracy",
            "brier_score",
            "ece_10bin",
            "truth_advantage",
            "position_flip_rate",
            "hack_rate",
            "rhetoric_sensitivity",
            "cross_lingual",
        ):
            if key in result:
                print(f"  {key}: {result[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bedir-eval",
        description="Audit Bedir debates, validate Mizan-Bench files and score judge predictions.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debates", type=Path, help="recorded debate JSON file or directory")
    group.add_argument("--validate-bench", type=Path, help="Mizan-Bench JSONL file to validate")
    group.add_argument("--predictions", type=Path, help="evaluation predictions JSONL file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.debates:
        result = audit_debates(args.debates)
    elif args.validate_bench:
        result = validate_bench(args.validate_bench)
    else:
        result = evaluate_predictions(args.predictions)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
