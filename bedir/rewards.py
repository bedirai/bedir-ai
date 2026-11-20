"""Deterministic reward functions for Mizan (judge), Bedir (debater) and Tutulma (red team).

Every reward function in this module is pure Python (standard library only) and
deterministic given a transcript and source document, so the exact same scoring
logic runs during local unit tests, offline evaluation and ``verl`` RL training.

References:
- Kenton et al. (2026), *Debate Training Reduces Reward Hacking in RLAIF* (arXiv:2608.17776)
- Damani et al. (2025/ICLR 2026), *RLCR: Training LMs to Reason About Their Uncertainty* (arXiv:2507.16806)
- Turtel et al. (2025), *Outcome-Based RL to Predict the Future* (arXiv:2505.17989)
- Post-rationalization in RLVR-trained search agents (arXiv:2609.23053)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .verify import locate

DEFAULT_TURN_WORD_LIMIT = 150


def _clamp_prob(p: float) -> float:
    if math.isnan(p) or math.isinf(p):
        raise ValueError(f"probability must be finite, got {p!r}")
    return max(0.0, min(1.0, float(p)))


def _word_count(text: str) -> int:
    return len((text or "").split())


@dataclass(frozen=True)
class MizanReward:
    """Breakdown of one Mizan judge reward step."""

    total: float
    brier: float
    position_penalty: float
    format_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DebaterReward:
    """Breakdown of one Bedir debater reward step."""

    total: float
    judge_term: float
    quote_gate_passed: bool
    verified_quotes: int
    total_quotes: int
    length_penalty: float
    counterfactual_penalty: float
    format_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TutulmaReward:
    """Breakdown of one Tutulma red-team reward step."""

    total: float
    wrong_shift: float
    quote_gate_passed: bool
    length_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mizan_reward(
    p_for: float,
    outcome: int,
    *,
    p_swapped: float | None = None,
    lambda_pos: float = 0.25,
    format_ok: bool = True,
    format_penalty_value: float = 0.25,
) -> MizanReward:
    """Proper-scoring reward for the Mizan judge model.

    Uses negative Brier score ``-(p - y)^2`` (in ``[-1, 0]``), combined with a
    position-invariance penalty when the same debate is also scored with the
    two sides swapped: ``|p(A, B) - (1 - p(B, A))|``.

    Use without GRPO std-normalisation (``norm_adv_by_std_in_grpo=false``) so
    calibration gradients are preserved rather than pushed toward 0/1 extremes.
    """
    if outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
    p = _clamp_prob(p_for)
    brier = -((p - float(outcome)) ** 2)
    pos_pen = 0.0
    if p_swapped is not None:
        ps = _clamp_prob(p_swapped)
        pos_pen = lambda_pos * abs(p - (1.0 - ps))
    fmt_pen = 0.0 if format_ok else float(format_penalty_value)
    total = brier - pos_pen - fmt_pen
    return MizanReward(
        total=round(total, 6),
        brier=round(brier, 6),
        position_penalty=round(pos_pen, 6),
        format_penalty=round(fmt_pen, 6),
    )


def debater_reward(
    side: str,
    p_for: float,
    claims: Sequence[Mapping[str, Any]],
    document: str,
    *,
    word_limit: int = DEFAULT_TURN_WORD_LIMIT,
    lambda_len: float = 0.005,
    counterfactual_deltas: Sequence[float] | None = None,
    counterfactual_min_delta: float = 0.02,
    lambda_cf: float = 0.15,
    format_ok: bool = True,
    format_penalty_value: float = 0.1,
    eclipsed_floor: float = -1.0,
) -> DebaterReward:
    """Zero-sum, quote-gated reward for the Bedir debater.

    Rules enforced deterministically:
    1. Every claim must carry a quote that passes :func:`bedir.verify.locate`
       against ``document``. If any quote fails (or no claims are supplied),
       the deterministic evidence gate returns ``eclipsed_floor`` (default
       ``-1.0``) regardless of how persuasive the prose is.
    2. Base payoff is zero-sum from the judge's probability:
       ``2 * (p_side - 0.5)`` in ``[-1, 1]``.
    3. Turns exceeding ``word_limit`` words (default 150, per Kenton et al.
       2026) receive a linear length penalty to prevent verbosity hacking.
    4. Optional counterfactual citation necessity check: when
       ``counterfactual_deltas`` gives ``|p_with_quote - p_without_quote|`` per
       claim, quotes below ``counterfactual_min_delta`` incur a post-
       rationalisation penalty (arXiv:2609.23053).
    """
    if side not in ("for", "against"):
        raise ValueError(f"side must be 'for' or 'against', got {side!r}")
    p = _clamp_prob(p_for)
    p_side = p if side == "for" else (1.0 - p)
    judge_term = 2.0 * (p_side - 0.5)

    total_quotes = len(claims)
    verified_quotes = 0
    total_words = 0
    for claim in claims:
        total_words += _word_count(str(claim.get("text", "")))
        quote = str(claim.get("quote", ""))
        if locate(document, quote).verified:
            verified_quotes += 1

    gate_passed = total_quotes > 0 and verified_quotes == total_quotes
    excess_words = max(0, total_words - int(word_limit))
    len_pen = lambda_len * excess_words

    cf_pen = 0.0
    if counterfactual_deltas is not None and total_quotes > 0:
        unnecessary = sum(1 for d in counterfactual_deltas if abs(float(d)) < counterfactual_min_delta)
        cf_pen = lambda_cf * (unnecessary / len(counterfactual_deltas)) if counterfactual_deltas else 0.0

    fmt_pen = 0.0 if format_ok else float(format_penalty_value)

    if not gate_passed:
        return DebaterReward(
            total=round(float(eclipsed_floor), 6),
            judge_term=round(judge_term, 6),
            quote_gate_passed=False,
            verified_quotes=verified_quotes,
            total_quotes=total_quotes,
            length_penalty=round(len_pen, 6),
            counterfactual_penalty=round(cf_pen, 6),
            format_penalty=round(fmt_pen, 6),
        )

    total = max(-1.0, min(1.0, judge_term - len_pen - cf_pen - fmt_pen))
    return DebaterReward(
        total=round(total, 6),
        judge_term=round(judge_term, 6),
        quote_gate_passed=True,
        verified_quotes=verified_quotes,
        total_quotes=total_quotes,
        length_penalty=round(len_pen, 6),
        counterfactual_penalty=round(cf_pen, 6),
        format_penalty=round(fmt_pen, 6),
    )


def tutulma_reward(
    p_before: float,
    p_after: float,
    true_outcome: int,
    claims: Sequence[Mapping[str, Any]],
    document: str,
    *,
    word_limit: int = DEFAULT_TURN_WORD_LIMIT,
    lambda_len: float = 0.005,
) -> TutulmaReward:
    """Adversarial red-team reward used strictly to harden Mizan.

    Measures how far a quote-verified argument moves the judge *away* from the
    ground-truth outcome. Attacks that fail the quote-hash gate are rejected
    so Tutulma only surfaces vulnerabilities that survive the evidence gate.
    Tutulma weights are kept private; only aggregate hack-rate curves are
    published.
    """
    if true_outcome not in (0, 1):
        raise ValueError(f"true_outcome must be 0 or 1, got {true_outcome!r}")
    pb = _clamp_prob(p_before)
    pa = _clamp_prob(p_after)
    # Positive when Mizan moves toward the wrong outcome.
    wrong_shift = (pb - pa) if true_outcome == 1 else (pa - pb)

    total_words = sum(_word_count(str(c.get("text", ""))) for c in claims)
    gate_passed = len(claims) > 0 and all(locate(document, str(c.get("quote", ""))).verified for c in claims)
    len_pen = lambda_len * max(0, total_words - int(word_limit))

    if not gate_passed:
        return TutulmaReward(
            total=-1.0,
            wrong_shift=round(wrong_shift, 6),
            quote_gate_passed=False,
            length_penalty=round(len_pen, 6),
        )

    total = max(-1.0, min(1.0, wrong_shift - len_pen))
    return TutulmaReward(
        total=round(total, 6),
        wrong_shift=round(wrong_shift, 6),
        quote_gate_passed=True,
        length_penalty=round(len_pen, 6),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw model output (optionally fenced)."""
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            s = "\n".join(lines[1:-1]).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def verl_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
) -> float:
    """Drop-in ``custom_reward_function`` hook compatible with ``verl``.

    Supported ``data_source`` values:
    - ``"bedir_mizan_judge"``: expects JSON with ``p_for`` (0..1 or 0..100).
    - ``"bedir_debater_selfplay"``: expects JSON with ``claims`` and
      ``extra_info`` containing ``document``, ``side`` and ``p_for``.
    - ``"bedir_tutulma_redteam"``: expects JSON with ``claims`` and
      ``extra_info`` containing ``document``, ``p_before`` and ``p_after``.
    """
    info = extra_info or {}
    payload = _extract_json_object(solution_str)

    if data_source == "bedir_mizan_judge":
        if payload is None or "p_for" not in payload:
            return -1.0
        raw_p = float(payload["p_for"])
        p = raw_p / 100.0 if raw_p > 1.0 else raw_p
        p_swapped = info.get("p_swapped")
        if p_swapped is not None:
            raw_ps = float(p_swapped)
            p_swapped = raw_ps / 100.0 if raw_ps > 1.0 else raw_ps
        outcome = int(ground_truth["outcome"] if isinstance(ground_truth, Mapping) else ground_truth)
        return mizan_reward(p, outcome, p_swapped=p_swapped, format_ok=True).total

    if data_source == "bedir_debater_selfplay":
        if payload is None or not isinstance(payload.get("claims"), list):
            return -1.0
        raw_p = float(info.get("p_for", 0.5))
        p = raw_p / 100.0 if raw_p > 1.0 else raw_p
        return debater_reward(
            side=str(info.get("side", "for")),
            p_for=p,
            claims=payload["claims"],
            document=str(info.get("document", "")),
            word_limit=int(info.get("word_limit", DEFAULT_TURN_WORD_LIMIT)),
            counterfactual_deltas=info.get("counterfactual_deltas"),
            format_ok=True,
        ).total

    if data_source == "bedir_tutulma_redteam":
        if payload is None or not isinstance(payload.get("claims"), list):
            return -1.0
        outcome = int(ground_truth["outcome"] if isinstance(ground_truth, Mapping) else ground_truth)
        raw_pb = float(info.get("p_before", 0.5))
        raw_pa = float(info.get("p_after", 0.5))
        pb = raw_pb / 100.0 if raw_pb > 1.0 else raw_pb
        pa = raw_pa / 100.0 if raw_pa > 1.0 else raw_pa
        return tutulma_reward(
            p_before=pb,
            p_after=pa,
            true_outcome=outcome,
            claims=payload["claims"],
            document=str(info.get("document", "")),
            word_limit=int(info.get("word_limit", DEFAULT_TURN_WORD_LIMIT)),
        ).total

    raise ValueError(f"unknown BedirAI data_source: {data_source!r}")


def summarise_batch_rewards(rewards: Iterable[float]) -> dict[str, float]:
    """Compute mean, std, min, max and non-zero-variance flag for a RL batch."""
    vals = [float(x) for x in rewards]
    if not vals:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "zero_variance": 1.0}
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    return {
        "count": float(len(vals)),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "zero_variance": 1.0 if std < 1e-9 else 0.0,
    }
