"""BedirAI: evidence-bound debate client, quote-hash verifier, RL rewards, metrics and data lineage."""

from .client import API, DebateError, debate
from .lineage import check_source_policy, redact_pii, scan_pii, verify_temporal_holdout
from .metrics import (
    brier_score,
    cross_lingual_consistency,
    expected_calibration_error,
    hack_rate,
    outcome_accuracy,
    performance_gap_recovered,
    persuasion_shift,
    position_flip_rate,
    quote_validity_audit,
    rhetoric_sensitivity,
    truth_advantage,
)
from .rewards import debater_reward, mizan_reward, tutulma_reward, verl_compute_score
from .verify import Match, check, locate, sha256

__all__ = [
    "API",
    "DebateError",
    "Match",
    "brier_score",
    "check",
    "check_source_policy",
    "cross_lingual_consistency",
    "debate",
    "debater_reward",
    "expected_calibration_error",
    "hack_rate",
    "locate",
    "mizan_reward",
    "outcome_accuracy",
    "performance_gap_recovered",
    "persuasion_shift",
    "position_flip_rate",
    "quote_validity_audit",
    "redact_pii",
    "rhetoric_sensitivity",
    "scan_pii",
    "sha256",
    "truth_advantage",
    "tutulma_reward",
    "verify_temporal_holdout",
    "verl_compute_score",
]
__version__ = "0.3.0"
