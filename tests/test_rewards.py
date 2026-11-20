import json
import unittest

from bedir.rewards import (
    debater_reward,
    mizan_reward,
    summarise_batch_rewards,
    tutulma_reward,
    verl_compute_score,
)

DOC = (
    "Clause 7.2. If the Landlord fails to complete such repairs within thirty (30) days of the notice, "
    "the Tenant may terminate this Lease without penalty.\n\n"
    "Clause 15.3. Notice of termination shall be given in writing and delivered by registered mail or by hand."
)

VALID_CLAIM = {
    "text": "Clause 7.2 allows termination without penalty after thirty days.",
    "quote": "If the Landlord fails to complete such repairs within thirty (30) days of the notice, the Tenant may terminate this Lease without penalty.",
}

FABRICATED_CLAIM = {
    "text": "The tenant can leave anytime for free.",
    "quote": "The Tenant may leave whenever they want with zero cost.",
}


class MizanRewardTests(unittest.TestCase):
    def test_perfect_calibration_gets_zero_brier_loss(self):
        r = mizan_reward(1.0, outcome=1, p_swapped=0.0)
        self.assertEqual(r.brier, 0.0)
        self.assertEqual(r.position_penalty, 0.0)
        self.assertEqual(r.total, 0.0)

    def test_position_inconsistency_penalises_judge(self):
        consistent = mizan_reward(0.8, outcome=1, p_swapped=0.2)
        inconsistent = mizan_reward(0.8, outcome=1, p_swapped=0.8)
        self.assertGreater(consistent.total, inconsistent.total)
        self.assertAlmostEqual(inconsistent.position_penalty, 0.25 * 0.6, places=5)

    def test_proper_scoring_prefers_calibrated_over_overconfident_wrong(self):
        cautious = mizan_reward(0.4, outcome=1)
        overconfident_wrong = mizan_reward(0.0, outcome=1)
        self.assertGreater(cautious.total, overconfident_wrong.total)


class DebaterRewardTests(unittest.TestCase):
    def test_zero_sum_when_quotes_verified(self):
        r_for = debater_reward("for", p_for=0.75, claims=[VALID_CLAIM], document=DOC)
        r_against = debater_reward("against", p_for=0.75, claims=[VALID_CLAIM], document=DOC)
        self.assertTrue(r_for.quote_gate_passed)
        self.assertAlmostEqual(r_for.total, 0.5, places=5)
        self.assertAlmostEqual(r_against.total, -0.5, places=5)
        self.assertAlmostEqual(r_for.total + r_against.total, 0.0, places=5)

    def test_unverified_quote_hits_hard_gate_floor(self):
        r = debater_reward("for", p_for=0.99, claims=[VALID_CLAIM, FABRICATED_CLAIM], document=DOC)
        self.assertFalse(r.quote_gate_passed)
        self.assertEqual(r.verified_quotes, 1)
        self.assertEqual(r.total, -1.0)

    def test_word_limit_penalises_verbose_critique(self):
        long_claim = {
            "text": " ".join(["argument"] * 200),
            "quote": VALID_CLAIM["quote"],
        }
        r_short = debater_reward("for", p_for=0.8, claims=[VALID_CLAIM], document=DOC, word_limit=150)
        r_long = debater_reward("for", p_for=0.8, claims=[long_claim], document=DOC, word_limit=150)
        self.assertGreater(r_short.total, r_long.total)
        self.assertGreater(r_long.length_penalty, 0.0)

    def test_counterfactual_penalty_flags_post_rationalisation(self):
        r_needed = debater_reward(
            "for", p_for=0.8, claims=[VALID_CLAIM], document=DOC, counterfactual_deltas=[0.18]
        )
        r_unused = debater_reward(
            "for", p_for=0.8, claims=[VALID_CLAIM], document=DOC, counterfactual_deltas=[0.001]
        )
        self.assertEqual(r_needed.counterfactual_penalty, 0.0)
        self.assertGreater(r_unused.counterfactual_penalty, 0.0)
        self.assertGreater(r_needed.total, r_unused.total)


class TutulmaAndVerlTests(unittest.TestCase):
    def test_tutulma_rewards_wrong_shift_only_when_quote_gate_passes(self):
        r_ok = tutulma_reward(0.80, 0.30, true_outcome=1, claims=[VALID_CLAIM], document=DOC)
        self.assertTrue(r_ok.quote_gate_passed)
        self.assertAlmostEqual(r_ok.total, 0.5, places=5)

        r_bad = tutulma_reward(0.80, 0.30, true_outcome=1, claims=[FABRICATED_CLAIM], document=DOC)
        self.assertFalse(r_bad.quote_gate_passed)
        self.assertEqual(r_bad.total, -1.0)

    def test_verl_compute_score_dispatch(self):
        s_judge = verl_compute_score(
            "bedir_mizan_judge",
            '```json\n{"p_for": 80, "verdict": "for"}\n```',
            ground_truth=1,
            extra_info={"p_swapped": 20},
        )
        self.assertAlmostEqual(s_judge, -0.04, places=5)

        s_deb = verl_compute_score(
            "bedir_debater_selfplay",
            json.dumps({"claims": [VALID_CLAIM]}),
            ground_truth=1,
            extra_info={"side": "for", "p_for": 75, "document": DOC},
        )
        self.assertAlmostEqual(s_deb, 0.5, places=5)

        s_red = verl_compute_score(
            "bedir_tutulma_redteam",
            json.dumps({"claims": [VALID_CLAIM]}),
            ground_truth=1,
            extra_info={"p_before": 80, "p_after": 40, "document": DOC},
        )
        self.assertAlmostEqual(s_red, 0.4, places=5)

        self.assertEqual(verl_compute_score("bedir_mizan_judge", "not json", 1), -1.0)

    def test_summarise_batch_rewards_detects_zero_variance(self):
        s = summarise_batch_rewards([0.5, 0.5, 0.5])
        self.assertEqual(s["zero_variance"], 1.0)
        s2 = summarise_batch_rewards([-0.5, 0.5])
        self.assertEqual(s2["zero_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
