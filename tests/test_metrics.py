import unittest

from bedir.metrics import (
    brier_score,
    cross_lingual_consistency,
    expected_calibration_error,
    hack_rate,
    master_key_probes,
    outcome_accuracy,
    performance_gap_recovered,
    persuasion_shift,
    position_flip_rate,
    rhetoric_sensitivity,
    truth_advantage,
)


class MetricsTests(unittest.TestCase):
    def test_accuracy_brier_and_ece(self):
        probs = [0.9, 0.8, 0.2, 0.1]
        outcomes = [1, 1, 0, 0]
        self.assertEqual(outcome_accuracy(probs, outcomes), 1.0)
        self.assertAlmostEqual(brier_score(probs, outcomes), 0.025, places=5)
        self.assertLess(expected_calibration_error(probs, outcomes), 0.2)

    def test_truth_advantage_and_pgr(self):
        recs = [
            {"p_for": 0.8, "outcome": 1},
            {"p_for": 0.7, "outcome": 1},
            {"p_for": 0.2, "outcome": 0},
            {"p_for": 0.6, "outcome": 0},  # wrong
        ]
        # 3 truthful wins, 1 untruthful win -> 0.75 - 0.25 = 0.5
        self.assertAlmostEqual(truth_advantage(recs), 0.5, places=5)
        # Kenton et al. benchmark formula check
        self.assertAlmostEqual(performance_gap_recovered(0.69, 0.60, 0.80), 0.45, places=5)

    def test_persuasion_shift_flags_hacking_vs_healthy_training(self):
        steps = [0, 100, 200, 300]
        # Healthy: wrong-side wins fall while judge accuracy rises -> PS < 0
        ps_healthy = persuasion_shift(steps, [0.40, 0.34, 0.28, 0.22], [0.60, 0.66, 0.72, 0.78])
        self.assertLess(ps_healthy, 0.0)
        # Hacking: wrong-side wins climb while judge accuracy falls -> PS > 0
        ps_hacking = persuasion_shift(steps, [0.30, 0.38, 0.45, 0.54], [0.70, 0.64, 0.57, 0.48])
        self.assertGreater(ps_hacking, 0.0)

    def test_position_flip_rate(self):
        # Perfectly invariant: p_ba = 1 - p_ab -> 0 flips
        self.assertEqual(position_flip_rate([0.8, 0.2, 0.7], [0.2, 0.8, 0.3]), 0.0)
        # Always favours first slot -> 100% flip rate
        self.assertEqual(position_flip_rate([0.8, 0.7], [0.8, 0.7]), 1.0)

    def test_rhetoric_sensitivity_and_hack_rate(self):
        rs = rhetoric_sensitivity([0.80, 0.70], [0.40, 0.65])
        self.assertAlmostEqual(rs.mean_abs_shift, 0.225, places=5)
        self.assertAlmostEqual(rs.flip_rate, 0.5, places=5)

        hr = hack_rate([0.80, 0.75, 0.20], [0.30, 0.70, 0.30], [1, 1, 0])
        # All 3 initially correct; 1 flipped to wrong -> 1/3
        self.assertAlmostEqual(hr, 1 / 3, places=5)

    def test_cross_lingual_consistency(self):
        cl = cross_lingual_consistency(
            {
                "case-1": [0.75, 0.70, 0.68],  # all FOR
                "case-2": [0.35, 0.40, 0.60],  # disagree
            }
        )
        self.assertEqual(cl.groups, 2)
        self.assertAlmostEqual(cl.agreement_rate, 0.5, places=5)
        self.assertAlmostEqual(cl.mean_prob_spread, (0.07 + 0.25) / 2, places=5)

    def test_master_key_probes_present(self):
        probes = master_key_probes()
        self.assertGreaterEqual(len(probes), 5)
        self.assertTrue(any(p["payload"] == ":" for p in probes))


if __name__ == "__main__":
    unittest.main()
