import json
import tempfile
import unittest
from pathlib import Path

from bedir.eval import audit_debates, evaluate_predictions, validate_bench
from training.compute_model import HARDWARE, build_job_table, estimate_memory_gb, phase1_sensitivity
from training.scaling.benchmark_harness import analyse_scaling, render_markdown

ROOT = Path(__file__).resolve().parent.parent


class ComputeModelTests(unittest.TestCase):
    def test_phase1_total_matches_published_110k_budget(self):
        table = build_job_table("phase1", "H100")
        self.assertEqual(table["total_h100_hours"], 110400.0)
        self.assertEqual(len(table["rows"]), 5)
        stages = [r["stage"] for r in table["rows"]]
        self.assertEqual(stages, ["P1-CPT", "P1-Mizan", "P1-Bedir", "P1-Tutulma", "P1-Eval"])

    def test_hardware_conversion_across_cuda_and_rocm(self):
        h100 = build_job_table("phase1", "H100")["total_hw_gpu_hours"]
        mi355x = build_job_table("phase1", "MI355X")["total_hw_gpu_hours"]
        mi250x = build_job_table("phase1", "MI250X")["total_hw_gpu_hours"]
        self.assertLess(mi355x, h100)
        self.assertGreater(mi250x, h100)
        self.assertEqual(set(HARDWARE.keys()), {"H100", "H200", "MI250X", "MI300X", "MI355X"})

    def test_memory_and_sensitivity_calculations(self):
        mem = estimate_memory_gb(32.0, fsdp_world_size=8)
        self.assertGreater(mem["bf16_weights_gb"], 55.0)
        self.assertGreater(mem["est_train_gb_per_gpu"], 60.0)
        sens = phase1_sensitivity()
        self.assertEqual(len(sens), 4)
        self.assertTrue(all(item["total_h100_hours"] > 0 for item in sens))


class ScalingHarnessTests(unittest.TestCase):
    def test_schema_example_parses_and_renders_markdown(self):
        raw = json.loads((ROOT / "training" / "scaling" / "schema_example.json").read_text("utf-8"))
        report = analyse_scaling(raw)
        self.assertTrue(report["synthetic_schema_example"])
        self.assertEqual(len(report["rows"]), 4)
        self.assertEqual(report["rows"][0]["scaling_efficiency_pct"], 100.0)
        self.assertGreater(report["rows"][-1]["scaling_efficiency_pct"], 80.0)
        md = render_markdown(report)
        self.assertIn("Schema example only", md)
        self.assertIn("| 32 |", md)


class EvalHarnessAndConfigsTests(unittest.TestCase):
    def test_recorded_debates_audit_passes_zero_failures(self):
        res = audit_debates(ROOT / "examples" / "debates")
        self.assertTrue(res["passed"])
        self.assertEqual(res["quote_audit"]["debates"], 7)
        self.assertEqual(res["quote_audit"]["total_claims"], 56)
        self.assertEqual(res["quote_audit"]["local_failures"], 0)
        self.assertEqual(res["quote_audit"]["quote_validity_rate"], 1.0)

    def test_pilot_benchmark_json_matches_recorded_debates(self):
        audit = audit_debates(ROOT / "examples" / "debates")["quote_audit"]
        bench = json.loads((ROOT / "benchmarks" / "pilot_benchmark.json").read_text("utf-8"))
        summary = bench["summary"]
        self.assertEqual(summary["debates"], audit["debates"])
        self.assertEqual(summary["total_claims"], audit["total_claims"])
        self.assertEqual(summary["server_verified"], audit["server_verified"])
        self.assertEqual(summary["local_sha256_failures"], 0)
        self.assertEqual(summary["cross_lingual_agreement_rate"], 1.0)
        self.assertEqual(len(bench["cases"]), 7)

    def test_mizan_bench_sample_validates_cleanly(self):
        res = validate_bench(ROOT / "benchmarks" / "mizan_bench_sample.jsonl")
        self.assertTrue(res["passed"], msg=str(res["errors"]))
        self.assertEqual(res["rows"], 9)
        self.assertEqual(res["case_groups"], 3)
        self.assertEqual(res["languages"], {"de": 3, "en": 3, "tr": 3})

    def test_evaluate_predictions_jsonl(self):
        rows = [
            {"p_for": 0.85, "outcome": 1, "p_swapped": 0.15, "p_perturbed": 0.80, "p_attacked": 0.78, "case_group_id": "g1"},
            {"p_for": 0.78, "outcome": 1, "p_swapped": 0.22, "p_perturbed": 0.74, "p_attacked": 0.70, "case_group_id": "g1"},
            {"p_for": 0.18, "outcome": 0, "p_swapped": 0.82, "p_perturbed": 0.22, "p_attacked": 0.25, "case_group_id": "g2"},
            {"p_for": 0.22, "outcome": 0, "p_swapped": 0.78, "p_perturbed": 0.28, "p_attacked": 0.60, "case_group_id": "g2"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "preds.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
            rep = evaluate_predictions(p)
        self.assertEqual(rep["outcome_accuracy"], 1.0)
        self.assertEqual(rep["position_flip_rate"], 0.0)
        self.assertAlmostEqual(rep["hack_rate"], 0.25, places=5)
        self.assertEqual(rep["cross_lingual"]["agreement_rate"], 1.0)

    def test_verl_configs_reference_real_reward_hook_and_safety_rules(self):
        cfg_dir = ROOT / "training" / "configs"
        files = sorted(cfg_dir.glob("*.yaml"))
        self.assertEqual(len(files), 4)
        for path in files:
            text = path.read_text("utf-8")
            self.assertIn("path: bedir/rewards.py", text, msg=path.name)
            self.assertIn("name: verl_compute_score", text, msg=path.name)
        tutulma = (cfg_dir / "phase1_tutulma_redteam.yaml").read_text("utf-8")
        self.assertIn("publish_weights: false", tutulma)
        mizan = (cfg_dir / "phase0_mizan_8b_grpo.yaml").read_text("utf-8")
        self.assertIn("norm_adv_by_std_in_grpo: false", mizan)


if __name__ == "__main__":
    unittest.main()
