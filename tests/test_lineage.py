import unittest

from bedir.lineage import (
    check_source_policy,
    is_valid_tckn,
    manifest_summary,
    redact_pii,
    scan_pii,
    verify_temporal_holdout,
)


class LineagePolicyTests(unittest.TestCase):
    def test_frontier_api_blocked_in_train_allowed_in_eval(self):
        src = {
            "id": "frontier-oversight-01",
            "license": "Frontier-API-Output",
            "provider": "anthropic-api",
            "pii_redacted": True,
        }
        self.assertFalse(check_source_policy(src, split="train").allowed)
        self.assertFalse(check_source_policy(src, split="val").allowed)
        self.assertTrue(check_source_policy(src, split="eval").allowed)
        self.assertTrue(check_source_policy(src, split="baseline").allowed)

    def test_echr_eval_only_blocked_in_train(self):
        src = {"id": "hudoc-tr-01", "license": "ECHR-Eval-Only", "pii_redacted": True}
        self.assertFalse(check_source_policy(src, split="train").allowed)
        self.assertTrue(check_source_policy(src, split="eval").allowed)

    def test_unmasked_public_decision_blocked_until_redacted(self):
        unmasked = {"id": "yargitay-batch", "license": "Statutory-Official-Decision", "pii_redacted": False}
        masked = {"id": "yargitay-batch", "license": "Statutory-Official-Decision", "pii_redacted": True}
        self.assertFalse(check_source_policy(unmasked, split="train").allowed)
        self.assertTrue(check_source_policy(masked, split="train").allowed)

    def test_manifest_summary_reports_rejections(self):
        sources = [
            {"id": "openreview", "license": "CC-BY-4.0", "pii_redacted": True},
            {"id": "openai-distill-attempt", "license": "Apache-2.0", "provider": "openai-api", "pii_redacted": True},
        ]
        summary = manifest_summary(sources, split="train")
        self.assertFalse(summary["all_approved"])
        self.assertEqual(summary["approved"], 1)
        self.assertEqual(summary["rejected"], 1)


class PIIRedactionTests(unittest.TestCase):
    def test_tckn_checksum_validation(self):
        # 10000000146 is a standard synthetic checksum-valid TCKN example
        self.assertTrue(is_valid_tckn("10000000146"))
        self.assertFalse(is_valid_tckn("10000000147"))
        self.assertFalse(is_valid_tckn("01234567890"))

    def test_scan_and_redact_pii(self):
        raw = (
            "Applicant ID 10000000146, IBAN TR330006100519786457841326, "
            "contact test.person@example.org or +90 532 123 45 67 from 192.168.10.22."
        )
        redacted, findings = redact_pii(raw)
        kinds = {f.kind for f in findings}
        self.assertEqual(kinds, {"TCKN", "IBAN", "EMAIL", "PHONE", "IPV4"})
        self.assertNotIn("10000000146", redacted)
        self.assertNotIn("TR330006100519786457841326", redacted)
        self.assertNotIn("test.person@example.org", redacted)
        self.assertEqual(scan_pii(redacted), [])


class TemporalHoldoutTests(unittest.TestCase):
    def test_clean_temporal_split_passes(self):
        train = [
            {"decision_date": "2025-06-10", "document": "The appellate chamber reviewed the commercial lease termination clause carefully and affirmed."},
        ]
        test = [
            {"decision_date": "2026-03-14", "document": "The administrative panel evaluated an environmental permit dispute regarding coastal zoning regulations."},
        ]
        audit = verify_temporal_holdout(train, test, cutoff_date="2025-12-31")
        self.assertTrue(audit.passed)
        self.assertEqual(audit.temporal_violations, 0)
        self.assertEqual(audit.exact_hash_overlaps, 0)

    def test_date_leakage_and_duplicate_doc_fail(self):
        doc = "Repeated court decision text that appears verbatim in both train and test splits without separation."
        train = [{"decision_date": "2025-05-01", "document": doc}]
        test = [{"decision_date": "2025-08-01", "document": doc}]
        audit = verify_temporal_holdout(train, test, cutoff_date="2025-12-31")
        self.assertFalse(audit.passed)
        self.assertEqual(audit.temporal_violations, 1)
        self.assertEqual(audit.exact_hash_overlaps, 1)
        self.assertEqual(audit.max_ngram_overlap, 1.0)


if __name__ == "__main__":
    unittest.main()
