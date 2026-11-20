import unittest

from bedir.verify import check, clean_quote, locate, sha256, slice_utf16

DOC = (
    "Clause 7.2. If the Landlord fails to complete such repairs within thirty (30) days of the notice, "
    "the Tenant may terminate this Lease without penalty.\n\n"
    "The manager replies the same day: \u201cReceived, we will send a roofer.\u201d"
)


class LocateTests(unittest.TestCase):
    def test_exact_quote_is_verified_and_hashed(self):
        m = locate(DOC, "the Tenant may terminate this Lease without penalty")
        self.assertTrue(m.verified)
        self.assertEqual(m.quote, "the Tenant may terminate this Lease without penalty")
        self.assertEqual(m.sha256, sha256(m.quote))
        self.assertEqual(slice_utf16(DOC, m.start, m.end), m.quote)

    def test_whitespace_case_and_quote_marks_are_normalised(self):
        m = locate(DOC, '"received, WE will   send a roofer."')
        self.assertTrue(m.verified)
        self.assertEqual(m.quote, "Received, we will send a roofer.")

    def test_quote_across_a_line_break(self):
        m = locate(DOC, "without penalty. The manager replies")
        self.assertTrue(m.verified)
        self.assertIn("\n\n", m.quote)

    def test_paraphrase_is_rejected(self):
        self.assertFalse(locate(DOC, "the Tenant can end the Lease without a penalty").verified)

    def test_short_quote_is_rejected(self):
        self.assertFalse(locate(DOC, "Clause 7.2").verified)

    def test_ellipsis_and_wrapping_marks_are_stripped(self):
        self.assertEqual(clean_quote("\u201e\u2026within thirty (30) days\u201c"), "within thirty (30) days")

    def test_turkish_dotted_capital_i(self):
        doc = "İadeli taahhütlü bir mektup dosyada bulunmuyor."
        self.assertTrue(locate(doc, "iadeli taahhütlü bir mektup").verified)

    def test_offsets_are_utf16_like_javascript(self):
        doc = "Note 🌕 first. The roof leaks again during rain on 2 July."
        m = locate(doc, "The roof leaks again during rain")
        self.assertTrue(m.verified)
        self.assertEqual(m.start, doc.index("The roof") + 1)  # the emoji is two UTF-16 units
        self.assertEqual(slice_utf16(doc, m.start, m.end), m.quote)


class CheckTests(unittest.TestCase):
    def claim(self, quote):
        m = locate(DOC, quote)
        return {"text": "claim", "quote": m.quote, "verified": m.verified, "start": m.start, "end": m.end, "sha256": m.sha256}

    def test_honest_claim_passes(self):
        ok, _ = check(DOC, self.claim("fails to complete such repairs within thirty (30) days"))
        self.assertTrue(ok)

    def test_tampered_hash_fails(self):
        c = self.claim("fails to complete such repairs within thirty (30) days")
        c["sha256"] = "0" * 64
        self.assertFalse(check(DOC, c)[0])

    def test_shifted_offsets_fail(self):
        c = self.claim("fails to complete such repairs within thirty (30) days")
        c["start"] += 1
        self.assertFalse(check(DOC, c)[0])

    def test_wrongly_eclipsed_quote_is_flagged(self):
        c = {"quote": "the Tenant may terminate this Lease", "verified": False}
        self.assertFalse(check(DOC, c)[0])

    def test_rightly_eclipsed_quote_passes(self):
        c = {"quote": "the Landlord promised a new roof", "verified": False}
        self.assertTrue(check(DOC, c)[0])


if __name__ == "__main__":
    unittest.main()
