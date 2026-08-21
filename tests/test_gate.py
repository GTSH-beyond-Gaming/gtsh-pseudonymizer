import unittest

from gtsh_pseudonymizer import PrivacyGateError, Pseudonymizer, evaluate_gate, protect_fields


class PrivacyGateTests(unittest.TestCase):
    def test_missing_ner_is_fail_closed(self):
        result = Pseudonymizer().pseudonymize("Hallo")
        decision = evaluate_gate(result, ner_available=False)
        self.assertFalse(decision.allowed)
        self.assertIn("NER_MISSING", decision.warning_types)

    def test_safe_wrapper_cannot_claim_ner_when_backend_is_missing(self):
        with self.assertRaises(PrivacyGateError) as caught:
            protect_fields(
                "Unkritischer Text",
                pseudonymizer=Pseudonymizer(),
                ner_available=True,
            )
        self.assertIn("no local NER backend", str(caught.exception))

    def test_residual_warning_is_fail_closed(self):
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        result = engine.pseudonymize("Termin bei Beispieltechnik GmbH.")
        decision = evaluate_gate(result, ner_available=True)
        self.assertFalse(decision.allowed)

    def test_fields_share_one_mapping_and_restore(self):
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        protected = protect_fields(
            "Kontakt: ada@example.test",
            "Bitte ada@example.test antworten",
            pseudonymizer=engine,
        )
        self.assertEqual(protected.values[0].count("[EMAIL_1]"), 1)
        self.assertEqual(protected.values[1].count("[EMAIL_1]"), 1)
        self.assertEqual(protected.restore("Antwort an [EMAIL_1]"), "Antwort an ada@example.test")

    def test_boundary_collision_is_avoided(self):
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        protected = protect_fields(
            "Text [[[PRIVACY_BOUNDARY_FIXED_1]]]",
            "zweites Feld",
            pseudonymizer=engine,
        )
        self.assertEqual(len(protected.values), 2)

    def test_gate_exception_contains_no_original_value(self):
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        with self.assertRaises(PrivacyGateError) as caught:
            protect_fields(
                "Termin bei Beispieltechnik GmbH.",
                pseudonymizer=engine,
            )
        self.assertNotIn("Beispieltechnik", str(caught.exception))

    def test_explicit_ner_unavailable_flag_still_fails_closed(self):
        with self.assertRaises(PrivacyGateError):
            protect_fields(
                "Hallo",
                pseudonymizer=Pseudonymizer(ner_backend=lambda _text: []),
                ner_available=False,
            )

    def test_non_string_field_is_rejected(self):
        with self.assertRaises(TypeError):
            protect_fields(
                "Hallo",
                None,  # type: ignore[arg-type]
                pseudonymizer=Pseudonymizer(ner_backend=lambda _text: []),
            )


if __name__ == "__main__":
    unittest.main()
