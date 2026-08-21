import unittest

from gtsh_pseudonymizer import Pseudonymizer, Span


class CorePseudonymizerTests(unittest.TestCase):
    def test_structured_identifiers_roundtrip_without_ner(self):
        source = (
            "Kundennummer: KD-847291\n"
            "USt-IdNr.: DE123456789\n"
            "IPv4: 203.0.113.42\n"
            "IPv6: 2001:db8::1\n"
            "MAC: 00:1A:2B:3C:4D:5E\n"
            "Geraete-ID: 550e8400-e29b-41d4-a716-446655440000\n"
            "Benutzername: max.muster\n"
            "Handle: @max_muster\n"
            "Kontaktperson: Ada Beispiel"
        )
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        result = engine.pseudonymize(source)
        for original in (
            "KD-847291",
            "DE123456789",
            "203.0.113.42",
            "2001:db8::1",
            "00:1A:2B:3C:4D:5E",
            "550e8400-e29b-41d4-a716-446655440000",
            "max.muster",
            "@max_muster",
            "Ada Beispiel",
        ):
            self.assertNotIn(original, result.masked_text)
        self.assertFalse(result.residual_warnings)
        self.assertEqual(engine.depseudonymize(result.masked_text, result.mapping), source)

    def test_email_wins_over_handle_overlap(self):
        result = Pseudonymizer(ner_backend=lambda _text: []).pseudonymize(
            "Kontakt: max.mustermann@example.test"
        )
        self.assertIn("[EMAIL_1]", result.masked_text)
        self.assertNotIn("[BENUTZERNAME_", result.masked_text)

    def test_existing_placeholder_is_reserved(self):
        source = "Vorlage [EMAIL_1], Kontakt max@example.test"
        engine = Pseudonymizer(ner_backend=lambda _text: [])
        result = engine.pseudonymize(source)
        self.assertIn("Vorlage [EMAIL_1]", result.masked_text)
        self.assertIn("[EMAIL_2]", result.masked_text)
        self.assertEqual(engine.depseudonymize(result.masked_text, result.mapping), source)

    def test_ner_failure_is_a_residual_warning(self):
        def failing_ner(_text):
            raise RuntimeError("offline")

        result = Pseudonymizer(ner_backend=failing_ner).pseudonymize("Unkritischer Text")
        self.assertEqual(result.residual_warnings[0]["typ"], "NER_FEHLER")
        self.assertNotIn("fund", result.residual_warnings[0])
        self.assertNotIn("offline", str(result.residual_warnings))

    def test_invalid_third_party_ner_span_fails_closed_without_leaking_source(self):
        source = "Geheimwert Ada Beispiel"
        bad_spans = [Span(0, 3, "NAME", source, "ner")]
        result = Pseudonymizer(ner_backend=lambda _text: bad_spans).pseudonymize(source)

        self.assertEqual(result.residual_warnings, [{"typ": "NER_FEHLER", "pos": 0}])
        self.assertNotIn(source, str(result.residual_warnings))

    def test_ner_exception_text_is_never_copied_into_warnings(self):
        def leaking_ner(_text):
            raise RuntimeError("Failed while processing ada@example.test")

        result = Pseudonymizer(ner_backend=leaking_ner).pseudonymize("Hallo")
        self.assertNotIn("ada@example.test", str(result.residual_warnings))

    def test_residual_warning_reports_type_and_position_only(self):
        result = Pseudonymizer(ner_backend=lambda _text: []).pseudonymize(
            "Termin bei Beispieltechnik GmbH."
        )
        self.assertTrue(result.residual_warnings)
        self.assertTrue(all("fund" not in warning for warning in result.residual_warnings))

    def test_non_string_input_is_rejected(self):
        with self.assertRaises(TypeError):
            Pseudonymizer().pseudonymize(None)  # type: ignore[arg-type]

    def test_extra_terms_are_validated_and_support_non_word_prefixes(self):
        engine = Pseudonymizer(
            ner_backend=lambda _text: [],
            extra_terms=[("@intern", "BENUTZERNAME")],
        )
        result = engine.pseudonymize("Kontakt über @intern.")
        self.assertNotIn("@intern", result.masked_text)

        for bad_terms in ([('', 'NAME')], [('Ada', 'name')], ['Ada']):
            with self.subTest(bad_terms=bad_terms), self.assertRaises((TypeError, ValueError)):
                Pseudonymizer(extra_terms=bad_terms)  # type: ignore[arg-type]

    def test_strong_miss_hints_remain_blockable(self):
        result = Pseudonymizer(ner_backend=lambda _text: []).pseudonymize(
            "Termin mit Frau Alina Qwertz bei Beispieltechnik GmbH."
        )
        warning_types = {warning["typ"] for warning in result.residual_warnings}
        self.assertIn("Personenname mit Titel/Anrede", warning_types)
        self.assertIn("Firma mit Rechtsform", warning_types)


if __name__ == "__main__":
    unittest.main()
