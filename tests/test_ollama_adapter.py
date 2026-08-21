import json
import unittest
from unittest.mock import Mock, patch

from gtsh_pseudonymizer import make_ollama_ner, spans_from_ner_json
from gtsh_pseudonymizer.adapters import ollama


class OllamaAdapterTests(unittest.TestCase):
    def test_remote_hosts_are_rejected_before_use(self):
        for url in (
            "https://example.invalid:11434",
            "http://192.0.2.10:11434",
            "http://user:pass@127.0.0.1:11434",
            "http://127.0.0.1:11434/api",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                make_ollama_ner(host=url)

    def test_loopback_hosts_are_accepted(self):
        for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
            with self.subTest(url=url):
                self.assertTrue(callable(make_ollama_ner(host=url)))

    def test_empty_model_and_non_positive_timeout_are_rejected(self):
        with self.assertRaises(ValueError):
            make_ollama_ner(model="")
        with self.assertRaises(ValueError):
            make_ollama_ner(timeout=0)

    def test_strict_json_parsing_and_offsets(self):
        text = "Ada Beispiel besucht Teststadt. Ada Beispiel antwortet."
        content = json.dumps(
            {"entities": [{"text": "Ada Beispiel", "label": "PERSON"}, {"text": "Teststadt", "label": "GPE"}]}
        )
        spans = spans_from_ner_json(content, text)
        self.assertEqual([span.label for span in spans], ["NAME", "NAME", "ORT"])

    def test_duplicate_entities_are_deduplicated(self):
        text = "Ada Beispiel antwortet."
        content = json.dumps(
            {"entities": [
                {"text": "Ada Beispiel", "label": "PERSON"},
                {"text": "Ada Beispiel", "label": "NAME"},
            ]}
        )
        self.assertEqual(len(spans_from_ner_json(content, text)), 1)

    def test_oversized_entity_list_is_rejected(self):
        content = json.dumps(
            {"entities": [{"text": "Ada", "label": "NAME"}] * 1001}
        )
        with self.assertRaisesRegex(ValueError, "too many"):
            spans_from_ner_json(content, "Ada")

    def test_localhost_is_canonicalized_before_request(self):
        response = Mock()
        response.read.return_value = json.dumps(
            {"message": {"content": '{"entities":[]}'}}
        ).encode("utf-8")
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = response
        with patch.object(ollama.urllib.request, "build_opener", return_value=opener):
            ner = make_ollama_ner(host="http://localhost:11434")
            ner("Hallo")
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")

    def test_invalid_json_is_not_an_empty_success(self):
        with self.assertRaises(ValueError):
            spans_from_ner_json("not-json", "Text")


if __name__ == "__main__":
    unittest.main()
