"""Optional adapters for local entity-recognition backends."""

from .ollama import make_ollama_ner, spans_from_ner_json

__all__ = ["make_ollama_ner", "spans_from_ner_json"]
