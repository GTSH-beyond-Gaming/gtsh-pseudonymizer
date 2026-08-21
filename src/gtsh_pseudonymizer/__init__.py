"""Public API for gtsh-pseudonymizer."""

from .adapters.ollama import _spans_from_ner_json, make_ollama_ner, spans_from_ner_json
from .core import PseudoResult, Pseudonymizer, Span
from .gate import (
    GateDecision,
    PrivacyGateError,
    ProtectedFields,
    evaluate_gate,
    protect_fields,
    require_safe,
)

__all__ = [
    "GateDecision",
    "PrivacyGateError",
    "ProtectedFields",
    "PseudoResult",
    "Pseudonymizer",
    "Span",
    "evaluate_gate",
    "make_ollama_ner",
    "protect_fields",
    "require_safe",
    "spans_from_ner_json",
]

__version__ = "0.2.0"
