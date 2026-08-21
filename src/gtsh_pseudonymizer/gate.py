"""Fail-closed facade for protecting one or more text fields."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from .core import PseudoResult, Pseudonymizer


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""
    warning_types: tuple[str, ...] = ()


@dataclass
class ProtectedFields:
    values: tuple[str, ...]
    result: PseudoResult
    pseudonymizer: Pseudonymizer

    def restore(self, text: str) -> str:
        return self.pseudonymizer.depseudonymize(text, self.result.mapping)


class PrivacyGateError(RuntimeError):
    """The text could not be proven safe for the requested boundary."""


def evaluate_gate(result: PseudoResult, *, ner_available: bool) -> GateDecision:
    if not ner_available:
        return GateDecision(False, "no local NER backend is available", ("NER_MISSING",))
    if result.residual_warnings:
        warning_types = tuple(
            sorted({str(warning.get("typ") or "RESIDUAL_RISK") for warning in result.residual_warnings})
        )
        return GateDecision(
            False,
            "residual risk scan found possible PII or an NER error: " + ", ".join(warning_types),
            warning_types,
        )
    return GateDecision(True)


def require_safe(result: PseudoResult, *, ner_available: bool) -> None:
    decision = evaluate_gate(result, ner_available=ner_available)
    if not decision.allowed:
        raise PrivacyGateError(decision.reason)


def _bundle(values: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if not values:
        raise ValueError("at least one field is required")
    if len(values) == 1:
        return values[0], ()
    alphabet = "GHIJKLMNOPQRSTUV"
    for _attempt in range(10):
        nonce = "".join(alphabet[int(char, 16)] for char in secrets.token_hex(16))
        separators = tuple(
            f"\n\n[[[PRIVACY_BOUNDARY_{nonce}_{index}]]]\n\n"
            for index in range(1, len(values))
        )
        if not any(separator in value for separator in separators for value in values):
            chunks = [values[0]]
            for separator, value in zip(separators, values[1:]):
                chunks.extend((separator, value))
            return "".join(chunks), separators
    raise RuntimeError("could not create collision-free privacy boundaries")


def _split(masked_text: str, separators: tuple[str, ...]) -> tuple[str, ...] | None:
    parts: list[str] = []
    remainder = masked_text
    for separator in separators:
        before, found, remainder = remainder.partition(separator)
        if not found:
            return None
        parts.append(before)
    parts.append(remainder)
    return tuple(parts)


def protect_fields(
    *values: str,
    pseudonymizer: Pseudonymizer,
    ner_available: bool | None = None,
) -> ProtectedFields:
    """Pseudonymize fields with one mapping and raise unless the gate is safe."""
    if any(not isinstance(value, str) for value in values):
        raise TypeError("all protected fields must be strings")
    normalized = tuple(value or "" for value in values)
    source, separators = _bundle(normalized)
    result = pseudonymizer.pseudonymize(source)
    backend_present = pseudonymizer.ner_backend is not None
    effective_ner_available = backend_present and ner_available is not False
    require_safe(result, ner_available=effective_ner_available)
    masked_values = _split(result.masked_text, separators)
    if masked_values is None or len(masked_values) != len(normalized):
        raise PrivacyGateError("protected fields could not be split safely")
    return ProtectedFields(masked_values, result, pseudonymizer)
