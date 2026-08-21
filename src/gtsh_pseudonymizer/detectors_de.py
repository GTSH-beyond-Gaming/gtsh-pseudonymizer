"""German detector metadata exposed for integrations and diagnostics."""

from __future__ import annotations

from .core import Pseudonymizer, Span


def detect(text: str, *, mask_number_clusters: bool = False) -> list[Span]:
    """Run the deterministic German detector layer without an NER backend."""
    result = Pseudonymizer(
        ner_backend=None,
        mask_number_clusters=mask_number_clusters,
    ).pseudonymize(text)
    return result.entities


__all__ = ["detect"]
