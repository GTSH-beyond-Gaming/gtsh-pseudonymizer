"""Loopback-only Ollama adapter for cleartext named-entity recognition."""

from __future__ import annotations

import ipaddress
import json
import re
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

from ..core import Span


_MAX_NER_RESPONSE_BYTES = 1_000_000
_MAX_NER_ENTITIES = 1_000


def _validated_loopback_http_url(raw_url: str) -> str:
    """Accept only direct HTTP loopback URLs for cleartext NER."""
    raw_url = (raw_url or "").strip()
    try:
        parsed = urlsplit(raw_url)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama NER host is not a valid URL") from exc
    if parsed.scheme.lower() != "http":
        raise ValueError("Ollama NER host must use http on loopback")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Ollama NER host must be a direct loopback target")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Ollama NER host must not include a path, query, or fragment")
    host = parsed.hostname.lower()
    if host == "localhost":
        # Do not rely on a mutable hosts-file entry for the security boundary.
        canonical_host = "127.0.0.1"
    else:
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("Ollama NER host must resolve to loopback")
        except ValueError as exc:
            if "must resolve to loopback" in str(exc):
                raise
            raise ValueError("Ollama NER host must resolve to loopback") from exc
        canonical_host = f"[{host}]" if ":" in host else host
    netloc = canonical_host if parsed.port is None else f"{canonical_host}:{parsed.port}"
    return urlunsplit(("http", netloc, "", "", ""))


def spans_from_ner_json(content: str, text: str) -> list[Span]:
    """Parse strict NER JSON and locate every literal entity in the input."""
    if not isinstance(content, str) or not isinstance(text, str):
        raise TypeError("content and text must be strings")
    if len(content.encode("utf-8")) > _MAX_NER_RESPONSE_BYTES:
        raise ValueError("NER JSON response is too large")
    label_map = {
        "NAME": "NAME",
        "PER": "NAME",
        "PERSON": "NAME",
        "ORT": "ORT",
        "LOC": "ORT",
        "GPE": "ORT",
        "FIRMA": "FIRMA",
        "ORG": "FIRMA",
    }
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("NER lieferte kein gueltiges JSON") from exc
    if isinstance(obj, dict):
        if not isinstance(obj.get("entities"), list):
            raise ValueError("NER JSON does not contain an entities list")
        entities = obj["entities"]
    elif isinstance(obj, list):
        entities = obj
    else:
        raise ValueError("NER JSON has an invalid shape")
    if len(entities) > _MAX_NER_ENTITIES:
        raise ValueError("NER JSON contains too many entities")

    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        surface = str(entity.get("text") or "").strip()
        label = label_map.get(str(entity.get("label") or "").upper())
        if not surface or not label:
            continue
        for match in re.finditer(re.escape(surface), text):
            key = (match.start(), match.end(), label)
            if key in seen:
                continue
            seen.add(key)
            spans.append(Span(match.start(), match.end(), label, surface, "ner", 0.9))
    return spans


# Backwards-compatible private name used by the original mail-assistant tests.
_spans_from_ner_json = spans_from_ner_json


def make_ollama_ner(
    model: str = "mistral",
    host: str = "http://127.0.0.1:11434",
    timeout: float = 60,
) -> Callable[[str], list[Span]]:
    """Return a proxy-free, loopback-only local Ollama NER callable."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Ollama NER model must not be empty")
    if timeout <= 0:
        raise ValueError("Ollama NER timeout must be positive")
    base_url = _validated_loopback_http_url(host)
    system = (
        "Du bist ein praeziser NER-Extraktor. Gib AUSSCHLIESSLICH JSON zurueck im "
        "Format {\"entities\":[{\"text\":\"...\",\"label\":\"NAME|ORT|FIRMA\"}]}. "
        "NAME = Personennamen, ORT = Orte/Staedte/Regionen, FIRMA = Organisationen. "
        "Nur Entitaeten, die woertlich im Text stehen. Technische Marker der Form "
        "[[[PRIVACY_BOUNDARY_...]]] sind niemals Entitaeten. Keine Erklaerung."
    )
    direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _ner(text: str) -> list[Span]:
        payload = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with direct_opener.open(request, timeout=timeout) as response:
            raw_response = response.read(_MAX_NER_RESPONSE_BYTES + 1)
            if len(raw_response) > _MAX_NER_RESPONSE_BYTES:
                raise ValueError("Ollama NER response is too large")
            data = json.loads(raw_response.decode("utf-8"))
        return spans_from_ner_json(data.get("message", {}).get("content", "{}"), text)

    return _ner
