"""
Reversible, deutschsprachige Pseudonymisierung fuer lokale KI-Workflows.

Zweck
-----
Erkennt personenbezogene Daten (PII) in einem Eingabetext, ersetzt sie durch
stabile, *umkehrbare* Platzhalter (z. B. [NAME_1]) und liefert die Zuordnung
zurueck. Nach der Modell-Antwort werden die Platzhalter wieder in die
Originalwerte uebersetzt (de-pseudonymisieren), bevor ein Mensch prueft.

Wichtige Eigenschaften
----------------------
- Pseudonymisierung (Art. 4 Nr. 5 DSGVO), NICHT Anonymisierung: umkehrbar,
  Daten bleiben personenbezogen. Genau so wird es benannt.
- Mehrere Entitaetstypen: NAME, ORT, FIRMA (via NER) + EMAIL, TEL, IBAN,
  ADRESSE, PLZ_ORT, DATUM, URL, IP, MAC, UUID, UST_ID, BENUTZERNAME,
  KENNUNG und optional BETRAG (Regex/Standardbibliothek, deps-frei).
- Stabil: gleiche Person -> gleicher Platzhalter (innerhalb eines Vorgangs).
- NER-Backend austauschbar: Der Kern akzeptiert jede Funktion ``text -> Spans``.
  Ein abgesicherter lokaler Ollama-Adapter liegt in ``adapters.ollama``.
- Restrisiko-Scan: zweiter Durchgang, der *strukturierte* PII und starke
  Personen-/Firmenhinweise findet, die durchgerutscht sind — das
  Sicherheitsnetz, bevor irgendetwas in die Cloud geht.
- Blockliste (extra_terms): kuratierte Begriffe, die IMMER maskiert werden —
  die praktische Antwort auf NER-Luecken (z. B. bekannte Mandanten, Orte).
- Nur Python-Standardbibliothek. Laeuft offline auf jeder Maschine.

Ehrliche Grenze
---------------
NER erkennt Namen/Orte nie zu 100 %. Die Recall-Qualitaet haengt am NER-Modell.
Produktive Empfehlung: gutes NER (Presidio + spaCy de) + Blockliste +
menschliche Pruefung der Zuordnung. Der Restrisiko-Scan deckt verlaesslich
strukturierte PII ab, nicht jeden Eigennamen.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Datentypen
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Span:
    """Ein erkanntes PII-Vorkommen im Originaltext."""
    start: int
    end: int
    label: str          # interne Kategorie, z. B. "NAME", "EMAIL"
    text: str
    source: str         # "regex" | "ner" | "blockliste"
    score: float = 1.0


@dataclass
class PseudoResult:
    masked_text: str
    mapping: dict[str, str]
    entities: list[Span]
    residual_warnings: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Regex-Schicht (praezise, deps-frei) — Reihenfolge = Prioritaet bei Ueberlappung
# -----------------------------------------------------------------------------

_MONTHS = ("Januar|Februar|M\u00e4rz|April|Mai|Juni|Juli|August|"
           "September|Oktober|November|Dezember")

_UST_ID_RX = re.compile(r"\bDE(?:[ -]?\d{3}){3}\b", re.IGNORECASE)
_UUID_RX = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_MAC_RX = re.compile(
    r"\b(?:(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}|(?:[0-9A-F]{4}\.){2}[0-9A-F]{4})\b",
    re.IGNORECASE,
)
_HANDLE_RX = re.compile(r"(?<![\w@])@[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}\b")
_EMAIL_RX = re.compile(
    r"(?<![\w@])[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w-]+\.)+[\w-]{2,}(?![\w@])",
    re.UNICODE,
)
_IPV4_RX = re.compile(
    r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
)

# Kandidaten werden anschliessend mit ipaddress validiert. Dadurch werden auch
# komprimierte IPv6-Adressen erkannt, ohne Datums-/Uhrzeitfragmente zu maskieren.
_IP_CANDIDATE_RX = re.compile(
    r"(?<![0-9A-Za-z_.:])(?P<value>[0-9A-Fa-f:.]{2,})(?![0-9A-Za-z_.:])"
)

# Bei beschrifteten IDs wird nur der Wert ersetzt; das fachliche Label bleibt
# fuer das Sprachmodell sichtbar. Eine Ziffer im Wert reduziert Fehlalarme bei
# Formulierungen wie 'Bestellnummer erhalten'.
_LABELED_IDENTIFIER_RX = re.compile(
    r"\b(?:"
    r"Kunden(?:nummer|[- ]?Nr\.?)|"
    r"Auftrag(?:snummer|\s+Nr\.?)|"
    r"Bestell(?:nummer|ung\s+Nr\.?)|"
    r"Rechnung(?:snummer|\s+Nr\.?)|"
    r"Beleg(?:nummer|\s+Nr\.?)|"
    r"Vorgang(?:snummer|\s+Nr\.?)|"
    r"Ticket(?:nummer|[- ]?ID)|"
    r"Aktenzeichen|Personalnummer|Versichertennummer|Steuernummer|"
    r"Kennzeichen|Ger(?:\u00e4|ae)te[- ]?ID|Kennung"
    r")\s*(?:[:#=]\s*)?"
    r"(?P<value>(?=[A-Z0-9._/-]{4,}\b)(?=[A-Z0-9._/-]*\d)"
    r"[A-Z0-9](?:[A-Z0-9._/-]*[A-Z0-9]))",
    re.IGNORECASE,
)
_LABELED_ALPHA_IDENTIFIER_RX = re.compile(
    r"\b(?:"
    r"Kunden(?:nummer|[- ]?Nr\.?)|"
    r"Auftrag(?:snummer|\s+Nr\.?)|"
    r"Bestell(?:nummer|ung\s+Nr\.?)|"
    r"Rechnung(?:snummer|\s+Nr\.?)|"
    r"Beleg(?:nummer|\s+Nr\.?)|"
    r"Vorgang(?:snummer|\s+Nr\.?)|"
    r"Ticket(?:nummer|[- ]?ID)|"
    r"Aktenzeichen|Personalnummer|Versichertennummer|Steuernummer|"
    r"Kennzeichen|Ger(?:\u00e4|ae)te[- ]?ID|Kennung"
    r")\s*[:#=]\s*"
    r"(?P<value>[A-Z0-9](?:[A-Z0-9._/-]{2,62}[A-Z0-9]))\b",
    re.IGNORECASE,
)
_LABELED_USERNAME_RX = re.compile(
    r"\b(?:Benutzername|Nutzerkonto|User(?:name|[- ]?ID)|Login)"
    r"\s*(?:[:#=]\s*|(?:ist|lautet)\s+)"
    r"(?P<value>[A-Za-z0-9][A-Za-z0-9._@-]{2,63})\b",
    re.IGNORECASE,
)
_LABELED_PERSON_RX = re.compile(
    r"\b(?:Ansprechpartner(?:in)?|Kontaktperson|Bearbeiter(?:in)?|Kundenname|Name)"
    r"\s*(?:[:#=]\s*|(?:ist|lautet)\s+)"
    r"(?P<value>(?:(?:Dr|Prof)\.?\s+)?"
    r"[A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+"
    r"(?:['\u2019][A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+)?"
    r"(?:[ -](?:(?:von|van|der|den|de|del|da|di|la|le|zu|zur|zum)|"
    r"[A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+"
    r"(?:['\u2019][A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+)?)){0,4})\b"
)
_LABELED_ADDRESS_RX = re.compile(
    r"\b(?:Adresse|Anschrift|Wohnadresse|Firmensitz)\s*[:#=]\s*"
    r"(?P<value>[^\r\n;]{4,160})",
    re.IGNORECASE,
)

# Diese Muster werden nicht blind ersetzt, sondern dienen als zweite,
# fail-closed Kontrolle fuer plausible NER-Auslassungen.
_TITLED_PERSON_HINT_RX = re.compile(
    r"\b(?:Herrn?|Frau|Dr\.?|Prof\.?)\s+"
    r"[A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+"
    r"(?:[ -][A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+){0,2}\b"
)
_LEGAL_ENTITY_HINT_RX = re.compile(
    r"\b(?:[A-Z\u00c4\u00d6\u00dc][\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df&.'-]*\s+){1,5}"
    r"(?:GmbH(?:\s*&\s*Co\.?\s*KG)?|AG|UG(?:\s*\(haftungsbeschr\u00e4nkt\))?|"
    r"KG|OHG|GbR|e\.?\s*V\.?)"
    r"(?=$|[\s,.;:()])"
)

# Jede Regel: (label, compiled_regex). Strukturierte/lange Treffer zuerst.
_REGEX_RULES: list = [
    # IBAN: getrennte 4er-Gruppen (mit Leerzeichen) ODER kompakt am Stueck.
    # Erzwungenes Leerzeichen zwischen Gruppen -> frisst kein nachfolgendes Wort ("... abgebucht").
    ("IBAN", re.compile(
        r"\b[A-Z]{2}\d{2}(?:[ ][A-Za-z0-9]{2,4})+\b"
        r"|\b[A-Z]{2}\d{2}[A-Za-z0-9]{11,30}\b")),
    ("EMAIL", _EMAIL_RX),
    ("URL", re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)),
    ("UST_ID", _UST_ID_RX),
    ("UUID", _UUID_RX),
    ("MAC", _MAC_RX),
    ("BENUTZERNAME", _HANDLE_RX),
    ("IP", _IPV4_RX),
    # Strasse + Hausnummer (deutsche Strassen-Suffixe)
    ("ADRESSE", re.compile(
        r"\b[A-Z\u00c4\u00d6\u00dc][\wA-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df.\-]*"
        r"(?:stra\u00dfe|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer)\s+\d+\s*[a-z]?\b",
        re.IGNORECASE)),
    # Mehrwortige deutsche Strassennamen ohne Suffix, z. B. "Unter den Linden 77".
    ("ADRESSE", re.compile(
        r"\b(?:An der|An den|Auf der|Auf dem|Unter den|Am)\s+"
        r"[A-Z\u00c4\u00d6\u00dc][\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df.\-]*"
        r"(?:\s+[A-Z\u00c4\u00d6\u00dc][\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df.\-]*){0,2}"
        r"\s+\d+\s*[a-z]?\b")),
    # PLZ + Ort (5-stellige PLZ, dann Ortsname)
    ("PLZ_ORT", re.compile(
        r"\b\d{5}\s+[A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+"
        r"(?:[ \-/][A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+)*")),
    # Telefon (DE): muss mit +49 / 0049 / 0 beginnen, danach >= 6 Ziffern gesamt
    ("TEL", re.compile(
        r"(?:\+49|0049|0)(?:[\s\-/]?\(?\d{1,5}\)?)(?:[\s\-/]?\d{2,}){2,}")),
    # Datum: 12.05.2026  |  2026-05-12  |  12. Mai 2026
    ("DATUM", re.compile(
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\b\d{1,2}\.?\s*(?:" + _MONTHS + r")\s+\d{4}\b")),
]

# Optional, standardmaessig AUS: Geldbetraege sind i. d. R. keine PII und helfen
# dem Modell beim Entwurf. Auf Wunsch (maximale Schwaerzung) aktivierbar.
_MONEY_RULE = ("BETRAG", re.compile(
    r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?\s*(?:\u20ac|EUR|Euro)(?!\w)"))

# Restrisiko-Scan: was nach dem Maskieren NICHT mehr stehen darf (strukturiert).
# Mindestens fuenf tatsaechliche Ziffern, nur innerhalb derselben Zeile. Die alte
# Fassung zaehlte auch Leerzeichen/Zeilenumbrueche und verband dadurch z. B. eine
# Mengenangabe mit einem spaeteren Betrag zu einem falschen PII-Treffer.
_NUMBER_CLUSTER_RX = re.compile(
    r"(?<!\w)(?=(?:[ \t\-/]*\d){5})\d[\d \t\-/]*\d(?!\w)"
)
_RESIDUAL_RULES: list = [
    ("E-Mail-Rest", _EMAIL_RX),
    ("IBAN-Muster", re.compile(
        r"\b[A-Z]{2}\d{2}(?:[ ][A-Za-z0-9]{2,4})+\b"
        r"|\b[A-Z]{2}\d{2}[A-Za-z0-9]{11,30}\b")),
    ("USt-ID-Muster", _UST_ID_RX),
    ("UUID-Muster", _UUID_RX),
    ("MAC-Muster", _MAC_RX),
    ("Benutzername/Handle", _HANDLE_RX),
    ("Beschriftete Kennung", _LABELED_IDENTIFIER_RX),
    ("Beschriftete Kennung", _LABELED_ALPHA_IDENTIFIER_RX),
    ("Beschrifteter Benutzername", _LABELED_USERNAME_RX),
    ("Beschrifteter Personenname", _LABELED_PERSON_RX),
    ("Beschriftete Adresse", _LABELED_ADDRESS_RX),
    ("Personenname mit Titel/Anrede", _TITLED_PERSON_HINT_RX),
    ("Firma mit Rechtsform", _LEGAL_ENTITY_HINT_RX),
    ("Zahlencluster (>=5)", _NUMBER_CLUSTER_RX),
]

_VALUE_RULES = (
    ("KENNUNG", _LABELED_IDENTIFIER_RX),
    ("KENNUNG", _LABELED_ALPHA_IDENTIFIER_RX),
    ("BENUTZERNAME", _LABELED_USERNAME_RX),
    ("NAME", _LABELED_PERSON_RX),
    ("ADRESSE", _LABELED_ADDRESS_RX),
)


def _validated_ip_spans(text: str) -> list:
    """Validierte IPv4/IPv6-Spans aus breit gescannten Kandidaten."""
    spans: list = []
    for match in _IP_CANDIDATE_RX.finditer(text):
        raw_value = match.group("value")
        value = raw_value.rstrip(".,;!?")
        if not value:
            continue
        try:
            ipaddress.ip_address(value)
        except ValueError:
            continue
        spans.append(Span(
            match.start("value"),
            match.start("value") + len(value),
            "IP",
            value,
            "regex",
        ))
    return spans


# -----------------------------------------------------------------------------
# Namen-Heuristik (Anrede + Grussformel) — erkennt Personennamen OHNE NER.
# Ergaenzt die strukturierte Regex-Schicht und NER, ersetzt fuer externe
# Vertrauensgrenzen aber nicht das volle fail-closed Datenschutz-Tor.
# -----------------------------------------------------------------------------
_HEUR_NAME = r"((?:Dr\.\s+|Prof\.\s+)?[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2})"

_HEUR_NAME_STOP = {
    "Damen", "Herren", "Herr", "Frau", "Team", "Kundenservice", "Kundendienst",
    "Support", "Service", "Ihr", "Ihre", "Euer", "Wir", "Sie", "Mit", "Viele",
    "Beste", "Herzliche", "Freundliche", "Liebe", "Lieber", "Sehr", "Hallo", "Guten",
    "Per", "Betreff", "Anbei", "Vielen", "Danke", "Gerne", "Respektvolle", "Respektvollen",
}

_HEUR_SALUTATION = re.compile(
    r"(?i:sehr\s+geehrte[r]?|liebe[r]?|hallo|guten\s+(?:tag|morgen|abend))"
    r"\s+(?:(?i:herr|frau)\s+)?" + _HEUR_NAME
)

_HEUR_GRUSS = r"gr(?:ü|ue|u)(?:ß|ss)(?:e[n]?)?"
_HEUR_GREETING = re.compile(
    r"(?i:mit\s+freundlichen\s+" + _HEUR_GRUSS +
    r"|(?:viele|beste|herzliche|liebe|freundliche|respektvolle[n]?|mit\s+besten)\s+" + _HEUR_GRUSS +
    r"|" + _HEUR_GRUSS + r"|mfg|lg)"
)

_HEUR_NAME_LINE = re.compile(
    r"^\s*(?:(?:Dr|Prof)\.?\s+)?(?:(?i:herr|frau)\s+)?"
    r"(?P<name>[A-ZÄÖÜ][a-zäöüß]+(?:[ \-][A-ZÄÖÜ][a-zäöüß]+){0,2})\s*$"
)


def _heur_clean_name(cand):
    cand = (cand or "").strip()
    if not cand or cand.split()[0] in _HEUR_NAME_STOP:
        return None
    return cand


def _heur_salutation_names(text: str) -> list:
    """Namen aus der Anrede (Empfaenger)."""
    out: list = []
    for m in _HEUR_SALUTATION.finditer(text):
        nm = _heur_clean_name(m.group(1))
        if nm and nm not in out:
            out.append(nm)
    return out


def _heur_closing_names(text: str) -> list:
    """Name(n) aus Grussformel/Signatur (Absender). Robust gegen Smileys/Zusatzzeilen."""
    lines = text.splitlines()
    greet_idx = -1
    for i, ln in enumerate(lines):
        if _HEUR_GREETING.search(ln):
            greet_idx = i
    if greet_idx < 0:
        return []
    gline = lines[greet_idx]
    m = _HEUR_GREETING.search(gline)
    rest = gline[m.end():] if m else ""
    rest = re.sub(r"^[\s,;:!.()\-=^_*~]+", "", rest).strip()
    if rest:
        mm = _HEUR_NAME_LINE.match(rest)
        if mm:
            nm = _heur_clean_name(mm.group("name"))
            if nm:
                return [nm]
    for ln in lines[greet_idx + 1: greet_idx + 6]:
        if not ln.strip():
            continue
        mm = _HEUR_NAME_LINE.match(ln)
        if mm:
            nm = _heur_clean_name(mm.group("name"))
            if nm:
                return [nm]
    return []


def _heuristic_name_spans(text: str) -> list:
    """Anrede-/Gruss-Namen als Spans (alle Vorkommen). Greift OHNE NER."""
    names = set(_heur_salutation_names(text)) | set(_heur_closing_names(text))
    spans: list = []
    for name in names:
        for m in re.finditer(re.escape(name), text):
            spans.append(Span(m.start(), m.end(), "NAME", name, "heuristik", 0.85))
    return spans


# -----------------------------------------------------------------------------
# Kern: Pseudonymizer
# -----------------------------------------------------------------------------

class Pseudonymizer:
    def __init__(self,
                 ner_backend: Optional[Callable[[str], list]] = None,
                 enable_money: bool = False,
                 mask_number_clusters: bool = False,
                 extra_terms: Optional[Sequence[tuple[str, str]]] = None,
                 token_format: str = "[{label}_{n}]"):
        """
        ner_backend : Funktion text -> list[Span] (z. B. make_ollama_ner()).
        enable_money: Geldbetraege mit maskieren (Standard: nein).
        mask_number_clusters: Lange Rest-Zahlenfolgen vorsorglich als NUMMER
                              maskieren, statt sie nur im Restrisiko-Tor zu melden.
                              Gedacht fuer irreversible Remote-Kontexte.
        extra_terms : Blockliste [(begriff, label), ...], immer maskiert.
        token_format: Platzhalter-Muster. {label} und {n} werden ersetzt.
        """
        if "{label}" not in token_format or "{n}" not in token_format:
            raise ValueError("token_format muss {label} und {n} enthalten")
        if ner_backend is not None and not callable(ner_backend):
            raise TypeError("ner_backend must be callable")
        normalized_terms: list[tuple[str, str]] = []
        for item in extra_terms or ():
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise TypeError("extra_terms entries must be (term, label) pairs")
            term, label = item
            if not isinstance(term, str) or not term.strip():
                raise ValueError("extra_terms must not contain empty terms")
            if not isinstance(label, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", label):
                raise ValueError("extra_terms labels must be uppercase token labels")
            normalized_terms.append((term, label))
        self.ner_backend = ner_backend
        self.enable_money = enable_money
        self.mask_number_clusters = mask_number_clusters
        self.extra_terms = tuple(normalized_terms)
        self.token_format = token_format
    # ---- oeffentliche API ----

    def pseudonymize(self, text: str) -> PseudoResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        spans, ner_failed = self._collect_spans(text)
        spans = self._resolve_overlaps(spans)
        masked, mapping = self._apply(text, spans)
        residual = self.scan_residuals(masked, mapping)
        if ner_failed:
            # Never echo exception text: third-party NER implementations may put
            # source PII in their error messages.
            residual.append({"typ": "NER_FEHLER", "pos": 0})
        return PseudoResult(masked, mapping, spans, residual)

    def depseudonymize(self, model_output: str, mapping: Mapping[str, str]) -> str:
        """Platzhalter -> Originalwerte. Laengste Tokens zuerst (NAME_12 vor NAME_1)."""
        if not isinstance(model_output, str):
            raise TypeError("model_output must be a string")
        out = model_output
        for token in sorted(mapping, key=len, reverse=True):
            out = out.replace(token, mapping[token])
        return out

    def scan_residuals(
        self,
        masked_text: str,
        mapping: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        """Return type and position of possible PII without echoing its value."""
        warnings: list[dict[str, Any]] = []
        tokens = set(mapping.keys())
        for label, rx in _RESIDUAL_RULES:
            for m in rx.finditer(masked_text):
                hit = m.group().strip()
                if any(tok in hit for tok in tokens):
                    continue  # ist Teil eines Platzhalters
                warnings.append({"typ": label, "pos": m.start()})
        for span in _validated_ip_spans(masked_text):
            if any(token in span.text for token in tokens):
                continue
            warnings.append({"typ": "IP-Muster", "pos": span.start})
        return warnings

    # ---- intern ----

    def _collect_spans(self, text: str) -> tuple[list[Span], bool]:
        spans: list[Span] = []
        ner_failed = False
        rules = list(_REGEX_RULES)
        if self.enable_money:
            rules.append(_MONEY_RULE)
        if self.mask_number_clusters:
            # Das Restrisiko-Tor bleibt unveraendert fail-closed. In
            # Remote-Kontexten werden potenziell identifizierende Zahlen aber
            # bereits vorher sicher ersetzt, sodass harmlose Rechnungs-/Beleg-
            # und Registerfolgen nicht die komplette Entscheidung blockieren.
            rules.append(("NUMMER", _NUMBER_CLUSTER_RX))
        for label, rx in rules:
            for m in rx.finditer(text):
                spans.append(Span(m.start(), m.end(), label, m.group(), "regex"))
        for label, rx in _VALUE_RULES:
            for m in rx.finditer(text):
                spans.append(Span(
                    m.start("value"),
                    m.end("value"),
                    label,
                    m.group("value"),
                    "regex",
                ))
        spans.extend(_validated_ip_spans(text))
        # Blockliste (immer, case-insensitive, ganze Woerter wenn moeglich)
        for term, label in self.extra_terms:
            left = r"(?<!\w)" if term[0].isalnum() or term[0] == "_" else ""
            right = r"(?!\w)" if term[-1].isalnum() or term[-1] == "_" else ""
            for m in re.finditer(left + re.escape(term) + right, text, re.IGNORECASE):
                spans.append(Span(m.start(), m.end(), label, m.group(), "blockliste", 1.0))
        # Namen-Heuristik (Anrede/Gruss) — IMMER, auch ohne NER.
        spans.extend(_heuristic_name_spans(text))
        # NER zuletzt (niedrigere Prioritaet als strukturierte Treffer)
        if self.ner_backend is not None:
            try:
                ner_spans = self.ner_backend(text)
                spans.extend(self._validated_ner_spans(ner_spans, text))
            except Exception:  # Aufrufer entscheidet ueber das Fail-closed-Tor.
                ner_failed = True
        return [s for s in spans if s.end > s.start], ner_failed

    @staticmethod
    def _validated_ner_spans(raw_spans: Sequence[Span], text: str) -> list[Span]:
        """Reject malformed third-party NER output as one fail-closed operation."""
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, (str, bytes)):
            raise TypeError("NER backend must return a sequence of Span objects")
        validated: list[Span] = []
        for span in raw_spans:
            if not isinstance(span, Span):
                raise TypeError("NER backend returned a non-Span value")
            if not (0 <= span.start < span.end <= len(text)):
                raise ValueError("NER backend returned invalid offsets")
            if text[span.start:span.end] != span.text:
                raise ValueError("NER backend span does not match source text")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", span.label):
                raise ValueError("NER backend returned an invalid label")
            validated.append(span)
        return validated

    @staticmethod
    def _resolve_overlaps(spans: list[Span]) -> list[Span]:
        """Ueberlappungen aufloesen: laengerer Treffer gewinnt; bei Gleichstand
        gewinnt die hoehere Prioritaet (blockliste > regex > ner)."""
        prio = {"blockliste": 3, "regex": 2, "heuristik": 2, "ner": 1}
        ordered = sorted(
            spans,
            key=lambda s: (-(s.end - s.start), -prio.get(s.source, 0), s.start),
        )
        kept: list[Span] = []
        for s in ordered:
            if any(not (s.end <= k.start or s.start >= k.end) for k in kept):
                continue  # ueberlappt einen bereits akzeptierten Span
            kept.append(s)
        return sorted(kept, key=lambda s: s.start)

    def _apply(self, text: str, spans: list[Span]) -> tuple[str, dict[str, str]]:
        """Spans durch stabile Platzhalter ersetzen (gleicher Wert -> gleicher Token)."""
        counters: dict[str, int] = {}
        seen: dict[tuple[str, str], str] = {}
        mapping: dict[str, str] = {}
        out: list[str] = []
        cursor = 0
        for s in spans:
            if s.label.startswith("_"):  # interne Fehler-Marker nicht ersetzen
                continue
            key = (s.label, s.text.casefold())
            token = seen.get(key)
            if token is None:
                next_number = counters.get(s.label, 0) + 1
                token = self.token_format.format(label=s.label, n=next_number)
                # Bereits im Eingabetext vorhandene Platzhalter duerfen niemals
                # in den reversiblen Tresor geraten, sonst wuerde die spaetere
                # Rueckuebersetzung auch Nutzereingaben veraendern.
                while token in text or token in mapping:
                    next_number += 1
                    token = self.token_format.format(label=s.label, n=next_number)
                counters[s.label] = next_number
                seen[key] = token
                mapping[token] = s.text
            out.append(text[cursor:s.start])
            out.append(token)
            cursor = s.end
        out.append(text[cursor:])
        return "".join(out), mapping
