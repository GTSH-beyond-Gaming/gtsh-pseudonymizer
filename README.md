# gtsh-pseudonymizer

Reversible pseudonymization for German-language local-AI workflows. The package
masks structured identifiers and detected names, organizations, and locations
with stable placeholders such as `[NAME_1]` and `[EMAIL_1]`.

> This is pseudonymization, not anonymization. The mapping remains personal data
> and must be kept local, short-lived, and out of logs and persistent storage.

## Safe default

Use `protect_fields` when text may cross a trust boundary. It requires a working
local NER backend and a clean residual-risk scan, otherwise it raises
`PrivacyGateError`.

```python
from gtsh_pseudonymizer import Pseudonymizer, make_ollama_ner, protect_fields

engine = Pseudonymizer(
    ner_backend=make_ollama_ner(
        model="your-local-ner-model",
        host="http://127.0.0.1:11434",
    )
)

protected = protect_fields(
    "Kontakt: ada@example.test",
    "Bitte Ada Beispiel erwaehnen",
    pseudonymizer=engine,
)

model_input, instruction = protected.values
final_text = protected.restore("Antwort an [NAME_1]")
```

The lower-level `Pseudonymizer.pseudonymize()` API remains available for local
workflows that deliberately manage their own policy. It must not be mistaken for
a successful trust-boundary decision; inspect `residual_warnings` or use the safe
facade.

`protect_fields` verifies that the supplied `Pseudonymizer` actually has an NER
backend. A caller cannot turn an absent backend into a successful decision with a
boolean flag. Residual warnings contain only a type and position, never the
matched source value or an exception message.

## Detector coverage

The deterministic German layer covers email addresses, telephone numbers, IBANs,
postal addresses, dates, URLs, IP addresses, MAC addresses, UUIDs, VAT IDs,
usernames/handles, labelled identifiers, and greeting/signature name heuristics.
An optional local NER backend adds people, organizations, and locations.

No detector or NER model guarantees 100 percent recall. The package therefore
combines detectors, a blocklist, local NER, and a residual-risk gate. Human review
remains required.

## Security boundaries

- The bundled Ollama NER adapter accepts direct HTTP loopback URLs only.
- It ignores `HTTP_PROXY` and `HTTPS_PROXY` for cleartext NER calls.
- Multi-field processing uses randomized collision-resistant boundary markers.
- Existing placeholders in source text are reserved during reversible mapping.
- Original values are never returned by the public gate decision.
- Malformed spans from third-party NER backends fail closed.
- NER responses and entity counts are bounded before parsing.

See [SECURITY.md](SECURITY.md) for reporting and deployment guidance.
Contributions must use synthetic data; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Development

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

The package supports Python 3.11 through 3.13 and ships a `py.typed` marker.

The repository is intentionally history-free and contains no mail-assistant code,
OAuth material, mailbox data, internal hostnames, or production configuration.

## Installation

Install the reviewed release directly from GitHub:

```powershell
uv pip install "gtsh-pseudonymizer @ git+https://github.com/GTSH-beyond-Gaming/gtsh-pseudonymizer.git@v0.2.0"
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
