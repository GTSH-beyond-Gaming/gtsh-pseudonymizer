# Changelog

## 0.2.0 - 2026-08-21

- Derive NER availability from the configured backend instead of trusting a
  caller-provided boolean.
- Remove matched values and exception text from residual warnings.
- Validate third-party NER spans and fail closed on malformed output.
- Remove shared per-call NER error state so one engine can be used concurrently.
- Bound and deduplicate Ollama NER responses and canonicalize `localhost` to a
  literal loopback target.
- Publish typing metadata.

## 0.1.0 - 2026-08-09

- Extract reversible German PII pseudonymization from the GTSH mail assistant.
- Add a fail-closed multi-field privacy facade.
- Add a proxy-free, loopback-only Ollama NER adapter.
- Add adversarial structured-identifier and boundary tests.
