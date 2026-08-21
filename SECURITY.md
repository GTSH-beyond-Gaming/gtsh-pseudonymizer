# Security policy

## Supported versions

Version `0.1.x` is an alpha integration line. Pin an exact tag or commit and run
your own privacy-contract tests before production use.

## Reporting

Do not post suspected privacy leaks, original mappings, real personal data, or
mail samples in a public issue. Use GitHub's private vulnerability reporting or a
private security advisory for this repository.

## Operational requirements

- Keep the reversible mapping in memory only.
- Never log source values, mappings, or residual-warning contents.
- Use a trusted local NER backend; the bundled adapter enforces loopback.
- Treat every pseudonymized output as personal data.
- Use `protect_fields` for any external-model or remote-memory boundary.
- Keep a human review step before messages or records are changed or sent.
