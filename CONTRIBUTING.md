# Contributing

Small, reviewable changes are welcome after the repository becomes public.

## Before opening a pull request

1. Use synthetic examples only. Never commit real messages, names, addresses,
   identifiers, mappings, access tokens, or production configuration.
2. Keep the fail-closed boundary intact. A detector or NER failure must not be
   interpreted as a safe decision.
3. Add a regression test for every detector, adapter, or gate change.
4. Run the complete local suite and package build:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
uv build
```

For a suspected privacy leak, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.
