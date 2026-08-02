# @declarative-ai/validate

Structural JSON Schema subtyping, one generic binding checker parameterized by ref family, and one ajv wrapper with an injectable $ref resolver.

The only package carrying a heavy dependency (ajv) — and nothing below it imports it, which is the point.

```bash
npm i @declarative-ai/validate
```

## Where it sits

Depends on `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/ops`.
Outside the workspace: `ajv`.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
