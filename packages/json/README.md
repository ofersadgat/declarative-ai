# @declarative-ai/json

The JSON vocabulary at the bottom of the declarative-ai graph: values and codecs (x-type), schema templates and inference, RFC 8785 canonicalization and hashing, and the classified error/result envelope.

It is the bottom of the graph: nothing here can be declined, and it knows nothing about operations, execution, providers, or validation.

```bash
npm i @declarative-ai/json
```

## Where it sits

Depends on no other package in the workspace.
Outside the workspace: `@noble/hashes`, `canonicalize`.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
