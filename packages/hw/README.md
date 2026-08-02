# @declarative-ai/hw

The hierarchical-workflow formalism: expression language, state-file loader and validator, snapshot hashing, evaluator engine, and its executor.

It takes the prompt executor as a plain `Executor`, so the AI SDK stays out of its dependency graph. The formalism is normative in [SPEC.md](https://github.com/ofersadgat/declarative-ai/blob/main/SPEC.md).

```bash
npm i @declarative-ai/hw
```

## Where it sits

Depends on `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/ops`, `@declarative-ai/permissions`, `@declarative-ai/validate`.
Outside the workspace: `yaml`.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
