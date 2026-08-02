# @declarative-ai/promptop

PromptOp to LlmCallDefinition lowering, the prompt Executor, and the llm-aware wrappers for rate limiting, budget, and sessions.

The op SHAPE stays in `ops`; the lowering to an `LlmCallDefinition` is LLM-specific and lives here, with the wrappers that need LLM knowledge.

```bash
npm i @declarative-ai/promptop
```

## Where it sits

Depends on `@declarative-ai/exec`, `@declarative-ai/json`, `@declarative-ai/llm`, `@declarative-ai/ops`.
It has no dependencies outside the workspace.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
