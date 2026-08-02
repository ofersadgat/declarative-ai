# @declarative-ai/llm

One structured LLM call, end to end and executor-free: model routing across Anthropic and OpenRouter, streaming with cache-split cost accounting, and per-provider schema and reasoning adaptation.

The direct call path is executor-free: `executeLlmCall(definition, environment)` needs nothing but this package and `@declarative-ai/json`, so installing it installs no ajv.

```bash
npm i @declarative-ai/llm
```

## Where it sits

Depends on `@declarative-ai/json`.
Outside the workspace: `@ai-sdk/anthropic`, `@openrouter/ai-sdk-provider`, `ai`, `undici`.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
