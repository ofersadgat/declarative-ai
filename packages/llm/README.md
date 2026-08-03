# @declarative-ai/llm

One structured LLM call, end to end and executor-free: model routing across Anthropic and OpenRouter, streaming with cache-split cost accounting, and per-provider schema and reasoning adaptation.

The direct call path is executor-free: `executeLlmCall(definition, environment)` needs nothing but this package and `@declarative-ai/json`, so installing it installs no ajv.

```bash
npm i @declarative-ai/llm
```

## Where it sits

Depends on `@declarative-ai/json`.
Outside the workspace: `@ai-sdk/anthropic`, `@ai-sdk/openai-compatible`, `@openrouter/ai-sdk-provider`,
`ai`, `undici`.

## Running models locally

Four routes: `anthropic` and `openrouter` reach remote fleets; `local` reaches an OpenAI-compatible
server on your machine (Ollama, LM Studio, `llama-server`, vLLM) and `embedded` loads weights into this
process. The `embedded` route needs the **optional peer** `node-llama-cpp`, imported only when an
`embedded/` model is actually resolved — a consumer who never uses the route neither installs nor loads
its ~100 MB of native binaries, and bundlers never walk into it.

Beyond the router, the local half of this package is:

- **`ManagedServer`** — probe-before-spawn lifecycle for a `local` endpoint. A configured `serve` means
  "make sure one is there", not "start one": an already-running server is adopted, and `close()` stops
  only what the router itself started.
- **`WeightsStore`** — resumable downloads to a caller-supplied directory, split-model parts,
  checksum-before-publish, and `WeightsCredentialRequired` for gated repos. Needs no `node-llama-cpp`:
  provisioning weights and running them are separate jobs.
- **`ResidencyManager`** — which models are loaded, given that memory is a resource that does not
  regenerate while you wait. Shared leases, per-model queues that drain before a swap, LRU eviction of
  idle models only, and no preemption.
- **`catalogRowForGguf`** — a catalog row read out of a GGUF's header (footprint, layers, trained
  context, quantization), over a range request, so an 18 GB model can be catalogued in about a second
  without downloading it. Its zero rates are a claim: local inference is free, and a row saying so is
  what lets `costSource` report `"table"` rather than `"unknown"`.

## Documentation

- [README](https://github.com/ofersadgat/declarative-ai#readme) — the package graph and how the pieces compose
- [API.md](https://github.com/ofersadgat/declarative-ai/blob/main/API.md) — the full API
- [DESIGN.md](https://github.com/ofersadgat/declarative-ai/blob/main/DESIGN.md) — why the packages split where they do

## License

MIT
