# @declarative-ai/promptop

PromptOp to LlmCallDefinition lowering, the prompt Executor, and the llm-aware wrappers for rate limiting, budget, sessions, and model residency.

The op SHAPE stays in `ops`; the lowering to an `LlmCallDefinition` is LLM-specific and lives here, with the wrappers that need LLM knowledge.

`withRateLimit` and `withModelManager` both gate on a bounded resource, and each takes an `appliesTo`
predicate naming the models it governs. Those sets must be **disjoint** — provider rate headroom and
local memory residency are different scarcities, and an overlap admits a lock ordering where one call
holds a residency lease waiting for a rate slot while another holds the slot waiting for that lease:

```ts
compose(prompt)
  .with(withRateLimit({ limiter, appliesTo: (id) => !isEmbeddedModel(id) }))
  .with(withModelManager({ manager, appliesTo: isEmbeddedModel }))
```

Compose `withModelManager` **outside** `withDeadline`: queueing and model loading are not the call's own
latency, and charging a minute-long load to its window turns "the machine was busy" into a deadline
failure that reads like provider slowness. The wait is reported separately as `metrics.queuedMs`.

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
