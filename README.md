# ai-exec

Shared AI execution library: one uniform contract for executing AI units — LLM calls,
hierarchical workflows, and (later) process-based agents — with schema validation and
bounded repair, error classification, retries, rate limiting, cost metering,
cancellation, and a normalized event stream.

Consumers: **findmyprompt** (prompt/config optimizer; searches over units) and
**JaiRA** (interactive agent-orchestration app; runs units with humans in the loop).

- [DESIGN.md](DESIGN.md) — architecture, contract, extraction map, consumer migration plans.
- [SPEC.md](SPEC.md) — the hierarchical-workflow formalism (normative for `@ai-exec/hw`).

## Packages

| Package | Contents |
| --- | --- |
| `@ai-exec/core` | Edge-safe contract (`Executor`/`ExecutionSpec`/`Outcome`/events), error classification, RFC 8785 content hashing + memo keys, LLM config types |
| `@ai-exec/services` | Ajv schema validation, budget-gated retry with full-jitter backoff, AIMD rate limiting + token buckets, deadline arithmetic |
| `@ai-exec/llm` | The `llm-call` executor: provider router (Anthropic/OpenRouter), structured streaming generation with cache-split cost accounting, schema/reasoning adaptation, model catalog, repair loop |
| `@ai-exec/hw` | The `hierarchical-workflow` formalism: expression language, state-file loader/validator, snapshot hashing, evaluator engine, and its executor |

Packages are consumed as TypeScript source (`exports` → `src/index.ts`); consumers
bundle (Next: `transpilePackages`; Electron: esbuild/vite).

## Development

```sh
npm install
npm run typecheck   # tsc across all packages
npm test            # vitest across packages/*/test
```

## Quick example

```ts
import { MapExecutorRegistry } from "@ai-exec/core";
import { SchemaValidator } from "@ai-exec/services";
import { createLlmCallExecutor } from "@ai-exec/llm";
import {
  createHierarchicalWorkflowExecutor,
  loadBundle,
  snapshotHash,
  llmCallBinding,
} from "@ai-exec/hw";

const registry = new MapExecutorRegistry().register(createLlmCallExecutor());
const hw = createHierarchicalWorkflowExecutor({
  providers: { planner: llmCallBinding({ model: "claude-sonnet-5" }) },
});

const states = /* state-file JSON map, see SPEC.md */ {};
const bundle = loadBundle(states, "feature/plan");
const handle = hw.start(
  {
    kind: "hierarchical-workflow",
    definition: { rootId: bundle.rootId, states },
    definitionHash: snapshotHash(bundle),
    inputs: { issue: "…" },
  },
  { registry, validator: new SchemaValidator() },
);
const outcome = await handle.outcome; // never throws for unit failure
```

## License

MIT — see [LICENSE](LICENSE).
