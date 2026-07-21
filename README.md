# ai-exec

A shared AI-execution library built on one idea: **declare what you want, then execute it against a
runtime.** An AI operation is a *declaration* — a portable, serializable, content-hashable description of a
call — that you run against an injected *environment* (provider keys, validators, stores). You `plan` a
declaration to see what would happen, then `execute` it to make it happen.

> **"Terraform for AI" is a handy first intuition** — a declaration you inspect before you apply, an
> environment that supplies the credentials, a plan/apply split — and nothing more. The analogy gets you
> oriented; it isn't load-bearing, and the model below stands on its own terms. Don't read it too literally.

Around that core the library provides schema validation with bounded repair, error classification, retries,
rate limiting, cost metering, cancellation, sessions, file I/O, and a normalized event stream — each an
**opt-in, composable** concern rather than a monolith.

Consumers: **findmyprompt** (prompt/config optimizer; searches over units) and **JaiRA** (interactive
agent-orchestration app; runs units with humans in the loop).

- [DESIGN.md](DESIGN.md) — architecture, the settled declarative model, the contract, extraction map, and
  consumer migration plans (the canonical design doc).
- [API.md](API.md) — the full API reference: every exported type and function, package by package, with
  intended usage.
- [SPEC.md](SPEC.md) — the hierarchical-workflow formalism (normative for `@declarative-ai/hw`).

## Packages

| Package | Contents |
| --- | --- |
| `@declarative-ai/core` | Edge-safe contract (`Executor<R>`/`ExecutionSpec`/`Outcome`/events), the `LlmConfiguration` declaration + strict parsing + resolution, error classification, RFC 8785 content hashing + memo keys, session/blob store seams, composition (`compose(...).with(...)` builder + loose `composeExecutors`) |
| `@declarative-ai/services` | Ajv schema validation, budget-gated retry with full-jitter backoff, AIMD rate limiting + token buckets, deadline arithmetic, token estimation |
| `@declarative-ai/llm` | The `llm-call` core: model router (Anthropic/OpenRouter), structured streaming generation with cache-split cost accounting, schema/reasoning adaptation, tools, file I/O, model catalog, `plan`, and the composable wrappers (`withRetry`/`withDeadline`/`withRateLimit`/`withSession`/`withMemoize`) |
| `@declarative-ai/hw` | The `hierarchical-workflow` formalism: expression language, state-file loader/validator, snapshot hashing, evaluator engine, and its executor |
| `@declarative-ai/tools` | Workspace-backed agent tools (`read_file`/`write_file`/`edit_file`/`list_dir`/`grep`/`glob`/`run_command`) that operate on `ctx.workspace` with a path-escape guard — the impls that make the composed `llm` runtime a coding agent |
| `@declarative-ai/claude-code` | The delegated `claude-code` runtime (`createClaudeCodeRuntime`): drives the Claude Agent SDK behind an injectable seam, mapping `runtime.tools`→allow-list, `ctx.workspace`→cwd, and routing the agent's tool approvals through `ctx.approve` |

Packages are consumed as TypeScript source (`exports` → `src/index.ts`); consumers bundle (Next:
`transpilePackages`; Electron: esbuild/vite).

## The model

Every operation splits cleanly into three layers:

1. **Declaration** — pure, serializable, portable *"what"*. A `LlmConfiguration` (model **id**,
   prompt/messages, decoding knobs, reasoning, tool **declarations**, schema, session ids, output
   modalities). No functions, no secrets, no live handles — so it is content-hashable and its hash is its
   identity.
2. **Environment** — the injected *"how/where"*: a provider router (keys/endpoints), a validator, a blob
   store, a session store, tool **executors**, a config registry, a clock. Secret-bearing, non-serializable,
   swappable per deployment. **Every seam is optional**; the floor is one provider.
3. **Resolved transport** — internal only. Never user-facing.

### Declare, plan, execute

The declaration moves through a fixed lifecycle, and the two verbs mark the boundary between "knowing" and
"doing":

- **Declare** — write the call as an `LlmConfiguration` (or the fuller `LlmCallDefinition`, which adds the
  prompt and time budget): model id, decoding knobs, schema, tools. Pure JSON — no secrets, no handles.
- **`plan(declaration)`** — resolve it and report everything knowable *before* execution, entirely from the
  local model catalog: which provider serves it, its content-hash identity, how a schema would be enforced,
  which params/modalities the model accepts, and a token + cost estimate. **No network, no spend.**
- **`execute`** — run the declaration against an environment through the composed executor stack. The run
  record / memo cache is the durable state that a later identical call can be served from.

The resolution pipeline (where `plan` stops):

```text
resolve  (defaults ← configRef ← inline; family-aware, replace-with-warning)
  → parse  (parseLlmConfig: strict, sampling XOR reasoning)
  → hash   (content hash = identity / memo key)     ← plan stops here (+ capability/modality/cost)
  → execute (composed wrapper stack)
```

### Separation of concerns via composition

The `llm-call` core does exactly **one thing**: one declaration → one outcome. Every cross-cutting
concern — repair, rate limiting, deadline fail-fast, sessions, memoization — is an `ExecutorWrapper`
(`Executor<RIn> → Executor<ROut>`) you stack around the core, either with the inside-out builder
`compose(core).with(a).with(b)` or by plain function application `b(a(core))`. **Stacking order encodes
semantics** (memoize outermost caches the final repaired result; rate-limit/deadline innermost are
per-attempt), and the builder **type-tracks requirements**: a wrapper that reads a ctx seam (e.g.
`withDeadline` → `deadline`/`stepStartMs`) adds it to what `.start` demands, so forgetting one is a
compile error.

This comes with a **loud-failure** guarantee: each wrapper *consumes* its own trigger (`withDeadline`
strips `ctx.deadline`; `withSession` strips the declaration's session ids), and the bare core **refuses**
anything left unconsumed. A mis-composed stack fails immediately with a clear message instead of silently
degrading.

## Install & develop

```sh
npm install
npm run typecheck   # tsc across all packages
npm test            # vitest across packages/*/test
```

---

## Examples

### 1. A one-shot structured call

`executeRequest` is the ergonomic convenience: a full declaration with its environment attached under `env`.
It is the *only* place declaration and environment co-exist — it strips `env` before anything hashes or
serializes the declaration. `typedSchema<T>` threads the output type through to `outcome.value`.

```ts
import { executeRequest, createModelRouter, typedSchema } from "@declarative-ai/llm";
import { SchemaValidator } from "@declarative-ai/services";

interface Answer { answer: string; confidence: number }

const outcome = await executeRequest<Answer>({
  // --- declaration ---
  model: "anthropic/claude-sonnet-5",
  system: "You are terse.",
  prompt: "What is 2 + 2? Include a confidence from 0 to 1.",
  temperature: 0.2,
  schema: typedSchema<Answer>({
    type: "object",
    properties: { answer: { type: "string" }, confidence: { type: "number" } },
    required: ["answer", "confidence"],
    additionalProperties: false,
  }),
  timeoutMs: 30_000,
  // --- environment ---
  env: {
    modelRouter: createModelRouter(),        // reads ANTHROPIC_API_KEY / OPENROUTER_API_KEY lazily
    validator: new SchemaValidator(), // Ajv boundary check on the way out
  },
});

// NEVER throws for a call failure — always a best-effort populated outcome.
if (outcome.error) {
  console.error(outcome.error.classification, outcome.error.reason);
} else {
  console.log(outcome.value.answer, outcome.value.confidence); // typed as Answer
  console.log(outcome.metrics.cost, outcome.metrics.costSource); // "provider" | "table"
}
```

> **Model ids are route-prefixed** `{route}/{model}`, where route is `anthropic` (native Anthropic API)
> or `openrouter` (everything else). The remainder is the provider-native id: `anthropic/claude-sonnet-5`,
> `openrouter/openai/gpt-5`. The same underlying model can be reached either way —
> `anthropic/claude-opus-4-8` (native) vs `openrouter/anthropic/claude-opus-4.8` (via OpenRouter) — with
> no ambiguity. Routing is **explicit**: a bare, unprefixed id is a fail-fast error, never guessed.

The declaration is a **union**: a model is *sampling* (`temperature`/`topP`/`topK`/penalties) **XOR**
*reasoning* (a neutral `reasoning: { effort?, budgetTokens? }`), never both. Illegal "both at once" states
are unrepresentable once parsed:

```ts
// a reasoning call — no sampling knobs allowed alongside `reasoning`
await executeRequest({
  model: "openrouter/openai/o3",
  prompt: "Prove there are infinitely many primes.",
  reasoning: { effort: "high" },
  timeoutMs: 120_000,
  env: { modelRouter: createModelRouter() },
});
```

### 2. `plan` — the dry run (no network, no spend)

Given a resolved declaration, `plan` reports everything knowable before execution, entirely from the local
model catalog: which provider serves it, its content-hash identity, how a schema would be enforced, which
params/modalities the model actually accepts, and a token + table-cost estimate.

```ts
import { plan } from "@declarative-ai/llm";

const p = plan({
  model: "anthropic/claude-sonnet-5",
  system: "be terse",
  prompt: "what is 2+2?",
  temperature: 0.7,
  maxOutputTokens: 100,
  timeoutMs: 30_000,
  schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
});

p.provider;          // { family: "anthropic", modelId: "anthropic/claude-sonnet-5" }
p.contentHash;       // 64-hex identity — same as the memo key
p.structuredOutput;  // "strict" | "advisory" | "text" — how the schema would be enforced
p.estimate;          // { inputTokens, outputTokens, costUsd? }
p.unsupportedParams; // sampling params this model would silently drop
p.issues;            // human-readable fit problems (unsupported params, modality mismatch, …)
```

### 3. The contract path — a composed executor stack

For the full contract (`Executor`/`ExecutionSpec`/`Outcome`, event stream, cancellation, and the wrapper
stack), start from the bare core and compose the behaviors you want. Order matters — see the comment.

```ts
import { compose, composeExecutors, type Outcome } from "@declarative-ai/core";
import { SchemaValidator, AdaptiveRateController } from "@declarative-ai/services";
import {
  createLlmCallExecutor, createModelRouter,
  withRateLimit, withDeadline, withRetry, withMemoize,
  type LlmCallDefinition,
} from "@declarative-ai/llm";

// A MemoCache is any { get, set } — an in-memory map, or a durable store.
const store = new Map<string, Outcome>();
const memo = { get: (k: string) => store.get(k), set: (k: string, o: Outcome) => void store.set(k, o) };

const core = createLlmCallExecutor({ router: createModelRouter() });
const limiter = new AdaptiveRateController({ maxConcurrency: 8 });

// Wrappers nest so the INNERMOST applies per attempt and the OUTERMOST wraps the whole call:
//   withMemoize outermost  → caches the FINAL (post-retry) result
//   withRetry               → one re-attempt policy: transient (backoff) + validation (feedback repair)
//   withDeadline / withRateLimit innermost → apply per attempt
// Pick whichever form reads clearer — they nest identically.

// Every wrapper takes a `config` object mirroring the ctx seams it reads, then an optional inner executor.
// Form 1 — direct nesting (inner as the last arg), reads inside-out:
const e1 = withMemoize({ cache: memo }, withRetry({ transient: 3, validation: { turns: 2, feedback: true } }, withDeadline(withRateLimit({ limiter }, core))));

// Form 2 — inside-out builder (core first, each `.with` adds an OUTER layer). TYPE-TRACKS requirements:
const exec = compose(core)
  .with(withRateLimit({ limiter }))
  .with(withDeadline())   // ADDS { deadline, stepStartMs } to what `.start` requires
  .with(withRetry({ validation: { turns: 2, feedback: true } }))
  .with(withMemoize({ cache: memo }));

// Loose variadic convenience (flat list, no requirement tracking):
//   composeExecutors(core, withRateLimit({ limiter }), withDeadline(), withRetry({ transient: 3 }), withMemoize({ cache: memo }));
// Provide a seam at CONSTRUCTION and it drops out of `.start` — e.g. withDeadline({ deadline: { maxDurationMs: 60_000 } })
// supplies the deadline, so `.start` then requires only `stepStartMs` (still per-execution).
void [e1, composeExecutors];

const def: LlmCallDefinition = {
  model: "anthropic/claude-sonnet-5",
  prompt: "Summarize declarative infrastructure in one sentence.",
  temperature: 0.3,
};

const spec = {
  kind: "llm-call" as const,
  definition: def, // its content identity is derived by withMemoize when you memoize — not carried here
  inputs: {},
  outputSchema: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] },
  limits: { timeoutMs: 30_000 },
};

// Because the stack composed withDeadline(), `start` now REQUIRES deadline + stepStartMs at the type level
// (forgetting stepStartMs is a compile error, not a runtime surprise):
const handle = exec.start(spec, {
  validator: new SchemaValidator(),
  deadline: { maxDurationMs: 60_000 },
  stepStartMs: Date.now(),
});

for await (const ev of handle.events) {
  if (ev.type === "output_partial") process.stdout.write(ev.text);
}
const outcome = await handle.outcome; // resolves; never rejects for a unit failure
```

> **Composition rules the types enforce.** `withMemoize` *throws at composition time* if it would wrap a
> session layer (session state isn't in the memo key, so a hit would replay a stale answer). Compose
> `withSession` **outside** `withMemoize` instead — sound, because `withSession` rewrites the sent
> definition to carry the full transcript, and `withMemoize` derives its key by hashing that definition.

### 4. Sessions — client-managed conversations

A declaration carries a **logical** `sessionId`. `withSession` resolves it against an injected
`SessionStore`: it prepends the stored transcript to the new turn, runs the call, then folds the reply back
into the transcript (only on success). The session fields are consumed, the sent definition carries the
full history (so an inner `withMemoize` keys on the real content), and `outcome.session.id` returns the
continuation token.

```ts
import { composeExecutors, MapSessionStore } from "@declarative-ai/core";
import { createLlmCallExecutor, createModelRouter, withSession, type LlmCallDefinition } from "@declarative-ai/llm";

const store = new MapSessionStore();
const exec = composeExecutors(createLlmCallExecutor({ router: createModelRouter() }), withSession({ sessions: store }));

const ask = (prompt: string): LlmCallDefinition => ({ model: "anthropic/claude-sonnet-5", prompt, sessionId: "chat-1" });
const run = (def: LlmCallDefinition) =>
  exec.start({ kind: "llm-call", definition: def, inputs: {}, limits: { timeoutMs: 30_000 } }, {}).outcome;

await run(ask("My name is Dana."));      // seeds the "chat-1" transcript
const out = await run(ask("What's my name?")); // prior turns are prepended automatically
out.session?.id;                         // "chat-1"
```

> Loud failures, not silent degradation: a `sessionId` with no store available is an error, and
> `providerSessionId` (a provider-side handle) is refused until the agent-sdk executor lands. A run-scoped
> store can also be injected via `ctx.sessions` (how a workflow shares one conversation across states).

### 5. Tools — serializable declarations + injected executors

Tool **declarations** (name / description / input schema, or a provider-tool id) live in the portable
declaration; the runtime `execute` implementations are injected via `env.toolExecutors`, keyed by name. A
declared tool **with** an executor runs a bounded loop; **without** one it's single-turn — the model's call
is returned in the outcome, unexecuted.

```ts
const weather = {
  name: "get_weather",
  description: "look up weather",
  inputSchema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
};

const outcome = await executeRequest({
  model: "anthropic/claude-sonnet-5",
  prompt: "What's the weather in NYC?",
  tools: [
    weather,
    // a PROVIDER tool runs server-side — no local implementation needed:
    { type: "provider", name: "search", id: "anthropic.web_search", args: { maxUses: 3 } },
  ],
  toolChoice: "auto",
  maxSteps: 4,
  timeoutMs: 30_000,
  env: {
    modelRouter: createModelRouter(),
    toolExecutors: {
      get_weather: async (input) => {
        const { city } = input as { city: string };
        return { tempF: 72, city };
      },
    },
  },
});

outcome.toolCalls;   // [{ toolName: "get_weather", input: { city: "NYC" } }] — what the model asked for
outcome.toolResults; // [{ output: { tempF: 72, city: "NYC" } }] — what your executor returned
outcome.value;       // the model's final answer after the tool result

// Omit the executor and the call is returned instead of run:
//   tools: [weather]  (no toolExecutors.get_weather) → outcome.toolCalls populated, no toolResults
```

### 6. Files — attachments in, artifacts out, modality gating

File/media inputs are a neutral, serializable `FileInput` (`{ mediaType, data }`) where the bytes travel
inline as base64, as a URL, or **by reference** (content hash / workspace path) resolved through an injected
`BlobStore` — so large media stays out of the declaration and the memo key. Model-generated files come back
as a parallel `outcome.artifacts` channel, never folded into the typed `value`.

```ts
// INPUT: attach an image (inline base64), merged into the user turn at the boundary
const described = await executeRequest({
  model: "anthropic/claude-sonnet-5",
  prompt: "Describe this image.",
  attachments: [{ mediaType: "image/png", data: { base64: "<...>" } }],
  timeoutMs: 30_000,
  env: { modelRouter: createModelRouter() },
});

// Large media by reference (needs env.blobs: BlobStore):
//   attachments: [{ mediaType: "application/pdf", data: { contentHash: "<sha256>" } }]

// OUTPUT: a model-generated file lands in artifacts (base64 + mediaType + contentHash)
described.artifacts?.[0]; // { content: "<base64>", format: "image/png", contentHash: "<sha256>" }
```

`plan` gates media **before** you spend — each input by the modality its media type requires, and each
requested output modality against the model's catalog capabilities:

```ts
plan({
  model: "openrouter/some-text-only-model",
  prompt: "look",
  attachments: [{ mediaType: "image/png", data: { base64: "x" } }],
  timeoutMs: 1000,
}).issues;
// => ["model some-text-only-model does not accept image inputs (modalities.input: text)"]
```

### 7. Config resolution — compose fragments into one valid declaration

`resolveConfig` merges raw config fragments low→high (later layers win per key), then strict-parses the
result. The merge is **family-aware** (introducing `reasoning` clears inherited sampling knobs, with a
warning — "replace, don't explode") and it splits the definition-layer fields
(`system`/`prompt`/`messages`/`attachments`/`timeoutMs`) out of the config bag. Identity is always the
resolved content hash; registry ids are provenance only.

```ts
import { resolveConfig, MapConfigurationRegistry } from "@declarative-ai/core";

const registry = new MapConfigurationRegistry().set("fast", {
  model: "anthropic/claude-haiku-4-5", temperature: 0.2, maxOutputTokens: 256,
});

const { config, definition, warnings } = resolveConfig([
  { model: "anthropic/claude-haiku-4-5", temperature: 0.7 }, // engine default   (low priority)
  registry.get("fast"),                            // a named preset
  { temperature: 0.9, system: "be terse" },        // inline override  (high priority)
]);

config.temperature;  // 0.9  — later layer wins
config.maxOutputTokens; // 256 — inherited from the "fast" preset
definition.system;   // "be terse" — split out of the config surface
warnings;            // e.g. a family switch that cleared the opposite family's knobs
```

Everything is **parse, don't validate**: a present-but-wrong-typed field, an unknown key, or an
irreconcilable sampling+reasoning bag throws loudly at the boundary rather than being coerced or dropped.

### 8. Hierarchical workflows

`@declarative-ai/hw` runs a declarative state-machine (see [SPEC.md](SPEC.md)) as an execution unit. A
state's operation is a `runtime` (dispatched through `registry.runtimes` — the `llm` runtime runs through
the same `llm-call` core) or a `function` (`registry.functions` — host code, incl. interactive UI). A
runtime's prompt comes from an inline `template` or a named `skill` (`registry.skills`). A runtime may also
be given **tools** (`registry.tools`, referenced by logical name in `runtime.tools`) that it calls
mid-loop, gated by a **profile × mode** permission system — see
[RUNTIMES-AND-PERMISSIONS.md](RUNTIMES-AND-PERMISSIONS.md).

```ts
import { MapCapabilityRegistry } from "@declarative-ai/core";
import { SchemaValidator } from "@declarative-ai/services";
import { createLlmRuntime, createLlmCallExecutor, withRetry } from "@declarative-ai/llm";
import { createHierarchicalWorkflowExecutor, loadBundle, snapshotHash } from "@declarative-ai/hw";

// One typed registry: the named things a state can reference. No modelRouter passed → env-key default;
// per-state defaults + configRef presets live on the runtime, not in a separate binding table.
const registry = new MapCapabilityRegistry();
registry.runtimes.register(
  "llm",
  createLlmRuntime({
    defaults: { model: "anthropic/claude-sonnet-5", temperature: 0.3 },
    // The composed llm-call executor stack the runtime delegates to (retry transient + repair validation):
    executor: withRetry({ transient: 3, validation: { turns: 2, feedback: true } }, createLlmCallExecutor()),
  }),
);
// registry.functions.register("choose_option", …)  // host UI / functions
// registry.skills.register("critique", "Review …")   // named prompt templates

const hw = createHierarchicalWorkflowExecutor({ registry });

const states = /* state-file JSON map, see SPEC.md */ {};
const bundle = loadBundle(states, "feature/plan"); // { rootId, states } — validated, hashable

const handle = hw.start(
  { kind: "hierarchical-workflow", definition: bundle, inputs: { issue: "…" } },
  { validator: new SchemaValidator() },
);
// To memoize a workflow run, supply its snapshot identity to the memoize wrapper:
//   withMemoize({ cache, identify: () => snapshotHash(bundle) })

const outcome = await handle.outcome; // never throws for unit failure; metrics fold child cost/calls
```

## Status

The declarative refactor has landed across all five phases plus a hardening pass — 378
tests green, clean typecheck. No test touches a real provider or the network: fakes throughout (fake router,
`MockLanguageModelV3`, in-memory stores). Remaining work is tracked as deferred follow-ups in
[DESIGN.md](DESIGN.md) §10.1. Publishing compiled artifacts is deferred until the contract stabilizes across
the two consumers.

## License

MIT — see [LICENSE](LICENSE).
