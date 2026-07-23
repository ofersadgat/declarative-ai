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

- [DESIGN.md](DESIGN.md) — architecture, the settled declarative model, the contract, the runtime/tool/
  permission model, the extraction map, and consumer migration plans.
- [API.md](API.md) — the API reference, package by package, with intended usage.
- [SPEC.md](SPEC.md) — the hierarchical-workflow formalism (normative for `@declarative-ai/hw`).

## Packages

```text
                          json
                     ┌──────┴──────┐
                    llm           ops
                     │             │
                     │            exec
                     │   ┌─────┬───┴────┬────────────┬────────────┐
                     └ promptop │   validate    permissions   agents-api
                             tools      └───────┬──────┘            │
                                                hw             agents-cli
```

Read the edges as "depends on the package above it": `ops → json`; `llm → json` only; `exec → ops`;
`promptop → exec + llm`; `validate`/`permissions`/`tools`/`agents-api` → `exec`; `hw → exec + validate +
permissions`; `agents-cli → agents-api`. **`hw` does not depend on `promptop`** — it takes the prompt
executor as a plain `Executor`, which is what keeps the AI SDK out of the workflow engine's graph.

| Package | Contents | Heavy deps |
| --- | --- | --- |
| `@declarative-ai/json` | The bottom of the graph, and nothing in it can be declined: `JsonValue`/`Jsonify<T>`/`JsonSchema<T>`/`SchemaDocument`/`Serializable`, the codec + type-name registry (`x-type`), schema templates (`$param`), schema inference, `selectType`, RFC 8785 canonicalization + hashing, the classified error vocabulary (`ErrorClass`/`Failure`), and the `Result`/`ResultWithMetrics` envelope all three result types build on | `canonicalize`, `@noble/hashes` |
| `@declarative-ai/ops` | The typed operation spine: the op model generic over a REF FAMILY (`PromptOp`/`FunctionOp`/`Parameter`/`Ref`, id-addressed or inline), the ONE function registry of discriminated entries (`pure` \| `host` \| `runtime`) with required per-variant capabilities, the `Signature` ⇄ schema bridge, the `Metrics` floor, `OperationRecord`, op metadata, and the `FromSchema` typed layer | `json-schema-to-ts` (types only) |
| `@declarative-ai/exec` | The ONE execution seam: `Executor.start(op, ctx)`, `ExecHandle`, `ExecResult`, the augmentable `ExecServices`, composition (`compose(...).with(...)`), memoization, AIMD rate limiting + token buckets, deadline arithmetic, retry, `SessionStore` | — |
| `@declarative-ai/llm` | One structured LLM call, end to end and `exec`-free: `executeLlmCall(definition, environment)`, the model router (Anthropic/OpenRouter), streaming generation with cache-split cost accounting, `LlmConfiguration` + strict parsing/resolution, schema/reasoning adaptation, tools, files, the model catalog, and `plan` | `ai`, `@ai-sdk/*`, `undici` |
| `@declarative-ai/promptop` | `PromptOp → LlmCallDefinition` lowering, the prompt `Executor`, and the llm-aware wrappers (`withRateLimit`/`withBudget`/`withSession`) | — |
| `@declarative-ai/validate` | Structural JSON-Schema subtyping, the ONE generic binding checker (parameterized by ref family), and one ajv wrapper with an injectable `$ref` resolver. The only package carrying a heavy dependency | **ajv** |
| `@declarative-ai/permissions` | The tool-call permission model: `ExecPolicy`, `Approver`, profile × mode resolution, baselines | — |
| `@declarative-ai/tools` | Workspace-backed agent tools (`read_file`/`write_file`/`edit_file`/`list_dir`/`grep`/`glob`/`run_command`) with a path-escape guard — the impls that make a composed prompt executor a coding agent | `node:*` |
| `@declarative-ai/hw` | The `hierarchical-workflow` formalism: expression language, state-file loader/validator, snapshot hashing, evaluator engine, and its executor. It takes the prompt executor as a plain `Executor`, so the AI SDK stays out of its graph | **ajv**, transitively — it depends on `validate` for the binding checker |
| `@declarative-ai/agents-api` | Delegated agents reached through an in-process SDK (`createClaudeCodeFunction`), plus the normalized `AgentQuery` seam both agent packages share. Its entry declares `policyEnforcement: "callback"` — it routes the agent's tool approvals back through `ctx.approve` | peer |
| `@declarative-ai/agents-cli` | The same adapter over a CLI subprocess, reaching back for approvals and host tools over an MCP bridge (`--mcp-config` + `--permission-prompt-tool`), so its entry declares `policyEnforcement: "callback"` too. `CLI_CONFIG_ONLY_CAPS` is the honest record for a run with no approver at all | — |

Packages are consumed as TypeScript source (`exports` → `src/index.ts`); consumers bundle (Next:
`transpilePackages`; Electron: esbuild/vite).

**They are independently usable, and that is enforced.** A structured LLM call needs `json + llm` and
nothing else — `npm i @declarative-ai/llm` installs no ajv. Optional capabilities declare their own seams
by augmenting `ExecServices`, so `exec` does not know that validation, permissions, model routing, or
workspaces-with-filesystems exist.

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
serializes the declaration. `typedSchema<T>` threads the output type through to `result.value.parsed`.

```ts
import { executeRequest, createModelRouter, typedSchema } from "@declarative-ai/llm";
import { isOk } from "@declarative-ai/json";
import { SchemaValidator } from "@declarative-ai/validate";

interface Answer { answer: string; confidence: number }

const result = await executeRequest<Answer>({
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

// NEVER throws for a call failure — always a best-effort populated result. The success branch has no
// `error` key, so `isOk` (or `"error" in result`) is the narrowing check; the model's payload is the
// `LlmOutput` under `.value`, with the parsed structured value on `.value.parsed`.
if (!isOk(result)) {
  console.error(result.error.classification, result.error.reason);
} else {
  const { parsed } = result.value;                               // LlmOutput<Answer>
  console.log(parsed?.answer, parsed?.confidence);               // typed as Answer
  console.log(result.metrics.costUsd, result.metrics.costSource); // "provider" | "table" | "unknown"
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

### 3. The contract path — one seam, a composed executor stack

An `Executor` takes an **`Operation`** and returns an `ExecHandle`; that is the whole contract. Dispatch is
by op kind — `"prompt"` to the prompt executor, `"function"` to a registry lookup — so wrapper composition
applies uniformly to both, which it could not when function ops went through a separate registry path.

```ts
import {
  MapMemoCache, compose, composeExecutors, withDeadline, withMemoize, withRetry,
  type InlineFamily, type Operation,
} from "@declarative-ai/exec";
import { AdaptiveRateController } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { createPromptExecutor, withRateLimit } from "@declarative-ai/promptop";
import { createModelRouter } from "@declarative-ai/llm";

const core = createPromptExecutor({ router: createModelRouter() });
const limiter = new AdaptiveRateController({ maxConcurrency: 8 });
const memo = new MapMemoCache(); // a MemoCache is any { get, set } — a map, or a durable store

// Wrappers nest so the INNERMOST applies per attempt and the OUTERMOST wraps the whole call:
//   withMemoize outermost  → caches the FINAL (post-retry) result
//   withRetry              → one re-attempt policy: transient (backoff) + validation (feedback repair)
//   withDeadline / withRateLimit innermost → apply per attempt
// Pick whichever form reads clearer — they nest identically.

// Form 1 — direct nesting (inner as the last arg), reads inside-out:
const e1 = withMemoize({ cache: memo }, withRetry({ transient: 3, validation: { turns: 2, feedback: true } }, withDeadline(withRateLimit({ limiter }, core))));

// Form 2 — inside-out builder (core first, each `.with` adds an OUTER layer). TYPE-TRACKS requirements:
const exec = compose(core)
  .with(withRateLimit({ limiter }))
  .with(withDeadline())   // ADDS { deadline, stepStartMs } to what `.start` requires
  .with(withRetry({ validation: { turns: 2, feedback: true } }))
  .with(withMemoize({ cache: memo }));

// Loose variadic convenience (flat list, no requirement tracking):
//   composeExecutors(core, withRateLimit({ limiter }), withDeadline(), withRetry({ transient: 3 }));
// Provide a seam at CONSTRUCTION and it drops out of `.start` — e.g. withDeadline({ deadline: { maxDurationMs: 60_000 } })
// supplies the deadline, so `.start` then requires only `stepStartMs` (still per-execution).
void [e1, composeExecutors];

// The payload is an OPERATION. A resolved op carries its inputs as bound literals, which is why the memo
// key is just the op's content hash — there is no `definition` + `inputs` pair to hash separately.
const op: Operation<InlineFamily> = {
  kind: "prompt",
  user: "Summarize declarative infrastructure in one sentence.",
  config: { model: "anthropic/claude-sonnet-5", temperature: 0.3 },
  input: {},
  output: { name: "output", kind: "json", schema: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] } },
};

// Because the stack composed withDeadline(), `start` now REQUIRES deadline + stepStartMs at the type level
// (forgetting stepStartMs is a compile error, not a runtime surprise):
const handle = exec.start(op, {
  validator: new SchemaValidator(),
  deadline: { maxDurationMs: 60_000 },
  stepStartMs: Date.now(),
  timeoutMs: 30_000,
});

for await (const ev of handle.events) {
  if (ev.type === "output_partial") process.stdout.write(ev.text);
}
const result = await handle.result;   // resolves; never rejects for a unit failure
// `handle.events` is SINGLE-CONSUMER — the loop above drains it; a second `for await` would throw.
// Read it as `isOk(result) ? result.value : result.error`.
```

> **Composition rules the types enforce.** `withMemoize` *throws at composition time* if it would wrap a
> session layer (session state isn't in the memo key, so a hit would replay a stale answer). Compose
> `withSession` **outside** `withMemoize` instead — sound, because `withSession` rewrites the sent op to
> carry the full transcript, and `withMemoize` keys on that op.

### 4. Sessions — client-managed conversations

A declaration carries a **logical** `sessionId`. `withSession` resolves it against an injected
`SessionStore`: it prepends the stored transcript to the new turn, runs the call, then folds the reply back
into the transcript (only on success). The session fields are consumed, the sent definition carries the
full history (so an inner `withMemoize` keys on the real content), and the conversation continues under
the same logical `sessionId` the caller passed — the execution result carries no session field.

```ts
import { MapSessionStore, composeExecutors, type InlineFamily, type Operation } from "@declarative-ai/exec";
import { createPromptExecutor, withSession } from "@declarative-ai/promptop";
import { createModelRouter } from "@declarative-ai/llm";

const store = new MapSessionStore();
const exec = composeExecutors(createPromptExecutor({ router: createModelRouter() }), withSession({ sessions: store }));

const ask = (user: string): Operation<InlineFamily> => ({
  kind: "prompt",
  user,
  config: { model: "anthropic/claude-sonnet-5", sessionId: "chat-1" },
  input: {},
  output: { name: "output", kind: "json" },
});

await exec.start(ask("My name is Dana."), {}).result;      // seeds the "chat-1" transcript
const out = await exec.start(ask("What's my name?"), {}).result; // prior turns are prepended automatically
// The store now holds the "chat-1" transcript, keyed by the logical id the declaration carries.
void out;
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

// `outcome` is an LlmCallResult; the model's output rides in the `LlmOutput` under `.value`:
outcome.value?.toolCalls;   // [{ toolName: "get_weather", input: { city: "NYC" } }] — what the model asked for
outcome.value?.toolResults; // [{ output: { tempF: 72, city: "NYC" } }] — what your executor returned
outcome.value?.parsed;      // the model's final answer after the tool result

// Omit the executor and the call is returned instead of run:
//   tools: [weather]  (no toolExecutors.get_weather) → value.toolCalls populated, no toolResults
```

### 6. Files — attachments in, blob outputs, modality gating

Binary data is a **leaf value**, so hydration is the ref family's business — the same as text and json.
There is no blob store to inject and no reference form to resolve: **sources are the caller's problem.**
The library takes bytes, a base64 string, or a URL the provider fetches itself, which is what keeps
`json`, `ops`, and `llm` free of `fetch` and `node:fs`.

```ts
// INPUT: attach an image, merged into the user turn at the boundary
const described = await executeRequest({
  model: "anthropic/claude-sonnet-5",
  prompt: "Describe this image.",
  attachments: [{ mediaType: "image/png", data: { base64: "<...>" } }],
  timeoutMs: 30_000,
  env: { modelRouter: createModelRouter() },
});

// Raw bytes work too — no store, no reference form to resolve:
//   attachments: [{ mediaType: "application/pdf", data: pdfBytes }]

// OUTPUT: a model-generated file comes back as BYTES on the LlmOutput...
described.value?.files?.[0]; // { mediaType: "image/png", bytes: Uint8Array }
```

A produced artifact is a **blob-kind output slot**, not a parallel output channel — there is no
`artifacts` side-channel on the result. An op declaring a blob output gets the bytes in `outcome.value`:

```ts
// `kindFor` derives `blob` from JSON Schema's OWN binary keywords, never a bespoke marker:
output: { name: "output", kind: "blob", schema: { type: "string", contentEncoding: "base64", contentMediaType: "image/png" } }
```

An inline-family blob leaf holds `Uint8Array | ByteStream`. **Stream materialization is not implemented**:
where a stream would have to become bytes — hashing for a memo key, fan-out to two consumers — the
machinery raises rather than draining it for you, so a live stream must be materialized by the caller
first. See [DESIGN.md](DESIGN.md) §10.1.

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
warning — "replace, don't explode"). It returns ONE `LlmCallDefinition` — config knobs and signature
together — because the old `{ config, definition }` pair existed only so a strict parse would not choke on
the prompt-shaped keys, and every caller immediately merged the two halves back. Identity is always the
resolved content hash; registry ids are provenance only.

```ts
import { resolveConfig, MapConfigurationRegistry } from "@declarative-ai/llm";

const registry = new MapConfigurationRegistry().set("fast", {
  model: "anthropic/claude-haiku-4-5", temperature: 0.2, maxOutputTokens: 256,
});

const { definition, warnings } = resolveConfig([
  { model: "anthropic/claude-haiku-4-5", temperature: 0.7 }, // engine default   (low priority)
  registry.get("fast"),                            // a named preset
  { temperature: 0.9, system: "be terse" },        // inline override  (high priority)
]);

definition.temperature;     // 0.9  — later layer wins
definition.maxOutputTokens; // 256 — inherited from the "fast" preset
definition.system;          // "be terse" — the signature half, merged the same way
warnings;                   // e.g. a family switch that cleared the opposite family's knobs
```

Everything is **parse, don't validate**: a present-but-wrong-typed field, an unknown key, or an
irreconcilable sampling+reasoning bag throws loudly at the boundary rather than being coerced or dropped.

### 8. Hierarchical workflows

`@declarative-ai/hw` runs a declarative state-machine (see [SPEC.md](SPEC.md)) as an execution unit. A
state has ONE `operation`, and it is an **op** from `@declarative-ai/ops`: a `PromptOp` (one structured
LLM call, dispatched to the injected prompt `Executor`) or a `FunctionOp` (`registry.functions`).
Everything that isn't a bare LLM call is a FunctionOp — host code, interactive UI, a sub-workflow, and a
delegated agent like `claude-code` alike; what distinguishes them is the resolved registry entry's
capabilities, never the op's shape. A prompt op's text comes from an inline `template` or a named `skill`
(`registry.skills`).
Session, tool, conversation, and permission concerns sit in a sibling `environment` block — tools
(`registry.tools`) are gated by a **profile × mode** permission system, see [DESIGN.md](DESIGN.md) §5.1.

```ts
import { newCapabilityRegistry, withRetry } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { createPromptExecutor } from "@declarative-ai/promptop";
import { createClaudeCodeFunction } from "@declarative-ai/agents-api";
import { createWorkflowExecutor, workflowIdentify } from "@declarative-ai/hw";

// One typed registry: the named things a state can reference — three plain Maps (`functions`, `skills`,
// `tools`). No modelRouter passed → env-key default.
const registry = newCapabilityRegistry();
// registry.functions.set("choose_option", hostFunction(impl, { interactive: true, readOnly: true, memoizable: false }));
// const agent = createClaudeCodeFunction();
// registry.functions.set("claude-code", runtimeFunction(agent.run, agent.capabilities));
// registry.skills.set("critique", "Review …");

// The executor a `PromptOp` dispatches to — hw takes it as a plain `Executor`, so it never learns that a
// prompt op HAS an llm lowering, and the AI SDK stays out of hw's dependency graph.
const prompt = withRetry({ transient: 3, validation: { turns: 2, feedback: true } }, createPromptExecutor({
  defaults: { model: "anthropic/claude-sonnet-5", temperature: 0.3 },
}));

const states = /* state-file JSON map, see SPEC.md */ {};
const hw = createWorkflowExecutor({ definition: { rootId: "feature/plan", states }, registry, prompt });

// A workflow run is started by a FUNCTION OP whose bound inputs are the workflow's inputs.
const handle = hw.start(
  {
    kind: "function",
    functionRef: "planning-workflow",
    input: { issue: { kind: "json", binding: { json: "…" } } },
    output: { name: "output", kind: "json" },
  },
  { validator: new SchemaValidator() },
);
// To memoize a workflow run, hand the memoize wrapper its snapshot identity:
//   withMemoize({ cache, identify: workflowIdentify({ rootId: "feature/plan", states }) })
void workflowIdentify;

const result = await handle.result;   // never throws for unit failure; metrics fold child calls
```

## Status

All eleven packages are implemented: one execution seam, serialization typing that tells the truth about
the wire form, and `blob` as a ref kind. Clean typecheck across every package; the full suite is green
(731 tests). No test touches a real provider or the network — fakes throughout (fake router,
`MockLanguageModelV3`, in-memory stores).

Known limits and non-blocking follow-ups are tracked in [DESIGN.md](DESIGN.md) §10.1. Publishing compiled
artifacts is deferred until the contract stabilizes across the two consumers.

## License

MIT — see [LICENSE](LICENSE).
