# API Reference

The complete public API of the `declarative-ai` packages, package by package, with intended usage. This is
the reference companion to the three narrative docs:

- **[DESIGN.md](DESIGN.md)** — the architecture and the settled declarative model (read §1 first: the three
  layers, declare/plan/execute, composition, resolution, sessions).
- **[SPEC.md](SPEC.md)** — the hierarchical-workflow formalism (normative for `@declarative-ai/hw`).
- **[README.md](README.md)** — runnable, copy-pasteable examples for each capability.

This file is the *what-and-why of every export*; the README is the *how-to*. Where an example would repeat
the README, this doc links to it instead.

## Contents

- [Orientation](#orientation)
- [`@declarative-ai/core`](#declarative-aicore)
  - [The execution contract](#the-execution-contract)
  - [Composition](#composition)
  - [Injected service seams](#injected-service-seams)
  - [The LLM declaration](#the-llm-declaration)
  - [Configuration resolution](#configuration-resolution)
  - [Store seams](#store-seams)
  - [Hashing & identity](#hashing--identity)
  - [Error classification](#error-classification)
- [`@declarative-ai/services`](#declarative-aiservices)
  - [Schema validation](#schema-validation)
  - [Rate limiting](#rate-limiting)
  - [Retry](#retry)
  - [Deadline arithmetic](#deadline-arithmetic)
- [`@declarative-ai/llm`](#declarative-aillm)
  - [One-shot calls](#one-shot-calls)
  - [The contract-path executor](#the-contract-path-executor)
  - [Wrappers](#wrappers)
  - [`plan` — the dry run](#plan--the-dry-run)
  - [Provider router](#provider-router)
  - [Model catalog](#model-catalog)
  - [Cost estimation](#cost-estimation)
  - [Schema & provider adaptation](#schema--provider-adaptation)
- [`@declarative-ai/hw`](#declarative-aihw)
  - [The workflow executor](#the-workflow-executor)
  - [Bundles & identity](#bundles--identity)
  - [Provider bindings & ports](#provider-bindings--ports)
  - [The engine (lower level)](#the-engine-lower-level)
  - [State-file format types](#state-file-format-types)
  - [Expression language](#expression-language)

---

## Orientation

**Packages.** Four packages, layered `core ← services ← llm` with `hw` depending on `core` (+ `services` for
validation). `core` is edge-safe (no `node:*` imports, no heavyweight deps); `llm` pulls in the `ai` SDK +
provider packages. Consumed as TypeScript source (`exports` → `src/index.ts`); consumers bundle.

**The three layers** (DESIGN §1.2). Everything below is one of:

- a **declaration** — pure serializable data (`LlmConfiguration`/`LlmCallDefinition`, tool *declarations*,
  schema), content-hashable, its hash is its identity;
- an **environment** — injected, secret-bearing, non-serializable seams (`LlmCallEnvironment` /
  `ExecServices`): provider router, validator, stores, tool *executors*, clock. Every seam optional;
- **resolved transport** — internal only, never exported for use.

**Two ways to run a call.** The ergonomic direct path (`executeRequest`, returns a `CallOutcome<T>`), and
the full contract path (`Executor.start(spec, ctx)`, returns an `ExecHandle` with an event stream, cancel,
and a never-rejecting `outcome` promise). The contract path is what you compose wrappers and registries
around.

**Model ids are route-prefixed** `{route}/{model}` — `anthropic/…` (native Anthropic) or `openrouter/…`
(everything else). A bare id is a fail-fast error; routing is never guessed.

**Outcomes never throw for a unit failure.** Both `CallOutcome` and `Outcome` are always returned,
best-effort populated; a failure is on `.error`, and partials (`rawText`, `value`, `artifacts`) are kept.

---

## `@declarative-ai/core`

Edge-safe contract, the LLM declaration + parsing/resolution, store seams, hashing, and error
classification.

### The execution contract

The uniform way to execute any "AI unit" — an LLM call, a hierarchical workflow, or (later) a process-based
agent. Source: `contract.ts`.

#### `UnitKind`

```ts
type UnitKind = "llm-call" | "hierarchical-workflow" | "agent-sdk" | "claude-cli" | "generic-cli";
```

The discriminator for what an executor runs. `llm-call` and `hierarchical-workflow` are implemented; the
process kinds are specified but deferred (DESIGN §4.4).

#### `ExecutionSpec`

The input to `Executor.start` — *what* to run, minus the environment.

```ts
interface ExecutionSpec {
  kind: UnitKind;
  definition: unknown;                 // for llm-call: an LlmCallDefinition; for hw: { rootId, states }
  inputs: Record<string, unknown>;     // named input values; schemas live in the definition
  workspace?: { rootDir: string; treeHash?: string };
  session?: { id?: string; transcript?: unknown };
  outputSchema?: Record<string, unknown>;   // JSON Schema the data payload must satisfy
  artifactTargets?: { name: string; path: string; format?: string }[];
  limits?: { timeoutMs?: number; maxCostUsd?: number };
  policy?: unknown;
  interaction?: InteractionPort;       // required iff the executor is interactive
  abortSignal?: AbortSignal;
}
```

The definition's **content identity** is *not* a spec field — `withMemoize` derives it (default
`hashCanonical(definition)`, or a unit-supplied `identify`), so no caller computes a hash it may never use
(DESIGN §3.4).

#### `Outcome`

The result of one unit execution. **Never thrown for a unit failure** — always returned, best-effort
populated.

```ts
interface Outcome {
  value?: unknown;                     // schema-validated payload (kept even on late failure)
  rawText?: string;                    // raw/partial output text
  artifacts?: ProducedArtifact[];      // files produced (parallel channel, never in `value`)
  thinking?: ReasoningSegment[];
  toolCalls?: ToolCall[];              // what the model asked for (primary output for single-turn tools)
  toolResults?: ToolResult[];          // what your executors returned in an executed loop
  finishReason?: string;
  metrics: ExecMetrics;
  session?: { id?: string };           // continuation token when the executor supports sessionResume
  error?: ExecFailure;                 // present iff the execution failed
}
```

Supporting shapes:

| Type | Purpose |
| --- | --- |
| `ExecMetrics` (extends `TokenCounts`) | `cost`, `costSource` (`"provider"`\|`"table"`), `rawUsage`, `durationMs`, `startMs`, and composite `childCalls`/`childCost` (a composite `cost` already **includes** `childCost`). |
| `TokenCounts` | cache-split token counts: `inputTokens`/`outputTokens` (cache-inclusive) + `noCacheTokens`/`cacheReadTokens`/`cacheWriteTokens`/`cacheWrite1hTokens`/`reasoningTokens`/`totalTokens`. |
| `ExecFailure` | `{ classification: ErrorClass; reason: string; retryAfterMs?; rateLimited? }`. |
| `ProducedArtifact` | `{ name; path?; content?; format?; contentHash? }` — a file output (path for workspace units, inline `content` for pure units). |
| `ToolCall` / `ToolResult` | `{ toolCallId?; toolName; input }` / `{ toolCallId?; toolName?; output }`. |
| `ReasoningSegment` | `{ type: "reasoning"\|"tool-call"; text; textOffset; toolName?; providerMetadata? }` — a positioned thinking-trace segment. |

#### `Executor<R>` and `ExecHandle`

```ts
interface Executor<R = ExecServices> {
  readonly kind: UnitKind;
  readonly capabilities: ExecutorCapabilities;
  start(spec: ExecutionSpec, ctx: R): ExecHandle;
}

interface ExecHandle {
  events: AsyncIterable<ExecEvent>;
  outcome: Promise<Outcome>;   // resolves when done; NEVER rejects for a unit failure
  cancel(): Promise<void>;
}
```

`Executor` is **generic in `R`**, the environment `start` still requires. The bare core and
registry-dispatched executors use the default `R = ExecServices` (every seam optional). Composition
**narrows `R`**: a wrapper that reads a ctx seam adds it to `R`, so a composed stack's `start` demands
exactly the fields its wrappers consume — a missing one is a compile error (see [`compose`](#composition)).

#### `ExecutorCapabilities`

```ts
interface ExecutorCapabilities {
  structuredOutput: boolean; sessionResume: boolean; streaming: boolean;
  interactive: boolean; mutatesWorkspace: boolean;
  policyEnforcement: "callback" | "config" | "none";
  memoizable: boolean; runtime: "edge-safe" | "node";
}
```

Static capability flags used to decide fit (e.g. `withMemoize` refuses a `sessionResume: true` inner).

#### `ExecEvent`

The normalized event stream (`handle.events`):

```ts
type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: unknown }
  | { type: "child_outcome"; ref: { kind: UnitKind; label? }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed? }
  | { type: "command_result"; decision: "allowed"|"blocked"|"approved"|"denied" }
  | { type: "interaction_request"; stateId: string; component: string; payload: unknown }
  | { type: "output_partial"; text: string };
```

> The bare `llm-call` core emits no events in v1 (its `events` is an empty stream); wrappers forward inner
> streams, and the hw executor emits `progress`/`child_outcome`.

#### `ExecutorRegistry` / `MapExecutorRegistry`

```ts
interface ExecutorRegistry { get(kind: UnitKind): Executor | undefined; }
class MapExecutorRegistry { register(executor: Executor): this; get(kind): Executor | undefined; }
```

How composite units (hw) dispatch child states. Register the executors you support; `.get(kind)` looks one
up.

#### `InteractionPort`

```ts
interface InteractionPort { request(req: { stateId; component; inputs }): Promise<unknown>; }
```

The human-in-the-loop seam (DESIGN §3.3). Apps supply a UI-backed port; search/batch supply a
fixture-scripted port or one that rejects (making interactive states a `permanent` failure). Caller-supplied
and never readable inside a unit — which is what makes approval gates user-controlled by construction.

### Composition

Stack cross-cutting behaviors around a core executor. Source: `contract.ts`.

#### `ExecutorWrapper<RIn, ROut>`

```ts
type ExecutorWrapper<RIn = ExecServices, ROut = RIn> = (inner: Executor<RIn>) => Executor<ROut>;
```

A composable behavior. A construction-injected wrapper leaves the requirement unchanged
(`ExecutorWrapper<R, R>`); a ctx-reading one adds its seam (`withDeadline()` →
`ExecutorWrapper<R, R & { deadline; stepStartMs }>`). The concrete wrappers live in `@declarative-ai/llm`
([Wrappers](#wrappers)).

#### `compose` / `ComposableExecutor`

The **inside-out builder** — the recommended form, because it type-tracks requirements.

```ts
function compose<R>(core: Executor<R>): ComposableExecutor<R>;
class ComposableExecutor<R> implements Executor<R> {
  with<ROut>(wrap: ExecutorWrapper<R, ROut>): ComposableExecutor<ROut>;
  start(spec, ctx): ExecHandle;   // it IS an Executor — drops into a registry unchanged
}
```

`compose(core).with(a).with(b)` = `b(a(core))`, read core-first with each `.with` adding an **outer** layer.
Each wrapper that adds a ctx seam narrows `R`, so the final `.start` requires exactly the union of what the
stack consumes — forgetting one is a compile error. See [README example 3](README.md#3-the-contract-path--a-composed-executor-stack).

#### `composeExecutors`

```ts
function composeExecutors(core: Executor, ...wrappers: ExecutorWrapper[]): Executor;
```

The loose variadic convenience (flat list, **no** requirement tracking). Handy; `compose(...).with(...)` is
clearer about ordering and compile-time-checked.

#### `MemoCache` / `MemoizeOptions`

```ts
interface MemoCache {
  get(key: string): Promise<Outcome | undefined> | Outcome | undefined;
  set(key: string, outcome: Outcome): Promise<void> | void;
}
interface MemoizeOptions { identify?(spec: ExecutionSpec): string; }
```

The cache injected into [`withMemoize`](#wrappers). Any `{ get, set }` (a `Map`, or a durable store); both
methods may be sync or async. `identify` supplies a unit's cheaper/canonical identity (e.g.
`workflowDefinitionHash`) instead of the default `hashCanonical(spec.definition)`. Only successful outcomes
should be cached.

### Injected service seams

`ExecServices` is the environment an executor runs with — the seam bundle passed as `ctx` to `start`. All
fields optional; an absent service is a no-op (unthrottled, unmetered, unvalidated).

```ts
interface ExecServices {
  meter?: BudgetMeter;
  validator?: OutputValidator;
  clock?: Clock;
  deadline?: DeadlineConfig;
  stepStartMs?: number;            // step-start origin for deadline arithmetic
  registry?: ExecutorRegistry;     // composite units execute children through this
  providers?: unknown;             // the ProviderRouter (typed in @declarative-ai/llm)
  sessions?: SessionStore;         // run-scoped, logical-id-keyed
  blobs?: BlobStore;
}
```

The seam interfaces:

| Interface | Shape | Notes |
| --- | --- | --- |
| `OutputValidator` | `validateValue(schema, value): { ok; errors? }` | Implemented by `@declarative-ai/services` `SchemaValidator`. |
| `RateLimiter` | `schedule<T>(est: CallEstimate, run): Promise<T>`; `reportOutcome({ rateLimited?; modelId? })` | Construction-injected into `withRateLimit` (not a ctx seam). `CallEstimate = { inputTokens; outputTokens; modelId? }`. |
| `BudgetMeter` | `reserve(estCostUsd): Promise<BudgetReservation \| null>`; `availableCostUsd(): Promise<number>` | Per-call wallet reservation; `null` ⇒ balance can't cover it. `BudgetReservation.settle(actualCostUsd)` corrects the reserve. |
| `Clock` | `now(): number` | Injectable time source (tests, deterministic replay). |
| `DeadlineConfig` | `{ maxDurationMs; safetyMarginMs?; floorMs? }` | The window budget consumed by `withDeadline` (DESIGN §3.5). |

### The LLM declaration

The one canonical "how to call an LLM" type — pure serializable data, no provider/DB coupling, no AI-SDK
import. Source: `llmConfig.ts`.

#### `LlmConfiguration` and its variants

```ts
interface LlmConfiguration {                 // universal base — every model accepts these
  model: string;                             // route-prefixed id
  maxOutputTokens?; stopSequences?: string[]; seed?;
  providerOptions?: ProviderOptions;         // raw per-provider passthrough (escape hatch)
  tools?: ToolDefinition[]; toolChoice?: LlmToolChoice; maxSteps?;
  sessionId?; providerSessionId?;            // session ids (DESIGN §1.6)
  outputModalities?: string[];               // gated against modalities.output at plan time
}
interface SamplingConfiguration extends LlmConfiguration {   // sampling family
  temperature?; topP?; topK?; presencePenalty?; frequencyPenalty?;
}
interface ReasoningConfiguration extends LlmConfiguration {  // reasoning family
  reasoning: ReasoningSpec;                  // presence is the discriminant; rejects sampling knobs
}
type LlmCallConfig = SamplingConfiguration | ReasoningConfiguration;   // sampling XOR reasoning
```

A model is **sampling XOR reasoning** — never both. The union is discriminated by `reasoning` (present ⇒
reasoning). Once parsed, an illegal "both at once" state is unrepresentable.

| Type | Shape |
| --- | --- |
| `ReasoningSpec` | `{ effort?: "low"\|"medium"\|"high"; budgetTokens? }` — provider-neutral, adapted at the boundary; at least one field required. |
| `ToolDefinition` | a **function** tool `{ type?: "function"; name; description?; inputSchema; strict? }` **or** a **provider** tool `{ type: "provider"; name; id: "<provider>.<name>"; args }`. Serializable declaration only — the `execute` impl is injected. |
| `LlmToolChoice` | `"auto" \| "none" \| "required" \| { type: "tool"; toolName }`. |
| `ProviderOptions` | `Record<string, Record<string, unknown>>` — per-provider raw options, merged with adapted reasoning at the call boundary. |

#### Parsing (`parseLlmConfig`, `parseReasoningSpec`, `LlmConfigParseError`)

```ts
function parseLlmConfig(json: unknown): LlmCallConfig;      // strict; throws on malformed/unknown-key
function parseReasoningSpec(v: unknown): ReasoningSpec;
function isReasoningConfig(cfg: LlmCallConfig): cfg is ReasoningConfiguration;
class LlmConfigParseError extends Error {}
```

**Parse, don't validate.** `parseLlmConfig` turns a stored JSON blob into a concrete variant and **throws**
on a present-but-wrong-typed field, an unknown key, or a reasoning config that also carries sampling knobs.
Errors naming a definition-layer key (`prompt`/`system`/…) hint that `resolveConfig` should have split it
out.

### Configuration resolution

Compose config fragments into one valid declaration. Source: `llmConfig.ts`.

```ts
function resolveConfig(layers: Array<Record<string, unknown> | undefined>): ResolveResult;
interface ResolveResult {
  config: LlmCallConfig;                 // fully resolved + strict-parsed
  definition: Record<string, unknown>;   // split-out definition-layer fields (see below)
  warnings: string[];                    // e.g. a family switch that cleared the opposite knobs
}
```

Merge raw property bags **low→high** (`[engineDefault, workflowDefault, registry.get(ref), inline]` — later
wins per key), split out the **definition-layer fields**, then strict-parse the config bag. The merge is
**family-aware** (introducing `reasoning` clears accumulated sampling knobs, and vice-versa, each with a
warning). Identity is always the resolved content hash; registry ids are provenance only. See
[README example 7](README.md#7-config-resolution--compose-fragments-into-one-valid-declaration).

```ts
const SAMPLING_KEYS = ["temperature","topP","topK","presencePenalty","frequencyPenalty"] as const;
const LLM_DEFINITION_KEYS = ["system","prompt","messages","attachments","timeoutMs"] as const;
```

`LLM_DEFINITION_KEYS` are the prompt inputs + per-call budget that sit *alongside* the config; `resolveConfig`
returns them under `ResolveResult.definition` so the config bag itself parses strictly.

#### `ConfigurationRegistry` / `MapConfigurationRegistry`

```ts
interface ConfigurationRegistry {
  get(id: string): Record<string, unknown> | undefined;
  idOf?(config: Record<string, unknown>): string | undefined;   // reverse lookup — provenance only
}
class MapConfigurationRegistry { set(id, config): this; get(id): Record<string, unknown> | undefined; }
```

Named presets resolved into a resolution layer. `idOf` is best-effort provenance; identity is never the id.

> **Search-space helpers.** `LlmParameters`, `MakeMembersArrays<T>`, `AllKeys<T>`, `FlattenUnion<T>` are
> type-level utilities findmyprompt uses to build a search space over config dimensions. Not needed for
> normal execution.

### Store seams

Two optional environment seams (DESIGN §3.6). Pure interfaces (no AI-SDK types). Source: `stores.ts`.

```ts
interface SessionState { messages?: unknown[]; providerSessionId?: string; }   // transcript OR handle
interface SessionStore {
  get(logicalId: string): SessionState | undefined | Promise<SessionState | undefined>;
  put(logicalId: string, state: SessionState): void | Promise<void>;
}
class MapSessionStore implements SessionStore {}                  // in-memory

interface BlobRef { contentHash?: string; url?: string; path?: string; }
interface FileInput {
  mediaType: string; filename?: string;
  data: { base64: string } | { url: string } | { contentHash: string } | { path: string };
}
interface BlobStore {
  load(ref: BlobRef): Promise<{ bytes?: Uint8Array; url?: string }>;
  put(bytes: Uint8Array, mediaType: string): Promise<{ contentHash: string }>;
}
```

- `SessionStore` is **mutable, keyed by a logical id** — a conversation's transcript (client-managed) or a
  provider handle. Backs [`withSession`](#wrappers); a workflow injects a run-scoped one via `ctx.sessions`.
- `BlobStore` is **immutable, content-addressed** — files by hash (memo-sound) plus URL/path references.
  Backs `FileInput` references and generated-file storage.
- `FileInput` is the neutral, serializable media input; large media travels **by reference** so it stays out
  of the declaration and memo key.

### Hashing & identity

RFC 8785 (JCS) canonicalization + sha256. Source: `hashing.ts`.

```ts
function canonicalize(value: unknown): string;    // RFC 8785 JSON Canonicalization
function sha256Hex(payload: string): string;
function hashCanonical(value: unknown): string;   // = sha256Hex(canonicalize(value)) — content identity
function memoKey(params: {
  kind: string; definitionHash: string;
  inputs: Record<string, unknown>; workspaceTreeHash?: string;
}): string;                                        // the canonical §3.4 memo key
```

`hashCanonical` is a declaration's content identity. `memoKey` is the canonical execution key: JCS sorts
`inputs` keys, so it is invariant to caller assembly order; `workspaceTreeHash` is required for
`mutatesWorkspace` units and omitted for pure ones.

### Error classification

One shared error vocabulary both consumers reason over. Source: `classification.ts`, `encodedError.ts`.

```ts
type ErrorClass = "network-retriable" | "api-retriable" | "permanent"
                | "deadline" | "out-of-credits" | "canceled" | "policy-denied";
type ClassifiedErrorClass = "network-retriable" | "api-retriable" | "permanent";  // classifyError's subset

function classifyError(err: unknown): ErrorClass;   // walks the cause chain; default → permanent
```

| Function | Returns |
| --- | --- |
| `isNetworkError(err)` | `boolean` — connection reset/refused, DNS, undici socket. |
| `isRateLimit(err)` | `boolean` — HTTP 429. |
| `isTimeoutOrAbort(err)` | `boolean` — timeout / abort. |
| `retryAfterMs(err)` | `number \| undefined` — parsed `retry-after`. |
| `extractRateLimitInfo(err)` | `Record<string, string> \| undefined` — rate-limit headers. |
| `describeError(err)` | `string` — human-readable rendering. |
| `encodeError(e: EncodedError)` / `decodeError(text)` | round-trip an error to/from a stored string. |

`classifyError` reads an explicit retryable flag first (`isRetryable`/`retryable`), then HTTP 429/5xx, then
timeout/abort, then low-level network codes; unknown errors are **permanent** (never blindly retried).

---

## `@declarative-ai/services`

Validation, retry, rate limiting, and deadline arithmetic — each interface-coupled and independently usable.

### Schema validation

Ajv at the boundaries. Source: `validator.ts`.

```ts
class SchemaValidator implements OutputValidator {
  constructor(resolver?: SchemaResolver);
  validateValue(schema: Record<string, unknown>, value: unknown): ValidationResult;   // inline schema
  compile(schemaId: string, schemaDoc?: unknown): Promise<ValidateFunction>;          // store-backed
  validate(schemaId: string, value: unknown, schemaDoc?: unknown): Promise<ValidationResult>;
  errorsText(fn: ValidateFunction): string;
}
interface ValidationResult { ok: boolean; errors?: string; }
interface SchemaResolver { getSchema(id: string): Promise<unknown | undefined>; }
function collectRefs(node: unknown, out?: Set<string>): Set<string>;
```

Two modes: **inline** (`validateValue`, compiled + cached by the schema's content hash — this is the
`OutputValidator` used by `executeStructuredCall` and the hw engine) and **store-backed** (`compile`/
`validate`, resolving `$ref` ids lazily through an injected `SchemaResolver`). `collectRefs` gathers every
`$ref` target in a document.

### Rate limiting

Concurrency pool + token buckets with AIMD. Source: `concurrency.ts`.

```ts
class AdaptiveRateController implements RateLimiter {
  constructor(opts?: AdaptiveRateControllerOptions);
  schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T>;
  reportOutcome(outcome: { rateLimited?: boolean }): void;
  get concurrency(): number; get activeCount(): number;
}
interface AdaptiveRateControllerOptions {
  initialConcurrency?; maxConcurrency?; minConcurrency?;
  increaseEvery?;              // +1 concurrency after this many consecutive successes (default 8)
  tokensPerMinute?;           // legacy combined bucket
  modelLimits?: ModelLimitResolver;   // per-model RPM/ITPM/OTPM buckets
  now?; wait?;
}
```

The v1 controller: a concurrency pool + optional per-model token buckets, with **AIMD** on concurrency —
halve on a 429 (`reportOutcome({ rateLimited: true })`), +1 after sustained success. Header backoff
(`retry-after`) is the retry loop's job (see [Retry](#retry)); this is the orthogonal "how fast to go next
time" control. This is the concrete `RateLimiter` you inject into [`withRateLimit`](#wrappers).

Building blocks and helpers:

| Export | Purpose |
| --- | --- |
| `ConcurrencyLimiter` | a bare async concurrency gate (`run`, `setLimit`, `currentLimit`, `activeCount`). |
| `TokenBucket` | a refilling token bucket (`remove(n)` waits for capacity). |
| `PassthroughRateLimiter` | a no-op `RateLimiter` (unthrottled). |
| `ProviderDispatchRateLimiter` | routes to a per-provider `RateLimiter`. |
| `ModelRateLimits` / `ModelLimitResolver` | per-model `{ rpm?, inputTpm?, outputTpm? }` and `modelId → { key, limits }`. |
| `prefixModelLimitResolver(map)` | build a resolver from a `{ prefix: limits }` map. |
| `estimateCallTokens(prompt, system, maxOutputTokens): CallTokenEstimate` | rough input/output token split for pre-admission (chars/4 + the output ceiling). |

### Retry

Generic, budget-gated, outcome-driven retry. Source: `retry.ts`.

```ts
function retryLoop<T extends { failure?: ExecFailure }>(
  attempt: (attemptIndex: number) => Promise<T>,
  options?: RetryOptions,
): Promise<T>;
interface RetryOptions {
  retryCap?;                  // max retries beyond the first (default 5)
  budget?: RetryBudget;       // { allowMore(): boolean } — gate before each retry
  retryApiRetriable?;         // also retry api-retriable failures (explicit output re-roll opt-in)
  baseBackoffMs?; maxBackoffMs?; waitMs?; random?;
}
function backoffDelayMs(attempt, retryAfterMs, { baseBackoffMs, maxBackoffMs }, random): number;
```

**Discipline:** only `network-retriable` failures auto-retry; an `api-retriable` output failure is a re-roll
you must opt into (`retryApiRetriable`) — silently re-rolling a stochastic output until it passes biases
scores. A server `retry-after` wins over exponential backoff; otherwise full-jitter exponential
(`backoffDelayMs`). Budget-exhausted and deadline-floor reasons short-circuit immediately.

Constants: `DEFAULT_BASE_BACKOFF_MS` (500), `DEFAULT_MAX_BACKOFF_MS` (60_000).

### Deadline arithmetic

The window-budget math the `withDeadline` wrapper is built on. Source: `deadline.ts`.

```ts
function deadlineDecision(stepStartMs: number, cfg: DeadlineConfig, now: number): DeadlineDecision;
function computeRemainingMs(stepStartMs: number, cfg: DeadlineConfig, now: number): number;
interface DeadlineDecision { proceed: boolean; remainingMs: number; }
const systemClock: Clock;                 // { now: () => Date.now() }
function isDeadlineFloor(reason: string): boolean;
```

`remainingMs = stepStartMs + maxDurationMs − safetyMargin − now`; `proceed` is `remainingMs >= floor`. The
window budget (DESIGN §3.5) is *not* a per-call duration — it's shared across every step of a serverless
invocation, and `withDeadline` is the only thing that reads it. Constants: `DEFAULT_SAFETY_MARGIN_MS`
(10_000), `DEFAULT_FLOOR_MS` (5_000), `DEADLINE_FLOOR_REASON`.

---

## `@declarative-ai/llm`

The `llm-call` core, its wrappers, `plan`, the provider router, and the model catalog.

### One-shot calls

The ergonomic direct path — no contract, no wrappers. Source: `llmStep.ts`, `generate.ts`.

```ts
function executeRequest<T = unknown>(req: LlmCallRequest<T>): Promise<CallOutcome<T>>;
type LlmCallRequest<T = unknown> = StructuredCallParams<T> & { env: LlmCallEnvironment };
```

`executeRequest` is the **only** place a declaration and its environment co-exist; it strips `env` and calls
`executeStructuredCall`. See [README example 1](README.md#1-a-one-shot-structured-call).

The declaration + environment types:

```ts
type LlmCallDefinition = LlmCallConfig & CallPromptInput & { timeoutMs?: number };
type StructuredCallParams<T = unknown> = LlmCallDefinition & { schema?: JsonSchema<T>; schemaId?; timeoutMs: number };

interface LlmCallEnvironment {
  providers?: ProviderRouter;                          // required to actually reach a model
  validator?: OutputValidator;
  toolExecutors?: Record<string, ToolExecutor>;        // function-tool impls, keyed by name
  abortSignal?: AbortSignal;
  blobs?: BlobStore;
}
type CallDeps = LlmCallEnvironment & { providers: ProviderRouter };   // providers required
type ToolExecutor = (input: unknown, options: ToolCallOptions) => unknown | Promise<unknown>;
```

- `LlmCallDefinition` is the serializable definition of one call (the `LlmCallConfig` union + prompt inputs +
  optional `timeoutMs`) — and exactly `spec.definition` for kind `llm-call`.
- `LlmCallEnvironment` is the injected environment. Every seam optional; a declared tool **with** a
  `toolExecutors[name]` runs a bounded loop, **without** one is single-turn (the call is returned,
  unexecuted).

The lower-level executor and the prompt/schema helpers:

```ts
function executeStructuredCall<T>(params: StructuredCallParams<T>, deps: CallDeps): Promise<CallOutcome<T>>;
type StructuredCallExecutor = <T>(params: StructuredCallParams<T>) => Promise<CallOutcome<T>>;

interface CallPromptInput {
  system?: string | SystemModelMessage | SystemModelMessage[];
  prompt?: string | ModelMessage[];      // provide exactly one of prompt/messages
  messages?: ModelMessage[];
  attachments?: FileInput[];             // lowered to provider file parts at the boundary
}
function promptAsMessages(p: CallPromptInput): ModelMessage[];   // normalize to message-list form
function promptText(p: CallPromptInput): string;                 // extract all plain text

function typedSchema<T>(schema: Record<string, unknown>): JsonSchema<T>;   // brand a schema with its output type
type JsonSchema<T = unknown> = Record<string, unknown> & { readonly __out?: T };
```

`typedSchema<T>` threads the output type from a call's `schema` through to `CallOutcome<T>.value` (the brand
is phantom — runtime identity — and the Ajv boundary enforces conformance).

`CallOutcome<T>` is the direct-path result (the `Outcome` shape, typed): `{ value?: T; rawText; thinking?;
toolCalls?; toolResults?; artifacts?; finishReason; metrics: CallMetrics; error?: CallFailure }`.
`CallMetrics`/`CallFailure` mirror `ExecMetrics`/`ExecFailure`.

### The contract-path executor

The bare core for the full contract path (event stream, cancel, composition). Source: `executor.ts`.

```ts
function createLlmCallExecutor(options?: LlmCallExecutorOptions): Executor;
class LlmCallExecutor implements Executor { constructor(options?: LlmCallExecutorOptions); }
interface LlmCallExecutorOptions {
  router?: ProviderRouter;    // else ctx.providers, else a lazy env-key router
  runner?: CallRunner;        // the injectable call seam (tests); defaults to executeStructuredCall
}
type CallRunner = (params: StructuredCallParams, deps: CallRunnerDeps) => Promise<CallOutcome>;
type CallRunnerDeps = LlmCallEnvironment;
```

The core does exactly one call → one `Outcome`, honors caller cancel, and **refuses** any unconsumed wrapper
field (a `ctx.deadline` with no `withDeadline`, a `sessionId` with no `withSession`, a def-timeout above the
spec limit) — loud failure, not silent degradation. Capabilities: `structuredOutput: true, streaming: true,
memoizable: true, runtime: edge-safe`. Compose the behaviors you want around it (below).

### Wrappers

The composable cross-cutting concerns (`ExecutorWrapper`s). Source: `wrappers.ts`. Each is **dual-mode**:
called with a `config` object it returns the curried wrapper (for `compose(...).with(...)` /
`composeExecutors`); called with a trailing `inner` executor it applies immediately (direct nesting).

| Wrapper | Signature (config form) | What it does |
| --- | --- | --- |
| `withRepair` | `withRepair({ turns }): ExecutorWrapper` | On a schema-**validation** failure, re-invoke the inner executor with the concrete errors appended to the prompt, up to `turns` extra turns, accumulating metrics. Any non-validation failure (or success) stops immediately. |
| `withRateLimit` | `withRateLimit({ limiter }): ExecutorWrapper` | Admit the call through the injected `RateLimiter` (concurrency slot + rate headroom) and feed the outcome back (drives AIMD). A cancel while queued prevents it from ever starting. |
| `withDeadline` | `withDeadline(config?): ExecutorWrapper` | Reads `{ deadline, stepStartMs }` from config **or** ctx. Below the start floor it short-circuits with a `deadline` failure and never starts the call; otherwise clamps `timeoutMs` to the remaining window. |
| `withSession` | `withSession(config?): ExecutorWrapper` | Resolves the declaration's logical `sessionId` against a `SessionStore` (from config or `ctx.sessions`): prepend the stored transcript, run, fold the reply back on success, stamp `outcome.session.id`. Refuses `providerSessionId` (agent-sdk only). |
| `withMemoize` | `withMemoize({ cache, identify? }): ExecutorWrapper` | Key by the §3.4 memo key; hit ⇒ return cached, miss ⇒ execute and cache **on success only**. |

**Requirement tracking.** `withDeadline`/`withSession` add the ctx seams they read to what the composed
`.start` requires — unless you supply them at construction (`withDeadline({ deadline })` needs only
`stepStartMs`; `withDeadline({ deadline, stepStartMs })` needs neither). With the `compose(...).with(...)`
builder this is compile-time-checked.

**Composition rules the types/runtime enforce.**

- **Order encodes semantics** (DESIGN §1.4): `withMemoize` outermost caches the final post-repair result;
  `withRateLimit`/`withDeadline` innermost apply per attempt.
- `withMemoize` **throws at composition time** if it would wrap a `sessionResume` layer — session state
  isn't in the memo key, so a hit would replay a stale answer. Compose `withSession` **outside**
  `withMemoize` (sound: `withSession` rewrites the sent definition to carry the transcript, so the inner
  memo key sees the real content).

See [README example 3](README.md#3-the-contract-path--a-composed-executor-stack).

### `plan` — the dry run

Everything knowable before execution, from the local catalog — no network, no spend. Source: `plan.ts`.

```ts
function plan(def: LlmCallDefinition & { schema?: Record<string, unknown> }): CallPlan;
interface CallPlan {
  provider: { family: ModelFamily; modelId: string };
  contentHash: string;                    // identity / memo key
  unsupportedParams: string[];            // sampling params the model would drop
  modalities?: Modalities;
  structuredOutput?: Enforcement;         // "strict" | "advisory" | "text"
  estimate: { inputTokens: number; outputTokens: number; costUsd?: number };
  issues: string[];                       // human-readable fit problems
}
```

`plan` uses the **same** acceptance gate and cost/catalog lookups `executeStructuredCall` uses, so the
dry-run can never drift from what execution sends. It gates media inputs per media-type against
`modalities.input` and requested `outputModalities` against `modalities.output`. See
[README example 2](README.md#2-plan--the-dry-run-no-network-no-spend).

### Provider router

Explicit `{route}/{model}` routing over native Anthropic + OpenRouter. Source: `router.ts`.

```ts
function createRouter(options?: RouterOptions): ProviderRouter;
interface ProviderRouter {
  resolveModel(modelId: string, opts?: ResolveModelOptions): LanguageModel;
  isAnthropic(modelId: string): boolean;
}
interface RouterOptions {
  anthropicApiKey?; openRouterApiKey?; skipDispatcher?;
  openRouterUsageAccounting?;             // real charged cost per response (default ON)
  openRouterStrictStructuredOutputs?;     // send strict json_schema (default OFF; Ajv is the gate)
}
interface ResolveModelOptions { strictStructuredOutput?: boolean; }
```

`createRouter` creates provider clients **lazily** — a process that only calls Anthropic never needs an
OpenRouter key. Reads `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` from the environment by default.

Route parsing helpers:

| Export | Purpose |
| --- | --- |
| `type ModelFamily = "anthropic" \| "openrouter"` / `ModelRoute` | the two serving routes. |
| `parseModelRoute(modelId): { route; providerId }` | parse `{route}/{model}`; **throws** on a bare id. |
| `providerNativeId(modelId): string` | the route-stripped provider id (the catalog/pricing key). |
| `familyForModel(modelId): ModelFamily` | the serving route. |
| `isAnthropicModel(nativeId): boolean` | true for a bare `claude-*` **native** id (native-id space only). |
| `installLongTimeoutDispatcher(opts?)` | node-only undici dispatcher for long-running calls (auto-installed by `createRouter` unless `skipDispatcher`). |

### Model catalog

Local pricing, capability, and modality data. Source: `model-catalog.ts`, `model-catalog-source.ts`.

```ts
function computeCostUsd(modelId: string, inputTokens: number, outputTokens: number): number | null;
function computeCost(modelId: string, usage: UsageForCost): number | null;   // cache-aware
function hasPricing(modelId: string): boolean;
function modalitiesFor(modelId: string): Modalities | undefined;
function paramAcceptanceFor(modelId: string): ParamAcceptance;   // { accepts(key); acceptsReasoning }
function supportedParametersFor(modelId: string): string[] | undefined;
function requiredParametersFor(modelId: string): string[] | undefined;
function isReasoningModel(modelId: string): boolean;
```

| Export | Purpose |
| --- | --- |
| `ModelCatalog` | the catalog class (lookup + refresh); `defaultModelCatalog` is the shared instance backing the free functions. |
| `DEFAULT_MODELS` / `modelsSeed()` | the seeded `ModelInfo[]`. |
| `ModelInfo` / `RateSet` / `Modalities` | a catalog row: pricing rates + `modalities` (`{ input?; output? }`) + capability. |
| `UsageForCost` / `PricingOptions` | cache-split usage input to `computeCost`. |
| `ParamAcceptance` | `{ accepts(key): boolean; acceptsReasoning: boolean }` — the single gate `plan` and execute share. |
| `SAMPLING_PARAM_NAMES` | map of config-knob → provider param name. |
| `deriveIdentity` / `canonicalIdFor` / `servingProviderFor` / `displayProviderFor` | id-normalization helpers. |

The **catalog-source** module (`model-catalog-source.ts`) refreshes pricing/capabilities from the network
(node-only): `makeAnthropicDocsSource`, `makeOpenRouterSource`, `parseAnthropicDocsPricing`,
`parseOpenRouterModels`, `validatePricingRows`, `sanitizePricingRows`, plus the `PricingSource`/`FetchText`
seams and the `*_URL` constants. Only needed if you refresh the catalog yourself.

### Cost estimation

Pre-call token/cost estimates. Source: `costEstimate.ts`.

| Export | Purpose |
| --- | --- |
| `estimateInputTokens(...texts): number` | chars/4 proxy over prompt texts. |
| `estimateOutputTokens(...)` / `OutputTokenStats` / `noteOutputTokens(...)` | rolling output-size estimate from observed history. |
| `estimateCallCostUsd(modelId, inputTokens, outputTokens): number` | table-price a call. |
| `affordableOutputTokens(modelId, inputTokens, availableUsd): number` | max output tokens a budget affords. |
| `DEFAULT_HOLD_OUTPUT_MULTIPLIER`, `MIN_USEFUL_OUTPUT_TOKENS` | tuning constants. |

### Schema & provider adaptation

Advanced — how a JSON Schema is adapted to each provider's structured-output tier. Most callers never touch
this directly (`executeStructuredCall`/`plan` use it internally); it's public for consumers that need the
enforcement decision. Source: `schema/`, `structured.ts`, `reasoning.ts`, `providerConfig.ts`.

```ts
type Enforcement = "strict" | "advisory" | "text";
function profileForModelId(modelId: string): ProviderSchemaProfile;
function adaptSchemaCached(original: SchemaNode, profile: ProviderSchemaProfile): AdaptResult;  // { enforce, outgoing, postProcess, notes }
```

`Enforcement` is how a schema would be enforced for a model: `"strict"` (grammar-constrained), `"advisory"`
(json_object hint + Ajv), or `"text"` (schema described in the prompt + Ajv). Also exported: the built-in
`ProviderSchemaProfile`s (`OPENAI_STRICT`, `OPENROUTER_STRICT`, `ANTHROPIC_AI_SDK`, `ANTHROPIC_RAW`,
`JSON_OBJECT`, `ADVISORY`), the `PROFILE_REGISTRY`, `profileForCaps`, `adaptReasoning`, and the
provider-config schemas (`configSchemaFor`, `ANTHROPIC_CONFIG_SCHEMA`, `OPENROUTER_CONFIG_SCHEMA`).

---

## `@declarative-ai/hw`

The hierarchical-workflow formalism (see [SPEC.md](SPEC.md) for the semantics): loader/validator, expression
language, evaluator engine, and the executor exposing a workflow run as an execution unit.

### The workflow executor

Run a state-file bundle as a `hierarchical-workflow` unit. Source: `executor.ts`.

```ts
function createHierarchicalWorkflowExecutor(options: HwExecutorOptions): HierarchicalWorkflowExecutor;
interface HwExecutorOptions {
  providers: Record<string, ProviderBinding>;   // agent.provider name → executor binding
  skills?: SkillResolver;
  persistence?: Persistence;
  interactionPolicy?: "eager" | "lazy";          // "eager": refuse interactive defs up front (search)
}
interface HierarchicalWorkflowDefinition { rootId: string; states: Record<string, StateDef | Record<string, unknown>>; }
```

The executor loads + validates the bundle, wires abort/timeout, runs the engine, folds child cost/calls into
its metrics, and maps termination to an `Outcome` (`deadline`/`canceled`/`permanent` classifications for the
failure cases). Child states execute through `ctx.registry` (required); `spec.interaction` supplies the UI
port. Capabilities: `interactive: true, memoizable: true, sessionResume: false, runtime: edge-safe`. See
[README example 8](README.md#8-hierarchical-workflows).

Identity helpers:

```ts
function workflowDefinitionHash(spec: ExecutionSpec): string;   // the snapshot hash (SPEC §12) — pass as withMemoize identify
function workflowMemoKey(spec: ExecutionSpec): string;          // the full §3.4 memo key for a run
function interactiveStatesOf(states: Record<string, StateDef>): string[];
```

To memoize a workflow run: `withMemoize({ cache, identify: workflowDefinitionHash })`.

### Bundles & identity

Load and validate a state-file map. Source: `loader.ts`, `validate.ts`.

```ts
function loadBundle(files: Record<string, unknown>, rootId: string): WorkflowBundle;
function snapshotHash(bundle: WorkflowBundle): string;          // the versioning identity (SPEC §12)
function stateIdFromPath(relPath: string): string;             // path → state id (drop suffix)
class WorkflowLoadError extends Error {}

function validateBundle(bundle: WorkflowBundle): ValidationReport;
interface ValidationReport { errors: ValidationIssue[]; warnings: ValidationIssue[]; }
interface ValidationIssue { stateId: string; path: string; message: string; }
```

`loadBundle` parses raw state files into a `WorkflowBundle`; `validateBundle` statically checks it (wiring,
transitions, references) before any execution.

### Provider bindings & ports

How states bind to executors, and the engine's outward-facing ports. Source: `ports.ts`.

```ts
function llmCallBinding(defaults: Record<string, unknown>, opts?: LlmCallBindingOptions): ProviderBinding;
interface LlmCallBindingOptions { registry?: ConfigurationRegistry; }
interface ProviderBinding {
  kind: UnitKind;
  definition(req: { prompt: string; system?: string; config: Record<string, unknown>; timeoutMs? }): unknown;
}
```

`llmCallBinding` declares each state's call through the same `resolveConfig` pipeline (DESIGN §1.5, §7):
`defaults ← registry.get(config.configRef) ← the state's inline config`, family-aware and strict-parsed. The
rendered template is the operation prompt; a config-layer `prompt` is an error. Map it under
`HwExecutorOptions.providers` keyed by the state file's `agent.provider` name.

The ports (apps implement these):

| Export | Purpose |
| --- | --- |
| `Persistence` | `record(event: EngineEvent, atMs: number): void` — the durable run-record sink (SPEC §10.2). |
| `InMemoryPersistence` | the bundled buffering implementation (embedding & tests). |
| `EngineEvent` | the run-record event union (`instance.entered`, `operation.completed`, `transition.taken`, `instance.terminated`, …). |
| `SkillResolver` / `SkillDef` | project skill-library lookup (`get(name): SkillDef`). |
| `ArtifactRef` / `isArtifactRef` | an artifact value flowing through workflow inputs/outputs. |

The `InteractionPort` is `@declarative-ai/core`'s (supplied via `spec.interaction`).

### The engine (lower level)

The evaluator engine underneath the executor. Most callers use the executor; use the engine directly for
custom hosting. Source: `engine.ts`.

```ts
class WorkflowEngine {
  constructor(config: EngineConfig);
  run(options: WorkflowRunOptions): Promise<WorkflowRunResult>;
}
interface EngineConfig {
  bundle: WorkflowBundle;
  providers: Record<string, ProviderBinding>;
  registry: ExecutorRegistry; validator?: OutputValidator; interaction?: InteractionPort;
  skills?: SkillResolver; persistence?: Persistence; services?: ExecServices; clock?: Clock;
  onEvent?: (event: EngineEvent) => void;
}
interface WorkflowRunOptions { inputs: Record<string, unknown>; abortSignal?: AbortSignal; }
interface WorkflowRunResult {
  outcome: TerminationOutcome; outputs?: Record<string, unknown>; failure?: ExecFailure;
  artifacts: ArtifactRef[]; metrics: { childCalls: number; childCost: number; durationMs: number };
}
```

### State-file format types

The typed shape of a state file (see [SPEC.md](SPEC.md) §5 for authoring semantics). Source: `format.ts`.

| Export | Purpose |
| --- | --- |
| `StateDef` | one state file: `id`, `label`, `inputs`/`outputs`/`params`, `agent`/`skill`/`ui`, `children`, `sequence`, `transitions`, `limits`. |
| `WorkflowBundle` | `{ rootId; states: Record<string, StateDef> }`. |
| `AgentConfig` / `SkillConfig` / `UiConfig` | the three operation configs. |
| `ChildDecl` / `TransitionDecl` / `LimitsDecl` | child wiring, transitions, iteration/timeout limits. |
| `FieldSchema` / `OutputFieldSchema` | input/output field schemas (the latter adds `from`). |
| `WiringValue` | `string \| { value: unknown }` — a reference expression or a wrapped literal. |
| `ConversationMode` | `"full_history" \| "summary" \| "fresh" \| "selected_artifacts"`. |
| `TerminationOutcome` | `"success" \| "error" \| "canceled" \| "timeout"`. |
| `RunStatus` | the state-run status enum (SPEC §10.1). |
| `CONTEXT_NAMESPACES` / `TERMINATE_TARGETS` | the expression namespaces and terminal transition targets. |

### Expression language

The pure transition/wiring expression language (SPEC §6). Source: `expr.ts`.

```ts
function evaluateExpression(src: string, context: Record<string, unknown>): ExprValue;
function parseExpression(src: string): Expr;
function evaluate(expr: Expr, context: Record<string, unknown>): ExprValue;   // value may be PENDING
function referencesOf(expr: Expr): string[][];      // the paths an expression reads
class ExprError extends Error {}
const PENDING: unique symbol;                        // an async child's not-yet-resolved reference
function isPending(v: unknown): v is Pending;
```

JavaScript evaluation semantics with one deviation — property access on `undefined`/missing yields
`undefined` (implicit optional chaining) — and no side effects (no calls, imports, loops, or I/O). A
reference to a started-but-unfinished async child is `PENDING`, which skips a transition for that round
(SPEC §10.4). `referencesOf` powers the dataflow join (waiting on the inputs an expression reads).
