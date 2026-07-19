# ai-exec: Shared AI Execution Library — Design

Status: implemented, declarative model landed — 2026-07-19 (canonical; supersedes `JaiRA/AI-EXEC-DESIGN.md`)

> **Implementation status.** `@declarative-ai/core`, `@declarative-ai/services`, `@declarative-ai/llm`,
> and `@declarative-ai/hw` are implemented and tested (378 tests, typecheck-clean),
> including the SPEC worked examples (§7.3, §9, §10.4) as golden tests. The **declarative
> refactor** has landed in full: the declaration/environment split
> (§1.2), the composable wrapper stack (§1.4/§3.2), the resolution pipeline + `plan`
> (§1.5/§4.1), the session + blob store seams (§1.6/§3.6), and file I/O + modality
> gating (§3.7). This document is the canonical home for that settled design; the
> deferred follow-ups are tracked in §10.
> `@declarative-ai/agents` is not started (deferred per §4.4). Notes: the hw engine
> executes runs in-process and emits a full `EngineEvent` stream through the
> `Persistence` port — step-level durable resume is future work
> (`sessionResume: false`); a ui state reached with no `InteractionPort`
> configured is run-fatal (a rejecting port stays a state-level failure), and
> the executor's `interactionPolicy: "eager"` refuses interactive definitions
> up front for search contexts; conversation mode `summary` currently degrades
> to `full_history`. Neither consumer has migrated yet: JaiRA's app build is
> next (JaiRA `DESIGN.md` §14), then findmyprompt's registry-seam swap (§8.1).
Consumers: **findmyprompt** (`C:\UbuntuCode\findmyprompt`) and **JaiRA** (`C:\UbuntuCode\JaiRA`).
Companions: [API.md](API.md) (the full API reference), [SPEC.md](SPEC.md) (the
hierarchical-workflow formalism, migrated from JaiRA), JaiRA `DESIGN.md` (the app on top of
this library), findmyprompt
`DESIGN.md`/`IMPLEMENTATION.md`.

## 1. Purpose

Both consumer projects execute "AI units" and need the same machinery around
each execution: provider routing, structured output + validation + bounded
repair, error classification, retries, rate limiting, cost metering,
cancellation, and a normalized observable event stream. findmyprompt has
battle-tested implementations of most of this, extracted here from its
`src/engine/providers/**` and `src/engine/execution/**`. JaiRA's design
specifies a nearly congruent contract that is implemented here directly.

This library additionally owns the **hierarchical-workflow formalism** — the
declarative state-machine format and engine specified in [SPEC.md](SPEC.md) —
because it is app-agnostic shared infrastructure: JaiRA is the interactive
product built on the engine, findmyprompt is the optimizer built on it, and
neither owns it.

Confirmed decisions this design implements:

- One shared package set consumed by both projects.
- findmyprompt defers process-based units (no worker host yet); the contract
  supports them from day one, implementations land on JaiRA's schedule.
- Memoization treats side-effecting units soundly by including a workspace
  snapshot hash in the memo key.
- First searchable composite unit: the **hierarchical workflow**.
  findmyprompt's native op-graph remains its other multi-step formalism.

### 1.1 The declarative model (the principles every part inherits)

The organizing idea: **declare what you want, then execute it against a runtime.** An AI operation is a
*declaration* — a portable, serializable, content-hashable description of a call — that you run against an
injected *environment*. You `plan` a declaration to learn everything knowable without spending, then
`execute` it. ("Terraform for AI" is a useful first intuition — declare, inspect, apply — but only that; the
model stands on its own and the analogy is not load-bearing.) The principles below are settled and inherited
by every package.

### 1.2 Three layers, cleanly separated

Every operation splits into three layers, and keeping them apart is what makes calls portable, cacheable,
and swappable per deployment:

- **Declaration** — the pure, serializable, portable *"what"*: a `LlmConfiguration` (model **id**,
  prompt/messages, decoding knobs, reasoning, tool **declarations**, schema, output modalities, session
  ids). No functions, no secrets, no live handles — so it is content-hashable, and **its content hash is its
  identity** (§1.5, §3.4).
- **Environment** — the injected *"how/where"*: a provider router (keys/endpoints), a validator, a blob
  store, a session store, tool **executors**, a configuration registry, observers (logging/verification), a
  clock. Secret-bearing, non-serializable, swappable per deployment. **Every seam is optional**; the floor
  is one provider (a call that actually reaches a model errors at execution if none is present). Concretely
  this is `LlmCallEnvironment` for a direct call and `ExecServices` for the contract path.
- **Resolved transport** — internal only (`GenerateStructuredParams`): the live model handle, the
  provider-adapted schema, merged provider options, the built tool set. Never user-facing, never persisted.

`LlmCallRequest = LlmConfiguration & { env }` is the one ergonomic bundle where a declaration and its
environment co-exist; the `executeRequest` convenience is the **only** place they mix, and it strips `env`
before anything hashes or serializes the declaration.

### 1.3 Declare, plan, execute

Two verbs mark the boundary between knowing and doing:

- **`plan(declaration)`** resolves the declaration and reports what *would* happen, entirely from the local
  model catalog — provider/model, content-hash identity, how a schema would be enforced, which
  params/modalities the model accepts, a token + cost estimate, and any fit issues. **No network, no spend.**
- **`execute`** runs the declaration against an environment through the composed executor stack. The run
  record / memo cache is the durable **state** a later identical call can be served from.

### 1.4 Separation of concerns via composition

A unit is the smallest thing that delivers its value. The `llm-call` core does exactly one thing: one
declaration → one outcome. Every cross-cutting concern — repair, rate limiting, deadline fail-fast,
sessions, memoization — is a **wrapping executor** (`Executor<RIn> → Executor<ROut>`) stacked around the
core (§3.2). Two properties make this safe rather than fragile:

- **Order encodes semantics.** `memoize` outermost caches the final (post-repair) result; per-attempt
  concerns (`rateLimit`/`deadline`) sit innermost so they apply to each attempt; `memoize` must not wrap a
  `session` layer (session state isn't in the memo key), and it throws at composition time if you try.
- **Loud failure, not silent degradation.** Each wrapper *consumes* its own trigger (`withDeadline` strips
  `ctx.deadline`; `withSession` strips the declaration's session ids), and the bare core **refuses** anything
  left unconsumed. A mis-composed stack fails immediately with a clear message. The `compose(core).with(…)`
  builder additionally **type-tracks** the ctx seams each wrapper reads, so forgetting one (e.g. `stepStartMs`
  after `withDeadline`) is a compile error (§3.2).

### 1.5 Resolution, parse-don't-validate, and identity

The pipeline every call config flows through, and where `plan` stops:

```text
resolve (defaults ← configRef ← inline; family-aware, replace-with-warning)
  → parse (parseLlmConfig: strict, sampling XOR reasoning)
  → hash  (content hash = identity / memo key)     ← plan stops here (+ capability/modality/cost)
  → execute (composed wrapper stack)
```

- **Parse, don't validate.** `resolveConfig` merges config fragments loosely low→high, then `parseLlmConfig`
  parses strictly — throwing on a malformed bag, an unknown key, or an illegal sampling+reasoning
  combination. A present-but-wrong-typed field is an error, never silently coerced or dropped.
- **Family-aware merge.** A higher layer that introduces `reasoning` clears accumulated sampling knobs (and
  vice versa), each with a warning — "replace, don't explode".
- **Identity = resolved content hash.** Registry ids and inline fragments are a *composition strategy*; the
  memo key is the content hash of the fully-resolved declaration that actually reaches the model. Registry
  ids are provenance only (§3.4).

### 1.6 Sessions

A declaration carries a **logical** `sessionId` (portable, store-resolved) and/or an **explicit**
`providerSessionId` (an exact server handle, usually threaded as runtime data). The provider handle never
enters the portable declaration; it round-trips via the session store + `Outcome.session.id`. Resolution
precedence: `sessionId` present in the store → resume it; `sessionId` absent → new session, seed it; else
`providerSessionId` → resume exactly; else stateless. Client-side and provider-side sessions are the same
seam storing different things (a transcript vs a provider handle) — a wiring choice, not two mechanisms.
`withSession` implements the client-managed (transcript) path; the provider-handle path lands with the
agent-sdk executor (§4.4). The store seams themselves are §3.6.

### 1.7 Naming

- Packages are scoped `@declarative-ai/*` (`core`, `services`, `llm`, `hw`).
- `LlmConfiguration` **is** the declaration — a `Configuration` never contains a function. `LlmCallDefinition`
  adds the prompt inputs + time budget; `LlmCallConfig` is the sampling-XOR-reasoning union underneath.
- `LlmCallEnvironment` is the injected environment for a direct call; `ExecServices` is its contract-path
  counterpart.
- `LlmCallRequest = LlmConfiguration & { env }` — the ergonomic bundle consumed by `executeRequest` (§1.2).

## 2. Packages

```text
ai-exec/
  packages/
    core/        @declarative-ai/core       edge-safe: types, contract, classification, hashing
    services/    @declarative-ai/services   validation+repair, retry, rate limiting, metering
    llm/         @declarative-ai/llm        the llm-call executor + provider layer (node & edge-capable)
    hw/          @declarative-ai/hw         the hierarchical-workflow formalism: state-file
                                     loader/validator, expression language, evaluator
                                     engine, snapshot hashing, and its executor (§7)
    agents/      @declarative-ai/agents     (future) process executors: agent-sdk, claude-cli,
                                     generic-cli — built when JaiRA needs them (§4.4)
```

```text
@declarative-ai/core  ◄─  @declarative-ai/services  ◄─  @declarative-ai/llm
     ▲                                        ▲
     └───────────  @declarative-ai/hw  ──────────────┘   (engine + hierarchical-workflow executor)
                        ▲
          ┌─────────────┴──────────────┐
     JaiRA app                    findmyprompt
     (SQLite persistence impl,    (ExecOp leaf, memo,
      Electron UI, policy          search strategies)
      authoring, worktrees)
```

Rules: `@declarative-ai/core` has zero heavyweight runtime deps and no `node:*`
imports (findmyprompt's `hash.ts` discipline — `@noble/hashes`,
edge/Workflow-runtime safe). `@declarative-ai/llm` depends on the `ai` SDK + provider
packages. `@declarative-ai/hw` depends on core (+ services for validation) and
executes child states through the injected `ExecutorRegistry` — it never
imports `llm` or `agents` directly. Its outward-facing ports (`Persistence`,
`InteractionPort`, `ExecutorRegistry`) are defined in the library; apps supply
implementations (JaiRA: SQLite + real UI; findmyprompt: the bundled in-memory
persistence + scripted interaction). Neither consumer app is ever a dependency
of this library.

Packages are consumed as TypeScript source (`exports` → `src/index.ts`);
consumers bundle (JaiRA via esbuild/vite; findmyprompt via Next
`transpilePackages`). Publishing compiled artifacts is deferred until the
contract stabilizes.

## 3. Core Contract (`@declarative-ai/core`)

Sketches are normative in shape, not in every field name; the source of truth
is `packages/core/src/contract.ts`.

### 3.1 Spec and Outcome

```ts
type UnitKind =
  | "llm-call"                // one structured LLM call (MVP)
  | "hierarchical-workflow"   // a hierarchical state-machine run (MVP)
  | "agent-sdk" | "claude-cli" | "generic-cli";  // process units (deferred impls)

interface ExecutionSpec {
  kind: UnitKind;
  /** Unit definition. For llm-call: the call params. For composite/process
   *  units: a content-addressed definition bundle. Its content identity (the memo
   *  key's definition-hash component, §3.4) is derived by `withMemoize` — NOT a spec field. */
  definition: unknown;
  /** Named input values, resolved by the caller. Schemas live in the definition. */
  inputs: Record<string, unknown>;
  /** Workspace binding for units that read/mutate files. Absent for pure units. */
  workspace?: { rootDir: string; treeHash?: string };   // treeHash: git tree sha
  /** Session/conversation continuity token (provider session id, transcript ref). */
  session?: { id?: string; transcript?: unknown };
  /** Output contract: JSON Schema for the data payload; artifact targets for
   *  file outputs (engine-assigned paths). Validation executor-performed,
   *  caller-observable (repair via the opt-in `withRepair` wrapper, §5.1). */
  outputSchema?: Record<string, unknown>;
  artifactTargets?: { name: string; path: string; format?: string }[];
  limits?: { timeoutMs?: number; maxCostUsd?: number };
  policy?: unknown;                 // compiled policy; enforced per executor capability
  interaction?: InteractionPort;    // required iff the executor is interactive (§3.3)
  abortSignal?: AbortSignal;
}

/** Never thrown for unit failure — always returned, best-effort populated
 *  (findmyprompt's CallOutcome discipline). */
interface Outcome {
  value?: unknown;                  // schema-validated data payload (kept even on late failure)
  rawText?: string;                 // best-effort raw body, partial on failure
  artifacts?: ProducedArtifact[];   // name, path/content, format, contentHash
  thinking?: ReasoningSegment[];
  finishReason?: string;
  metrics: ExecMetrics;
  session?: { id?: string };        // continuation token for resume
  error?: ExecFailure;
}

interface ExecMetrics {              // = findmyprompt Metrics, generalized
  inputTokens?; outputTokens?; noCacheTokens?; cacheReadTokens?;
  cacheWriteTokens?; cacheWrite1hTokens?; reasoningTokens?; totalTokens?;
  cost?: number; costSource?: "provider" | "table"; rawUsage?: unknown;
  durationMs: number; startMs?: number;
  /** Composite units: aggregate of child executions + count, so budget gates and
   *  cost folds see through nesting without needing child records. */
  childCalls?: number; childCost?: number;
}

interface ExecFailure {
  classification: ErrorClass;
  reason: string;
  retryAfterMs?: number; rateLimited?: boolean;
}
type ErrorClass = "api-retriable" | "permanent" | "infra" | "deadline"
                | "out-of-credits" | "canceled" | "policy-denied";
```

`ErrorClass → action` mapping (halt / broken / defer / bad-draw) stays with
each caller (findmyprompt's evaluator, JaiRA's app); the *classification* is
shared so both callers reason over one vocabulary.

### 3.2 Executor and Events

```ts
// Generic in R — the env still REQUIRED at start(). Bare core + registry use the default (all seams
// optional); composing a ctx-reading wrapper NARROWS R so a stack's start demands exactly what it consumes.
interface Executor<R = ExecServices> {
  readonly kind: UnitKind;
  readonly capabilities: ExecutorCapabilities;
  start(spec: ExecutionSpec, ctx: R): ExecHandle;
}
// Every wrapper is `withX(config, inner?)`: `config` mirrors the ctx SEAMS it reads; providing a seam there
// drops it from what `.start` requires (`Omit`-tracked). Two ways to stack (identical nesting): direct
// nesting `withMemoize({cache}, withDeadline(core))` (inner as the last arg), or the inside-out builder
// `compose(core).with(a).with(b)` (type-tracks requirements). `composeExecutors(core, …)` is the loose
// variadic convenience. A wrapper is `ExecutorWrapper<RIn, ROut> = (Executor<RIn>) => Executor<ROut>`;
// `withDeadline()` needs `{ deadline, stepStartMs }` at start, `withDeadline({ deadline })` only `stepStartMs`,
// `withDeadline({ deadline, stepStartMs })` neither.
interface ExecutorCapabilities {
  structuredOutput: boolean;        // native schema-constrained output
  sessionResume: boolean;
  streaming: boolean;
  interactive: boolean;             // may emit interaction_request; needs spec.interaction
  mutatesWorkspace: boolean;        // requires spec.workspace; memo key must include treeHash
  policyEnforcement: "callback" | "config" | "none";
  memoizable: boolean;              // sound under §3.4 keying
  runtime: "edge-safe" | "node";
}
interface ExecHandle {
  events: AsyncIterable<ExecEvent>;
  outcome: Promise<Outcome>;        // resolves when done; never rejects for unit failure
  cancel(): Promise<void>;
}
type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: unknown }          // transcript stream
  | { type: "child_outcome"; ref: unknown; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: unknown }  // process units
  | { type: "command_result"; decision: string }
  | { type: "interaction_request"; stateId: string; component: string; payload: unknown }
  | { type: "output_partial"; text: string };
```

`ExecServices` is the injected seam bundle — the shared-library descendant of
findmyprompt's `RunCtx`, reduced to what *execution* (not search) needs:

```ts
interface ExecServices {
  meter?: BudgetMeter;              // reserve/settle/available (WalletMeter shape)
  validator?: SchemaValidator;      // extracted ajv.ts
  clock?: Clock; deadline?: DeadlineConfig;   // deadline consumed by the withDeadline wrapper
  providers?: ProviderRouter;       // for llm-backed executors
  registry?: ExecutorRegistry;      // composite units execute children through this
  sessions?: SessionStore; blobs?: BlobStore; // store seams (Phase 4/5)
}
// Rate limiting is NOT a ctx seam: it is construction-injected into the withRateLimit wrapper
// (composition encodes the policy; the bare core refuses unconsumed wrapper fields).
```

### 3.3 Interaction Port

The unification of "workflow UI states" and "search can't wait for humans":

```ts
interface InteractionPort {
  request(req: { stateId: string; component: string; inputs: unknown }): Promise<unknown>;
}
```

JaiRA passes its real renderer-backed bridge. findmyprompt passes a
**scripted port** whose responses come from the dataset item's fixtures, or —
default for search — a port that rejects, making any interactive state a
`permanent` failure (score 0). Search contexts therefore require either
non-interactive definitions or a fixture script; both are checked before
spending money. The approval-gate security property (SPEC §11.4) is preserved
because the port is caller-supplied: agents inside a unit never hold it.

### 3.4 Memo Keying

`@declarative-ai/core` exports the canonical key function (extracted from
findmyprompt `artifacts/hash.ts` + `canonicalize.ts`, RFC 8785 JCS + sha256):

```text
memoKey = sha256(canonicalize({
  kind, definitionHash, inputs,            // inputs sorted by name
  workspaceTreeHash?                       // required iff mutatesWorkspace
}))
```

`definitionHash` is a **memoization concern, not a spec field**: the `withMemoize`
wrapper derives it — `hashCanonical(definition)` by default, or a unit-supplied
`identify(spec)` — so no caller computes a hash it may never use or keep in sync
with `definition`. Nondeterminism is the caller's concern via findmyprompt's
existing unhashed `runId` draw-scope token — the library only guarantees the key
excludes it. For `hierarchical-workflow`, that identity **is** the workflow snapshot
hash (`workflowDefinitionHash`) computed by `@declarative-ai/hw` — one identity for
"this exact workflow version" across both consumers, passed as
`withMemoize({ cache, identify: workflowDefinitionHash })`.

### 3.5 Time budgets: `timeoutMs` vs `deadline` vs `withDeadline`

These three look redundant but sit at three different layers. The distinction only
matters for multi-step serverless runs; for a single call the deadline collapses
entirely into `timeoutMs` (see the note at the end).

- **`timeoutMs` — the primitive.** A per-call wall-clock *duration*. It resolves to the
  required `StructuredCallParams.timeoutMs` and is applied as
  `AbortSignal.timeout(timeoutMs)` at the actual provider call. "Cut off *this one call*
  after N ms." It appears at three layers that all collapse into that one number:
  the definition's own `timeoutMs` (serializable, baked into the "what"), the
  `spec.limits.timeoutMs` (per-execution limit), and the resolved value
  (`def.timeoutMs ?? spec.limits?.timeoutMs ?? DEFAULT`, with `def > limit` **refused**,
  never silently clamped). This is *layering of one value*, not three timers.

- **`ctx.deadline` (+ `ctx.stepStartMs`) — the window budget.** *Not* a duration for one
  call: "this serverless invocation started at `stepStartMs` and must finish by
  `stepStartMs + maxDurationMs − safetyMargin`, or the platform hard-kills it." It is
  **dynamic** (depends on `now`) and **shared** across every step in the invocation. The
  bare `llm-call` core does not interpret it — it **refuses** it (loud-failure discipline),
  because turning a window ceiling into a call action is a separable concern.

- **`withDeadline()` — the adapter.** The only thing that reads `ctx.deadline`/
  `ctx.stepStartMs`, translating the window budget into call behavior. It (1) **fails fast**:
  if `remainingMs = stepStartMs + maxDurationMs − margin − now` is below `floorMs`, it does
  NOT start the call and returns a `deadline`-classified failure so the caller *yields* to
  the next window (§time-vs-money) — which a plain `timeoutMs` cannot do (it would start a
  doomed call and let it get hard-killed mid-flight, losing the partial and the graceful
  classification); and (2) **clamps** `timeoutMs` (the spec limit, and the definition's if
  larger) down to `remainingMs`.

| field | answers | shape | applied by |
| --- | --- | --- | --- |
| `timeoutMs` | how long may *this call* run | absolute duration, static | always (abort signal) |
| `deadline` | when must the *whole window* end; worth starting? | relative to shared origin, dynamic | only via `withDeadline` |
| `withDeadline` | translate the window into a start/skip gate + a clamp | wrapper | you, by composing |

**Why it feels redundant (and when it genuinely is):** after `withDeadline` runs, the
deadline expresses itself *entirely* as a clamped `timeoutMs` — at the moment of the call
there is just one number. The distinction earns its keep only through (1) the fail-fast/yield
decision and (2) the shared origin across many steps. If you make a single call and don't care
about yielding, `deadline` collapses into `timeoutMs` and you don't need it. Two cleanups are
noted for later (not done): consider whether `timeoutMs` belongs in the serializable
*definition* at all (by the "declaration is pure" rule it is arguably an execution/`spec.limits`
concern), and move `withDeadline`'s static config to construction — leaving only `stepStartMs`
(genuinely per-execution) in `ctx`, consistent with the other wrappers.

### 3.6 Store seams: session store + blob store

Two optional environment seams (`@declarative-ai/core` `stores.ts`), kept as pure interfaces so the engine,
the LLM layer, and consumers all share them. They are deliberately separate because they have opposite
shapes:

```ts
// Mutable, keyed by a LOGICAL session id — a conversation's transcript, or a provider handle.
interface SessionState { messages?: unknown[]; providerSessionId?: string; }
interface SessionStore {
  get(logicalId: string): SessionState | undefined | Promise<SessionState | undefined>;
  put(logicalId: string, state: SessionState): void | Promise<void>;
}

// Immutable, content-addressed — files by hash (memo-sound), plus URL/path references.
interface BlobStore {
  load(ref: { contentHash?: string; url?: string; path?: string }): Promise<{ bytes?: Uint8Array; url?: string }>;
  put(bytes: Uint8Array, mediaType: string): Promise<{ contentHash: string }>;
}
```

- **Both optional.** An absent seam means the capability is simply unavailable: a `sessionId` with no session
  store, or a `FileInput` reference with no blob store, is an error at resolve/execute time — never a silent
  no-op (loud-failure discipline).
- **`SessionStore`** backs the client-managed conversation model of §1.6. The `withSession` wrapper resolves
  the logical id against it, prepends the stored transcript to the new turn, and folds the successful reply
  back. A workflow injects a **run-scoped** store via `ctx.sessions` so states sharing a `sessionId` continue
  one conversation (§7); an app-provided store takes precedence.
- **`BlobStore`** backs file references (§3.7): large media travels **by content hash** — small and
  memo-sound — instead of being inlined into the declaration and the memo key. `MapSessionStore` is the
  bundled in-memory `SessionStore`; apps supply durable implementations.

### 3.7 File I/O and modality gating

Media is a first-class, serializable part of a declaration, closed symmetrically on input and output:

```ts
// A neutral, serializable file/media input. Exactly one source of bytes.
interface FileInput {
  mediaType: string;              // IANA type: application/pdf, image/png, audio/mp3, …
  filename?: string;
  data: { base64: string } | { url: string } | { contentHash: string } | { path: string };
}
```

- **Input.** `attachments: FileInput[]` on the declaration are lowered to provider file/image message parts
  and merged into the user turn at the call boundary; a `contentHash`/`path` reference is resolved through
  `env.blobs`. Inline base64 or a URL needs no store; a reference does.
- **Output.** Model-generated files come back as a **parallel channel**, `outcome.artifacts` (base64
  `content` + `format` + `contentHash`), never folded into the typed `value: T`. Binary bypasses the Ajv
  boundary.
- **Modality gating at `plan`.** Each input is gated by the modality its media type requires
  (`image/*` → `image`, `audio/*` → `audio`, `video/*` → `video`, else `file`) against the model's
  `modalities.input`; each requested `outputModalities` entry against `modalities.output`. A text-only model
  handed an image input is an `issues` entry **before** you spend, not a runtime rejection after.

## 4. Unit Kinds

### 4.1 `llm-call` (`@declarative-ai/llm`)

The extracted findmyprompt leaf, behavior-identical at the wire: definition = the
structured call params (model, prompts, sampling, reasoning, timeout);
executor = the `executeStructuredCall` → `generateStructured` pipeline,
including schema adaptation, reasoning adaptation, param filtering, streaming,
cache-split token accounting, and provider-reported cost. Capabilities:
`structuredOutput: true, interactive: false, mutatesWorkspace: false,
memoizable: true, runtime: edge-safe` (the undici long-timeout dispatcher
install is node-only and conditional).

The declarative model of §1 lands entirely here:

- **The bare core (`createLlmCallExecutor`) is minimal** — resolve the model, run one structured call, map
  the result onto `Outcome`, honor caller cancel, and *refuse* any unconsumed wrapper field. Repair, rate
  limiting, deadline fail-fast, sessions, and memoization are the composable wrappers of §3.2 (source:
  `wrappers.ts`), not core behavior.
- **The environment is `LlmCallEnvironment`** (§1.2): `providers` (the one near-floor seam), `validator`,
  `toolExecutors`, `blobs`, plus an `abortSignal`. `executeRequest(req)` is the convenience over the whole
  thing — a full declaration with `env` attached, split back apart before execution.
- **`resolveConfig(layers)` composes the declaration** (§1.5): merge `[engineDefault, workflowDefault,
  registry.get(configRef), inline]` family-aware, split out the definition-layer fields
  (`system`/`prompt`/`messages`/`attachments`/`timeoutMs`), and strict-parse the rest. A
  `ConfigurationRegistry` (e.g. `MapConfigurationRegistry`) resolves named presets; identity remains the
  resolved content hash.
- **`plan(declaration)` is the dry run** (§1.3): resolve + content-hash identity + structured-output
  enforcement tier + param/modality fit + token/table-cost estimate, all from the local catalog — no
  network, no spend. It uses the *same* acceptance gate `executeStructuredCall` filters with, so plan and
  execute cannot drift.
- **Routing is explicit.** Model ids are route-prefixed `{route}/{model}` (`anthropic` = native Anthropic,
  `openrouter` = everything else); a bare id is a fail-fast error, never guessed. The `ProviderRouter`
  (`createRouter`) creates provider clients lazily, so a process that only calls one route never needs the
  other's key.

### 4.2 `hierarchical-workflow` (`@declarative-ai/hw`)

The shared state-machine engine and its executor. See §7.

### 4.3 Multi-step LLM workflows (findmyprompt)

**Not a new unit kind.** findmyprompt's content-addressed `Operation` graphs
keep executing through `runOperation` natively, with each `PromptOp` leaf
dispatched to the shared `llm-call` executor. Where a *hierarchical* pipeline
(guarded transitions, retry loops, sub-states) is wanted, that is a
`hierarchical-workflow` definition restricted to `llm-call`-backed states —
searchable and serverless-safe today.

### 4.4 Process units (deferred implementations)

`agent-sdk`, `claude-cli`, `generic-cli` are specified by the contract now
(`mutatesWorkspace: true`, `runtime: node`) and built as `@declarative-ai/agents`,
on JaiRA's schedule (its phases 4–7 are the driving consumer). The canonical
policy *model* and command-intent parsing land here with them; policy
authoring and the approval UI stay in JaiRA. findmyprompt adopts the
executors unchanged when it gains a worker host; until then its registry
simply doesn't register them.

## 5. Services (`@declarative-ai/services`)

Extracted with minimal edits; all already interface-coupled:

- **Validation + repair** — `SchemaValidator` (ajv) plus a bounded repair
  loop: on validation failure the executor re-invokes the same session with
  concrete Ajv errors, ≤ 2 repair turns recorded in metrics, then
  `error.classification` per cause. Repair is opt-in per spec (default off)
  so findmyprompt's current "bad-draw scores 0, never re-roll" semantics are
  preserved by default; JaiRA turns it on.
- **Retry** — budget-gated, `retryAfterMs`-honoring retry driver + provider
  retry policy.
- **Rate limiting** — `ConcurrencyLimiter`, `TokenBucket`,
  `AdaptiveRateController` (AIMD), model/provider limit resolution.
  Generalizes per-model → per-executor-pool by key.
- **Metering** — the `BudgetMeter`/reservation interfaces and cost
  estimation. Implementations (Stripe/ledger, local budgets) stay app-side.

## 6. Extraction Map (from findmyprompt `src/engine/`)

**Moved here** (mostly as-is; adapted imports only):

| From (findmyprompt) | To | Notes |
| --- | --- | --- |
| `providers/generate.ts` | llm | the one LLM call site |
| `providers/router.ts`, `dispatcher.ts` | llm | `ProviderRouter`, lazy clients, undici dispatcher |
| `providers/structured.ts`, `reasoning.ts`, `providers/schema/*` | llm | provider divergence adapters + capability profiles |
| `providers/model-catalog.ts`, `model-catalog-source.ts`, `registry/providerConfig.ts` | llm | catalog; refresh-from-network part node-only |
| `execution/llmStep.ts` | llm | becomes the `llm-call` executor |
| `execution/retry.ts`, `providers/retry-policy.ts` | services | |
| `execution/concurrency.ts` | services | |
| `execution/deadline.ts`, `execution/errors.ts`, `execution/costEstimate.ts` | services | |
| `schema/ajv.ts` | services | store-backed `$ref` resolution stays injectable |
| `artifacts/hash.ts`, `artifacts/canonicalize.ts` | core | hashing primitives + memo key |
| `config/llm.ts` | core | `LlmConfiguration` hierarchy, `ReasoningSpec` |
| meter/scope/classification types from `execution/context.ts` | core | interfaces only |

**Stays in findmyprompt** (the optimizer's domain): `model/` (Operation/Ref
content model), `runOperation.ts`, `evaluator.ts`, `aggregation.ts`,
`functionRegistry.ts`, `stores.ts`, `searchResolvers.ts`, `yield.ts`,
`critique.ts`, all of `search/**`, `strategies/**`, `judge/**`, `meta/**`,
the remaining `schema/*` and `artifacts/*`, plus `src/server/**`, `db/**`.

## 7. The `hierarchical-workflow` Package (`@declarative-ai/hw`)

One package containing both the **engine** — state-file loader/validator,
expression language, evaluator, snapshot hashing per [SPEC.md](SPEC.md) — and
the **executor** exposing a workflow run as an execution unit. The engine
keeps an injected-seams architecture (`Persistence` / `ExecutorRegistry` /
`InteractionPort`); those injections are exactly the executor's constructor
arguments, and exactly what JaiRA's app supplies in its richer form.

- **Definition**: a bundle of state files (the SPEC format) + root state id,
  content-addressed; its snapshot hash (`workflowDefinitionHash`, SPEC §12) is the
  identity `withMemoize` keys on (§3.4). In findmyprompt the bundle is stored as
  `json_artifacts` rows; in JaiRA it comes from `.jaira/snapshots/<hash>/`.
- **Execution**: instantiate the engine with the bundled in-memory
  persistence (durable SQLite persistence is a JaiRA-side implementation of
  the `Persistence` port), an `ExecutorRegistry` from `ctx.registry` (child
  states execute through the same shared registry — `llm-call` today, process
  units later), and an `InteractionPort` from `spec.interaction`. Run the
  root state with `spec.inputs`; terminal outputs become `Outcome.value`,
  produced artifacts become `Outcome.artifacts`.
- **Metrics rollup**: every child execution emits `child_outcome`; the
  executor folds `childCalls`/`childCost` into its own metrics so budget
  gates and wallet reconciliation see through the nest.
- **Cancellation/deadline**: `spec.abortSignal` → engine cancel (cancels
  descendant operations per SPEC §10.4); a deadline hit yields a well-formed
  `Outcome` with `classification: "deadline"` and partial metrics — never a
  hang.
- **Capabilities**: `interactive: true` (UI states supported when a port is
  supplied); `mutatesWorkspace: true` *iff* the definition contains
  process-unit states — computed from the definition, and rejected up front
  under search (no worker host). `memoizable: true` under §3.4 keying.
  `sessionResume: false` in v1.
- **Search restriction (current)**: only definitions whose states are
  `llm-call`-backed or pure-data, and either non-interactive or
  fixture-scripted. The validator checks this statically before execution.

- **States declare their call through the same declarative pipeline.** A state whose `agent.provider` is
  llm-backed builds its call via `llmCallBinding` (`ports.ts`), which runs the §1.5 `resolveConfig` pipeline
  per operation: `defaults ← registry.get(config.configRef) ← the state's inline config`, merged
  family-aware and strict-parsed. Each layer may carry definition-layer fields (a shared `system` prompt, a
  per-state `timeoutMs`) alongside the config knobs; the rendered template becomes the operation prompt (a
  config-layer `prompt` is an error — there's nothing to do with two). So a workflow state's call is an
  `LlmConfiguration` declaration like any other, not a parallel config surface.

- **Sessions coordinate by logical id.** The engine holds a run-scoped `SessionStore` (§3.6) exposed to child
  executors as `ctx.sessions`; states sharing a logical `sessionId` continue one conversation when the
  registered llm executor is composed with `withSession`. This is orthogonal to the state file's built-in
  `conversationMode` preamble (reconciling the two is a deferred follow-up, §10.1).

## 8. Consumer Migration Plans

### 8.1 findmyprompt

1. Introduce `ExecutorRegistry` and re-point the `RunCtx.executeCall` seam:
   the default inline path and the WDK step both call
   `registry.get("llm-call").start(...)` instead of importing
   `executeStructuredCall`. Pure refactor, gated on the existing test suite.
2. Replace the moved modules (§6) with `@declarative-ai/*` imports; delete the
   originals.
3. Add the composite leaf: `ExecOp { kind: "exec"; unitKind; definitionJsonId;
   input; output }` alongside `PromptOp`/`FunctionOp`, with `runExecOp`
   building an `ExecutionSpec` and recording the `Outcome` as a
   `GenerationResult` (aggregate metrics; `child_outcome` events optionally
   persisted as child GRs).
4. Register the `@declarative-ai/hw` executor; a hierarchical-workflow candidate is
   an `ExecOp` whose search space ranges over its definition bundle.
5. WDK note: an `ExecOp` step's replay-safety comes from the memo (§3.4),
   the same property `PromptOp` steps rely on today.

### 8.2 JaiRA

1. The engine is developed here as `@declarative-ai/hw`; JaiRA's repo keeps the app:
   Electron shell + renderer, durable SQLite `Persistence`, task/board model,
   policy authoring + approvals UI, Git worktree management, WSL exec layer.
2. JaiRA `DESIGN.md` §8's `RunnerAdapter`/`RunSpec`/`RunHandle` are this
   library's `Executor`/`ExecutionSpec`/`ExecHandle`. The `llm_api` adapter
   is `@declarative-ai/llm`; process executors are `@declarative-ai/agents`.
3. The engine's `RunnerRegistry` port *is* `ExecutorRegistry`; its `UiBridge`
   port *is* `InteractionPort` (JaiRA's renderer-backed implementation
   preserves the SPEC §11.4 approval-gate guarantee).
4. JaiRA design phases 1–3 (expression language, loader/validator, engine
   core) execute in this repo; JaiRA phases 4+ consume it.

## 9. Phasing

1. **Extract** — this repo; moved files with tests; findmyprompt's suite is
   the acceptance gate for the eventual swap (no behavior change intended).
2. **Registry seam** — findmyprompt `ExecutorRegistry` + `ExecOp`; `@declarative-ai/hw`
   engine work proceeds here against the same core types.
3. **hierarchical-workflow** — the `@declarative-ai/hw` executor once the engine
   core is headless-runnable; findmyprompt registers it; first end-to-end:
   optimize a small llm-call-only workflow's prompts against a dataset.
4. **Process units** — `@declarative-ai/agents`, built as JaiRA phases 4–7 need
   them; findmyprompt adoption waits on an execution host.

## 10. Risks and Open Points

- **Contract churn across three repos.** Mitigated by sequencing: extracted
  from working findmyprompt code; JaiRA consumes types before writing
  executors. File-linked workspaces until the contract survives phase 3;
  publish then.
- **`ExecOp` in a content-addressed model.** New op kinds change hashing
  surface area; `ExecOp` ids must hash `unitKind + definitionJsonId + input`
  exactly like other ops. Needs focused review of findmyprompt
  `artifacts/factories.ts` when added.
- **Engine embedding assumptions.** `@declarative-ai/hw` must never acquire
  Electron/SQLite hard dependencies — better-sqlite3 stays behind
  `Persistence` (implemented JaiRA-side).
- **Repair loop vs findmyprompt eval semantics.** Repair is opt-in per spec
  (default off in findmyprompt) to avoid perturbing its statistical
  discipline; JaiRA turns it on.
- **`interactive` fixtures under search.** Scripted `InteractionPort`
  responses become part of the dataset item's identity; they must be included
  in the item's hashed content or scores aren't reproducible.
- **Spec/implementation co-location.** [SPEC.md](SPEC.md) here is the
  canonical formalism spec (see its preamble for which sections are normative
  for this library vs. JaiRA-product context).

### 10.1 Deferred follow-ups (none blocking)

The declarative refactor landed complete; these are known, non-blocking extensions (each verified green on
`npm run typecheck` + `npx vitest run` before it lands):

- **`withBudget` wrapper.** The `BudgetMeter` seam exists (§3, `reserve`/`settle`); the composable wrapper
  that reserves before a call and settles after does not yet.
- **Shrink `ExecServices`** toward only what the bare core needs (the wrappers take their deps at
  construction now; the ctx bundle can lose the fields no core reads).
- **Key `withMemoize` off `plan`'s resolved content hash** — unify the memo-key identity with the plan
  identity (§1.5, §3.4).
- **Fold tool-call / file outputs into the `withSession` transcript** (today it folds text + structured
  value only).
- **Reconcile HW's built-in `conversationMode` preamble with the session-store path** (§7 — an app currently
  picks one).
- **Forward `outputModalities` per-provider** (declarative + `plan` surfaces exist; the per-provider request
  wiring, like reasoning adaptation, does not).
- **Store output files by content hash** (`blobs.put`) rather than inline base64 in `artifacts`.
- **Build the `agent-sdk` executor** (§4.4) — unlocks the provider-side session path (`providerSessionId`).
- **Verify provider tools + schema-with-tools against a live provider** (wired structurally, unverified).
