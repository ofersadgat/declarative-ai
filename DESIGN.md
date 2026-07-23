# ai-exec: Shared AI Execution Library — Design

Status: implemented — 2026-07-22 (canonical; supersedes `JaiRA/AI-EXEC-DESIGN.md`)

> **Implementation status.** All eleven packages — `@declarative-ai/json`, `ops`, `exec`, `llm`,
> `promptop`, `validate`, `permissions`, `tools`, `hw`, `agents-api`, and `agents-cli` — are implemented
> and tested (731 tests, typecheck-clean), including the SPEC worked examples (§7.3, §9, §10.4) as
> golden tests. This document is the canonical home for the settled design: the declaration/environment
> split (§1.2), the composable wrapper stack (§1.4/§3.2), the resolution pipeline + `plan` (§1.5/§4.1),
> the session seam (§1.6/§3.6), file I/O + modality gating (§3.7), the typed operation spine (§3.1),
> the one execution seam (§3.2), and the runtime/tool/permission model (§5.1). Known limits and
> non-blocking follow-ups are tracked in §10.
> Notes: the hw engine
> executes runs in-process and emits a full `EngineEvent` stream through the
> `Persistence` port — step-level durable resume is future work
> (`sessionResume: false`); a function operation reached with no matching function
> registered is run-fatal (a rejecting function stays a state-level failure), while
> how interaction flows — block, auto-approve, refuse in a search context — is the
> workflow designer's composition through the registered functions, not an executor
> policy; conversation mode `summary` currently degrades
> to `full_history`. Neither consumer has migrated yet: JaiRA's app build is
> next (JaiRA `DESIGN.md` §14), then findmyprompt's op-vocabulary + registry-seam swap (§8.1).

Consumers: **findmyprompt** (`C:\UbuntuCode\findmyprompt`) and **JaiRA** (`C:\UbuntuCode\JaiRA`).
Companions: [API.md](API.md) (the full API reference), [SPEC.md](SPEC.md) (the
hierarchical-workflow formalism, migrated from JaiRA), [README.md](README.md) (orientation and runnable
examples), JaiRA `DESIGN.md` (the app on top of this library), findmyprompt
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
- **Environment** — the injected *"how/where"*: a model router (keys/endpoints), a validator, a session
  store, tool **executors**, a configuration registry, observers (logging/verification), a
  clock. Secret-bearing, non-serializable, swappable per deployment. **Every seam is optional**; the floor
  is one provider (a call that actually reaches a model errors at execution if none is present). Concretely
  this is `LlmCallEnvironment` for a direct call and `ExecServices` for the contract path.
- **Resolved transport** — internal only (`GenerateEnvironment`): the live model handle, the
  provider-adapted schema and its reverse transform, merged provider options, the built tool set. Never
  user-facing, never persisted.

`LlmCallRequest = LlmCallDefinition & { env }` is the one ergonomic bundle where a declaration and its
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

A unit is the smallest thing that delivers its value. The prompt-executor core does exactly one thing: one
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
enters the portable declaration; it lives in the session store, keyed by the logical id — the execution result carries no session field. Resolution
precedence: `sessionId` present in the store → resume it; `sessionId` absent → new session, seed it; else
`providerSessionId` → resume exactly; else stateless. Client-side and provider-side sessions are the same
seam storing different things (a transcript vs a provider handle) — a wiring choice, not two mechanisms.
`withSession` implements the client-managed (transcript) path; the provider-handle path lands with the
agent-sdk executor (§4.4). The store seams themselves are §3.6.

### 1.7 Naming

- Packages are scoped `@declarative-ai/*` (§2 lists all eleven).
- `LlmConfiguration` **is** the declaration — a `Configuration` never contains a function. `LlmCallDefinition`
  adds the `CallSignature` (prompt inputs, output schema, time budget); `LlmCallConfig` is the
  sampling-XOR-reasoning union underneath.
- `LlmCallEnvironment` is the injected environment for a direct call; `ExecServices` is its contract-path
  counterpart.
- `LlmCallRequest = LlmCallDefinition & { env }` — the ergonomic bundle consumed by `executeRequest` (§1.2).

## 2. Packages

```text
ai-exec/
  packages/
    json/        @declarative-ai/json        the bottom of the graph: JsonValue/Jsonify/JsonSchema,
                                             SchemaDocument, the codec + type-name registry (x-type),
                                             schema templates ($param), inference, selectType,
                                             canonicalization + hashing, the classified error
                                             vocabulary (ErrorClass/Failure), and the shared
                                             Result/ResultWithMetrics envelope all three result types build on
    ops/         @declarative-ai/ops         the typed operation spine — the op model (ref families),
                                             the ONE function registry of discriminated entries,
                                             the Signature ⇄ schema bridge, the Metrics floor,
                                             OperationRecord, op metadata, the typed layer (§3.1)
    exec/        @declarative-ai/exec        the ONE execution seam: Executor/ExecHandle/ExecResult,
                                             the augmentable ExecServices, composition, memoization,
                                             rate limiting, deadline, retry, SessionStore
    llm/         @declarative-ai/llm         one structured LLM call end to end + the provider layer
                                             (node & edge-capable); exec-FREE
    promptop/    @declarative-ai/promptop    PromptOp → LlmCallDefinition lowering, the prompt
                                             Executor, and the llm-aware wrappers
    validate/    @declarative-ai/validate    structural subtyping, the ONE generic binding checker,
                                             and the ajv wrapper — the only heavy dependency
    permissions/ @declarative-ai/permissions the tool-call permission model (profile × mode)
    tools/       @declarative-ai/tools       the Claude-Code-parity workspace tool library
    hw/          @declarative-ai/hw          the hierarchical-workflow formalism: state-file
                                             loader/validator, expression language, evaluator
                                             engine, snapshot hashing, and its executor (§7)
    agents-api/  @declarative-ai/agents-api  delegated agents reached through an in-process SDK,
                                             plus the normalized AgentQuery seam
    agents-cli/  @declarative-ai/agents-cli  the same adapter over a CLI subprocess
```

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
                                                 ▲
                               ┌─────────────────┴──────────────┐
                          JaiRA app                        findmyprompt
                     (SQLite persistence impl,      (op model + registry imported
                      Electron UI, policy            from ops at IdFamily; memo,
                      authoring, worktrees)          search strategies stay put)
```

The edges, exactly: `ops → json`; `llm → json` and nothing else in the workspace; `exec → ops`;
`promptop → exec + llm`; `validate`, `permissions`, `tools`, and `agents-api` each → `exec`;
`hw → exec + validate + permissions`; `agents-cli → agents-api`.

Rules: `@declarative-ai/json` is the floor and **nothing in it can be declined** — its only
dependencies are `canonicalize` and `@noble/hashes`, both tiny and runtime-agnostic, with no `node:*`
imports (findmyprompt's `hash.ts` discipline — edge/Workflow-runtime safe). `@declarative-ai/ops`
adds only `json-schema-to-ts`, and that is TYPES ONLY, which is what lets findmyprompt adopt the same
op vocabulary without taking on the execution contract. **Packages are independently usable, and that
is enforced by test**: `npm i @declarative-ai/llm` installs no ajv, and a structured LLM call runs with
`json + llm` and nothing else. `@declarative-ai/llm` depends on the `ai` SDK + provider packages;
`@declarative-ai/promptop` is the only thing that joins it to `exec`. `@declarative-ai/hw` **does not
depend on `promptop`** — that would drag the AI SDK and every provider package into the workflow engine.
It takes the prompt executor as a plain `Executor` (`exec`'s own type), so it never
learns that a `PromptOp` has an llm lowering; it executes child
operations through the injected `CapabilityRegistry` and `ctx.executor`. Its outward-facing ports
(`Persistence`, `CapabilityRegistry`) are defined in the library; apps supply
implementations (JaiRA: SQLite + real UI; findmyprompt: the bundled in-memory
persistence + scripted functions). Interactive behavior is no longer a
separate port: an interactive UI component is a registry entry of kind `host`
with `capabilities.interactive`. Neither consumer app is ever a dependency
of this library.

Optional capabilities declare their own seams by **augmenting `ExecServices`** (declaration merging),
so `exec` itself does not know that permissions or model routing exist (§3.2). Where a lower package
needs a capability a higher one implements, it declares instead the **minimal structural interface** it
consumes and never learns the concrete type — `json`'s three-line `OutputValidator`, which `validate`'s
ajv-backed `SchemaValidator` implements, is the load-bearing case: `promptop` needs the validation seam
without depending on `validate`, because that would put ajv back in the LLM path. Declaration merging is
used where it earns its global-namespace cost (`permissions` declares `policy`/`approve`; `promptop`
declares `modelRouter`); a plain interface is used where the type is small and universally needed
(`Workspace`'s `root` + `treeHash` live in `exec`, while the filesystem that reads them lives in `tools`).

Packages are consumed as TypeScript source (`exports` → `src/index.ts`);
consumers bundle (JaiRA via esbuild/vite; findmyprompt via Next
`transpilePackages`). Publishing compiled artifacts is deferred until the
contract stabilizes.

## 3. The Execution Contract (`@declarative-ai/exec`)

Sketches are normative in shape, not in every field name; the source of truth
is `packages/exec/src/contract.ts` (with the op vocabulary in `packages/ops/src/model.ts` and the
shared error/telemetry vocabulary in `packages/json/src/failure.ts`).

### 3.1 Operation and ExecResult

**What flows through the contract is an `Operation`.** There is ONE execution seam, and its payload is
the op model of `@declarative-ai/ops`: a `PromptOp` (one structured LLM call) or a `FunctionOp` (a
registered function). There is no third kind and no separate execution-spec taxonomy: dispatch is
`op.kind === "prompt"` → the prompt executor, `"function"` → a registry lookup by `functionRef`, and
that is the whole of it. Because that is one ordinary seam, wrapper composition (§3.2) reaches function
ops as well as prompt ops.

Everything an op needs sits on the op or on the ctx, never on a wrapper payload in between: an op's
`input` parameters carry their bindings (so the op IS its inputs), the output schema is the op's
`output.schema`, and `workspace`/`timeoutMs`/`maxCostUsd`/`abortSignal`/`policy` are `ExecServices`
seams (§3.2).

```ts
/** What execution RETURNS — the value-or-failure envelope every layer shares, with its failure type
 *  pinned to the classified `Failure` and its metrics to `ExecMetrics`. NEVER thrown for a unit
 *  failure: always returned, and the failure branch may still carry a partial `value`. */
type ExecResult<O = ResolvedValue, M extends ExecMetrics = ExecMetrics> = ResultWithMetrics<O, Failure, M>;

// The shared envelope, declared once in `json` (§2.2):
type Result<S, E>               = { value: S } | { error: E; value?: S };  // success has NO `error` key
type ResultWithMetrics<S, E, M> = Result<S, E> & { metrics: M };
function isOk<S, E>(r: Result<S, E>): r is { value: S };                   // the narrowing check

/** What EXECUTION measures — timing and a child-call count, and nothing else. No tokens and no money:
 *  those belong to whatever ran (llm's `LlmMetrics`), and a richer `M` satisfies this floor
 *  structurally while adding its own fields. */
interface ExecMetrics {
  durationMs: number;               // wall-clock duration of the execution, ms
  startMs?: number;                 // when it started (ms epoch)
  childLlmCalls?: number;              // LLM calls a composite's children fanned out to, rolled up (a prompt op is one; a non-LLM function is none)
}

/** The classified failure — declared in `json`, and the SAME value for an llm call, an execution, and
 *  a stored record: it is not execution-specific, and it does not live in `exec`. */
interface Failure {
  classification: ErrorClass;
  reason: string;                   // the REAL underlying cause, never "retries exhausted"
  retryAfterMs?: number; rateLimited?: boolean;
}
type ErrorClass = "network-retriable" | "api-retriable" | "permanent" | "deadline"
                | "out-of-credits" | "canceled" | "policy-denied";
```

**Reading a result is `isOk(r) ? r.value : r.error`.** The success branch has no `error` key at all, so
`if (r.error)` does not compile — the narrowing checks are `isOk(r)` and `"error" in r`. A failed call
still keeps its partial `value`, which is what makes a failure diagnosable rather than empty.

**Execution returns only the op's output value.** What a *model* produced — `rawText`, `thinking`,
`toolCalls`, `finishReason` — is `@declarative-ai/llm`'s `LlmOutput`, and it stops at `promptop`: none
of it rides on an `ExecResult`, because none of it is meaningful for a function op. A caller who wants
the trace asks the llm layer, which is the layer that has it (§4.1).

**There is no `artifacts` channel.** A produced artifact is a `blob`-kind output slot like any other,
not a parallel output channel (§3.7).

`ExecResult` is generic in the produced value type, defaulting to `ResolvedValue` (JSON, or the bytes of
a blob slot) — never `unknown`.

`ErrorClass → action` mapping (halt / broken / defer / bad-draw) stays with
each caller (findmyprompt's evaluator, JaiRA's app); the *classification* is
shared so both callers reason over one vocabulary.

### 3.2 Executor and Events

```ts
// Generic in R — the env still REQUIRED at start(). Bare core + registry use the default (all seams
// optional); composing a ctx-reading wrapper NARROWS R so a stack's start demands exactly what it consumes.
interface Executor<R = ExecServices, M extends ExecMetrics = ExecMetrics> {
  readonly capabilities: Capabilities;      // the SAME total record a `runtime` registry entry carries
  readonly metrics: MetricsAlgebra<M>;      // how its measurements combine (across retries, child→parent)
  capabilitiesFor?(op: Operation<InlineFamily>): Capabilities;  // a DISPATCHER's per-op record, when its
                                            // entries differ; absent ⇒ `capabilities` is total for every op
  start(op: Operation<InlineFamily>, ctx: R): ExecHandle<ResolvedValue, M>;
}
// Every wrapper is `withX(config, inner?)`: `config` mirrors the ctx SEAMS it reads; providing a seam there
// drops it from what `.start` requires (`Omit`-tracked). Two ways to stack (identical nesting): direct
// nesting `withMemoize({cache}, withDeadline(core))` (inner as the last arg), or the inside-out builder
// `compose(core).with(a).with(b)` (type-tracks requirements). `composeExecutors(core, …)` is the loose
// variadic convenience. A wrapper is `ExecutorWrapper<RIn, ROut> = (Executor<RIn>) => Executor<ROut>`;
// `withDeadline()` needs `{ deadline, stepStartMs }` at start, `withDeadline({ deadline })` only `stepStartMs`,
// `withDeadline({ deadline, stepStartMs })` neither.
// The capability record is `ops`' — an executor IS what a `runtime` registry entry delegates to, so the
// two share ONE record and cannot drift. Required and TOTAL, never all-optional.
interface RuntimeCapabilities {
  structuredOutput: boolean;        // native schema-constrained output
  sessionResume: boolean;
  streaming: boolean;
  interactive: boolean;             // needs a human/renderer — search callers refuse it up front
  readOnly: boolean;                // what the read-only/plan profiles gate on
  mutatesWorkspace: boolean;        // requires ctx.workspace; memo key must include treeHash
  policyEnforcement: "callback" | "config" | "none";
  memoizable: boolean;              // sound under §3.4 keying
  runtime: "edge-safe" | "node";
}
type Capabilities = RuntimeCapabilities;

interface ExecHandle<O = ResolvedValue, M extends ExecMetrics = ExecMetrics> {
  events: AsyncIterable<ExecEvent>; // SINGLE-CONSUMER: events are delivered, not broadcast — attaching a
                                    // second iterator throws rather than splitting or hanging the stream
  result: Promise<ExecResult<O, M>>;  // resolves when done; never rejects for a unit failure
  cancel(): Promise<void>;          // genuinely stops the operation and settles `result` (a `canceled`
                                    // failure unless it finished); equivalent to aborting ctx.abortSignal
}
type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: JsonValue }         // transcript stream
  | { type: "child_result"; ref: { label?: string }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: JsonValue }  // process units
  | { type: "command_result"; decision: "allowed" | "blocked" | "approved" | "denied" }
  | { type: "output_partial"; text: string };
```

`ExecServices` is the injected seam bundle — the shared-library descendant of
findmyprompt's `RunCtx`, reduced to what *execution* (not search) needs. It is **augmentable**: each
optional package declares its own seam by declaration merging, so `exec` names only what it itself
understands:

```ts
interface ExecServices {
  meter?: BudgetMeter;              // reserve/settle/available (WalletMeter shape)
  validator?: OutputValidator;      // json's three-line seam; validate's SchemaValidator implements it
  clock?: Clock; deadline?: DeadlineConfig;   // deadline consumed by the withDeadline wrapper
  stepStartMs?: number;             // step-start origin for deadline arithmetic
  executor?: Executor;              // composite ops execute children through this
  tools?: Record<string, Tool>;     // executables the current operation may call mid-loop
  sessions?: SessionStore;          // run-scoped, logical-id-keyed
  workspace?: Workspace;            // { root, treeHash? } — a Session-owned resource
  timeoutMs?: number;               // per-call wall-clock budget
  maxCostUsd?: number;              // per-call cost ceiling
  abortSignal?: AbortSignal;
}

// Declared elsewhere, by the package that owns the concept (§2, "augmenting ExecServices"):
declare module "@declarative-ai/exec" {          // in @declarative-ai/permissions
  interface ExecServices { policy?: ExecPolicy; approve?: Approver }
}
declare module "@declarative-ai/exec" {          // in @declarative-ai/promptop
  interface ExecServices { modelRouter?: ModelRouter }
}
// Rate limiting is NOT a ctx seam: it is construction-injected into the withRateLimit wrapper
// (composition encodes the policy; the bare core refuses unconsumed wrapper fields).
```

### 3.3 Interactive functions

The unification of "workflow UI states" and "search can't wait for humans". There is **no separate
interaction port**: an interactive UI component is simply a registry entry of kind `host` whose
capabilities declare `interactive: true` — it renders and awaits human input, and its promise is
treated like any other async value in the dataflow.

```ts
// One registry — a plain Map from `functionRef` to entry — with entries discriminated by HOW they run
// and capabilities REQUIRED and TOTAL per variant, so nothing can be "registered but uncharacterized".
type RegisteredFunction<Ctx, M> =
  | { kind: "pure";    impl: FunctionImpl;              capabilities: PureCapabilities }
  | { kind: "host";    impl: AsyncFunctionImpl<…, Ctx>; capabilities: HostCapabilities }
  | { kind: "runtime"; impl: AsyncFunctionImpl<…, Ctx>; capabilities: RuntimeCapabilities };

interface HostCapabilities { interactive: boolean; readOnly: boolean; memoizable: boolean }

registry.functions.set("choose_option", hostFunction(impl, { interactive: true, readOnly: true, memoizable: false }));
```

Errors are **data**: an impl RESOLVES a `Result` (`{ value } | { error: Failure; value? }`, plus an
optional `metrics` report) rather than throwing, so a 429 raised inside a function impl carries its
classification to the retry machinery instead of being reconstructed from `err.name`. `liftThrowing`
is the documented `catch` fallback for impls that throw anyway, and `runFunction` applies it at the one
place every dispatch goes through.

JaiRA registers its real renderer-backed implementations. findmyprompt registers **scripted
functions** whose responses come from the dataset item's fixtures, or — default for search — a
function that rejects, making any interactive state a `permanent` failure (score 0). Search contexts
therefore require either non-interactive definitions or a fixture script; both are checked before
spending money, by reading the resolved entry's `interactive` capability. The approval-gate security
property (SPEC §11.4) is preserved because registration is host-side: agents inside a unit never hold
the renderer.

### 3.4 Memo Keying

`@declarative-ai/exec` exports the canonical key function, over hashing primitives that live in
`@declarative-ai/json` (extracted from findmyprompt `artifacts/hash.ts` + `canonicalize.ts`,
RFC 8785 JCS + sha256):

```text
memoKey = sha256(canonicalize({
  operationHash,                           // hashOperation(op) — the op IS its inputs
  workspaceTreeHash?,                      // folded in whenever there IS a workspace;
                                           // REQUIRED for a `mutatesWorkspace` executor
  executorId?                              // WHO answered — so two executors sharing one cache (different
                                           // routing/registry, a real one and a stub) never collide
}))
```

That collapse is what the single execution seam buys: a **resolved op's `input` parameters carry
their bindings**, so the op already embeds the values a run was given. There is no `definition` +
`inputs` pair to hash separately and nothing asserting serializability on the caller's behalf. A blob
leaf is replaced by the hash of its bytes; a live stream throws with the remedy rather than silently
keying on object identity (§10.1).

`identify(op)` is the seam for an op with a cheaper or more canonical identity, so no caller computes
a hash it may never use. Nondeterminism is the caller's concern via findmyprompt's
existing unhashed `runId` draw-scope token — the library only guarantees the key
excludes it. For a hierarchical workflow, that identity **is** the workflow snapshot
hash folded with the op's own hash (`workflowIdentify(definition)`) — one identity for
"this exact workflow version and these inputs" across both consumers, passed as
`withMemoize({ cache, identify: workflowIdentify(definition) })`. `withMemoize` supplies the `executorId`
itself (a per-inner-executor token by default; a stable string for a durable cache). It also
**deduplicates in-flight**: N identical calls issued together share one execution rather than each
missing the not-yet-written cache, and a cache **hit reports zero** metrics, so an outer retry or budget
fold does not re-charge the time and money of the run that filled the cache.

### 3.5 Time budgets: `timeoutMs` vs `deadline` vs `withDeadline`

These three look redundant but sit at three different layers. The distinction only
matters for multi-step serverless runs; for a single call the deadline collapses
entirely into `timeoutMs` (see the note at the end).

- **`timeoutMs` — the primitive.** A per-call wall-clock *duration*, applied as
  `AbortSignal.timeout(timeoutMs)` at the actual provider call. "Cut off *this one call*
  after N ms." It appears at two layers that collapse into one number: the definition's own
  `timeoutMs` (serializable, baked into the "what") and `ctx.timeoutMs` (the per-execution limit,
  which a caller — or `withDeadline` — passes as the argument). Resolution is
  `argument ?? def.timeoutMs ?? DEFAULT`: the call-site value wins, because it is the clamp. With one
  field on `ExecServices` rather than one on a spec AND one on a definition, there is no longer a
  "definition budget above the spec limit" conflict for the core to refuse.

- **`ctx.deadline` (+ `ctx.stepStartMs`) — the window budget.** *Not* a duration for one
  call: "this serverless invocation started at `stepStartMs` and must finish by
  `stepStartMs + maxDurationMs − safetyMargin`, or the platform hard-kills it." It is
  **dynamic** (depends on `now`) and **shared** across every step in the invocation. The
  bare prompt executor does not interpret it — it **refuses** it (loud-failure discipline),
  because turning a window ceiling into a call action is a separable concern.

- **`withDeadline()` — the adapter.** The only thing that reads `ctx.deadline`/
  `ctx.stepStartMs`, translating the window budget into call behavior. It (1) **fails fast**:
  if `remainingMs = stepStartMs + maxDurationMs − margin − now` is below `floorMs`, it does
  NOT start the call and returns a `deadline`-classified failure so the caller *yields* to
  the next window (§time-vs-money) — which a plain `timeoutMs` cannot do (it would start a
  doomed call and let it get hard-killed mid-flight, losing the partial and the graceful
  classification); and (2) **enforces** the remaining window on the call itself — it clamps
  `ctx.timeoutMs` to `remainingMs`, and because that number is only advisory to an inner executor that
  may not read it, it also races the in-flight call against the window and cancels it with a `deadline`
  failure if it overruns, rather than trusting the clamp alone.

| field | answers | shape | applied by |
| --- | --- | --- | --- |
| `timeoutMs` | how long may *this call* run | absolute duration, static | always (abort signal) |
| `deadline` | when must the *whole window* end; worth starting? | relative to shared origin, dynamic | only via `withDeadline` |
| `withDeadline` | translate the window into a start/skip gate + an enforced cutoff | wrapper | you, by composing |

**Why it feels redundant (and when it genuinely is):** after `withDeadline` runs, the
deadline has been reduced to one remaining-window number — clamped onto `timeoutMs` and enforced as an
in-flight cutoff — so at the moment of the call there is just one bound. The distinction earns its keep
only through (1) the fail-fast/yield
decision and (2) the shared origin across many steps. If you make a single call and don't care
about yielding, `deadline` collapses into `timeoutMs` and you don't need it. `withDeadline`'s static
config **can** now be supplied at construction — `withDeadline({ deadline })` leaves only
`stepStartMs` (genuinely per-execution) required at `start`, and the builder tracks that in the type.
One cleanup is noted for later (not done): whether `timeoutMs` belongs in the serializable
*definition* at all, since by the "declaration is pure" rule it is arguably an execution concern.

### 3.6 The session store seam

One optional environment seam (`@declarative-ai/exec` `contract.ts`), kept as a pure interface so the
engine, the LLM layer, and consumers all share it:

```ts
// Mutable, keyed by a LOGICAL session id — a conversation's transcript, or a provider handle.
interface SessionState<Msg = JsonValue> { messages?: Msg[]; providerSessionId?: string; }
interface SessionStore<Msg = JsonValue> {
  get(logicalId: string): SessionState<Msg> | undefined | Promise<SessionState<Msg> | undefined>;
  put(logicalId: string, state: SessionState<Msg>): void | Promise<void>;
}
```

- **Optional.** An absent seam means the capability is simply unavailable: a `sessionId` with no session
  store is an error at resolve/execute time — never a silent no-op (loud-failure discipline).
- **`SessionStore`** backs the client-managed conversation model of §1.6. The `withSession` wrapper resolves
  the logical id against it, prepends the stored transcript to the new turn, and folds the successful reply
  back. A workflow injects a **run-scoped** store via `ctx.sessions` so states sharing a `sessionId` continue
  one conversation (§7); an app-provided store takes precedence. `MapSessionStore` is the bundled in-memory
  implementation; apps supply durable ones.

**There is no blob store.** Binary data is a leaf VALUE, so hydration is the ref family's business — the
same as `text` and `json` — and a separate injected store beside that would be a second mechanism doing
the first one's job. There is no reference form to resolve: the library takes bytes, a base64 string, or
a URL, and a caller holding a content hash or a filesystem path resolves it **before** calling. That is
what keeps `json`, `ops`, and `llm` free of `fetch` and `node:fs`.

The other Session-owned resource is the **workspace** — `{ root, treeHash? }` on `ctx.workspace`. Two
plain fields, no filesystem: `root` is what every consumer needs and `treeHash` is what memoization
needs, while the fs-backed tools that actually read the directory live in `@declarative-ai/tools`.

### 3.7 File I/O and modality gating

Media is a first-class, serializable part of a declaration, closed symmetrically on input and output:

```ts
// A neutral file/media input. Exactly one source of bytes — resolved by the CALLER (§3.6).
interface FileInput {
  mediaType: string;              // IANA type: application/pdf, image/png, audio/mp3, …
  filename?: string;
  data: Uint8Array | { base64: string } | { url: string };
}
```

- **Input.** `attachments: FileInput[]` on the declaration are lowered to provider file/image message parts
  and merged into the user turn at the call boundary. Raw bytes, inline base64, and a URL the provider
  fetches are the three forms; there is no store to consult and no reference form to resolve.
- **Output.** A model-generated file lands in a **`blob`-kind output slot** — `outcome.value` carries the
  bytes when the op's output is a blob. There is deliberately no parallel `artifacts` channel: a produced
  artifact is an output slot like any other. `kindFor` derives `blob` from JSON Schema's own
  `contentEncoding`/`contentMediaType`, never a bespoke marker. Binary still bypasses the Ajv boundary.
- **Streams.** An inline-family blob leaf holds `Uint8Array | ByteStream`, and streaming is an
  input/output optimization that is **never part of a definition** — an authored document is JSON, so a
  stream only ever appears in a runtime value. Automatic materialization of a stream into bytes is *not*
  implemented (§10.1): the paths that need bytes raise instead, so the caller materializes first.
- **Modality gating at `plan`.** Each input is gated by the modality its media type requires
  (`image/*` → `image`, `audio/*` → `audio`, `video/*` → `video`, else `file`) against the model's
  `modalities.input`; each requested `outputModalities` entry against `modalities.output`. A text-only model
  handed an image input is an `issues` entry **before** you spend, not a runtime rejection after.

## 4. What Executes

There is one `Executor` interface and one payload (`Operation`), so "unit kinds" are gone as a
taxonomy. What remains is *which executor an op reaches*: a `PromptOp` reaches the prompt executor, a
`FunctionOp` reaches a registry entry. The sections below are the concrete implementations.

### 4.1 One structured LLM call (`@declarative-ai/llm` + `@declarative-ai/promptop`)

The extracted findmyprompt leaf, behavior-identical at the wire. It splits across two packages, and
the split is what makes `llm` usable on its own:

- **`@declarative-ai/llm` is `exec`-free.** `executeLlmCall(definition, environment)` →
  `generateStructured` is the whole direct path, including schema adaptation, reasoning adaptation,
  param filtering, streaming, cache-split token accounting, and provider-reported cost. It knows
  nothing about `Executor`, `ExecHandle`, or wrappers — a structured call runs with `json + llm` and
  nothing else, and that is asserted by test.
- **`@declarative-ai/promptop` is the `Executor`.** It owns the `PromptOp → LlmCallDefinition`
  lowering and the class that applies it. Capabilities: `structuredOutput: true, streaming: true,
  interactive: false, readOnly: true, mutatesWorkspace: false, memoizable: true, runtime: edge-safe`
  (the undici long-timeout dispatcher install is node-only and conditional).

The declarative model of §1 lands entirely here:

- **The bare core (`createPromptExecutor`) is minimal** — lower the op, resolve the model, run one
  structured call, map the result onto an `ExecResult`, honor caller cancel, and *refuse* any unconsumed
  wrapper field. Repair, rate limiting, deadline fail-fast, sessions, budget, and memoization are the
  composable wrappers of §3.2, not core behavior — the generic ones in `exec`, the llm-aware ones
  (`withRateLimit`/`withBudget`/`withSession`) in `promptop`.
- **The environment is `LlmCallEnvironment`** (§1.2): `modelRouter` (the one near-floor seam),
  `validator`, `toolExecutors`, plus an `abortSignal`. `executeRequest(req)` is the convenience over
  the whole thing — a full declaration with `env` attached, split back apart before execution.
- **Two call shapes plus an environment.** A call is a `PromptOp` (the op-graph form) or an
  `LlmCallDefinition` (the direct form), each run against a separately-passed environment. The output
  `schema` lives IN the definition (via `CallSignature`) because it is declarative and serializable —
  keeping it out would force the lowering to smuggle it alongside and cast the phantom away. `timeoutMs`
  is a call-site *argument*, not a declaration field, because it is a clamp. A TEXT-mode call yields
  `LlmCallResult<string>`, not `LlmCallResult<JsonValue>` — a text call produces text.
- **`resolveConfig(layers)` composes the declaration** (§1.5): merge `[engineDefault, workflowDefault,
  registry.get(configRef), inline]` family-aware, split out the `SIGNATURE_KEYS`
  (`system`/`prompt`/`messages`/`attachments`/`timeoutMs`/`schema`) so the config bag parses strictly,
  then layer them back on. It returns ONE `LlmCallDefinition` — config knobs and signature together,
  which is possible because the config module lives *in* `llm` and can name the AI-SDK prompt types
  precisely. A `ConfigurationRegistry` (e.g. `MapConfigurationRegistry`) resolves named presets;
  identity remains the resolved content hash.
- **`plan(declaration)` is the dry run** (§1.3): resolve + content-hash identity + structured-output
  enforcement tier + param/modality fit + token/table-cost estimate, all from the local catalog — no
  network, no spend. It uses the *same* acceptance gate `executeLlmCall` filters with, so plan and
  execute cannot drift.
- **Routing is explicit.** Model ids are route-prefixed `{route}/{model}` (`anthropic` = native Anthropic,
  `openrouter` = everything else); a bare id is a fail-fast error, never guessed. The `ModelRouter`
  (`createModelRouter`) creates provider clients lazily, so a process that only calls one route never needs
  the other's key.

### 4.2 Hierarchical workflows (`@declarative-ai/hw`)

The shared state-machine engine and its executor. See §7.

### 4.3 Multi-step LLM workflows (findmyprompt)

**Not a new kind of anything.** findmyprompt's content-addressed `Operation` graphs
keep executing through `runOperation` natively, with each `PromptOp` leaf
dispatched to the shared prompt executor. Since the ops redesign the
`Operation` VOCABULARY is shared (`@declarative-ai/ops` at `IdFamily`) even though the
execution is not — so the two systems now describe operations in one language.
Where a *hierarchical* pipeline (guarded transitions, retry loops, sub-states)
is wanted, that is a workflow definition — searchable and serverless-safe today.

### 4.4 Delegated agents (`@declarative-ai/agents-api`, `@declarative-ai/agents-cli`)

There are two strategies for reaching agentic behavior, and a workflow author chooses **per operation**
which one they want:

| Quality | **Composed** (a prompt op with tools) | **Delegated** (`claude-code`, `opencode`) |
| --- | --- | --- |
| Owns the multi-turn loop | No — we drive a bounded tool loop | Yes — the agent loops internally |
| Operates on a workspace | Only via the tools we give it | Yes (a cwd + its own permissions) |
| Emits tool calls we can see | Yes, each in-loop call | Mostly no (a black box) |
| Tools | Our impls, run in-loop | Its built-ins by name, or ours, MCP-injected |

Two consequences are worth stating outright. **CLI is a transport, not a category** — a command-line
invocation is sometimes a plain function (`git status`, a linter) and sometimes the launch mechanism for
a delegated agent, so "CLI" never appears in the taxonomy; and **a prompt op with reasoning, tools, and
a step budget already *is* an agent**, because the bounded tool loop is the same machinery. Closing the
gap between composed and delegated is therefore mostly wiring and supplying tools, not building a second
loop — which is what makes "one model for coding, a second for review, a third for Q&A" a composition
choice rather than an integration project.

The adapters are built, and split by **invocation mechanism** rather than gathered into one `agents`
package:
`agents-api` reaches an agent through an in-process SDK (an optional peer on
`@anthropic-ai/claude-agent-sdk`), `agents-cli` through a CLI subprocess. Both drive the same
normalized `AgentQuery` seam and produce the same shape of `runtime` registry entry, so a workflow
authored against one runs against the other. Both declare `policyEnforcement: "callback"` — the SDK path
routes each tool approval back through `ctx.approve` in-process, and the CLI path does the same over an
MCP bridge (`--mcp-config` + `--permission-prompt-tool`), because the *guarantee* is what the capability
describes, not the mechanism. `CLI_CONFIG_ONLY_CAPS` (`policyEnforcement: "config"`) is the honest record
for a caller that deliberately runs with no approver, on an up-front posture alone. Both are
`mutatesWorkspace: true, runtime: node`.

The canonical policy *model* lives in `@declarative-ai/permissions`; policy authoring and the approval
UI stay in JaiRA. findmyprompt adopts the adapters unchanged when it gains a worker host; until then
its registry simply doesn't register them.

## 5. Cross-cutting services

Extracted with minimal edits; all already interface-coupled. Each lives in the package that owns the
concern, and none of them is barrelled into a shared "services" module — that barrelling is exactly what
would put ajv back into `llm`'s module graph:

- **Validation** (`@declarative-ai/validate`) — `SchemaValidator` (ajv, with an injectable `$ref`
  resolver), the structural subtype checker, and the ONE generic binding checker parameterized by ref
  family. `exec` and `llm` consume only `json`'s three-line `OutputValidator` seam, so neither learns
  about ajv. **Repair** is no longer a separate loop: it is the `validation: { feedback: true }` axis
  of `withRetry` (§3.2), off by default so findmyprompt's "bad-draw scores 0, never re-roll" semantics
  are preserved; JaiRA turns it on.
- **Retry** (`@declarative-ai/exec`) — budget-gated, `retryAfterMs`-honoring backoff arithmetic plus
  the unified `withRetry` wrapper. Now that a function impl resolves a *classified* failure, this
  reaches function ops too — a 429 raised inside a registered async function is retried instead of
  being blanket-`permanent`.
- **Rate limiting** (`@declarative-ai/exec`) — `ConcurrencyLimiter`, `TokenBucket`,
  `AdaptiveRateController` (AIMD), model/provider limit resolution.
  Generalizes per-model → per-executor-pool by key. The prompt-aware `withRateLimit` wrapper and the
  token estimate it prices on live one layer up (`promptop` / `llm`).
- **Metering** (`@declarative-ai/exec` + `promptop`) — the `BudgetMeter`/reservation interfaces, and
  the `withBudget` wrapper that reserves before a call and settles after. Implementations
  (Stripe/ledger, local budgets) stay app-side.
- **Permissions** (`@declarative-ai/permissions`) — profile × mode resolution, the scope chain, and
  the `withPermission` tool wrapper. It augments `ExecServices` with `policy`/`approve`, so `exec`
  does not know that permissions exist. The model is §5.1.

### 5.1 Tools, sessions, and permissions

One primitive underlies all three: **name → binding, resolved through a scope chain**. The same shape
appears at every layer, which is why they are described together rather than as three subsystems.

#### Functions and tools

- A **function** is named host code, `inputs → output`, sync or async. It can be anything, and it is
  graph-invokable as a `FunctionOp`.
- A **tool** is a function *plus* the call-metadata a model needs to decide to call it: a `description`,
  an `inputSchema`, and a `readOnly` declaration. So **`Tool ⊂ Function`**, and one impl can be surfaced
  either way — as a graph op or as an agent tool — from one schema, with no drift.

Because a tool *is* a function, a tool's body may itself invoke another operation or a sub-workflow;
tools are not restricted to leaf host code. `@declarative-ai/tools` ships the Claude-Code-parity set
(`read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, `glob`, `run_command`) — these are the
impls that turn a composed prompt operation into a coding agent, and they are **where the workspace
lives**: handing an agent "a workspace" is really handing it these tools, closed over `ctx.workspace`.

One asymmetry is deliberate: a tool's `run` **may throw**, because a tool failure travels back to the
*model* as a result it reads and reacts to. That is not the classified-failure channel a `FunctionOp`
impl uses (§3.3), and conflating them would surface a model-visible tool error as an execution failure.

#### The scope chain: static registry, run-scoped overlay

Two containers of the same shape but different lifetime and mutability:

- **Static base — the `CapabilityRegistry`**: `{ functions, skills, tools }`, three plain maps.
  Declaration-time, host-provided, shared across many runs, effectively immutable during a run.
- **Run overlay — the environment**: mutable, run-scoped state keyed by `sessionId`.

Name resolution walks **overlay → base** (local shadows global). That is what makes a rebind safe: it
writes to the overlay, so it is automatically run-scoped and can never leak into another run or mutate
shared host config. This lines up exactly with the declaration/environment split of §1.2 — the registry
is *declaration*, the overlay is *environment*.

#### Sessions: the run-scoped resource bundle

The run-scoped identity is a **session**, keyed by `sessionId`, owning
`{ conversation, workspace, permission bindings, tool renames }`. Operations sharing a `sessionId` share
the whole bundle — the same lever as sharing a conversation, generalized from "transcript" to
"transcript + workspace + permissions + tools".

- A session is strictly more than a transcript: the conversation is one facet of it.
- **Sharing is explicit, isolation is the default**: an operation's `environment.session` names an id;
  the same id means a shared session, absent means the run's default session.
- **The workspace is session-owned, not runtime-owned**, and not always shared. It is default-shared
  within a subtree — a review agent reading what a coding agent wrote is the point — and overridable to
  isolate (a parallel fan-out into worktrees). Two different runtimes sharing one workspace is common
  and correct.

#### Tool renames are just overlay bindings

The **logical tool name is the vocabulary**: an operation says it may use `read_file` and `run_command`,
and the same name means the same thing everywhere. Per-runtime differences are handled by binding that
logical name in the session-scoped overlay, so a rename is not a separate map type — it is one more
name → binding, whose value is `Tool | NativeToolRef`:

- Composed: `read_file → our impl`, run in the bounded loop.
- Delegated, native tool: `read_file → { native: "Read" }` — a redirect the adapter translates into the
  agent's alias/allow-list, because we cannot execute the agent's built-in ourselves.
- Delegated, our behavior: `read_file → our impl`, MCP-injected, so `run_command` on a delegated agent
  runs the same code as on a composed one.

`native` vs `mcp` is therefore a delegated-only concern; MCP is purely the transport for pushing our
tools into a black box.

#### Permissions: two orthogonal axes

- **Profile** — *which effects are in scope*: `read-only`, `plan`, `full`. Out-of-profile tools are
  denied outright, before mode is even consulted. Any other name is a **custom** profile resolved
  through a host-supplied `ProfilePredicate` (e.g. a `search` profile admitting only `grep`/`glob`); an
  unknown custom name admits nothing, which is the safe default.
- **Mode** — *how an in-scope call is authorized*: `allow`, `deny`, `ask`, `smart`.

This generalizes Claude Code's flat mode list rather than copying it: `default` = full × ask,
`acceptEdits` = full × smart, `plan` = read-only × ask + an exit gate, `bypassPermissions` = full ×
allow. **Plan mode is not a special case**: it is the read-only profile plus a human-gated transition
that rebinds the session's profile `read-only → full`, using the same approval-rebind mechanism as
everything else. A delegated runtime in `plan` sets its agent's native plan mode where it has one, and
routes that agent's exit gate back through our approval seam.

#### Enforcement

A wrapper around tool execution decides each call:

```text
tool call → resolve the mode for (tool, session) through the scope chain →
  allow → run the tool
  deny  → return a "denied" tool-result to the MODEL (it keeps going; not terminal)
  ask   → invoke the approver (renders UI, awaits the human)
  smart → a bound arg-inspecting policy decides, or returns `ask` to escalate
```

Three properties make this sound rather than merely convenient:

- The approval request is **wrapper-internal and never exposed to the model** — a model must never be
  able to self-authorize. It reuses the name→function machinery only as the resolution mechanism.
- The approver **returns a decision as data** (`{ decision, scope }`) and the *engine* applies the
  rebind. The approver stays a pure input-collector and the mode change becomes an observable event
  rather than a hidden side-effect.
- An operation whose tools are in `ask` mode is **not soundly memoizable** — its behavior depends on live
  human input, exactly as an interactive function is impure. This is an authoring rule the entry's
  `memoizable` capability records; nothing derives it from the resolved mode automatically.

The gate is only active when the host supplies an approver; otherwise tools run unguarded.

#### Persistence granularity — a scope chain, all in memory

When a human authorizes, they choose how long it sticks. **Every one of these is ephemeral**; nothing
here crosses the durability boundary:

| Granularity | Written to | Lifetime |
| --- | --- | --- |
| `once` | nowhere | this call |
| `session` | the session overlay | this session |
| `workflow-run` | the run overlay | this run, across all its sessions |
| `always` | the process overlay | the host process, across runs — gone on restart |

Resolution walks most-specific → least, so a narrower decision shadows a broader one:

```text
session → workflow-run → process → definition-authored baseline → hard default (ask)
```

The process overlay lives in the *host*, above any single run's engine, and is threaded into each run.

Beneath the ephemeral overlays sits the one durable layer: the **definition-authored baseline** —
a profile plus per-tool modes, authored at two levels (a workflow default with a per-state override).
This is the only permission state that is *declaration* rather than *environment*, and it is deliberately
the only place a cross-run decision can live. "Remember this forever" means editing the definition, not
clicking a permission dialog; there is no durable cross-run policy store in the decision path.

#### Delegated approval fidelity

A delegated agent routes its own loop's tool calls back through **our** approver via the SDK's
permission hook (a `canUseTool`-style callback) or, for the CLI transport, over an MCP bridge — so the
ask-UX is uniform regardless of which runtime executes. The engine hands such an entry (one declaring
`policyEnforcement: "callback"`) **raw** tools, with no engine-side wrapper, so authorization flows
through exactly one gate and injected tools are never double-gated. Where an agent offers no hook, the
fallback is to translate the policy into its native permission config — that is what
`policyEnforcement: "config"` honestly records, and it is a weaker guarantee, not an equivalent one.

## 6. Extraction Map (from findmyprompt `src/engine/`)

**Moved here** (mostly as-is; adapted imports only):

| From (findmyprompt) | To | Notes |
| --- | --- | --- |
| `providers/generate.ts` | llm | the one LLM call site |
| `providers/router.ts`, `dispatcher.ts` | llm | `ModelRouter`, lazy clients, undici dispatcher |
| `providers/structured.ts`, `reasoning.ts`, `providers/schema/*` | llm | provider divergence adapters + capability profiles |
| `providers/model-catalog.ts`, `model-catalog-source.ts`, `registry/providerConfig.ts` | llm | catalog; refresh-from-network part node-only |
| `execution/llmStep.ts` | llm | becomes `executeLlmCall` (`call.ts`) |
| `execution/retry.ts`, `providers/retry-policy.ts` | exec | |
| `execution/concurrency.ts` | exec | |
| `execution/deadline.ts` | exec | |
| `execution/errors.ts` | json | `EncodedError` + classification |
| `execution/costEstimate.ts` | llm | it prices a PROMPT's footprint |
| `schema/ajv.ts` | validate | store-backed `$ref` resolution stays injectable |
| `artifacts/hash.ts`, `artifacts/canonicalize.ts` | json | canonical serialization is definitionally a json concern; memo keying stays in exec |
| `config/llm.ts` | llm | `LlmConfiguration` hierarchy, `ReasoningSpec` — its only consumer |
| meter/scope/classification types from `execution/context.ts` | exec / json | interfaces only; the error + telemetry vocabulary lands at the bottom in json |
| `model/index.ts` | **ops** | the op MODEL, made generic over a ref family (§3.1); findmyprompt re-imports it at `IdFamily` |
| `execution/functionRegistry.ts` | **ops** | the registry, made generic in `Ctx`, entries now a discriminated union (§3.3) |
| `schema/subtype.ts` (+ its test suite) | **validate** | ported verbatim, then re-homed out of ops — a checker is not part of the op model |
| `schema/template.ts`, `infer.ts`, `selectType.ts` | **json** | pure schema/data transforms with no model dependency |
| `schema/signatureSchema.ts` | **ops** | the `Signature` ⇄ schema bridge |
| `schema/checker.ts` | **validate** | unified with hw's hand-rolled twin into ONE generic binding checker |

**Stays in findmyprompt** (the optimizer's domain): `runOperation.ts` and the rest of the EXECUTION
machinery, `evaluator.ts`, `aggregation.ts`, `stores.ts`, `searchResolvers.ts`, `yield.ts`,
`critique.ts`, all of `search/**`, `strategies/**`, `judge/**`, `meta/**`, the remaining `schema/*`
and `artifacts/*`, plus `src/server/**`, `db/**`. Note the split within `model/` and
`functionRegistry.ts`: the **vocabulary** moved to `ops`, the **execution** stayed — findmyprompt
imports the types and registry from `@declarative-ai/ops` and keeps running them itself.

## 7. The Hierarchical-Workflow Package (`@declarative-ai/hw`)

One package containing both the **engine** — state-file loader/validator,
expression language, evaluator, snapshot hashing per [SPEC.md](SPEC.md) — and
the **executor** exposing a workflow run as an `Executor`. The engine
keeps an injected-seams architecture (`Persistence` / `CapabilityRegistry` / the prompt `Executor`);
those injections are exactly the executor's constructor
arguments, and exactly what JaiRA's app supplies in its richer form.

- **Definition**: a bundle of state files (the SPEC format) + root state id, held at
  CONSTRUCTION rather than re-supplied per run — a workflow's identity is its snapshot, not a payload.
  Its snapshot hash (SPEC §12), folded with the run op's own hash, is the
  identity `withMemoize` keys on via `workflowIdentify(definition)` (§3.4). In findmyprompt the bundle
  is stored as `json_artifacts` rows; in JaiRA it comes from `.jaira/snapshots/<hash>/`.
- **Execution**: instantiate the engine with the bundled in-memory
  persistence (durable SQLite persistence is a JaiRA-side implementation of
  the `Persistence` port), a `CapabilityRegistry` of the named things states reference
  (`functions` / `skills` / `tools`), and the prompt `Executor` a `PromptOp` dispatches to. A run is
  started by a `FunctionOp` whose bound inputs are the workflow's declared inputs; terminal outputs
  become the `ExecResult`'s `value`. hw takes the prompt executor as a **plain `Executor`**, so it never learns
  that a prompt op has an llm lowering and the AI SDK stays out of its dependency graph.
- **Metrics rollup**: every child execution emits `child_result`; the
  executor folds child metrics (call counts, and spend where the metrics carry
  it) into its own so budget gates and wallet reconciliation see through the nest.
- **Cancellation/deadline**: `ctx.abortSignal` → engine cancel (cancels
  descendant operations per SPEC §10.4); a deadline hit yields a well-formed
  `ExecResult` with `classification: "deadline"` and partial metrics — never a
  hang.
- **Capabilities**: `interactive: true` (UI states supported when an interactive entry is
  registered); `mutatesWorkspace: false` today, to become true per-definition once process units
  exist. `memoizable: true` under §3.4 keying. `sessionResume: false` in v1.
- **Validation is a function of *(document, registry)*.** `validateBundle(bundle, env)` reads the
  registry it will run against, so "a `functionRef` naming nothing registered" and "an interactive
  function in a search-only workflow" are authoring errors caught before anything spends money. That
  is the deliberate consequence of capabilities being required and total on every entry (§3.3).

- **States declare their call through the same declarative pipeline.** A state whose operation is a
  `PromptOp` is dispatched to the injected prompt `Executor` — `@declarative-ai/promptop` — which runs
  the §1.5 `resolveConfig` pipeline per operation:
  `defaults ← configs.get(config.configRef) ← the state's inline config`, merged
  family-aware and strict-parsed. Each layer may carry signature fields (a shared `system` prompt, a
  per-state `timeoutMs`) alongside the config knobs; the rendered template becomes the operation prompt (a
  config-layer `prompt` is an error — there's nothing to do with two). So a workflow state's call is an
  `LlmConfiguration` declaration like any other, not a parallel config surface.

- **Sessions coordinate by logical id.** A state's `environment.session` is the sharing key for its owned
  resources — conversation transcript, workspace, and permissions (§5.1); absent ⇒
  the run's default session (so a plain workflow is one shared session). The engine keys the built-in
  `conversationMode` preamble per session, aligning it with the run-scoped `SessionStore` (§3.6, exposed as
  `ctx.sessions` for the llm `withSession` path) — both key on `sessionId`.

- **Tools and permissions.** An operation may be given **tools** (`registry.tools`, referenced by
  logical name in `environment.tools`) it calls mid-loop — a composed prompt operation runs them in its
  bounded tool loop; a delegated agent gets the allow-list. Each call is gated by a **profile × mode**
  permission system (`read-only`/`plan`/`full` × `allow`/`deny`/`ask`) whose human decisions persist across
  an in-memory scope chain (session → workflow-run → process) over the workflow-authored baseline. The full
  model — composed-vs-delegated runtimes, the `Tool` seam, the session-keyed environment overlay, and
  the permission granularities — is specified in §5.1.

## 8. Consumer Migration Plans

### 8.1 findmyprompt

1. Re-point the `RunCtx.executeCall` seam at the shared call: the default inline path and the WDK
   step both call `executeLlmCall(definition, environment)` (or, for the contract path, an injected
   `Executor.start(op, ctx)`) instead of findmyprompt's own `executeStructuredCall`. Pure refactor,
   gated on the existing test suite.
2. Replace the moved modules (§6) with `@declarative-ai/*` imports; delete the
   originals.
3. Replace `model/index.ts` and `functionRegistry.ts` with `@declarative-ai/ops` imports at
   `IdFamily` (+ `Ctx = RunCtx`). The property renames (`textId` → `text`, …)
   change the canonical serialized content and therefore the content ids: findmyprompt RESETS its
   storage and re-seeds. No compatibility layer, no dual canonical form (decision #7).
   `runOperation`, stores, search, and judges stay put — only the vocabulary is shared.
4. **No third op kind is needed.** What was sketched here as a composite `ExecOp` leaf is subsumed
   by a plain `FunctionOp` (§3.1): a sub-workflow, a composite unit, and a
   delegated agent runtime are all registered async functions, distinguished by the resolved
   registry entry's capabilities rather than by the op shape. A hierarchical-workflow candidate is
   a `FunctionOp` whose search space ranges over its definition bundle.
5. WDK note: such a step's replay-safety comes from the memo (§3.4), the same property `PromptOp`
   steps rely on today.

### 8.2 JaiRA

1. The engine is developed here as `@declarative-ai/hw`; JaiRA's repo keeps the app:
   Electron shell + renderer, durable SQLite `Persistence`, task/board model,
   policy authoring + approvals UI, Git worktree management, WSL exec layer.
2. JaiRA `DESIGN.md` §8's `RunnerAdapter`/`RunSpec`/`RunHandle` are this
   library's `Executor`/`Operation`/`ExecHandle`. The `llm_api` adapter
   is `@declarative-ai/promptop` over `@declarative-ai/llm`; delegated agents are
   `@declarative-ai/agents-api` / `@declarative-ai/agents-cli`.
3. The engine's `RunnerRegistry` port *is* the `CapabilityRegistry`'s `functions` facet; its `UiBridge`
   port is an interactive `host` entry registered there (JaiRA's renderer-backed implementation
   preserves the SPEC §11.4 approval-gate guarantee).
4. JaiRA design phases 1–3 (expression language, loader/validator, engine
   core) execute in this repo; JaiRA phases 4+ consume it.

## 9. Phasing

1. **Extract** — this repo; moved files with tests; findmyprompt's suite is
   the acceptance gate for the eventual swap (no behavior change intended). **Done.**
2. **Execution seam** — findmyprompt re-points its call site at the shared `Executor`; `@declarative-ai/hw`
   engine work proceeds here against the same op vocabulary.
3. **Hierarchical workflows** — the `@declarative-ai/hw` executor once the engine
   core is headless-runnable; findmyprompt registers it; first end-to-end:
   optimize a small prompt-only workflow's prompts against a dataset. **The executor is done**;
   findmyprompt's adoption is not.
4. **Delegated agents** — `@declarative-ai/agents-api` / `agents-cli`, built as JaiRA phases 4–7 need
   them; findmyprompt adoption waits on an execution host. **The adapters are done.**

## 10. Risks and Open Points

- **Contract churn across three repos.** Mitigated by sequencing: extracted
  from working findmyprompt code; JaiRA consumes types before writing
  executors. File-linked workspaces until the contract survives phase 3;
  publish then.
- **Content ids move when the op vocabulary does.** The ref property renames (`textId` → `text`, …)
  change the canonical serialized content and therefore every content id: findmyprompt resets its
  storage and re-seeds rather than carrying a compatibility layer. Needs focused review of
  findmyprompt `artifacts/factories.ts` at the swap.
- **Engine embedding assumptions.** `@declarative-ai/hw` must never acquire
  Electron/SQLite hard dependencies — better-sqlite3 stays behind
  `Persistence` (implemented JaiRA-side).
- **Repair vs findmyprompt eval semantics.** Repair is opt-in per composition
  (`withRetry({ validation: { feedback: true } })`, default off) to avoid perturbing its statistical
  discipline; JaiRA turns it on.
- **`interactive` fixtures under search.** Scripted interactive-function
  responses become part of the dataset item's identity; they must be included
  in the item's hashed content or scores aren't reproducible.
- **Spec/implementation co-location.** [SPEC.md](SPEC.md) here is the
  canonical formalism spec (see its preamble for which sections are normative
  for this library vs. JaiRA-product context).

### 10.1 Known limits and deferred follow-ups (none blocking)

The design above is implemented; these are the places where it is knowingly incomplete. Each is
non-blocking, and each lands green on `npm run typecheck` + `npx vitest run`.

- **Stream materialization is not implemented.** A blob leaf may hold a `ByteStream`, but nothing
  converts one to `Uint8Array` on demand. Three paths genuinely need bytes — hashing an op for a memo
  key, fanning one blob producer out to two consumers, and storing a result inline — and today they
  raise with the remedy rather than draining the stream. The intended semantics when it lands: an
  idempotent, in-place upgrade of the *runtime* value (never the definition), with the in-flight promise
  cached on the leaf so two consumers never both drain it, `abortSignal` tied to `reader.cancel()` so
  the source cannot leak, and a consumption failure failing the op immediately and non-retriably with
  the underlying message under a context prefix. Note fan-out is *statically* known — the validator
  already sees two consumers of one producer — so it is a bind-time decision, not a runtime discovery.
- **Shrink `ExecServices`** toward only what the bare core needs (the wrappers take their deps at
  construction now; the ctx bundle can lose the fields no core reads). The `ExecServices` augmentation
  mechanism (§3.2) is the first half of this — `exec` no longer *names* the optional capabilities.
- **Key `withMemoize` off `plan`'s resolved content hash** — unify the memo-key identity with the plan
  identity (§1.5, §3.4).
- **Fold tool-call / file outputs into the `withSession` transcript** (today it folds text + structured
  value only).
- **Forward `outputModalities` per-provider** (declarative + `plan` surfaces exist; the per-provider request
  wiring, like reasoning adaptation, does not).
- **The provider-side session path (`providerSessionId`) is unimplemented** — `withSession` refuses it,
  loudly, rather than silently starting a fresh conversation.
- **Register codecs for a real rich type** — the `x-type` / `Codec` machinery
  is in place with an empty `TypeRegistry`; no type has claimed a name yet.
- **Consumer-side `anyOf` in the subtype checker** — still the documented next addition, still unforced
  by any real workflow.
- **Verify provider tools + schema-with-tools against a live provider** (wired structurally, unverified).
- **The delegated-agent boundary mappings are unverified against the live SDKs.** `agents-api`'s message
  and `canUseTool` field names, and `agents-cli`'s subprocess/MCP mapping, are documented and tested
  against fakes but not build-checked against an installed `@anthropic-ai/claude-agent-sdk`. An
  `opencode` adapter is not built.
- **Whether `x-type` constraint checking wants variance.** A `DateTime` slot rejecting a bare-number
  producer is clearly right; whether a *wider* named type should accept a narrower one needs a real case
  before it is modeled.
- **hw's loader does not use `kindFor`.** `loader.ts`'s local `kindOf` defaults a string-typed schema to
  `text` without consulting `contentEncoding`/`contentMediaType`, so an artifact slot must currently
  declare `kind: "blob"` explicitly even though `ops`' `kindFor` would derive it. Routing the loader
  through `kindFor` is the fix.
