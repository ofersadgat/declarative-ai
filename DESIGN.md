# ai-exec: Shared AI Execution Library — Design

Status: v1 implemented — 2026-07-17 (canonical; supersedes `JaiRA/AI-EXEC-DESIGN.md`)

> **Implementation status.** `@ai-exec/core`, `@ai-exec/services`, `@ai-exec/llm`,
> and `@ai-exec/hw` are implemented and tested (296 tests, typecheck-clean),
> including the SPEC worked examples (§7.3, §9, §10.4) as golden tests.
> `@ai-exec/agents` is not started (deferred per §4.4). v1 notes: the hw engine
> executes runs in-process and emits a full `EngineEvent` stream through the
> `Persistence` port — step-level durable resume is future work
> (`sessionResume: false`); a ui state reached with no `InteractionPort`
> configured is run-fatal (a rejecting port stays a state-level failure), and
> the executor's `interactionPolicy: "eager"` refuses interactive definitions
> up front for search contexts; conversation mode `summary` currently degrades
> to `full_history`. Neither consumer has migrated yet: JaiRA's app build is
> next (JaiRA `DESIGN.md` §14), then findmyprompt's registry-seam swap (§8.1).
Consumers: **findmyprompt** (`C:\UbuntuCode\findmyprompt`) and **JaiRA** (`C:\UbuntuCode\JaiRA`).
Companions: [SPEC.md](SPEC.md) (the hierarchical-workflow formalism, migrated from JaiRA),
JaiRA `DESIGN.md` (the app on top of this library), findmyprompt `DESIGN.md`/`IMPLEMENTATION.md`.

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

## 2. Packages

```text
ai-exec/
  packages/
    core/        @ai-exec/core       edge-safe: types, contract, classification, hashing
    services/    @ai-exec/services   validation+repair, retry, rate limiting, metering
    llm/         @ai-exec/llm        the llm-call executor + provider layer (node & edge-capable)
    hw/          @ai-exec/hw         the hierarchical-workflow formalism: state-file
                                     loader/validator, expression language, evaluator
                                     engine, snapshot hashing, and its executor (§7)
    agents/      @ai-exec/agents     (future) process executors: agent-sdk, claude-cli,
                                     generic-cli — built when JaiRA needs them (§4.4)
```

```text
@ai-exec/core  ◄─  @ai-exec/services  ◄─  @ai-exec/llm
     ▲                                        ▲
     └───────────  @ai-exec/hw  ──────────────┘   (engine + hierarchical-workflow executor)
                        ▲
          ┌─────────────┴──────────────┐
     JaiRA app                    findmyprompt
     (SQLite persistence impl,    (ExecOp leaf, memo,
      Electron UI, policy          search strategies)
      authoring, worktrees)
```

Rules: `@ai-exec/core` has zero heavyweight runtime deps and no `node:*`
imports (findmyprompt's `hash.ts` discipline — `@noble/hashes`,
edge/Workflow-runtime safe). `@ai-exec/llm` depends on the `ai` SDK + provider
packages. `@ai-exec/hw` depends on core (+ services for validation) and
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

## 3. Core Contract (`@ai-exec/core`)

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
   *  units: a content-addressed definition bundle + its hash (§3.4). */
  definition: unknown;
  definitionHash: string;
  /** Named input values, resolved by the caller. Schemas live in the definition. */
  inputs: Record<string, unknown>;
  /** Workspace binding for units that read/mutate files. Absent for pure units. */
  workspace?: { rootDir: string; treeHash?: string };   // treeHash: git tree sha
  /** Session/conversation continuity token (provider session id, transcript ref). */
  session?: { id?: string; transcript?: unknown };
  /** Output contract: JSON Schema for the data payload; artifact targets for
   *  file outputs (engine-assigned paths). Validation executor-performed,
   *  caller-observable (repair loop, §5.1). */
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
interface Executor {
  readonly kind: UnitKind;
  readonly capabilities: ExecutorCapabilities;
  start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle;
}
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
  rateLimiter?: RateLimiter;        // extracted concurrency.ts (AIMD + token bucket)
  meter?: BudgetMeter;              // reserve/settle/available (WalletMeter shape)
  validator?: SchemaValidator;      // extracted ajv.ts
  clock?: Clock; deadline?: DeadlineConfig;
  providers?: ProviderRouter;       // for llm-backed executors
  registry?: ExecutorRegistry;      // composite units execute children through this
}
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

`@ai-exec/core` exports the canonical key function (extracted from
findmyprompt `artifacts/hash.ts` + `canonicalize.ts`, RFC 8785 JCS + sha256):

```text
memoKey = sha256(canonicalize({
  kind, definitionHash, inputs,            // inputs sorted by name
  workspaceTreeHash?                       // required iff mutatesWorkspace
}))
```

Nondeterminism is the caller's concern via findmyprompt's existing unhashed
`runId` draw-scope token — the library only guarantees the key excludes it.
For `hierarchical-workflow`, `definitionHash` **is** the workflow snapshot
hash computed by `@ai-exec/hw` — one identity for "this exact workflow
version" across both consumers.

## 4. Unit Kinds

### 4.1 `llm-call` (`@ai-exec/llm`)

The extracted findmyprompt leaf, behavior-identical: definition = the
structured call params (model, prompts, sampling, reasoning, timeout);
executor = the `executeStructuredCall` → `generateStructured` pipeline,
including schema adaptation, reasoning adaptation, param filtering, streaming,
cache-split token accounting, and provider-reported cost. Capabilities:
`structuredOutput: true, interactive: false, mutatesWorkspace: false,
memoizable: true, runtime: edge-safe` (the undici long-timeout dispatcher
install is node-only and conditional).

### 4.2 `hierarchical-workflow` (`@ai-exec/hw`)

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
(`mutatesWorkspace: true`, `runtime: node`) and built as `@ai-exec/agents`,
on JaiRA's schedule (its phases 4–7 are the driving consumer). The canonical
policy *model* and command-intent parsing land here with them; policy
authoring and the approval UI stay in JaiRA. findmyprompt adopts the
executors unchanged when it gains a worker host; until then its registry
simply doesn't register them.

## 5. Services (`@ai-exec/services`)

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

## 7. The `hierarchical-workflow` Package (`@ai-exec/hw`)

One package containing both the **engine** — state-file loader/validator,
expression language, evaluator, snapshot hashing per [SPEC.md](SPEC.md) — and
the **executor** exposing a workflow run as an execution unit. The engine
keeps an injected-seams architecture (`Persistence` / `ExecutorRegistry` /
`InteractionPort`); those injections are exactly the executor's constructor
arguments, and exactly what JaiRA's app supplies in its richer form.

- **Definition**: a bundle of state files (the SPEC format) + root state id,
  content-addressed; `definitionHash` = the snapshot hash computed here. In
  findmyprompt the bundle is stored as `json_artifacts` rows; in JaiRA it
  comes from `.jaira/snapshots/<hash>/`.
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

## 8. Consumer Migration Plans

### 8.1 findmyprompt

1. Introduce `ExecutorRegistry` and re-point the `RunCtx.executeCall` seam:
   the default inline path and the WDK step both call
   `registry.get("llm-call").start(...)` instead of importing
   `executeStructuredCall`. Pure refactor, gated on the existing test suite.
2. Replace the moved modules (§6) with `@ai-exec/*` imports; delete the
   originals.
3. Add the composite leaf: `ExecOp { kind: "exec"; unitKind; definitionJsonId;
   input; output }` alongside `PromptOp`/`FunctionOp`, with `runExecOp`
   building an `ExecutionSpec` and recording the `Outcome` as a
   `GenerationResult` (aggregate metrics; `child_outcome` events optionally
   persisted as child GRs).
4. Register the `@ai-exec/hw` executor; a hierarchical-workflow candidate is
   an `ExecOp` whose search space ranges over its definition bundle.
5. WDK note: an `ExecOp` step's replay-safety comes from the memo (§3.4),
   the same property `PromptOp` steps rely on today.

### 8.2 JaiRA

1. The engine is developed here as `@ai-exec/hw`; JaiRA's repo keeps the app:
   Electron shell + renderer, durable SQLite `Persistence`, task/board model,
   policy authoring + approvals UI, Git worktree management, WSL exec layer.
2. JaiRA `DESIGN.md` §8's `RunnerAdapter`/`RunSpec`/`RunHandle` are this
   library's `Executor`/`ExecutionSpec`/`ExecHandle`. The `llm_api` adapter
   is `@ai-exec/llm`; process executors are `@ai-exec/agents`.
3. The engine's `RunnerRegistry` port *is* `ExecutorRegistry`; its `UiBridge`
   port *is* `InteractionPort` (JaiRA's renderer-backed implementation
   preserves the SPEC §11.4 approval-gate guarantee).
4. JaiRA design phases 1–3 (expression language, loader/validator, engine
   core) execute in this repo; JaiRA phases 4+ consume it.

## 9. Phasing

1. **Extract** — this repo; moved files with tests; findmyprompt's suite is
   the acceptance gate for the eventual swap (no behavior change intended).
2. **Registry seam** — findmyprompt `ExecutorRegistry` + `ExecOp`; `@ai-exec/hw`
   engine work proceeds here against the same core types.
3. **hierarchical-workflow** — the `@ai-exec/hw` executor once the engine
   core is headless-runnable; findmyprompt registers it; first end-to-end:
   optimize a small llm-call-only workflow's prompts against a dataset.
4. **Process units** — `@ai-exec/agents`, built as JaiRA phases 4–7 need
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
- **Engine embedding assumptions.** `@ai-exec/hw` must never acquire
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
