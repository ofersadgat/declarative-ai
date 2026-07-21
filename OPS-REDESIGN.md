# Ops Redesign: the typed operation spine (`@declarative-ai/ops`)

> **Status: DESIGN — nothing implemented.** This document specifies importing findmyprompt's
> `PromptOp`/`FunctionOp` operation system (including its async-function registry) into this repo as a new
> package, and rewiring `core`/`llm`/`hw` so operations become the typed spine of the whole library.
> It supersedes DESIGN.md §6's line "model/, runOperation.ts, functionRegistry.ts stay in findmyprompt":
> the **model and registry** move here; the **execution machinery** (runOperation, stores, memo, search)
> stays in findmyprompt.

## Why

The current contract is stringly/loosely typed at every seam a consumer touches:

- `RuntimeOp.config: Record<string, unknown>` and `ExecutionSpec.definition: unknown` — the two payloads
  every executor reads are untyped (`core/src/contract.ts`).
- `Outcome.value?: unknown` — a caller that just executed a schema-validated call gets `unknown` back.
- `HostFunction.run(inputs: Record<string, unknown>)` and hw's `FunctionConfig` open index signature
  (`[key: string]: unknown`) — function inputs/outputs have no types at all.
- hw wiring is `WiringValue` expression **strings** (`"children.plan.outputs.steps"`) checked only by the
  expression parser, never against what the target slot expects (`hw/src/format.ts`).

findmyprompt already has the strongly-typed vocabulary this repo lacks: `PromptOp`/`FunctionOp`
(`src/engine/model/index.ts`) with `Parameter`/`Ref` bindings — every slot carries a `kind` + JSON Schema,
every wire is a typed binding, an op's parameter tree IS the graph, and a `FunctionRegistry` registers
both sync `FunctionImpl`s and async ctx-bearing `AsyncFunctionImpl`s. What it lacks — and what this port
adds — is **compile-time** typing: the runtime-schema system gets a TS-generic layer on top.

Two consumers drive the shape:

- **findmyprompt** should later replace its `model/index.ts` + `functionRegistry.ts` with imports from
  this package **with behavior stable** — in particular its content-addressed ids and memo keys must not
  change.
- **JaiRA** builds its workflow basis on this package using **inline values** (no content-addressed store
  yet). A future goal is JaiRA adopting the same content-addressed substrate; the current goal is to align
  the vocabularies so that transition is a substitution, not a rewrite.

## Decisions (settled)

| # | Question | Decision |
| --- | --- | --- |
| 1 | What to import | **Types + registry only**: the op model and the function registry (sync + async). `runOperation`, stores, memoization, retry, evaluator, search all stay in findmyprompt. |
| 2 | How deep the rewiring goes | **Ops become the spine**: hw states compile to operations, core's untyped `Record` payloads are replaced by typed op shapes, the llm executor becomes the `PromptOp` leaf runner. |
| 3 | Typing level | **Both**: keep the runtime JSON-Schema typing (schemas on every slot) *and* add a TS-generic layer with compile-time checked wiring. |
| 4 | Schema technology | **JSON Schema + const-generic inference** (`json-schema-to-ts`-style `FromSchema`). No zod; no new runtime dependency — inference is type-level only. |
| 5 | Value representation | **Generic over the reference substrate**: one op vocabulary parameterized by a "ref family". findmyprompt instantiates with content ids; hw/JaiRA instantiate with inline values (and findmyprompt may also pass inline values where the backend has already resolved an id). |
| 6 | Non-llm runtimes | **`PromptOp` stays LLM-only** (faithful). A separate **`ExecOp`** kind (per DESIGN.md §8.1's sketch) covers delegated/agent runtimes (`claude-code`, `agent-sdk`, …) and composite units (hw itself). |
| 7 | hw wiring | **Bindings replace expression wiring.** The `Parameter`/`Ref` binding model becomes hw's input wiring; the expression DSL remains for transitions/guards, and an `{ expr }` **binding leaf** is kept as an escape hatch for small computations. |
| 8 | hw migration | **Big-bang.** Pre-release repo: `format.ts`, SPEC §4.2/§6–§8, validator, engine, and all fixtures rewritten in one change. No dual wiring support. |

## 1. The package

**`@declarative-ai/ops`** at `packages/ops`. Dependency-free (the async-function context is a generic
parameter, so ops does not import `ExecServices` from core). Layering becomes:

```
ops  ←  core  ←  llm, hw, tools, claude-code, services
```

`core` imports the op vocabulary; `ops` imports nothing. Exports:

- `model` — the generic op types: `Operation`, `PromptOp`, `FunctionOp`, `ExecOp`, `Parameter`,
  `NamedParameter`, `Signature`, the ref-family machinery (§2).
- `families` — the two standard ref-family instantiations: `IdFamily` (content-addressed) and
  `InlineFamily` (direct values), plus their `Ref` unions.
- `registry` — `FunctionRegistry` with sync `FunctionImpl` and async `AsyncFunctionImpl<Ctx>` registration
  (a faithful port of findmyprompt's `functionRegistry.ts`, made generic in `Ctx`).
- `typed` — the TS-generic layer: `defineFunction`, typed op builders, `FromSchema` inference helpers,
  `TypedOperation<I, O>` (§4).

**Not** exported: execution. There is no `runOperation` here; findmyprompt keeps its own, and ai-exec
executes ops through the existing `Executor`/`Runtime` machinery (§5–§7).

## 2. The generic reference model

The single most important design element. findmyprompt leaves are ids into artifact stores
(`{ textId }`, `{ jsonId }`, `{ resultId }`, `jsonRefs` trees, producer ops by id); hw/JaiRA leaves are
inline values and references into a run context. Both are instantiations of one shape, parameterized by a
**ref family** — the three slot types every op field is built from:

```ts
interface RefFamily {
  text: unknown;    // what a text-valued slot holds (a prompt body, an error)
  json: unknown;    // what a json-valued slot holds (a config, a schema, a value)
  binding: unknown; // what a Parameter's `binding` holds (the wiring edge)
}
```

The model is generic in `F extends RefFamily`:

```ts
interface Parameter<F extends RefFamily> {
  kind: RefKind;              // "text" | "json" | "prompt" | "function" — unchanged
  schema?: F["json"];         // JSON Schema for the slot (inline: the schema value; ids: an Id)
  binding?: F["binding"];     // absent = free/external input slot — unchanged semantics
  index?: number;             // positional sort key — unchanged semantics
}

interface PromptOp<F extends RefFamily> {
  kind: "prompt";
  system?: F["text"];
  user: F["text"];
  config: F["json"];          // typed by the LlmConfiguration schema (§6)
  input: Record<string, Parameter<F>>;
  output: NamedParameter<F>;
}

interface FunctionOp<F extends RefFamily> {
  kind: "function";
  functionRef: string;        // registry name, or (id family) another op's Id → partial application
  input: Record<string, Parameter<F>>;
  output: NamedParameter<F>;
}
```

### 2.1 The two standard families

**`IdFamily`** — the findmyprompt substrate, semantically identical to today's `model/index.ts`:

```ts
type IdFamily = {
  text: Id;                                   // -> Text artifact
  json: Id;                                   // -> Json artifact
  binding: ValueRef | { resultId: Id } | JsonRefs
         | (OperationRef & { parameters?: Record<string, Parameter<IdFamily>> });
};
```

**`InlineFamily`** — the hw/JaiRA substrate:

```ts
type InlineFamily = {
  text: string;
  json: unknown;                              // the value itself (a JSON Schema, a config, a literal)
  binding: { value: unknown }                 // inline literal
         | { ref: string }                   // pure context reference ("children.plan.outputs.steps")
         | { expr: string }                  // expression escape hatch (§7.2)
         | { op: Operation<InlineFamily>;    // inline producer — the graph edge, op embedded not by id
             parameters?: Record<string, Parameter<InlineFamily>> };
};
```

The producer-binding semantics are unchanged from findmyprompt: `kind ∈ {text, json}` runs the producer
and its output fills the slot; `kind ∈ {prompt, function}` passes the op definition itself (higher-order).
An explicitly-passed value overrides a binding. What differs per family is only *how a leaf names a value*.

A backend that has resolved ids may hand inline values through the same vocabulary — a mixed family
(`binding: IdFamily["binding"] | { value: unknown }`) is a legal instantiation; nothing in the model
assumes leaves are homogeneous.

### 2.2 Field-name compatibility with findmyprompt (⚠ the one behavior-stability risk)

The generic model uses substrate-neutral names (`system`, `user`, `config`, `schema`) where findmyprompt
today has id-suffixed names (`systemPromptTextId`, `userPromptTextId`, `configJsonId`, `schemaId`).
findmyprompt **content-addresses ops by hashing their canonical serialized content — field names
included** — so adopting the neutral names naively would change every op id and break memo/dedup against
its existing database.

Decision: the package ships the neutral names, plus a `legacy` module with lossless converters
(`toLegacyShape`/`fromLegacyShape`) between `Operation<IdFamily>` and the exact current findmyprompt
shapes. findmyprompt applies the legacy shape **at its hash/store boundary only** (its `artifacts/hash.ts`
canonicalizes the legacy form), keeping every existing content id stable while its in-memory code moves to
the shared types. The converters are pure field renames — mechanically verifiable by round-trip tests
against findmyprompt's fixtures. If this proves too fiddly in practice, the fallback is to keep the
findmyprompt field names verbatim in the shared model and tolerate `userPromptTextId: string` reading
oddly in inline mode; that choice is reversible until findmyprompt migrates.

## 3. Op kinds

```ts
type Operation<F extends RefFamily> = PromptOp<F> | FunctionOp<F> | ExecOp<F>;
```

- **`PromptOp`** — one structured LLM call, exactly findmyprompt's semantics. LLM-only: its dedicated
  prompt-template fields are the searchable identity findmyprompt's optimizer mutates, and its config is
  the `LlmConfiguration` surface. Runs through `@declarative-ai/llm` (§6).
- **`FunctionOp`** — a registered function (sync or async) or, in the id family, a partial application of
  another op. Runs through the registry (§3.1). findmyprompt's async functions (`registerAsyncFunction` /
  `AsyncFunctionImpl`) are FunctionOps whose impl is async — that trio (PromptOp, FunctionOp, async
  function ops) is the imported system.
- **`ExecOp`** *(new here; DESIGN.md §8.1 sketched it for findmyprompt)* — the leaf for everything that is
  neither a bare LLM call nor host code: delegated agent runtimes (`claude-code`, future `agent-sdk`) and
  composite units (a hierarchical workflow run as a child op).

```ts
interface ExecOp<F extends RefFamily> {
  kind: "exec";
  runtime: string;            // resolved through registry.runtimes (e.g. "claude-code", "hw")
  definition: F["json"];      // runtime-specific definition: config surface, or a workflow bundle ref
  input: Record<string, Parameter<F>>;
  output: NamedParameter<F>;
}
```

An ExecOp's *prompt* is not a dedicated field: a delegated agent's prompt is normally computed by wiring
(a rendered template), so it is an ordinary input `Parameter` named `prompt` (kind `"text"`) — wiring
stays uniform, and only PromptOp privileges its templates (because they are the optimizer's search
surface). Memoizability, workspace mutation, and structured-output support come from the selected
runtime's `ExecutorCapabilities`, not the op kind.

### 3.1 The function registry

A faithful port of findmyprompt's `FunctionRegistry`, generic in the async context:

```ts
type FunctionImpl = (inputs: FunctionInputs) => unknown;                       // pure, sync
type AsyncFunctionImpl<Ctx> = (inputs: FunctionInputs, ctx: Ctx) => Promise<unknown>;

interface FunctionRegistry<Ctx> {
  register(ref: string, impl: FunctionImpl): void;
  registerAsyncFunction(ref: string, impl: AsyncFunctionImpl<Ctx>, opts?: AsyncFunctionOptions): void;
  get / getAsyncFunction / has ...
}
```

findmyprompt instantiates `Ctx = RunCtx`; ai-exec instantiates `Ctx = ExecServices`. Lookup order is
preserved (sync registry first; id-family miss falls through to partial application).

## 4. The typed layer (TS generics over JSON Schema)

Runtime typing (schemas on every slot) is preserved; on top of it, a type-level layer gives compile-time
safety with **zero runtime cost** — `FromSchema` inference over `as const` JSON Schemas
(`json-schema-to-ts` as a types-only dependency):

```ts
const summarize = defineFunction({
  name: "wordCount",
  input:  { type: "object", properties: { text: { type: "string" } }, required: ["text"] } as const,
  output: { type: "number" } as const,
  impl: ({ text }) => text.split(/\s+/).length,   // `text: string` INFERRED; return checked as number
});
```

- `defineFunction` returns a `FunctionDef<I, O>`: the schemas (runtime truth) + phantom `I`/`O` (inferred
  compile-time truth). Registering it also registers the plain string-keyed impl, so dynamic
  (`functionRef` by name) resolution still works — the typed handle is *additional*, not required.
- Typed op builders (`promptOp`, `functionOp(def, bindings)`, `execOp`) return
  `TypedOperation<I, O>`; binding a producer whose `O` doesn't match the consuming slot's type is a
  **compile error**. Untyped/dynamic construction (deserialized ops, findmyprompt's search mutating op
  graphs) bypasses the typed builders and relies on the runtime schema checker — both layers check the
  same schemas, so they cannot drift.
- Where inference degrades (schemas with `$ref`s into stores, dynamic schemas), the type falls back to
  `unknown` — runtime validation still applies; nothing is *less* checked than today.

## 5. Rewiring `core`

| Today (`core/src/contract.ts`) | Becomes |
| --- | --- |
| `RuntimeOp` (prompt/config/`Record<string, unknown>`) | **removed** — a `Runtime` runs an `ExecOp<InlineFamily>` (or, for the llm runtime, a `PromptOp<InlineFamily>`); the payload is the typed op. |
| `HostFunction.run(inputs: Record<string, unknown>)` | `FunctionDef<I, O>` from ops; `run` typed by the def. `CapabilityRegistry.functions` becomes the ops `FunctionRegistry<ExecServices>`. |
| `Tool extends HostFunction` + hand-written `inputSchema` | `Tool = FunctionDef + description`; `inputSchema` **derived** from the def's input schema — one schema, two uses (graph op + agent tool), no drift. |
| `Outcome` with `value?: unknown` | `Outcome<O = unknown>` with `value?: O`; executors thread the op's output type through, so a typed call site gets a typed result. |
| `ExecutionSpec.definition: unknown` | `ExecutionSpec<K extends UnitKind>` with a per-kind definition-type map (llm-call → `LlmCallDefinition`, hierarchical-workflow → `WorkflowBundle`, …). The generic default keeps dynamic dispatch working. |

`Executor`/`ExecHandle`/`ExecServices`/wrapper composition are unchanged in role — ops are what flows
*through* the contract, not a replacement for it.

## 6. Rewiring `llm`

`@declarative-ai/llm` becomes the **PromptOp leaf runner**: `createLlmRuntime` accepts a
`PromptOp<InlineFamily>` instead of the removed `RuntimeOp`. The §1.5 `resolveConfig` pipeline is
unchanged — but `PromptOp.config`'s slot is typed by the `LlmConfiguration` schema
(`core/src/llmConfig.ts`), so the config surface stops being `Record<string, unknown>` at the seam where
authors touch it. `plan()` gains a typed entry point taking a `PromptOp` directly.

## 7. Rewiring `hw` (big-bang)

The largest change. A state's operation and wiring become ops-and-bindings; the expression DSL survives
only where control flow genuinely needs it.

### 7.1 Format

- `StateDef.runtime` / `StateDef.function` blocks → one `operation: Operation<InlineFamily>` (a
  `runtime` state with name `llm` is a `PromptOp`; other runtimes are `ExecOp`s; `function` states are
  `FunctionOp`s). Runtime-block extras that are session/permission concerns (`session`, `permissions`,
  `tools`, `conversation`) stay as sibling state fields — they configure the *execution environment*
  (RUNTIMES-AND-PERMISSIONS.md), not the op.
- `StateDef.inputs`/`outputs` (`FieldSchema` maps) → `Record<string, Parameter<InlineFamily>>` /
  `NamedParameter` — same information (schema, optional/default, description) in the op vocabulary;
  `OutputFieldSchema.from` becomes the output's binding.
- `WiringValue` (`string | { value }`) → the `InlineFamily` binding union: `{ value }` literals,
  `{ ref }` pure context references, `{ expr }` for computation, `{ op }` inline producers.
  `ChildDecl.inputs` wires with the same bindings.
- SPEC §4.1–§4.2 (schemas/wiring), §6 (expression contexts), §7–§8 (runtime/function states) rewritten;
  transitions (`TransitionDecl.when`), `limits`, and guard expressions keep the expression DSL unchanged.
- All fixtures/tests rewritten. No dual support, no deprecation shims.

### 7.2 The `{ expr }` leaf

Small computations (`"children.retry.outputs.count + 1"`, template interpolation) keep the DSL as a
*binding kind* rather than the wiring default. An expr leaf may declare its expected output `schema`; the
engine validates the evaluated value against it at run time. Exprs are the one wiring form the static
checker cannot type — the validator flags schema-less expr leaves as warnings, nudging toward `{ ref }`
or a `FunctionOp` producer where typing matters.

### 7.3 Static checking — where the strong typing lands at authoring time

The hw validator grows a binding type-checker: every `{ ref }` is resolved against the schema of the slot
it points at, every producer's output schema is checked against the consuming slot's schema, free inputs
must be covered by the parent's wiring. v1 compatibility is exact-schema-match plus obvious widenings;
structural subtyping is a later refinement. This replaces "expression parses" with "the wire type-checks"
— the same property the TS layer gives programmatic authors, enforced for JSON authors.

### 7.4 Engine

The engine stops rendering a bespoke `RuntimeOp`: it resolves a state's `operation`'s bindings against the
run context (refs/exprs/producers → `Parameter` values), then dispatches the resolved op — `PromptOp` →
the llm runtime, `ExecOp` → `registry.runtimes`, `FunctionOp` → the ops function registry. Conversation
preambles, sessions, and permission gating attach exactly where they do today; only the payload shape and
the wiring resolution change.

## 8. Consumers

- **findmyprompt** (behavior stable): replace `model/index.ts` types and `functionRegistry.ts` with
  `@declarative-ai/ops` imports at `IdFamily` (+ `Ctx = RunCtx`), applying the §2.2 legacy converters at
  the hash/store boundary so every content id, memo key, and DB row survives. `runOperation`, stores,
  search, judges stay put. `ExecOp` gives it DESIGN.md §8.1's composite leaf for free when it wants it.
- **JaiRA / hw**: authors workflows against `InlineFamily` ops with the typed builders. The future move to
  content addressing is: adopt an artifact store, switch the family instantiation, intern inline values as
  artifacts — the op graphs, registry, and typed layer are unchanged.

## 9. Out of scope

`runOperation` and any shared execution engine for ops; memoization stores and content-addressed artifact
storage (hashing primitives already live in core per DESIGN.md §6); search/strategies/evaluator; the
permission model (RUNTIMES-AND-PERMISSIONS.md, unchanged); process-unit executors beyond the existing
`claude-code` runtime adapter.

## 10. Open questions / risks

1. **§2.2 converter fidelity** — must be proven with round-trip tests against real findmyprompt fixtures
   before findmyprompt migrates; the fallback (keep legacy field names in the shared model) stays open
   until then.
2. **`FromSchema` inference limits** — deeply composed schemas (`allOf`, store-resolved `$ref`) degrade to
   `unknown`; acceptable (runtime checking remains), but the typed builders should make the degradation
   visible rather than silent.
3. **Binding compatibility strictness (§7.3)** — exact-match v1 may be too strict for real workflows
   (e.g. a producer emitting a superset object); the widening rules need worked examples from JaiRA's
   first workflows.
4. **`ExecOp` memoization** — capabilities-gated as today, but a delegated agent's `definition` identity
   (what exactly hashes) needs pinning down when memoizing ExecOps becomes real.
5. **hw expression contexts after the rewrite** — with wiring moved to bindings, which of the nine
   `CONTEXT_NAMESPACES` remain reachable from guard expressions (and whether `function.*` results stay a
   namespace or become ordinary op outputs) needs a pass in the SPEC rewrite.
