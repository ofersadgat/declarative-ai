# HW API Redesign

> **Status: FULLY LANDED — code + all four docs (README/API/DESIGN/SPEC).** Implemented and green:
> typecheck clean across all packages, 393 tests passing. All slices done, including the retry/repair
> unification (`withRetry({ transient, validation: { feedback } })`; `withRepair` kept as a deprecated
> alias) and the SPEC §7 (Runtime Execution) / §8 (Function States) prose.
>
> **Follow-up (see RUNTIMES-AND-PERMISSIONS.md):** the `ui.*` expression namespace was later renamed to
> `function.*` (the expr DSL treats `function` as an ordinary identifier — only `true`/`false`/`null` are
> reserved), and the `agent`/`ui` operation model grew into registerable runtimes, tools, and a
> profile × mode permission system.

## Why

Examples 1–7 are all one package (`@declarative-ai/llm`), so `router`, `repair`, and
`providers` read as native vocabulary. Example 8 (hw) is the **only example that composes
two packages**: a workflow is a state machine whose child states happen to dispatch through
`llm-call`. The example is dominated by `llm-call` plumbing that has nothing to do with
workflows, and nothing signals which concepts are hw's versus substrate leaking through.

Five concrete confusions motivated this:

1. **`router`** reads like state-routing; it actually routes model ids to endpoints.
2. **`repair`** reads like state recovery; it's just a validation-retry that mutates the prompt.
3. **`providers`** means three different things across three layers of one example.
4. **`bundle` vs `states`** — the example builds a bundle, discards it, and passes raw `states`.
5. **Two registration sources** — a flat `registry` *and* a `providers` binding table.

## Decisions

### Renames / removals

| Today | Becomes | Why |
| --- | --- | --- |
| `createRouter` / `ProviderRouter` | `createModelRouter` / `ModelRouter`, **optional** | "router" was ambiguous vs state-routing; it routes model ids to endpoints. Env-key default already exists — drop it from the hello-world path. |
| `ExecServices.providers` (`unknown`) + `LlmCallEnvironment.providers` | one seam `modelRouter` | It was always one slot: generic (`unknown`) in core, narrowed to the router type in llm. One concept. |
| `withRepair({ turns })` | a mode on retry: `withRetry({ validation: { turns, feedback: true } })` | Repair = validation-retry with the prompt mutated; not a separate noun. See "Retry/repair" below. |
| `agent` operation block + its `provider` field | `runtime` block + `name` field | "agent" is loaded (means a tool-using loop in AI; a bare `llm` call isn't one) and no longer matches its contents. See "Operation blocks" below. |
| `HwExecutorOptions.providers: Record<string, ProviderBinding>` | **removed** | The redundant registration source. Folded into `registry.runtimes`. |
| `ProviderBinding` / `llmCallBinding` / `LlmCallBindingOptions` | **removed** | Its op→definition logic moves inside the runtime executor, where it belongs. |
| `HierarchicalWorkflowDefinition` (a separate `{ rootId, states }`) | `= WorkflowBundle` | One shape, one name; `start` takes the bundle directly. |
| flat `registry` (`kind → Executor`) + `providers` table | one typed `CapabilityRegistry` | Single registration source, typed facets. See "Registry" below. |

## Operation blocks — name the facet, select by `name`

A state performs one of three operations, today keyed `agent` / `skill` / `ui`. Two problems:
`agent` is a loaded AI term (a bare `llm` runtime call is not an agent), and the three blocks
already use **three different selector fields** — `provider`, `name`, `component` — for the same
idea ("which entry in the facet"). Consolidate on the convention that already exists
(`skill.name`): **block = facet name, selector = `name`.**

There are **two** operation types (down from three — see "`skill` is a prompt source" below):

| Operation | Block | Selector | Resolves via |
| --- | --- | --- | --- |
| run a runtime | `runtime` | `name` | `registry.runtimes.get(op.name)` |
| invoke a function | `function` | `name` | `registry.functions.get(op.name)` |

```json
{ "runtime": { "name": "claude-code", "config": { "configRef": "critic" }, "prompt": { "template": "…" } } }
```

Each executor operation reads `<facet>: { name, …params }` resolving through `registry[facet].get(name)`.
Naming the block `runtime` with the field `name` (not `runtime`) avoids the `runtime: { runtime }`
collision. The discriminator (which key is present) is unchanged. Types `AgentConfig` / `UiConfig`
become `RuntimeConfig` / `FunctionConfig`; `SkillConfig` is removed; the engine op union
`"agent" | "skill" | "ui"` becomes `"runtime" | "function"`.

### `skill` is a prompt source, not an operation

Today `runSkillOperation` (`engine.ts`) forwards a skill through the **same** executor path as an
agent op — a skill was never a distinct execution, just a runtime call whose prompt came from a
library. And `SkillDef = { provider, template, config }` wrongly bundles a runtime (`provider`) and
`config` into the skill. A skill is fundamentally **name → prompt**: provider-agnostic,
config-agnostic. Strip it to the template.

Consequences:

- **`skill` is not an operation type.** A prompt can't run without a runtime, so a skill alone is
  incomplete. A `runtime` op's prompt comes from **either** an inline template **or** a named skill
  (exactly one): `prompt: { template: "…" }` or `prompt: { skill: "critique-plan" }`.
- **`registry.skills` is a content facet, not an executor facet** — a template library (name →
  template) the runtime op's prompt reads from. The registry holds behavior facets (`runtimes`,
  `functions`) and this one content facet.
- **Conversation policy decouples from prompt source.** Today skills force `conversationMode:
  "fresh"`; in the new model the runtime op sets its own conversation mode regardless of where its
  prompt came from.

### `function` generalizes `ui` (was: `component`)

The `ui` operation is a special case of "invoke a registered host function": inputs → structured
output, where a UI component is simply a function that renders and awaits human input. Generalizing
`component` → `function`:

- **Fits the existing async dataflow.** An async function's outputs are `PENDING` (SPEC §10.4,
  `expr.ts`) until the promise resolves — identical to how async child states are handled. No new
  "waiting" concept; UI-await stops being special.
- **Dissolves the InteractionPort-vs-components question.** An interactive function *is* the port:
  the host registers it, the engine calls and awaits it. Any renderer plumbing lives inside that
  function, not as an engine seam.
- **Opens the non-LLM, non-UI middle** — deterministic transforms, data fetches, validators,
  computed joins, tool calls — which today have no operation type and get abused as `runtime` calls.

What UI-as-a-distinct-op bought (blocked-state events, approval gates, memoization purity) is preserved by
moving it from the op keyword to the function's declared **capabilities** — exactly as `runtime`
capabilities distinguish `llm` from `claude-code`. Interaction *flow* (block, auto-approve, refuse in a
search context) is not an executor policy at all: it's the designer's composition through the registered
`function`s — a search caller refuses a human gate by registering a rejecting function or not registering
it (the state then fails when reached), so no `eager`/`lazy` mode is needed:

```ts
interface HostFunction {
  capabilities: { interactive?: boolean; pure?: boolean };
  run(inputs: Record<string, unknown>, services: ExecServices): unknown | Promise<unknown>;
}
```

## The registry — one typed `CapabilityRegistry`

The core move. Instead of one flat `registry.get(kind): Executor` returning a
`definition: unknown` (the impedance mismatch that forced an adapter), the registry is
**typed by category**. Each facet's native interface already matches how that category is
invoked — so there is nothing to bridge.

```ts
interface CapabilityRegistry {
  runtimes:  Registry<Runtime>;       // `runtime` operations: claude-code, codex, llm (SPEC §7.1) — behavior
  functions: Registry<HostFunction>;  // `function` operations: host code, incl. interactive UI (was `ui`/components) — behavior
  skills:    Registry<Template>;      // named prompt templates a runtime op's prompt references (data, NOT an executor)
  // workflows: deferred — see open questions. A black-box sub-workflow may just be a HostFunction.
}

interface Registry<T> {
  get(name: string): T | undefined;
  register(name: string, value: T): this;
}

interface Runtime {
  capabilities: Capabilities;
  run(
    op: { prompt: string; system?: string; config: Config; outputSchema: Schema; timeoutMs?: number },
    services: ExecServices,
  ): ExecutionHandle;   // ← the operation IS the interface; no adapter, no `unknown`
}
```

Naming: the **agent** is the operation in a state (`agent: { runtime, config, prompt }`); the
**runtime** is the adapter that executes it. The facet is `runtimes` so the lookup matches the
renamed field: `registry.runtimes.get(state.agent.runtime)`.

What this unifies (four scattered seams → one object):

| Today | Becomes |
| --- | --- |
| `HwExecutorOptions.providers` (agent bindings) | `registry.runtimes` |
| `SkillResolver` (`skills.get`) | `registry.skills` |
| `registry.get(kind)` for nested workflows | `registry.workflows` |
| duck-typed `ctx.providers` | typed facet, no duck-typing |

Boundaries that keep the registry bounded:

- **In-bundle child states are not registry entries** — they are states in the same bundle,
  run by the same engine. `registry.workflows` is only for composing a *different* workflow as
  an opaque unit (the rare cross-bundle case).
- **The standalone one-shot `llm-call`** (example 1) stays a direct entry point (the request
  API), not a registry lookup — so there is no `units`/`kinds` facet duplicating `runtimes`.

`llm` is simply the simplest runtime: `registry.runtimes.get('llm')` and `.get('claude-code')`
share one interface and differ only in `capabilities` (bare model call vs file-editing agent) —
exactly SPEC §7.1's "runtime adapters with capabilities, not interchangeable strings."

## Retry / repair

There is no separate "repair" concept. A schema-validation failure classifies as
`api-retriable` and, by default, is **not** auto-retried (re-sending an identical prompt just
burns budget). All "repair" does is **mutate the prompt** — append the concrete validation
errors so the model can fix its output. So it is one re-attempt policy with two axes:

```ts
withRetry({
  transient: 3,                              // network-retriable: backoff, auto, generic over any op
  validation: { turns: 2, feedback: true },  // api-retriable; feedback:true == today's `withRepair`
})
```

The typed registry makes the split fall out cleanly: `transient` and `withMemoize` touch only
the *outcome*, so they stay generic over any category's input. `feedback: true` must mutate the
prompt — which only a `runtimes` op has — so feedback-repair is inherently a runtime-category
concern, not a generic wrapper whose composition order is a footgun.

## Bundle is the definition

`loadBundle` now earns its keep (validation + closure restriction + hash identity) and the
bundle flows through as the definition — raw `states` never appears at the call site, and
`rootId` is not re-derived from an argument it was handed.

## Revised example 8, end to end

```ts
import { MapExecutorRegistry } from "@declarative-ai/core";
import { SchemaValidator } from "@declarative-ai/services";
import { createLlmRuntime, withRetry } from "@declarative-ai/llm";
import { createHierarchicalWorkflowExecutor, loadBundle, snapshotHash } from "@declarative-ai/hw";

// ONE registry, typed facets. No modelRouter passed → env-key default.
// Defaults + configRef presets live on the runtime, not in a separate binding table.
const registry = new MapExecutorRegistry();
registry.runtimes.register(
  "llm",
  withRetry({ transient: 3, validation: { turns: 2, feedback: true } })(
    createLlmRuntime({
      defaults: { model: "anthropic/claude-sonnet-5", temperature: 0.3 },
      configs: presetRegistry,   // resolves a state's agent.config.configRef
    }),
  ),
);

const hw = createHierarchicalWorkflowExecutor();     // no providers table

const bundle = loadBundle(states, "feature/plan");   // { rootId, states } — validated, hashable
const handle = hw.start(
  { kind: "hierarchical-workflow", definition: bundle, inputs: { issue: "…" } },
  { registry, validator: new SchemaValidator() },
);
const outcome = await handle.outcome;
// memoize identity from the same object:
//   withMemoize({ cache, identify: () => snapshotHash(bundle) })
```

A state file's agent operation now reads:

```json
{ "agent": { "runtime": "llm", "config": { "configRef": "critic" }, "prompt": { "template": "…" } } }
```

The `"planner"` / `"critic"` names that used to be provider keys are just **presets** in
`presetRegistry` — which is what they always were.

## Engine dispatch: before / after

Before — two hops plus an adapter (`engine.ts`):

```ts
const binding  = providers[req.provider];               // hop 1: name → binding
const executor = registry.get(binding.kind);            // hop 2: kind → executor
const def      = binding.definition({ prompt, config }); // adapter: op → definition
executor.start({ kind, definition: def, /* … */ });
```

After — one typed hop:

```ts
registry.runtimes.get(req.runtime).run(
  { prompt, system, config, outputSchema, timeoutMs },
  services,
);
```

## Open questions

**1. UI components — RESOLVED by generalizing `ui` → `function`.** An interactive function *is*
the port; the `components`-vs-`InteractionPort` seam question disappears (see "`function`
generalizes `ui`"). Any renderer plumbing lives inside the interactive function, not as an engine
seam.

**2. `Executor<I>` — the wrapper split (gate run; result below).** Parametrize the executor by
input type, `Executor<I = unknown, R = ExecServices>` with `start(input: I, ctx: R)`. The gate —
"which wrappers actually read the llm `definition` shape?" — was run against `wrappers.ts`. Result:

- **Only `withMemoize` is genuinely generic.** It keys off an `identify(input)` seam (hw supplies
  `snapshotHash`) and otherwise hashes the definition opaquely (`hashCanonical`) — no llm-shape
  assumption. It moves to core, typed `Executor<I>`.
- **Everything else is legitimately llm-coupled** and stays typed to the runtime facet in llm:
  `withRepair` and `withSession` mutate the prompt/transcript (input-shaped); `withRateLimit` and
  `withBudget` read `def.model` + token estimates to rate-limit/price (model-metadata-coupled);
  `withDeadline` is mostly generic (clamps `spec.limits`) with one llm-specific `def.timeoutMs`
  rewrite that stays behind.

This is **lower-risk than first assumed**: no big generic-wrapper migration. The llm wrapper stack
stays where it is, now typed to `Executor<RuntimeOp>`; core gains only input-generic `Executor<I>`
plus a generic `withMemoize`. Typing still converts today's footgun (composing `withSession` onto a
non-LLM executor, silently a no-op because everything is `Executor<ExecServices>` with
`definition: unknown`) into a compile error.

**3. `registry.workflows` — defer the facet, keep the concept.** In-bundle nesting (`children` by
state id) needs no registry. A `workflows` facet would serve only *cross-bundle* black-box
composition, which has **no operation syntax today** — so it is YAGNI for v1, and adding it later is
purely additive (a workflow executor already satisfies `Executor<I>`). Triggers to add it:
cross-bundle reuse, or per-sub-workflow memoization under `snapshotHash`. Note a black-box
sub-workflow is itself `(inputs) => Promise<outputs>` — i.e. a `HostFunction` — so it may ride
`registry.functions` and need no dedicated facet. The caveat that decides facet-vs-function when it
lands: sub-workflows also fold cost/calls into metrics (`childCost`/`childCalls`) and expose a
`snapshotHash` memo identity, which a bare function op does not.
