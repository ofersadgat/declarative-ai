# Hierarchical Workflow Specification

> **Provenance and scope.** This file is the canonical specification of the
> hierarchical-workflow formalism implemented by `@declarative-ai/hw`, migrated from
> `JaiRA/SPEC.md` (which remains the product spec for the JaiRA app).
> Normative for this library: §2.3–§2.4 (states, state IDs), §3 (state machine
> semantics), §4 (inputs/outputs/artifacts/conversations), §5 (state
> file format), §6 (expressions and static validation), §7.3 and §9 (worked examples,
> used as golden tests), §8 (function states / interactive UI, realized here as
> interactive functions in `registry.functions`), §10.1–§10.4 (statuses, run records, durability, async
> children), §12 (versioning → snapshot hashing). Sections about tasks,
> boards, Git isolation, safety policy, and MVP scope describe the JaiRA
> product and are context for the library, not requirements on it.
>
> **How the library realizes this formalism** — the executor, the injected prompt `Executor` a
> `PromptOp` is dispatched to (`@declarative-ai/promptop`; config resolution per operation),
> session coordination by logical id, snapshot hashing/memoization, the typed
> `CapabilityRegistry`, and the `Persistence` port — is documented in
> [DESIGN.md](DESIGN.md) §7 (with the settled declarative model in §1). The typed operation
> vocabulary a state compiles to (`Operation`, `Parameter`, `Ref`, the ref families) lives in
> `@declarative-ai/ops`; its design is [DESIGN.md](DESIGN.md) §3.1 and its type surface is
> [API.md](API.md), which is also the precise `@declarative-ai/hw` API reference.

## 1. Purpose

JaiRA is a local-first, single-user project management and agent orchestration app.
It combines a Jira-style Kanban interface with hierarchical, declarative state
machines that can run local AI coding agents, collect human input, produce
artifacts, and move tasks through project-defined workflows.

The central design goal is to make AI-assisted work observable, resumable,
auditable, and configurable without letting agents directly own workflow control.

## 2. Core Concepts

### 2.1 Project

A project is the root unit of configuration and execution. It contains:

- Workflow state files.
- Tasks.
- Artifact references.
- Conversation artifacts.
- Execution history.
- Runtime configuration for local agents.
- Safety policy.

For the MVP, a JaiRA project maps to one local project directory. Workflow
state files live in a reserved, engine-owned directory (`.jaira/workflows/`)
inside the project. Agents have no read or write access to `.jaira/`.

### 2.2 Task

A task is a unit of work shown on a board. A task has exactly one active state at
a time within its current workflow level.

A task may:

- Move through workflow states.
- Spawn subtasks.
- Enter child workflows.
- Produce artifacts.
- Accumulate conversation history.
- Wait for user input.
- Wait for external events.

Subtasks are separate tasks. Substates are child states inside a workflow tree.
These are distinct concepts.

### 2.3 Workflow State

A workflow state is defined by one state file. State files form a tree.

Each state may define:

- Inputs (including configuration knobs — inputs with a `default`).
- Outputs.
- Child states.
- Default child order.
- Agent execution behavior.
- UI behavior.
- Transition rules.
- Iteration limits.
- Safety requirements.

A state only knows about its own declared children. It does not know siblings,
parents, or arbitrary external states.

### 2.4 State ID

A state ID is the state file path relative to the workflow root, without the file
suffix.

Examples:

```text
feature
feature/plan
feature/plan/critique
feature/plan/critique/address_weaknesses
```

### 2.5 Board

A board is a visual projection of the active states of tasks.

At any level, columns correspond to visible child states. Double-clicking a task
that is inside a state with a child workflow opens the sub-board for that state.

Example:

```text
feature
  plan
    goals
    context
    critique
  design
  implement
  review
```

At the feature board level, the task appears in `plan`, `design`, `implement`,
or `review`.

When the task is in `plan`, double-clicking opens the planning sub-board with
columns such as `goals`, `context`, and `critique`.

## 3. Hierarchical State Machine Semantics

### 3.1 Tree Structure

The workflow definition is a tree. Runtime movement is controlled by transitions
within each node's local scope.

A state can transition to:

- One of its own child states.
- A typed termination outcome.

A state cannot transition directly to a sibling, parent, or ancestor. If a child
needs to influence its parent, it terminates with structured outputs. The parent
then decides what to do.

### 3.2 Operations

A state does its work by running **one operation** and its child states. A state
declares at most one `operation`, of one of two kinds:

- `prompt`: one structured model call, driven by the state's prompt. The prompt comes from an
  inline `template` or a named `skill` (a reusable prompt template from `registry.skills`).
  Dispatched to the injected prompt `Executor`.
- `function`: a registered function invoked by name (`registry.functions`) that returns structured
  data. This one kind covers host code, an interactive UI component that collects input from the
  user, a sub-workflow, a composite unit, and a **delegated agent adapter** (`claude-code`, …)
  alike. Nothing about the operation distinguishes them — the resolved registry entry's
  capabilities do (§7.1).

Child states are entered by sequence order or by explicit transition.

A state may declare an operation, children, or both. Operations run one at a time
in a fixed priority order: the state's `operation`, then child states in
`sequence` order. Async child states are the only exception to one-at-a-time
execution.

How an operation runs — the session it belongs to, the tools it may call, the conversation
preamble it receives, its permission baseline — is declared in a sibling `environment` block, not
in the operation (§7.1).

### 3.3 Evaluation Loop

1. Entering a state creates a new state instance. Declared inputs are resolved
   and validated; validation failure blocks the state.
2. The engine runs the highest-priority operation that has not yet run in this
   instance.
3. When an operation completes, its outputs are validated and the state's
   transitions are evaluated in declared order. The first transition whose
   `when` expression is true is taken.
4. A taken transition either enters a child state or terminates the state.
5. If no transition matches, the engine runs the next operation in priority
   order.
6. If no operations remain and no transition matches: if any children are
   still running, the state waits, and each child completion triggers another
   round of transition evaluation. Only when all children have finished and no
   transition applies does the state terminate with `terminate.success`.

If an operation fails unrecoverably, or a child terminates with `error` or
`timeout` and no transition handles it, the state terminates with
`terminate.error` instead of continuing.

Transition order matters: child-completion conditions should be declared before
child-entry conditions, so that the evaluation that runs after a child
completes does not immediately re-enter it.

Taking a transition to a child that appears in the `sequence` resets the
sequence cursor to that child and clears the recorded results of that child and
every later child in the sequence; default ordering then resumes from there.
For example, with sequence `[a, b, c, d]`, a transition from `d` back to `b`
clears `b`, `c`, and `d`; only `a` retains its results. Children outside the
sequence keep their results.

Starting an async child does not trigger transition evaluation; evaluation runs
when the child completes. A transition whose `when` expression references
outputs of a child that has started but not yet finished is skipped for that
evaluation round and becomes eligible again when the child resolves
(Section 10.4).

### 3.4 State Instances

Entering a state — including re-entering it via a transition — creates a fresh
instance. Outputs and child results belong to the instance. References such as
`children.<id>.outputs` resolve to the most recent instance of that child
within the current parent instance, and evaluate to `undefined` if the child
has not run.

`run.iteration` is the count of transitions taken so far within the current
state instance, starting at 0. It is the standard guard for cycles.

### 3.5 Active Path

A task has an active path through the state tree.

Example:

```text
feature/plan/critique
```

When `critique` terminates, control returns to the active `plan` instance,
which validates the child's outputs and continues its own evaluation loop.

### 3.6 Termination

Termination is a typed return from a child state to its parent. The termination
outcome describes how the state finished, not what it decided:

```text
terminate.success
terminate.error
terminate.canceled
terminate.timeout
```

Domain-level results such as "approved" or "needs changes" are ordinary
schema-validated outputs. Parents branch on `children.<id>.outputs.*` for
decisions and on `children.<id>.outcome` for failure handling.

A child termination includes:

- A termination outcome.
- Validated outputs.
- Produced artifacts.
- Execution metadata.

Output validation failure or an unrecoverable operation failure terminates the
state with `terminate.error`.

### 3.7 Parent-Owned Control

Parents own:

- Child ordering.
- Child input wiring.
- Child result handling.
- Output remapping.
- Parent-level transitions.

Children do not directly mutate parent state.

A parent's declared outputs are resolved when the parent terminates.

## 4. Inputs, Outputs, and Artifacts

### 4.1 Declared Slots

`inputs` and `outputs` are maps of name → **slot**. A slot is a `Parameter`:

```text
kind         "text" | "json" | "blob" | "prompt" | "function" — the slot's value type.
             Defaults to "text" for a string-typed schema, "json" otherwise; a `blob`
             (artifact) slot declares its kind explicitly (§4.6).
schema       A plain JSON Schema document. Absent = unconstrained (accepts anything).
binding      Where the value comes from (§4.2). Absent = a FREE slot, filled by the
             caller: the parent's wiring for an input, the operation for an output.
index        Positional sort key for bare/tuple ingestion. Wiring, not type.
default      Value used when nothing is wired in. Also the explicit opt-out from the
             reachability rule (§6.2).
optional     Slots are required by default; `true` relaxes that.
description  Human-readable documentation for the slot.
```

The whole type of a slot lives in `schema` — there is no parallel `type`/`enum`/`items`/
`properties`/`required`/`format` vocabulary beside it. An artifact-typed slot is a `blob`-kind
slot whose schema names the content's media type (§4.6); an unconstrained slot (no `schema`) is
the generic/passthrough case (§4.4).

`kind`, `schema`, `binding`, and `index` are the slot proper; `default`, `optional`, and
`description` are authoring metadata carried alongside it. An output slot may also declare a
`name`, which otherwise defaults to its key in the `outputs` map.

Inputs and outputs are required by default.

A field is optional only when:

- It declares `optional: true`.
- It declares a `default` value.

Schema validation failure blocks the state.

### 4.2 Inputs and Wiring

Inputs are named values available to a state. Inputs may come from:

- Task fields.
- Parent-provided values.
- Previous child outputs.
- Artifacts.
- Conversation artifacts.
- Literal values (including a `default`, which makes an input a configuration knob).

A child state receives only its declared inputs. It does not know where those
inputs came from.

Good:

```json
{
  "critique": {
    "state": "feature/plan/critique",
    "inputs": {
      "plan_doc": { "child": "context", "output": "plan_doc" },
      "goals": { "child": "goals", "output": "goals" }
    }
  }
}
```

Bad:

```text
The critique state directly reads ../context.outputs.plan_doc.
```

A wiring value is a **binding**: a structured object, never a path string. Five binding
forms are the base vocabulary; the rest are authoring **sugar** that the loader lowers onto
the base forms, so the validator, the snapshot hasher, and the engine only ever see base
cases.

| Authored binding | Meaning | Lowers to |
| --- | --- | --- |
| `{ "text": "significant" }` | A string literal. | itself (base) |
| `{ "json": { "n": 3 } }` | A JSON literal of any shape. | itself (base) |
| `{ "result": … }` | Reuse of an already-existing generation result. | itself (base) |
| `{ "refs": … }` | An inline arrangement (array/object) whose leaves are refs. | itself (base) |
| `{ "op": … }` | A producer edge: a declared child's key, or an embedded operation. | itself (base) |
| `{ "child": "context" }` | The child's outputs object. | `{ "op": "context" }` |
| `{ "child": "context", "output": "plan_doc" }` | One named output of the child. | the producer edge above, plus a `select` producer projecting that property |
| `{ "input": "issue" }` | This state's declared input, by name. | a `scope.get` producer |
| `{ "expr": "outputs.weaknesses" }` | A small computation in the expression DSL (§6). | an `expr.eval` producer whose output schema is the inferred type |
| `{ "artifact": "design_doc" }` | A session-owned artifact, by name. | an `artifact.get` producer |
| `{ "conversation": "review", "message": 3 }` | A session's transcript, or one message of it. | a `conversation.get` producer |

Every sugar becomes a **producer edge** (or a literal), so the base vocabulary stays closed
and one uniform mechanism resolves all wiring: a producer edge on a declared child resolves
to that child's outputs when it has run, and parks the consumer while it is still in flight
(§10.4).

`{ "result": … }` and `{ "child": … }` are different concepts: the former references a result
that already exists, the latter is an edge the engine may still have to run.

Literals are wired by their own form; there is no wrapper object:

```json
{ "severity_threshold": { "text": "significant" } }
```

The same binding forms wire a child's inputs (`children.<key>.inputs`), fill an operation's
input slots (`operation.input`), and derive a state's outputs (§4.3).

### 4.3 Outputs

Outputs are named values produced by a state. An output slot with no `binding` is produced by
the state's operation and validated when the operation completes. An output slot **with** a
`binding` is *derived*: the binding is resolved against the state's context when the state
terminates.

```json
{
  "outputs": {
    "plan_doc": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" },
      "binding": { "child": "context", "output": "plan_doc" }
    },
    "outcome": {
      "schema": { "type": "string", "enum": ["complete", "blocked"] },
      "binding": { "expr": "children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'" }
    }
  }
}
```

Outputs should be schema-validated when they are used for:

- Transitions.
- UI decisions.
- Parent control flow.
- Automation.

### 4.4 Generic Passthrough Outputs

Some wrapper or delegation states need to return whatever a child state produces.
This is supported explicitly, as a slot that declares no `schema` — an unconstrained
slot constrains nothing, so any producer satisfies it.

Example:

```json
{
  "outputs": {
    "child_outputs": {
      "binding": { "child": "critique" }
    }
  }
}
```

Rule:

```text
Generic outputs may be stored or passed upward. Transition-relevant outputs
should be explicitly typed.
```

### 4.5 Configuration Knobs

There is no separate `params` concept: a state that configures reusable behavior does so with
ordinary **inputs** that carry a `default`. An input with a default is optional — the caller may
override it by wiring the input, and otherwise the default stands. This is the same collapse the
operation model already makes (an operation has inputs, not params); a state is no different.

Example:

```json
{
  "inputs": {
    "severity_threshold": {
      "schema": { "type": "string", "enum": ["minor", "significant", "critical"] },
      "default": "significant"
    },
    "max_findings": {
      "schema": { "type": "number" },
      "default": 10
    }
  }
}
```

A configuration input is read exactly like any other input: by a binding
(`{ "input": "severity_threshold" }`), by an expression (`inputs.severity_threshold`), or by
prompt interpolation (`{{inputs.severity_threshold}}`).

### 4.6 Artifacts

Artifacts are durable work products. Examples:

- Markdown plans.
- Design documents.
- Patches.
- Test reports.
- Review notes.
- Conversation summaries.
- Full conversation logs.

An artifact-typed slot is a **`blob`-kind slot** (§4.1) whose schema names the content's media type:

```json
{ "kind": "blob", "schema": { "type": "string", "contentMediaType": "markdown" } }
```

The slot's `kind` is what marks it; `contentMediaType` names the artifact's content format. Both are
ordinary JSON Schema — there is no bespoke marker keyword, because a produced artifact is simply a
blob-kind output slot rather than a parallel output channel. The value travelling through such a slot
is bytes, an artifact reference, or inline string content, and a prompt operation that produces one is
asked for its content as a string.

Git is responsible for artifact versioning. JaiRA records artifact references
and workflow metadata, but artifact history is delegated to the project Git
repository.

### 4.7 Conversations

Conversations are artifacts. A state may choose how to use prior conversation
context.

Supported conversation modes:

```text
full_history
summary
fresh
selected_artifacts
```

The initial default is `full_history`, but the schema must support all modes so
projects can move toward more controlled context selection over time.

A state selects its mode in `environment.conversation` (§7.1); the selected preamble is injected
into that state's own call.

Transcripts are scoped per **session** (`environment.session`, DESIGN.md §5.1):
`full_history` threads the prior exchanges of the *same* session. States that declare no session share
the run's default session (so a plain workflow threads history across all its states); a distinct
`environment.session` isolates a subtree's conversation.

Injecting a preamble and *reading a transcript as data* are different things. The `environment.conversation`
block is the preamble; a `{ "conversation": "<session>", "message": n }` binding (§4.2) wires a
transcript — or one message of it — into a slot as an ordinary value, which is how a state summarizes or
answers questions about an earlier session.

## 5. State File Format

State files are declarative JSON for the MVP.

The implementation may later support JSON5, YAML, or editor-friendly syntaxes,
but the MVP should define canonical JSON semantics first.

### 5.1 Top-Level Fields

```text
id
label
description
inputs
outputs
operation
environment
children
sequence
transitions
limits
```

### 5.2 Field Summary

`id`
: State ID, equal to relative path without suffix.

`label`
: Human-readable state name.

`description`
: Short explanation of the state purpose.

`inputs`
: Declared slots for values the state receives — including configuration knobs (inputs with a `default`, §4.5).

`outputs`
: Declared slots for values the state emits.

`operation`
: The state's single operation (§7.1): `{ "kind": "prompt", … }` (a structured model call from an
  inline `template` or a named `skill`) or `{ "kind": "function", "function": "<name>", … }` (a
  registered function — host code, an interactive UI component, a sub-workflow, or a delegated
  agent adapter).

`environment`
: Execution-environment configuration for that operation: `session`, `tools`, `conversation`
  preamble, and `permissions`.

`children`
: Declared child states, their input wiring, and async flags.

`sequence`
: Default order for child execution.

`transitions`
: Local transition rules.

`limits`
: Iteration and timeout limits.

## 6. Transition Expressions

Expressions use a small language with JavaScript evaluation semantics:
equality, comparison, and truthiness behave exactly as in JavaScript. One
deviation: property access on `undefined` or missing values yields `undefined`
instead of throwing (implicit optional chaining), so `children.x.outputs.y` is
safely `undefined` when `x` has never started.

References to a child that has started but not yet finished are pending, not
`undefined`: a transition using a pending reference is skipped for that
evaluation round, and input wiring using one waits for it to resolve
(Section 10.4).

The same language is used for transition conditions (`when`), for `{ "expr": … }` binding leaves
(§4.2), and for `{{…}}` interpolation in prompt templates. It is no longer the wiring default:
ordinary data references are structured bindings, not expression strings, so the DSL survives only
where a computation is genuinely needed.

The expression language should support:

- Literals.
- Identifiers.
- Property access.
- Parentheses.
- Unary `!`.
- Binary comparison operators.
- Boolean operators.
- Conditional (ternary) expressions.
- Numeric comparisons.
- String comparisons.
- `.length` for arrays and strings.

Supported operators:

```text
==
!=
===
!==
<
<=
>
>=
&&
||
!
?:
```

The expression language must be pure. It must not support:

- Arbitrary JavaScript execution.
- Function calls.
- Imports.
- Mutation.
- Loops.
- Filesystem access.
- Network access.
- Async execution.

### 6.1 Expression Context

Expressions may read from a controlled context. The namespaces split by **role**.

The **ref vocabulary** — the data namespaces authored bindings address. They are readable from
`{ "expr": … }` leaves and from guards too, since an expr leaf is itself a producer over the same
data:

```text
inputs.*
outputs.*
children.<id>.outputs.*
children.<id>.outcome
artifacts.*
conversations.*
```

The **guard-only scalars** — control-flow state, never addressable by a reference binding,
reachable only from `when` guards and `{ "expr": … }` leaves:

```text
run.iteration
limits.max_iterations
limits.timeout
```

There is no `function.*` namespace. A function operation's result is an ordinary state output, so
guards read `outputs.*` and `children.<id>.outputs.*` uniformly.

A reference whose root is not one of these namespaces is a validation error, as is a reference to
an undeclared input, output, or child.

### 6.2 Static Validation

A workflow is validated before it is accepted for execution. Errors block execution; warnings do
not. Beyond the structural checks (a child naming an unknown state, a required child input left
unwired, a duplicate `sequence` entry, a transition target that is neither a declared child nor a
`terminate.*` outcome, an unknown slot `kind`), three checks make the wiring itself type-safe.

**Binding compatibility.** Every binding is checked against the schema of the slot it fills:
`isSubschema(producer, consumer)` — "is every value the producer can emit necessarily valid for
this consumer?". The producer's schema is read from what the binding lowers to:

- a `{ "text" }` / `{ "json" }` literal — the exact value, as a `const`-constrained schema, so a
  literal satisfies an `enum`-constrained consumer;
- a producer edge on a declared child — that child's declared outputs as one object schema
  (each output's own schema, required unless it is `optional` or has a `default`);
- `{ "child", "output" }` — the named property's schema; selecting an output the child does not
  declare is an error;
- `{ "input" }` — the declared slot's own schema;
- `{ "expr" }` — the inferred result type (below);
- `{ "artifact" }` / `{ "conversation" }` — session-owned resources whose contents are known only
  at run time, so the check defers to run-time validation.

An unconstrained consumer slot accepts anything and is skipped. Compatibility is sound structural
subtyping (object width honoring `additionalProperties: false`, `required` coverage, `integer` ⊆
`number`, `enum`/`const` ⊆) and **conservative** otherwise: an unmodeled keyword or a union
rejects with a precise reason rather than passing silently.

**Expression typing.** Every `when` guard and every `{ "expr": … }` leaf is type-inferred against
the namespaces of §6.1: member access projects property schemas, comparison and `!` yield boolean,
`&&`/`||` and `?:` yield the join of their branches, and a literal infers to its exact value — so
`cond ? 'complete' : 'blocked'` infers as the enum `["complete", "blocked"]` and satisfies an
enum-constrained slot instead of widening to `string`. A guard **must infer to boolean**: this is
strict, with no truthiness coercion, so a `when` that infers to a number is a validation error
rather than a falsy surprise at run time. A `schema` declared on an `{ "expr" }` leaf is an
*assertion*, checked against the inferred type; it is not the only source of typing.

**Reachability.** The *type* of a producer edge is always statically known; whether the producer
has *run* by the time the edge is resolved is a control-flow property, settled by definite-assignment
analysis over `sequence` and `transitions`. The rule is strict: a reference to a child not proven
to have run on every path reaching its evaluation point is an **error**, so an absent value never
propagates silently. Reading a child's outputs from an expression carries the same obligation as
wiring it.

- Members of `sequence` are proven, in order.
- An `async` sequence member is also proven: async means "started but not awaited", so its outputs
  may be *pending* at read time — and pending is a run-time park (the dataflow join, §10.4), not a
  permanently-missing value. The engine parks the consumer until the producer resolves.
- A child reachable only through a conditional transition is **not** proven.
- `optional: true` or a `default` on the *consuming* slot is the explicit opt-out: both declare
  that an absent value is acceptable here.

The analysis is deliberately conservative: it proves ordered sequences and refuses everything else.

Warnings, which do not block execution, cover the cases that are suspicious rather than wrong: a
child state that is not a descendant path of its parent (legal, so shared library states stay
expressible), a transition back into a `sequence` member with neither `limits.max_iterations` nor a
`run.iteration` guard, a prompt operation with neither a template nor a skill, and a state that
declares no operation and no children.

Static validation cannot settle values, only types. Run-time validation of actual values against
declared schemas (a nondeterministic producer can emit anything) remains at every boundary.

## 7. Operations and the Execution Environment

### 7.1 The Operation and Its Environment

A state declares at most one `operation`, of one of two kinds.

A **prompt operation** is one structured model call:

```text
kind        "prompt"
prompt      { "template": "…" } or { "skill": "<name>" } — exactly one. Both render with
            {{inputs.*}} interpolation; a skill resolves through registry.skills.
system      Optional system prompt.
config      The model-configuration surface (model, sampling, configRef, …).
input       Slots (§4.1) feeding the call; a bound slot is resolved before the call runs. The op's
            resolved inputs ARE the template's {{inputs.*}} scope, so a render variable (e.g. a
            skill invocation's arguments) is just a bound input.
output      The operation's output slot. Defaults to one object slot built from the state's
            declared outputs — which is what a `{ "child", "output" }` binding projects against.
```

A **function operation** invokes a registered function:

```text
kind        "function"
function    A name in registry.functions.
config      The authored surface for this invocation, bound as the operation's `config` input.
input       Slots feeding the call.
output      As above.
```

**There is no separate runtime concept.** A delegated agent runtime (`claude-code`, and future
adapters) is a plain function operation naming a registered adapter. So are sub-workflows,
composite units, interactive UI components, and pure host transforms — one op shape for all of
them. What distinguishes them is the **capabilities of the resolved registry entry**
(`mutatesWorkspace`, `memoizable`, `policyEnforcement`, …), never the shape of the operation. The
`llm` runtime is not one of these entries: a prompt operation is dispatched to an injected prompt
`Executor` instead, which is the same seam every other executor implements.

Adapters are therefore still adapters with capabilities, not interchangeable strings: a bare model
call and a file-editing agent differ in what they can do. See [DESIGN.md](DESIGN.md) §4.4 for the
composed-vs-delegated distinction.

Everything about *how* an operation runs — as opposed to what it is — lives in the sibling
`environment` block, because it is not part of the operation's identity:

```text
session       Logical session id this state runs under; owns the conversation transcript,
              workspace, and permissions. Same id across states ⇒ a shared session; absent ⇒
              the run's default session.
tools         Logical names of tools the operation may call mid-loop, resolved through
              registry.tools. A composed prompt operation runs them in a bounded loop; a
              delegated agent is handed the allow-list.
conversation  { "mode": "full_history" | "summary" | "fresh" | "selected_artifacts",
                "artifacts": [ … ] } — the preamble injected into this call (§4.7).
permissions   The authored per-state permission baseline: `profile`, `default`, per-tool modes
              (§7.4).
```

### 7.2 Agent Responsibilities

An agent operation (e.g. a delegated local code agent) may:

- Inspect the project.
- Modify files within the project directory.
- Run allowed commands.
- Produce artifacts.
- Produce structured outputs matching the state schema.
- Propose transitions when allowed by the state.

An agent operation may not:

- Bypass human approval gates.
- Mutate workflow state directly.
- Manipulate Git history.
- Access files outside the project directory.
- Execute blocked commands.

### 7.3 Operation State Example

```json
{
  "id": "feature/plan/critique",
  "label": "Critique Plan",
  "description": "Review the current plan for significant weaknesses.",
  "inputs": {
    "plan_doc": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" }
    },
    "severity_threshold": {
      "schema": {
        "type": "string",
        "enum": ["minor", "significant", "critical"]
      },
      "default": "significant"
    }
  },
  "outputs": {
    "outcome": {
      "schema": {
        "type": "string",
        "enum": ["clean", "needs_changes", "blocked"]
      }
    },
    "weaknesses": {
      "schema": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "critique_report": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" }
    },
    "human_decision": {
      "schema": {
        "type": "string",
        "enum": ["approve", "request_changes", "block"]
      },
      "optional": true,
      "binding": { "child": "human_review", "output": "decision" }
    }
  },
  "operation": {
    "kind": "prompt",
    "config": { "model": "critic" },
    "prompt": {
      "template": "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema."
    }
  },
  "environment": {
    "conversation": {
      "mode": "full_history"
    }
  },
  "children": {
    "address_weaknesses": {
      "state": "feature/plan/critique/address_weaknesses",
      "inputs": {
        "plan_doc": { "input": "plan_doc" },
        "weaknesses": { "expr": "outputs.weaknesses" },
        "critique_report": { "expr": "outputs.critique_report" }
      }
    },
    "human_review": {
      "state": "feature/plan/critique/human_review",
      "inputs": {
        "plan_doc": { "input": "plan_doc" },
        "critique_report": { "expr": "outputs.critique_report" }
      }
    }
  },
  "transitions": [
    {
      "to": "terminate.success",
      "when": "children.human_review.outcome === 'success'"
    },
    {
      "to": "terminate.success",
      "when": "children.address_weaknesses.outcome === 'success'"
    },
    {
      "to": "terminate.success",
      "when": "outputs.outcome === 'clean'"
    },
    {
      "to": "human_review",
      "when": "outputs.outcome === 'blocked'"
    },
    {
      "to": "address_weaknesses",
      "when": "outputs.outcome === 'needs_changes'"
    }
  ]
}
```

Execution walk-through: the state's operation runs first. `clean` terminates
immediately, before any child runs. `needs_changes` runs one fix pass and then
terminates so the parent can decide whether to re-plan. `blocked` collects a
human decision, surfaced through the `human_decision` output — an output derived
from a binding rather than produced by the operation, and `optional` because the
child that produces it runs only on the conditional path (§6.2). The
child-completion transitions are declared first so the evaluation that runs
after a child completes does not re-enter it. The retry loop lives in the
parent (Section 9), which re-runs the whole planning pass and gets a fresh
critique instance each time.

Handing the same state to a delegated agent instead of the prompt runner changes only the
operation block — the slots, wiring, children, and transitions are untouched:

```json
{
  "operation": {
    "kind": "function",
    "function": "claude-code",
    "config": { "permissionMode": "plan" },
    "input": {
      "prompt": {
        "binding": { "text": "Review the plan document and report significant weaknesses." }
      }
    }
  },
  "environment": {
    "session": "planning",
    "tools": ["read_file"],
    "permissions": { "profile": "plan" }
  }
}
```

The adapter reads its instruction from the `prompt` input and its authored surface from `config`;
the engine hands a delegated adapter raw tools, because such an entry declares that it authorizes
its own loop's calls (§7.4).

### 7.4 Tool-Call Permissions

When an operation is given tools, each tool call is authorized by a **profile × mode** (full detail in
[DESIGN.md](DESIGN.md) §5.1):

- **profile** — which effects are in scope: `read-only`, `plan`, or `full`. A `read-only`/`plan` profile
  admits only tools that declare themselves read-only; `plan` stays read-only until a human approves an
  exit that rebinds the session to `full`.
- **mode** — how an in-scope call is authorized: `allow`, `deny`, or `ask`. An `ask` invokes a human
  approval gate whose decision persists at a chosen scope — `once`, `always this session`, `always this
  workflow run`, or `always` (the host process) — all in-memory; durable policy is authored, not decided.

A state authors its starting policy via `environment.permissions` (`profile` / `default` / per-tool
modes), overriding a workflow-wide default; live human decisions overlay on top, most-specific scope
winning. The gate is only active when the host supplies an approver; otherwise tools run unguarded. A
delegated adapter — a registry entry declaring that it enforces policy through its own callback —
receives raw tools and routes its native approval callback back through the same approver, so it is
gated once, not twice.

## 8. Function States (Interactive UI)

Human interaction is modeled as a `function` operation whose registered function is interactive — not as a
special human runtime. An interactive function displays state inputs and returns structured data;
transition logic remains in the state file. (Non-interactive functions — pure transforms, data fetches,
validators — use the same operation kind; a UI component is just the interactive case, marked by the
registry entry's capabilities, not by the operation.)

### 8.1 MVP Built-In Components

Suggested MVP interactive functions:

- `choose_option`
- `review_artifact`
- `edit_markdown`
- `fill_form`
- `confirm_action`

### 8.2 Function State Example

```json
{
  "id": "feature/plan/critique/human_review",
  "label": "Human Review",
  "inputs": {
    "plan_doc": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" }
    },
    "critique_report": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" }
    }
  },
  "outputs": {
    "decision": {
      "schema": {
        "type": "string",
        "enum": ["approve", "request_changes", "block"]
      }
    },
    "comments": {
      "schema": { "type": "string", "format": "markdown" },
      "optional": true
    }
  },
  "operation": {
    "kind": "function",
    "function": "choose_option",
    "config": {
      "prompt": "Review the critique result.",
      "options": ["approve", "request_changes", "block"]
    }
  }
}
```

The interactive function's authored surface — its prompt text and options — rides the operation's
`config`, bound as an ordinary input; the operation shape gains nothing for being interactive.

This state declares no transitions: once the operation completes, no
operations remain, so the state terminates with `terminate.success` and its
validated outputs. The parent branches on `outputs.decision`.

## 9. Parent State Example

```json
{
  "id": "feature/plan",
  "label": "Planning",
  "inputs": {
    "issue": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" }
    }
  },
  "outputs": {
    "outcome": {
      "schema": {
        "type": "string",
        "enum": ["complete", "blocked"]
      },
      "binding": {
        "expr": "children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'"
      }
    },
    "plan_doc": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" },
      "binding": { "child": "context", "output": "plan_doc" }
    },
    "critique": {
      "binding": { "child": "critique" }
    }
  },
  "children": {
    "goals": {
      "state": "feature/plan/goals",
      "inputs": {
        "issue": { "input": "issue" }
      }
    },
    "context": {
      "state": "feature/plan/context",
      "inputs": {
        "issue": { "input": "issue" },
        "goals": { "child": "goals", "output": "goals" }
      }
    },
    "critique": {
      "state": "feature/plan/critique",
      "inputs": {
        "plan_doc": { "child": "context", "output": "plan_doc" },
        "severity_threshold": { "text": "significant" }
      }
    }
  },
  "sequence": ["goals", "context", "critique"],
  "transitions": [
    {
      "to": "terminate.success",
      "when": "children.critique.outputs.outcome === 'clean'"
    },
    {
      "to": "goals",
      "when": "children.critique.outputs.outcome === 'needs_changes' && run.iteration < limits.max_iterations"
    },
    {
      "to": "terminate.success",
      "when": "children.critique.outcome === 'success'"
    }
  ],
  "limits": {
    "max_iterations": 3
  }
}
```

The re-plan loop lives here: `needs_changes` transitions back to `goals`,
which resets the sequence and clears the recorded results of `goals`,
`context`, and `critique`, so the next pass runs fresh instances (and the
first two transitions evaluate to false until `critique` runs again). When the
iteration limit is reached, or critique reports `blocked`, the final
transition fires and the `outcome` output resolves to `blocked`.

The whole file type-checks statically (§6.2): the `outcome` expression is a conditional over two
string literals, so it infers as the enum `["complete", "blocked"]` and satisfies the slot's
schema; `plan_doc` projects a `markdown` artifact off `context`'s declared outputs; `critique` is
the passthrough case, an unconstrained slot bound to the child's whole outputs object; and every
child edge is reachable-proven because `goals`, `context`, and `critique` are all `sequence`
members.

## 10. Execution Lifecycle

### 10.1 State Run Status

A state run may be in one of these statuses:

```text
queued
running
waiting_for_user
waiting_for_event
sleeping
blocked
failed
completed
canceled
```

### 10.2 State Run Record

Each state run records:

- Task ID.
- State ID.
- Workflow version.
- State file content hash.
- Input artifact references.
- Output artifact references.
- Conversation references.
- Operation kind, target (prompt template/skill, or function name), and configuration.
- Commands requested.
- Commands executed.
- Commands blocked.
- Function data returned.
- Transition decisions.
- Validation errors.
- Start time.
- End time.
- Token or cost metadata when available.

### 10.3 Durable Execution

The engine must persist enough information to resume after app restart.

At minimum, the persisted data must answer:

- Which task was active?
- Which state path was active?
- Was an operation running?
- Was user input pending?
- Which artifacts had been produced?
- Did a transition already occur?
- Which workflow version was in use?

### 10.4 Async Child States

A child entry may declare `"async": true`. When the sequence cursor reaches an
async child, or a transition targets it, the child starts and the engine
immediately continues with the next operation instead of waiting. Multiple
async children may run concurrently.

Waiting is dataflow-driven; no explicit join construct is needed:

- A child whose input wiring references outputs of an unresolved async child
  does not start until those outputs resolve.
- A transition whose `when` expression references outputs of an unresolved
  async child is skipped until that child resolves.
- Each child completion triggers another round of transition evaluation.
- If no transitions apply and no children are left to run, still-running async
  children are waited for. The state terminates only when all children have
  finished and no transition applies.
- Terminating a state cancels any still-running descendant states.

Example — fan-out reviews with a dataflow join:

```json
{
  "children": {
    "claude_review": {
      "state": "review/agent_review",
      "async": true,
      "inputs": {
        "change": { "input": "change" }
      }
    },
    "codex_review": {
      "state": "review/agent_review",
      "async": true,
      "inputs": {
        "change": { "input": "change" }
      }
    },
    "synthesize": {
      "state": "review/synthesize",
      "inputs": {
        "review_a": { "child": "claude_review", "output": "report" },
        "review_b": { "child": "codex_review", "output": "report" }
      }
    }
  },
  "sequence": ["claude_review", "codex_review", "synthesize"]
}
```

Both reviews start without blocking. The sequence cursor reaches `synthesize`
immediately, but its inputs reference both review outputs, so it waits for
both to resolve before starting. With no transitions declared, the state
terminates with `terminate.success` once all three children finish.

The two producer edges into `synthesize` pass the reachability check (§6.2) even though the
reviews are async: an async sequence member is still proven to run, and "in flight" is a run-time
park, not a permanently-missing value.

### 10.5 Branch Isolation and Concurrent Tasks

Tasks may run concurrently. Isolation is Git-based:

- A task may be bound to a Git branch. The engine materializes concurrently
  active branches as separate worktrees, so concurrent tasks never share a
  working tree.
- A task may spawn subtasks that share its branch, for collaborative patterns
  such as multiple agents reviewing the same change. Coordinating writes
  within a shared branch is the workflow author's responsibility in the MVP.
- Async children of a single state run in the task's own branch context.

## 11. Safety Policy

### 11.1 Required MVP Restrictions

Agents are limited to their task's worktree of the project directory.

Agents have no read or write access to the engine-owned `.jaira/` directory,
including workflow state files and run records.

Agents cannot bypass human approval gates.

Agents cannot destroy Git history.

Agents cannot execute denied commands.

### 11.2 Git Operations

Agents may use Git constructively: commit, create branches, switch branches,
and inspect history.

The policy layer blocks operations that destroy history or discard work,
equivalent to:

```text
git push --force
git push --mirror
git reset --hard
git rebase
git filter-branch
git filter-repo
git gc / git prune
git reflog expire / git reflog delete
rm -rf .git
```

The actual implementation should block by parsed command intent, not only by raw
string matching.

### 11.3 Commands Requiring Approval

The MVP should require explicit user approval for:

- Pushes.
- Merges.
- Deployments.
- Package publishing.
- Commands that access network resources.
- Commands that modify global configuration.
- Commands that access secrets.
- Commands that install or execute remote scripts.

### 11.4 Approval Gates

Approval gates are ordinary interactive-`function` states; there is no separate policy mechanism.
The engine guarantees that an interactive function's outputs can only be produced by a
real user interaction in the app, and runtimes have no channel to write another
state's outputs. A transition guarded on such a state's outputs is therefore
user-controlled by construction.

An agent may include a recommendation in its artifacts or outputs, but a
recommendation never satisfies a gate.

## 12. Workflow Versioning

Each task run pins the workflow version it started with.

The MVP can represent a workflow version as:

- The Git commit hash when available.
- The hash of all referenced state files.
- The project-local workflow version ID.

If state files change while a task is active, the task continues using its pinned
version unless the user explicitly migrates it. The engine executes a task from
its pinned snapshot of the state files, not from the current on-disk copies.

## 13. History and Pruning

JaiRA stores state run history by default.

Stored history includes:

- State runs.
- Transitions.
- Agent outputs.
- UI outputs.
- Artifact references.
- Conversation references.

Users may prune historical records.

Pruning should preserve task correctness. A task cannot prune data required to
resume its current active state.

## 14. MVP Scope

### 14.1 Included

- Single local project.
- Single user.
- One file per state.
- Declarative JSON state files.
- Hierarchical state execution.
- Default child ordering.
- Local child-only transitions.
- Async child states.
- Typed termination outcomes.
- Schema-validated inputs and outputs.
- Generic passthrough outputs.
- Built-in UI components.
- Local agent runtime abstraction.
- Conversation artifacts.
- Artifact references.
- Durable state run records.
- Kanban board and sub-board navigation.
- Branch-per-task isolation via Git worktrees.
- Safety policy for local agents.
- Blocking on schema validation failure.

### 14.2 Deferred

- Multi-user collaboration.
- Remote hosted agents.
- GitHub or GitLab integration.
- Deployment automation.
- Arbitrary custom UI components.
- Arbitrary webhooks.
- General DAG execution beyond async child states.
- Cross-project workflow libraries.
- Rich workflow migration tooling.

## 15. Open Design Questions

These questions are not required for the first implementation, but should be
resolved before expanding beyond the MVP:

1. Should artifacts be stored in a reserved JaiRA directory, or can users choose
   project-specific artifact paths?
2. Should the state file extension be `.json`, `.state.json`, or something else?
3. Should state files be allowed to omit `id` and derive it from the file path?
4. What is the minimum built-in UI component set for a useful MVP?
5. Which local agent provider should be implemented first?
6. How should agent adapters report partial progress?
7. Should workflow validation happen continuously, on save, or only when a task
   starts?
8. What is the first task database format: SQLite, plain files, or embedded app
   storage?
9. How should subtasks relate to parent task completion?
10. Should spawned subtasks block parent completion by default?
11. How are skills (named prompt templates) authored and registered into
    `registry.skills` (project-level library format), and what inputs does a
    skill-sourced prompt render take?
12. How does a task get bound to a branch: at creation, by a state, or by the
    user?

## 16. Design Principles

1. State files are declarative.
2. States only know their own children.
3. Children return structured outputs; parents decide what those outputs mean.
4. UI components produce data; state files own transition logic.
5. Agents produce artifacts and proposals; the engine owns workflow control.
6. Transition expressions are pure and limited.
7. Schema validation is mandatory for state outputs.
8. Generic passthrough is explicit.
9. Cycles require iteration limits.
10. Workflow versions are pinned per task run.
11. Human approval gates cannot be bypassed.
12. Agent filesystem access is scoped to the task's worktree.
13. Destructive Git operations are blocked; constructive Git use is allowed.
14. Execution history is stored by default and can be pruned safely.
15. Operations run in fixed priority order: the state's single operation, then
    children in `sequence` order.
16. Entering a state creates a fresh instance; results never leak across
    instances.
17. Workflow definitions are engine-owned; agents cannot read or modify them.
