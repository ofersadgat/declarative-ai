# Hierarchical Workflow Specification

> **Provenance and scope.** This file is the canonical specification of the
> hierarchical-workflow formalism implemented by `@declarative-ai/hw`, migrated from
> `JaiRA/SPEC.md` (which remains the product spec for the JaiRA app).
> Normative for this library: §2.3–§2.4 (states, state IDs), §3 (state machine
> semantics), §4 (inputs/outputs/artifacts/conversations), §5 (state
> file format), §6 (expressions and static validation), §7.3 and §9 (worked examples,
> used as golden tests), §7.5 (function definitions — the callee side of §6, including the
> TypeScript signature extractor, the wire/marshalling boundary, and the hash-and-freeze
> integrity model), §8 (function states / interactive UI, realized here as
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

Child states are entered by sequence order or by explicit transition. `children` may be omitted
entirely, in which case they are INFERRED from the state's own namespace: every state one path
segment below it, keyed by basename, in alphabetical order. `"children": {}` declares none.

A state may declare an operation, children, or both. Operations run one at a time
in a fixed priority order: the state's `operation`, then child states in
`sequence` order. The cursor HOLDS on the child it entered until that child resolves, so one child
runs at a time; `async: true` on a child is the sole exception and the sole meaning of the flag.
Concurrency is something an author asks for, never what a plain sequence falls into.

How an operation runs — the session it belongs to, the tools it may call, the conversation
preamble it receives, its permission baseline — is declared in the operation itself (§7.1). The
sibling `environment` block carries DEFAULTS for the operation instead, inherited by this state and
every descendant.

### 3.3 Evaluation Loop

1. Entering a state creates a new state instance. Declared inputs are resolved
   and validated; validation failure blocks the state.
2. The engine runs the highest-priority operation that has not yet run in this
   instance.
3. When an operation completes, its outputs are validated and transitions are
   evaluated. A child that finished since the last evaluation contributes its
   OWN `transitions` first — in the order the state runs its children — and the
   state's list is considered after them. Within each list, declared order; the
   first transition whose `when` expression is true is taken.
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

A transition written on a child mount (`children.<key>.transitions`) says the
same thing structurally, and is the better place for a rule that is about one
child: it is eligible ONLY in the round that child's completion triggered, so a
child that finished earlier no longer diverts anything and an unconditional `to`
means "after this child, go here" rather than "from now on, always go here".

The guarantee is **once per completion**: every child that finishes has its list
evaluated exactly once, in the first evaluation round that STARTS after it
finished. A round fixes the set of children it answers for before it runs any
guard's call, so a child that finishes while a round is already in flight is
answered by the next one rather than being swept into a round whose guards were
prepared before it existed. Two async children may therefore share a round or take
one each — nothing an author writes should depend on which — and a looped child is
evaluated once per pass, because the eligibility is granted per completion and not
per child. Its
guards resolve in the enclosing state's scope like any other — this child is
`.children.<key>` from there, spelled out. Taking one HANDLES that child's
`error`/`timeout` termination, which is what makes per-child recovery expressible
without a state-level guard for each child:

```json
{
  "children": {
    "implement": {
      "state": "feature/implement",
      "transitions": [
        { "to": "repair", "when": ".children.implement.outcome === 'error'" }
      ]
    },
    "repair": {
      "state": "feature/repair",
      "inputs": { "failed_change": ".inputs.change" }
    },
    "review": { "state": "feature/review" }
  },
  "sequence": ["implement", "review"]
}
```

`implement` fails, its own rule routes to `repair`, and because a taken transition
handles the failure the state does not terminate with `error`. `repair` is not a
sequence member, so entering it does not move the cursor: when it finishes, the
spine resumes at `review`. When `implement` succeeds the guard is false, nothing
else matches, and the cursor reaches `review` the same way.

**Write a transition on the mount by default.** The state's own `transitions` are
for the rules that hold *whichever* child just finished, or none did — a decision
made from the state's own operation output, an entry into a child, an iteration
limit. A rule that is about one child belongs on that child, and the difference is
not stylistic: written at state level it has to name the child in its guard, stay
correctly ordered against every other rule in the list, and be re-evaluated after
every unrelated child completion. Written on the mount, the round it is eligible
in is already the one it is about.

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

#### A guard that WAITS stops the list

Skipping is right for a guard that reads a child still running: the guard is a
question about data, the next transition is a different question, and the round
can answer that one meanwhile.

A guard may instead CALL an operation that waits on something outside the run —
a person, an event, a deadline. A registered function declares this with the
`deferred` capability, and it changes who waits: the engine STARTS such a call
and reads its answer in a later round, rather than awaiting it inside the round
that needed it. A call that may never finish must not hold a round open, because
a round holding open is a state that can neither report what it is waiting for
nor be woken by anything else that happens to it.

While that answer is outstanding the guard is `PENDING`, and this `PENDING`
**stops the evaluation and does not skip**:

- no later transition in the same list is considered, and neither is the state's
  own list if the waiting rule was on a child mount;
- the round consumes no eligibility — the children it was answering for are still
  owed an answer, and they get it from the round that runs when the call settles;
- the state does not terminate. A state with nothing left to run WAITS instead of
  succeeding, which is what makes it *paused on a decision* rather than one that
  quietly finished while somebody was still looking at it;
- the sequence does not advance. Nothing new starts in a state that is waiting to
  be told where to go.

The reason is that such a guard is not a question about data. It is a DECISION
that has been asked for and not yet made, and every transition after it is a rule
about what to do given that decision. Letting a later rule fire while the answer
is outstanding would take a branch the author wrote to be considered only if the
wait came back false.

**A wait is only STARTED for a rule the round can reach.** Guards are prepared in
the same order they are evaluated, and the preparation stops where the evaluation
will: at the first rule that fires (unconditional, or a guard already true) and at
the first rule that waits. A rule behind either of those is a rule about a decision
this round will not reach, so its call is not made — which for an ordinary call
saves the work, and for a deferred one is the difference between one outstanding
question and several. Two offers on a screen for one decision, only one of which
does anything, is not a cosmetic problem: it is a person told they may do something
the engine would ignore.

**A taken transition CANCELS the waits it did not answer.** A rule earlier in the
list can become true in a later round while a rule behind it is still waiting — the
list is walked from the top each time. When that earlier rule fires, every deferred
call the instance still has outstanding is cancelled, because the decision they were
asking about has been made by something else. The same happens when the instance
ends for any reason: a terminated, timed-out or superseded state leaves no
registration standing.

**A taken transition CONSUMES the answer.** The result of a deferred call is
forgotten when the instance takes a transition, and a call still in flight when
one is taken is cancelled. An event is not a memo: a guard that kept reading the
same `true` would re-take the same transition on every following round, and an
offer nobody withdrew would stand for a state that had already moved on. The next
round asks again, which registers a fresh wait — so a rule that says "let them do
it again" means that, rather than firing on the memory of the last time.

A wait is recorded in the run journal as a `call.waiting`/`call.settled` pair, so
a run parked on a person does not read as one that hung.

### 3.4 State Instances

Entering a state — including re-entering it via a transition — creates a fresh
instance. Outputs and child results belong to the instance. References such as
`children.<id>.outputs` resolve to the most recent instance of that child
within the current parent instance, and evaluate to `undefined` if the child
has not run.

`run.iteration` is the count of transitions taken so far within the current
state instance, starting at 0. It is the standard guard for cycles.

`run.cursor` is the key of the child most recently ENTERED — where the sequence cursor is — and
`run.position` its index in the sequence (`-1` before any child is entered). Transitions are
evaluated after an operation completes or a child terminates, so "the cursor is at x" means x has
run; a guard can therefore say *if we are at x and y holds, go to z*.

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
      "plan_doc": ".children.context.outputs.plan_doc",
      "goals": ".children.goals.outputs.goals"
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
| `{ "child": "context" }` | The child output named by THE SLOT BEING BOUND. | a producer edge on the child, plus a `select` producer projecting that property |
| `".children.context.outputs.plan_doc"` | One named output of the child. | the same, projecting `plan_doc` |
| `".children.context.outputs"` | The child's whole outputs object, as one value. | `{ "op": "context" }` |
| `".inputs.issue"` | This state's declared input, by name. | a `scope.get` producer |
| `{ "expr": ".outputs.weaknesses" }` | A small computation in the expression DSL (§6). | an `expr.eval` producer whose output schema is the inferred type |
| `".artifacts.design_doc"` | A session-owned artifact, by name. | an `artifact.get` producer |
| `"messages(<session ref>)"` | A conversation, by ref — a session is a position, not a name. | a `conversation.get` producer |

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
      "binding": ".children.context.outputs.plan_doc"
    },
    "outcome": {
      "schema": { "type": "string", "enum": ["complete", "blocked"] },
      "binding": { "expr": ".children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'" }
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

`output` DEFAULTS to the name of the slot being bound, because
`"plan_doc": { "binding": ".children.context.outputs.plan_doc" }` says the same word twice
and the repeat is the one people forget to change. `"*"` is the escape hatch for the whole object.

Example — the three forms side by side:

```json
{
  "outputs": {
    "plan_doc": { "binding": ".children.context.outputs.plan_doc" },
    "summary": { "binding": ".children.context.outputs.notes" },
    "child_outputs": { "binding": ".children.critique.outputs" },
    "ctx_*": { "binding": ".children.context.outputs.binding" }
  }
}
```

The last is a **spread**: a slot key ending in `*` republishes every output of the named child as an
output of this state, prefixed with whatever precedes the `*` (so `ctx_plan_doc`, `ctx_notes`, …),
each keeping its own schema and optionality. A bare `"*"` spreads them unprefixed, and an explicitly
declared slot always wins over one a spread would have produced.

The marker is in the slot KEY rather than in `output` because a spread declares N slots and that has
to be visible where the slots are declared — and because `{ "output": "ctx_" }` could not be told
apart from selecting an output genuinely named `ctx_`.

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
(`".inputs.severity_threshold"`), by an expression (`inputs.severity_threshold`), or by
prompt interpolation (`{{.inputs.severity_threshold}}`).

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

A state selects its mode in `operation.conversation` (§7.1); the selected preamble is injected
into that state's own call. It may be inherited from an ancestor's `environment`, so a subtree can
be put on one mode in one place.

Transcripts are scoped per **session** (`operation.session`, DESIGN.md §1.6/§5.1):
`full_history` threads the prior exchanges of the *same* session. A state that declares no session gets
its OWN conversation — there is no run-wide default to fall into, since an implicit shared transcript is
what drives unbounded context growth. Threading across states is asked for, by naming a session once at a
common ancestor's `environment` and letting the inheritance chain carry it down.

Injecting a preamble and *reading a transcript as data* are different things. The `operation.conversation`
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
: DEFAULTS for `operation` — the same shape with every field optional, inherited by this state and
  every descendant, nearest layer winning (§7.1a). Only a state that declares an `operation` gets
  one.

`children`
: Declared child states, their input wiring, async flags, per-mount `environment`
  defaults, and per-mount `transitions` — considered when that child finishes,
  ahead of the state's own list (§3.3).

`sequence`
: Order the cursor advances through children. Optional — absent means the order the children were
  declared in; `[]` means no spine, so a child runs only when a transition enters it.

`transitions`
: Transition rules that apply to the state as a whole — a decision from its own
  operation output, an entry into a child, an iteration limit. A rule about one
  child belongs on that child's mount instead (§3.3), which is the default.

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
- Object literals — `{ to_state: 'deploy', urgent: .inputs.severity > 2 }`. Keys are bare
  identifiers or quoted strings, never computed (`get(o, k)` is the spelling for a computed read);
  values are full expressions. The aggregate literal, standing beside the scalar ones — its use is
  an OPTIONS BAG at a call site, where naming what an argument means beats counting positions, and
  where a call whose options grow renumbers nobody.

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

The expression language is CLOSED — it has no way to reach anything the host did not put in front
of it. It must not support:

- Arbitrary JavaScript execution.
- Imports.
- Mutation.
- Loops.
- Filesystem access.
- Network access.

⚠️ **"No function calls" and "no async execution" no longer hold, and were the wrong line to draw.**
An expression may CALL an operation resolved along the search path (EXPRESSIONS.md §3) — which is
not an escape from the sandbox but the opposite: the callee is a document or a registered function
the host chose to make available, resolved by name, exactly like the built-in operators it is
indistinguishable from. What purity buys — no ambient authority, no mutation, no way to name
something that was not offered — is preserved by that resolution rule and not by the absence of
parentheses.

A call may also WAIT (§3.3, "A guard that WAITS stops the list"). The expression itself stays
synchronous: it resolves to `PENDING` and is re-resolved in a later round, which is the same
protocol a reference to a running child already follows. Nothing in the language awaits anything.

A callee may also be a **user's own js/ts module** (§7.5), which genuinely does step outside the
closure above: it can import, mutate, and reach the filesystem. That is not an exception smuggled
into this section but a different guarantee, made elsewhere and by different means. The language
stays closed because nothing it can *name* is unvetted — a module runs only when its content hash
carries an approval, and only from the frozen copy taken before the run started (§7.5.5). Purity is
preserved for expressions; for modules, provenance replaces it.

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
operation.*
children.<id>.operation.*
artifacts.*
```

`operation.*` is the state's **own** call as an addressable node, which is what makes engine metadata
reachable without going through the events journal. It is a namespace rather than a child on purpose:
a child would perturb the instance tree and make `run.cursor` / `run.position` / `sequence`
ambiguous. Its shape is a **typed union** — a common core on every kind plus llm-only extras — so
`operation.output.session` on a `ui` operation is a load-time authoring error rather than a runtime
`undefined`:

| Field | Kinds | Notes |
| --- | --- | --- |
| `outcome` | all | `success` \| `error` \| `timeout` \| `canceled`; mirrors `children.<id>.outcome` |
| `usage` | all | the measurement record, passed through rather than re-shaped |
| `cost` | all | USD — lifted out of `usage` because it is the field asked for by name, and a *failed* call still spends money and still reports it |
| `model` | all | the model the call was actually made with, post-resolution |
| `outputs.session` | prompt only | a `SessionRef` — `{ id }`, plus `end`, which is another `SessionRef` |

`provider` and `attempts` are **absent on purpose**. Neither reaches the engine's seam today — they
are things an executor knows and does not report — so declaring them would hand the lint a field it
could never resolve and every author a value that is always `undefined`.

The ref schema is closed (`additionalProperties: false`), which is what makes
`operation.output.session.position` — a plausible thing to reach for, given how the notation reads —
a lint error rather than a runtime `undefined`. A ref is opaque, and the schema says so.

**`operation.output.session` is the END position.** A call appends *at* a position but does not know
its end until the provider resolves, so the end marker is the only value that can exist when the
engine reads it — and it is what a consumer actually wants ("append after me", "fork after me").
There is deliberately no start marker: recovery after an error does not need one, since instance-scoped
resolution (§4.7) forks from the right place on its own.

**`operation.output.session.end` is the same conversation with no position.** The two are what an
author chooses between when they wire a conversation into a later state's `session`:

| Written | Means | When the conversation moved on in between |
| --- | --- | --- |
| `.operation.output.session` | continue from exactly the point this call ended at | branches, so the later state never silently inherits turns it was not shown |
| `.operation.output.session.end` | continue from wherever the conversation has got to | appends after them — three states in a row are one thread, not a fork each |

A conversation reaches a later state as DATA — the parent wires the ref into the consumer's input, and
the consumer names that input as its session — so the choice is made at the point of wiring:

```json
{
  "children": {
    "plan": { "state": "feature/plan" },
    "refine": {
      "state": "feature/refine",
      "inputs": { "thread": ".children.plan.operation.output.session.end" }
    },
    "explore_risks": {
      "state": "feature/explore",
      "inputs": { "thread": ".children.plan.operation.output.session" }
    },
    "explore_cost": {
      "state": "feature/explore",
      "inputs": { "thread": ".children.plan.operation.output.session" }
    }
  },
  "sequence": ["plan", "refine", "explore_risks", "explore_cost"]
}
```

Each consumer declares `"session": { "expr": ".inputs.thread" }`. `refine` CONTINUES the planning
thread — `.end` resolves at the head, so it appends after `plan` and would append after anything else
that had spoken since. The two `explore_*` children wire the POSITION instead, and `refine` has since
claimed it: each branches from the point immediately after `plan`, so each sees `plan`'s turn and
neither sees `refine`'s or the other's. One primitive does both jobs, and the ref that was wired is
what decides which.

`end` is the ONE property a ref carries beyond `id`, and it is a real value, materialized when the
node is published — not a marker interpreted at the consuming site. So it survives a spread, a
round-trip through JSON, and the events journal, and it is an ordinary `SessionRef` everywhere
downstream. There is no `.end.end`: `end` is already unpositioned, so a second hop is a lint error.

A plain session **name** already behaves as `end` does — a name is unpositioned by construction — so
this distinction only arises where a conversation reaches a state as data.

Note the granularity. **Authored forking is per-operation**: one agentic call that appends forty
entries cannot be branched at entry twenty from a workflow. A store may address finer positions so a
human can scrub a transcript in a UI, but the expression language exposes operation boundaries only.
`{}` before the operation has run, so a guard reading it early sees absence rather than an error.

The **guard-only scalars** — control-flow state, never addressable by a reference binding,
reachable only from `when` guards and `{ "expr": … }` leaves:

```text
run.iteration
run.cursor
run.position
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

**An output's schema is optional, and an undeclared one is INFERRED from its binding.** A declared
schema constrains twice — the binding filling the slot is checked against it, and so is every
consumer reading it. Declaring none must not therefore mean "the top type": a schema every value
satisfies is one no typed consumer accepts, so an undeclared output would be unwirable into
anything typed, and the error would be reported against the parent, about a slot the parent did not
write — making the declaration mandatory wherever the value was actually used. So an output with a
binding and no schema takes the producer type of that binding, computed by the same rule as above,
in the declaring state's own scope. An output with NO binding is filled by the operation, whose
result is not statically typed, and stays unconstrained — genuinely unknown rather than merely
undeclared. Inference is cut off at a state already being inferred, so a child that mounts an
ancestor terminates rather than looping.

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
declares no operation, no children **and no bound output**.

That last one takes all three. An output with a binding is resolved when the state terminates
(§3.7), so a state whose outputs bind is a pure computation — it has no operation and no children
*because* it needs neither, and terminating immediately is precisely what it is for. Warning on that
shape made the message unreadable in the workflows that lean on it: a scoring state over signals
already in the run, or an arithmetic verdict over its inputs.

Static validation cannot settle values, only types. Run-time validation of actual values against
declared schemas (a nondeterministic producer can emit anything) remains at every boundary.

## 7. Operations and the Execution Environment

### 7.1 The Operation and Its Environment

A state declares at most one `operation`, of one of two kinds.

A **prompt operation** is one structured model call:

```text
kind        "prompt"
prompt      { "template": "…" } or { "skill": "<name>" } — exactly one. Both render with
            {{.inputs.*}} interpolation; a skill resolves through registry.skills.
system      Optional system prompt.
config      The model-configuration surface (model, sampling, configRef, …).
input       Slots (§4.1) feeding the call; a bound slot is resolved before the call runs. The op's
            resolved inputs ARE the template's {{.inputs.*}} scope, so a render variable (e.g. a
            skill invocation's arguments) is just a bound input.
output      The operation's output slot. Defaults to one object slot built from the state's
            declared outputs — which is what a `{ "child", "output" }` binding projects against.
```

A **function operation** invokes a registered function:

```text
kind        "function"
function    A name in registry.functions.
args        The authored arguments, bound to the call's input slots BY NAME. Shorthand for `input`
            where the value is a constant and there is nothing to say about its type; a slot the
            author declared in `input` wins. They used to arrive as one blob in a slot called
            `config`, because a registered function had no way to declare named parameters and
            there were no slots to bind to — there are now (§7.5.2), and an impl reads `inputs.mode`.
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

Everything about *how* an operation runs — as opposed to what it is — is written in the operation
alongside the rest, because each of these is a per-CALL decision:

```text
session       The conversation this call joins (DESIGN.md §1.6). A NAME shares an
              append-only stream by declaration; {"expr": …} names an exact position
              computed at run time, normally from `operation.output.session` — or its
              `.end`, the same conversation with no position, which continues the thread
              rather than branching from a point (§6.1); `null` starts a fresh one, and
              absent means this state gets its own. The DECLARED name separately keys the
              state's workspace and permissions, which are inherited when nothing is
              declared.
tools         Logical names of tools the operation may call mid-loop, resolved through
              registry.tools. A composed prompt operation runs them in a bounded loop; a
              delegated agent is handed the allow-list.
conversation  { "mode": "full_history" | "summary" | "fresh" | "selected_artifacts",
                "artifacts": [ … ] } — the preamble injected into this call (§4.7).
permissions   The authored per-operation permission baseline: `profile`, `default`, per-tool modes
              (§7.4).
```

#### 7.1a Inheritance

`environment` is an `operation` with every field optional. A state's effective operation is

```text
merge(root.environment, …, parent.environment, mount.environment, own.environment, own.operation)
```

with the nearest layer winning, so a root can set the model, the session and the tool set once for
a whole subtree. Only a state that DECLARES an `operation` gets one — `{}` is the opt-in to a fully
inherited one, and without it a pure composite under an `environment`-declaring root stays a pure
composite instead of inheriting an operation and running it.

`mount.environment` is `children.<key>.environment`: a layer the parent applies to ONE child rather
than to all of them. Without it the chain was per-state, so two children of one parent could not
differ in it — which is exactly what "review this change with two different agents" needs (§10.4).
It sits under the child's own layers, so a state that names its own `operation.function` still wins;
a state meant to be mounted under several runtimes leaves that field to the chain.

A state mounted under two parents that give it different environments is running as two different
things, so it loads as two entries: the first mount keeps the plain id and any later one that
inherits something different gets a `#`-suffixed VARIANT id, hashed from the inherited environment.
Two parents passing the same environment collapse back onto one entry. Everything downstream — the
checker, snapshots, the event log — goes on seeing one id with one operation per state.

Most fields merge per key (`config`, `input`, `permissions.tools`); `prompt`, `schema` and `binding`
are replaced whole, because merging them produces documents that are not prompts or bindings at all.
Arrays (`tools`) replace, which is what makes `[]` the way to drop an inherited tool. A layer that
changes `kind` drops the inherited `config`/`prompt`/`system`/`function`, since those mean different
things per kind.

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
      "binding": ".children.human_review.outputs.decision"
    }
  },
  "operation": {
    "kind": "prompt",
    "config": { "model": "critic" },
    "conversation": { "mode": "full_history" },
    "prompt": {
      "template": "Review the plan document. Find significant weaknesses at or above the configured severity threshold. Return structured output matching this state's output schema."
    }
  },
  "children": {
    "address_weaknesses": {
      "state": "feature/plan/critique/address_weaknesses",
      "inputs": {
        "plan_doc": ".inputs.plan_doc",
        "weaknesses": { "expr": ".outputs.weaknesses" },
        "critique_report": { "expr": ".outputs.critique_report" }
      },
      "transitions": [
        {
          "to": "terminate.success",
          "when": ".children.address_weaknesses.outcome === 'success'"
        }
      ]
    },
    "human_review": {
      "state": "feature/plan/critique/human_review",
      "inputs": {
        "plan_doc": ".inputs.plan_doc",
        "critique_report": { "expr": ".outputs.critique_report" }
      },
      "transitions": [
        {
          "to": "terminate.success",
          "when": ".children.human_review.outcome === 'success'"
        }
      ]
    }
  },
  "transitions": [
    {
      "to": "terminate.success",
      "when": ".outputs.outcome === 'clean'"
    },
    {
      "to": "human_review",
      "when": ".outputs.outcome === 'blocked'"
    },
    {
      "to": "address_weaknesses",
      "when": ".outputs.outcome === 'needs_changes'"
    }
  ]
}
```

Execution walk-through: the state's operation runs first. `clean` terminates
immediately, before any child runs. `needs_changes` runs one fix pass and then
terminates so the parent can decide whether to re-plan. `blocked` collects a
human decision, surfaced through the `human_decision` output — an output derived
from a binding rather than produced by the operation, and `optional` because the
child that produces it runs only on the conditional path (§6.2). The two
child-completion rules live on the mounts they are about, so nothing depends on
their position in a list: each is eligible only in the round its own child ended.
What stays in the state's own `transitions` is the decision the state's OPERATION
makes — terminate on `clean`, enter one child or the other on `blocked` /
`needs_changes` — which is about no child in particular and would be wrong
anywhere else. The retry loop lives in the parent (Section 9),
which re-runs the whole planning pass and gets a fresh critique instance each time.

Handing the same state to a delegated agent instead of the prompt runner changes only the
operation block — the slots, wiring, children, and transitions are untouched:

```json
{
  "operation": {
    "kind": "function",
    "function": "claude-code",
    "args": { "permissionMode": "plan" },
    "session": "planning",
    "tools": ["read_file"],
    "permissions": { "profile": "plan" },
    "input": {
      "prompt": {
        "binding": { "text": "Review the plan document and report significant weaknesses." }
      }
    }
  }
}
```

Every authored value reaches the adapter as one named input — `prompt` from the `input` block,
`permissionMode` from `args`, both in one flat `FunctionInputs`. The engine hands a delegated adapter
raw tools, because such an entry declares that it authorizes its own loop's calls (§7.4).

### 7.4 Tool-Call Permissions

When an operation is given tools, each tool call is authorized by a **profile × mode** (full detail in
[DESIGN.md](DESIGN.md) §5.1):

- **profile** — which effects are in scope: `read-only`, `plan`, or `full`. A `read-only`/`plan` profile
  admits only tools that declare themselves read-only; `plan` stays read-only until a human approves an
  exit that rebinds the session to `full`.
- **mode** — how an in-scope call is authorized: `allow`, `deny`, or `ask`. An `ask` invokes a human
  approval gate whose decision persists at a chosen scope — `once`, `always this session`, `always this
  workflow run`, or `always` (the host process) — all in-memory; durable policy is authored, not decided.

A state authors its starting policy via `operation.permissions` (`profile` / `default` / per-tool
modes), possibly inherited from an ancestor's `environment` (§7.1a), overriding a workflow-wide default; live human decisions overlay on top, most-specific scope
winning. The gate is only active when the host supplies an approver; otherwise tools run unguarded. A
delegated adapter — a registry entry declaring that it enforces policy through its own callback —
receives raw tools and routes its native approval callback back through the same approver, so it is
gated once, not twice.

### 7.5 Function Definitions

A state's `operation` is one call that state makes. A **function definition** is the other direction:
a callable an expression names, resolved along the search `path` exactly as §6 describes, and
indistinguishable at the call site from `eq` or `max`.

There is one rule, and everything below is its consequence:

> **A callee is an operation document.** Its declared `input` slots are the signature its positional
> arguments bind against (§6, "the CALLEE's own parameter order binds the arguments"). What differs
> between a built-in, a registered function, and a user's own file is only where the name resolves
> and where the body comes from — never the shape of the call.

#### 7.5.1 Three body forms

A function definition supplies its body in one of three ways. The first two are documents on the
search path; the third is a module beside them.

```text
expression   { "expr": "max(0, 1 - 0.35 * .inputs.severity)" }
             The closed language of §6. No file to approve, no compiler, no runtime.

embedded     { "kind": "function", "input": {…}, "outputs": {…},
               "body": "Math.max(0, 1 - 0.35 * severity)" }
             A JSON document declaring its slots as a state does, with a js/ts body. The
             engine wraps the body in a synthetic function whose parameters are the declared
             inputs, in `index` order.

module       functions/confidence.ts
             A js/ts file. The source is definitive: the signature is READ from it (§7.5.2)
             rather than declared alongside it.
```

The three are one authoring surface at three levels of power, and the choice between them is real
rather than a matter of taste. An expression cannot loop. An embedded body can, and pays for it with
an approval prompt and a compile step. A module can additionally `require` other code, and pays for
that with a resolution story.

**Why an embedded body needs no separate approval.** A callee document is *inlined* into the state
that names it during loading, and `snapshotHash` hashes the resolved form (§12) — so an embedded
body is already part of the workflow's identity, and editing one produces a different workflow. A
module cannot be inlined that way, which is exactly why it needs §7.5.5. The two forms differ in
integrity machinery *because* they differ in whether the body is part of the document.

**A body is a statement list or a single expression**, the distinction an arrow function already
draws between `x => { return f(x); }` and `x => f(x)`. All three of these are the same function:

```text
"return Math.max(0, 1 - 0.35 * severity);"
"Math.max(0, 1 - 0.35 * severity);"
"Math.max(0, 1 - 0.35 * severity)"
```

The rule is the obvious one: **a body that parses as a single expression is one, and its value is
returned.** Anything else is a statement list and must `return` for itself. A trailing semicolon
decides nothing, since an expression followed by one still parses as an expression statement.

This matters more than the keystrokes it saves. The expression form is what makes the three body
forms a genuine progression rather than three unrelated syntaxes: an `{ "expr" }` document that
outgrows the closed language of §6 — it needs a loop, or a built-in the language does not have —
becomes an embedded body by changing which key it is written under, and the text between the quotes
often does not change at all.

One inherited ambiguity comes with it. A body beginning with `{` is read as a **statement list**,
exactly as JavaScript reads it, so an object literal must be parenthesised:

```text
"{ score: s, reasons: r }"      a block, and a syntax error
"({ score: s, reasons: r })"    the record a multi-output function returns
```

**An embedded body may not import.** It is self-contained by construction, which is what keeps the
property above true. Code that needs imports is a module.

#### 7.5.2 The signature

For a module, TypeScript is the source of truth. A parameter list already carries almost everything
a `ParameterDecl` (§4.1) holds, and reading it is strictly better than asking an author to restate it
in JSON where the two can drift:

| Slot field | Read from |
| --- | --- |
| name | the parameter name |
| `index` | the parameter's **position** — so positional binding needs no annotation |
| `schema` | the parameter's type, converted to the wire schema (§7.5.3) |
| `optional` | `?`, or a `\| undefined` member |
| `default` | a parameter default — `function f(limit = 3)` declares the slot's default |
| `description` | a JSDoc `@param` tag, if present |

A module therefore declares **no** signature in JSON; there is nowhere for one to disagree. An
embedded body is the opposite case — it has no parameter list to read — so it declares its slots the
way a state does, and its `index` (or key order) orders the synthetic function's parameters.

Three degradations, all to the same place:

- A `.js` module has no annotations. Every slot is untyped.
- A parameter typed `any` or `unknown` is untyped.
- A **generic** function has no instantiation at extraction time, so its type variables are
  unconstrained and the slots they appear in are untyped. Generic functions are legal and not
  usefully typed.

"Untyped" means the unconstrained schema, which §6.2 already defines as accepting anything. It is
reported as a warning naming the parameter, never silently: a slot that accepts anything and is then
wired into a typed consumer is the failure §6.2 argues against for undeclared outputs, and it should
be visible in both places.

⚠️ **`noImplicitAny` is load-bearing.** Under a configuration that does not require annotations,
every unannotated parameter is implicitly `any`, and the rule above turns the whole feature off with
no error anywhere. The default compiler configuration (§7.5.6) sets it, and disabling it warns.

**Exports, and what a module contributes.** A document on the search path contributes exactly one
symbol: its own name. A module is different — a file can hold a library — so it contributes a set:

| The module holds | Contributes |
| --- | --- |
| a **default export** that is a function | the module's **filename**. `functions/confidence.ts` is called as `confidence`. |
| a **default export** that is an object | one symbol per key, nested to any depth. `{ text: { slug } }` contributes `text.slug`. |
| **named exports** | their names, and their contents where they are objects. `export const text = { slug }` also contributes `text.slug`. |
| **no export at all** | its top-level declarations, which the transpile step exports for it. |

Note what the second row means: for an object export the **filename contributes nothing**. That is
deliberate, and it is what lets a symbol live wherever its author put it — `text.slug` is `text.slug`
whether it was written in `text.ts`, in `strings.ts`, or beside forty other helpers in `lib.ts`.

**A prefix match is not a match.** This is the whole resolution rule, and it is ordinary `PATH`
semantics applied at full symbol depth rather than at the first segment:

> Resolving `text.slug` looks for a file that provides *that whole symbol*. A file providing `text`,
> or `text.trim`, but not `text.slug`, has **missed** — and a miss continues the search.

**A module is not found by its name.** This is what makes the rule above true rather than merely
intended, and it is worth stating as mechanism. A *document* is located by matching the reference
against the directory listing, longest prefix first — which works precisely because a document
contributes one symbol and its name IS that symbol. Applying the same machinery to modules would
answer only the coincidental case: `text.slug` living in a file called `text`. It could never find
`text.slug` in `strings.ts`, because nothing in the reference names `strings`.

So the two are located differently, and in this order:

1. **The document split**, exactly as before. A state file or JSON document always wins, so no
   existing reference can be captured by a module contributing a matching symbol.
2. **The module index**, asked for the whole dotted symbol in that directory. Reached only where
   step 1 finds nothing — which is why this section adds a resolution route rather than altering
   one.

Then the next entry on the search path, and so on. `text.ts` exporting `{ text: { trim } }` never
becomes a candidate for `text.slug` at all: it is not a document, and the index is asked for a symbol
rather than for a filename. Without this the first file whose *first segment* matched would swallow
the reference and report "no `slug` in `text`" — a lie whenever another file has one.

**The rule applies to documents too**, which is what keeps it a rule rather than a special case for
modules. `plan.json` is a candidate for `plan.inner` and does not necessarily *hold* an `inner`: if it
does, it answers and the index is never consulted; if it does not, it has missed, and the search moves
on to the module contributing that symbol. Without this a document's name would claim its whole dotted
subtree, and `plan.inner` could never be reached in a directory that also held a `plan.json`.

This is the one place resolution reads a file rather than only listing a directory. Parsed documents
are cached per file, so the cost is one read each however many references are resolved against them.

**A total miss is RE-RUN with the rule relaxed**, and only to recover the better error. The strict
pass fails with "names no symbol on the path", which is right when something else might have held the
symbol and unhelpful when nothing did — the reader wanted to be told that `plan.json` has no `inner`.
So the relaxed pass, which is exactly the behavior that predates modules, runs again: the first name
match wins, and the missing property is reported against the file it is missing from, at the point of
use. Nothing is traded away; the precise diagnostic is recovered rather than replaced.

One consequence worth stating, because it makes an existing warning sharper: two candidate files are
only AMBIGUOUS when both genuinely provide the symbol. A `user.json` that lacks `address` is not
competing with `user.address.json` for `user.address`, so the "also matches" warning no longer fires
for it.

Three consequences:

- **Only a whole-symbol match at two places is shadowing**, and it is reported by the existing
  search-path shadowing warning (§7.1). Falling through a partial match is silent, because a file
  that lacks the symbol is not competing for it.
- **Partial matches are remembered, for the error.** When nothing on the path provides the symbol,
  the failure names what was found instead — "no `text.slug` on the search path; `lib/text.ts`
  provides `text.trim` and `text.format`". A near miss is almost always a typo or a stale rename, and
  the diagnosis is worth more than the refusal.
- **An explicit reference still pins a file.** Spelling the extension makes the module a document
  split again — `functions/strings.ts.text.slug` is the file `strings.ts` and the property
  `text.slug`, through the same `selectProperty` a JSON document's property reference goes through.
  That is the escape hatch when the path would otherwise answer with someone else's `text.slug`.

A resolved symbol that is **not callable** is data, not an operation — the same shape-mismatch rule
`bindingForDocument` applies to a resolved document, so a module exporting a table of constants
contributes a table of constants, readable in a binding and an error in callee position.

#### 7.5.3 The wire boundary

hw's values are JSON. A function's parameters are TypeScript. These are not the same vocabulary, and
the spec is precise about which one each part of the system speaks:

> **The JSON Schema is the wire type. Marshalling is the adapter between the wire and TypeScript.**

Every static check — `isSubschema`, binding compatibility, expression inference (§6.2) — operates on
wire types and is untouched by this section. The marshalling table is metadata on the slot, beside
`kind`, and not part of its schema.

Marshalling is defined **per leaf type**, and its traversal is derived structurally: `Date` has one
rule, and `Date[]`, `{ when: Date }` and `Date | null` need no further instruction. The table runs in
both directions — a function returning `Date` is demarshalled back to a wire string before the value
reaches an output slot.

Two shapes are rejected at signature extraction rather than guessed at:

- **A union whose members are indistinguishable on the wire once marshalled.** `Date | string` is
  `{"type":"string","format":"date-time"}` against `{"type":"string"}`; nothing at a leaf can decide
  which rule to apply, so the signature is an authoring error rather than a coin flip.
- **A type with no wire form at all** — `bigint`, `symbol`, a function, a class instance whose
  identity is its methods. Accepting one produces a signature that does not describe the call.

Recursive types are fine: the walker follows `$ref` and does not revisit a node it is already inside,
by the same rule §6.2 uses to cut off output inference.

#### 7.5.4 Module resolution

A module is transpiled to **CommonJS**, because `require` is synchronous and interceptable and the
ESM loader hooks are neither. Its resolution is not Node's ambient one: the require path is derived
from the state's search `path` (§7.1), each entry contributing itself and its `node_modules`:

```text
path            ["$JAIRA/functions", "./ops"]
require path    ["$JAIRA/functions", "$JAIRA/functions/node_modules",
                 "./ops",            "./ops/node_modules"]
```

So a function resolves other code the same way a state resolves a document, and one search path
governs both.

**A specifier is a reference.** The require hook resolves what a module asks for through the same
scheme every other reference in this system goes through (§5), not through Node's:

```text
require('./helper')            relative — resolved against the requiring file, as expected
require('$JAIRA/lib/review')   a ROOT VARIABLE, the same `$JAIRA` a state's `path` and a
                               `{"$ref": …}` already resolve
require('/opt/shared/x')       absolute
require('helper')              bare — searched along the require path above
require('zod')                 bare — same search, found in a `node_modules` entry
```

The point is that an author writes one spelling of "where things are" and it means the same thing in
a state file, in a `$ref`, and in an import. A function reaching for a shared library should not have
to know it is inside a module rather than beside one.

Two consequences are worth stating outright, because each is easy to assume the other way:

- **A bare specifier is not the same thing as a dependency.** `require('helper')` finds
  `$JAIRA/functions/helper.js` before it looks in any `node_modules`. Every rule that treats
  `node_modules` differently is therefore keyed on the **resolved path** — "is this file under a
  `node_modules` directory" — never on how the specifier was spelled.
- **The require path is resolution, not containment.** Node builtins resolve ahead of it, so
  `require('fs')` works regardless of what the path holds. Nothing in this section is a sandbox;
  §7.5.5 is what makes that acceptable, and the path is not.

**Type checking uses the same resolver.** The compiler's module resolution is overridden with the
function above rather than left to `tsconfig` defaults. Two resolvers would let a signature be
checked against one `helper.ts` and executed against another, with no error anywhere.

#### 7.5.5 Integrity: hashing, approval, and the freeze

The threat model is a single user running their own code on their own machine. The risk is therefore
not privilege — it is **code changing without the user knowing**, which in a system where an agent
can write files includes code the user never wrote in the first place.

**Every user file is hashed independently**, and the property to hold is the strong form:

> Every file loaded through the user-module resolver must carry an approved content hash.
> **An unknown file is an unapproved file.**

The strong form is what covers shadowing: a new file earlier on the search path changes which module
a specifier resolves to without modifying anything that already existed, and only "unknown means
unapproved" stops it.

Files resolving under a `node_modules` directory are exempt — their integrity is the lockfile's
concern, and re-hashing a dependency tree per run buys nothing. Per §7.5.4 this is a test on the
resolved path, not on the specifier.

**Approval shows a diff, never a hash.** A changed hash carries nothing a person can act on, and
§7.4's approval gate already holds that an approver must be able to see what it is authorizing. First
approval and re-approval are distinct questions and are presented as such: the first is "should this
run at all", the second is "here is what changed".

**Approvals are machine-local and are never synced.** An approval is a statement about a file on one
disk; propagating it would let one compromised machine confer trust on the rest.

**The freeze.** Before a run starts, the workflow's definition is made immutable, and user files are
part of that definition:

1. every reachable function module is resolved, and its hash checked against its approval;
2. any mismatch, and any file with no approval, **stops the run before anything executes** — an
   error, not a prompt, because a run is not the moment to be deciding what code to trust;
3. the **transpiled** output of each module is copied into the snapshot directory beside the state
   files.

Copying the transpiled form rather than the source is what makes a frozen run actually frozen. A
stored hash can only *detect* drift and refuse; it cannot execute the version that was approved.
Storing the emitted code also removes the compiler from replay entirely, so a later toolchain upgrade
cannot change what a pinned run does.

**Those hashes fold into the snapshot hash** (§12). A workflow's identity is what it will do, and a
module reached by name is part of that. Without this, a task pinned to a snapshot would run edited
code under an unchanged version — the precise failure snapshot hashing exists to prevent, and the
reason it hashes the resolved form rather than the authored bytes.

On resume, a run whose frozen copies no longer match the files on disk still executes the **frozen**
copies, and reports drift. The choice that offers — continue the old run, or start a new one against
the current code — belongs to the user, so drift surfaces as a decision rather than a log line.

#### 7.5.6 Execution

**Nothing reaches a function but its parameters.** There is no context object, no session handle, no
ambient binding to the run. This is the closure §6 states for expressions, held one level further
out, and it is why a function can be understood from its signature alone.

The one carve-out is a **cancellation signal**, which a function may accept as a trailing
`AbortSignal` parameter. It is a carve-out and not a hole: it carries no workflow data, is absent
from the wire signature, and exists so that a function can cooperate with the deadline below rather
than be abandoned by it.

**Nothing leaves but the return value**, and every call is wrapped: a throw, a rejection, or a
module-load failure becomes a classified `Failure` rather than an exception crossing the seam, so a
retriable error raised inside a function reaches the retry machinery with its classification intact.

**A deadline is a deadline, not a kill.** A call may be bounded in wall-clock time, and on expiry the
operation reports `outcome: "timeout"` and the run proceeds. The function itself is *abandoned*, not
terminated: JavaScript offers no way to stop a promise, and synchronous code cannot be interrupted at
all — a function that never yields the event loop prevents even the timer from firing. This is stated
rather than papered over, because a spec promising termination would be promising what the runtime
cannot deliver. A function that wants to be cancellable takes the signal above.

**Function results are not memoized.** A call is dispatched afresh each time an expression demands
it, which keeps a function reading a clock, a file, or a network from replaying a stale answer. Two
consequences follow, and both are already the engine's behavior for deferred calls:

- A call that WAITS is registered **once**, however many guards or rounds demand it (§3.3), and a
  taken transition consumes its answer — so waiting does not re-dispatch.
- A call that **completes within a round** has nothing pending, so a guard re-evaluated in a later
  round calls it again. **A function called from a guard must be idempotent.** This is an authoring
  rule rather than machinery: guards are re-evaluated by design, and a function with side effects does
  not belong in one.

**The compiler configuration** has a default that sets `strict` (see §7.5.2 on `noImplicitAny`) and
`isolatedModules` — the latter because emit is per-file, and an author is better told at check time
that a construct will not survive that than at run time. A project may override the configuration;
overriding `noImplicitAny` warns, since it disables typed signatures rather than loosening them.

#### 7.5.7 Example

The three scoring documents of a phase gate, as one module. Every slot below is read from the
parameter list; nothing about this signature is declared in JSON.

```ts
// $JAIRA/functions/confidence.ts
export interface Confidence {
  score: number;
  reasons: string[];
  must_ask: string[];
}

/**
 * @param maxSeverityRank blocker=3 … note=0.
 * @param iteration Converging on pass 3 is not converging.
 */
export default function confidence(
  maxSeverityRank: number,
  iteration: number,
  maxIterations = 3,
  runnerUpMargin = 1,
  mandatoryMargin = 1,
): Confidence {
  const score = Math.max(
    0,
    1 - 0.35 * (maxSeverityRank / 3) - 0.25 * (iteration / maxIterations)
      - 0.2 * (1 - Math.min(1, runnerUpMargin)) - 0.2 * (1 - Math.min(1, mandatoryMargin)),
  );
  const reasons: string[] = [];
  if (maxSeverityRank >= 2) reasons.push("the critique exited at or above the severity threshold");
  if (iteration > 1) reasons.push("it took more than one pass to converge");
  const must_ask: string[] = [];
  if (iteration >= maxIterations) must_ask.push("three rounds that did not converge");
  return { score, reasons: reasons.slice(0, 3), must_ask };
}
```

Called from a state, positionally, in the parameter order the file already fixes:

```json
{
  "outputs": {
    "confidence": { "binding": { "expr": "confidence(.inputs.max_severity_rank, .inputs.iteration)" } }
  }
}
```

**One call, one output.** The three fields reach their consumer as properties of that one value —
`.children.confidence.outputs.confidence.score` — rather than as three sibling outputs each binding
its own call. That is not a stylistic preference: function results are not memoized (§7.5.6), so
three bindings that each name `confidence(…)` are three invocations of it. A function returning a
record is called once and projected many times.

Three files sharing an unwritten contract over eight caller-declared inputs become one file whose
contract is its signature. `maxIterations` and the two margins carry defaults, so a caller with no
exploration to report omits them rather than wiring a constant — and the `empty` seed that existed
only because the expression grammar has no array literal has nowhere left to be.

## 8. Function States (Interactive UI)

> **Not to be confused with §7.5.** A *function definition* is a callable an expression names. A
> *function state*, below, is a state whose operation happens to be of kind `function` and whose
> registered entry is interactive. They share a word and nothing else.

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
        "expr": ".children.critique.outputs.outcome === 'clean' ? 'complete' : 'blocked'"
      }
    },
    "plan_doc": {
      "kind": "blob",
      "schema": { "type": "string", "contentMediaType": "markdown" },
      "binding": ".children.context.outputs.plan_doc"
    },
    "critique": {
      "binding": ".children.critique.outputs.critique"
    }
  },
  "children": {
    "goals": {
      "state": "feature/plan/goals",
      "inputs": {
        "issue": ".inputs.issue"
      }
    },
    "context": {
      "state": "feature/plan/context",
      "inputs": {
        "issue": ".inputs.issue",
        "goals": ".children.goals.outputs.goals"
      }
    },
    "critique": {
      "state": "feature/plan/critique",
      "inputs": {
        "plan_doc": ".children.context.outputs.plan_doc",
        "severity_threshold": { "text": "significant" }
      },
      "transitions": [
        {
          "to": "terminate.success",
          "when": ".children.critique.outputs.outcome === 'clean'"
        },
        {
          "to": "goals",
          "when": ".children.critique.outputs.outcome === 'needs_changes' && .run.iteration < .limits.max_iterations"
        },
        {
          "to": "terminate.success",
          "when": ".children.critique.outcome === 'success'"
        }
      ]
    }
  },
  "sequence": ["goals", "context", "critique"],
  "limits": {
    "max_iterations": 3
  }
}
```

Every rule here is about `critique` — what the critique said, and whether to go
round again — so all three live on its mount and the state declares no
`transitions` of its own. The re-plan loop still lives here: `needs_changes`
transitions back to `goals`, which resets the sequence and clears the recorded
results of `goals`, `context`, and `critique`, so the next pass runs fresh
instances. On the mount they are not merely false until `critique` runs again;
they are not evaluated at all until it finishes. When the iteration limit is
reached, or critique reports `blocked`, the third rule fires and the `outcome`
output resolves to `blocked`.

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
      "environment": { "kind": "function", "function": "claude-cli" },
      "inputs": {
        "change": ".inputs.change"
      }
    },
    "codex_review": {
      "state": "review/agent_review",
      "async": true,
      "environment": { "kind": "function", "function": "codex-cli" },
      "inputs": {
        "change": ".inputs.change"
      }
    },
    "synthesize": {
      "state": "review/synthesize",
      "inputs": {
        "review_a": ".children.claude_review.outputs.report",
        "review_b": ".children.codex_review.outputs.report"
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

Note what makes the two reviews DIFFERENT: one `review/agent_review` state, mounted twice under two
per-mount `environment` layers (§7.1a), so the reviewed state names no agent of its own and needs no
duplicate file. The two mounts load as two variants of one id.

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
18. User code runs only by approved content hash, and only from the copy frozen
    before the run began.
