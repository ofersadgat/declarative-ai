# Scripts — a state machine written as code

**Status: PROPOSED.** Nothing is built. §16 lists what is still open.

Code is a better notation than JSON for a complicated state machine. A loop is a loop, a branch is an
`if`, and a value handed from one step to the next is a variable. In JSON the same things are a
backward transition with an iteration guard, two guarded rules, and a wire. A **workflow script**
lets an author write that code and have hw compile it into the state tree the JSON would have been.

The script language is a SUPERSET of the Claude Code workflow script. Every valid Claude workflow
script is a valid hw script (§14). On top of that, a script may be TypeScript, may `import`, and may
call anything on the search path.

Four rules, and the rest of the document works them out:

> 1. **A script file IS a state.** `meta` sets its standard properties: label, title, description,
>    environment, limits. Its inputs and outputs are those of the function it defines (§5).
> 2. **The code COMPILES to states.** `phase("Find")` names a child state. Code runs in whichever
>    phase is current, calling the current phase is a no-op, and calling another phase is a
>    TRANSITION (§6).
> 3. **`llm()` is the one primitive.** It takes the full LLM operation and goes through route
>    binding. `agent()` and every other helper are `llm()` with different defaults (§9).
> 4. **Compiling is a mode, not a requirement.** A script can instead run as ONE state (Claude's
>    exact execution model) or as a plain callable function. A `.ts` under a workflows root compiles
>    to states by default (§3).

---

## 1. Why

hw's JSON format says exactly what a state machine is, and that is the problem when the machine is
complicated. The run-11 `feature/product` loop, a critique loop with an escape hatch, or "find
until two rounds come back dry, then verify each" are each a few lines of code and a page of
transitions, guards, wires and iteration limits. Every one of those has to be kept consistent by
hand. LOOPS.md exists largely because a wire that looked right in JSON silently carried nothing.

Writing the machine as code and COMPILING it keeps everything hw is for: the board shows the phases,
a person can steer between them, the journal records every step, a restart resumes from the last
state, and the validator types every value. Durability is not reinvented, because the output is
ordinary states. What changes is who writes the transitions: the compiler, from control flow the
author already wrote.

Being a superset of Claude's workflow scripts is what makes this a notation people already know.
`agent()`, `parallel()`, `pipeline()` and `phase()` already describe multi-step agent work, and a
Claude script's `phase()` calls already mark the places a board would draw columns.

## 2. A script, and what it compiles to

```ts
// .jaira/workflows/review/sweep.ts
import triage from "./triage";                          // a state — mounted, called like a function
import { dedupeByLocation } from "$/lib/findings";      // code — callable from compiled expressions

export const meta = {
  name: "Sweep review",
  description: "Find issues along several dimensions until a round finds nothing new; verify; triage.",
  title: { binding: { $expr: "'Review ' + .inputs.change.ref" } },
  environment: { config: { model: "anthropic/claude-opus-5-5" } },
  limits: { max_iterations: 3 },
};

export default async function sweep(change: Change, dimensions: string[] = ["bugs", "perf"]) {
  let seen: string[] = [];
  let confirmed: Finding[] = [];
  for (let round = 0; round < 3; round++) {
    phase("Find");
    const found = await parallel(dimensions.map((d) => () =>
      agent(`Review ${change.ref} for ${d}. Skip anything in: ${seen.join(", ")}`, { schema: FINDINGS })));
    const fresh = dedupeByLocation(found.filter(Boolean).flatMap((r) => r.findings));
    if (fresh.length === 0) break;
    seen = [...seen, ...fresh.map((f) => f.id)];

    phase("Verify");
    const verdicts = await parallel(fresh.map((f) => () =>
      llm<{ refuted: boolean; reason: string }>({ prompt: `Try to refute: ${f.claim}`, config: { model: "anthropic/claude-sonnet-5" } })));
    confirmed = [...confirmed, ...fresh.filter((_, i) => verdicts[i] && !verdicts[i].refuted)];
  }
  phase("Triage");
  return { confirmed: await Promise.all(confirmed.map((f) => triage({ finding: f }))) };
}
```

The compiler produces this tree:

```text
review/sweep            "Sweep review" — inputs change, dimensions; output confirmed
├─ find                 "Find"    — fans `agent` out over dimensions (an `each` child), then dedupes
├─ verify               "Verify"  — fans `llm` out over fresh findings
└─ triage               "Triage"  — `triage` mounted with `each` over confirmed
```

The transitions are the control flow, lifted into guards the author could have written:

```jsonc
"children": {
  "find": {
    "state": "./find",
    "transitions": [
      { "when": "length(.children.find.output.fresh) === 0", "to": "triage", "inputs": { … } },  // break
      { "to": "verify", "inputs": { … } }
    ]
  },
  "verify": {
    "state": "./verify",
    "transitions": [
      { "when": ".children.verify.output.round < 3", "to": "find", "inputs": { … } },          // loop
      { "to": "triage", "inputs": { … } }
    ]
  },
  "triage": { "state": "./triage" }
},
"sequence": ["find"]
```

It is the file a careful author would have written by hand, with every wire already correct.

## 3. Three modes

`meta.compile` chooses how a file becomes something hw runs:

| `compile` | The file becomes | Calls are recorded by | Default for |
| --- | --- | --- | --- |
| `"states"` | a state TREE: the root plus one child per phase (§6), plus whatever structure the code inside a phase needs (§7) | the states themselves: every step is an ordinary state or operation | a `.ts`/`.js` under a **workflows** root |
| `"state"` | ONE state whose operation is the whole script. `phase()` only groups calls, as in Claude. | replay: the body re-runs against the journal (§11) | none (asked for explicitly) |
| `"function"` | a CALLABLE on the search path, exactly as a function module is today (§7.5), but allowed to call `llm()` | the caller's call site (§11) | a `.ts`/`.js` anywhere else (`functions/`, `lib/`) |

A file with no `meta` gets its location's default. A `"states"` compile that cannot segment the code
(§8, §14) is an ERROR that names the construct and suggests `compile: "state"`. It never silently
falls back, because a script that quietly stopped being a state tree would lose the board, the
steering and per-state resume without the author knowing.

`"state"` is Claude's exact execution model. It is what guarantees the superset: every Claude script
loads in it, even one the compiler cannot segment.

## 4. `meta`

`meta` holds the STANDARD PROPERTIES of the state the file is: the fields that describe it and say how
it runs, and none of the fields that describe its structure. Structure comes from the code.

| Key | Meaning |
| --- | --- |
| `name` / `label` | The static display name. `name` is Claude's spelling, and writing both is refused. |
| `description` | As in §5.2. |
| `whenToUse` | Claude's key. Documentation for whatever chooses among workflows. |
| `title` | The instance's display name, evaluated once at entry. It may read `.inputs` (§5). |
| `environment` | Defaults for every operation in the compiled tree, inherited as usual (§7.1a). |
| `limits` | `max_iterations` (which bounds the compiled loops, §6), `timeout`, `budget`, `max_calls`. |
| `compile` | The mode (§3). |
| `phases` | **Claude compatibility only**, and it declares nothing (§6.5). |
| `inputs`, `outputs`, `children`, `sequence`, `transitions`, `operation`, `id` | **Refused.** The signature gives the inputs and outputs, the phases give the children, and the control flow gives the rest. |

`meta` is a PURE LITERAL, as in Claude: literals, comments, `as const`/`satisfies` stripped, and no
identifiers, calls, spreads or interpolation. The loader reads it without running anything. A
computed value is a binding written as data (`title: { binding: { $expr: … } }`), as in JSON.

## 5. Signature

**If the file exports a function, its signature is the state's.** Its parameters are the inputs and
its return type is the output, read by the existing extractor (§7.5.2): names, positions, types
converted to wire schemas, `?` and defaults as optional, JSDoc `@param` as descriptions. The function
is the default export, or else the only exported function. Several exported functions and no default
is an error in `"states"`/`"state"` mode (which one is the state?), and ordinary in `"function"` mode,
where a module contributes a symbol per export as it does today.

**If it does not, the file's body is the function.**

- **Input: `args`**, Claude's global: one optional, unconstrained input named `args`. A parent wires
  it as `"inputs": { "args": … }`.
- **Output: the export.** `export default <value>` is the whole output, and named exports
  (`export const confirmed = …`) are outputs by name. Claude's top-level `return x` is also the whole
  output, since that is how a Claude script returns. Writing both a `return` and an export is an
  error.

Module-level code outside the exported function (constants, helper functions, schemas) is STATIC:
it is available to every compiled state and is not itself a step.

## 6. Phases

### 6.1 The meaning

> **`phase("X")` says: the code from here on belongs in child state X.** If X is already the current
> phase, it does nothing. If another phase is current, it is a transition to X.

That is the whole semantics, and it is the same whether the call sits in a loop, an `if`, or a helper
function. Before the first `phase()` no child is current, and code runs in the root state itself.

- **Phase → key.** `"Find"` becomes key `find` and label `"Find"`. Two titles that map to one key
  are an error. The title must be a string literal, because a computed phase has no state to be.
- **Where a call can move the phase.** Only the script's single control thread can, so `phase()`
  inside a `parallel` thunk or a `pipeline` stage is a compile error. Claude's `opts.phase` on an
  `agent()` inside one keeps its Claude meaning, a display group, and moves nothing.
- **Recursion.** A phase call reachable only through recursion cannot be compiled, and is an error.
  Script-local helpers are otherwise inlined, so a helper that calls `phase()` works.

### 6.2 How a phase becomes a state

A child state's code is every piece of the program that can run while that phase is current. The
compiler finds it by walking the control flow and tracking the current phase:

- **Entry points.** A phase is entered at a `phase("X")` call site, and there may be several (the
  first pass through a loop, and every later one). The child takes an `$at` input naming which entry
  it was entered through, plus that entry's live variables (§8).
- **Exits.** The child ends where control reaches a `phase()` naming another phase, a `return`, or
  the end of the function. It ends SUCCESSFULLY and outputs its continuation: the phase to go to,
  the entry point there, and the live variables the target needs.
- **Transitions go on the mount.** hw lets a child reach a sibling only through its parent (§3.1),
  so every exit becomes a rule on the child's mount (§3.3): `{ when, to, inputs }`. The rule's
  `inputs` hand the target its live variables, so the target can read them.
- **Readable guards.** When a branch condition is an expression over values the child outputs, the
  compiler LIFTS it into the guard: `fresh.length === 0` becomes
  `length(.children.find.output.fresh) === 0`. Otherwise the child outputs the decision, and the
  guard reads `.children.find.output.$next === "triage"`. §6's expression language has JavaScript
  semantics, so most conditions lift.
- **`return`** from inside a phase ends that child with the return value, and the mount's rule is
  `{ to: "terminate.success" }`. The root's outputs are bound to whichever child returned.
- **A join reached under two phases.** Code after an `if` whose branches entered different phases
  runs in whichever phase is current, so it is compiled into BOTH children. This is tail
  duplication, and it is invisible to the author.

### 6.3 Loops

A loop whose body crosses phases is a backward transition, and backward transitions in hw require a
bound (§16.9). The compiler takes it from a literal loop bound when the loop has one
(`round < 3`), and from `meta.limits.max_iterations` otherwise. A loop across phases with neither
is a compile error. The run's passes stay addressable as LOOPS.md describes: `.children.find[-2]` is
the previous Find.

### 6.4 Before, between, after

- Code before the first `phase()` is the ROOT's own operation. It runs before any child, as an
  operation always does. A script with no `phase()` at all is a single state whose content is the
  whole body (§7 may still give it inner structure).
- The root's `sequence` is the one phase control first reaches. If that depends on a branch, the
  root outputs the decision and a root rule enters the right phase.
- A throw that nothing catches is the child's `error`, which becomes the root's `error` (§3.3),
  exactly as a JS exception propagates.

### 6.5 Claude's `meta.phases`

In Claude, `meta.phases` is **display only**. It pre-declares the phase groups so the progress view
and the permission dialog can show the plan before anything runs, with a `detail` line per phase and
an optional `model` label. It is matched to `phase()` calls by exact title, a `phase()` call with no
entry just gets its own group, and nothing executes from it.

A compiled script already knows its phases before it runs, because they are the literals in its
`phase()` calls, so the list is redundant as structure. It is ACCEPTED as descriptions of the
compiled children: `detail` becomes the child's `description`. It declares nothing. An entry no
`phase()` call names is a warning, and so is a `model` that disagrees with what the phase's calls
actually use, since in hw a model is set on the call or the environment and not on a label.

## 7. Inside a phase

A phase's code is cut at its AWAIT POINTS: the places it waits on an `llm()`, a state, or a
registered function. The pieces become hw's ordinary constructs:

| Code | Compiles to |
| --- | --- |
| a straight run of calls | an operation LIST (§7.1d), with each call's inputs bound to earlier calls' outputs |
| JS computing a value between calls | an `{ $expr }` binding when it fits the expression language (§6), otherwise an embedded function body (§7.5.1) |
| a call to imported code (`dedupeByLocation(…)`) | a callee call inside that binding, as in any state (§7.5) |
| `parallel(xs.map(x => () => call(x)))`, `Promise.all(xs.map(call))` | a child mounted with **`each`** over `xs` (WORKFLOWS.md §6.2), `async: true` |
| `pipeline(items, s1, s2, …)` | an `each` child, `async: true`, whose state is the stage chain for one item |
| an `if` or loop around calls, with no phase change | generated sub-states inside the phase, with transitions as in §6.2 but unnamed (`find/1`, `find/2`) |
| `try { … } catch` around a call | a transition that handles the call's outcome, by naming `.outcome` (§3.3) |
| `agent()`'s `null` on failure | the call's binding with `failureValue: null` (§4.2) |

Generated sub-states are real states. They show on the phase's sub-board and resume like any other,
but no name the author wrote refers to them.

**One gap in `each`.** An element that fails ends the whole batch today. `parallel` and `pipeline`
turn a failed element into `null` and carry on, so the compiler needs a per-element failure value:
`each` gains `failureValue`, so an element that fails reads as that value and the batch continues.
The feature is useful on its own too.

## 8. Values across states

A variable that is live where control crosses a state boundary becomes a WIRE: an output of the state
it leaves and an input of the state it enters. The compiler does the liveness analysis, and nothing
the author writes names it.

- **Only wire values cross.** A value crossing a boundary is marshalled by the §7.5.3 rules: JSON as
  it is, and `Date`/`Set`/`Map` through their leaf rules. That covers the `seen = new Set()` of
  Claude's loop-until-dry pattern. A closure, a class instance, or a pending promise living across a
  boundary is a compile error that names the variable and the boundary.
- **Mutation is by value.** `confirmed.push(x)` in one phase is seen by the next because the array
  travels as a wire. Two concurrent elements of one `each` cannot share a mutable variable, and the
  compiler refuses a write to an outer variable from inside a `parallel` thunk or a `pipeline` stage.
  Returning the value is the fix, and the error says so.
- **Static code is not a value.** Module-level constants and functions are available to every state
  and never travel.

## 9. `llm()` and the functions built on it

```ts
llm<T = string>(op: LlmOperation): Promise<T>
llm<T = string>(prompt: string, op?: Partial<LlmOperation>): Promise<T>
```

`LlmOperation` is the full prompt operation of §7.1 together with its environment fields: `prompt`,
`system`, `config` (model and every knob, `configRef` included), `input`, `output`, `tools`,
`session`, `workspace`, `conversation` and `permissions`, plus the binding-level `failureValue` and a
display `label`.

**The type argument IS the output contract.**

```ts
const destructive = await llm<boolean>({ prompt: "Is the following command destructive? " + command });
const { foo } = await llm<{ foo: number }>({ prompt: "Tell me the value of foo in: " + JSON.stringify(command) });
```

The compiler converts `T` to the call's output schema with the converter that already reads a
function module's parameter types (§7.5.2, §7.5.3). There is one TS → wire path, and the result is
the call's `output: { schema }`: the structured-output contract the executor asks the model for,
and the schema the answer is validated against. `T` is a compile-time fact the compiler turns into
data. At run time the call has no type argument, only the schema.

- **`T` is any type the checker can resolve, not only a literal.** `llm<Person>(…)` works whether
  `Person` is a `type` or `interface` in the script, or imported (`import type { Person } from
  "$/lib/people"`). It also works for anything built from such types: `Person[]`,
  `Pick<Person, "name">`, `Page<Person>`, a union of named types. The compiler asks the TypeScript
  checker for the type at the call site and converts that, so an imported type resolves through the
  same resolver the type-check uses (§7.5.4). That resolver is the search path, `$`-roots and
  `node_modules` included, so `Person` means the same file for the checker and for the schema.
- **Named types stay named.** Each named type the conversion reaches becomes a `$defs` entry and is
  referenced by `$ref`, titled with its name. A type used by several calls is written once per
  schema, and a recursive type (`type Node = { children: Node[] }`) converts without looping,
  because the walker follows `$ref` and does not revisit a node it is already inside (§7.5.3).
- **Descriptions travel.** A JSDoc comment on a property of `T` becomes that property's
  `description` in the schema, which the model reads. `llm<{ /** 0–1, how sure */ confidence: number
  }>` tells the model what the field means.
- **A non-object `T` is wrapped for the wire.** Structured output generally wants an object at the
  root, so `llm<boolean>` asks for `{ value: boolean }` and returns the `boolean`. The wrapping is the
  executor's concern and invisible to the script, the journal's `output` and every consumer.
- **`T = string` is text.** With no type argument and no `output`, the call asks for text.
- **`T` and `output.schema` together must agree.** Writing both is allowed (the schema can carry
  what a type cannot: `pattern`, `minimum`, `format`), but the type converted from `T` and the schema
  must describe the same values, checked with `isSubschema` both ways. A disagreement is an error,
  because either one alone would validate the answer against something the other says it isn't.
- **`T` must have a wire form.** A function, `bigint`, a class instance, or a union the marshaller
  cannot tell apart (§7.5.3) is refused at the call, naming the type.
- **Generic helpers are fine.** Script-local helpers are inlined before compilation (below), so
  `const ask = <T,>(p: string) => llm<T>(p)` is instantiated at each call site, and `ask<boolean>(…)`
  gets `{ type: "boolean" }`. A `T` still unresolved after inlining (a helper in an imported module,
  called generically) cannot be converted, and is an error that asks for `output` instead.
- **JavaScript has no type arguments.** A `.js` script (every Claude script) says the same thing with
  `output: { schema }`, or Claude's `agent(p, { schema })`. The call is typed `unknown` in the
  checker, as an unconstrained slot is.

`agent<T>()` takes the same type argument with the same meaning. On an `agent`, `T` and Claude's
`schema` option are the two spellings of one contract, and the agreement rule above applies.

**It goes through ROUTE BINDING, never to a named agent.** `config.model` resolves through the model
catalog to a route, and the route decides the executor: the API, or an agent binary through
`agents-cli`. A workflow that wants Claude Code's loop asks for a model and a tool set. It does not
name `claude-code`, and the binding decides how to run it. A registered function can still be called
where one is meant (§10); it is just not what `llm()` is.

**Every other helper is `llm()` with different defaults**, and each default is a layer under the
call's own options:

| Helper | Defaults over `llm()` |
| --- | --- |
| `llm` | `session: null` (each call stands alone unless it names a conversation), `output` text |
| `agent(prompt, opts)` | `llm`'s, plus the agent tool set (`tools` from the environment, or the host's agent profile), `failureValue: null` (Claude's null contract) |

A script can define its own helpers the same way (`const critic = (p) => llm({ ...CRITIC, prompt:
p })`). Script-local helpers are inlined before compilation, so the calls inside them compile like any
other.

**Claude's `agent()` options, lowered:**

| `opts` | Becomes |
| --- | --- |
| `schema` | `output: { schema }`, the same contract as `agent<T>()`'s type argument (they must agree) |
| `model` | `config.model`, route-bound |
| `effort` | `config.reasoning.effort`, fitted to the model (`fitReasoning`) |
| `isolation: "worktree"` | `workspace: null`, a fresh private bundle. What that is (a worktree) is the host's decision. |
| `agentType` | `config.configRef`: a named configuration, which is what "a kind of agent" is under route binding |
| `label` | the call's display label |
| `phase` | a display group only (§6.1) |
| anything else | the same-named `LlmOperation` field |

**A prompt is TEXT, never a template.** A JS template literal has already interpolated what the
author meant to interpolate, so `{{…}}` in a script's prompt is literal text. The compiler lowers
`\`Review ${change.ref} for ${d}\`` to an expression binding that builds the string, and the
operation receives it verbatim. hw needs a literal prompt form for this (§15).

## 10. Calling states, and imports

**An import specifier is a reference** (§7.5.4), resolved along the same search path as everything
else. What it resolves to decides what the import is:

| Resolves to | Binds | In a compiled script |
| --- | --- | --- |
| a state (JSON, YAML, or another script) | `(inputs) => outputs` | a MOUNT. A call enters it, and `xs.map(s)` under `Promise.all` is an `each` over it. |
| a prompt (`.md`, a skill) | a prompt callable | an `llm()` with that prompt |
| a registry entry (`$REGISTRY`) | a typed callable | a function operation |
| a code module | its exports | a callee in compiled bindings (§7), or inline code in the other modes |
| a data document, or `with { type: "json" }` | the value | a constant |

Claude's `workflow(nameOrRef, args)` calls a state by reference, and the reference must be a
literal. Imports and literal `workflow()` targets are the state's declared children (§2.3), so the
validator sees and types every state a script can reach. A dynamic `import()`, a computed
`workflow()` and an inline Node builtin are refused, because none of them can be declared, frozen or
typed.

In `"state"` and `"function"` modes, where imported code runs inline, a code module that touches the
world is imported `with { as: "operation" }` so its calls are recorded rather than replayed (§11).
In `"states"` mode every call to imported code is already a recorded step, and the attribute changes
nothing.

## 11. The other two modes

**`"state"`: one state, Claude's execution model.** The whole body is the state's operation. Every
`llm()`, state call and registered-function call is recorded at a SITE whose key is
`hash(call) # k`, where `k` counts earlier identical calls from this instance. This reuses the
engine's content-keyed `callSiteScope`, plus the ordinal, so three identical skeptic calls are three
draws and not one memo hit. On resume the body re-runs from the top:

- a call with a settled record returns it;
- a called state still running is re-attached;
- a call with no record runs live.

Recorded results are released in their journaled completion order, so a race resolves as it did the
first time. `Date.now()`/`Math.random()` throw as in Claude, and `now()`/`random()` from
`@declarative-ai/hw/script` are the recorded versions. `phase()` journals a group, and nothing else.

**`"function"`: a callable.** The file is a function module on the search path, as `functions/*.ts`
is today, but `llm()` is allowed inside it. Its calls are made under the CALLER's services and
recorded at the caller's site with the same ordinal scheme, so a function called from a state's
binding shows its model calls under that state. Because §7.5.6 re-runs guards, a function that calls
`llm()` from a guard is re-asked on every round, and the validator warns about it.

## 12. The compiled files

The compiled tree can be MATERIALIZED as ordinary state files beside the script, the way
`feature.md` sits beside `feature.json` (WORKFLOWS.md §11.3):

```text
workflows/review/sweep.ts          ← the source
workflows/review/sweep.json        ← generated: the root
workflows/review/sweep/find.json   ← generated: a phase
workflows/review/sweep/find/1.json ← generated: a sub-state inside a phase
```

- **The source owns what it generates**, by the same nearest-owner rule descriptions use, so each
  generated file answers to exactly one script.
- **Unlike a description's sync, compiling is deterministic.** It involves no model, no proposal and
  no review round. A changed source recompiles exactly.
- **A generated file records its provenance** (`"generated": { "from": "sweep.ts", "inputs": { <path>:
  <sha256>, … } }`), so the loader can tell a stale or hand-edited file from a current one. `inputs`
  lists EVERY file the compile read, not only the script. A module that only supplied a type (the
  `Person` of an `llm<Person>`) never runs, but the schema was generated from it, so an edit to it
  makes the generated files stale exactly as an edit to the script does. The script is the source of
  truth. A generated file that no longer matches its source is a load error that says to edit the
  script, rather than a silent choice between the two.
- **Ejecting** is deleting the script. The JSON files are then the source, and editing them is
  ordinary authoring.

The library side is one entry point, `compileScript(source, options) → { documents, sourceMap,
diagnostics }`. The loader calls it in memory, so a script runs whether or not its files were ever
written. WHETHER to write the files, and the ownership bookkeeping, belong to the host (JaiRA's
`sync.json`). The source map is how a failed state, a board card or a validation error points back
to the line of the script that produced it.

## 13. Types, validation, integrity

- **Types.** A TypeScript script is type-checked under the module compiler configuration (§7.5.6).
  Imported states are typed by their signatures, `llm<T>()` by `T` (or by an `output` schema written
  `as const`), and `args` as `unknown` unless the file exports a typed function. The compiled tree is then
  validated like any authored tree (§6.2), so a wire the compiler produced is checked like one a
  person wrote.
- **Diagnostics speak source.** Every error in the compiled tree is reported at the source line
  through the source map, never as "`find/1.json`, input `$at`".
- **Integrity.** The script is user code (§7.5.5): it is hashed, approved by DIFF, and frozen.
  Embedded bodies the compiler emits are part of the compiled documents and hashed with them, like
  any embedded body. Imported modules are frozen through the same resolver. A `type`-only import
  needs no approval, because nothing from it executes. Its effect is the schema it produced, which
  is data in the compiled documents and hashed with them, so a change to the type is a change to the
  workflow's identity.

## 14. The superset, precisely

**Every Claude workflow script loads.** In `"state"` mode it runs with Claude's execution model and
the same meaning. Under a workflows root it first tries `"states"`, which it passes unless it does
one of the things below, and then the error names the construct and `compile: "state"`.

| Claude construct | In `"states"` mode |
| --- | --- |
| `phase(title)` | a state boundary (§6), not a display group. The code runs the same way. |
| `meta.phases` | descriptions of the compiled children (§6.5) |
| `phase()` inside `parallel`/`pipeline` | **refused**: only the control thread moves the phase |
| a non-wire value (closure, class instance) live across a phase | **refused** (§8) |
| writing an outer variable from a concurrent thunk | **refused** (§8) |
| an unbounded loop across phases with no `limits.max_iterations` | **refused** (§6.3) |
| `workflow(x)` with a computed `x` | **refused** in every mode (§10) |
| `workflow("name")` finds a saved workflow by `meta.name` | resolves as a reference along the search path in every mode. A host maps names through `$REGISTRY`. |
| `agent()` runs Claude Code's subagent | runs the model the call resolves to, route-bound (§9) |
| `budget` is the turn's shared pool | this state's `limits.budget` |

**Additions no Claude script can notice:** TypeScript, imports, an exported-function signature,
`llm()`, the extra `meta` keys and `agent` options, and unlimited nesting.

## 15. What changes where

**hw.**
- `compileScript` (§12): the `meta` reader; inlining script-local helpers; converting each
  `llm<T>`/`agent<T>` type argument to its output schema through the §7.5.3 converter; the control-flow walk that
  tracks the current phase; liveness analysis into wires; cutting a phase at its await points;
  lifting conditions into guards; emitting the documents and the source map.
- Loading: state resolution recognizes a script (`export const meta` first, or an exported function
  under a workflows root) and compiles it. `compile` selects the mode. `"function"` goes to the
  module index as today.
- `each` gains `failureValue` (§7). A literal prompt form (§9). The `generated` provenance field and
  its load check (§12). `limits.budget` and `limits.max_calls`.
- The `"state"` operation kind: the realm, the hook implementations, the `#k` site ordinal, and the
  replay scheduler (§11). `llm()` inside `"function"` modules, under the caller's site.
- `@declarative-ai/hw/script`: `llm`, `agent`, `parallel`, `pipeline`, `phase`, `log`, `workflow`,
  `now`, `random`, with the Claude names also installed as globals for `.js` scripts.
- SPEC: a new §7.6 carrying this document once settled. §5.1 (`generated`, `whenToUse`), WORKFLOWS
  §6.2 (`each.failureValue`). API.md.

**JaiRA.** The workflows pane lists scripts and marks the files a script generated. The host writes
the generated files and records ownership in `sync.json`. Approval covers scripts. The board and
the projection point back to source lines through the source map.

## 16. Open questions

1. **Phase keys.** `"Find"` → `find` is a slug. Should an author be able to name the key separately
   (`phase("Find the bugs", { key: "find" })`), or is the slug enough? Recommendation: the slug, plus
   an error on collision.
2. **Nested phases.** A phase can only be a direct child of the script. Hierarchy comes from calling
   another script. Should `phase("Verify/Refute")` make a grandchild, or is calling a sub-script the
   one way to nest? Recommendation: sub-scripts only, so one mechanism.
3. **Is `$next` acceptable where a condition cannot be lifted?** It is correct but opaque on a board.
   The alternative is refusing such a branch, which rejects legitimate code. Recommendation: allow
   it, and label the guard with the source condition's text.
4. **Tail duplication.** A join after branches that entered different phases puts the same code in
   two states (§6.2). If that code makes model calls, the call appears on two sub-boards. Is that
   fine, or should such a join be an error? Recommendation: fine. Each occurrence is where the run
   actually was.
5. **Should a generated file ever be editable?** §12 makes the script authoritative and refuses a
   hand edit. The other option is letting an edit "detach" that one state from the script.
   Recommendation: no detaching, since one owner per file is what keeps the pair honest.
6. **`budget` units**: output tokens as in Claude, `costUsd`, or both? Recommendation: both keys,
   with `budget.total` reporting tokens.
