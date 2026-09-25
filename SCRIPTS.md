# Scripts — a state machine written as code

**Status: BUILT in hw, with the gaps listed below. JaiRA not migrated.** `compileScript`
(`scriptCompile.ts`), the hooks (`scriptHooks.ts`) and the engine's `$script` operation
(`scriptRun.ts`, `engine.ts`) implement §3–§14. `loadBundleFromDir` compiles a directory's scripts.
`scripts.test.ts` has 31 tests, and the whole suite (2838) passes.

Six things the implementation found that this document did not anticipate:

- **Inside a phase, the code runs as it was written; it is not decomposed further.** §7 below
  describes the finer cut: operation lists, `each` children and generated sub-states. What is
  built is the coarse half: each phase is a state, and its code runs as one recorded operation. Every
  `llm()` it makes is dispatched and journaled at a site of its own (`script.call.settled`), and a
  restart part-way through a phase re-runs its code and is handed back what was recorded (§11). That
  already gives per-call durability and visibility. The per-call STATES are the next step, and
  `each.failureValue` goes with them.
- **The wiring checker does not reason about unions**, and a variable's TypeScript type often has
  one (`(Finding | null)[]` out of `parallel`). A union inside a wire's type is widened to "anything".
  Both ends of every generated wire are the same variable, so the checker loses nothing it could
  have used.
- **Reachability cannot see which phase returned.** The root's outputs read "whichever phase returned
  the value", which is a run-time fact. With phases, the root's outputs are `optional` (the §6.2
  opt-out), and the script's return type is what guarantees them.
- **Guards read the continuation; none are lifted yet.** Every exit is
  `.children.<phase>.output._next === '<target>'`. §6.2's readable guards are not built.
- **`Set` and `Map` have no wire form** (`wireType.ts` refuses both), so one live across a phase is a
  compile error, as §8 says of a closure. A named type is inlined where it is used, not placed in
  `$defs`, and a recursive one is unconstrained at the cycle.
- **Metrics carry dollars, not tokens**, so `limits.budget` and `budget.total` are USD.

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
import classify from "./triage";                        // a state — mounted, called like a function
import { dedupeByLocation } from "$/lib/findings";      // code — runs inline

export const meta = {
  name: "Sweep review",
  description: "Find issues along several dimensions until a round finds nothing new; verify; triage.",
  title: { binding: { $expr: "'Review ' + .inputs.change.ref" } },
  environment: { model: "anthropic/claude-opus-5-5" },
};

export default async function sweep(change: Change, dimensions: string[] = ["bugs", "perf"]) {
  let seen: string[] = [];
  let confirmed: Finding[] = [];
  for (let round = 0; round < 3; round++) {
    phase("Find");
    const found = await parallel(dimensions.map((d) => () =>
      agent<{ findings: Finding[] }>(`Review ${change.ref} for ${d}. Skip anything in: ${seen.join(", ")}`)));
    const fresh = dedupeByLocation(found.filter((r) => r !== null).flatMap((r) => r!.findings));
    if (fresh.length === 0) break;
    seen = [...seen, ...fresh.map((f) => f.id)];

    phase("Verify");
    const verdicts = await parallel(fresh.map((f) => () =>
      llm<{ refuted: boolean; reason: string }>({ prompt: `Try to refute: ${f.claim}`, model: "anthropic/claude-sonnet-5" })));
    confirmed = [...confirmed, ...fresh.filter((_, i) => verdicts[i] !== null && !verdicts[i]!.refuted)];
  }
  phase("Triage");
  return { confirmed: await Promise.all(confirmed.map((f) => classify({ finding: f }))) };
}
```

The compiler produces this tree:

```text
review/sweep            "Sweep review" — inputs change, dimensions; output confirmed
├─ find                 "Find"    — takes change, dimensions, seen, confirmed, round; hands on fresh too
├─ verify               "Verify"  — its code makes one recorded llm() call per fresh finding
├─ triage               "Triage"  — takes confirmed; calls `classify` once per finding
└─ classify             review/triage, mounted for the calls (never entered by a spine)
```

The transitions are the control flow, one rule per place a phase can go, on the phase's mount:

```jsonc
"children": {
  "find": {
    "transitions": [
      { "name": "to_triage", "when": ".children.find.output._next === 'triage'", "to": "triage",
        "inputs": { "_entry": ".children.find.output._entry", "confirmed": ".children.find.output.confirmed" } },
      { "name": "to_verify", "when": ".children.find.output._next === 'verify'", "to": "verify", "inputs": { … } }
    ]
  },
  "verify": { "transitions": [ { "name": "to_find", … }, { "name": "to_triage", … } ] },
  "triage": { "transitions": [ { "name": "return", "when": ".children.triage.output._next === '$return'", "to": "terminate.success" } ] },
  "classify": { "state": "review/triage", "called": true }
},
"sequence": []
```

Each phase's code returns where it goes next (`_next`), where it enters (`_entry`), and the variables
the next phase reads. The rule for that exit hands those variables over as the target's inputs.

## 3. Three modes

`meta.compile` chooses how a file becomes something hw runs:

| `compile` | The file becomes | Calls are recorded by | Default for |
| --- | --- | --- | --- |
| `"states"` | a state TREE: the root plus one child per phase (§6) | the states, one per phase, and the sites of the calls inside each (§7, §11) | a `.ts`/`.js` under a **workflows** root (`loadBundleFromDir`) |
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
| `limits` | `max_iterations` and `timeout` as for any state, and `budget` (USD, what `budget.total` reports). The call cap is the host's (`EngineConfig.scripts.maxCalls`, default 1000). |
| `compile` | The mode (§3). |
| `phases` | **Claude compatibility only**, and it declares nothing (§6.5). |
| `inputs`, `outputs`, `children`, `sequence`, `transitions`, `operation`, `id` | **Refused.** The signature gives the inputs and outputs, the phases give the children, and the control flow gives the rest. |

`meta` comes FIRST, as Claude requires: before it there may be only imports and type declarations,
which have no run-time effect. `hasScriptMeta` is that test, done without compiling anything.

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

A return that is a RECORD (an object type with properties) becomes one output per property. Any
other return, and a body's whole return, is ONE output named `result`, and the root records that
(`generated.whole`) so a caller of the state gets the value itself back (§10).

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
  first pass through a loop, and every later one). The child takes an `_entry` input naming which
  entry it was entered through, plus the live variables its entries need (§8).
- **Exits.** The child ends where control reaches a `phase()` naming another phase, a `return`, or
  the end of the function. It ends SUCCESSFULLY and outputs its continuation: the phase to go to,
  the entry point there, and the live variables the target needs.
- **Transitions go on the mount.** hw lets a child reach a sibling only through its parent (§3.1),
  so every exit becomes a rule on the child's mount (§3.3): `{ when, to, inputs }`. The rule's
  `inputs` hand the target its live variables, so the target can read them.
- **Guards read the continuation.** The child outputs the decision, and each rule reads it:
  `.children.find.output._next === 'triage'`. *Not built:* LIFTING a branch condition that is an
  expression over the child's outputs into the guard itself (`length(.children.find.output.fresh) ===
  0`), so the rule reads like one a person wrote. §6's expression language has JavaScript semantics,
  so most conditions would lift.
- **`return`** from inside a phase ends that child with the return value (`_next` is `'$return'`),
  and the mount's rule is `{ to: "terminate.success" }`. The root's outputs are bound to whichever
  child returned, which is why they are `optional` with phases (the reachability opt-out).
- **A join reached under two phases.** Code after an `if` whose branches entered different phases
  runs in whichever phase is current, so it is compiled into BOTH children. This is tail
  duplication, and it is invisible to the author.

### 6.3 Loops

A loop whose body crosses phases is a transition back into an earlier phase, and it ends when the
script's own condition does, exactly as the JavaScript would. The phases are mounted outside the
root's sequence (`sequence: []`), so a re-entry is a new pass of that child and resets nothing
else. The run's passes stay addressable as LOOPS.md describes: `.children.find[-2]` is the previous
Find. `meta.limits.max_iterations` is carried to the root like any limit. *Not built:* deriving a
bound from the loop, or enforcing one on the transitions the compiler writes (§16).

### 6.4 Before, between, after

- Code before the first `phase()` is the ROOT's own operation. It runs before any child, as an
  operation always does. A script with no `phase()` at all is a single state whose content is the
  whole body (§7 may still give it inner structure).
- The root's `sequence` is empty. Its own operation returns the first continuation, and the root's
  own rules enter the phase it names, whether or not that depends on a branch.
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

**Built:** a phase's code runs as ONE recorded operation: the `$script` segment the compiler emits for
it, a `switch` over the phase's blocks. Every `llm()`, `agent()` and called state inside it is a call
at a site of its own, dispatched through the engine and journaled (`script.call.settled`), and a
restart part-way through re-runs the phase's code and is handed back what it had already recorded
(§11). `parallel`, `pipeline`, `try`/`catch` and every other construct inside a phase are ordinary
JavaScript.

**Not built — the finer cut.** A phase's code is cut at its AWAIT POINTS, the places it waits on an
`llm()`, a state or a registered function, and the pieces become hw's ordinary constructs:

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

**One gap in `each`, for when the finer cut is built.** An element that fails ends the whole batch
today. `parallel` and `pipeline` turn a failed element into `null` and carry on, so the compiler needs
a per-element failure value: `each` gains `failureValue`, so an element that fails reads as that
value and the batch continues.

**What the cut refuses.** A `try`, a `switch` or a `for…in` whose body contains a `phase()` is a
compile error naming the construct; inside one phase all three are ordinary JavaScript. A `for…of`
that changes phase iterates an array it copies into a variable, so the iterated value must have a
wire form.

## 8. Values across states

A variable that is live where control crosses a state boundary becomes a WIRE: an output of the state
it leaves and an input of the state it enters. The compiler does the liveness analysis, and nothing
the author writes names it.

- **Only wire values cross.** A value crossing a boundary must have a wire form (§7.5.3): JSON, or
  `Date` through its marshaller. A `Set` or a `Map`, a closure, a class instance, or a pending
  promise living across a boundary is a compile error naming the variable and the phase. Claude's
  `seen = new Set()` pattern works as long as the set stays within one phase. Across phases, it is an
  array.
- **Liveness is ordinary backward dataflow over the blocks**, with the `phase()` edges being where a
  live value crosses. A variable used only inside one phase never becomes a wire, and a phase hands
  on only what the phases after it read.
- **Mutation is by value.** `confirmed.push(x)` in one phase is seen by the next because the array
  travels as a wire.
- **Hoisting.** Scoping does not survive being cut into `switch` cases, so a variable declared
  directly in a structure that contains a `phase()` (the function body, or a loop body that changes
  phase) lives in the continuation record `$v`. Everything inside a statement with no `phase()` in it
  keeps its own scoping. The one visible difference: a closure created in a loop that changes phase,
  and called after the iteration, sees the variable's latest value.
- **Static code is not a value.** Module-level code, function declarations, and `const`s whose value
  is a literal or a function are re-declared in every phase's code and never travel.

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
- **Named types are inlined.** A named type is written out where it is used, and a recursive type
  (`type Node = { children: Node[] }`) is unconstrained at the cycle, which is the converter's
  existing rule (§7.5.3). *Not built:* a `$defs` entry per named type, referenced by `$ref`.
- **Descriptions travel.** A JSDoc comment on a property of `T` becomes that property's
  `description` in the schema, which the model reads. `llm<{ /** 0–1, how sure */ confidence: number
  }>` tells the model what the field means.
- **A non-object `T` is wrapped for the wire.** Structured output generally wants an object at the
  root, so `llm<boolean>` asks for `{ value: boolean }` and returns the `boolean`. The hook does the
  wrapping (`lowerLlmCall`), so the script never sees it.
- **`T = string` is text.** With no type argument and no `output`, the call asks for text.
- **`T` and `output.schema` together must agree.** Writing both is allowed (the schema can carry
  what a type cannot: `pattern`, `minimum`, `format`), but the type converted from `T` and the schema
  must describe the same values, checked with `isSubschema` both ways when the call is made. A
  disagreement is an error, because either one alone would validate the answer against something the
  other says it isn't.
- **`T` must have a wire form.** A function, `bigint`, a class instance, or a union the marshaller
  cannot tell apart (§7.5.3) is refused at the call, naming the type.
- **Generic helpers are not.** *Not built:* inlining a helper so `const ask = <T,>(p: string) =>
  llm<T>(p)` is instantiated at each call site. Today a helper's `T` is a type parameter, which
  converts to "anything" with a warning. Write `llm<X>` where `X` is known, or pass `output`.
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
| `llm` | the state's environment chain as literals (`LoadedState.scriptDefaults`: model and knobs, tools, permissions), then `session: null` (each call stands alone unless it names a conversation), `output` text |
| `agent(prompt, opts)` | `llm`'s, under the host's agent layer (`EngineConfig.scripts.agent`), plus `failureValue: null` (Claude's null contract) |

A script can define its own helpers the same way (`const critic = (p) => llm({ ...CRITIC, prompt:
p })`). A helper is re-declared in every phase's code, so the calls inside it are made wherever it
is called.

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
author meant to interpolate, so `{{…}}` in a script's prompt is literal text. No new prompt form was
needed: the text is bound as the call's `prompt` input and the template is `{{.inputs.prompt}}`, and
a render does not re-render what it substitutes.

## 10. Calling states, and imports

**An import specifier is a reference** (§7.5.4), resolved along the same search path as everything
else. What it resolves to decides what the import is:

| Resolves to | Binds | Built |
| --- | --- | --- |
| a state: a `.json`/`.yaml` document, or a script with a `meta` (not `compile: "function"`) | `(inputs) => outputs`, a call of `workflow(<id>, inputs)` | yes. The state is mounted on the script's root with `called: true` — outside the sequence, its inputs each call's rather than wired — so the bundle holds it and the validator sees it. |
| a code module (a `.ts`/`.js` with no `meta`) | its exports | yes. Runs inline in the phase's code, prepared through the module loader and `EngineConfig.scripts.approved`. |
| a prompt (`.md`, a skill) | a prompt callable | not yet |
| a registry entry (`$REGISTRY`) | a typed callable | not yet |
| a data document, or `with { type: "json" }` | the value | not yet. A `.json` beside the script is read as a STATE. |

A module under a workflows root compiles to states when a loader MOUNTS it, but an import treats it as
a state only when it declares a `meta`. A helper module sitting among workflows stays code.

**A call of a state** (an import, or `workflow("…")`) is a real instance under the calling one, with
`instance.entered.calledAt` naming the call's site. It is journaled, on the board, and resumable in
its own right. What differs from an entry the spine or a rule makes is who waits: the script does.
Its end triggers no round and moves no cursor. Its failure is thrown into the script, where a `try`
can catch it. Two calls of one state run side by side, and neither supersedes the other. What it
returns is its outputs, or, for a script whose return is not a record, that return whole
(`generated.whole`), which is what a Claude `workflow()` hands back. The answer is recorded at the
call's site (`script.call.settled`).

Claude's `workflow(nameOrRef, args)` calls a state by reference, and the reference must be a literal:
relative to the script, or from the workflow root. Claude's `{ scriptPath }` is a path. A computed
`workflow()` is refused at compile. A dynamic `import()` and an inline Node builtin fail when the code
reaches them, because the module loader prepared neither (§7.5.4).

*Not built:* `with { as: "operation" }`, which makes an imported function a recorded call rather than
inline code.

## 11. The other two modes

**`"state"`: one state, Claude's execution model.** The whole body is the state's operation. Every
`llm()`, `agent()`, called state, `now()` and `random()` is recorded at a SITE whose key is
`<content hash>#<k>`, where `k` counts the earlier identical calls this run of the code made. Three
identical skeptic calls are therefore three draws, not one memo hit. A `"states"` phase's code is
recorded the same way. On resume the code re-runs from its entry:

- a call with a settled record returns it (`LoadedInstance.scriptCalls`);
- a call with no record runs live.

Recorded results are released in their journaled completion order whenever the code is quiescent, so
a race resolves as it did the first time. `Date.now()`, `Math.random()` and argless `new Date()`
throw in the script's own code, as in Claude, and `now()`/`random()` from `@declarative-ai/hw/script`
are the recorded versions. `phase()` journals `script.phase`, and groups the calls after it.

*Not built:* re-attaching a called state that was still running when the run stopped. Today the call
is made again, and a host building the loaded run leaves an instance with `calledAt` out of its
parent's children. Also not built: the determinism ban reaching imported modules, which run in the
host realm.

**`"function"`: a callable.** The file is a function module on the search path, as `functions/*.ts`
is today, but `llm()` is allowed inside it (`import { llm } from "@declarative-ai/hw/script"`). The
engine dispatches every function call inside a script host for the CALLING instance, so its calls
are recorded under that state, at sites prefixed with the call's own (`<functionRef>@<site>/…`).
*Not built:* a validator warning for a function that calls `llm()` from a guard, which §7.5.6
re-runs every round.

## 12. The compiled files

The compiled tree can be MATERIALIZED as ordinary state files beside the script, the way
`feature.md` sits beside `feature.json` (WORKFLOWS.md §11.3):

```text
workflows/review/sweep.ts          ← the source
workflows/review/sweep.json        ← generated: the root
workflows/review/sweep/find.json   ← generated: a phase
```

- **Compiling is deterministic.** Unlike a description's sync it involves no model, no proposal and
  no review round, and a changed source recompiles exactly.
- **A generated file records its provenance** (`"generated": { "from": "sweep.ts", "inputs": { <path>:
  <sha256>, … } }`). `inputs` lists EVERY file the compile read, not only the script. A module that
  only supplied a type (the `Person` of an `llm<Person>`) never runs, but the schema was generated
  from it, so an edit to it makes the generated files stale exactly as an edit to the script does.
- **The script is the source of truth.** `loadBundleFromDir` compiles every script it finds and
  checks each generated file against it (`staleGenerated`). A stale or hand-edited file is a load
  error that says to edit the script and regenerate. A JSON file that is NOT generated and defines
  the same state as a script is refused too.
- **Ejecting** is deleting the script. The JSON files are then the source, and editing them is
  ordinary authoring.

The library side is `compileScript(options) → { mode, meta, documents, warnings, generated }` and
`generatedFiles(compiled)`, which renders them as `<id>.json` texts. The loader compiles in memory,
so a script runs whether or not its files were ever written. WHETHER to write them, and the ownership
bookkeeping (the nearest-owner rule, `sync.json`), belong to the host. *Not built:* a source map, so
a failed state or a board card can point back to the script line that produced it.

## 13. Types, validation, integrity

- **Types.** A script is compiled under the module compiler configuration (§7.5.6), with the hooks'
  declarations in scope (the globals, and the `@declarative-ai/hw/script` module). A `.js` script is
  read with `allowJs` and its types are inferred. The exported function's signature types the
  state's inputs and outputs, `llm<T>()` is typed by `T`, and a crossing variable by its declared
  or inferred type. The compiled tree is then validated like any authored tree (§6.2), so a wire the
  compiler produced is checked like one a person wrote.
- **Diagnostics.** A compile error names the script's line (`sweep.ts:14:5: …`). An error in the
  compiled tree names the generated state, until the source map exists.
- **Integrity.** The phase's code is carried IN the state document (`operation.script.code`), so it
  is part of the snapshot hash like an embedded body, and a changed script is a different workflow.
  Imported modules go through the module loader, and so through the host's approval gate
  (`EngineConfig.scripts.approved`). A `type`-only import runs nothing. Its effect is the schema it
  produced, which is in the compiled documents and their hash.

## 14. The superset, precisely

**Every Claude workflow script loads.** In `"state"` mode it runs with Claude's execution model and
the same meaning. Under a workflows root it compiles to `"states"`, which it does unless it uses one of
the refused constructs below. The error then names the construct, and `compile: "state"` is the way
out.

| Claude construct | In `"states"` mode |
| --- | --- |
| `phase(title)` | a state boundary (§6), not a display group. The code runs the same way. |
| `meta.phases` | descriptions of the compiled children (§6.5) |
| `phase()` inside a function, a `parallel` thunk or a `pipeline` stage | **refused**: only the control thread moves the phase |
| `phase()` inside a `try`, a `switch` or a `for…in` | **refused** (§7) |
| a value with no wire form (closure, class instance, `Set`, `Map`) live across a phase | **refused** (§8) |
| `workflow(x)` with a computed `x` | **refused** in every mode (§10) |
| `workflow("name")` finds a saved workflow by `meta.name` | resolves as a state id along the workflow root in every mode |
| `agent()` runs Claude Code's subagent | runs the model the call resolves to, route-bound, under the host's agent layer (§9) |
| `budget` is the turn's shared token pool | this state's `limits.budget`, in USD |

**Additions no Claude script can notice:** TypeScript, imports, an exported-function signature,
`llm()`, the extra `meta` keys and `agent` options, and unlimited nesting.

## 15. What changed where

**hw (built).**
- `scriptCompile.ts`: `compileScript`, `generatedFiles` and `staleGenerated`; the `meta` reader, the
  signature, the control-flow cut at `phase()`, block liveness into wires, the phase segments, the
  root's rules and outputs, and the classification of imports into code and called states.
- `scriptHooks.ts`: the hooks and their declarations; the `AsyncLocalStorage` host a script (or a
  function module) finds its caller through. `moduleLoader.ts` serves `@declarative-ai/hw/script` as
  a provided module, and `signature.ts` resolves its types.
- `scriptRun.ts` and `engine.ts`: the `$script` operation, `lowerLlmCall`/`agentCall`, the site
  ordinal and replay scheduler, called states, and the function host. `EngineConfig.scripts`.
- `format.ts`/`loader.ts`: `operation.script`, `SCRIPT_FUNCTION`, `SCRIPT_CONTROL`,
  `LoadedState.scriptDefaults`, `StateDef.generated` and `whenToUse`, `limits.budget`; `load.ts`:
  `LoadedInstance.scriptCalls`; `ports.ts`: `script.call.settled`, `script.log`, `script.phase`,
  `instance.entered.calledAt`. `loadBundleFromDir` compiles scripts. API.md, "Workflow scripts".

**hw (not built).** The finer cut inside a phase and `each.failureValue` (§7). Lifted guards
(§6.2). A loop bound (§6.3). `$defs` for named types and generic-helper inlining (§9). Prompt,
registry and data imports, and `with { as: "operation" }` (§10). Re-attaching a running called
state, and the determinism ban in imported modules (§11). The source map (§12). SPEC §7.6.

**JaiRA (must migrate).**
- Read the scripts in the workflow layers: compile them (`compileScript`) into the `files` map
  `loadBundle` takes, with `stateIdOf` for the layered roots.
- Write the generated files and record their ownership, if they are to be materialized.
- Wire `EngineConfig.scripts` (the layers' vfs and require path, the approval gate, and the agent
  layer that means Claude Code).
- Build `LoadedInstance.scriptCalls` from the `script.*` rows, and leave `calledAt` instances out of
  their parent's children.
- Project `script.call.settled` as the calls of a state, grouped by `phase`, and draw a script's
  phases as its sub-board.

## 16. Open questions

1. **The finer cut (§7).** Per-call states are more visible and resume at a finer grain, but they
   cost a JavaScript → expression translation. Is the recorded phase segment enough for now?
2. **Guard lifting (§6.2).** `_next` guards are correct but opaque on a board. Lift the conditions
   that fit the expression language, and label the rest with the source condition's text?
3. **Loop bounds (§6.3).** Should a loop across phases require `limits.max_iterations`, and should the
   compiler enforce it on the transitions it writes? A phase change is a transition, so
   `run.iteration` already counts them.
4. **Phase keys.** `"Find"` → `find` is a slug, with an error on collision. Should an author be able
   to name the key separately?
5. **Nested phases.** A phase is a direct child of the script, and nesting comes from calling another
   script. Keep it that way?
6. **Tail duplication.** A join after branches that entered different phases puts the same code in
   both states, so a call there appears under whichever phase was current. Acceptable?
7. **Called states and `.children`.** A called state is mounted on the root but read by nobody. Should
   `.children.<key>` read a script's calls of it, so a rule after the body can?
