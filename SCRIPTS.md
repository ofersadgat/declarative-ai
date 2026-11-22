# Scripts — a state machine written as code

**Status: BUILT in hw. JaiRA not migrated.** `compileScript` (`scriptCompile.ts`), the hooks
(`scriptHooks.ts`) and the engine's `$script` operation (`scriptRun.ts`, `engine.ts`) implement §3–§14.
`loadBundleFromDir` compiles a directory's scripts. `scripts.test.ts` has 53 tests, and the whole suite
(2860) passes. SPEC §7.6 states what the rest of the specification relies on.

What the implementation found that the first draft of this document did not anticipate:

- **The wiring checker does not reason about unions or tuples**, and a variable's TypeScript type
  often has one (`(Finding | null)[]` out of `parallel`). Inside a generated wire's type both are
  widened to "anything". Both ends of every generated wire are the same variable, so the checker loses
  nothing it could have used.
- **Reachability cannot see which state returned.** The root's outputs read "whichever state returned
  the value", which is a run-time fact, so they are `optional` (the §6.2 opt-out) and the script's
  return type is what guarantees them.
- **`Set` and `Map` have no wire form** (`wireType.ts` refuses both), so one live across a cut is a
  compile error, as §8 says of a closure.
- **TypeScript infers an untyped `llm()`'s `T` from context**, and the compiler follows the checker:
  `flag || await llm("…")` has `T = boolean`, because `||`'s right operand is contextually typed by its
  left. Write `llm<string>` where the context would say otherwise.
- **A failed state publishes no outputs**, so a rule that catches one reads what it was handed and
  how it failed: `.children.<key>.inputs` and `.children.<key>.failure` joined the ref vocabulary.
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
>    TRANSITION (§6). Inside a phase, every non-deterministic call is a state of its own (§7).
> 3. **`llm()` is the one primitive.** It takes the full LLM operation and goes through route
>    binding. `agent()` and every other helper are `llm()` with different defaults (§9).
> 4. **How finely to compile is a mode.** A script can instead be cut only at its phases, run as ONE
>    state (Claude's exact execution model), or be a plain callable function (§3).

---

## 1. Why

hw's JSON format says exactly what a state machine is, and that is the problem when the machine is
complicated. The run-11 `feature/product` loop, a critique loop with an escape hatch, or "find
until two rounds come back dry, then verify each" are each a few lines of code and a page of
transitions, guards, wires and iteration limits. Every one of those has to be kept consistent by
hand. LOOPS.md exists largely because a wire that looked right in JSON silently carried nothing.

Writing the machine as code and COMPILING it keeps everything hw is for: the board shows the phases
and the calls inside them, a person can steer between them, the journal records every step, a
restart resumes from the last state, and the validator types every value. Durability is not
reinvented, because the output is ordinary states. What changes is who writes the transitions: the
compiler, from control flow the author already wrote.

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

The compiler produces this tree (in the default `"calls"` mode, §3):

```text
review/sweep                      "Sweep review" — inputs change, dimensions; output confirmed
├─ find                           "Find"   — takes change, dimensions, seen, confirmed, round
│  ├─ map_0 → find/map_0          "each: parallel" — one element per dimension, async, failureValue null
│  │  └─ call_3                   "agent: Review {change.ref} for {d}. …" — the call, then the code after it
│  └─ after_0                     "after parallel" — dedupe, the `break`, `seen`; decides Verify or Triage
├─ verify                         "Verify"
│  ├─ map_1 → verify/map_1        one element per fresh finding
│  │  └─ call_4                   "llm: Try to refute: {f.claim}" — a prompt operation, model literal
│  └─ after_1                     `confirmed`, the loop's step and test
└─ triage                         "Triage"
   ├─ map_2 → triage/map_2        one element per confirmed finding
   │  ├─ triage_5 → review/triage called: true — the imported state, mounted where it is called
   │  └─ after_5
   └─ after_2                     returns
```

The transitions are the control flow. Between phases, one rule per place a phase can go, on the
phase's mount; the loop's test, which reads only variables, is LIFTED into the guard:

```jsonc
"children": {
  "find": { "transitions": [
    { "name": "to_triage", "when": ".children.find.output._next === 'triage'", "to": "triage",
      "inputs": { "_entry": ".children.find.output._entry", "confirmed": ".children.find.output.confirmed" } },
    { "name": "to_verify", "when": ".children.find.output._next === 'verify'", "to": "verify", "inputs": { … } } ] },
  "verify": { "transitions": [ { "name": "to_find", … }, { "name": "to_triage", … } ] },
  "triage": { "transitions": [ { "name": "return", "when": ".children.triage.output._next === '$return'", "to": "terminate.success" } ] }
},
"sequence": [],
"transitions": [
  { "name": "to_find", "when": ".operation.output.round < 3", "to": "find", "inputs": { … } },
  { "name": "to_triage", "when": "!(.operation.output.round < 3)", "to": "triage", "inputs": { … } }
]
```

Each state's code returns where it goes next (`_next`), where it enters (`_entry`), and the variables
the states after it read. The rule for that exit hands those variables over as the target's inputs.

## 3. Four modes

`meta.compile` chooses how finely a file is cut, coarsest to finest:

| `compile` | The file becomes | A call is recorded as |
| --- | --- | --- |
| `"function"` | a CALLABLE on the search path, exactly as a function module is (SPEC §7.5), but allowed to call `llm()` | a site under the caller's call (§11) |
| `"state"` | ONE state whose operation is the whole script. `phase()` only groups calls, as in Claude. | a journaled site, replayed on resume (§11) |
| `"phases"` | a state TREE: the root plus one child per phase (§6), each phase's code one operation | a journaled site inside the phase's state |
| `"calls"` | the phases, and inside each a state per non-deterministic call (§7) | the call's own state |

`"calls"` is the default, both of `compileScript` and of a `.ts`/`.js` under a **workflows** root
(`loadBundleFromDir`). A `.ts`/`.js` anywhere else (`functions/`, `lib/`) is a function module, as it
always was. A compile that cannot cut the code (§14) is an ERROR that names the construct. It never
silently falls back, because a script that quietly stopped being a state tree would lose the board,
the steering and per-state resume without the author knowing. A coarser `compile` is the way out.

`"state"` is Claude's exact execution model. It is what guarantees the superset: every Claude script
loads in it, even one the compiler cannot cut.

## 4. `meta`

`meta` holds the STANDARD PROPERTIES of the state the file is: the fields that describe it and say how
it runs, and none of the fields that describe its structure. Structure comes from the code.

| Key | Meaning |
| --- | --- |
| `name` / `label` | The static display name. `name` is Claude's spelling, and writing both is refused. |
| `description` | As in SPEC §5.2. |
| `whenToUse` | Claude's key. Documentation for whatever chooses among workflows. |
| `title` | The instance's display name, evaluated once at entry. It may read `.inputs` (§5). |
| `environment` | Defaults for every operation in the compiled tree, inherited as usual (SPEC §7.1a). |
| `limits` | `max_iterations` and `timeout` as for any state, and `budget` (USD, what `budget.total` reports). The call cap is the host's (`EngineConfig.scripts.maxCalls`, default 1000). |
| `compile` | The mode (§3). |
| `phases` | **Claude compatibility only**, and it declares nothing (§6.5). |
| `inputs`, `outputs`, `children`, `sequence`, `transitions`, `operation`, `id` | **Refused.** The signature gives the inputs and outputs, the phases and calls give the children, and the control flow gives the rest. |

`meta` comes FIRST, as Claude requires: before it there may be only imports and type declarations,
which have no run-time effect. `hasScriptMeta` is that test, done without compiling anything.

`meta` is a PURE LITERAL, as in Claude: literals, comments, `as const`/`satisfies` stripped, and no
identifiers, calls, spreads or interpolation. The loader reads it without running anything. A
computed value is a binding written as data (`title: { binding: { $expr: … } }`), as in JSON.

## 5. Signature

**If the file exports a function, its signature is the state's.** Its parameters are the inputs and
its return type is the output, read by the existing extractor (SPEC §7.5.2): names, positions, types
converted to wire schemas, `?` and defaults as optional, JSDoc `@param` as descriptions. The function
is the default export, or else the only exported function. Several exported functions and no default
is an error when compiling to states (which one is the state?), and ordinary in `"function"` mode,
where a module contributes a symbol per export as it does today.

**If it does not, the file's body is the function.**

- **Input: `args`**, Claude's global: one optional, unconstrained input named `args`. A parent wires
  it as `"inputs": { "args": … }`.
- **Output: the export.** `export default <value>` is the whole output, and named exports
  (`export const confirmed = …`) are outputs by name, typed by the variables they are. Claude's
  top-level `return x` is also the whole output, since that is how a Claude script returns. Writing
  both a `return` and an export is an error.

A return that is a RECORD (an object type with properties) becomes one output per property. Any
other return, and a body's whole return, is ONE output named `result`, and the root records that
(`generated.whole`) so a caller of the state gets the value itself back (§10).

Module-level code outside the exported function (constants, helper functions, schemas) is STATIC:
it is available to every compiled state and is not itself a step.

## 6. Phases

### 6.1 The meaning

> **`phase("X")` says: the code from here on belongs in child state X.** If X is already the current
> phase, it does nothing. If another phase is current, it is a transition to X.

That is the whole semantics, and it is the same whether the call sits in a loop, an `if`, a `switch`
or a `try`. Before the first `phase()` no child is current, and code runs in the root state itself.

- **Phase → key.** `"Find"` becomes key `find` and label `"Find"`. Two titles that map to one key
  are an error. The title must be a string literal, because a computed phase has no state to be.
- **Where a call can move the phase.** Only the script's own control flow can, so `phase()` inside a
  function, a `parallel` thunk or a `pipeline` stage is a compile error. Claude's `opts.phase` on an
  `agent()` inside one keeps its Claude meaning, a display group, and moves nothing.

### 6.2 How a phase becomes a state

A child state's code is every piece of the program that can run while that phase is current. The
compiler finds it by walking the control flow and tracking the current phase:

- **Entry points.** A phase is entered at a `phase("X")` call site, and there may be several (the
  first pass through a loop, and every later one). The child takes an `_entry` input naming which
  entry it was entered through, plus the live variables its entries need (§8).
- **Exits.** The child ends where control reaches a `phase()` naming another phase, a `return`, or
  the end of the function. It ends SUCCESSFULLY and outputs its continuation: the phase to go to,
  the entry point there, and the live variables the target needs.
- **Transitions go on the mount.** hw lets a child reach a sibling only through its parent (SPEC
  §3.1), so every exit becomes a rule on the child's mount (SPEC §3.3): `{ name, when, to, inputs }`.
  The rule's `inputs` hand the target its live variables. The composite's OWN code — what runs
  before its first call — has no mount, so the rules after it are on the composite's own list,
  written `after: "operation"`: they answer that code finishing, once, and no later round that finds
  nothing else to do enters its first call again (SPEC §3.3).
- **Guards are LIFTED where they can be.** A branch whose condition reads only variables the state
  outputs becomes the guard itself, in the expression language (`.operation.output.round < 3`,
  `.children.call_0.output.r.ok`), so the rule reads like one a person wrote. A condition that
  cannot be written that way (it calls a helper, say) is decided by the code, and the rule reads the
  continuation: `.children.find.output._next === 'triage'`.
- **`return`** ends the state with the return value (`_next` is `'$return'`), and the rule is
  `{ to: "terminate.success" }`. The root's outputs read whichever state returned, which is why they
  are `optional` (the reachability opt-out).
- **A join reached under two phases.** Code after an `if` whose branches entered different phases
  runs in whichever phase is current, so it is compiled into BOTH children. This is tail
  duplication, and it is invisible to the author.

### 6.3 Loops

A loop whose body crosses a cut is a transition back to an earlier state, and it ends when the
script's own condition does, exactly as the JavaScript would. The compiler adds no bound of its own:
a loop that runs forever in JavaScript runs forever here, and `meta.limits.max_iterations` is the
author's to set, as for any state. The phases are mounted outside the root's sequence
(`sequence: []`), so a re-entry is a new pass of that child and resets nothing else. The run's passes
stay addressable as LOOPS.md describes: `.children.find[-2]` is the previous Find.

### 6.4 Before, between, after

- Code before the first `phase()` or call is the machine's OWN operation. It runs before any child,
  as an operation always does, and returns the first continuation; the machine's own rules enter the
  child it names. The root's `sequence` is empty.
- A throw that nothing catches is the state's `error`, which becomes its parent's `error` (SPEC
  §3.3), exactly as a JS exception propagates. A `try` around a cut is compiled (§7).

### 6.5 Claude's `meta.phases`

In Claude, `meta.phases` is **display only**. It pre-declares the phase groups so the progress view
and the permission dialog can show the plan before anything runs, with a `detail` line per phase and
an optional `model` label. It is matched to `phase()` calls by exact title, a `phase()` call with no
entry just gets its own group, and nothing executes from it.

A compiled script already knows its phases before it runs, because they are the literals in its
`phase()` calls, so the list is redundant as structure. It is ACCEPTED as descriptions of the
compiled children: `detail` becomes the child's `description`. It declares nothing. An entry no
`phase()` call names is a warning. A `model` label is ignored, since in hw a model is set on the call
or the environment and not on a label.

## 7. Inside a phase: a state per call

**The rule: no state makes more than one non-deterministic call.** In `"calls"` mode the code is cut
at every call whose answer is not a function of its arguments — `llm()`, `agent()`, a called state
(§10), a registered function, an imported operation, `now()` and `random()` — and each becomes a state
of its own. Everything between calls is deterministic JavaScript, and runs as the code of the state
it falls in. The cut does not depend on phases: a script with no `phase()` is a root whose children
are its calls.

**`now()` and `random()` are calls like any other.** They are non-deterministic by definition, so
they are never memoized: two `now()` calls measuring an interval are two readings, each its own state.
What they return is recorded like any answer, so a restart reads it back rather than asking again.
An author who wants one value twice keeps it in a variable.

**What a call becomes:**

| Code | Compiles to |
| --- | --- |
| `await llm(…)` whose options the compiler can read | a state `call_<n>` whose operation is the LIST (SPEC §7.1d) `[the prompt operation, the code after it]`. A literal option is literal (`model: "anthropic/claude-sonnet-5"`); a computed one is a typed input of the state (`model: { $expr: ".inputs.model_1" }`), so the wiring checker reads it. `T` and `output` must be known when compiling. |
| `agent(…)`, an `llm()` built at run time, a registered function, an imported operation, `now()`, `random()` | the same list, with the call made by a `$script` operation of its own |
| a called state | a mount `<name>_<n>` of that state with `called: true`, the call's arguments its inputs, then a state `after_<n>` for the code after it |
| `parallel(xs.map(x => () => …))`, `Promise.all(xs.map(…))`, `pipeline(items, s1, s2, …)` | a FAN-OUT: a mount `map_<n>` with `each` over the items, `async: true`, whose state (`<machine>/map_<n>`) is the callback compiled as a machine of its own — then `after_<n>` |
| `Promise.all([llm(a), llm(b)])`, `parallel([() => …, () => …])` — a list of calls | a fan-out over the list's indices, each element running its call |
| a call inside `&&`, `\|\|`, `??` or `?:` | a BRANCH: the call is made only when JavaScript would make it |
| a call buried in an expression | lifted into a statement of its own, with everything JavaScript evaluates before it held in a temporary, so evaluation order is kept exactly |
| a helper that makes a call | INLINED where it is called, once per call of it; a generic one with its caller's `T` substituted. A helper that calls itself is refused. |
| an `if`, a loop or a `switch` around calls | rules between the call states, forward or backward, as §6.2 |
| `try { … } catch` around a call | a rule on the call's mount that handles its failure, named `catch`, entering the state the `catch` block compiled to with the variables as they were before the call (`.children.<call>.inputs.<name>`) and the error (`.children.<call>.failure`) |
| a promise stored and awaited later (`const p = llm(…); …; await p`) | the call, made where it is written, as if awaited there |

**Fan-outs.** Parallel and pipeline turn a failed element into `null` and carry on, so their mount
carries `failureValue: null` (a mount's `failureValue` is what a failed element reads as); a failed
element of `Promise.all` fails the fan-out, as it rejects in JavaScript. A pipeline's element runs its
stages in order for its item, and the items run side by side. An element reads the variables it needs
from outside it as inputs, and may not WRITE one: each item runs apart from the others, so it returns
what it computed instead.

Generated states are real states. They show on the phase's sub-board, resume like any other, and
carry the source line they came from (§12), but no name the author wrote refers to them.

**What the cut refuses**, each at its line with the way out: a `finally` around a cut; a `for…in` or
`for await` that is cut; a call in a `for` loop's condition or step, a `do…while` condition, or a
`case` label; a fan-out over a list made elsewhere (`parallel(thunks)`); a callback that is neither
written in place nor a helper's name; a recursive helper that makes a call. A `for…of` that is cut
iterates an array it copies into a variable, so the iterated value must have a wire form.

## 8. Values across states

A variable that is live where control crosses a cut becomes a WIRE: an output of the state it leaves
and an input of the state it enters. The compiler does the liveness analysis, and nothing the author
writes names it.

- **Only wire values cross.** A value crossing a cut must have a wire form (SPEC §7.5.3): JSON, or
  `Date` through its marshaller. A `Set` or a `Map`, a closure, a class instance, or a pending
  promise living across a cut is a compile error naming the variable. Claude's `seen = new Set()`
  pattern works as long as the set stays between two calls. Across one, it is an array.
- **Liveness is ordinary backward dataflow over the blocks**, with the cuts being where a live value
  crosses. A variable used only between two cuts never becomes a wire, and a state hands on only what
  the states after it read. A fan-out's element takes only the outer variables its callback reads.
- **Mutation is by value.** `confirmed.push(x)` before a call is seen after it because the array
  travels as a wire.
- **Hoisting.** Scoping does not survive being cut into `switch` cases, so a variable declared
  directly in a structure that contains a cut (the function body, or a loop body that is cut) lives
  in the continuation record `$v`. Everything inside a statement with no cut in it keeps its own
  scoping. The one visible difference: a closure created in a loop that is cut, and called after the
  iteration, sees the variable's latest value.
- **Static code is not a value.** Module-level code, function declarations, and `const`s whose value
  is a literal nobody changes or a function are re-declared in every state's code and never travel.

## 9. `llm()` and the functions built on it

```ts
llm<T = string>(op: LlmOperation): Promise<T>
llm<T = string>(prompt: string, op?: Partial<LlmOperation>): Promise<T>
```

`LlmOperation` is the full prompt operation of SPEC §7.1 together with its environment fields:
`prompt`, `system`, `config` (model and every knob, `configRef` included), `input`, `output`, `tools`,
`session`, `workspace`, `conversation` and `permissions`, plus the binding-level `failureValue` and a
display `label`.

**The type argument IS the output contract.**

```ts
const destructive = await llm<boolean>({ prompt: "Is the following command destructive? " + command });
const { foo } = await llm<{ foo: number }>({ prompt: "Tell me the value of foo in: " + JSON.stringify(command) });
```

The compiler converts `T` to the call's output schema with the converter that already reads a
function module's parameter types (SPEC §7.5.2, §7.5.3). There is one TS → wire path, and the result
is the call's `output: { schema }`: the structured-output contract the executor asks the model for,
and the schema the answer is validated against. `T` is a compile-time fact the compiler turns into
data. At run time the call has no type argument, only the schema.

- **`T` is any type the checker can resolve, not only a literal.** `llm<Person>(…)` works whether
  `Person` is a `type` or `interface` in the script, or imported (`import type { Person } from
  "$/lib/people"`). It also works for anything built from such types: `Person[]`,
  `Pick<Person, "name">`, `Page<Person>`. The compiler asks the TypeScript checker for the type at
  the call site and converts that, so an imported type resolves through the same resolver the
  type-check uses (SPEC §7.5.4).
- **With no type argument, `T` is what TypeScript infers** at the call, from its context included
  (`const ok: boolean = await llm(…)`). No context leaves the default, `string`.
- **Named types go in `$defs`.** A named object type below the root of `T` is converted once, placed
  in the schema's `$defs`, and referenced by `$ref` wherever it is used; a recursive one refers to
  itself there.
- **Descriptions travel.** A JSDoc comment on a property of `T` becomes that property's
  `description` in the schema, which the model reads.
- **A string `T` is text.** The call asks for text and returns it, as with no type argument.
- **Any other non-object `T` is wrapped for the wire.** Structured output wants an object at the
  root, so `llm<boolean>` asks for `{ value: boolean }` and returns the `boolean`. The script never
  sees the wrapper.
- **`T` and `output.schema` together must agree.** Writing both is allowed (the schema can carry
  what a type cannot: `pattern`, `minimum`, `format`), but they must describe the same values,
  checked with `isSubschema` both ways. A disagreement is an error.
- **`T` must have a wire form.** A function, `bigint`, a class instance, or a union the marshaller
  cannot tell apart (SPEC §7.5.3) is refused at the call, naming the type.
- **Generic helpers work.** `const ask = async <T,>(p: string) => llm<T>(p)` is inlined at each call
  in `"calls"` mode with the caller's `T` substituted; in the coarser modes the caller's `T` is handed
  to the helper as a schema, so `ask<Verdict>("…")` asks for a `Verdict` either way.
- **JavaScript has no type arguments.** A `.js` script (every Claude script) says the same thing with
  `output: { schema }`, or Claude's `agent(p, { schema })`.

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
| `llm` | the state's environment chain (`LoadedState.scriptDefaults`: model and knobs, tools, permissions), then `session: null` (each call stands alone unless it names a conversation), `output` text |
| `agent(prompt, opts)` | `llm`'s, under the host's agent layer (`EngineConfig.scripts.agent`), plus `failureValue: null` (Claude's null contract) |

A script can define its own helpers the same way (`const critic = (p) => llm({ ...CRITIC, prompt:
p })`).

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
author meant to interpolate, so `{{…}}` in a script's prompt is literal text: the text is bound as an
input and the template is `{{.inputs._call.prompt}}`, and a render does not re-render what it
substitutes. A prompt IMPORT is the exception, and on purpose (§10).

## 10. Calling states, and imports

**An import specifier is a reference** (SPEC §7.5.4), resolved along the same search path as
everything else. What it resolves to decides what the import is:

| Resolves to | Binds |
| --- | --- |
| a state: a `.json`/`.yaml` document, or a script with a `meta` (not `compile: "function"`) | `(inputs) => outputs` — a CALLED STATE (below) |
| a code module (a `.ts`/`.js` with no `meta`) | its exports, run inline, prepared through the module loader and `EngineConfig.scripts.approved` |
| code, `with { as: "operation" }` | its functions as RECORDED calls: a state per call in `"calls"` mode, a journaled site (`recorded:<name>#k`) in the coarser ones. Recording is asynchronous, so the call is awaited where it stands, and one in a synchronous function is refused. |
| a prompt (`.md`) | `(inputs) => answer`: the file is the call's template, and its `{{.inputs.x}}` holes are the call's inputs |
| `$REGISTRY` | its named imports as calls of those registered functions |
| a data document, `with { type: "json" }` | the value, inlined when compiling |

A module under a workflows root compiles to states when a loader MOUNTS it, but an import treats it as
a state only when it declares a `meta`. A helper module sitting among workflows stays code.

**A call of a state** (an import, or `workflow("…")`) is a real instance. In `"calls"` mode it is a
mount of that state at the call (`called: true`, its inputs the call's arguments). In the coarser modes
it is an instance under the calling one, with `instance.entered.calledAt` naming the call's site, and
the code waits on it: its end triggers no round and moves no cursor, its failure is thrown into the
script where a `try` can catch it, and two calls of one state run side by side. What it returns is its
outputs, or, for a script whose return is not a record, that return whole (`generated.whole`), which is
what a Claude `workflow()` hands back.

Claude's `workflow(nameOrRef, args)` calls a state by reference, and the reference must be a literal:
relative to the script, or from the workflow root. Claude's `{ scriptPath }` is a path. A computed
`workflow()` is refused at compile. A dynamic `import()` and an inline Node builtin fail when the code
reaches them, because the module loader prepared neither (SPEC §7.5.4).

## 11. The coarser modes

**`"state"`: one state, Claude's execution model, and `"phases"`: a state per phase.** The code runs as
written. Every `llm()`, `agent()`, called state, registered function, recorded import, `now()` and
`random()` is recorded at a SITE whose key is `<content hash>#<k>`, where `k` counts the earlier
identical calls this run of the code made. Three identical skeptic calls are therefore three draws,
not one memo hit. On resume the code re-runs from its entry:

- a call with a settled record returns it (`LoadedInstance.scriptCalls`);
- a called state still running when the run stopped, or ended with its answer unsettled, is
  RE-ATTACHED (`LoadedInstance.calledAt`): continued if live, read if it ended — never called again;
- any other call runs live.

Recorded results are released in their journaled completion order whenever the code is quiescent, so
a race resolves as it did the first time. `phase()` journals `script.phase`, and groups the calls
after it.

**Determinism, in every mode.** `Date.now()`, `Math.random()` and argless `new Date()` throw in the
script's own code and in every module it imports, as in Claude; `now()` and `random()` from
`@declarative-ai/hw/script` are the recorded versions.

**`"function"`: a callable.** The file is a function module on the search path, as `functions/*.ts`
is today, but `llm()` is allowed inside it (`import { llm } from "@declarative-ai/hw/script"`). The
engine dispatches every function call inside a script host for the CALLING instance, so its calls
are recorded under that state, at sites prefixed with the call's own (`<functionRef>@<site>/…`). A
function whose module — or anything it imports — imports the hooks is marked `callsModels`, and one
called from a guard is warned about: a guard is re-evaluated every round and its completed calls are
not memoized (SPEC §7.5.6), so it would ask, and pay, again each time.

## 12. The compiled files

The compiled tree can be MATERIALIZED as ordinary state files beside the script, the way
`feature.md` sits beside `feature.json` (WORKFLOWS.md §11.3):

```text
workflows/review/sweep.ts                ← the source
workflows/review/sweep.json              ← generated: the root
workflows/review/sweep/find.json         ← generated: a phase
workflows/review/sweep/find/map_0.json   ← generated: a fan-out's element
```

- **Compiling is deterministic.** Unlike a description's sync it involves no model, no proposal and
  no review round, and a changed source recompiles exactly.
- **A generated file records its provenance** (`"generated": { "from": "sweep.ts", "inputs": { <path>:
  <sha256>, … }, "at": { "line", "column" } }`). `inputs` lists EVERY file the compile read, not only
  the script. A module that only supplied a type (the `Person` of an `llm<Person>`) never runs, but the
  schema was generated from it, so an edit to it makes the generated files stale exactly as an edit to
  the script does.
- **The source map.** `at` is the script line the state came from, and `CompiledScript.sourceMap`
  has it for every state, so a failed state or a board card points back to the code. A failure thrown
  by a state's code names the line that threw (`sweep.ts:14: …`).
- **The script is the source of truth.** `loadBundleFromDir` compiles every script it finds and
  checks each generated file against it (`staleGenerated`). A stale or hand-edited file is a load
  error that says to edit the script and regenerate. A JSON file that is NOT generated and defines
  the same state as a script is refused too.
- **Ejecting** is deleting the script. The JSON files are then the source, and editing them is
  ordinary authoring.

The library side is `compileScript(options) → { mode, meta, documents, warnings, generated,
sourceMap }` and `generatedFiles(compiled)`, which renders them as `<id>.json` texts. The loader
compiles in memory, so a script runs whether or not its files were ever written. WHETHER to write
them, and the ownership bookkeeping (the nearest-owner rule, `sync.json`), belong to the host.

## 13. Types, validation, integrity

- **Types.** A script is compiled under the module compiler configuration (SPEC §7.5.6), with the
  hooks' declarations in scope (the globals, and the `@declarative-ai/hw/script` module). A `.js`
  script is read with `allowJs` and its types are inferred. The exported function's signature types
  the state's inputs and outputs, `llm<T>()` is typed by `T`, a computed call option by the option's
  type, and a crossing variable by its declared or inferred type. The compiled tree is then validated
  like any authored tree (SPEC §6.2), so a wire the compiler produced is checked like one a person
  wrote.
- **Diagnostics.** A compile error names the script's line (`sweep.ts:14:5: …`); a generated state
  names it through the source map (§12).
- **Integrity.** A state's code is carried IN its document (`operation.script.code`), so it is part
  of the snapshot hash like an embedded body, and a changed script is a different workflow. Imported
  modules go through the module loader, and so through the host's approval gate
  (`EngineConfig.scripts.approved`); `generated.modules` names the ones the compiled states run, so a
  host can ask for approval and freeze them before a run starts, as it does a function module. A
  `type`-only import runs nothing and is not listed. Its effect is the schema it produced, which is
  in the compiled documents and their hash.

## 14. The superset, precisely

**Every Claude workflow script loads.** In `"state"` mode it runs with Claude's execution model and
the same meaning. Under a workflows root it compiles to `"calls"`, which it does unless it uses one of
the refused constructs below. The error then names the construct, and a coarser `compile` is the way
out.

| Claude construct | Compiled (`"phases"`, `"calls"`) |
| --- | --- |
| `phase(title)` | a state boundary (§6), not a display group. The code runs the same way. |
| `meta.phases` | descriptions of the compiled children (§6.5) |
| `phase()` inside a function, a `parallel` thunk or a `pipeline` stage | **refused**: only the script's own control flow moves the phase |
| a `finally`, a `for…in` or a `for await` around a cut | **refused** (§7) |
| a value with no wire form (closure, class instance, `Set`, `Map`) live across a cut | **refused** (§8) |
| a fan-out element that writes an outer variable (`"calls"`) | **refused** (§7) |
| `workflow(x)` with a computed `x` | **refused** in every mode (§10) |
| `workflow("name")` finds a saved workflow by `meta.name` | resolves as a state id along the workflow root in every mode |
| `agent()` runs Claude Code's subagent | runs the model the call resolves to, route-bound, under the host's agent layer (§9) |
| `budget` is the turn's shared token pool | this state's `limits.budget`, in USD |

**Additions no Claude script can notice:** TypeScript, imports, an exported-function signature,
`llm()`, the extra `meta` keys and `agent` options, and unlimited nesting.

## 15. What changed where

**hw.**
- `scriptCompile.ts`: `compileScript`, `generatedFiles` and `staleGenerated`; the `meta` reader, the
  signature, the classification of imports, the lifting of calls out of expressions, helper
  inlining, fan-outs, the control-flow cut at `phase()` and every call, block liveness into wires,
  lifted guards, and the machines — root, phase, element — each a composite of call states.
- `scriptHooks.ts`: the hooks and their declarations; the `AsyncLocalStorage` host a script (or a
  function module) finds its caller through. `moduleLoader.ts` serves `@declarative-ai/hw/script` as
  a provided module and injects the determinism globals, and `signature.ts` resolves its types.
- `scriptRun.ts` and `engine.ts`: the `$script` operation, `lowerLlmCall`/`agentCall`, the site
  ordinal and replay scheduler, called states and their re-attachment, a fan-out's `failureValue`,
  and the function host. `EngineConfig.scripts`.
- `format.ts`/`loader.ts`: `operation.script`, `SCRIPT_FUNCTION`, `SCRIPT_CONTROL`,
  `LoadedState.scriptDefaults`, `StateDef.generated` and `whenToUse`, `limits.budget`,
  `ChildDecl.called` and `failureValue`; `load.ts`: `LoadedInstance.scriptCalls` and `calledAt`;
  `ports.ts`: `script.call.settled`, `script.log`, `script.phase`, `instance.entered.calledAt`;
  `validate.ts`: `.children.<key>.inputs`/`.failure`, and the guard warning. `loadBundleFromDir`
  compiles scripts. `wireType.ts`: `$defs`.
- `userFunctions.ts` and ops' `HostCapabilities.callsModels`.
- SPEC §6.1 and §7.6; API.md, "Workflow scripts".

**JaiRA (must migrate).**
- Read the scripts in the workflow layers: compile them (`compileScript`) into the `files` map
  `loadBundle` takes, with `stateIdOf` for the layered roots.
- Write the generated files and record their ownership, if they are to be materialized.
- Wire `EngineConfig.scripts` (the layers' vfs and require path, the approval gate, and the agent
  layer that means Claude Code).
- Build `LoadedInstance.scriptCalls` from the `script.*` rows, and list an instance with
  `instance.entered.calledAt` among its caller's children with `calledAt` set.
- Show the generated states on their phase's sub-board, labelled and pointing at `generated.at`;
  project `script.call.settled` as the calls of a state in the coarser modes, grouped by `phase`.

## 16. Open questions

1. **Phase keys.** `"Find"` → `find` is a slug, with an error on collision. Should an author be able
   to name the key separately?
2. **Nested phases.** A phase is a direct child of the script, and nesting comes from calling another
   script. Keep it that way?
3. **Tail duplication.** A join after branches that entered different phases puts the same code in
   both states, so a call there appears under whichever phase was current. Acceptable?
4. **Guards that do not lift.** A condition the expression language cannot say is decided by the
   code and read as `_next`. Label such a rule with the source condition's text?

## 17. A script is also a program: durable and non-durable

**Status: BUILT** (2026-11-18, for JaiRA decision 0029). One file runs two ways:

| | Durable | Non-durable |
| --- | --- | --- |
| How | compiled to states and run by the engine (§3) | `dai-script my-workflow.ts [args…]` — `node` with the hooks wired |
| `llm()`, `agent()`, `input()` | a state each: journaled, resumable | just run |
| An imported workflow | mounted as a child state | a function call |
| After a crash | resumes from the last state | gone |

- **The hooks are a real import.** `@declarative-ai/hw/script` is an export of hw (`src/script.ts`), so
  plain `node` resolves it. Under the engine a hook reaches the run's host through `AsyncLocalStorage`, as
  before; outside any run it reaches the STANDALONE host, if one is installed
  (`installStandaloneHost`, kept on `globalThis` so every copy of the module sees one). The engine's host
  always wins inside a run.
- **`@declarative-ai/script`** is the standalone host and the runner. `llm()` calls the model through
  `@declarative-ai/llm`'s router, with conversations held in memory and addressed as the engine's are
  (`<stream>@<seq>`; continuing an earlier position branches). `agent()` runs a coding agent through
  `agents-cli` (Claude Code by default) and keeps Claude's null-on-failure contract. `input()` reads a line
  from stdin, `null` at end of input. `workflow()` imports the named script and calls it, binding a named
  argument object by the function's parameter names. A registered function (`$REGISTRY`) exists only where
  an engine registered it, and is refused plainly. Importing the package installs its host when none is.
- **`input(options?)`** waits for a person's next message. Durable, it is the host's registered `input`
  function (`INPUT_FUNCTION`), dispatched and recorded like any call, so a replay hands back the message
  rather than asking again; a host that registered none fails it plainly. It resolves the message
  (`{ text, files?, settings? }`), `{ idle: true }` once `idle` elapses, or `null` when the run is ended.
  In `"calls"` mode it is a cut of its own.
- **`llm({ …, withSession: true })`** resolves `{ value, session }`: the answer, and the position the call
  ENDED at, which the next call continues from (`session: turn.session`) or forks. `T` is still the value's
  contract — the compiler reads it off `value` — so the model is asked for `T`, never for the pair. Where
  the options can be read, the call is still a PROMPT STATE: its `session` may be computed (a typed input
  the field reads as `{ $expr }`), and its result is bound as
  `{ value: …, session: { id: .operation[0].usage.sessionRef } }` — the position the session layer reports
  on the call's metrics, whatever the answer's shape. So a conversation written as a loop is journaled
  turn by turn, as an authored one is. `withSession` must be written `true` or `false`.
- **A failed `withSession` call hands its error the position too** — `error.session`, where its
  conversation ended — when it wrote a record (a call stopped or failed mid-reply did; one refused before
  it was made did not). The failed call is a record: what it was asked, and what it said before it
  stopped. A `catch` that continues from `error.session` goes on AFTER it, so the words that call was
  given stay in the conversation; one that continues from the last good position replaces it. In a prompt
  state the `catch` reads it off the failed call's node (`.operation[0].usage.sessionRef`); a call made as
  code records it on `script.call.settled.session`, so a replay hands back the same; the standalone host
  appends the asked turn and hands its position.
- **A computed `null` session is a fresh conversation** (`sessionFromExpr`), as a written `null` is: a
  program holds `conv = null` until its first call hands it a position. A computed `undefined` is still a
  wiring error.
- **A computed knob that comes to `undefined` is unset.** `llm({ config: { model: chosen.model } })`
  runs on the chosen model when there is one, and when there is none the key is left out of the call
  (`writeField`) — so the defaults answer it, as they would a call that never named one — rather than
  failing the knob's type. A program hands each message's chosen settings through this way.

A conversation, then, is ordinary code:

```ts
import { llm, input } from "@declarative-ai/hw/script";
export default async function chat(first: string) {
  let text: string | null = first;
  let conv: { id: string } | null = null;
  while (text !== null) {
    try {
      const turn = await llm({ prompt: text, session: conv, withSession: true });
      conv = turn.session;
    } catch (stopped) {
      // Stopped or failed: go on after what was said, and wait for the next message.
      conv = (stopped as { session?: { id: string } }).session ?? conv;
    }
    const next = await input();
    text = next !== null && next.idle !== true ? next.text : null;
  }
}
```

### 17.1 A running program takes a newer version at a call

**Status: BUILT** (2026-11-18). A run keeps the workflow it started with — except at the boundary a
script's CALL of another workflow is. There the host may offer a newer version of the called state
(`EngineConfig.versions`, asked with the running version's hash), and the call enters it instead:

- **Same inputs, or not at all.** The new version is taken only when every input the call hands it is one
  it declares and every one it requires is handed. Otherwise the call runs what it was running, and
  `version.declined` says why.
- **Side by side, under a versioned id.** The new version's states join the run's bundle as
  `<state>@v<hash>` (and `…@v<hash>/call_0` below it) — the run's own copy of the bundle, never the host's
  — so what ran under the old version replays as it ran. The hash is the subtree's content with its own
  id taken out (`versionHashOf`), computed for every called state before the run starts (a running engine
  annotates the states it runs), so the same content is never "newer".
- **Journaled once.** `version.adopted { stateId, version, snapshot? }`, then `instance.entered` naming the
  versioned id. A replay lays the version back in first (`withVersions`, from the host's snapshot) and
  re-attaches what was entered.
- **Non-durable**, the boundary is `workflow()`: the standalone host imports the called script by its
  content (`?v=<hash>`), so an unchanged file is the module already loaded and an edited one is a fresh
  import. An ES `import` of a workflow is bound when the program starts, as any import is.
- **A call of an imported script is positional.** `turn(text, conv)` binds the script's parameters in
  order; one object of names is still how a JSON state, or a script called that way, is handed its inputs.

Two things make a program resumable at all, version or none:

- **A segment's value is journaled.** A compiled script's code (`$script`) is run by the engine itself,
  so no executor records what it returned — the continuation the states after it read. Its
  `operation.completed` carries it (`value`), and a host loading a run from its records reads it there.
- **A call the run was stopped under is asked again.** A script call settled `canceled` or `interrupted`
  (the process or the run went away while it waited — a program parked at `input()` when the app closed)
  was never answered, so a replay does not hand it back as a failure (`ScriptRunState`).

### 17.2 A running workflow takes a newer version of itself at any state boundary

**Status: BUILT** (2026-11-18). The person's ruling: a run may switch "at any state boundary given that
the state boundary has the same inputs in the new version as the old one", and `input()` "should always
be a state boundary". So the boundary is not only a call: every child the run's ROOT is about to enter.

- **The same boundary in both versions.** The host offers a newer root (`versions`, asked about the root
  with its hash). It is taken at the child being entered when the new root has that child, its state takes
  the same inputs — by name and by type; for a compiled script's state, the variables live there — and,
  for a compiled call, makes the same call (its label); and when the new root takes the run's own inputs.
  An `input()` is its key, whatever it now asks. Otherwise the run goes on, and `version.declined` names
  the boundary and why — once per version and boundary.
- **A child of the old version.** The root mounts the new one as its `$version` child (`withVersionMount`,
  each input it requires wired from the root's own), entered with the root's inputs, its way down
  directed straight at the boundary with the inputs that child was about to be handed (`by: "version"`).
  The old root takes no rule of its own again: it ends with `$version`'s outcome. A version that took the
  run hands it on the same way, one level deeper — `version.adopted { at, from }` says where, and which
  version handed off. A replay mounts it again (`AdoptedVersion.root`).
- **An `input()`'s key is stable**: `input`, `input_2`, … in the order the code waits, not a call number —
  so it is the same boundary in a version that adds a call before it.
- **From the wait itself.** A program spends its life waiting at `input()`, so an update nearly always
  lands PAST the boundary it stands at. `DirectedTransitions.takeNewerVersion()` asks the run to switch from
  there: when the root is idle, waiting on its children, the one thing running is an `input()`, and a
  newer version shares that boundary, the wait — which has done nothing — is stepped past as a directed
  skip steps past a state (recorded `skipped`, journaled before it is interrupted), and the new version
  asks again. Made beside the root's loop, never by waking it — a wake that changes nothing is a round
  that answers nothing. It resolves whether it switched; a host asks when a message arrives, and answers
  whichever `input()` then waits.

Not built: a boundary below the root — entering a version's tree in its middle needs its ancestors'
instances too. The root's children, and a call's, are the boundaries.

### 17.3 A plain program reports to an engine that will listen

**Status: BUILT** (2026-11-19, for JaiRA decision 0029, ruling 12: "A non-durable run reports to JaiRA
when it can reach it"). A script run as a plain program — `dai-script my-workflow.ts`, or `node` with
`@declarative-ai/script` imported — tells an engine that takes reports what it does, as it does it, so the
run is visible and gated there. It is still a plain program: it makes its own calls, with the keys of its
own environment, reads `input()` from its own terminal, and is gone when its process is. Nothing about it
can be resumed. With no engine listening, nothing changes.

- **Finding one.** An engine that takes reports advertises itself (`advertiseEndpoint`, from
  `@declarative-ai/script/report`, which installs no host): a JSON file `{ name, socket, token, protocol,
  pid }` in `$DAI_ENDPOINTS`, else this user's directory in the system's temporary one —
  `/tmp/declarative-ai-endpoints-<uid>` on macOS and Linux, `%TEMP%\declarative-ai-endpoints` on Windows —
  written when it starts listening, written again while it listens (so the system's cleaning of its
  temporary directory never takes it for long) and removed when it stops. An advert lasts as long as its
  process, so nothing of it is kept in the person's home. `/tmp` is the same for a terminal, ssh, cron and
  the desktop, where `$TMPDIR` and `$XDG_RUNTIME_DIR` are not; it is shared, so a directory that is not
  this user's alone (owned by them, mode `0700`) is neither read nor written. One whose `pid` is gone is
  passed over (`findEndpoints`).
  `$DAI_SCRIPT_ENDPOINT` names one directly — the advert, or a file holding it — and
  `DAI_SCRIPT_REPORT=0` turns reporting off. `runScript` looks before it calls the script (its `meta`
  goes with the hello); the host `import "@declarative-ai/script"` installs looks at the run's first call
  and ends the report when the process is about to exit (`createDiscoveringHost`).
- **The wire** (`report.ts`, protocol 1) is newline-delimited JSON on the advertised socket — a Unix
  socket, or a named pipe on Windows. A request `{ id, method, params }` is answered `{ id, result }` or
  `{ id, error: { message } }`; a notification has no `id`.

  | method | kind | params → result |
  | --- | --- | --- |
  | `hello` | request | `{ token, protocol, run: { file, args, cwd, pid, meta? } }` → `{ run, name, version?, url? }` |
  | `call.started` | notification | `{ call: { id, kind: "llm" \| "agent", model?, system?, prompt, session, schema?, cwd? } }` |
  | `call.settled` | request | `{ id, outcome: { value? \| error?, session?, messages?, costUsd?, providerSession? } }` → `{}` |
  | `approve` | request | `{ id, toolName, input }` → `{ allow, reason? }` |
  | `note` | notification | `{ kind: "phase" \| "log" \| "input" \| "workflow", … }` |
  | `end` | request | `{ value }` or `{ error }` → `{}` |

  A session is named as the program holds it (`s1@2`); the endpoint keeps its own and maps between them.
  A connection that closes before `end` is a run that ended without saying how.
- **Gated there.** A reported `agent()` hands the agent a permission callback that asks the endpoint about
  each tool use, by the agent's own tool name; what the endpoint cannot answer is refused, never allowed.
  A report that cannot be delivered is lost, never the run's failure.

## 18. How a call is drawn, and reaching a call in flight

**Status: BUILT** (2026-11-22, for JaiRA decision 0033: "one view of a run").

### 18.1 `renderUI`

```ts
import { llm, renderUI } from "@declarative-ai/hw/script";

const reply = await renderUI(llm({ prompt: text, session: conv, withSession: true }), { title: "Reply" });
const check = renderUI(async (text: string) => { … }, { role: "aside", title: "Checking for a correction" });
```

`renderUI(target, config)` says how a call, or every call a function makes, is drawn by whatever shows
the run. It changes nothing about how anything runs: it returns its argument, and the engine never reads
the config.

- **A call.** `renderUI(<call>, config)`, `renderUI(await <call>, config)` or `renderUI(<promise of a
  call>, config)`: in `"calls"` mode the compiler ties the config to that call site, and writes it on
  the call's state as `present`, beside `label`. The wrapper itself is dropped from the lowered code.
- **A function.** `renderUI(fn, config)` is `fn`, configured: every call inlined from it, through however
  many helpers, takes the config as its default. Bound to a `const` (`const check = renderUI(…)`) or
  called where it is written (`renderUI(async () => …, config)()`).
- **Scopes nest.** A call takes every scope around it, outermost first, the nearest naming a field
  winning; their `renderers` add up, the inner scope's after the outer's. A `label` in a scope names the
  step, as the call's own `label` does.
- **The config is a literal** the compiler can read (a constant will do), but for `interrupt`; anything
  else computed is a compile error. A first argument the compiler cannot tie to a call site — a variable holding an earlier promise —
  is a warning, and draws nothing differently.
- **Its fields** are open, read by whatever shows the run: `role` (a string — `aside`, `hidden`, or one the
  viewer's registry knows), `title`, `label`, `renderer` (an id, `{ id, options }`, or `null` for nothing)
  and `renderers` (`[function, role, renderer]` rows, `"*"` for either).
- **`interrupt`** — an `interruptSignal()` id, and so the one field that may be computed — registers every
  `llm()` and `agent()` call inside the scope under it (§18.2), unless a call names one of its own. It is no
  part of the config written on a state. A call that runs as code (its options unreadable) is not reached,
  and the compiler warns.
- **In the coarser modes** (`"phases"`, `"state"`) a call is not a state of its own. There
  `renderUI(promise, config)` tells the host which call the promise is, and the config is recorded on the
  call's `script.call.settled` as `present`; each `operation.dispatched` a script's code makes names its
  `site`, so a reader pairs the call's record with what was said of it. `renderUI(await llm(…))` hands the
  host a value, not a call, and draws nothing differently; a scope's `interrupt` does not reach these calls.

### 18.2 `interruptSignal`

```ts
const s = interruptSignal();
const a = await llm({ prompt: "…", interrupt: s });
```

`interruptSignal()` returns a fresh id, recorded like `now()`, so a replay hands back the same one. A call
made with `interrupt: s` (or a list of ids) is registered under it while it runs, as well as under its own
state; whatever runs the engine can then interrupt every call an id reaches. The id reaches the executor
as the knob `config.interrupt`, which the host reads and takes out before a model sees it. An interrupted
call fails as `canceled`, its reason `interrupted: <what it was interrupted with>`: the script may catch
it, and a run that does not ends stopped rather than failed. `Interrupted` is exported for scripts that
throw one of their own.
