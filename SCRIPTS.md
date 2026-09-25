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
  The rule's `inputs` hand the target its live variables.
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
  (`EngineConfig.scripts.approved`). A `type`-only import runs nothing. Its effect is the schema it
  produced, which is in the compiled documents and their hash.

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
