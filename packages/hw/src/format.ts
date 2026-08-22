/**
 * The hierarchical-workflow state-file format (SPEC §5). Declarative JSON; one file per
 * state; state ID = file path relative to the workflow root, without suffix (SPEC §2.4).
 *
 * Since the ops redesign (DESIGN §3.1) a state's operation IS an
 * `Operation<InlineFamily>` and its wiring IS `Parameter` bindings — the `runtime`/`function`
 * blocks and the `WiringValue` expression strings are gone. What an AUTHOR writes is sugar
 * (§2.1): `{ child, output }`, `{ input }`, `{ expr }`, `{ artifact }`,
 * `{ conversation }`. The loader desugars every one to a base `Ref<InlineFamily>` case — a
 * literal or a producer edge — so the checker, hasher, and engine only ever see base cases.
 * The expression DSL survives only where control flow needs it: transition guards, `limits`,
 * and the `{ expr }` binding leaf (which is itself a producer edge on the evaluator function).
 */
import type {
  FunctionOp,
  InlineFamily,
  JsonSchema,
  JsonValue,
  NamedParameter,
  Operation,
  Parameter,
  PromptOp,
  Ref,
  RefKind,
} from "@declarative-ai/exec";
import type { PermissionMode, PermissionProfile, ScopeDecl } from "@declarative-ai/permissions";
import { BUILTINS } from "./builtins.js";
import { OPERATION_ENGINE_OUTPUT, OPERATION_METADATA_FIELDS } from "./operationNode.js";

// The op vocabulary is hw's format vocabulary — re-exported so authors and consumers import
// one set of names.
export type { FunctionOp, InlineFamily, NamedParameter, Operation, Parameter, PromptOp, Ref, RefKind };

/** Termination outcomes (SPEC §3.6) — how a state finished, not what it decided. */
export type TerminationOutcome = "success" | "error" | "canceled" | "timeout";

export const TERMINATE_TARGETS = [
  "terminate.success",
  "terminate.error",
  "terminate.canceled",
  "terminate.timeout",
] as const;

/** State run statuses (SPEC §10.1). */
export type RunStatus =
  | "queued"
  | "running"
  | "waiting_for_user"
  | "waiting_for_event"
  | "sleeping"
  | "blocked"
  | "failed"
  | "completed"
  | "canceled";

// --- Well-known resolver functions (the desugaring targets, §2.1) -------------

/**
 * The registered functions authored binding sugar desugars ONTO. Each is an ordinary
 * `FunctionOp` producer — that is the whole point: after desugaring there is no special
 * wiring case left for the checker or the engine to know about, only producer edges.
 */
export const RESOLVER_REFS = {
  /** Project one property off a producer's object output (`{ child, output }` lowering). */
  select: "select",
  /** Read a declared `inputs.*` value by name (the model's by-name free-slot fill). */
  scope: "scope.get",
  /** Read a session-owned artifact by name. */
  artifact: "artifact.get",
  /** Read a session's transcript, or one message of it. */
  conversation: "conversation.get",

  // --- Operators (EXPRESSIONS.md §2) ------------------------------------------
  //
  // An expression is a tree of producer edges over THESE, rather than a source string handed to an
  // interpreter. That is what stops `{ expr }` being the one construct needing its own static
  // analysis: a dependency is a leaf of the tree, which the fan-out planner and the validator
  // already walk. The set is a registry rather than a grammar, so a user-defined pure function is
  // indistinguishable from `eq`.
  /** One root of the expression context by name (`inputs`, `children`, `run`, …). */
  context: "context.get",
  /** Property access, with implicit optional chaining — distinct from `select`, which REFUSES a
   *  missing key because a named child output that is not there is an authoring error. */
  member: "op.member",
  not: "op.not",
  eq: "op.eq",
  ne: "op.ne",
  strictEq: "op.strictEq",
  strictNe: "op.strictNe",
  lt: "op.lt",
  le: "op.le",
  gt: "op.gt",
  ge: "op.ge",
  /** The three LAZY forms. They resolve their arguments on demand, which is what preserves
   *  `false && PENDING === false` and keeps an untaken branch from running (EXPRESSIONS.md §6). */
  and: "op.and",
  or: "op.or",
  cond: "op.cond",
  /**
   * An OBJECT LITERAL — `{ to_state: 'deploy' }`.
   *
   * The one resolver whose parameter names are the author's rather than the language's: an object
   * literal's keys ARE the input slots, so there is no positional signature to write down and
   * `OPERATOR_PARAMS` has no entry for it. That is what makes it the natural target — a producer
   * edge's `input` is already a record of named parameters, so a record literal lowers onto the
   * shape it already has instead of needing a variadic call convention the model does not have.
   */
  record: "op.record",
} as const;

/** Every well-known resolver ref, for registry seeding and validator checks. */
export const RESOLVER_REF_VALUES: readonly string[] = Object.values(RESOLVER_REFS);

/**
 * Every name the engine computes INLINE — the resolvers above plus the built-in operation library.
 *
 * Membership decides two things: the validator does not look for a registry entry, and
 * `resolveProducer` runs it in-place rather than treating it as an embedded call to dispatch.
 */
export const RESOLVER_REF_SET: ReadonlySet<string> = new Set([...RESOLVER_REF_VALUES, ...Object.keys(BUILTINS), "map", "filter", "flatMap", "reduce"]);

// --- Authored binding sugar (§2.1) -------------------------------------------

/**
 * What an author may write in a `Parameter.binding` slot. The first five cases ARE the base
 * `Ref<InlineFamily>` union (literals, an existing result, a ref tree, a producer edge); the
 * rest are sugar the loader lowers onto producer edges over {@link RESOLVER_REFS}.
 *
 * `{ result }` is a reference to an ALREADY-EXISTING `GenerationResult` — a different concept
 * from `{ child, output }`, which is a producer EDGE the engine may still have to run.
 */
export type BindingDecl =
  | Ref<InlineFamily>
  /**
   * A RUNTIME reference (REFERENCES.md §5) — a leading-dot path into this instance's data:
   * `.children.critique.outputs.outcome`, `.inputs.issue`, `.artifacts.design_doc`.
   *
   * Always current-instance: a runtime reference into another file would have no instance to
   * resolve against, since a state can run many times. Cross-file references are transclusion,
   * resolved before anything runs.
   *
   * Also how an EXPRESSION is written without a wrapper — a string binding is parsed with the
   * expression grammar, and the leading dot is what separates a read of this instance's data
   * from a bare name resolved along the search `path` (EXPRESSIONS.md §14).
   */
  | string
  /** A small computation in the expression DSL. Lowers to a TREE of operator producer edges
   *  (EXPRESSIONS.md §1) — parsed once, at load, never carried as a source string. The bare
   *  string above says the same thing; this spelling is emphasis, for a value a reader would
   *  otherwise have to squint at to see is computed. */
  | { expr: string };

/** Every key that tags an authored binding form — the base `Ref` cases plus the sugar. */
const BINDING_TAGS: readonly string[] = ["text", "json", "result", "refs", "op", "expr"];

/**
 * True when a value is spelled as a BINDING rather than as data.
 *
 * A reference in a binding position can point at anything, and what it points at decides how it
 * reads: a binding form is a binding, an operation document is an operation, and anything else is
 * data. This is the test expansion applies to what it spliced — which is the shape-mismatch rule of
 * REFERENCES.md §3 turned on the target instead of the position.
 *
 * An operation document counts, because it lowers to the `{ op }` producer edge that §3.1 calls the
 * higher-order value of a named operation.
 */
export function isBindingDecl(value: unknown): boolean {
  if (typeof value === "string") return true;
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const o = value as Record<string, unknown>;
  return BINDING_TAGS.some((k) => k in o) || o.kind === "prompt" || o.kind === "function";
}

/**
 * How a RESOLVED document reads in binding position — the one definition of that dispatch.
 *
 * Two callers reach it from opposite directions and must agree: expansion, splicing a path written
 * as a whole binding, and lowering, resolving a bare name inside an expression. `"binding": "lib/x"`
 * and `"binding": "id(lib/x)"` have to see the same `lib/x`, so the rule lives in one place.
 *
 * `fromDataFile` is what separates a `.md`'s TEXT from a JSON string: both are strings in hand, and
 * only the file type says whether the author wrote prose or a binding.
 */
export function bindingForDocument(value: unknown, fromDataFile: boolean): BindingDecl {
  if (typeof value === "string" && !fromDataFile) return { text: value };
  return isBindingDecl(value) ? (value as BindingDecl) : { json: value as JsonValue };
}

/** A `Parameter` as AUTHORED: the binding may still be sugar. */
export interface ParameterDecl {
  kind?: RefKind;
  schema?: JsonSchema;
  binding?: BindingDecl;
  index?: number;
  /** Authoring convenience for a FREE slot: a value used when nothing is wired in. Also the
   *  explicit opt-out from the §7.2 reachability rule. */
  default?: JsonValue;
  /** SPEC §4.1: slots are required by default. */
  optional?: boolean;
  description?: string;
}

/**
 * The order slots bind POSITIONAL arguments in: by declared `index`, else declaration order.
 *
 * `index` is the model's own "positional sort key for bare/tuple ingestion", so an author who wants
 * to be called positionally says so there; a document declaring none still has an order, and using
 * it is friendlier than refusing.
 *
 * Shared because two things ask this question and must not drift: a CALL binds its arguments to the
 * callee's slots (`lowerExpr`), and an embedded body's synthetic wrapper declares its parameters in
 * the same order (`functionBody`). A caller and its callee disagreeing about argument order is the
 * kind of bug that type-checks.
 */
export function positionalOrder(input: Readonly<Record<string, { index?: number }>>): string[] {
  const entries = Object.entries(input);
  const indexed = entries.filter(([, p]) => p.index !== undefined);
  if (indexed.length === 0) return entries.map(([name]) => name);
  return indexed.sort(([, a], [, b]) => (a.index ?? 0) - (b.index ?? 0)).map(([name]) => name);
}

/** A standalone (named) slot as authored — a state's output. */
export interface NamedParameterDecl extends ParameterDecl {
  name?: string;
}

// --- Authored operations (§7.1) ----------------------------------------------

/** Conversation context modes (SPEC §4.7). */
export type ConversationMode = "full_history" | "summary" | "fresh" | "selected_artifacts";

/**
 * The EXECUTION-ENVIRONMENT fields of an operation: session, tools, conversation preamble, and the
 * authored permission baseline (DESIGN §5.1).
 *
 * These used to be a sibling `environment` block, on the reasoning that "how it runs" is not part of
 * the op's identity. They are fields of the operation now, because every one of them is a per-CALL
 * decision — which session the call joins, which tools it may reach mid-loop, how much transcript it
 * carries — and separating them bought nothing except a second place to look. The name `environment`
 * was then free for what authors actually kept asking for: DEFAULTS (see {@link EnvironmentDecl}).
 */
export interface ExecEnvironmentDecl {
  /**
   * The conversation this operation runs under (DESIGN.md §1.6). Three spellings:
   *
   *  - a **name** — same name across states ⇒ one shared stream, and the name is also the
   *    resource-bundle key (workspace, permissions);
   *  - a **session ref** (`{ id }`) — an exact position to continue or branch from. Opaque: nothing
   *    outside the session store parses it. Normally reached through the fourth spelling rather than
   *    written literally, since a ref is a run-time value;
   *  - an **expression** (`{ expr }`, e.g. `{"expr": ".children.plan.operation.output.session"}`) —
   *    the same thing computed per instance. This is the one form EVALUATED rather than read, and
   *    the only practical way to name an exact position, because `operation.output.session` does
   *    not exist until that operation has run;
   *  - **`null`** — start a fresh stream, overriding whatever the environment chain supplied.
   *
   * ABSENT no longer means a shared default. An undeclared operation gets its own stream, because
   * an implicit process-wide transcript is the thing that drives unbounded context growth; the
   * run's shared WORKSPACE is unaffected, being a separate concern (see {@link fork}'s neighbours
   * in `session.ts`). `""` is an error, never "fresh" — a template interpolating a bad reference
   * would otherwise silently produce an isolated conversation that looks like it worked.
   *
   * ONE spelling. `sessionId` used to be accepted as a synonym so an `LlmConfiguration`-shaped
   * block could paste in unchanged; it is refused now, because a second name for a field costs an
   * equality check at parse, a normalization that has to run before the merge, and a rule about
   * which one wins — all to save an author one rename.
   */
  session?: string | null | { id: string } | { expr: string };

  /**
   * Always branch, rather than appending when the position is still the head (DESIGN.md §1.6).
   *
   * Declared where a session is CONSUMED, not carried on the value produced: a position marker
   * should not encode an intent about how a later caller will use it. Absent and `false` mean the
   * SAME thing — append if the position is still the head, fork automatically if it is not — because
   * the stricter reading ("append, or fail") would be an exclusivity claim on the position, and
   * holding that claim is state the prompt executor must not carry. `true` stays useful because
   * deliberate divergence (fan three variants out of one point) cannot be inferred from stream state.
   */
  fork?: boolean;
  /** Logical names of tools the operation may call mid-loop — resolved through `registry.tools`. */
  tools?: string[];
  /** Conversation preamble injected into THIS call (distinct from a `{ conversation }` wire, which
   *  reads a transcript as data, §7.5). */
  conversation?: {
    mode: ConversationMode;
    /** For `selected_artifacts`: names of artifacts to inject. */
    artifacts?: string[];
  };
  /** Authored per-operation permission baseline (DESIGN §5.1, "the definition-authored baseline"). */
  permissions?: {
    profile?: PermissionProfile;
    default?: PermissionMode;
    tools?: Record<string, PermissionMode>;
    /** What a tool the host does not register resolves to — see `ProfileTable.other`. */
    other?: PermissionMode;
    /**
     * WHERE each tool may act, as authored on this operation.
     *
     * Carried, never resolved: the engine hands it to the host's `ExecPolicy.scopeOf`, which owns the
     * glob grammar and decides what an unmatched path means. A state's table NARROWS whatever floor
     * the host supplies — it cannot widen past it, for the same reason it cannot widen past a profile.
     */
    scopes?: ScopeDecl[];
  };
}

/**
 * Every field an operation may carry, all optional — the shape BOTH `operation` and `environment`
 * are written in (§5). `operation` narrows it to {@link MergedOperationDecl} once the inheritance
 * chain has been merged; `environment` never does, because a defaults layer is partial by nature.
 */
export interface OperationFields extends ExecEnvironmentDecl {
  kind?: "prompt" | "function";
  /**
   * The prompt, as text. `{{.inputs.x}}` interpolation applies.
   *
   * A reusable prompt is a REFERENCE to a file — `{"$ref": "$/prompts/review.md"}` — which is what
   * replaced the old `prompt.skill` and `registry.skills` (REFERENCES.md §7.1). A referenced `.md`
   * is still a template; nothing about interpolation changes.
   */
  prompt?: string;
  system?: string;
  /** Registry name — a host function, or a runtime adapter (`claude-code`, …). */
  function?: string;
  /**
   * A FUNCTION operation's authored arguments, bound to its input slots BY NAME.
   *
   * The shorthand for `input`: `{"args": {"mode": "plan"}}` is `{"input": {"mode": {"kind": "text",
   * "binding": {"text": "plan"}}}}`, which is what an author writes when the value is a constant and
   * there is nothing to say about its type. A slot the author declared in `input` wins, since `args`
   * merges per key down the environment chain and the typed declaration is the more specific of the
   * two statements.
   *
   * They used to arrive as one blob in a slot called `config`, because a registered function had no
   * way to declare named parameters and there were no slots to bind to. There are now (§7.5.2), and
   * an impl reads `inputs.mode`.
   *
   * Untyped by nature — only the function knows what it takes — which makes this the one position
   * where a reference must be written `{"$ref": …}` rather than as a bare string (§3.1).
   */
  args?: Record<string, JsonValue>;
  input?: Record<string, ParameterDecl>;
  /**
   * An EMBEDDED js/ts body (SPEC §7.5.1) — the middle of the three function-body forms.
   *
   * A statement list, or a single expression whose value is returned. The engine wraps it in a
   * synthetic function whose parameters are the declared `input` slots in `positionalOrder`, so an
   * embedded body and a module form compile to the same artifact and run down one path.
   *
   * It may not import: self-containment is what keeps the body INLINED into this document, and
   * therefore part of the snapshot identity (§12) rather than a separate file needing its own
   * approval. Code that needs imports is a module.
   */
  body?: string;
  /**
   * What the operation RETURNS, by name — and, for a prompt op, the structured-output contract the
   * model is held to.
   *
   * A map, like `input`, because an operation returns named values exactly as it takes them. It used
   * to be one `output` slot, and the contract was derived from the STATE's unbound outputs instead:
   * the operation borrowed its own signature from whatever the state around it happened to declare,
   * which is why the result had no address of its own and could only be received, never renamed or
   * transformed on the way through.
   *
   * These names are what `.operation.output.<name>` exposes, and what a state's outputs bind FROM.
   */
  outputs?: Record<string, NamedParameterDecl>;
  /**
   * The single lowered output slot, as authored.
   *
   * Still here because the executor seam takes ONE output — a prompt op asks the model for one
   * object, a delegated agent hands back one blob — so `outputs` is lowered INTO this. Declaring it
   * directly is how an author says something the map cannot: that the whole return value is a blob
   * (§4.4), rather than a record of named fields.
   */
  output?: NamedParameterDecl;
  /**
   * The ordered roots a BARE reference is searched along, inherited down the tree (EXPRESSIONS.md
   * §4). Shell `PATH` semantics: first match wins, and only the first entry produces bare ids.
   *
   * Two things about it are deliberate and easy to get wrong:
   *
   *  - **It is spliced, not unioned.** Arrays REPLACE everywhere else in this merge, and that rule
   *    is what makes `"tools": []` the way to drop an inherited tool. Rather than exempt this one
   *    field, a `"$INHERITED"` entry splices in what the chain supplied — the same idiom JaiRA's
   *    `artifacts.destination: "$DEFAULT"` already uses. `["./ops", "$INHERITED"]` prepends;
   *    `["./ops"]` shadows everything, built-ins included.
   *  - **Its own entries may not be bare**, or resolving the path would need the path. `$VAR`,
   *    absolute or relative only — exactly as a shell `PATH` holds directories rather than names to
   *    search for.
   *
   * It sits on `OperationFields` because that is the shape the environment chain merges, and a
   * second inheritance chain for one field would be worse. It is not an operation field in any
   * other sense — nothing dispatches on it — which is why it is in `OPERATION_OWN_FIELDS` (so it is
   * never hoisted into an LLM call) and out of `KIND_SPECIFIC` (a prompt op's guards resolve
   * references too).
   */
  path?: readonly string[];

  // --- The LlmConfiguration surface, inline on a prompt operation (REFERENCES.md §7.2) ----------
  //
  // Structurally restated rather than imported: hw does not depend on `@declarative-ai/llm`, and
  // the prompt runner is what type-checks the assembled call. Any field NOT owned by hw
  // (`OPERATION_OWN_FIELDS`) is passed through to the call config too, so a knob missing from this
  // list still reaches the model — it just is not statically known here.
  model?: string;
  maxOutputTokens?: number;
  stopSequences?: string[];
  seed?: number;
  maxSteps?: number;
  toolChoice?: JsonValue;
  providerOptions?: Record<string, JsonValue>;
  outputModalities?: string[];
  temperature?: number;
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  /** How hard to think. Mirrors `llm`'s `ReasoningSpec` by hand, as everything on this authoring
   *  surface does — `xhigh` included, because a delegated agent has such a tier and an author must be
   *  able to write it here or the level cannot be requested at all. */
  reasoning?: { effort?: "low" | "medium" | "high" | "xhigh"; budgetTokens?: number };
}

/**
 * The operation fields hw itself consumes. Everything else an author writes on a prompt operation
 * is call configuration and is hoisted into the lowered op's `config` — "the operation IS the call"
 * (REFERENCES.md §7.2).
 */
export const OPERATION_OWN_FIELDS: ReadonlySet<string> = new Set([
  "kind",
  "prompt",
  "system",
  "function",
  "args",
  "input",
  "output",
  "outputs",
  "path",
  "session",
  "sessionId",
  "fork",
  "tools",
  "conversation",
  "permissions",
]);

/**
 * A `prompt` operation: one structured LLM call, with exactly one of `prompt.template` /
 * `prompt.skill` (a skill is a named template resolved through `registry.skills` at render time).
 */
export interface PromptOpDecl extends OperationFields {
  kind: "prompt";
}

/**
 * A `function` operation: invoke a registered function (`registry.functions`) — sync or async, host
 * code or a DELEGATED AGENT ADAPTER (§3.1). A sub-workflow, a composite unit, and a `claude-code`
 * invocation are all this one shape; the resolved registry ENTRY's capabilities distinguish them,
 * never the op.
 */
export interface FunctionOpDecl extends OperationFields {
  kind: "function";
  function: string;
}

/** A state's `operation` block AS AUTHORED — partial, because an ancestor's `environment` may supply
 *  `kind`, `function`, `config`, or anything else it leaves out. `{}` is legal and means "inherit the
 *  whole operation". */
export type OperationDecl = OperationFields;

/** An operation after the environment chain is merged in: `kind` is settled and the shape is checked. */
export type MergedOperationDecl = PromptOpDecl | FunctionOpDecl;

/**
 * The DEFAULTS layer (§5): an operation shape, all fields optional, inherited by this state's
 * operation AND by every descendant's.
 *
 * A state's effective operation is the deep merge of every ancestor's `environment` (outermost
 * first), then its own `environment`, then its own `operation` — the nearest layer wins. Only a state
 * that declares an `operation` block gets one, so a pure composite under an `environment`-declaring
 * root stays a pure composite; `"operation": {}` is the explicit opt-in to a fully inherited one.
 */
export type EnvironmentDecl = OperationFields;

// --- States ------------------------------------------------------------------

export interface ChildDecl {
  /** The child's state reference. Absent ⇒ `./<key>` — the state the child's own key names. */
  state?: string;
  /** Wiring into the child's declared inputs — the same authored binding sugar (§2.1). */
  inputs?: Record<string, BindingDecl>;
  /** SPEC §10.4: starting this child does not block the sequence. */
  async?: boolean;
  /**
   * Defaults for THIS MOUNT of the child, and its subtree (§5).
   *
   * The `environment` chain was per-STATE, so every child of one parent inherited the same layer and
   * two children could not differ in it. That is fine until the difference is the point: mounting one
   * `review/agent_review` twice, once under `claude-code` and once under `codex-cli`, is a two-line
   * change with this and two near-duplicate state files without it.
   *
   * ```json
   * "children": {
   *   "claude_review": { "state": "$/review/agent_review", "async": true,
   *                      "environment": { "kind": "function", "function": "claude-code" } },
   *   "codex_review":  { "state": "$/review/agent_review", "async": true,
   *                      "environment": { "kind": "function", "function": "codex-cli" } }
   * }
   * ```
   *
   * It sits BETWEEN the parent's `environment` and the child's own, in the ordinary nearest-wins
   * order — so it is a default the child may still override, not an imposition. A state that names
   * its own `operation.function` therefore cannot be varied this way, which is correct: a state that
   * says what it runs means it. A state meant to be mounted under several runtimes leaves `function`
   * to the chain (`"operation": { "kind": "function" }`), which the loader already supports.
   *
   * Two mounts under two different environments load as two VARIANTS of the state (see `variantFor`),
   * so everything downstream — validation, snapshots, the board, events — still sees one id with one
   * operation. That machinery predates this field; per-child layers just reach it.
   */
  environment?: EnvironmentDecl;
  /**
   * Transitions considered ONLY when this child finishes, and BEFORE the state's own (SPEC §3.3).
   *
   * The DEFAULT place for a transition. The state's list ({@link StateDef.transitions}) is for the
   * rules that hold whichever child just finished, or none did.
   *
   * The state-level list answers "what does this state do next", which is the right question when the
   * answer depends on the state's whole situation and the wrong one when it depends on a single
   * child's outcome. Written there, "if the review failed, escalate" has to say WHICH review failed
   * and then stay true only while that is the freshest fact — so a list that grew a per-child branch
   * for each of five children was five guards, each of which every other child's completion also had
   * to be evaluated against.
   *
   * Here the "when" is structural: the child's completion is what makes the list eligible, and only
   * for the round that completion triggered. A child that finished two rounds ago no longer diverts
   * anything, so an unconditional `to` means "after this child, go here" rather than "from now on,
   * always go here".
   *
   * ONCE PER COMPLETION, in the first round that STARTS after the child finished — see the snapshot
   * the evaluation loop takes. Not once per child: a looped child is answered on every pass.
   *
   * Guards resolve in the ENCLOSING state's scope, exactly as the state-level ones do — this child is
   * `.children.<key>` from here, and it is spelled out. A `self` alias would be one namespace whose
   * meaning depended on where it was written, and the same guard would then mean two things depending
   * on which list it had been moved into.
   *
   * `to` may name any sibling child or a `terminate.*` outcome, and entering a sibling resets the
   * sequence exactly as a state-level transition does. Taking one HANDLES this child's failure: an
   * error or timeout routed here is dealt with, not merely reacted to.
   */
  transitions?: TransitionDecl[];
}

export interface TransitionDecl {
  /** A declared child key, or one of `terminate.*`. */
  to: string;
  /** Guard expression; absent = unconditional. Must INFER to boolean (§7.2) — strict, no
   *  truthiness coercion. */
  when?: string;
}

/**
 * A transition after loading: its guard LOWERED (EXPRESSIONS.md §1).
 *
 * Lowered at load rather than by the engine, because a guard may CALL an operation and resolving a
 * callee is load-time knowledge — it needs the search path, the referring state and a filesystem.
 * Lowering guards engine-side worked while expressions were operators only, and threw
 * `'x' is not a known operation` the moment one contained a call.
 *
 * `when` is kept alongside: it is what the validator reports against, and what a human reads.
 */
export interface LoadedTransition extends TransitionDecl {
  whenRef?: Ref<InlineFamily>;
  /**
   * Why the guard could not be lowered — carried as DATA, not thrown.
   *
   * The same rule `operationError` follows: an authoring mistake is reported by the validator
   * alongside every other one, because throwing at load aborts at the FIRST bad state and hides the
   * rest. A transition carrying this never fires; validation blocks the run anyway, and a guard that
   * failed to parse must not read as unconditional.
   */
  whenError?: string;
}

export interface LimitsDecl {
  /** Guard value exposed as `limits.max_iterations` in expressions (SPEC §3.4). */
  max_iterations?: number;
  /** State timeout in seconds → `terminate.timeout` when exceeded. */
  timeout?: number;
}

export interface StateDef {
  /** The state's PATH REFERENCE (§2.1) — equal to its own location, so it may be omitted and
   *  derived (a present-but-mismatched `id` is a load error). Bare paths hang off the default
   *  workflow root; `/…`, `$VAR/…`, `file:…`, and `./…` are the escape hatches. */
  id?: string;
  label?: string;
  description?: string;
  inputs?: Record<string, ParameterDecl>;
  outputs?: Record<string, NamedParameterDecl>;
  /** The state's operation (§7.1). A state with children and no operation is a pure composite;
   *  `{}` means "the operation my `environment` chain describes". */
  operation?: OperationDecl;
  /** Defaults for this state's operation AND every descendant's (§5). */
  environment?: EnvironmentDecl;
  children?: Record<string, ChildDecl>;
  /** Order the engine's cursor advances through `children`. Absent ⇒ declaration order (§6). */
  sequence?: string[];
  /**
   * Transitions that apply to the state AS A WHOLE — a decision made from its own operation's output,
   * an entry into a child, an iteration limit.
   *
   * A rule about one child goes on that child instead ({@link ChildDecl.transitions}), which is the
   * default. Written here it has to name the child in its guard, hold its position against every
   * other rule in the list, and be re-evaluated after every unrelated child completion; written on
   * the mount, the round it is eligible in is already the one it is about.
   */
  transitions?: TransitionDecl[];
  limits?: LimitsDecl;
}

/**
 * A state after loading: its operation desugared to base `Ref` cases, ready for the checker and the
 * engine. `operation` is a real `Operation<InlineFamily>`; the authored sugar is gone, and so is the
 * environment chain — the loader merged it in (§5), which is why `environment` here is the RESOLVED
 * execution environment rather than the partial defaults layer the author wrote.
 */
export interface LoadedState
  extends Omit<StateDef, "operation" | "environment" | "inputs" | "outputs" | "children" | "sequence" | "transitions"> {
  /** Transitions with their guards lowered (§1). */
  transitions?: LoadedTransition[];
  id: string;
  inputs?: Record<string, Parameter<InlineFamily>>;
  outputs?: Record<string, NamedParameter<InlineFamily>>;
  operation?: Operation<InlineFamily>;
  /** The merged execution environment this state's operation runs under. */
  environment?: ExecEnvironmentDecl;
  /**
   * The session this state's SUBTREE resolves in — its own `environment.session` merged over what the
   * chain supplied, whether or not this state declares an operation.
   *
   * Distinct from `environment.session`, which is the merged view for this state's OWN operation and
   * is therefore absent on a pure composite. That gap matters: declaring `environment.session` on a
   * composite root is the ordinary way to give a whole subtree one session, and the engine needs to
   * see it there to key the subtree's resource bundle — workspace, permission ledger, approval scope
   * — on the name the author actually wrote (DESIGN.md §1.6).
   *
   * Present-but-`null` is meaningful: it is an explicit "start fresh", not an absent declaration.
   */
  scopeSession?: string | null | { id: string } | { expr: string };
  children?: Record<string, LoadedChild>;
  /**
   * Why this state's `operation` could not be built — an incomplete merge (§5), reported by the
   * validator rather than thrown, so one broken state does not hide every other authoring error.
   * A state carrying this always has NO `operation`; the engine refuses to run it.
   */
  operationError?: string;
  /** Unexpanded `prefix*` outputs, pending the child's slots — expanded by `loadBundle` (§3.4). */
  outputSpreads?: OutputSpread[];
  /** Always present when the state has children: the authored order, or declaration order (§6). */
  sequence?: string[];
  /** Whether `sequence` was written in the file rather than derived — the lint surface treats an
   *  authored order as a claim a transition can contradict, and a derived one as no claim at all. */
  sequenceAuthored?: boolean;
  /** Per-slot authoring metadata the op model doesn't carry (defaults, optionality, docs),
   *  keyed `"<section>.<name>"` — read by the engine when filling free slots. */
  slotMeta?: Record<string, SlotMeta>;
  /**
   * The declared-child outputs this state's wiring FANS OUT — a producer output referenced by two or
   * more consumers (§7.3, rule 2). Computed statically at load time, because fan-out is a property of
   * the DOCUMENT (the validator/loader can already see every consumer of a producer), not something to
   * discover when a second reader shows up at run time. Keyed `"<childKey>\0<output>"` for a specific
   * output and `"<childKey>\0*"` when the whole child is read enough times that every output fans out.
   * The engine drains a matching blob output ONCE, when the producer child completes, so both consumers
   * receive the bytes instead of racing to read one stream. Absent ⇒ no fan-out.
   *
   * A sorted ARRAY rather than a `Set`, because a `LoadedState` is a resolved definition and a
   * definition has to be plain JSON: `JSON.stringify` renders a `Set` as `{}`, dropping every entry
   * without an error.
   */
  fanOut?: readonly string[];
}

/** One `prefix*` output: republish every output of `child`, prefixed, as this state's own (§3.4). */
export interface OutputSpread {
  prefix: string;
  child: string;
  optional?: boolean;
}

/** Authoring metadata for one declared slot, kept alongside (never inside) the op. */
export interface SlotMeta {
  default?: JsonValue;
  optional?: boolean;
  description?: string;
}

export interface LoadedChild {
  state: string;
  /** Desugared wiring into the child's declared inputs. */
  inputs?: Record<string, Ref<InlineFamily>>;
  async?: boolean;
  /** The per-mount defaults this child was declared with, carried through so the closure walk can
   *  fold them into the chain (and so a lint surface can see why a state loaded as two variants). */
  environment?: EnvironmentDecl;
  /** This mount's own transitions, guards lowered — considered when this child finishes, ahead of the
   *  state's list (`ChildDecl.transitions`). */
  transitions?: LoadedTransition[];
}

/** A loaded workflow: the root state ID plus every reachable state, keyed by state ID. */
export interface WorkflowBundle {
  rootId: string;
  states: Record<string, LoadedState>;
  /** The states AS AUTHORED (pre-desugaring), kept because the snapshot hash is the identity of
   *  what the author wrote — so improving the lowering never invalidates a stored snapshot. */
  source?: Record<string, StateDef>;
  /**
   * One value standing for every js/ts MODULE this workflow reaches — `FrozenModules.digest`
   * (SPEC §7.5.5).
   *
   * `snapshotHash` hashes the RESOLVED states on the stated grounds that "what was referenced is
   * inlined, and a change to what anything lowers to is a different hash by construction". A module
   * reached by name breaks that premise: it is not inlined, so without this a task pinned to a
   * snapshot would run edited code under an unchanged version — the exact failure resolved-form
   * hashing exists to prevent.
   *
   * Absent for a bundle that reaches no module, and absent is not the same as empty: the key is left
   * out of the hashed document entirely, so every snapshot taken before modules existed keeps its
   * identity.
   */
  moduleDigest?: string;
}

/**
 * Expression-context namespaces, split by ROLE after the rewrite (§7.5):
 *  - REF vocabulary — the data namespaces authored bindings point into. They are reachable from
 *    `{ expr }` leaves too, since an expr leaf IS a producer over the same data.
 *  - GUARD-ONLY scalars — control-flow state (`run`, `limits`), never a reference binding.
 *
 * `operation` is the state's OWN call as an addressable node (SPEC.md §6.1) — its outcome, what it
 * cost, which model served it, and for a prompt op the conversation position it ended at. It is a
 * namespace rather than a child on purpose: a child would perturb the instance tree, which is what
 * `run.cursor`, `run.position` and `sequence` are defined against.
 * The old `function.*` namespace is GONE: a function state's result is an ordinary state output,
 * so guards read `outputs.*` / `children.<key>.outputs.*` uniformly.
 */
export const REF_NAMESPACES = ["inputs", "outputs", "operation", "children", "artifacts"] as const;
export const GUARD_NAMESPACES = ["run", "limits"] as const;
export const CONTEXT_NAMESPACES = [...REF_NAMESPACES, ...GUARD_NAMESPACES] as const;

/**
 * What a read of a runtime path CONSUMES from a producing child — `undefined` for a read of nothing
 * that could be a stream, and an `output` of {@link WHOLE_CHILD} for one that reaches every output.
 *
 * This lives here, beside the namespace vocabulary it interprets, because the question is "what does
 * this SHAPE mean" and the shape is defined right above. It used to be answered inline in
 * `fanout.ts` by indexing path positions — `path[2] === "outputs"`, `path[2] !== "outcome"` — which
 * made the tally a third place hard-coding the `children` namespace, alongside the validator's
 * `childrenProps` and the engine's `exprContext`. Two of those three fail SILENTLY when the
 * namespace grows a segment, because a miscounted consumer is a wrong number and not an error.
 */
export type Consumption = { child: string; output: string } | undefined;

/** Marks a read that consumes EVERY output of a child. */
export const WHOLE_CHILD = "*";

/**
 * Decide what one lowered path consumes.
 *
 * The distinctions are load-bearing (see the `fanout.ts` header): a specific output is one blob
 * consumed, a coarser read consumes every output, and a read of how the child WENT consumes none.
 * Counting that last kind forces a materialization nothing needs, and §7.4 wants a single-consumer
 * output left a live stream so it can be piped.
 */
export function consumptionOf(path: readonly string[]): Consumption {
  if (path[0] !== "children" || path[1] === undefined) return undefined;
  const child = path[1];
  const whole: Consumption = { child, output: WHOLE_CHILD };
  const [section, ...tail] = path.slice(2);

  if (section === undefined) return whole; // `.children.k` — the child itself, so all of it
  if (section === "outcome") return undefined; // the termination STATUS, not an output
  if (section === "outputs") return tail[0] !== undefined ? { child, output: tail[0] } : whole;
  if (section !== "operation") return whole;

  const field = tail[0];
  // The node itself carries what the call returned, so a bare read reaches it.
  if (field === undefined) return whole;
  // How the call WENT, not what it produced.
  if (OPERATION_METADATA_FIELDS.has(field)) return undefined;
  if (field !== "output") return whole;
  // Engine-written and opaque: a session ref is never bytes.
  if (tail[1] === OPERATION_ENGINE_OUTPUT) return undefined;
  // Otherwise this IS the returned value. WHICH declared slot it feeds is not knowable from the
  // referring state alone — a state may publish it under another name — so it counts as all of them.
  // Over-materializing is the safe direction: under-counting races two readers on one stream,
  // silently.
  return whole;
}
