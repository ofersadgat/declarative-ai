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
import type { PermissionMode, PermissionProfile } from "@declarative-ai/permissions";

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
  /** Evaluate a DSL expression against the run context; output schema = the inferred type (§7.2). */
  expr: "expr.eval",
  /** Project one property off a producer's object output (`{ child, output }` lowering). */
  select: "select",
  /** Read a declared `inputs.*` value by name (the model's by-name free-slot fill). */
  scope: "scope.get",
  /** Read a session-owned artifact by name. */
  artifact: "artifact.get",
  /** Read a session's transcript, or one message of it. */
  conversation: "conversation.get",
} as const;

/** Every well-known resolver ref, for registry seeding and validator checks. */
export const RESOLVER_REF_VALUES: readonly string[] = Object.values(RESOLVER_REFS);

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
  /** A declared child's output. Lowers to a producer edge on the child + a `select` projection. */
  | { child: string; output?: string }
  /** This state's declared input, by name. Lowers to a `scope.get` producer. */
  | { input: string }
  /** A small computation in the expression DSL. Lowers to an `expr.eval` producer (§7.2). */
  | { expr: string }
  /** A session-owned artifact, by name. Lowers to an `artifact.get` producer. */
  | { artifact: string }
  /** A previous conversation, or one message of it. Lowers to a `conversation.get` producer. */
  | { conversation: string; message?: number };

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

/** A standalone (named) slot as authored — a state's output. */
export interface NamedParameterDecl extends ParameterDecl {
  name?: string;
}

// --- Authored operations (§7.1) ----------------------------------------------

/** Conversation context modes (SPEC §4.7). */
export type ConversationMode = "full_history" | "summary" | "fresh" | "selected_artifacts";

/**
 * A `prompt` operation as authored: one structured LLM call. `prompt.template` / `prompt.skill`
 * are the authored forms of the `PromptOp.user` slot (a skill is a named template resolved
 * through `registry.skills` at render time) — exactly one of the two.
 */
export interface PromptOpDecl {
  kind: "prompt";
  /** Prompt source — an inline `template` OR a named `skill`. `{{inputs.x}}` interpolation
   *  applies either way, resolving against the operation's inputs. */
  prompt?: { template?: string; skill?: string };
  system?: string;
  /** The `LlmConfiguration` surface (model, sampling, `configRef`, …), merged by the runner. */
  config?: Record<string, JsonValue>;
  input?: Record<string, ParameterDecl>;
  output?: NamedParameterDecl;
}

/**
 * A `function` operation as authored: invoke a registered function (`registry.functions`) —
 * sync or async, host code or a DELEGATED AGENT ADAPTER (§3.1). A sub-workflow, a composite
 * unit, and a `claude-code` invocation are all this one shape; the resolved registry ENTRY's
 * capabilities distinguish them, never the op.
 */
export interface FunctionOpDecl {
  kind: "function";
  /** Registry name — a host function, or a runtime adapter (`claude-code`, …). */
  function: string;
  /** The authored surface bound as the op's `config` input (permission baseline, mode, …). */
  config?: Record<string, JsonValue>;
  input?: Record<string, ParameterDecl>;
  output?: NamedParameterDecl;
}

/** A state's operation as authored — desugared to an `Operation<InlineFamily>` by the loader. */
export type OperationDecl = PromptOpDecl | FunctionOpDecl;

// --- States ------------------------------------------------------------------

export interface ChildDecl {
  /** State ID of the child state file. */
  state: string;
  /** Wiring into the child's declared inputs — the same authored binding sugar (§2.1). */
  inputs?: Record<string, BindingDecl>;
  /** SPEC §10.4: starting this child does not block the sequence. */
  async?: boolean;
}

export interface TransitionDecl {
  /** A declared child key, or one of `terminate.*`. */
  to: string;
  /** Guard expression; absent = unconditional. Must INFER to boolean (§7.2) — strict, no
   *  truthiness coercion. */
  when?: string;
}

export interface LimitsDecl {
  /** Guard value exposed as `limits.max_iterations` in expressions (SPEC §3.4). */
  max_iterations?: number;
  /** State timeout in seconds → `terminate.timeout` when exceeded. */
  timeout?: number;
}

/**
 * The EXECUTION-ENVIRONMENT block: session/permission/tool concerns that configure how an
 * operation runs rather than what it is (DESIGN §5.1). Kept a sibling of
 * `operation` precisely because it is not part of the op's identity (§7.1).
 */
export interface EnvironmentDecl {
  /** Logical session id this state runs under — owns its conversation transcript, workspace, and
   *  permissions. Same id across states ⇒ a shared session; absent ⇒ the run's default session. */
  session?: string;
  /** Logical names of tools the operation may call mid-loop — resolved through `registry.tools`. */
  tools?: string[];
  /** Conversation preamble injected into THIS call (distinct from a `{ conversation }` wire, which
   *  reads a transcript as data, §7.5). */
  conversation?: {
    mode: ConversationMode;
    /** For `selected_artifacts`: names of artifacts to inject. */
    artifacts?: string[];
  };
  /** Authored per-state permission baseline (DESIGN §5.1, "the definition-authored baseline"). */
  permissions?: {
    profile?: PermissionProfile;
    default?: PermissionMode;
    tools?: Record<string, PermissionMode>;
  };
}

export interface StateDef {
  /** Equal to the file path without suffix; may be omitted and derived (validated when present). */
  id?: string;
  label?: string;
  description?: string;
  inputs?: Record<string, ParameterDecl>;
  outputs?: Record<string, NamedParameterDecl>;
  /** The state's operation (§7.1). A state with children and no operation is a pure composite. */
  operation?: OperationDecl;
  /** Execution environment — session, tools, conversation preamble, permissions. */
  environment?: EnvironmentDecl;
  children?: Record<string, ChildDecl>;
  sequence?: string[];
  transitions?: TransitionDecl[];
  limits?: LimitsDecl;
}

/** A state after loading: its operation desugared to base `Ref` cases, ready for the checker
 *  and the engine. `operation` is a real `Operation<InlineFamily>`; the authored sugar is gone. */
export interface LoadedState extends Omit<StateDef, "operation" | "inputs" | "outputs" | "children"> {
  id: string;
  inputs?: Record<string, Parameter<InlineFamily>>;
  outputs?: Record<string, NamedParameter<InlineFamily>>;
  operation?: Operation<InlineFamily>;
  children?: Record<string, LoadedChild>;
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
   */
  fanOut?: ReadonlySet<string>;
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
}

/** A loaded workflow: the root state ID plus every reachable state, keyed by state ID. */
export interface WorkflowBundle {
  rootId: string;
  states: Record<string, LoadedState>;
  /** The states AS AUTHORED (pre-desugaring), kept because the snapshot hash is the identity of
   *  what the author wrote — so improving the lowering never invalidates a stored snapshot. */
  source?: Record<string, StateDef>;
}

/**
 * Expression-context namespaces, split by ROLE after the rewrite (§7.5):
 *  - REF vocabulary — the data namespaces authored bindings point into. They are reachable from
 *    `{ expr }` leaves too, since an expr leaf IS a producer over the same data.
 *  - GUARD-ONLY scalars — control-flow state (`run`, `limits`), never a reference binding.
 * The old `function.*` namespace is GONE: a function state's result is an ordinary state output,
 * so guards read `outputs.*` / `children.<key>.outputs.*` uniformly.
 */
export const REF_NAMESPACES = ["inputs", "outputs", "children", "artifacts", "conversations"] as const;
export const GUARD_NAMESPACES = ["run", "limits"] as const;
export const CONTEXT_NAMESPACES = [...REF_NAMESPACES, ...GUARD_NAMESPACES] as const;
