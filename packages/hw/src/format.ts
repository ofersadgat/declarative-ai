/**
 * The hierarchical-workflow state-file format (SPEC §5). Declarative JSON; one file per
 * state; state ID = file path relative to the workflow root, without suffix (SPEC §2.4).
 */
import type { PermissionMode, PermissionProfile } from "@declarative-ai/core";

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

/**
 * A declared input/output/param field: a JSON-Schema subset plus the format's own
 * types. `type: "artifact"` marks a durable work product (SPEC §4.6); for llm-backed
 * states its content travels inline and the engine converts it to an artifact.
 * `type: "passthrough"` (outputs only, SPEC §4.4) forwards a `from` expression's
 * result verbatim, without schema validation.
 */
export interface FieldSchema {
  type: string;
  enum?: unknown[];
  items?: Record<string, unknown>;
  properties?: Record<string, unknown>;
  required?: string[];
  format?: string;
  /** SPEC §4.1: fields are required by default. */
  optional?: boolean;
  /** Inputs/params only: a default makes the field effectively optional. */
  default?: unknown;
  description?: string;
}

export interface OutputFieldSchema extends FieldSchema {
  /** Expression evaluated against the state's context when the state terminates. */
  from?: string;
}

/** Conversation context modes (SPEC §4.7). */
export type ConversationMode = "full_history" | "summary" | "fresh" | "selected_artifacts";

/**
 * A `runtime` operation (SPEC §7.1; HW-REDESIGN.md): run a named runtime adapter (llm, claude-code, …,
 * resolved through `registry.runtimes`) with a prompt and a config surface. The prompt comes from EITHER
 * an inline `template` OR a named `skill` (a template from `registry.skills`) — exactly one.
 */
export interface RuntimeConfig {
  /** Runtime name — resolved through `registry.runtimes` (a runtime adapter with capabilities, SPEC §7.1). */
  name: string;
  conversation?: {
    mode: ConversationMode;
    /** For `selected_artifacts`: names of artifacts to inject. */
    artifacts?: string[];
  };
  /** Prompt source — an inline `template` OR a named `skill` from `registry.skills`. `{{inputs.x}}` /
   *  `{{params.x}}` interpolation applies either way. */
  prompt?: {
    template?: string;
    skill?: string;
  };
  /** Extra `{{params.*}}` values for this render (e.g. a skill invocation's params). */
  params?: Record<string, unknown>;
  /** Config surface (model, sampling, `configRef`, …), merged by the runtime over its defaults. */
  config?: Record<string, unknown>;
  /** Logical names of tools this runtime (agent) may call mid-loop — resolved through `registry.tools`
   *  and handed to the runtime as executables (RUNTIMES-AND-PERMISSIONS.md §2). A composed runtime (llm)
   *  runs them in a bounded loop; a delegated runtime declares them to its agent. */
  tools?: string[];
  /** Logical session id this operation runs under — owns its conversation transcript, workspace, and
   *  permissions (RUNTIMES-AND-PERMISSIONS.md §3). Same id across states ⇒ a shared session; absent ⇒ the
   *  run's default session, so a plain workflow is one shared session. Set a distinct id to isolate. */
  session?: string;
  /** Authored per-state permission baseline (RUNTIMES-AND-PERMISSIONS.md §4): the starting `profile` and
   *  optional per-tool modes, overriding the engine's workflow-wide default (and shadowed by live human
   *  decisions). Only enforced when the engine is given an approver. */
  permissions?: {
    profile?: PermissionProfile;
    default?: PermissionMode;
    tools?: Record<string, PermissionMode>;
  };
}

/**
 * A `function` operation (HW-REDESIGN.md): invoke a registered host function (`registry.functions`) —
 * inputs → structured output, sync or async. A UI component is just an interactive function.
 */
export interface FunctionConfig {
  /** Function name — resolved through `registry.functions`. */
  name: string;
  /** Function-specific params (options, form schema, prompt text, …), passed through to the function. */
  [key: string]: unknown;
}

/** A wiring value (SPEC §4.2): a bare string is an expression evaluated against the
 *  parent's context; literal values must be wrapped as `{ "value": ... }`. */
export type WiringValue = string | { value: unknown };

export interface ChildDecl {
  /** State ID of the child state file. */
  state: string;
  inputs?: Record<string, WiringValue>;
  /** SPEC §10.4: starting this child does not block the sequence. */
  async?: boolean;
}

export interface TransitionDecl {
  /** A declared child key, or one of `terminate.*`. */
  to: string;
  /** Transition expression; absent = unconditional. */
  when?: string;
}

export interface LimitsDecl {
  /** Guard value exposed as `limits.max_iterations` in expressions (SPEC §3.4). */
  max_iterations?: number;
  /** State timeout in seconds → `terminate.timeout` when exceeded. */
  timeout?: number;
}

export interface StateDef {
  /** Equal to the file path without suffix; may be omitted and derived (validated when present). */
  id?: string;
  label?: string;
  description?: string;
  params?: Record<string, FieldSchema>;
  inputs?: Record<string, FieldSchema>;
  outputs?: Record<string, OutputFieldSchema>;
  runtime?: RuntimeConfig;
  function?: FunctionConfig;
  children?: Record<string, ChildDecl>;
  sequence?: string[];
  transitions?: TransitionDecl[];
  limits?: LimitsDecl;
}

/** A loaded workflow: the root state ID plus every reachable state, keyed by state ID. */
export interface WorkflowBundle {
  rootId: string;
  states: Record<string, StateDef>;
}

/** Expression-context namespaces (SPEC §6.1). */
export const CONTEXT_NAMESPACES = [
  "inputs",
  "outputs",
  "params",
  "function",
  "children",
  "run",
  "limits",
  "artifacts",
  "conversations",
] as const;
