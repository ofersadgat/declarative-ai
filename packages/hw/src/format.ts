/**
 * The hierarchical-workflow state-file format (SPEC §5). Declarative JSON; one file per
 * state; state ID = file path relative to the workflow root, without suffix (SPEC §2.4).
 */

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

export interface AgentConfig {
  /** Provider name, resolved through the engine's provider-binding map (a runtime
   *  adapter with capabilities, per SPEC §7.1 — never an interchangeable string). */
  provider: string;
  conversation?: {
    mode: ConversationMode;
    /** For `selected_artifacts`: names of artifacts to inject. */
    artifacts?: string[];
  };
  prompt?: {
    /** `{{inputs.x}}` / `{{params.x}}` interpolation. */
    template?: string;
  };
  /** Provider-specific overrides (model, sampling, ...), merged over the binding's defaults. */
  config?: Record<string, unknown>;
}

export interface SkillConfig {
  name: string;
  params?: Record<string, unknown>;
}

export interface UiConfig {
  component: string;
  prompt?: string;
  /** Component-specific configuration (options, form schema, ...). */
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
  agent?: AgentConfig;
  skill?: SkillConfig;
  ui?: UiConfig;
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
  "ui",
  "children",
  "run",
  "limits",
  "artifacts",
  "conversations",
] as const;
