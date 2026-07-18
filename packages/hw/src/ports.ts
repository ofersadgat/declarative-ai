/**
 * The engine's outward-facing ports (DESIGN §7). Apps supply implementations:
 * JaiRA provides a renderer-backed InteractionPort and durable persistence;
 * findmyprompt provides a scripted InteractionPort and the bundled in-memory
 * persistence. The engine only ever talks to these interfaces.
 */
import type { ExecFailure, ExecMetrics, UnitKind } from "@ai-exec/core";
import type { TerminationOutcome } from "./format";

/** An artifact value flowing through workflow inputs/outputs (SPEC §4.6). For
 *  llm-backed states the content travels inline; process units use paths. */
export interface ArtifactRef {
  artifact: true;
  name: string;
  format?: string;
  content?: string;
  path?: string;
}

export function isArtifactRef(v: unknown): v is ArtifactRef {
  return typeof v === "object" && v !== null && (v as { artifact?: unknown }).artifact === true;
}

/** Engine events — the run-record stream (SPEC §10.2). Consumed live by the
 *  executor/UI and persisted by the `Persistence` port. */
export type EngineEvent =
  | { type: "instance.entered"; instanceId: number; stateId: string; childKey?: string; parentInstanceId?: number; inputs: Record<string, unknown> }
  | { type: "instance.blocked"; instanceId: number; stateId: string; reason: string }
  | { type: "operation.started"; instanceId: number; stateId: string; op: "ui" | "agent" | "skill" }
  | { type: "operation.completed"; instanceId: number; stateId: string; op: "ui" | "agent" | "skill"; metrics?: ExecMetrics }
  | { type: "operation.failed"; instanceId: number; stateId: string; op: "ui" | "agent" | "skill"; failure: ExecFailure }
  | { type: "transition.taken"; instanceId: number; stateId: string; to: string; iteration: number }
  | { type: "child.superseded"; instanceId: number; stateId: string; childKey: string }
  | { type: "instance.terminated"; instanceId: number; stateId: string; outcome: TerminationOutcome; failure?: ExecFailure };

/**
 * Durable run recording (SPEC §10.2/§10.3). The engine calls `record` at every step;
 * implementations persist (JaiRA: SQLite) or buffer (in-memory). v1 executes runs
 * in-process; cross-restart resume is built by consumers on top of this record
 * stream plus the snapshot hash.
 */
export interface Persistence {
  record(event: EngineEvent, atMs: number): void;
}

/** The bundled in-memory persistence — embedding & tests. */
export class InMemoryPersistence implements Persistence {
  readonly events: Array<{ event: EngineEvent; atMs: number }> = [];
  record(event: EngineEvent, atMs: number): void {
    this.events.push({ event, atMs });
  }
}

/**
 * Maps a state file's `agent.provider` name to an executor unit kind and builds the
 * executor definition for one operation (SPEC §7.1: providers are runtime adapters
 * with capabilities, not interchangeable strings). The built definition must match
 * what the target executor expects — e.g. for `llm-call`, @ai-exec/llm's
 * `LlmCallDefinition` shape.
 */
export interface ProviderBinding {
  kind: UnitKind;
  definition(req: {
    prompt: string;
    system?: string;
    /** Merged provider defaults + the state's `agent.config` overrides. */
    config: Record<string, unknown>;
    timeoutMs?: number;
  }): unknown;
}

/** Convenience binding for llm-call providers: `config.model` (+ passthrough sampling
 *  fields) become the definition; the rendered prompt rides along. */
export function llmCallBinding(defaults: Record<string, unknown>): ProviderBinding {
  return {
    kind: "llm-call",
    definition: ({ prompt, system, config, timeoutMs }) => ({
      ...defaults,
      ...config,
      prompt,
      ...(system !== undefined ? { system } : {}),
      ...(timeoutMs !== undefined ? { timeoutMs } : {}),
    }),
  };
}

/** Project skill library lookup (SPEC §3.2 `skill` operations; JaiRA DESIGN §7.4). */
export interface SkillDef {
  provider: string;
  template: string;
  config?: Record<string, unknown>;
}

export interface SkillResolver {
  get(name: string): SkillDef | undefined;
}
