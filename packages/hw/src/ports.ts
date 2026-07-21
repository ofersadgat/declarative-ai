/**
 * The engine's outward-facing ports (DESIGN §7). Apps supply implementations:
 * JaiRA provides durable persistence and renderer-backed interactive `functions`;
 * findmyprompt provides the bundled in-memory persistence and scripted functions.
 * Interactive/host behavior now lives in the `CapabilityRegistry` (`registry.functions`,
 * @declarative-ai/core); the engine only ever talks to these interfaces.
 */
import type { ExecFailure, ExecMetrics } from "@declarative-ai/core";
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
export type OperationKind = "runtime" | "function";

export type EngineEvent =
  | { type: "instance.entered"; instanceId: number; stateId: string; childKey?: string; parentInstanceId?: number; inputs: Record<string, unknown> }
  | { type: "instance.blocked"; instanceId: number; stateId: string; reason: string }
  | { type: "operation.started"; instanceId: number; stateId: string; op: OperationKind }
  | { type: "operation.completed"; instanceId: number; stateId: string; op: OperationKind; metrics?: ExecMetrics }
  | { type: "operation.failed"; instanceId: number; stateId: string; op: OperationKind; failure: ExecFailure }
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

// Runtime/function dispatch now goes through the typed `CapabilityRegistry` (@declarative-ai/core):
// `registry.runtimes` (was the `ProviderBinding` table + `llmCallBinding`, now the `llm` runtime absorbs
// that config pipeline — see @declarative-ai/llm `createLlmRuntime`), `registry.functions` (was the
// InteractionPort/ui components), and `registry.skills` (was `SkillResolver`, now name → prompt template).
