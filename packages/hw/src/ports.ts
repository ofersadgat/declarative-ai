/**
 * The engine's outward-facing ports (DESIGN §7). Apps supply implementations:
 * JaiRA provides a renderer-backed InteractionPort and durable persistence;
 * findmyprompt provides a scripted InteractionPort and the bundled in-memory
 * persistence. The engine only ever talks to these interfaces.
 */
import { resolveConfig, type ConfigurationRegistry, type ExecFailure, type ExecMetrics, type UnitKind } from "@declarative-ai/core";
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
 * what the target executor expects — e.g. for `llm-call`, @declarative-ai/llm's
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

/** Options for {@link llmCallBinding}. */
export interface LlmCallBindingOptions {
  /** Named-config registry: a state's `config.configRef` is resolved to a preset and merged UNDER the
   *  state's inline config (over the binding `defaults`). */
  registry?: ConfigurationRegistry;
}

/**
 * Convenience binding for llm-call providers: every state's call is declared through the SAME declarative
 * `resolveConfig` pipeline — `defaults` ← `registry.get(config.configRef)` ← the state's inline `config`,
 * merged family-aware and strict-parsed. Each layer may be a `Partial<LlmCallDefinition>`: alongside the
 * config knobs it may carry DEFINITION-LAYER fields (`system`, `messages`, `attachments`, `timeoutMs` —
 * `resolveConfig` splits them out), so e.g. a shared system prompt or a per-state time budget lives in the
 * config surface, never silently dropped. The rendered template is THE operation prompt: config-layer
 * `messages` become preamble turns with the rendered prompt appended as the final user turn; a config-layer
 * `prompt` is an ERROR (nothing to do with it but silently lose one of the two). An unknown config key or a
 * malformed merged config throws (the engine turns that into a permanent operation failure).
 */
export function llmCallBinding(defaults: Record<string, unknown>, opts: LlmCallBindingOptions = {}): ProviderBinding {
  return {
    kind: "llm-call",
    definition: ({ prompt, system, config, timeoutMs }) => {
      const { configRef, ...inline } = config as { configRef?: unknown } & Record<string, unknown>;
      const preset = typeof configRef === "string" ? opts.registry?.get(configRef) : undefined;
      const { config: resolved, definition: defLayer } = resolveConfig([defaults, preset, inline]);
      if (defLayer.prompt !== undefined) {
        throw new Error(
          "config supplies `prompt`, but an hw operation's prompt is rendered from its template — remove it (use `system` for instructions, or `messages` for preamble turns)",
        );
      }
      const preamble = defLayer.messages as unknown[] | undefined;
      return {
        ...defLayer,
        ...resolved,
        ...(preamble !== undefined ? { messages: [...preamble, { role: "user", content: prompt }] } : { prompt }),
        ...(system !== undefined ? { system } : {}),
        ...(timeoutMs !== undefined ? { timeoutMs } : {}),
      };
    },
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
