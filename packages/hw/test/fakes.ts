/**
 * Test doubles: an abort-aware fake executor scripted per model id, and a scripted
 * interaction port. The fake stands in for @declarative-ai/llm's executor — the engine only
 * sees the core `Executor` contract.
 */
import type {
  ExecHandle,
  ExecutionSpec,
  Executor,
  ExecutorCapabilities,
  ExecServices,
  InteractionPort,
  Outcome,
  UnitKind,
} from "@declarative-ai/core";

const CAPS: ExecutorCapabilities = {
  structuredOutput: true,
  sessionResume: false,
  streaming: false,
  interactive: false,
  mutatesWorkspace: false,
  policyEnforcement: "none",
  memoizable: true,
  runtime: "edge-safe",
};

async function* empty(): AsyncGenerator<never> {}

export type Script = (spec: ExecutionSpec) => Outcome | Promise<Outcome>;

export const ok = (value: unknown, cost = 0.01): Outcome => ({
  value,
  rawText: JSON.stringify(value),
  metrics: { durationMs: 1, cost, inputTokens: 10, outputTokens: 20 },
});

/** A promise you resolve from the test body. */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (v: T) => void;
}
export function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

export class FakeExecutor implements Executor {
  readonly kind: UnitKind = "llm-call";
  readonly capabilities = CAPS;
  readonly calls: ExecutionSpec[] = [];
  readonly ctxs: ExecServices[] = [];
  constructor(private readonly script: Script) {}

  start(spec: ExecutionSpec, ctx: ExecServices): ExecHandle {
    this.calls.push(spec);
    this.ctxs.push(ctx);
    const outcome = (async (): Promise<Outcome> => {
      const canceled = new Promise<Outcome>((resolve) => {
        const onAbort = (): void =>
          resolve({ metrics: { durationMs: 0 }, error: { classification: "canceled", reason: "aborted" } });
        if (spec.abortSignal?.aborted) onAbort();
        else spec.abortSignal?.addEventListener("abort", onAbort, { once: true });
      });
      try {
        return await Promise.race([Promise.resolve(this.script(spec)), canceled]);
      } catch (e) {
        return { metrics: { durationMs: 0 }, error: { classification: "permanent", reason: (e as Error).message } };
      }
    })();
    return { events: empty(), outcome, cancel: async () => {} };
  }
}

/** Model id of an llm-call spec built by `llmCallBinding` (the fixture bindings put it in config). */
export function modelOf(spec: ExecutionSpec): string {
  return String((spec.definition as { model?: unknown }).model ?? "");
}

export function promptOf(spec: ExecutionSpec): string {
  return String((spec.definition as { prompt?: unknown }).prompt ?? "");
}

/** The rendered template tail — the prompt with any conversation-history preamble
 *  stripped, so scripts can dispatch on content without matching history echoes. */
export function promptTail(spec: ExecutionSpec): string {
  const p = promptOf(spec);
  const marker = "</conversation-history>";
  const at = p.lastIndexOf(marker);
  return at < 0 ? p.trim() : p.slice(at + marker.length).trim();
}

/** Scripted interaction port: responses per stateId (fifo when array). */
export class ScriptedPort implements InteractionPort {
  readonly requests: Array<{ stateId: string; component: string; inputs: unknown }> = [];
  constructor(private readonly responses: Record<string, unknown[] | ((req: { stateId: string; component: string; inputs: unknown }) => unknown)>) {}
  async request(req: { stateId: string; component: string; inputs: unknown }): Promise<unknown> {
    this.requests.push(req);
    const r = this.responses[req.stateId];
    if (r === undefined) throw new Error(`no scripted response for ${req.stateId}`);
    if (typeof r === "function") return r(req);
    const next = r.shift();
    if (next === undefined) throw new Error(`scripted responses for ${req.stateId} exhausted`);
    return next;
  }
}

export const rejectingPort: InteractionPort = {
  request: async (req) => {
    throw new Error(`interactive state '${req.stateId}' not allowed in this context`);
  },
};
