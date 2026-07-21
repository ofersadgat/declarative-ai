/**
 * Test doubles for the new capability-registry model: a scripted `Runtime` set (one script dispatched by
 * runtime name), and scripted `HostFunction`s (was the InteractionPort). The fakes stand in for
 * @declarative-ai/llm's `createLlmRuntime` and host UI functions — the engine only sees the core contracts.
 */
import {
  MapCapabilityRegistry,
  type ExecHandle,
  type ExecServices,
  type ExecutorCapabilities,
  type HostFunction,
  type Outcome,
  type Runtime,
  type RuntimeOp,
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

/** A recorded runtime invocation: the normalized op plus the runtime NAME it was dispatched to. */
export type FakeCall = RuntimeOp & { name: string };

export type Script = (call: FakeCall) => Outcome | Promise<Outcome>;

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

/** A set of fake runtimes backed by ONE script (dispatched on the runtime name), sharing a call log. */
export class FakeRuntimes {
  readonly calls: FakeCall[] = [];
  readonly ctxs: ExecServices[] = [];
  constructor(private readonly script: Script) {}

  runtime(name: string): Runtime {
    return {
      capabilities: CAPS,
      run: (op: RuntimeOp, ctx: ExecServices): ExecHandle => {
        const call: FakeCall = { ...op, name };
        this.calls.push(call);
        this.ctxs.push(ctx);
        const outcome = (async (): Promise<Outcome> => {
          const canceled = new Promise<Outcome>((resolve) => {
            const onAbort = (): void =>
              resolve({ metrics: { durationMs: 0 }, error: { classification: "canceled", reason: "aborted" } });
            if (op.abortSignal?.aborted) onAbort();
            else op.abortSignal?.addEventListener("abort", onAbort, { once: true });
          });
          try {
            return await Promise.race([Promise.resolve(this.script(call)), canceled]);
          } catch (e) {
            return { metrics: { durationMs: 0 }, error: { classification: "permanent", reason: (e as Error).message } };
          }
        })();
        return { events: empty(), outcome, cancel: async () => {} };
      },
    };
  }

  /** Register the given runtime names into a registry, all backed by this script. */
  register(registry: MapCapabilityRegistry, names: string[]): MapCapabilityRegistry {
    for (const name of names) registry.runtimes.register(name, this.runtime(name));
    return registry;
  }
}

/** The runtime name a recorded call dispatched to (was the fixture's model id). */
export function modelOf(call: FakeCall): string {
  return call.name;
}

export function promptOf(call: FakeCall): string {
  return call.prompt ?? "";
}

/** The rendered template tail — the prompt with any conversation-history preamble
 *  stripped, so scripts can dispatch on content without matching history echoes. */
export function promptTail(call: FakeCall): string {
  const p = promptOf(call);
  const marker = "</conversation-history>";
  const at = p.lastIndexOf(marker);
  return at < 0 ? p.trim() : p.slice(at + marker.length).trim();
}

/** A scripted host function (was ScriptedPort): returns queued responses FIFO; records its invocations.
 *  Interactive by default (stands in for a UI component). */
export class ScriptedFunction implements HostFunction {
  readonly capabilities: { interactive?: boolean; pure?: boolean };
  readonly calls: Array<{ config: unknown; inputs: unknown }> = [];
  constructor(
    private readonly queue: unknown[],
    capabilities: { interactive?: boolean; pure?: boolean } = { interactive: true },
  ) {
    this.capabilities = capabilities;
  }
  run(args: Record<string, unknown>): unknown {
    this.calls.push(args as { config: unknown; inputs: unknown });
    const next = this.queue.shift();
    if (next === undefined) throw new Error("scripted function responses exhausted");
    return next;
  }
}

/** A function that always rejects — the search-context stand-in for a human gate. */
export const rejectingFunction: HostFunction = {
  capabilities: { interactive: true },
  run: (args) => {
    throw new Error(`interactive function '${(args as { config?: { name?: string } }).config?.name}' not allowed in this context`);
  },
};
