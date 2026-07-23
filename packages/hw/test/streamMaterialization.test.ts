/**
 * Stream MATERIALIZATION for blob leaves (SPEC §7.3). A `blob` leaf may hold a `ByteStream`; the engine
 * keeps it a stream where it can be piped (§7.4) and drains it to a `Uint8Array` where materialization is
 * required — a bind-time-known fan-out, the run RESULT, and a memo key. Each test below asserts one of
 * those, and the passthrough case that proves a single-consumer stream is NOT drained.
 */
import { describe, expect, it } from "vitest";
import {
  hashOperation,
  hostFunction,
  type ByteStream,
  type ByteStreamReader,
  type ExecServices,
  type FunctionInputs,
  type FunctionResult,
  type HostCapabilities,
  type InlineFamily,
  type Operation,
  type ResolvedValue,
} from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine";
import { loadBundle } from "../src/loader";
import { computeFanOut } from "../src/fanout";
import { isByteStream, materialize, MaterializeError } from "../src/materialize";
import { createWorkflowExecutor, type HierarchicalWorkflowDefinition } from "../src/executor";
import type { StateDef } from "../src/format";
import type { WorkflowMetrics } from "../src/ports";
import { errorOf, FakePromptExecutor, newRegistry } from "./fakes";

const BATCH: HostCapabilities = { interactive: false, readOnly: true, memoizable: true };
const CTX: ExecServices = { validator: new SchemaValidator() };
const bytes = (...n: number[]): Uint8Array => new Uint8Array(n);
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 5));

/** A `ByteStream` backed by fixed chunks. Counts `getReader()` calls (so a fan-out that drains twice is
 *  observable) and can be told to throw on the Nth `read()` (a mid-stream fault). */
class FakeByteStream implements ByteStream {
  readerCount = 0;
  canceled = false;
  constructor(
    private readonly chunks: Uint8Array[],
    private readonly failAt?: number,
  ) {}
  getReader(): ByteStreamReader {
    this.readerCount++;
    let i = 0;
    const self = this;
    return {
      read(): Promise<{ done: boolean; value?: Uint8Array }> {
        if (self.failAt !== undefined && i === self.failAt) return Promise.reject(new Error("read boom"));
        if (self.canceled || i >= self.chunks.length) return Promise.resolve({ done: true });
        return Promise.resolve({ done: false, value: self.chunks[i++] });
      },
      cancel(): Promise<void> {
        self.canceled = true;
        return Promise.resolve();
      },
      releaseLock(): void {},
    };
  }
}

/** A stream that yields one chunk, then blocks forever on the next `read()` until `cancel()` unblocks it
 *  with `{ done: true }` — the shape an abort has to interrupt. */
class HangingByteStream implements ByteStream {
  readerCount = 0;
  canceled = false;
  constructor(private readonly first: Uint8Array) {}
  getReader(): ByteStreamReader {
    this.readerCount++;
    let sent = false;
    let unblock: ((r: { done: boolean; value?: Uint8Array }) => void) | undefined;
    const self = this;
    return {
      read(): Promise<{ done: boolean; value?: Uint8Array }> {
        if (!sent) {
          sent = true;
          return Promise.resolve({ done: false, value: self.first });
        }
        return new Promise((resolve) => (unblock = resolve));
      },
      cancel(): Promise<void> {
        self.canceled = true;
        unblock?.({ done: true });
        return Promise.resolve();
      },
      releaseLock(): void {},
    };
  }
}

const blobSlot = { kind: "blob", schema: { type: "string", contentMediaType: "application/octet-stream" } } as const;

/** A parent with a `producer` (a function op that emits a blob stream) and `n` consumers, each a
 *  function op reading the producer's `img` output. With `n >= 2` the producer output fans out. */
function fanoutStates(consumerKeys: string[]): Record<string, StateDef> {
  const children: Record<string, unknown> = { producer: { state: "parent/producer" } };
  for (const key of consumerKeys) {
    children[key] = { state: "parent/consumer", inputs: { data: { child: "producer", output: "img" } } };
  }
  return {
    parent: { label: "Parent", children, sequence: ["producer", ...consumerKeys] } as StateDef,
    "parent/producer": {
      label: "Producer",
      outputs: { img: blobSlot },
      operation: { kind: "function", function: "gen", output: blobSlot },
    } as StateDef,
    "parent/consumer": {
      label: "Consumer",
      inputs: { data: blobSlot },
      outputs: { ok: { schema: { type: "boolean" } } },
      operation: { kind: "function", function: "consume" },
    } as StateDef,
  };
}

function fanoutEngine(states: Record<string, StateDef>, stream: ByteStream, seen: ResolvedValue[]): WorkflowEngine {
  const registry = newRegistry();
  registry.functions.set(
    "gen",
    hostFunction(async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({ value: stream as ResolvedValue }), BATCH),
  );
  registry.functions.set(
    "consume",
    hostFunction(async (inputs: FunctionInputs): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => {
      seen.push(inputs["data"] as ResolvedValue);
      return { value: { ok: true } };
    }, BATCH),
  );
  return new WorkflowEngine({
    bundle: loadBundle(states, "parent"),
    registry,
    prompt: new FakePromptExecutor(() => {
      throw new Error("no prompt op should run");
    }),
  });
}

describe("the materialize helper (§7.3)", () => {
  it("drains a multi-chunk stream into the concatenated Uint8Array; bytes pass through unchanged", async () => {
    const stream = new FakeByteStream([bytes(1, 2), bytes(3), bytes(4, 5, 6)]);
    expect(await materialize(stream, undefined, "x")).toEqual(bytes(1, 2, 3, 4, 5, 6));
    // Idempotent: an already-materialized leaf is returned as-is.
    const raw = bytes(9, 9);
    expect(await materialize(raw, undefined, "x")).toBe(raw);
  });

  it("dedupes concurrent drains of one stream — the reader is acquired ONCE", async () => {
    const stream = new FakeByteStream([bytes(1), bytes(2)]);
    const [a, b] = await Promise.all([materialize(stream, undefined, "x"), materialize(stream, undefined, "x")]);
    expect(a).toEqual(bytes(1, 2));
    expect(b).toBe(a); // the second caller shared the first drain's promise
    expect(stream.readerCount).toBe(1);
  });

  it("abort during a drain cancels the reader and fails NON-RETRIABLY", async () => {
    const stream = new HangingByteStream(bytes(1, 2, 3));
    const abort = new AbortController();
    const drained = materialize(stream, abort.signal, "operation input 'doc'");
    await tick(); // let the drain read the first chunk and block on the second
    abort.abort();
    await expect(drained).rejects.toBeInstanceOf(MaterializeError);
    expect(stream.canceled).toBe(true); // reader.cancel() was called, so the source is released
  });

  it("a drain fault surfaces as `<context>: <underlying message>`", async () => {
    const stream = new FakeByteStream([bytes(1)], 1); // fails on the second read
    await expect(materialize(stream, undefined, "operation input 'doc'")).rejects.toMatchObject({
      message: "operation input 'doc': read boom",
    });
  });
});

describe("fan-out materialization (§7.3, rule 2)", () => {
  it("detects a producer output read by two consumers, statically", () => {
    const bundle = loadBundle(fanoutStates(["c1", "c2"]), "parent");
    expect(computeFanOut(bundle.states["parent"]!)).toBeDefined();
    // A single consumer is NOT a fan-out — its stream must survive to be piped.
    expect(computeFanOut(loadBundle(fanoutStates(["c1"]), "parent").states["parent"]!)).toBeUndefined();
  });

  it("drains a fanned-out blob input for a function op — the impl receives the full bytes", async () => {
    const stream = new FakeByteStream([bytes(10, 20), bytes(30)]);
    const seen: ResolvedValue[] = [];
    const result = await fanoutEngine(fanoutStates(["c1", "c2"]), stream, seen).run({ inputs: {} });
    expect(result.outcome).toBe("success");
    // The first consumer's function op was handed the drained bytes, not a live stream.
    expect(seen[0]).toEqual(bytes(10, 20, 30));
    expect(seen[0]).toBeInstanceOf(Uint8Array);
  });

  it("the SAME stream fanned out to two consumers is drained ONCE; both see the full bytes", async () => {
    const stream = new FakeByteStream([bytes(10, 20), bytes(30)]);
    const seen: ResolvedValue[] = [];
    const result = await fanoutEngine(fanoutStates(["c1", "c2"]), stream, seen).run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(stream.readerCount).toBe(1); // one drain served both consumers
    expect(seen).toHaveLength(2);
    expect(seen[0]).toEqual(bytes(10, 20, 30));
    expect(seen[1]).toEqual(bytes(10, 20, 30));
  });

  it("a SINGLE-consumer stream stays UN-materialized — the impl receives the live stream (piping, §7.4)", async () => {
    const stream = new FakeByteStream([bytes(1, 2, 3)]);
    const seen: ResolvedValue[] = [];
    const result = await fanoutEngine(fanoutStates(["c1"]), stream, seen).run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(isByteStream(seen[0])).toBe(true); // still a stream — the engine did not drain it
    expect(stream.readerCount).toBe(0); // nor did anything acquire its reader
  });
});

describe("blob OUTPUT and RESULT materialization (§7.3, rule 3)", () => {
  function blobOutputWorkflow(failAt?: number): { definition: HierarchicalWorkflowDefinition; stream: FakeByteStream } {
    const stream = new FakeByteStream([bytes(0x89, 0x50), bytes(0x4e, 0x47)], failAt);
    const states: Record<string, StateDef> = {
      render: {
        label: "Render",
        outputs: { image: blobSlot },
        operation: { kind: "function", function: "gen", output: blobSlot },
      } as StateDef,
    };
    return { definition: { rootId: "render", states }, stream };
  }

  function executorFor(definition: HierarchicalWorkflowDefinition, produce: () => ResolvedValue) {
    const registry = newRegistry();
    registry.functions.set("gen", hostFunction(async (): Promise<FunctionResult<ResolvedValue, WorkflowMetrics>> => ({ value: produce() }), BATCH));
    return createWorkflowExecutor({
      definition,
      registry,
      prompt: new FakePromptExecutor(() => {
        throw new Error("no prompt op should run");
      }),
    });
  }

  const wfOp = (input: Operation<InlineFamily>["input"] = {}): Operation<InlineFamily> => ({
    kind: "function",
    functionRef: "wf",
    input,
    output: { name: "output", kind: "json" },
  });

  it("a blob-kind OUTPUT returning a stream is stored as the drained bytes", async () => {
    const { definition, stream } = blobOutputWorkflow();
    const outcome = await executorFor(definition, () => stream as ResolvedValue).start(wfOp(), CTX).result;
    expect(errorOf(outcome)).toBeUndefined();
    expect((outcome.value as Record<string, ResolvedValue>)["image"]).toEqual(bytes(0x89, 0x50, 0x4e, 0x47));
  });

  it("a drain fault storing a blob output fails the run NON-RETRIABLY, with the context-named reason", async () => {
    const { definition, stream } = blobOutputWorkflow(1); // the stream throws mid-drain
    const outcome = await executorFor(definition, () => stream as ResolvedValue).start(wfOp(), CTX).result;
    const failure = errorOf(outcome);
    expect(failure?.classification).toBe("permanent");
    expect(failure?.reason).toBe("workflow output 'image': read boom");
  });
});

describe("memoization materialization (§7.3, rule 1)", () => {
  function passthroughWorkflow(): HierarchicalWorkflowDefinition {
    const states: Record<string, StateDef> = {
      echo: {
        label: "Echo",
        inputs: { doc: blobSlot },
        outputs: { out: { ...blobSlot, binding: { input: "doc" } } },
      } as StateDef,
    };
    return { rootId: "echo", states };
  }

  it("materializes a stream blob INPUT so the op hashes — no exec hashOperation throw escapes", async () => {
    const stream = new FakeByteStream([bytes(7, 8), bytes(9)]);
    const op: Operation<InlineFamily> = {
      kind: "function",
      functionRef: "wf",
      input: { doc: { kind: "blob", binding: { blob: stream } } },
      output: { name: "output", kind: "json" },
    };
    // A live stream cannot be hashed — exec's hasher throws by design, the signal to materialize first.
    expect(() => hashOperation(op)).toThrow(/live stream/);

    const executor = createWorkflowExecutor({
      definition: passthroughWorkflow(),
      registry: newRegistry(),
      prompt: new FakePromptExecutor(() => {
        throw new Error("no prompt op should run");
      }),
    });
    const outcome = await executor.start(op, CTX).result;
    expect(errorOf(outcome)).toBeUndefined();
    expect((outcome.value as Record<string, ResolvedValue>)["out"]).toEqual(bytes(7, 8, 9));

    // The op's blob input was upgraded IN PLACE, so the memo hasher now sees bytes instead of throwing.
    expect((op.input["doc"]!.binding as { blob: unknown }).blob).toBeInstanceOf(Uint8Array);
    expect(() => hashOperation(op)).not.toThrow();
  });
});
