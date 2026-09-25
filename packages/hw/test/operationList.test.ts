/**
 * An operation LIST (SPEC §7.1d): `operation` written as an array of calls, run in order, whose
 * result is the array of their results.
 *
 * What is pinned here is the contract a list adds and nothing a single operation already had: the
 * calls run one after another and a failure stops the rest, a call can read the calls before it and
 * never itself or a later one, the list's value is typed as a tuple, every call inherits on its own,
 * the calls share the state's conversation unless one asks otherwise, each call is journaled at its
 * own index and site, and a loaded list resumes at the first call its journal did not record.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, MapSessionStore, withRecord, withSessionPosition, type ExecResult, type HostCapabilities, type JsonValue, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { environmentOf, operationsOf, type StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { FakePromptExecutor, newRegistry, ok, type Script } from "./fakes.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

/**
 * An engine whose `echo` function returns `{ answer: <its answer arg> }` and records every call, and
 * whose `boom` function fails. Prompt calls go to a fake executor behind a real session stack.
 */
function makeEngine(files: Record<string, StateDef>, rootId: string, script: Script = () => ok({ text: "said" })) {
  const calls: Record<string, unknown>[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "echo",
    hostFunction(async (inputs: Record<string, unknown>) => {
      calls.push(inputs);
      return ok({ answer: inputs["answer"] as JsonValue }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  registry.functions.set(
    "boom",
    hostFunction(async (inputs: Record<string, unknown>) => {
      calls.push(inputs);
      return { error: { classification: "permanent", reason: "boom" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } } as never;
    }, HOST),
  );
  const fake = new FakePromptExecutor(script);
  const sessions = new MapSessionStore();
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files, rootId),
    registry,
    prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
    sessions,
    validator: new SchemaValidator(),
    persistence,
  } as never);
  const events = (): EngineEvent[] => persistence.events.map(({ event }) => event);
  return { engine, calls, fake, events };
}

const echo = (answer: JsonValue, extra: Record<string, unknown> = {}) => ({ function: "echo", args: { answer }, ...extra });

describe("running a list", () => {
  it("runs the calls in order, and its result is the array of theirs", async () => {
    const { engine, calls } = makeEngine(
      {
        s: {
          operation: [echo("first"), echo("second")],
          outputs: {
            all: { binding: ".operation.output" },
            second: { binding: ".operation[1].output.answer" },
          },
        },
      },
      "s",
    );
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(calls.map((c) => c["answer"])).toEqual(["first", "second"]);
    expect(result.outputs).toEqual({ all: [{ answer: "first" }, { answer: "second" }], second: "second" });
  });

  it("returns a one-element array for a list of one — the shape follows what was written", async () => {
    const { engine } = makeEngine({ s: { operation: [echo("only")], outputs: { all: { binding: ".operation.output" } } } }, "s");
    expect((await engine.run({ inputs: {} })).outputs).toEqual({ all: [{ answer: "only" }] });
  });

  it("lets a call read the calls before it", async () => {
    const { engine, calls } = makeEngine(
      {
        s: {
          operation: [echo("plan"), { function: "echo", input: { answer: { binding: { $expr: "concat(.operation[0].output.answer, ' → acted')" } } } }],
          outputs: { acted: { binding: ".operation[1].output.answer" } },
        },
      },
      "s",
    );
    const result = await engine.run({ inputs: {} });
    expect(calls[1]!["answer"]).toBe("plan → acted");
    expect(result.outputs).toEqual({ acted: "plan → acted" });
  });

  it("stops at the first failure: the calls after it never run and the state ends in error", async () => {
    const { engine, calls, events } = makeEngine(
      { s: { operation: [echo("a"), { function: "boom" }, echo("never")], outputs: { a: { binding: ".operation[0].output.answer", optional: true } } } },
      "s",
    );
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(calls.map((c) => c["answer"])).toEqual(["a", undefined]);
    const failed = events().filter((e) => e.type === "operation.failed");
    expect(failed.map((e) => (e as { index?: number }).index)).toEqual([1]);
  });

  it("reads the list-wide outcome and cost, and a call's own", async () => {
    const { engine } = makeEngine(
      {
        s: {
          operation: [echo("a"), echo("b")],
          outputs: { outcome: { binding: ".operation.outcome" }, cost: { binding: ".operation.cost" }, first: { binding: ".operation[0].outcome" } },
        },
      },
      "s",
    );
    const result = await engine.run({ inputs: {} });
    expect(result.outputs?.["outcome"]).toBe("success");
    expect(result.outputs?.["cost"]).toBeCloseTo(0.02);
    expect(result.outputs?.["first"]).toBe("success");
  });

  it("journals every call at its own index and its own site", async () => {
    const { engine, events } = makeEngine({ s: { operation: [echo("same"), echo("same")], outputs: { all: { binding: ".operation.output" } } } }, "s");
    await engine.run({ inputs: {} });
    const started = events().filter((e) => e.type === "operation.started");
    const completed = events().filter((e) => e.type === "operation.completed") as Array<{ index?: number; operationId?: string }>;
    expect(started.map((e) => (e as { index?: number }).index)).toEqual([0, 1]);
    expect(completed.map((e) => e.index)).toEqual([0, 1]);
    // Two identical calls in one instance are still two records: the site is part of the identity.
    expect(completed[0]!.operationId).toBeDefined();
    expect(completed[0]!.operationId).not.toBe(completed[1]!.operationId);
  });

  it("carries no index for a single operation", async () => {
    const { engine, events } = makeEngine({ s: { operation: echo("one"), outputs: { answer: { binding: ".operation.output.answer" } } } }, "s");
    await engine.run({ inputs: {} });
    const started = events().find((e) => e.type === "operation.started") as { index?: number };
    expect("index" in started).toBe(false);
  });
});

describe("inheritance and conversations", () => {
  it("merges every call over the chain on its own", async () => {
    const { engine, fake } = makeEngine(
      {
        root: { environment: { model: "inherited" }, children: { s: { state: "root/s" } } },
        "root/s": {
          operation: [{ prompt: "one" }, { prompt: "two", model: "own" }],
          outputs: { all: { binding: ".operation.output" } },
        },
      },
      "root",
    );
    await engine.run({ inputs: {} });
    expect(fake.calls.map((c) => c.name)).toEqual(["inherited", "own"]);
  });

  it("shares the state's conversation across calls, unless a call asks for a fresh one", async () => {
    const { engine, fake } = makeEngine(
      {
        s: {
          operation: [{ prompt: "one", model: "m" }, { prompt: "two", model: "m" }, { prompt: "three", model: "m", session: null }],
          outputs: { all: { binding: ".operation.output" } },
        },
      },
      "s",
    );
    await engine.run({ inputs: {} });
    const [one, two, three] = fake.calls.map((c) => c.ctx.session!.at);
    expect(two!.id).toBe(one!.id);
    // Call 1 CONTINUES call 0 — the position after it, not a fork beside it.
    expect(two!.seq).toBe(one!.seq + 1);
    expect(three!.id).not.toBe(one!.id);
  });

  it("does not hand a call the state's outputs as its structured-output contract", async () => {
    const { engine, fake } = makeEngine(
      {
        s: {
          operation: [{ prompt: "one", model: "m", output: { text: { schema: { type: "string" } } } }],
          outputs: { text: { schema: { type: "string" }, binding: ".operation[0].output.text" }, extra: { schema: { type: "number" }, binding: { $expr: "1" } } },
        },
      },
      "s",
    );
    await engine.run({ inputs: {} });
    const schema = fake.calls[0]!.op.output.schema as { properties: Record<string, unknown> };
    expect(Object.keys(schema.properties)).toEqual(["text"]);
  });

  it("loads each call's environment aligned with the list, and a bound model under its index", () => {
    const bundle = loadBundle(
      {
        s: {
          inputs: { m: { schema: { type: "string" } } },
          operation: [{ prompt: "one", model: "a", tools: ["t"] }, { prompt: "two", model: { $expr: ".inputs.m" } }],
          outputs: { all: { binding: ".operation.output" } },
        },
      },
      "s",
    );
    const s = bundle.states["s"]!;
    expect(operationsOf(s)).toHaveLength(2);
    expect(environmentOf(s, 0)?.tools).toEqual(["t"]);
    expect(environmentOf(s, 1)?.tools).toBeUndefined();
    expect(environmentOf(s)).toBeUndefined();
    expect(s.fields?.map((f) => f.path)).toEqual(["operation.1.config.model"]);
  });
});

describe("validation", () => {
  const errorsOf = (files: Record<string, StateDef>, rootId = "s"): string[] =>
    validateBundle(loadBundle(files, rootId)).errors.map((e) => `${e.path}: ${e.message}`);
  const typed = (answer: string) => ({
    prompt: answer,
    model: "m",
    output: { answer: { schema: { type: "string" } }, score: { schema: { type: "number" } } },
  });

  it("accepts a list whose outputs all bind, reading any call", () => {
    expect(
      errorsOf({
        s: {
          operation: [typed("a"), { ...typed("b"), input: { prior: { schema: { type: "string" }, binding: ".operation[0].output.answer" } } }],
          outputs: { a: { schema: { type: "string" }, binding: ".operation[0].output.answer" }, b: { schema: { type: "number" }, binding: ".operation[1].output.score" } },
        },
      }),
    ).toEqual([]);
  });

  it("refuses an output that does not bind", () => {
    expect(errorsOf({ s: { operation: [typed("a")], outputs: { answer: { schema: { type: "string" } } } } })).toEqual([
      expect.stringMatching(/^outputs\.answer: a state whose operation is a list must bind every output/),
    ]);
  });

  it("refuses a call reading itself, a later call, or the whole list", () => {
    const errors = errorsOf({
      s: {
        operation: [
          { ...typed("a"), input: { self: { schema: {}, binding: ".operation[0].output.answer" } } },
          { ...typed("b"), input: { later: { schema: {}, binding: ".operation[2].output.answer" }, all: { schema: {}, binding: ".operation.output" } } },
          typed("c"),
        ],
        outputs: { a: { binding: ".operation[0].output.answer" } },
      },
    });
    expect(errors).toEqual(
      expect.arrayContaining([
        expect.stringMatching(/^operation\[0\]\.input\.self: call 0 reads '\.operation\[0\]', which is this call/),
        expect.stringMatching(/^operation\[1\]\.input\.later: call 1 reads '\.operation\[2\]', which runs after it/),
        expect.stringMatching(/^operation\[1\]\.input\.all: call 1 reads '\.operation' as a whole/),
      ]),
    );
  });

  it("types the list as a tuple — a name call 1 does not return is an error, and a wrong type is caught", () => {
    const errors = errorsOf({
      s: {
        operation: [typed("a"), { prompt: "b", model: "m", output: { verdict: { schema: { type: "boolean" } } } }],
        outputs: {
          missing: { schema: { type: "string" }, binding: ".operation[1].output.answer" },
          mistyped: { schema: { type: "string" }, binding: ".operation[1].output.verdict" },
        },
      },
    });
    expect(errors.some((e) => e.startsWith("outputs.missing:"))).toBe(true);
    expect(errors.some((e) => e.startsWith("outputs.mistyped:"))).toBe(true);
  });

  it("refuses an empty list", () => {
    expect(errorsOf({ s: { operation: [], outputs: {} } })).toEqual([expect.stringMatching(/^operation: operation is an empty list/)]);
  });

  it("names the call a list cannot build", () => {
    expect(errorsOf({ s: { operation: [typed("a"), {}], outputs: { a: { binding: ".operation[0].output.answer" } } } })).toContainEqual(
      expect.stringMatching(/^operation: operation\[1\]: operation declares neither a 'prompt' nor a 'function'/),
    );
  });
});

describe("a loaded list", () => {
  const files: Record<string, StateDef> = {
    s: { operation: [echo("a"), echo("b"), echo("c")], outputs: { all: { binding: ".operation.output" } } },
  };
  const record = (answer: string) => ({ value: { answer } as ResolvedValue });

  it("resumes at the first call its journal did not record", async () => {
    const { engine, calls } = makeEngine(files, "s");
    const loaded: LoadedInstance = { id: "i-s", stateId: "s", inputs: {}, live: true, operation: [record("recorded a")] };
    const result = await engine.loadRun(loaded);
    expect(calls.map((c) => c["answer"])).toEqual(["b", "c"]);
    expect(result.outputs).toEqual({ all: [{ answer: "recorded a" }, { answer: "b" }, { answer: "c" }] });
  });

  it("runs nothing when every call was recorded", async () => {
    const { engine, calls } = makeEngine(files, "s");
    const loaded: LoadedInstance = { id: "i-s", stateId: "s", inputs: {}, live: true, operation: [record("x"), record("y"), record("z")] };
    const result = await engine.loadRun(loaded);
    expect(calls).toEqual([]);
    expect(result.outputs).toEqual({ all: [{ answer: "x" }, { answer: "y" }, { answer: "z" }] });
  });
});
