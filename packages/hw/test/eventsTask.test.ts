/**
 * A workflow shaped like JaiRA's EVENTS task (decision 0010 §4–§5): tasks starting work on events.
 *
 * The shape: a root with NO sequence children, whose own rules are each `{ name, when: on_x(…), to:
 * <one async child> }`, and whose async children chain on through their own rules. What the engine
 * owes that shape, one test each:
 *
 *  - a rule's `name` rides onto `transition.taken`, is unique within its list, and is never part of a
 *    call's identity (§5.1);
 *  - `.event` in a rule's `inputs` is what the rule's deferred call resolved to — the first one's,
 *    when the guard read several (§5.2);
 *  - LISTENING waits (`HostCapabilities.listens`) stop nothing: every rule is armed at once, the first
 *    to come true fires, and the others stay armed through it (§5.3);
 *  - the state's own rules are armed while async children run — including a question that the
 *    rule which fired had withdrawn (§5.3);
 *  - a rule whose async target — or the chain it starts — is still running is not armed until that
 *    chain ends, so it never re-enters (and aborts) a running record; the event that arrived
 *    meanwhile is taken when it ends, in order (§5.4);
 *  - the root does not end when a chain does: it goes back to waiting.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecServices, type FunctionInputs, type InlineFamily, type ResolvedValue, type Signature } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { validateBundle } from "../src/validate.js";
import { newRegistry } from "./fakes.js";

const ON_EVENT_SIGNATURE = {
  input: { kind: { kind: "text", index: 0 } },
  output: { name: "value", kind: "json" },
} as const;

/**
 * A stand-in for JaiRA's event hub: per kind, a queue of events nobody was waiting for and a list of
 * waits nobody has answered. An event goes to the oldest wait of its kind, or is queued; a wait takes
 * the oldest queued event of its kind at once, or waits. A cancelled wait leaves the list.
 */
class Hub {
  readonly waits: Array<{ kind: string; resolve: (v: ResolvedValue) => void; canceled: boolean; answered: boolean }> = [];
  readonly queued = new Map<string, ResolvedValue[]>();
  registrations: string[] = [];

  wait(kind: string, signal?: AbortSignal): Promise<ResolvedValue> {
    this.registrations.push(kind);
    const queue = this.queued.get(kind) ?? [];
    if (queue.length > 0) return Promise.resolve(queue.shift()!);
    return new Promise((resolve) => {
      const w = { kind, resolve, canceled: false, answered: false };
      signal?.addEventListener("abort", () => {
        w.canceled = true;
      });
      this.waits.push(w);
    });
  }

  emit(kind: string, event: ResolvedValue): void {
    const w = this.waits.find((x) => x.kind === kind && !x.canceled && !x.answered);
    if (w !== undefined) {
      w.answered = true;
      w.resolve(event);
      return;
    }
    this.queued.set(kind, [...(this.queued.get(kind) ?? []), event]);
  }

  /** The kinds with a live (unanswered, uncancelled) wait registered. */
  armed(): string[] {
    return this.waits.filter((w) => !w.canceled && !w.answered).map((w) => w.kind).sort();
  }
}

/** Work that runs until the test releases it — a chain step that stays RUNNING for as long as needed. */
class Work {
  readonly started: Array<{ step: string; inputs: Record<string, unknown> }> = [];
  private readonly pending: Array<{ step: string; done: () => void; aborted: boolean }> = [];

  run(step: string, inputs: FunctionInputs, signal?: AbortSignal): Promise<{ value: ResolvedValue }> {
    this.started.push({ step, inputs: { ...inputs } });
    return new Promise((resolve) => {
      const p = { step, done: () => resolve({ value: { ok: step } as ResolvedValue }), aborted: false };
      signal?.addEventListener("abort", () => {
        p.aborted = true;
        resolve({ value: { aborted: step } as ResolvedValue });
      });
      this.pending.push(p);
    });
  }

  release(step: string): void {
    const i = this.pending.findIndex((p) => p.step === step && !p.aborted);
    if (i < 0) throw new Error(`nothing running for '${step}'`);
    this.pending.splice(i, 1)[0]!.done();
  }

  aborted(): string[] {
    return this.pending.filter((p) => p.aborted).map((p) => p.step);
  }
}

function harness(root: unknown, options: { listens?: boolean } = {}) {
  const hub = new Hub();
  const work = new Work();
  const registry = newRegistry();
  registry.functions.set(
    "on_x",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) => ({ value: await hub.wait(String(inputs.kind), ctx.abortSignal) }),
      { interactive: false, readOnly: true, memoizable: false, deferred: true, listens: options.listens ?? true },
      { signature: ON_EVENT_SIGNATURE as unknown as Signature<InlineFamily> },
    ),
  );
  // A QUESTION — an ordinary deferred call, which stops the list and is withdrawn when another rule
  // fires. What a person-facing offer is.
  registry.functions.set(
    "ask",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) => ({ value: await hub.wait(`ask:${String(inputs.kind)}`, ctx.abortSignal) }),
      { interactive: true, readOnly: true, memoizable: false, deferred: true },
      { signature: ON_EVENT_SIGNATURE as unknown as Signature<InlineFamily> },
    ),
  );
  for (const step of ["a1", "a2", "b1"]) {
    registry.functions.set(
      `work_${step}`,
      hostFunction<ExecServices, WorkflowMetrics>(
        async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) => work.run(step, inputs, ctx.abortSignal),
        { interactive: false, readOnly: false, memoizable: false },
      ),
    );
  }
  const step = (name: string) => ({
    inputs: { payload: { kind: "json", optional: true } },
    operation: { kind: "function", function: `work_${name}` },
  });
  const bundle = loadBundle({ "events.json": root, "a1.json": step("a1"), "a2.json": step("a2"), "b1.json": step("b1") }, "events", { functions: registry.functions });
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({ bundle, registry, validator: new SchemaValidator(), persistence });
  return { engine, hub, work, persistence, bundle };
}

async function settled(): Promise<void> {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

const taken = (persistence: InMemoryPersistence): Array<Extract<EngineEvent, { type: "transition.taken" }>> =>
  persistence.events
    .map(({ event }) => event as EngineEvent)
    .filter((e): e is Extract<EngineEvent, { type: "transition.taken" }> => e.type === "transition.taken");

/** The events task's shape: a root that only waits, each rule to one async chain. */
const eventsRoot = {
  sequence: [],
  children: {
    a1: { state: "a1", async: true, transitions: [{ when: "true", to: "a2" }] },
    a2: { state: "a2", async: true },
    b1: { state: "b1", async: true },
  },
  transitions: [
    { name: "on_a", when: "on_x('a')", to: "a1", inputs: { payload: ".event.n" } },
    { name: "on_b", when: "on_x('b')", to: "b1", inputs: { payload: ".event" } },
    { name: "stop", when: "on_x('stop')", to: "terminate.success" },
  ],
};

describe("the events task's shape", () => {
  it("validates clean", () => {
    const { bundle } = harness(eventsRoot);
    expect(validateBundle(bundle).errors).toEqual([]);
  });

  it("arms every listening rule at once, and does not end while it waits", async () => {
    const { engine, hub } = harness(eventsRoot);
    let done = false;
    const run = engine.run({ inputs: {} }).then((r) => {
      done = true;
      return r;
    });
    await settled();
    // Three rules, three waits — a listening wait stops nothing behind it.
    expect(hub.armed()).toEqual(["a", "b", "stop"]);
    expect(done).toBe(false);
    hub.emit("stop", {});
    expect((await run).outcome).toBe("success");
  });

  it("hears a second event while a chain runs, never aborts a running chain, and takes the queued event when it ends", async () => {
    const { engine, hub, work, persistence } = harness(eventsRoot);
    const run = engine.run({ inputs: {} });
    await settled();

    // (2) `.event` reaches the target's inputs — `.event.n` of the event that fired `on_a`.
    hub.emit("a", { n: 1 });
    await settled();
    expect(work.started).toEqual([{ step: "a1", inputs: { payload: 1 } }]);
    // (4) `on_a` is not re-armed while the chain it started runs; `on_b` and `stop` stay armed (3).
    expect(hub.armed()).toEqual(["b", "stop"]);

    // (3) A second, different event is heard while the first chain is still running.
    hub.emit("b", { from: "b" });
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1", "b1"]);
    expect(work.started[1]!.inputs).toEqual({ payload: { from: "b" } });

    // (4) The same event again while its chain runs: queued by the hub, and the running chain is
    // NOT re-entered — nothing aborted, nothing superseded.
    hub.emit("a", { n: 2 });
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1", "b1"]);

    // The chain's second step: a1's own rule enters a2. Still busy, so still not re-armed.
    work.release("a1");
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1", "b1", "a2"]);
    expect(hub.armed()).toEqual(["stop"]);

    // The chain ends — `on_a` re-arms, takes the queued event, and starts the chain again, in order.
    work.release("a2");
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1", "b1", "a2", "a1"]);
    expect(work.started[3]!.inputs).toEqual({ payload: 2 });

    expect(work.aborted()).toEqual([]);
    const superseded = persistence.events.map(({ event }) => event as EngineEvent).filter((e) => e.type === "child.superseded");
    expect(superseded).toEqual([]);
    // (1) Every taken rule reports its name; a child's unnamed rule reports none.
    expect(taken(persistence).map((t) => [t.to, t.name])).toEqual([
      ["a1", "on_a"],
      ["b1", "on_b"],
      ["a2", undefined],
      ["a1", "on_a"],
    ]);

    // The root keeps waiting after chains finish, until it is told to stop.
    work.release("b1");
    work.release("a1");
    await settled();
    work.release("a2");
    await settled();
    expect(hub.armed()).toEqual(["a", "b", "stop"]);
    hub.emit("stop", {});
    expect((await run).outcome).toBe("success");
  });

  it("takes the first matching rule when two answers land in one round, and keeps the other for the next", async () => {
    // Both rules listen for the SAME kind: each registers its own wait (distinct targets bind distinct
    // `inputs`, but the calls hash alike — so give them different kinds and emit both before a round).
    const { engine, hub, work } = harness(eventsRoot);
    const run = engine.run({ inputs: {} });
    await settled();
    hub.emit("b", { first: true });
    hub.emit("a", { n: 7 });
    await settled();
    // Whichever round saw both, neither answer was lost: both chains started.
    expect(work.started.map((s) => s.step).sort()).toEqual(["a1", "b1"]);
    expect(work.started.find((s) => s.step === "a1")!.inputs).toEqual({ payload: 7 });
    hub.emit("stop", {});
    await run;
  });
});

describe("a guard's wait is re-armed while an async child runs", () => {
  /**
   * An ordinary (question) wait stops the list, so only the first rule's is registered at first. When
   * it fires, the round withdraws the rest and enters an async child — and before this change the
   * state then blocked with no wait registered at all, deaf until the child finished. The second
   * question is now offered while the child runs, and answering it is heard.
   */
  it("registers the state's other question while the child it entered is still running", async () => {
    const { engine, hub, work } = harness({
      sequence: [],
      children: { a1: { state: "a1", async: true } },
      transitions: [
        { name: "go", when: "ask('go')", to: "a1" },
        { name: "halt", when: "ask('halt')", to: "terminate.canceled" },
      ],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(hub.armed()).toEqual(["ask:go"]); // a question stops the list

    hub.emit("ask:go", true);
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1"]);
    // `go` sits out (its target runs); `halt` is asked NOW, while a1 is still running.
    expect(hub.armed()).toEqual(["ask:halt"]);

    hub.emit("ask:halt", true);
    expect((await run).outcome).toBe("canceled");
    expect(work.aborted()).toEqual(["a1"]);
  });
});

describe("`.event`", () => {
  it("is the FIRST deferred answer the guard read, when it read several", async () => {
    const { engine, hub, work } = harness({
      sequence: [],
      children: { a1: { state: "a1", async: true } },
      transitions: [
        { when: "on_x('a') && on_x('b')", to: "a1", inputs: { payload: ".event" } },
        { name: "stop", when: "on_x('stop')", to: "terminate.success" },
      ],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    hub.emit("a", { which: "a" });
    await settled();
    hub.emit("b", { which: "b" });
    await settled();
    expect(work.started).toEqual([{ step: "a1", inputs: { payload: { which: "a" } } }]);
    hub.emit("stop", {});
    await run;
  });

  it("is refused anywhere but a transition's own inputs", () => {
    const { bundle } = harness({
      sequence: [],
      children: { a1: { state: "a1", async: true, inputs: { payload: ".event" } } },
      transitions: [{ when: "on_x('a') && .event === 1", to: "a1" }],
    });
    const messages = validateBundle(bundle).errors.map((e) => `${e.path}: ${e.message}`);
    expect(messages.some((m) => m.startsWith("children.a1.inputs.payload") && m.includes("'.event' is only readable in a transition's own inputs"))).toBe(true);
    expect(messages.some((m) => m.startsWith("transitions[0].when"))).toBe(true);
  });
});

describe("a rule's `name`", () => {
  it("must be unique within its list", () => {
    const { bundle } = harness({
      ...eventsRoot,
      transitions: [
        { name: "dup", when: "on_x('a')", to: "a1" },
        { name: "dup", when: "on_x('b')", to: "b1" },
      ],
    });
    const errors = validateBundle(bundle).errors;
    expect(errors.map((e) => [e.path, e.message])).toContainEqual(["transitions[1].name", "'dup' already names transitions[0] — a rule's name is unique within its list"]);
  });

  it("is never part of a call's identity", () => {
    const named = harness(eventsRoot).bundle.states.events!.transitions!;
    const renamed = harness({ ...eventsRoot, transitions: eventsRoot.transitions.map((t) => ({ ...t, name: `${t.name}_renamed` })) }).bundle.states.events!.transitions!;
    expect(renamed.map((t) => t.whenRef)).toEqual(named.map((t) => t.whenRef));
    expect(renamed.map((t) => t.inputRefs)).toEqual(named.map((t) => t.inputRefs));
  });
});

describe("a non-listening wait keeps its old meaning", () => {
  it("still stops the list: the rule behind a question is not armed until the question is gone", async () => {
    const { engine, hub, work } = harness(eventsRoot, { listens: false });
    const abort = new AbortController();
    const run = engine.run({ inputs: {}, abortSignal: abort.signal });
    await settled();
    expect(hub.armed()).toEqual(["a"]);
    hub.emit("a", { n: 1 });
    await settled();
    expect(work.started.map((s) => s.step)).toEqual(["a1"]);
    // `on_a` sits out while its chain runs, so the next question in the list is the one asked.
    expect(hub.armed()).toEqual(["b"]);
    abort.abort();
    expect((await run).outcome).toBe("canceled");
    expect(hub.armed()).toEqual([]);
  });
});
