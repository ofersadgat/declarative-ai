/**
 * DIRECTED transitions, `skipped`, and STANDING rules (SPEC §3.3).
 *
 * A directed transition is a move the host injects on somebody's behalf — "go there", said by
 * dragging a card or to a conversation — and three things about it are the engine's to get right:
 *
 *  - it is HELD while its source still runs, and taken when the source ends, ahead of every rule;
 *  - a SKIP decides before it interrupts: the interrupted call here COMPLETES on abort (an agent's
 *    `interrupt()` ends the turn early and the call succeeds), which is exactly the completion that
 *    would otherwise walk the run into the next member;
 *  - what it steps over ends `skipped`, which an expression reads as an outcome and not an absence —
 *    live, and again after a load.
 *
 * A STANDING rule is the authored half of the same idea: the `on_user_event('task_move', …)` rule a
 * host generates. It waits without holding anything, so a task nobody moves follows its workflow as
 * though the rule were not there.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type ExecServices, type FunctionInputs, type HostCapabilities, type InlineFamily, type ResolvedValue, type Signature } from "@declarative-ai/exec";
import type { JsonValue } from "@declarative-ai/json";
import { SchemaValidator } from "@declarative-ai/validate";
import { newRegistry, ok } from "./fakes.js";
import { DirectedTransitions, isSkipAbort } from "../src/directed.js";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { validateBundle } from "../src/validate.js";

const HOST: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

const AWAIT_EVENT_SIGNATURE = {
  input: {
    event: { kind: "text", index: 0 },
    options: { kind: "json", index: 1, optional: true },
    transition: { kind: "json", optional: true },
  },
  output: { name: "value", kind: "json" },
} as const;

interface Waiter {
  options: JsonValue | undefined;
  settle: (value: boolean) => void;
  canceled: boolean;
}

/**
 * A workflow whose leaves are `step` calls the test can HOLD OPEN by name.
 *
 * A held step that is aborted answers with a SUCCESS — the partial answer an interrupted agent turn
 * comes back with. That is the trap the engine has to be ahead of: a test whose interrupted call
 * failed instead would pass with the jump decided too late.
 */
function harness(files: Record<string, StateDef>, options: { hold?: string[] } = {}) {
  const dispatched: string[] = [];
  const held = new Set(options.hold ?? []);
  const gates = new Map<string, () => void>();
  const toldSkipped: string[] = [];
  const waiters: Waiter[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "step",
    hostFunction<ExecServices, WorkflowMetrics>(async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) => {
      const { name, note } = inputs as { name: string; note?: string };
      dispatched.push(name);
      if (held.has(name)) {
        await new Promise<void>((resolve) => {
          gates.set(name, resolve);
          ctx.abortSignal?.addEventListener(
            "abort",
            () => {
              // The REASON reaches the implementation through the executor stack — what lets a host's
              // parked question withdraw itself on a skip and on nothing else.
              if (isSkipAbort(ctx.abortSignal?.reason)) toldSkipped.push(name);
              resolve();
            },
            { once: true },
          );
        });
      }
      return ok({ answer: note ?? `from ${name}` }) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, HOST),
  );
  registry.functions.set(
    "await_event",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) =>
        new Promise((resolve) => {
          const waiter: Waiter = { options: inputs.options as JsonValue | undefined, settle: (value) => resolve({ value }), canceled: false };
          ctx.abortSignal?.addEventListener("abort", () => {
            waiter.canceled = true;
          });
          waiters.push(waiter);
        }),
      { interactive: true, readOnly: true, memoizable: false, deferred: true },
      { signature: AWAIT_EVENT_SIGNATURE as unknown as Signature<InlineFamily> },
    ),
  );
  const rootId = Object.keys(files)[0]!;
  const bundle = loadBundle(files, rootId, { functions: registry.functions });
  const persistence = new InMemoryPersistence();
  const directed = new DirectedTransitions();
  let n = 0;
  const engine = new WorkflowEngine({ bundle, registry, validator: new SchemaValidator(), persistence, directed, newInstanceId: () => `new-${++n}` });
  const events = (): EngineEvent[] => persistence.events.map(({ event }) => event);
  return {
    engine,
    bundle,
    registry,
    directed,
    dispatched,
    toldSkipped,
    waiters,
    events,
    /** Let a held step finish on its own. */
    release: (name: string): void => {
      held.delete(name);
      gates.get(name)?.();
    },
    taken: () => events().filter((e): e is Extract<EngineEvent, { type: "transition.taken" }> => e.type === "transition.taken"),
    /** `instance.terminated` outcomes by child key, in journal order. */
    ended: (): Array<[string, string]> => {
      const keyOf = new Map<string, string>();
      for (const e of events()) if (e.type === "instance.entered") keyOf.set(e.instanceId, e.childKey ?? "(root)");
      return events()
        .filter((e): e is Extract<EngineEvent, { type: "instance.terminated" }> => e.type === "instance.terminated")
        .map((e) => [keyOf.get(e.instanceId) ?? e.instanceId, e.outcome]);
    },
  };
}

async function settled(): Promise<void> {
  for (let i = 0; i < 3; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

const leaf = (name: string): StateDef => ({
  label: name,
  inputs: { note: { schema: { type: "string" }, optional: true } },
  outputs: { answer: { schema: {} } },
  operation: { kind: "function", function: "step", args: { name } },
});

const outcomesOf = (keys: string[]): StateDef["outputs"] =>
  Object.fromEntries(keys.map((key) => [`${key}_outcome`, { schema: {}, optional: true, binding: `.children.${key}.outcome` }]));

/** Root → a, b, c, d in sequence; the root's outputs say how each child ended, AS AN EXPRESSION READS IT. */
const FOUR: Record<string, StateDef> = {
  root: {
    label: "Root",
    outputs: { ...outcomesOf(["a", "b", "c", "d"]), last: { schema: {}, optional: true, binding: ".children.d.output.answer" } },
    children: { a: { state: "root/a" }, b: { state: "root/b" }, c: { state: "root/c" }, d: { state: "root/d" } },
    sequence: ["a", "b", "c", "d"],
  },
  "root/a": leaf("a"),
  "root/b": leaf("b"),
  "root/c": leaf("c"),
  "root/d": leaf("d"),
};

describe("a directed transition, forward", () => {
  it("is HELD while the source runs, taken when it ends, and records what it jumped over as skipped", async () => {
    const h = harness(FOUR, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();

    expect(h.directed.direct({ to: "d", by: "person" })).toEqual({ status: "held" });
    await settled();
    // Held: nothing has moved, and nothing was interrupted.
    expect(h.taken()).toEqual([]);
    expect(h.dispatched).toEqual(["a"]);

    h.release("a");
    const result = await run;
    expect(result.outcome).toBe("success");
    // `b` and `c` never ran — and the source, which was waited for, ended as itself.
    expect(h.dispatched).toEqual(["a", "d"]);
    expect(h.taken()).toMatchObject([{ to: "d", by: "person", index: 1, iteration: 0 }]);
    expect(h.taken()[0]).not.toHaveProperty("skip");
    expect(result.outputs).toMatchObject({ a_outcome: "success", b_outcome: "skipped", c_outcome: "skipped", d_outcome: "success", last: "from d" });
    expect(h.ended()).toEqual([["a", "success"], ["b", "skipped"], ["c", "skipped"], ["d", "success"], ["(root)", "success"]]);
  });

  it("SKIP decides before it interrupts: the interrupted call completes, and the run still does not walk on", async () => {
    const h = harness(FOUR, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();

    expect(h.directed.direct({ to: "c", by: "person", skip: true })).toEqual({ status: "taking" });
    const result = await run;
    expect(result.outcome).toBe("success");
    // `a`'s call came back a SUCCESS when it was aborted. Had the jump been decided after that
    // landed, the completion would have entered `b`.
    expect(h.dispatched).toEqual(["a", "c", "d"]);
    expect(h.toldSkipped).toEqual(["a"]);
    expect(result.outputs).toMatchObject({ a_outcome: "skipped", b_outcome: "skipped", c_outcome: "success", d_outcome: "success" });
    expect(h.taken()).toMatchObject([{ to: "c", by: "person", skip: true }]);
    // One end per instance, and the source's is the skip — journaled at the decision, ahead of the
    // interrupt, so a run that died right here would still load with `a` stepped past.
    expect(h.ended()).toEqual([["a", "skipped"], ["b", "skipped"], ["c", "success"], ["d", "success"], ["(root)", "success"]]);
    const types = h.events().map((e) => e.type);
    expect(types.indexOf("transition.taken")).toBeLessThan(types.indexOf("instance.terminated"));
  });

  it("hands the target the asker's inputs, over the mount's wiring", async () => {
    const h = harness(FOUR, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    h.directed.direct({ to: "d", by: "control", skip: true, inputs: { note: "what the person said" } });
    const result = await run;
    expect(result.outputs).toMatchObject({ last: "what the person said" });
    expect(h.taken()).toMatchObject([{ to: "d", by: "control", inputs: { note: "what the person said" } }]);
  });

  it("refuses a move that names no child, or no live instance — and says which", async () => {
    const h = harness(FOUR, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    expect(h.directed.direct({ to: "nowhere", by: "person" })).toEqual({ status: "refused", reason: "'nowhere' is not a declared child of 'root'" });
    expect(h.directed.direct({ instanceId: "gone", to: "d", by: "person" })).toEqual({ status: "refused", reason: "no live instance 'gone'" });
    h.release("a");
    expect((await run).outcome).toBe("success");
    expect(h.dispatched).toEqual(["a", "b", "c", "d"]);
    // The run is over: a later move waits on the port for whoever loads the run next.
    expect(h.directed.direct({ to: "d", by: "person" })).toEqual({ status: "queued" });
  });

  it("the latest held move wins", async () => {
    const h = harness(FOUR, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    h.directed.direct({ to: "b", by: "person" });
    h.directed.direct({ to: "d", by: "person" });
    h.release("a");
    await run;
    expect(h.dispatched).toEqual(["a", "d"]);
  });
});

describe("a directed transition, backward", () => {
  it("re-enters the target as the next occurrence, with the usual backward reset", async () => {
    const h = harness(FOUR, { hold: ["b"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    expect(h.dispatched).toEqual(["a", "b"]);

    expect(h.directed.direct({ to: "a", by: "person" })).toEqual({ status: "held" });
    h.release("b");
    const result = await run;
    expect(result.outcome).toBe("success");
    // Back to `a`, and the tail re-runs from there — the workflow's own machinery.
    expect(h.dispatched).toEqual(["a", "b", "a", "b", "c", "d"]);
    // A backward move is a PASS: `iteration` counts it, exactly as an authored back-transition's.
    expect(h.taken()).toMatchObject([{ to: "a", by: "person", index: 1, iteration: 1 }]);
    const superseded = h.events().filter((e) => e.type === "child.superseded").map((e) => (e as { childKey: string }).childKey);
    expect(superseded).toEqual(["a", "b"]);
    // Nothing was jumped over, so nothing is skipped.
    expect(h.ended().filter(([, outcome]) => outcome === "skipped")).toEqual([]);
  });

  it("with SKIP, the interrupted source is skipped in the pass it ran in", async () => {
    const h = harness(
      {
        ...FOUR,
        root: { ...FOUR.root!, outputs: { ...FOUR.root!.outputs, b_before: { schema: {}, optional: true, binding: ".children.b[-2].outcome" } } },
      },
      { hold: ["b"] },
    );
    const run = h.engine.run({ inputs: {} });
    await settled();
    h.directed.direct({ to: "a", by: "person", skip: true });
    await settled();
    h.release("b"); // the SECOND b, entered by the new pass
    const result = await run;
    expect(h.dispatched).toEqual(["a", "b", "a", "b", "c", "d"]);
    expect(result.outputs).toMatchObject({ b_before: "skipped", b_outcome: "success" });
  });
});

describe("skipped survives a load", () => {
  const done = (id: string, key: string, value: JsonValue): LoadedInstance => ({
    id,
    stateId: `root/${key}`,
    childKey: key,
    inputs: {},
    live: false,
    outcome: "success",
    operation: { value: { answer: value } as ResolvedValue },
  });
  const skipped = (id: string, key: string): LoadedInstance => ({ id, stateId: `root/${key}`, childKey: key, inputs: {}, live: false, outcome: "skipped" });

  it("a loaded skipped child reads as `outcome: \"skipped\"`, and is not run", async () => {
    const h = harness(FOUR);
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      index: 1,
      cursor: 2,
      children: [done("i-a", "a", "recorded a"), skipped("i-b", "b"), { id: "i-c", stateId: "root/c", childKey: "c", inputs: {}, live: true }],
    });
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["c", "d"]);
    expect(result.outputs).toMatchObject({ a_outcome: "success", b_outcome: "skipped", c_outcome: "success" });
  });

  it("a run that died between the transition and the target's entry makes the entry it owes — once, unjournaled", async () => {
    const h = harness(FOUR);
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      index: 1,
      cursor: 2,
      directed: { to: "d", inputs: { note: "handed over" } },
      children: [done("i-a", "a", "recorded a"), skipped("i-b", "b"), skipped("i-c", "c")],
    });
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["d"]);
    expect(result.outputs).toMatchObject({ last: "handed over" });
    // The transition is already in the journal this was loaded from.
    expect(h.taken()).toEqual([]);
  });
});

describe("a finished run takes a move by being reopened", () => {
  const done = (id: string, stateId: string, key: string, value: JsonValue, children?: LoadedInstance[]): LoadedInstance => ({
    id,
    stateId,
    childKey: key,
    inputs: {},
    live: false,
    outcome: "success",
    ...(children === undefined ? { operation: { value: { answer: value } as ResolvedValue } } : { children }),
  });

  it("onto a TERMINATED root: the queued move is claimed at load, and the target re-enters as the next occurrence", async () => {
    const h = harness(FOUR);
    // Nothing is running yet — the move waits on the port for the load.
    expect(h.directed.direct({ to: "c", by: "person" })).toEqual({ status: "queued" });
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: false,
      outcome: "success",
      cursor: 3,
      children: [done("i-a", "root/a", "a", "ra"), done("i-b", "root/b", "b", "rb"), done("i-c", "root/c", "c", "rc"), done("i-d", "root/d", "d", "rd")],
    });
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["c", "d"]);
    expect(h.taken()).toMatchObject([{ instanceId: "i-root", to: "c", by: "person", iteration: 1 }]);
    expect(h.directed.queued()).toEqual([]);
    // The root ended again — the reopened run's own end.
    expect(h.ended().at(-1)).toEqual(["i-root", "success"]);
    // `a` and `b` stay as recorded.
    expect(result.outputs).toMatchObject({ a_outcome: "success", b_outcome: "success", last: "from d" });
  });

  it("onto a terminated instance BELOW the root: every instance on the way down is reopened", async () => {
    const NESTED: Record<string, StateDef> = {
      root: {
        label: "Root",
        outputs: { last: { schema: {}, optional: true, binding: ".children.phase.output.last" } },
        children: { phase: { state: "root/phase" }, after: { state: "root/after" } },
        sequence: ["phase", "after"],
      },
      "root/phase": {
        label: "Phase",
        outputs: { last: { schema: {}, optional: true, binding: ".children.y.output.answer" } },
        children: { x: { state: "root/phase/x" }, y: { state: "root/phase/y" } },
        sequence: ["x", "y"],
      },
      "root/phase/x": leaf("x"),
      "root/phase/y": leaf("y"),
      "root/after": leaf("after"),
    };
    const h = harness(NESTED);
    h.directed.direct({ instanceId: "i-phase", to: "y", by: "person", inputs: { note: "again, differently" } });
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: false,
      outcome: "success",
      cursor: 1,
      children: [
        done("i-phase", "root/phase", "phase", null, [done("i-x", "root/phase/x", "x", "rx"), done("i-y", "root/phase/y", "y", "ry")]),
        done("i-after", "root/after", "after", "ra"),
      ],
    });
    expect(result.outcome).toBe("success");
    // Only the target runs: what came after the reopened phase already has its record.
    expect(h.dispatched).toEqual(["y"]);
    expect(result.outputs).toEqual({ last: "again, differently" });
    expect(h.taken()).toMatchObject([{ instanceId: "i-phase", to: "y", by: "person" }]);
  });

  it("a STOPPED run reopened with a skip steps past its cut state without continuing it first", async () => {
    const h = harness(FOUR);
    h.directed.direct({ to: "d", by: "person", skip: true });
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 1,
      children: [
        done("i-a", "root/a", "a", "ra"),
        // Entered, its call cut mid-flight: a plain load would dispatch it again.
        { id: "i-b", stateId: "root/b", childKey: "b", inputs: {}, live: true },
      ],
    });
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["d"]);
    expect(result.outputs).toMatchObject({ b_outcome: "skipped", c_outcome: "skipped", d_outcome: "success" });
    const ends = h.events().filter((e): e is Extract<EngineEvent, { type: "instance.terminated" }> => e.type === "instance.terminated");
    expect(ends.find((e) => e.instanceId === "i-b")?.outcome).toBe("skipped");
  });
});

describe("a standing rule — what a host generates for `on_user_event('task_move', …)`", () => {
  /** Root → a, b in sequence, `moved` outside it; ONE standing rule offers the move to `moved`. */
  const standing = (extra: Partial<StateDef> = {}, rules: StateDef["transitions"] = []): Record<string, StateDef> => ({
    root: {
      label: "Root",
      inputs: { go: { schema: { type: "boolean" }, optional: true } },
      children: { a: { state: "root/a" }, b: { state: "root/b" }, moved: { state: "root/moved" }, other: { state: "root/other" } },
      sequence: ["a", "b"],
      transitions: [{ to: "moved", when: "await_event('task_move', { to_state: 'moved' })", standing: true, inputs: { note: "'wired by the rule'" } }, ...(rules ?? [])],
      ...extra,
    },
    "root/a": leaf("a"),
    "root/b": leaf("b"),
    "root/moved": leaf("moved"),
    "root/other": leaf("other"),
  });

  it("does not park the sequence, and does not keep a finished state open: the implicit end stays ahead of it", async () => {
    const h = harness(standing());
    const result = await h.engine.run({ inputs: {} });
    // Nobody moved it, so it ran as written and ENDED — withdrawing the offer on the way out.
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["a", "b"]);
    expect(h.taken()).toEqual([]);
    expect(h.waiters.length).toBeGreaterThan(0);
    expect(h.waiters.every((w) => w.canceled)).toBe(true);
  });

  it("sits LAST, behind every authored rule, wherever it was written", async () => {
    // Written FIRST; the authored rule behind it is true. A waiting rule ahead of it would block it.
    const h = harness(standing({}, [{ to: "other", when: ".inputs.go === true && .run.index === 0" }]));
    const result = await h.engine.run({ inputs: { go: true } });
    expect(result.outcome).toBe("success");
    expect(h.taken().map((t) => t.to)).toEqual(["other"]);
    expect(h.taken()[0]).not.toHaveProperty("by");
  });

  it("answered while the source still runs, it is HELD and taken when the source ends", async () => {
    const h = harness(standing(), { hold: ["b"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    // `a` finished, its round registered the offer, and `b` is running.
    expect(h.dispatched).toEqual(["a", "b"]);
    expect(h.waiters).toHaveLength(1);
    expect(h.waiters[0]!.options).toEqual({ to_state: "moved" });

    h.waiters[0]!.settle(true);
    await settled();
    expect(h.taken()).toEqual([]);
    expect(h.dispatched).toEqual(["a", "b"]);

    h.release("b");
    const result = await run;
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["a", "b", "moved"]);
    expect(h.taken().map((t) => t.to)).toEqual(["moved"]);
  });

  it("a taken transition cancels the waits it did not answer", async () => {
    const files = standing({}, [{ to: "other", when: "await_event('task_move', { to_state: 'other' })", standing: true }]);
    const h = harness(files, { hold: ["b", "moved"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    expect(h.waiters.map((w) => w.options)).toEqual([{ to_state: "moved" }, { to_state: "other" }]);
    h.waiters[0]!.settle(true);
    h.release("b");
    await settled();
    // `moved` was entered; the offer of `other` went with the round that took it.
    expect(h.dispatched).toEqual(["a", "b", "moved"]);
    expect(h.waiters[1]!.canceled).toBe(true);
    h.release("moved");
    expect((await run).outcome).toBe("success");
  });

  it("a DIRECTED move to the same child carries the standing rule's wiring, the asker's inputs over it", async () => {
    const files = standing({ outputs: { said: { schema: {}, optional: true, binding: ".children.moved.output.answer" } } });
    const wired = harness(files, { hold: ["a"] });
    const first = wired.engine.run({ inputs: {} });
    await settled();
    wired.directed.direct({ to: "moved", by: "person", skip: true });
    expect((await first).outputs).toMatchObject({ said: "wired by the rule" });

    const over = harness(files, { hold: ["a"] });
    const second = over.engine.run({ inputs: {} });
    await settled();
    over.directed.direct({ to: "moved", by: "person", skip: true, inputs: { note: "the asker's" } });
    expect((await second).outputs).toMatchObject({ said: "the asker's" });
  });

  it("is refused by validation with no guard, or when it would end the state", () => {
    const h = harness({
      root: {
        label: "Root",
        children: { a: { state: "root/a" } },
        sequence: ["a"],
        transitions: [
          { to: "a", standing: true },
          { to: "terminate.success", when: "await_event('task_move')", standing: true },
        ],
      },
      "root/a": leaf("a"),
    });
    const messages = validateBundle(h.bundle, { functions: h.registry.functions }).errors.map((e) => `${e.path}: ${e.message}`);
    expect(messages).toEqual([
      "transitions[0].when: a standing rule needs a `when` — it is taken when its guard comes true, and has none",
      "transitions[1].to: a standing rule enters a child; it cannot terminate the state",
    ]);
  });

  it("is outside the reachability proof: it pre-empts nothing the sequence proves, and its own target's wires are not held to it", () => {
    const wiredLeaf = (name: string): StateDef => ({ ...leaf(name), inputs: { note: { schema: {} } } });
    const files = (rules: StateDef["transitions"]): Record<string, StateDef> => ({
      root: {
        label: "Root",
        children: {
          a: { state: "root/a" },
          // On the spine, wired from the member before it — proven before the rule existed, and still.
          b: { state: "root/b", inputs: { note: ".children.a.output.answer" } },
          // Off the spine, entered only because somebody sends the task there.
          moved: { state: "root/moved", inputs: { note: ".children.b.output.answer" } },
        },
        sequence: ["a", "b"],
        transitions: rules,
      },
      "root/a": leaf("a"),
      "root/b": wiredLeaf("b"),
      "root/moved": wiredLeaf("moved"),
    });
    const errorsOf = (rules: StateDef["transitions"]): string[] => {
      const h = harness(files(rules));
      return validateBundle(h.bundle, { functions: h.registry.functions }).errors.map((e) => e.path);
    };
    expect(errorsOf([{ to: "moved", when: "await_event('task_move', { to_state: 'moved' })", standing: true }])).toEqual([]);
    // The same rule WITHOUT the mark is an ordinary conditional transition: it can fire before `a`
    // ends, so nothing is proven, and its target is not proven to follow anything.
    expect(errorsOf([{ to: "moved", when: "await_event('task_move', { to_state: 'moved' })" }])).toEqual(["children.b.inputs.note", "children.moved.inputs.note"]);
    // And a child an ORDINARY rule also enters keeps the obligation, whatever else offers it.
    expect(
      errorsOf([
        { to: "moved", when: ".children.b.output.answer === 'go'" },
        { to: "moved", when: "await_event('task_move', { to_state: 'moved' })", standing: true },
      ]),
    ).toEqual(["children.moved.inputs.note"]);
  });
});

describe("`skipped` in an expression", () => {
  it("type-checks against a child's outcome, and a guard can act on it", async () => {
    const files: Record<string, StateDef> = {
      ...FOUR,
      root: {
        ...FOUR.root!,
        children: { ...FOUR.root!.children, noticed: { state: "root/noticed" } },
        transitions: [{ to: "noticed", when: ".run.cursor === 'd' && .children.b.outcome === 'skipped' && .children.noticed.outcome !== 'success'" }],
      },
      "root/noticed": leaf("noticed"),
    };
    const h = harness(files, { hold: ["a"] });
    expect(validateBundle(h.bundle, { functions: h.registry.functions }).errors).toEqual([]);
    const run = h.engine.run({ inputs: {} });
    await settled();
    h.directed.direct({ to: "d", by: "person", skip: true });
    expect((await run).outcome).toBe("success");
    expect(h.dispatched).toEqual(["a", "d", "noticed"]);
  });
});

describe("the way down — a move to a NESTED target (`path`)", () => {
  /**
   * Root → a, feat; feat → p, ux; ux → first, draft, last. What `draft` answers is carried up to the
   * root, so the outputs say whether the move reached it and with what.
   */
  const DEEP: Record<string, StateDef> = {
    root: {
      label: "Root",
      outputs: { ...outcomesOf(["a", "feat"]), got: { schema: {}, optional: true, binding: ".children.feat.output.got" } },
      children: { a: { state: "root/a" }, feat: { state: "root/feat" } },
      sequence: ["a", "feat"],
    },
    "root/a": leaf("a"),
    "root/feat": {
      label: "Feature",
      outputs: { ...outcomesOf(["p"]), got: { schema: {}, optional: true, binding: ".children.ux.output.got" } },
      children: { p: { state: "root/feat/p" }, ux: { state: "root/feat/ux" } },
      sequence: ["p", "ux"],
    },
    "root/feat/p": leaf("p"),
    "root/feat/ux": {
      label: "UX",
      outputs: { ...outcomesOf(["first"]), got: { schema: {}, optional: true, binding: ".children.draft.output.answer" } },
      children: { first: { state: "root/feat/ux/first" }, draft: { state: "root/feat/ux/draft" }, last: { state: "root/feat/ux/last" } },
      sequence: ["first", "draft", "last"],
    },
    "root/feat/ux/first": leaf("first"),
    "root/feat/ux/draft": leaf("draft"),
    "root/feat/ux/last": leaf("last"),
  };
  const done = (id: string, key: string, value: JsonValue): LoadedInstance => ({
    id,
    stateId: `root/${key}`,
    childKey: key,
    inputs: {},
    live: false,
    outcome: "success",
    operation: { value: { answer: value } as ResolvedValue },
  });

  it("takes each step as the composite above it is entered, and hands the asker's inputs to the LAST state only", async () => {
    const h = harness(DEEP, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    expect(h.directed.direct({ to: "feat", path: ["ux", "draft"], by: "control", skip: true, inputs: { note: "deep" } })).toEqual({ status: "taking" });
    const result = await run;
    expect(result.outcome).toBe("success");
    // Nothing before the target ran at either level below: the composites went straight down.
    expect(h.dispatched).toEqual(["a", "draft", "last"]);
    expect(result.outputs).toMatchObject({ a_outcome: "skipped", got: "deep" });
    // One directed row per level, each on the composite that took it, with the rest of the way still
    // to go — the inputs travel with the descent and are handed only at the bottom.
    const taken = h.taken();
    expect(taken.map((row) => [row.stateId, row.to, row.by, row.skip])).toEqual([
      ["root", "feat", "control", true],
      ["root/feat", "ux", "control", true],
      ["root/feat/ux", "draft", "control", true],
    ]);
    expect(taken[0]).toMatchObject({ descent: { path: ["ux", "draft"], inputs: { note: "deep" }, by: "control", skip: true } });
    expect(taken[0]).not.toHaveProperty("inputs");
    expect(taken[1]).toMatchObject({ descent: { path: ["draft"], inputs: { note: "deep" } } });
    expect(taken[2]).toMatchObject({ inputs: { note: "deep" } });
    expect(taken[2]).not.toHaveProperty("descent");
    // What each composite stepped over is recorded, as it is at the top.
    expect(h.ended()).toEqual(expect.arrayContaining([["p", "skipped"], ["first", "skipped"]]));
  });

  it("a way down that is not there is refused before anything is taken", async () => {
    const h = harness(DEEP, { hold: ["a"] });
    const run = h.engine.run({ inputs: {} });
    await settled();
    expect(h.directed.direct({ to: "feat", path: ["ux", "nowhere"], by: "person" })).toEqual({
      status: "refused",
      reason: "'nowhere' is not a declared child of 'ux' on the way down",
    });
    h.release("a");
    expect((await run).outcome).toBe("success");
    expect(h.taken()).toEqual([]);
  });

  it("a run that stopped after entering a composite on the way down takes the next step when it is loaded", async () => {
    const h = harness(DEEP);
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      index: 1,
      cursor: 1,
      children: [
        done("i-a", "a", "ra"),
        // Entered by the root's directed step, and the process died before `feat` took its own.
        { id: "i-feat", stateId: "root/feat", childKey: "feat", inputs: {}, live: true, descent: { path: ["ux", "draft"], inputs: { note: "after a restart" }, by: "person" } },
      ],
    });
    expect(result.outcome).toBe("success");
    // Not `p` first: the loaded composite goes straight down, as it would have had it not stopped.
    expect(h.dispatched).toEqual(["draft", "last"]);
    expect(result.outputs).toMatchObject({ got: "after a restart" });
    expect(h.taken().map((row) => [row.instanceId, row.to])).toEqual([
      ["i-feat", "ux"],
      [expect.any(String), "draft"],
    ]);
    expect(h.taken()[0]).toMatchObject({ descent: { path: ["draft"], inputs: { note: "after a restart" } } });
  });

  it("an owed entry on the way down carries the rest of the way with it", async () => {
    const h = harness(DEEP);
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      index: 1,
      cursor: 1,
      // The root's step is in the journal; the process died before `feat` was entered.
      directed: { to: "feat", descent: { path: ["ux", "draft"], inputs: { note: "owed" }, by: "person" } },
      children: [done("i-a", "a", "ra")],
    });
    expect(result.outcome).toBe("success");
    expect(h.dispatched).toEqual(["draft", "last"]);
    expect(result.outputs).toMatchObject({ got: "owed" });
    // The root's own row is not journaled again; the two below it are new.
    expect(h.taken().map((row) => [row.stateId, row.to])).toEqual([
      ["root/feat", "ux"],
      ["root/feat/ux", "draft"],
    ]);
  });

  it("a move the port hands a loaded composite is the later word over the step it owed", async () => {
    const h = harness(DEEP);
    h.directed.direct({ instanceId: "i-feat", to: "ux", by: "person" });
    const result = await h.engine.loadRun({
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      index: 1,
      cursor: 1,
      children: [
        done("i-a", "a", "ra"),
        { id: "i-feat", stateId: "root/feat", childKey: "feat", inputs: {}, live: true, descent: { path: ["ux", "draft"], by: "person" } },
      ],
    });
    expect(result.outcome).toBe("success");
    // `ux` walked its own spine: the way down it owed was replaced, not taken as well.
    expect(h.dispatched).toEqual(["first", "draft", "last"]);
    expect(h.taken()[0]).toMatchObject({ instanceId: "i-feat", to: "ux" });
    expect(h.taken()[0]).not.toHaveProperty("descent");
  });
});
