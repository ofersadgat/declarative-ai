/**
 * A call that WAITS — `HostCapabilities.deferred` (SPEC §3.3, EXPRESSIONS.md §3).
 *
 * The seam a host uses to make a transition wait on something outside the run: a person, an event, a
 * deadline. The registered function returns a promise it settles later; the engine starts the call,
 * reports the guard as pending, and runs the round again when the answer arrives.
 *
 * What the feature IS, one test each:
 *
 *  - the state does not terminate while a guard is waiting (a wait is not a "no match");
 *  - transitions AFTER the waiting one do not fire — the decision is outstanding, so a rule about it
 *    is not yet a rule about anything;
 *  - a guard's other conditions are the PRECONDITION: `severity > 2 && waits()` asks nobody at
 *    severity 1, because the short-circuit reaches the resolver and not just the value;
 *  - one registration per call, however many guards or rounds demand it;
 *  - a taken transition CONSUMES the answer, so the state does not act on it twice.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecServices, type FunctionInputs, type InlineFamily, type JsonValue, type ResolvedValue, type Signature } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { newRegistry } from "./fakes.js";

/**
 * The signature the deferred entry declares — the slots its positional arguments bind against.
 *
 * This used to be a hand-written operation DOCUMENT shipped through `LoadBundleOptions.documents`,
 * for the one reason that option existed: registering an implementation said what it does and
 * nothing about how it is called, so the only place to declare slots was a file. An entry declares
 * them itself now, and the document — an exact restatement of the impl's own parameters, kept in
 * step by hand — is gone.
 */
const AWAIT_EVENT_SIGNATURE = {
  input: {
    event: { kind: "text", index: 0 },
    options: { kind: "json", index: 1, optional: true },
    // Declared, never positional: the loader fills it with the rule the call sits in
    // (`TRANSITION_INPUT`).
    transition: { kind: "json", optional: true },
  },
  output: { name: "value", kind: "json" },
} as const;

/** One registration the test can answer, exactly as a UI would. */
interface Waiter {
  event: string;
  options: JsonValue | undefined;
  transition: JsonValue | undefined;
  settle: (value: ResolvedValue) => void;
  /** Set when the ENGINE cancelled the call — the offer being withdrawn, as a UI would see it. */
  canceled: boolean;
}

function harness(def: unknown) {
  const waiters: Waiter[] = [];
  const registry = newRegistry();
  registry.functions.set(
    "await_event",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs, ctx: { abortSignal?: AbortSignal }) =>
        new Promise((resolve) => {
          const waiter: Waiter = {
            event: String(inputs.event ?? ""),
            options: inputs.options as JsonValue | undefined,
            transition: inputs.transition as JsonValue | undefined,
            settle: (value) => resolve({ value }),
            canceled: false,
          };
          // A real hub withdraws its request here — the offer disappearing from the board is this
          // event, so a test that does not watch it cannot tell a cancelled wait from a live one.
          ctx.abortSignal?.addEventListener("abort", () => {
            waiter.canceled = true;
          });
          waiters.push(waiter);
        }),
      // The declaration that makes it a WAIT rather than a computation. `memoizable: false` because
      // what it returns is an event, not a function of its arguments.
      { interactive: true, readOnly: true, memoizable: false, deferred: true },
      { signature: AWAIT_EVENT_SIGNATURE as unknown as Signature<InlineFamily> },
    ),
  );
  registry.functions.set(
    "noop",
    hostFunction<ExecServices, WorkflowMetrics>(async () => ({ value: {} as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  /** A child that is still running when the wait is registered — see the dedup test. */
  registry.functions.set(
    "sleep",
    hostFunction<ExecServices, WorkflowMetrics>(
      async () =>
        new Promise((resolve) => {
          setTimeout(() => resolve({ value: {} as ResolvedValue }), 5);
        }),
      { interactive: false, readOnly: true, memoizable: true },
    ),
  );
  /** Long enough that a test can observe the round BEFORE it lands, and act between the two. */
  registry.functions.set(
    "dawdle",
    hostFunction<ExecServices, WorkflowMetrics>(
      async () =>
        new Promise((resolve) => {
          setTimeout(() => resolve({ value: {} as ResolvedValue }), 60);
        }),
      { interactive: false, readOnly: true, memoizable: true },
    ),
  );

  /** A child that ends badly — a classified failure travels as DATA (§4.2), which is the shape a
   *  real failing child has, and the case a wait must not hold open (see the last describe). */
  registry.functions.set(
    "fail",
    hostFunction<ExecServices, WorkflowMetrics>(
      async () => ({ error: { classification: "permanent" as const, reason: "risky failed" } }),
      { interactive: false, readOnly: true, memoizable: false },
    ),
  );

  const leaf = { operation: { kind: "function", function: "noop" } };
  const bundle = loadBundle(
    {
      "plan.json": def,
      "moved.json": leaf,
      "slow.json": { operation: { kind: "function", function: "sleep" } },
      "dawdling.json": { operation: { kind: "function", function: "dawdle" } },
      "failing.json": { operation: { kind: "function", function: "fail" } },
    },
    "plan",
    { functions: registry.functions },
  );
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({ bundle, registry, validator: new SchemaValidator(), persistence });
  return { engine, waiters, persistence };
}

/**
 * Let the engine reach its wait — the round has to run before there is anything to answer.
 *
 * A macrotask turn, not a count of microtasks: settling a wait can start a child, which runs an
 * operation, which starts another round, and how many `await`s that is depends on the state. Yielding
 * to the timer queue drains all of it however deep it goes.
 */
async function settled(): Promise<void> {
  for (let i = 0; i < 3; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

const takenTo = (persistence: InMemoryPersistence): string[] =>
  persistence.events
    .map(({ event }) => event as EngineEvent)
    .filter((e): e is Extract<EngineEvent, { type: "transition.taken" }> => e.type === "transition.taken")
    .map((e) => e.to);

const eventsOfType = (persistence: InMemoryPersistence, type: string): EngineEvent[] =>
  persistence.events.map(({ event }) => event as EngineEvent).filter((e) => e.type === type);

/**
 * A state whose first rule waits, with a rule behind it that would otherwise fire at once.
 *
 * Both targets terminate, so what the run RETURNS says which rule won.
 */
const waitsThenFallsBack = {
  inputs: { severity: { kind: "json", schema: { type: "number" } } },
  operation: { kind: "function", function: "noop" },
  transitions: [
    { to: "terminate.error", when: "await_event('task_drag', { to_state: 'moved' })" },
    { to: "terminate.success" },
  ],
};

describe("a deferred call makes a transition wait", () => {
  it("registers the wait with the options the call named", async () => {
    const { engine, waiters } = harness(waitsThenFallsBack);
    const run = engine.run({ inputs: { severity: 3 } });
    await settled();

    expect(waiters).toHaveLength(1);
    expect(waiters[0]!.event).toBe("task_drag");
    expect(waiters[0]!.options).toEqual({ to_state: "moved" });

    waiters[0]!.settle(false as ResolvedValue);
    expect((await run).outcome).toBe("success");
  });

  /**
   * The property the whole design turns on. The second rule is unconditional: with the wait SKIPPED
   * — the rule every other PENDING follows — it would fire immediately and the run would be over
   * before anybody could answer.
   */
  it("does not take a later transition while the wait is outstanding", async () => {
    const { engine, waiters, persistence } = harness(waitsThenFallsBack);
    const run = engine.run({ inputs: { severity: 3 } });
    await settled();

    expect(takenTo(persistence)).toEqual([]);

    waiters[0]!.settle(true as ResolvedValue);
    expect((await run).outcome).toBe("error");
    expect(takenTo(persistence)).toEqual(["terminate.error"]);
  });

  /** A wait that comes back FALSE hands the round on: the next rule is now a rule about a decision. */
  it("falls through to the next transition when the wait answers false", async () => {
    const { engine, waiters, persistence } = harness(waitsThenFallsBack);
    const run = engine.run({ inputs: { severity: 3 } });
    await settled();

    waiters[0]!.settle(false as ResolvedValue);
    expect((await run).outcome).toBe("success");
    expect(takenTo(persistence)).toEqual(["terminate.success"]);
  });

  /**
   * A state with nothing left to run would terminate — except that it is waiting. This is the tail of
   * the evaluation loop, and it is what makes such a state PAUSED on a decision rather than one that
   * quietly succeeded while somebody was still looking at it.
   */
  it("does not terminate while a guard is waiting", async () => {
    const { engine, waiters } = harness({
      operation: { kind: "function", function: "noop" },
      transitions: [{ to: "terminate.success", when: "await_event('task_drag')" }],
    });
    let done = false;
    const run = engine.run({ inputs: {} }).then((r) => {
      done = true;
      return r;
    });
    await settled();
    expect(done).toBe(false);
    expect(waiters).toHaveLength(1);

    waiters[0]!.settle(true as ResolvedValue);
    expect((await run).outcome).toBe("success");
  });
});

describe("a call is told which rule it is part of", () => {
  /**
   * The destination is already written on the transition, so a call that offers somebody the move
   * that rule describes should not have to be told it twice — the second telling is the one that
   * ends up disagreeing.
   */
  it("fills the declared `transition` input with the rule's target", async () => {
    const { engine, waiters } = harness(waitsThenFallsBack);
    const run = engine.run({ inputs: { severity: 3 } });
    await settled();

    expect(waiters[0]!.transition).toEqual({ to: "terminate.error" });

    waiters[0]!.settle(false as ResolvedValue);
    await run;
  });

  it("leaves an argument the author bound alone", async () => {
    const { engine, waiters } = harness({
      operation: { kind: "function", function: "noop" },
      transitions: [{ to: "terminate.success", when: "await_event('task_drag', { to_state: 'elsewhere' })" }],
    });
    const run = engine.run({ inputs: {} });
    await settled();

    // The rule still says where IT goes; the options still say what the author wrote. Neither is
    // overwritten by the other, which is what makes an explicit `to_state` an escape hatch rather
    // than a duplicate.
    expect(waiters[0]!.options).toEqual({ to_state: "elsewhere" });
    expect(waiters[0]!.transition).toEqual({ to: "terminate.success" });

    waiters[0]!.settle(true as ResolvedValue);
    await run;
  });
});

describe("the guard's other conditions are the precondition", () => {
  /**
   * `.inputs.severity > 2 && await_event(…)` at severity 1 must ask NOBODY. Under a static walk over
   * guard bindings every call in the tree ran before the guard was evaluated, so the left-hand side
   * could not stop the right-hand side from happening — which for a computation is a wasted call, and
   * for a wait is a request shown to somebody who should never have seen it.
   */
  const conditional = {
    inputs: { severity: { kind: "json", schema: { type: "number" } } },
    operation: { kind: "function", function: "noop" },
    transitions: [
      { to: "terminate.error", when: ".inputs.severity > 2 && await_event('task_drag', { to_state: 'moved' })" },
      { to: "terminate.success" },
    ],
  };

  it("never registers the wait when the condition ahead of it is false", async () => {
    const { engine, waiters, persistence } = harness(conditional);
    expect((await engine.run({ inputs: { severity: 1 } })).outcome).toBe("success");
    expect(waiters).toHaveLength(0);
    expect(takenTo(persistence)).toEqual(["terminate.success"]);
  });

  it("registers it when the condition ahead of it holds", async () => {
    const { engine, waiters } = harness(conditional);
    const run = engine.run({ inputs: { severity: 5 } });
    await settled();
    expect(waiters).toHaveLength(1);

    waiters[0]!.settle(true as ResolvedValue);
    expect((await run).outcome).toBe("error");
  });
});

describe("one wait per call", () => {
  /**
   * A guard is re-evaluated whenever anything finishes, and here something does: `fast` completes and
   * registers the wait, then `slow` completes and the round runs again with the wait still
   * outstanding. A second registration would put a second offer on somebody's screen for one rule.
   */
  it("registers once however many rounds evaluate the guard", async () => {
    const { engine, waiters } = harness({
      sequence: ["fast", "slow"],
      children: { fast: { state: "moved", async: true }, slow: { state: "slow", async: true } },
      transitions: [{ to: "terminate.success", when: "await_event('task_drag')" }],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters).toHaveLength(1);

    // `slow` is still going; when it lands the guard is asked again and finds its own question
    // already in flight.
    await new Promise((resolve) => setTimeout(resolve, 20));
    await settled();
    expect(waiters).toHaveLength(1);

    waiters[0]!.settle(false as ResolvedValue);
    // `false` takes no rule, so the state runs out of rules and succeeds.
    expect((await run).outcome).toBe("success");
  });
});

describe("a taken transition consumes the answer", () => {
  /**
   * The rule that keeps an event from being a memo. `moved` runs and finishes, which starts another
   * round; the guard is evaluated again, and if the old `true` were still readable it would re-enter
   * `moved` forever. Instead the round asks again — a FRESH registration, which is what "offer the
   * drag again" has to mean.
   */
  it("asks again rather than re-reading the answer it already acted on", async () => {
    const { engine, waiters, persistence } = harness({
      operation: { kind: "function", function: "noop" },
      children: { moved: { state: "moved" } },
      transitions: [{ to: "moved", when: "await_event('task_drag')" }],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters).toHaveLength(1);

    waiters[0]!.settle(true as ResolvedValue);
    await settled();
    expect(takenTo(persistence)).toEqual(["moved"]);
    // The child ran, the round came back round, and the state is waiting again — on a NEW question,
    // not on the answer to the old one.
    expect(waiters).toHaveLength(2);

    waiters[1]!.settle(false as ResolvedValue);
    expect((await run).outcome).toBe("success");
  });
});

describe("the journal says what is being waited on", () => {
  it("records the wait and its answer", async () => {
    const { engine, waiters, persistence } = harness(waitsThenFallsBack);
    const run = engine.run({ inputs: { severity: 3 } });
    await settled();

    const waiting = eventsOfType(persistence, "call.waiting");
    expect(waiting).toHaveLength(1);
    expect((waiting[0] as { call: string }).call).toBe("await_event");

    waiters[0]!.settle(false as ResolvedValue);
    await run;
    const done = eventsOfType(persistence, "call.settled");
    expect(done).toHaveLength(1);
    expect((done[0] as { outcome: string }).outcome).toBe("value");
  });
});

describe("a taken transition cancels the waits it did not answer", () => {
  /**
   * The rule that makes a list of transitions ONE decision rather than several.
   *
   * `slow` is still running when `fast` lands, so the round evaluates with the earlier rule still
   * false and the later one waits. When `slow` finishes, the earlier rule fires — and the wait it
   * overtook must be withdrawn, or a person is left holding an offer belonging to a decision that
   * has already been made somewhere else.
   */
  it("cancels a wait when an earlier rule fires in a later round", async () => {
    const { engine, waiters } = harness({
      sequence: ["fast", "slow"],
      children: { fast: { state: "moved", async: true }, slow: { state: "dawdling", async: true } },
      transitions: [
        { to: "terminate.success", when: ".children.slow.outcome === 'success'" },
        { to: "terminate.error", when: "await_event('task_drag')" },
      ],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters).toHaveLength(1);
    expect(waiters[0]!.canceled).toBe(false);

    // `slow` lands, the first rule is now true, and the round takes it.
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect((await run).outcome).toBe("success");
    expect(waiters[0]!.canceled).toBe(true);
  });

  /** A state that ends for any other reason withdraws its offers too — a terminated run holds none. */
  it("cancels a wait when the run is aborted", async () => {
    const { engine, waiters } = harness({
      operation: { kind: "function", function: "noop" },
      limits: { timeout: 0.02 },
      transitions: [{ to: "terminate.success", when: "await_event('task_drag')" }],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters).toHaveLength(1);

    expect((await run).outcome).toBe("timeout");
    expect(waiters[0]!.canceled).toBe(true);
  });
});

describe("a wait is only started for a rule the round can reach", () => {
  /**
   * Two rules, both waiting. Evaluation stops at the first — so the second is a rule about a
   * decision that has not been made, and starting its call would offer somebody a move the engine
   * could not act on if they made it. One question at a time is what "the pipeline stops here"
   * has to mean for the calls as well as for the evaluation.
   */
  it("does not start the wait behind a waiting rule", async () => {
    const { engine, waiters } = harness({
      operation: { kind: "function", function: "noop" },
      transitions: [
        { to: "terminate.error", when: "await_event('task_drag', { to_state: 'a' })" },
        { to: "terminate.success", when: "await_event('task_drag', { to_state: 'b' })" },
      ],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters.map((w) => (w.options as { to_state?: string }).to_state)).toEqual(["a"]);

    // …and once the first is answered `false`, the second gets its turn — the rules are still a
    // list, walked one at a time.
    waiters[0]!.settle(false as ResolvedValue);
    await settled();
    expect(waiters.map((w) => (w.options as { to_state?: string }).to_state)).toEqual(["a", "b"]);

    waiters[1]!.settle(true as ResolvedValue);
    expect((await run).outcome).toBe("success");
  });

  /** A rule that already FIRES makes every rule behind it irrelevant, calls included. */
  it("does not start a wait behind a rule that is already true", async () => {
    const { engine, waiters } = harness({
      inputs: { severity: { kind: "json", schema: { type: "number" } } },
      operation: { kind: "function", function: "noop" },
      transitions: [
        { to: "terminate.success", when: ".inputs.severity > 2" },
        { to: "terminate.error", when: "await_event('task_drag')" },
      ],
    });
    expect((await engine.run({ inputs: { severity: 5 } })).outcome).toBe("success");
    expect(waiters).toHaveLength(0);
  });
});

/**
 * A wait does not hold a FAILURE open.
 *
 * The rule on a child mount is "when this child ends, and somebody drags the card, move on". The child
 * ending in error is the case the rule never meant: only a TAKEN transition handles a failure (SPEC
 * §3.3), and a guard still asking has taken nothing. Seen live: a phase failed on its way out, the
 * root parked on the drag its rule was waiting for, the board showed the run as waiting on a person
 * over a state that was already dead — and the drag, when it came, walked the run into the next
 * phase with the failed one's outputs missing everywhere downstream.
 */
describe("a wait does not hold a child's failure open", () => {
  it("fails the state when the child the waiting rule guards terminated with error", async () => {
    const { engine, waiters } = harness({
      children: {
        risky: { state: "failing", transitions: [{ to: "moved", when: "await_event('task_drag')" }] },
        moved: { state: "moved" },
      },
      sequence: ["risky"],
    });
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toContain("child 'risky' terminated with error and no transition handled it");
    expect(result.failure?.reason).toContain("risky failed");
    // The offer was withdrawn with the state: a request left on somebody's screen for a run that has
    // ended is the thing this exists to prevent.
    expect(waiters).toHaveLength(1);
    expect(waiters[0]!.canceled).toBe(true);
  });

  it("still waits, and still moves, when the child it guards succeeded", async () => {
    const { engine, waiters, persistence } = harness({
      children: {
        first: { state: "moved", transitions: [{ to: "moved", when: "await_event('task_drag')" }] },
        moved: { state: "moved" },
      },
      sequence: ["first"],
    });
    const run = engine.run({ inputs: {} });
    await settled();
    expect(waiters).toHaveLength(1);
    waiters[0]!.settle(true as ResolvedValue);
    expect((await run).outcome).toBe("success");
    expect(takenTo(persistence)).toEqual(["moved"]);
  });
});
