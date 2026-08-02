/**
 * Calling an operation from an expression, END TO END (EXPRESSIONS.md §3).
 *
 * The expression lowers to a producer edge carrying the resolved operation; the engine runs it
 * before resolving the binding, memoized by the op's content hash; resolution reads the result. Same
 * division of labour a declared child already has — the engine produces, resolution reads.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecServices, type FunctionInputs, type InlineFamily, type Operation, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { FakePromptExecutor, newRegistry, ok, promptTail } from "./fakes.js";
import type { Vfs } from "../src/reference.js";
import type { WorkflowMetrics } from "../src/ports.js";

const ROOT = "/p/.jaira";
const WF = `${ROOT}/workflows`;
const FUNCTIONS = `${ROOT}/functions`;

function vfsOf(files: Record<string, string>): Vfs {
  return {
    list: (dir) => {
      const prefix = `${dir}/`;
      const names = new Set<string>();
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (!rest.includes("/")) names.add(rest);
      }
      return [...names];
    },
    read: (path) => files[path],
  };
}

/** `shout` upper-cases its argument, and counts how many times it actually ran. */
function registryWithShout(calls: { n: number }) {
  const registry = newRegistry();
  registry.functions.set(
    "noop",
    hostFunction<ExecServices, WorkflowMetrics>(async () => ({ value: {} as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  registry.functions.set(
    "joinTwo",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs) => {
        calls.n += 1;
        return { value: `${String(inputs.acc ?? "")}${String(inputs.item ?? "")}` as ResolvedValue };
      },
      { interactive: false, readOnly: true, memoizable: true },
    ),
  );
  registry.functions.set(
    "shout",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs) => {
        calls.n += 1;
        return { value: String(inputs.text ?? "").toUpperCase() as ResolvedValue };
      },
      { interactive: false, readOnly: true, memoizable: true },
    ),
  );
  return registry;
}

const files = {
  [`${WF}/plan.json`]: "{}",
  // A PROMPT operation as a callee — it declares its own output contract, unlike a state's prompt op
  // whose output is replaced by the state's produced outputs.
  [`${FUNCTIONS}/classify.json`]: JSON.stringify({
    kind: "prompt",
    prompt: "Classify: {{.inputs.text}}",
    model: "classifier",
    input: { text: { kind: "text", index: 0 } },
  }),
  // A two-parameter operation, for `reduce`: accumulator first, then the element.
  [`${FUNCTIONS}/joinTwo.json`]: JSON.stringify({
    kind: "function",
    function: "joinTwo",
    input: { acc: { kind: "text", index: 0 }, item: { kind: "text", index: 1 } },
  }),
  [`${FUNCTIONS}/shout.json`]: JSON.stringify({
    kind: "function",
    function: "shout",
    input: { text: { kind: "text", index: 0 } },
  }),
};

function bundleFor(def: unknown) {
  return loadBundle({ "plan.json": def }, "plan", {
    defaultRoot: [WF, FUNCTIONS],
    roots: { JAIRA: ROOT, PROJECT: "/p" },
    vfs: vfsOf(files),
  });
}

async function run(def: unknown, inputs: Record<string, ResolvedValue>, calls = { n: 0 }) {
  const registry = registryWithShout(calls);
  const engine = new WorkflowEngine({ bundle: bundleFor(def), registry, validator: new SchemaValidator() });
  const result = await engine.run({ inputs });
  return { result, calls };
}

describe("a call runs and its result reaches the binding", () => {
  /** The op the call names produces a value that lands in a declared output. */
  const withCall = {
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { loud: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("lowers to an edge carrying the resolved operation", () => {
    const binding = bundleFor(withCall).states.plan!.outputs!.loud!.binding as {
      op: Operation<InlineFamily>;
      parameters?: Record<string, unknown>;
    };
    expect(binding.op.kind).toBe("function");
    expect((binding.op as { functionRef: string }).functionRef).toBe("shout");
    expect(Object.keys(binding.parameters ?? {})).toEqual(["text"]);
  });
});

describe("running it", () => {
  const withCall = {
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { loud: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("runs the called operation and binds its result", async () => {
    const { result, calls } = await run(withCall, { issue: "ship it" });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.loud).toBe("SHIP IT");
    expect(calls.n).toBe(1);
  });

  /**
   * The memo key is the RESOLVED op's content hash — callee plus arguments — so one call evaluated
   * from two bindings is one execution, and a guard costs one however many rounds it is evaluated in.
   */
  it("runs one execution for the same call named twice", async () => {
    const { result, calls } = await run(
      {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: {
          a: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } },
          b: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } },
        },
        operation: { kind: "function", function: "noop" },
      },
      { issue: "once" },
    );
    expect(result.outputs?.a).toBe("ONCE");
    expect(result.outputs?.b).toBe("ONCE");
    expect(calls.n).toBe(1);
  });

  it("treats a different argument as a different call", async () => {
    const { result, calls } = await run(
      {
        inputs: { a: { kind: "text", schema: { type: "string" } }, b: { kind: "text", schema: { type: "string" } } },
        outputs: {
          x: { schema: { type: "string" }, binding: { expr: "shout(.inputs.a)" } },
          y: { schema: { type: "string" }, binding: { expr: "shout(.inputs.b)" } },
        },
        operation: { kind: "function", function: "noop" },
      },
      { a: "one", b: "two" },
    );
    expect(result.outputs?.x).toBe("ONE");
    expect(result.outputs?.y).toBe("TWO");
    expect(calls.n).toBe(2);
  });
});

describe("a prompt operation as a callee", () => {
  const withPromptCall = {
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { verdict: { schema: { type: "string" }, binding: { expr: "classify(.inputs.issue)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("renders the callee's own template with the call's argument, and binds the result", async () => {
    const seen: string[] = [];
    const registry = registryWithShout({ n: 0 });
    const prompt = new FakePromptExecutor((call) => {
      seen.push(promptTail(call));
      return ok("high");
    });
    const engine = new WorkflowEngine({
      bundle: bundleFor(withPromptCall),
      registry,
      prompt,
      validator: new SchemaValidator(),
    });
    const result = await engine.run({ inputs: { issue: "the build is broken" } });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.verdict).toBe("high");
    // `{{.inputs.text}}` is the CALLEE's parameter, filled by the call's positional argument — not the
    // enclosing state's `issue` input under another name.
    expect(seen).toEqual(["Classify: the build is broken"]);
  });
});

/**
 * A call's ARGUMENTS are ordinary reads, and the static passes have to see them.
 *
 * They live in the edge's `parameters`, not in the callee's own `input`, so a walk that only descends
 * `input` misses them. Both failures are silent: fan-out under-counts and an un-materialized blob is
 * raced by two readers (EXPRESSIONS.md §1.3), and the reachability obligation simply is not applied.
 */
describe("a call's arguments are visible to the static passes", () => {
  const loadWith = (defs: Record<string, unknown>) =>
    loadBundle(defs, "plan", { defaultRoot: [WF, FUNCTIONS], roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs: vfsOf(files) });

  it("counts a child read inside a call argument as fan-out", () => {
    const bundle = loadWith({
      "plan.json": {
        children: { c: { state: "./c" } },
        outputs: {
          // TWO consumers of the same child output — one direct, one through a call.
          direct: { schema: { type: "string" }, binding: ".children.c.outputs.doc" },
          shouted: { schema: { type: "string" }, binding: { expr: "shout(.children.c.outputs.doc)" } },
        },
      },
      "plan/c.json": {
        outputs: { doc: { schema: { type: "string" } } },
        operation: { kind: "function", function: "noop" },
      },
    });
    // Without the call's argument being walked this is ONE consumer, and nothing fans out.
    expect([...(bundle.states.plan!.fanOut ?? [])]).toContain("c doc");
  });

  it("reports a call argument that reads an undeclared child", () => {
    const bundle = loadWith({
      "plan.json": {
        outputs: { v: { schema: { type: "string" }, binding: { expr: "shout(.children.ghost.outputs.x)" } } },
        operation: { kind: "function", function: "noop" },
      },
    });
    // The reference checks run over the WHOLE binding now, so it no longer matters whether the call
    // or an operator happens to be the root of the tree.
    expect(validateBundle(bundle).errors.map((e) => e.message).join(" | ")).toMatch(/undeclared child 'ghost'/);
  });

  /**
   * A call VALIDATES — it used to be rejected outright ("binds an embedded operation, which no
   * binding can run"), which was correct while nothing could run one and would now fail every
   * workflow that calls anything.
   */
  it("validates a workflow that calls an operation", () => {
    const bundle = loadWith({
      "plan.json": {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { v: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
        operation: { kind: "function", function: "noop" },
      },
    });
    expect(validateBundle(bundle).errors).toEqual([]);
  });

  /** A zero-argument call still carries `parameters` (empty), so it reads as a call, not a definition. */
  it("runs a call with no arguments", async () => {
    const { result } = await run(
      {
        outputs: { loud: { schema: { type: "string" }, binding: { expr: "shout()" } } },
        operation: { kind: "function", function: "noop" },
      },
      {},
    );
    expect(result.outcome).toBe("success");
    expect(result.outputs?.loud).toBe("");
  });
});

/**
 * Both spellings of the same callee resolve (EXPRESSIONS.md §3): bare, through the search path, and
 * explicitly rooted. A rooted one is what lets a workflow name an operation the path does not carry.
 */
describe("a rooted reference as a callee", () => {
  it("resolves the same operation as the bare name", async () => {
    const bare = await run(
      {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { v: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { issue: "hi" },
    );
    const rooted = await run(
      {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { v: { schema: { type: "string" }, binding: { expr: "$JAIRA/functions/shout(.inputs.issue)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { issue: "hi" },
    );
    expect(bare.result.outputs?.v).toBe("HI");
    expect(rooted.result.outputs?.v).toBe("HI");
  });
});

/**
 * A call inside a GUARD (EXPRESSIONS.md §6).
 *
 * This used to throw `'yes' is not a known operation` at run time: guards were lowered by the ENGINE,
 * which has no callee resolver — resolving one is load-time knowledge. And even lowered, nothing ran
 * a guard's calls. Both halves are fixed: guards lower at load, and their calls run before the
 * (synchronous) evaluation reads the result.
 */
describe("a call inside a guard", () => {
  const guarded = (when: string) => ({
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { done: { schema: { type: "string" }, binding: { text: "ok" } } },
    operation: { kind: "function", function: "noop" },
    transitions: [{ to: "terminate.success", when }],
  });

  it("runs the call and takes the transition on a truthy result", async () => {
    const { result, calls } = await run(guarded("shout(.inputs.issue) === 'GO'"), { issue: "go" });
    expect(result.outcome).toBe("success");
    expect(calls.n).toBe(1);
  });

  it("does not take it when the call says otherwise", async () => {
    const { result } = await run(guarded("shout(.inputs.issue) === 'NOPE'"), { issue: "go" });
    // The guard is false, so the transition is not taken; the state still ends successfully.
    expect(result.outcome).toBe("success");
  });

  /** An unparseable guard is carried as data and never fires — it must not read as unconditional. */
  it("keeps an unparseable guard from firing, and reports it", () => {
    const bundle = bundleFor({
      outputs: { done: { schema: { type: "string" }, binding: { text: "ok" } } },
      operation: { kind: "function", function: "noop" },
      transitions: [{ to: "terminate.error", when: ".inputs.x ===" }],
    });
    expect(bundle.states.plan!.transitions![0]!.whenError).toBeDefined();
    expect(bundle.states.plan!.transitions![0]!.whenRef).toBeUndefined();
    expect(validateBundle(bundle).errors.length).toBeGreaterThan(0);
  });
});

/**
 * The memo answers "would someone else making the identical call reuse this answer?" — so the KEY is
 * content-addressed (`hashOperation` of the resolved op IS "this callee with these arguments"), and
 * the STORE is the host's to supply. An in-run `Map` answers it only within one run.
 */
describe("the call memo is content-addressed and injectable", () => {
  const def = {
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { loud: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("reuses a result across separate runs when the host supplies the cache", async () => {
    const store = new Map<string, { value?: unknown; error?: string }>();
    const cache = { get: (k: string) => store.get(k) as never, set: (k: string, v: never) => void store.set(k, v) };
    const calls = { n: 0 };

    for (let i = 0; i < 2; i++) {
      const engine = new WorkflowEngine({
        bundle: bundleFor(def),
        registry: registryWithShout(calls),
        validator: new SchemaValidator(),
        callCache: cache,
      });
      const result = await engine.run({ inputs: { issue: "same" } });
      expect(result.outputs?.loud).toBe("SAME");
    }
    // Two runs, one execution — which is the whole point.
    expect(calls.n).toBe(1);
    expect(store.size).toBe(1);
  });

  it("does NOT reuse across runs without one, which is why the seam exists", async () => {
    const calls = { n: 0 };
    for (let i = 0; i < 2; i++) {
      const engine = new WorkflowEngine({ bundle: bundleFor(def), registry: registryWithShout(calls), validator: new SchemaValidator() });
      await engine.run({ inputs: { issue: "same" } });
    }
    expect(calls.n).toBe(2);
  });
});

/**
 * EVERY operation dispatches through the operation executor — a call and a state's own operation
 * alike. Supplying one is how a host gets the wrapper stack (retry, rate limiting, budget, a
 * content-addressed memo) around dispatch; absent one the engine builds the plain dispatcher, so the
 * path is the same shape either way rather than a wrapped path and a raw `runFunction` call.
 */
describe("operations dispatch through the operation executor", () => {
  const def = {
    inputs: { issue: { kind: "text", schema: { type: "string" } } },
    outputs: { loud: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("routes a FUNCTION callee AND the state's own operation through it, so wrappers apply to both", async () => {
    const seen: string[] = [];
    const calls = { n: 0 };
    const operations = {
      capabilities: { interactive: false, readOnly: true, memoizable: true, structuredOutput: false, mutatesWorkspace: false, policyEnforcement: "none", sessionResume: false, streaming: false, runtime: "node" },
      metrics: { merge: (a: never) => a },
      start: (op: { kind: string; functionRef?: string }) => {
        seen.push(op.functionRef ?? op.kind);
        return {
          events: (async function* () {})(),
          result: Promise.resolve({ value: "WRAPPED" as never, metrics: { durationMs: 1, costUsd: 0 } }),
          cancel: async () => undefined,
        };
      },
    } as never;

    const engine = new WorkflowEngine({
      bundle: bundleFor(def),
      registry: registryWithShout(calls),
      validator: new SchemaValidator(),
      operations,
    });
    const result = await engine.run({ inputs: { issue: "x" } });
    expect(result.outputs?.loud).toBe("WRAPPED");
    // `noop` is the STATE's operation and `shout` the callee in its output binding — both arrive here,
    // in that order. A wrapper that only saw callees would leave every state operation unwrapped,
    // which is the gap this closes: `noop` used to reach `runFunction` from the engine directly.
    expect(seen).toEqual(["noop", "shout"]);
    // The registry entry was never invoked directly — the stack owns dispatch now.
    expect(calls.n).toBe(0);
  });

  it("resolves a state's operation ONCE, not on every dispatch", async () => {
    const registry = registryWithShout({ n: 0 });
    let lookups = 0;
    const counting = new Map(registry.functions);
    registry.functions = Object.assign(counting, {
      get(name: string) {
        lookups += 1;
        return Map.prototype.get.call(counting, name);
      },
    }) as never;

    const engine = new WorkflowEngine({
      bundle: bundleFor({
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { out: { schema: { type: "object" } } },
        operation: { kind: "function", function: "noop" },
      }),
      registry,
      validator: new SchemaValidator(),
    });
    await engine.run({ inputs: { issue: "x" } });
    const first = lookups;
    lookups = 0;
    await engine.run({ inputs: { issue: "y" } });
    const second = lookups;

    // The second run reuses the first's resolution: the operation carries its entry, so dispatch reads
    // it off the op instead of asking the registry again. A few lookups remain either way — the engine
    // reads an entry's capabilities to decide tool gating — so the claim is the DELTA, which is
    // precisely what the hoist removes. Without it the two runs cost the same.
    expect(second).toBeLessThan(first);
  });
});

/**
 * A callee DOCUMENT declares the parameters a call binds to; the implementation it names lives in the
 * registry. When the entry declares a signature the two must agree — the only check that catches
 * them drifting, since otherwise the document is the sole authority and a renamed parameter surfaces
 * as a missing argument at run time.
 */
describe("a callee document is checked against its registered signature", () => {
  const withSignature = (props: Record<string, unknown>) => {
    const registry = registryWithShout({ n: 0 });
    const entry = registry.functions.get("shout") as { signature?: unknown };
    entry.signature = { input: { kind: "json", schema: { type: "object", properties: props } }, output: { name: "value", kind: "json" } };
    return new Map([["shout", entry as never]]);
  };

  const bundle = () =>
    bundleFor({
      inputs: { issue: { kind: "text", schema: { type: "string" } } },
      outputs: { v: { schema: { type: "string" }, binding: { expr: "shout(.inputs.issue)" } } },
      operation: { kind: "function", function: "noop" },
    });

  it("says nothing when the document agrees with the implementation", () => {
    const errors = validateBundle(bundle(), { functions: withSignature({ text: { type: "string" } }) }).errors;
    expect(errors.map((e) => e.message).join(" | ")).not.toMatch(/does not accept/);
  });

  it("reports a document declaring a parameter the implementation does not take", () => {
    // The document's callee declares `text`; the impl says it takes `body`.
    const errors = validateBundle(bundle(), { functions: withSignature({ body: { type: "string" } }) }).errors;
    expect(errors.map((e) => e.message).join(" | ")).toMatch(/does not accept/);
  });

  it("constrains nothing when the entry declares no signature", () => {
    const registry = registryWithShout({ n: 0 });
    const errors = validateBundle(bundle(), { functions: registry.functions as never }).errors;
    expect(errors.map((e) => e.message).join(" | ")).not.toMatch(/does not accept/);
  });
});

/**
 * HIGHER-ORDER operations (EXPRESSIONS.md §3.5): applying an operation to each element of an array.
 *
 * The shape no other machinery has — how many operations run is not known until the array resolves,
 * so it cannot be the static walk `embeddedOpsOf` is. The engine resolves the array, runs one bound
 * application per element, and resolution reads them back by rebuilding the same per-element
 * identities.
 */
describe("map applies an operation to every element", () => {
  const mapDef = {
    inputs: { issues: { kind: "json", schema: { type: "array", items: { type: "string" } } } },
    outputs: { loud: { schema: { type: "array" }, binding: { expr: "map(.inputs.issues, shout)" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("runs once per element and assembles the results in order", async () => {
    const { result, calls } = await run(mapDef, { issues: ["a", "b", "c"] as never });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.loud).toEqual(["A", "B", "C"]);
    expect(calls.n).toBe(3);
  });

  /** Per-element memoization: a repeated element is one execution, because the key is its value. */
  it("pays once for a repeated element", async () => {
    const { result, calls } = await run(mapDef, { issues: ["a", "a", "b"] as never });
    expect(result.outputs?.loud).toEqual(["A", "A", "B"]);
    expect(calls.n).toBe(2);
  });

  it("maps an empty array without running anything", async () => {
    const { result, calls } = await run(mapDef, { issues: [] as never });
    expect(result.outputs?.loud).toEqual([]);
    expect(calls.n).toBe(0);
  });

  it("filters by the operation's result, keeping the ELEMENT", async () => {
    const { result } = await run(
      {
        inputs: { issues: { kind: "json", schema: { type: "array" } } },
        outputs: { kept: { schema: { type: "array" }, binding: { expr: "filter(.inputs.issues, shout)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { issues: ["a", "", "b"] as never },
    );
    // `shout("")` is falsy, so that element drops; the others keep their ORIGINAL value.
    expect(result.outputs?.kept).toEqual(["a", "b"]);
  });

  it("refuses an operation position that is not a name", () => {
    expect(() =>
      bundleFor({
        inputs: { issues: { kind: "json", schema: { type: "array" } } },
        outputs: { x: { schema: { type: "array" }, binding: { expr: "map(.inputs.issues, 1 + 2)" } } },
        operation: { kind: "function", function: "noop" },
      }),
    ).toThrow();
  });
});

/**
 * `reduce` is a FOLD, so its applications cannot be built up front the way `map`'s can — each step
 * consumes the previous result. Both sides rebuild the same CHAIN: the engine to run it, resolution
 * to read it, which works because every step is recorded under its own content hash.
 */
describe("reduce folds sequentially", () => {
  const reduceDef = {
    inputs: { parts: { kind: "json", schema: { type: "array" } } },
    outputs: { joined: { schema: { type: "string" }, binding: { expr: "reduce(.inputs.parts, joinTwo, '')" } } },
    operation: { kind: "function", function: "noop" },
  };

  it("threads the accumulator through every element", async () => {
    const { result, calls } = await run(reduceDef, { parts: ["a", "b", "c"] as never });
    expect(result.outcome).toBe("success");
    expect(result.outputs?.joined).toBe("abc");
    expect(calls.n).toBe(3);
  });

  it("returns the seed for an empty array, running nothing", async () => {
    const { result, calls } = await run(reduceDef, { parts: [] as never });
    expect(result.outputs?.joined).toBe("");
    expect(calls.n).toBe(0);
  });

  /** Each STEP is keyed by its own (accumulator, element) pair, so a repeat is still one execution. */
  it("reuses a step whose accumulator and element both repeat", async () => {
    const { result, calls } = await run(
      {
        inputs: { parts: { kind: "json", schema: { type: "array" } } },
        outputs: {
          a: { schema: { type: "string" }, binding: { expr: "reduce(.inputs.parts, joinTwo, '')" } },
          b: { schema: { type: "string" }, binding: { expr: "reduce(.inputs.parts, joinTwo, '')" } },
        },
        operation: { kind: "function", function: "noop" },
      },
      { parts: ["x", "y"] as never },
    );
    expect(result.outputs?.a).toBe("xy");
    expect(result.outputs?.b).toBe("xy");
    // Two identical folds, two steps each — but every step is content-addressed, so two runs total.
    expect(calls.n).toBe(2);
  });
});

/**
 * The awkward cases — nesting, emptiness, PENDING, failure and self-reference.
 *
 * Happy-path tests prove a feature exists; these are where it either holds together or does not.
 */
describe("calls and higher-order under stress", () => {
  it("nests a call inside a map's argument", async () => {
    const { result } = await run(
      {
        inputs: { xs: { kind: "json", schema: { type: "array" } } },
        outputs: { out: { schema: { type: "array" }, binding: { expr: "map(slice(.inputs.xs, 1, 3), shout)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { xs: ["a", "b", "c", "d"] as never },
    );
    expect(result.outputs?.out).toEqual(["B", "C"]);
  });

  it("maps over a map", async () => {
    const { result } = await run(
      {
        inputs: { xs: { kind: "json", schema: { type: "array" } } },
        outputs: { out: { schema: { type: "array" }, binding: { expr: "map(map(.inputs.xs, shout), shout)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { xs: ["a", "b"] as never },
    );
    expect(result.outputs?.out).toEqual(["A", "B"]);
  });

  it("refuses a non-array where an array belongs, rather than guessing", async () => {
    const { result } = await run(
      {
        inputs: { notList: { kind: "text", schema: { type: "string" } } },
        outputs: { out: { schema: { type: "array" }, binding: { expr: "map(.inputs.notList, shout)" } } },
        operation: { kind: "function", function: "noop" },
      },
      { notList: "nope" },
    );
    expect(result.outcome).toBe("error");
    expect(result.failure?.reason).toMatch(/expects an array/);
  });

  /** A call inside a call: the inner result must be bound before the outer one is even identified. */
  it("chains calls, innermost first", async () => {
    const { result, calls } = await run(
      {
        inputs: { issue: { kind: "text", schema: { type: "string" } } },
        outputs: { out: { schema: { type: "string" }, binding: { expr: "shout(shout(.inputs.issue))" } } },
        operation: { kind: "function", function: "noop" },
      },
      { issue: "ab" },
    );
    expect(result.outputs?.out).toBe("AB");
    // `shout("ab")` and `shout("AB")` are DIFFERENT calls — different arguments, different identity.
    expect(calls.n).toBe(2);
  });

  it("carries a failing element as error DATA rather than losing the successful ones", async () => {
    const registry = registryWithShout({ n: 0 });
    registry.functions.set(
      "shout",
      hostFunction<ExecServices, WorkflowMetrics>(
        async (inputs: FunctionInputs) => {
          const text = String(inputs.text ?? "");
          if (text === "bad") return { error: { classification: "permanent" as const, reason: "refused" } };
          return { value: text.toUpperCase() as ResolvedValue };
        },
        { interactive: false, readOnly: true, memoizable: true },
      ),
    );
    const engine = new WorkflowEngine({
      bundle: bundleFor({
        inputs: { xs: { kind: "json", schema: { type: "array" } } },
        // The slot accepts an array of anything, errors included — so the failure travels (§5).
        outputs: { out: { schema: { type: "array" }, binding: { expr: "map(.inputs.xs, shout)" } } },
        operation: { kind: "function", function: "noop" },
      }),
      registry,
      validator: new SchemaValidator(),
    });
    const result = await engine.run({ inputs: { xs: ["a", "bad", "c"] as never } });
    const out = result.outputs?.out as unknown[];
    expect(out[0]).toBe("A");
    expect(out[2]).toBe("C");
    expect((out[1] as { error?: { reason?: string } }).error?.reason).toBe("refused");
  });
});
