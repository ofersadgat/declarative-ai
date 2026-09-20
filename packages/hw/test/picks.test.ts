/**
 * Alternatives, and the rule for choosing (NAMES.md §6).
 *
 * `$pick` is an expression and what it returns is the value used; it reads the alternatives as `.any`.
 * WHEN it runs is not written anywhere, because it belongs to the position: a model is fixed when its
 * session is created, and everything else when the instance that reads it is entered. Either way the
 * choice is made once and journaled, which is what these pin.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, type ExecResult, type HostCapabilities, type JsonValue, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent, type WorkflowMetrics } from "../src/ports.js";
import { FakePromptExecutor, modelOf, newRegistry, ok } from "./fakes.js";

/** Reads live state, so it must not be memoized — the capability NAMES.md §8 points `model_limits` at. */
const LIVE: HostCapabilities = { interactive: false, readOnly: true, memoizable: false };

interface Ran {
  models: string[];
  configs: Array<Record<string, unknown>>;
  /** How many times the limits were asked for — once per PICK, whatever the number of alternatives. */
  asked: number;
  args: Array<Record<string, unknown>>;
  events: EngineEvent[];
  outcome: string;
  reason?: string;
}

async function run(files: Record<string, unknown>, remaining: Record<string, number>, loaded?: LoadedInstance): Promise<Ran> {
  const fake = new FakePromptExecutor(() => ok({}));
  const registry = newRegistry();
  const picks = new Set<string>();
  const args: Array<Record<string, unknown>> = [];
  registry.functions.set(
    "model_limits",
    hostFunction(async (inputs: Record<string, unknown>) => {
      const alternative = inputs.alternative as { model?: string } | string;
      const model = typeof alternative === "string" ? alternative : String(alternative.model);
      picks.add(model);
      return ok({ remaining: remaining[model] ?? 0 } as JsonValue) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, LIVE),
  );
  registry.functions.set(
    "record",
    hostFunction(async (inputs: Record<string, unknown>) => {
      args.push(inputs);
      return ok({}) as ExecResult<ResolvedValue, WorkflowMetrics>;
    }, LIVE),
  );
  let asked = 0;
  const limits = registry.functions.get("model_limits")!;
  registry.functions.set("model_limits", {
    ...limits,
    signature: { input: { alternative: { kind: "json" } }, output: { name: "output", kind: "json", schema: {} } },
    impl: (async (...a: unknown[]) => {
      asked++;
      return (limits.impl as (...x: unknown[]) => unknown)(...a);
    }) as never,
  } as never);
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle: loadBundle(files as Record<string, StateDef>, "root", { functions: registry.functions as never }),
    registry,
    validator: new SchemaValidator(),
    prompt: fake as never,
    persistence,
  });
  const result = loaded !== undefined ? await engine.loadRun(loaded, { inputs: {} }) : await engine.run({ inputs: {} });
  return {
    models: fake.calls.map(modelOf),
    configs: fake.calls.map((call) => call.op.config as Record<string, unknown>),
    asked,
    args,
    events: persistence.events.map(({ event }) => event),
    outcome: result.outcome,
    ...(result.failure ? { reason: result.failure.reason } : {}),
  };
}

const ROLE = {
  $any: [{ model: "small" }, { model: "large", reasoning: { effort: "high" } }, { model: "mid" }],
  $pick: ".any[.any.map(model_limits).map('remaining').indexOf(max)]",
};

const prompt = (extra: Record<string, unknown> = {}): Record<string, unknown> => ({ outputs: {}, operation: { prompt: "go", ...extra } });

describe("what is chosen", () => {
  it("is the FIRST alternative when nothing says how to choose", async () => {
    const ran = await run({ root: prompt({ model: { $any: ["first", "second"] } }) }, {});
    expect(ran.models).toEqual(["first"]);
    expect(ran.asked).toBe(0);
  });

  it("treats `null` as a VALUE — it can be listed, and it can be what is chosen", async () => {
    const ran = await run(
      {
        root: { children: { a: {}, b: {} }, sequence: ["a", "b"] },
        "root/a": { outputs: {}, operation: { function: "record", args: { remote: { $any: [null, { to: "origin" }] } } } },
        "root/b": { outputs: {}, operation: { function: "record", args: { remote: { $any: [{ to: "origin" }, null], $pick: ".any[1]" } } } },
      },
      {},
    );
    expect(ran.reason).toBeUndefined();
    expect(ran.args).toEqual([{ remote: null }, { remote: null }]);
  });

  it("is what `$pick` RETURNS — read over the alternatives as `.any`", async () => {
    const ran = await run({ root: prompt({ model: ROLE }) }, { small: 10, large: 80, mid: 30 });
    if (ran.reason !== undefined) console.log("REASON", ran.reason, JSON.stringify(ran.events.filter((e) => e.type.includes("fail") || (e as {error?: unknown}).error !== undefined)).slice(0, 1500));
    expect(ran.reason).toBeUndefined();
    expect(ran.models).toEqual(["large"]);
  });

  it("lays an alternative's other keys UNDER what the state itself wrote", async () => {
    const ran = await run({ root: prompt({ model: ROLE, temperature: 0.2 }) }, { large: 80 });
    expect(ran.configs[0]).toEqual({ model: "large", reasoning: { effort: "high" }, temperature: 0.2 });
    const own = await run({ root: prompt({ model: ROLE, reasoning: { effort: "low" } }) }, { large: 80 });
    expect(own.configs[0]).toEqual({ model: "large", reasoning: { effort: "low" } });
  });

  it("is reached through a NAME as it is written in place — a role is a names entry", async () => {
    const ran = await run(
      { root: { environment: { names: { plan: ROLE }, model: { $ref: "plan" } }, children: { a: {} } }, "root/a": prompt() },
      { small: 5, large: 1, mid: 60 },
    );
    expect(ran.models).toEqual(["mid"]);
  });

  it("refuses an alternative that would itself have to be computed", () => {
    expect(() => loadBundle({ root: prompt({ model: { $any: [{ $expr: ".inputs.m" }] } }) as StateDef }, "root")).toThrow(
      /an alternative is a literal value .* it cannot itself be computed/,
    );
  });
});

describe("WHEN it is chosen belongs to the position", () => {
  const shared = (session: (key: string) => unknown) => ({
    root: { environment: { names: { plan: ROLE }, model: { $ref: "plan" } }, children: { a: {}, b: {} }, sequence: ["a", "b"] },
    "root/a": prompt({ session: session("a") }),
    "root/b": prompt({ session: session("b") }),
  });

  it("fixes a MODEL when its session is created: a second call in the conversation does not choose again", async () => {
    const ran = await run(shared(() => ({ $ref: "draft", $in: "parent" })), { large: 80 });
    expect(ran.models).toEqual(["large", "large"]);
    // Three alternatives, asked about once each — by the first call, and not at all by the second.
    expect(ran.asked).toBe(3);
  });

  it("chooses afresh for the NEXT session", async () => {
    const ran = await run(shared((key) => `draft-${key}`), { large: 80 });
    expect(ran.asked).toBe(6);
  });

  it("journals the choice under every instance that runs on it, so each record is complete alone", async () => {
    const ran = await run(shared(() => ({ $ref: "draft", $in: "parent" })), { large: 80 });
    const settled = ran.events.filter((e) => e.type === "value.settled" && e.field === "operation.config.model");
    expect(settled.map((e) => (e as { stateId: string }).stateId)).toEqual(["root/a", "root/b"]);
    expect(new Set(settled.map((e) => JSON.stringify((e as { value: unknown }).value))).size).toBe(1);
  });

  it("reads a journaled choice back on LOAD instead of making it again", async () => {
    const chosen = { model: "small" };
    const loaded: LoadedInstance = {
      id: "i-root",
      stateId: "root",
      inputs: {},
      live: true,
      cursor: 1,
      children: [
        { id: "i-a", stateId: "root/a", childKey: "a", inputs: {}, live: false, outcome: "success", operation: { value: {} }, fields: { "operation.config.model": chosen } },
      ],
    };
    // The limits now favour `large`, and the conversation still runs on what it was started with.
    const ran = await run(shared(() => ({ $ref: "draft", $in: "parent" })), { large: 80 }, loaded);
    if (ran.reason !== undefined) console.log("REASON", ran.reason, JSON.stringify(ran.events.filter((e) => e.type.includes("fail") || (e as {error?: unknown}).error !== undefined)).slice(0, 1500));
    expect(ran.reason).toBeUndefined();
    expect(ran.models).toEqual(["small"]);
    expect(ran.asked).toBe(0);
  });

  it("chooses per INSTANCE anywhere else — an argument is read when its state is entered", async () => {
    const ran = await run(
      {
        root: { children: { a: {}, b: {} }, sequence: ["a", "b"] },
        "root/a": { outputs: {}, operation: { function: "record", args: { tier: { $any: ["x", "y"], $pick: ".any[1]" } } } },
        "root/b": { outputs: {}, operation: { function: "record", args: { tier: { $any: ["x", "y"] } } } },
      },
      {},
    );
    expect(ran.args).toEqual([{ tier: "y" }, { tier: "x" }]);
  });
});

describe("a role reached by NAME is the identity the use resolved to", () => {
  it("reads the role `$in` names, past a nearer role of the same name", async () => {
    const ran = await run(
      {
        root: { environment: { names: { plan: { $any: [{ model: "outer" }] } } }, children: { mid: {} } },
        "root/mid": { environment: { names: { plan: { $any: [{ model: "inner" }] } } }, children: { a: {}, b: {} }, sequence: ["a", "b"] },
        "root/mid/a": prompt({ model: { $ref: "plan", $in: "global" } }),
        "root/mid/b": prompt({ model: { $ref: "plan" } }),
      },
      {},
    );
    expect(ran.models).toEqual(["outer", "inner"]);
  });

  it("lays the keys beside the `$ref` over whichever alternative is chosen — that use's own override", async () => {
    const ran = await run(
      {
        root: { environment: { names: { plan: ROLE } }, children: { a: {}, b: {} }, sequence: ["a", "b"] },
        "root/a": prompt({ model: { $ref: "plan", reasoning: { effort: "low" } } }),
        "root/b": prompt({ model: { $ref: "plan" } }),
      },
      { large: 80 },
    );
    expect(ran.configs).toEqual([
      { model: "large", reasoning: { effort: "low" } },
      { model: "large", reasoning: { effort: "high" } },
    ]);
  });

  it("refuses an override when an alternative has no keys to override", () => {
    const bundle = loadBundle(
      { root: { environment: { names: { plan: { $any: ["bare-id"] } } }, ...prompt({ model: { $ref: "plan", reasoning: { effort: "low" } } }) } as StateDef },
      "root",
    );
    expect(bundle.states.root!.sessionError?.message).toMatch(/alternative 0 of 'plan' is "bare-id", which has no keys to override/);
  });

  it("refuses alternatives CONTRIBUTED from below — a role is declared whole, where it is scoped", () => {
    const bundle = loadBundle(
      {
        root: { environment: { names: { plan: ROLE } }, children: { a: {} } } as StateDef,
        "root/a": { environment: { names: { plan: { $in: "parent", $any: [{ model: "mine" }] } } }, ...prompt() } as StateDef,
      },
      "root",
    );
    expect(bundle.states["root/a"]!.sessionError?.message).toMatch(/contributes '\$any'\/'\$pick' to an enclosing scope/);
  });
});

describe("a choice is the CONVERSATION's, however it is joined", () => {
  it("does not choose again for a call that continues the session BY REF — a position in it, not another session", async () => {
    const ran = await run(
      {
        root: {
          environment: { names: { plan: ROLE }, model: { $ref: "plan" } },
          children: { a: {}, b: { inputs: { thread: ".children.a.operation.output.session" } } },
          sequence: ["a", "b"],
        },
        "root/a": prompt({ session: "draft" }),
        "root/b": { inputs: { thread: { kind: "json" } }, ...prompt({ session: { $expr: ".inputs.thread" } }) },
      },
      { large: 80 },
    );
    if (ran.reason !== undefined) console.log("REASON", ran.reason, JSON.stringify(ran.events.filter((e) => e.type.includes("fail") || (e as {error?: unknown}).error !== undefined)).slice(0, 1500));
    expect(ran.reason).toBeUndefined();
    expect(ran.models).toEqual(["large", "large"]);
    expect(ran.asked).toBe(3);
  });
});

describe("a computed value reaches the call as the binding produced it", () => {
  it("keeps a value's own `$`-keys — a response schema's `$schema` and `$defs` are JSON Schema's", async () => {
    const schema = { $schema: "https://json-schema.org/draft/2020-12/schema", $defs: { a: { type: "string" } }, type: "object" };
    const ran = await run({ root: prompt({ model: "m", responseSchema: { $binding: { json: schema } } }) }, {});
    expect(ran.configs[0]).toEqual({ model: "m", responseSchema: schema });
  });

  it("takes off only the key the engine added, when a NAME fills a typed position", async () => {
    const ran = await run(
      { root: { environment: { names: { thinking: { effort: "high" } } }, ...prompt({ model: "m", reasoning: { $ref: "thinking" } }) } },
      {},
    );
    expect(ran.configs[0]).toEqual({ model: "m", reasoning: { effort: "high" } });
  });
});
