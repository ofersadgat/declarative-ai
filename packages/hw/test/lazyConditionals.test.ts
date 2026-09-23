/**
 * A call runs only if evaluation REACHES it (SPEC §6).
 *
 * `?:` evaluates one branch, `&&` its right side only past a truthy left, `||` only past a falsy one,
 * and `coalesce(a, b)` — the language's `??` — its fallback only when `a` is absent. The resolver was
 * always lazy about the VALUE; what these pin is that the engine is lazy about what it RUNS. It used
 * to walk every named call in a wire, an output or an operation's argument and run them all before
 * resolving any, so `.inputs.risky ? ask(.inputs.req) : 'allow'` asked every time — for a call that
 * shows a person a question, a question they should never have seen.
 *
 * Every position a binding can sit in is exercised, with a SYNC callee (a pure function), an ASYNC
 * one (a host function that awaits) and a prompt callee, and the untaken call is checked three ways:
 * its implementation never ran, the answer store was never asked about it (so no dispatch site was
 * minted for it), and nothing the run journaled names it.
 */
import { describe, expect, it } from "vitest";
import { hostFunction, pureFunction, type ExecServices, type FunctionInputs, type ResolvedValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { evaluateExpression } from "../src/expr.js";
import { FakePromptExecutor, newRegistry, ok } from "./fakes.js";
import type { Vfs } from "../src/reference.js";
import { InMemoryPersistence, type WorkflowMetrics } from "../src/ports.js";

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

const oneText = (fn: string) => JSON.stringify({ kind: "function", function: fn, input: { text: { kind: "text", index: 0 } } });

const files = {
  [`${WF}/plan.json`]: "{}",
  // ASYNC: a host function that awaits before it answers — the approval prompt's shape.
  [`${FUNCTIONS}/ask.json`]: oneText("ask"),
  // SYNC: a pure function, answered in the same tick.
  [`${FUNCTIONS}/askSync.json`]: oneText("askSync"),
  [`${FUNCTIONS}/shout.json`]: oneText("shout"),
  // A PROMPT callee: a model call.
  [`${FUNCTIONS}/judge.json`]: JSON.stringify({
    kind: "prompt",
    prompt: "Judge: {{.inputs.text}}",
    model: "judge",
    input: { text: { kind: "text", index: 0 } },
  }),
};

/** Every call each function actually received, by its argument. */
type Calls = Record<string, string[]>;

function registryFor(calls: Calls) {
  const registry = newRegistry();
  const log = (fn: string, inputs: FunctionInputs): string => {
    const text = String(inputs.text ?? "");
    (calls[fn] ??= []).push(text);
    return text;
  };
  registry.functions.set(
    "ask",
    hostFunction<ExecServices, WorkflowMetrics>(
      async (inputs: FunctionInputs) => {
        const text = log("ask", inputs);
        await new Promise((r) => setTimeout(r, 1));
        return { value: `asked:${text}` as ResolvedValue };
      },
      { interactive: true, readOnly: true, memoizable: false },
    ),
  );
  registry.functions.set(
    "askSync",
    pureFunction<WorkflowMetrics>((inputs: FunctionInputs) => ({ value: `sync:${log("askSync", inputs)}` as ResolvedValue })),
  );
  registry.functions.set(
    "shout",
    hostFunction<ExecServices, WorkflowMetrics>(async (inputs: FunctionInputs) => ({ value: log("shout", inputs).toUpperCase() as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  registry.functions.set(
    "noop",
    hostFunction<ExecServices, WorkflowMetrics>(async () => ({ value: {} as ResolvedValue }), { interactive: false, readOnly: true, memoizable: true }),
  );
  registry.functions.set(
    "echo",
    hostFunction<ExecServices, WorkflowMetrics>(async (inputs: FunctionInputs) => ({ value: { out: inputs.v ?? null } as ResolvedValue }), {
      interactive: false,
      readOnly: true,
      memoizable: true,
    }),
  );
  return registry;
}

function load(defs: Record<string, unknown>) {
  return loadBundle(defs, "plan", { defaultRoot: [WF, FUNCTIONS], roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs: vfsOf(files) });
}

async function runDefs(defs: Record<string, unknown>, inputs: Record<string, ResolvedValue>) {
  const calls: Calls = {};
  const asked: string[] = [];
  const persistence = new InMemoryPersistence();
  const prompt = new FakePromptExecutor((call) => {
    const text = (call.op.user ?? "").replace(/^[\s\S]*Judge: /, "");
    (calls.judge ??= []).push(text);
    return ok(`judged:${text}`);
  });
  const bundle = load(defs);
  const engine = new WorkflowEngine({
    bundle,
    registry: registryFor(calls),
    prompt,
    validator: new SchemaValidator(),
    persistence,
    answers: (sid) => {
      asked.push(sid);
      return undefined;
    },
  });
  const result = await engine.run({ inputs });
  return { result, calls, asked, journal: JSON.stringify(persistence.events.map((e) => e.event)), bundle };
}

/** A one-state workflow whose OUTPUT is the expression. */
const withOutput = (expr: string) => ({
  "plan.json": {
    inputs: {
      risky: { kind: "json", schema: { type: "boolean" } },
      req: { kind: "text", schema: { type: "string" } },
      given: { kind: "json", optional: true },
    },
    outputs: { decision: { binding: { $expr: expr } } },
    operation: { kind: "function", function: "noop" },
  },
});

async function outputOf(expr: string, inputs: Record<string, ResolvedValue>) {
  return runDefs(withOutput(expr), { req: "rm -rf build", ...inputs });
}

describe("?: runs only the branch it takes", () => {
  for (const callee of ["ask", "askSync", "judge"] as const) {
    const expected = callee === "ask" ? "asked:rm -rf build" : callee === "askSync" ? "sync:rm -rf build" : "judged:rm -rf build";

    it(`never calls ${callee} in the branch not taken`, async () => {
      const { result, calls, asked, journal } = await outputOf(`.inputs.risky ? ${callee}(.inputs.req) : 'allow'`, { risky: false });
      expect(result.outcome).toBe("success");
      expect(result.outputs?.decision).toBe("allow");
      expect(calls[callee]).toBeUndefined();
      // Never asked of the answer store — no dispatch site was minted for a call not made.
      expect(asked).toEqual([]);
      expect(journal).not.toContain(`"${callee}"`);
    });

    it(`calls ${callee} once in the branch taken`, async () => {
      const { result, calls, asked } = await outputOf(`.inputs.risky ? ${callee}(.inputs.req) : 'allow'`, { risky: true });
      expect(result.outcome).toBe("success");
      expect(result.outputs?.decision).toBe(expected);
      expect(calls[callee]).toEqual(["rm -rf build"]);
      expect(asked).toHaveLength(1);
    });
  }

  it("runs only the taken side when both branches are calls", async () => {
    const yes = await outputOf(".inputs.risky ? ask('yes') : askSync('no')", { risky: true });
    expect(yes.result.outputs?.decision).toBe("asked:yes");
    expect(yes.calls).toEqual({ ask: ["yes"] });
    const no = await outputOf(".inputs.risky ? ask('yes') : askSync('no')", { risky: false });
    expect(no.result.outputs?.decision).toBe("sync:no");
    expect(no.calls).toEqual({ askSync: ["no"] });
  });

  it("decides on a test that is itself a call, then runs only the branch it chose", async () => {
    const { result, calls } = await outputOf("shout(.inputs.req) === 'RM -RF BUILD' ? ask('a') : ask('b')", { risky: false });
    expect(result.outputs?.decision).toBe("asked:a");
    expect(calls).toEqual({ shout: ["rm -rf build"], ask: ["a"] });
  });
});

describe("nested conditionals run exactly one leaf", () => {
  const expr = ".inputs.risky ? (.inputs.req === 'x' ? ask('inner-then') : askSync('inner-else')) : judge('outer-else')";

  it("takes the outer then and the inner else", async () => {
    const { result, calls } = await outputOf(expr, { risky: true, req: "y" });
    expect(result.outputs?.decision).toBe("sync:inner-else");
    expect(calls).toEqual({ askSync: ["inner-else"] });
  });

  it("takes the outer then and the inner then", async () => {
    const { result, calls } = await outputOf(expr, { risky: true, req: "x" });
    expect(result.outputs?.decision).toBe("asked:inner-then");
    expect(calls).toEqual({ ask: ["inner-then"] });
  });

  it("takes the outer else and never reaches the inner test", async () => {
    const { result, calls } = await outputOf(expr, { risky: false, req: "x" });
    expect(result.outputs?.decision).toBe("judged:outer-else");
    expect(calls).toEqual({ judge: ["outer-else"] });
  });

  it("nests in the test position too", async () => {
    const { result, calls } = await outputOf("(.inputs.risky ? ask('t') : false) ? askSync('then') : askSync('else')", { risky: false });
    expect(result.outputs?.decision).toBe("sync:else");
    expect(calls).toEqual({ askSync: ["else"] });
  });
});

describe("&&, || and coalesce (the language's ??) run their right side only when needed", () => {
  for (const callee of ["ask", "askSync"] as const) {
    it(`&& skips ${callee} past a false left, and calls it past a true one`, async () => {
      const skipped = await outputOf(`.inputs.risky && ${callee}(.inputs.req)`, { risky: false });
      expect(skipped.result.outputs?.decision).toBe(false);
      expect(skipped.calls).toEqual({});
      expect(skipped.asked).toEqual([]);
      const taken = await outputOf(`.inputs.risky && ${callee}(.inputs.req)`, { risky: true });
      expect(taken.calls[callee]).toEqual(["rm -rf build"]);
    });

    it(`|| skips ${callee} past a true left, and calls it past a false one`, async () => {
      const skipped = await outputOf(`.inputs.risky || ${callee}(.inputs.req)`, { risky: true });
      expect(skipped.result.outputs?.decision).toBe(true);
      expect(skipped.calls).toEqual({});
      expect(skipped.asked).toEqual([]);
      const taken = await outputOf(`.inputs.risky || ${callee}(.inputs.req)`, { risky: false });
      expect(taken.calls[callee]).toEqual(["rm -rf build"]);
    });

    it(`coalesce skips ${callee} when its first value is there, and calls it when it is not`, async () => {
      const skipped = await outputOf(`coalesce(.inputs.given, ${callee}(.inputs.req))`, { risky: false, given: "allow" });
      expect(skipped.result.outputs?.decision).toBe("allow");
      expect(skipped.calls).toEqual({});
      expect(skipped.asked).toEqual([]);
      const taken = await outputOf(`coalesce(.inputs.given, ${callee}(.inputs.req))`, { risky: false });
      expect(taken.calls[callee]).toEqual(["rm -rf build"]);
    });
  }

  it("a chain stops at the first operand that decides it", async () => {
    const { result, calls } = await outputOf("askSync('first') === 'sync:first' || ask('second') || ask('third')", { risky: false });
    expect(result.outputs?.decision).toBe(true);
    expect(calls).toEqual({ askSync: ["first"] });
  });

  it("coalesce is lazy in the reference interpreter as well", () => {
    // The interpreter refuses to run a non-built-in, so reaching one would throw; not reaching it is
    // the laziness. A present `a` answers before `b` is looked at.
    expect(evaluateExpression("coalesce(.x, missing(.y))", { x: 1 })).toBe(1);
    expect(evaluateExpression("coalesce(.x, 2)", { x: null })).toBe(2);
    expect(() => evaluateExpression("coalesce(.x, missing(.y))", { x: null })).toThrow(/only the lowered form can run/);
  });
});

describe("a higher-order application in a branch not taken is not run", () => {
  it("skips a map in the untaken branch and runs it in the taken one", async () => {
    const defs = {
      "plan.json": {
        inputs: { go: { kind: "json", schema: { type: "boolean" } }, xs: { kind: "json", schema: { type: "array", items: { type: "string" } } } },
        outputs: { out: { binding: { $expr: ".inputs.go ? map(.inputs.xs, shout) : 'none'" } } },
        operation: { kind: "function", function: "noop" },
      },
    };
    const skipped = await runDefs(defs, { go: false, xs: ["a", "b"] });
    expect(skipped.result.outputs?.out).toBe("none");
    expect(skipped.calls).toEqual({});
    const taken = await runDefs(defs, { go: true, xs: ["a", "b"] });
    expect(taken.result.outputs?.out).toEqual(["A", "B"]);
    expect(taken.calls.shout?.sort()).toEqual(["a", "b"]);
  });
});

describe("every binding position is lazy", () => {
  it("an operation's argument", async () => {
    const defs = (risky: boolean) => ({
      "plan.json": {
        inputs: { req: { kind: "text", schema: { type: "string" } } },
        outputs: { out: { binding: ".operation.output.out" } },
        operation: {
          kind: "function",
          function: "echo",
          input: { v: { binding: { $expr: `${risky} ? ask(.inputs.req) : 'allow'` } } },
          output: { out: {} },
        },
      },
    });
    const skipped = await runDefs(defs(false), { req: "r" });
    expect(skipped.result.failure).toBeUndefined();
    expect(skipped.result.outputs?.out).toBe("allow");
    expect(skipped.calls).toEqual({});
    const taken = await runDefs(defs(true), { req: "r" });
    expect(taken.result.outputs?.out).toBe("asked:r");
    expect(taken.calls).toEqual({ ask: ["r"] });
  });

  it("a child mount's wire", async () => {
    const defs = (risky: boolean) => ({
      "plan.json": {
        inputs: { req: { kind: "text", schema: { type: "string" } } },
        children: { c: { state: "./c", inputs: { v: { $expr: `${risky} ? ask(.inputs.req) : 'allow'` } } } },
        outputs: { out: { binding: ".children.c.output.v" } },
      },
      "plan/c.json": {
        inputs: { v: { kind: "json" } },
        outputs: { v: { binding: ".inputs.v" } },
        operation: { kind: "function", function: "noop" },
      },
    });
    const skipped = await runDefs(defs(false), { req: "r" });
    expect(skipped.result.failure).toBeUndefined();
    expect(skipped.result.outputs?.out).toBe("allow");
    expect(skipped.calls).toEqual({});
    const taken = await runDefs(defs(true), { req: "r" });
    expect(taken.result.outputs?.out).toBe("asked:r");
    expect(taken.calls).toEqual({ ask: ["r"] });
  });

  it("a guard", async () => {
    const defs = (risky: boolean) => ({
      "plan.json": {
        inputs: { req: { kind: "text", schema: { type: "string" } } },
        outputs: { done: { binding: { text: "ok" } } },
        operation: { kind: "function", function: "noop" },
        transitions: [{ to: "terminate.success", when: `${risky} ? ask(.inputs.req) === 'asked:r' : true` }],
      },
    });
    const skipped = await runDefs(defs(false), { req: "r" });
    expect(skipped.result.outcome).toBe("success");
    expect(skipped.calls).toEqual({});
    const taken = await runDefs(defs(true), { req: "r" });
    expect(taken.result.outcome).toBe("success");
    expect(taken.calls).toEqual({ ask: ["r"] });
  });
});

describe("static checks still read both branches", () => {
  it("reports an unregistered callee in the branch a run would not take", () => {
    // `ghost` is a function document whose implementation nobody registered. A run with `risky`
    // false would never reach it, and the validator must say so anyway.
    const ghostFiles = { ...files, [`${FUNCTIONS}/ghost.json`]: oneText("ghost") };
    const bundle = loadBundle(
      {
        "plan.json": {
          inputs: { risky: { kind: "json", schema: { type: "boolean" } }, req: { kind: "text", schema: { type: "string" } } },
          outputs: { decision: { binding: { $expr: "false ? ghost(.inputs.req) : 'allow'" } } },
          operation: { kind: "function", function: "noop" },
        },
      },
      "plan",
      { defaultRoot: [WF, FUNCTIONS], roots: { JAIRA: ROOT, PROJECT: "/p" }, vfs: vfsOf(ghostFiles) },
    );
    const registry = registryFor({});
    const messages = validateBundle(bundle, { functions: registry.functions as never, strict: true }).errors.map((e) => e.message).join("\n");
    expect(messages).toMatch(/ghost/);
  });

  it("refuses at load a name that resolves to nothing, in either branch", () => {
    expect(() => load(withOutput(".inputs.risky ? nowhere(.inputs.req) : 'allow'"))).toThrow(/nowhere/);
    expect(() => load(withOutput(".inputs.risky ? 'allow' : nowhere(.inputs.req)"))).toThrow(/nowhere/);
  });

  it("reads a child reference in the branch not taken", () => {
    const bundle = load({
      "plan.json": {
        inputs: { risky: { kind: "json", schema: { type: "boolean" } } },
        outputs: { decision: { binding: { $expr: ".inputs.risky ? 'allow' : .children.nobody.output.x" } } },
        operation: { kind: "function", function: "noop" },
      },
    });
    expect(validateBundle(bundle).errors.map((e) => e.message).join("\n")).toMatch(/nobody/);
  });
});
