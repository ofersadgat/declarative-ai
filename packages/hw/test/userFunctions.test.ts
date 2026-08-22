/**
 * A workflow calling a TypeScript function, end to end (SPEC §7.5).
 *
 * Every other test in this feature pins one seam. These pin the seams meeting: a state file names a
 * bare symbol, the loader resolves it through the index, reads its signature out of the TypeScript,
 * builds the operation a call binds against, and the impl runs the real code with the real values.
 *
 * The property that matters most is the one SPEC §7.5 opens with — that a callee is an operation and
 * nothing downstream learns where it came from. So the assertions are about `FunctionOp`s and bound
 * parameters, not about anything named "user function".
 */
import { describe, expect, it } from "vitest";
import type { FunctionOp, InlineFamily } from "@declarative-ai/exec";
import { loadBundle } from "../src/loader.js";
import { createSymbolIndex } from "../src/moduleIndex.js";
import { requirePathFor } from "../src/moduleLoader.js";
import { createUserFunctions, userFunctionRef, type UserFunctions } from "../src/userFunctions.js";
import type { Vfs } from "../src/reference.js";

const FN = "/p/functions";

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

/** Everything a host wires once at startup: one index, one compiler, both synchronous after. */
async function harness(files: Record<string, string>) {
  const vfs = vfsOf(files);
  const symbols = await createSymbolIndex({ vfs });
  const userFunctions = await createUserFunctions({ vfs, requirePath: requirePathFor([FN]) });
  return { vfs, symbols, userFunctions };
}

const load = (
  states: Record<string, unknown>,
  h: { vfs: Vfs; symbols: Awaited<ReturnType<typeof createSymbolIndex>>; userFunctions: UserFunctions },
) =>
  loadBundle(states, "root", {
    defaultRoot: FN,
    vfs: h.vfs,
    symbols: h.symbols,
    userFunctions: h.userFunctions,
    documentCache: new Map(),
  });

/** The one operation a loaded state carries. */
const opOf = (bundle: ReturnType<typeof loadBundle>, id = "root"): FunctionOp<InlineFamily> =>
  bundle.states[id]!.operation as FunctionOp<InlineFamily>;

/**
 * The operation a lowered `{ expr }` output binding CALLS.
 *
 * A module symbol is named by an EXPRESSION — `confidence(.inputs.rank)` — rather than by
 * `operation.function`, which per §7.1 is a name in `registry.functions` and resolves nowhere near
 * the search path. So the callee lands in the binding's producer edge, not in `state.operation`.
 */
const calleeOf = (bundle: ReturnType<typeof loadBundle>, output: string, id = "root"): FunctionOp<InlineFamily> =>
  (bundle.states[id]!.outputs![output]!.binding as { op: FunctionOp<InlineFamily> }).op;

describe("a state calling a module symbol by bare name", () => {
  const files = {
    [`${FN}/lib.ts`]: [
      "/**",
      " * @param rank blocker=3 … note=0.",
      " */",
      "export function confidence(rank: number, iteration: number, maxIterations = 3): number {",
      "  return Math.max(0, 1 - 0.35 * (rank / 3) - 0.25 * (iteration / maxIterations));",
      "}",
    ].join("\n"),
  };

  const state = {
    root: {
      inputs: { rank: { schema: { type: "number" } }, iteration: { schema: { type: "number" } } },
      outputs: { score: { binding: { expr: "confidence(.inputs.rank, .inputs.iteration)" } } },
    },
  };

  it("resolves the callee and reads its signature out of the TypeScript", async () => {
    const h = await harness(files);
    const op = calleeOf(load(state, h), "score");

    // A plain `FunctionOp`, indistinguishable at this layer from a built-in or a host function.
    expect(op.kind).toBe("function");
    expect(op.functionRef).toBe(userFunctionRef(`${FN}/lib.ts`, ["confidence"]));

    // The slots came from the parameter list: names, positions, types, and the default.
    expect(op.input.rank).toMatchObject({ index: 0, schema: { type: "number" } });
    expect(op.input.iteration).toMatchObject({ index: 1, schema: { type: "number" } });
    expect(op.input.maxIterations).toMatchObject({ index: 2, binding: { json: 3 } });
    expect(op.output).toMatchObject({ schema: { type: "number" } });
  });

  it("binds the call's POSITIONAL arguments against those slots", async () => {
    // The whole reason `index` is read off the parameter list: nothing in the state file says which
    // argument is which, and `confidence(a, b)` still binds `rank` and `iteration` correctly.
    const bundle = load(state, await harness(files));
    const edge = bundle.states.root!.outputs!.score!.binding as { parameters?: Record<string, unknown> };
    expect(Object.keys(edge.parameters ?? {})).toEqual(["rank", "iteration"]);
  });

  it("carries the JSDoc through to the resolved signature", async () => {
    const h = await harness(files);
    const resolved = h.userFunctions.operationFor(`${FN}/lib.ts`, ["confidence"]);
    expect(resolved.signature.parameters[0]?.description).toContain("blocker=3");
  });

  it("RUNS it, with the arguments the caller supplied", async () => {
    const h = await harness(files);
    load(state, h);
    await h.userFunctions.prepare();

    const impl = h.userFunctions.impls.get(userFunctionRef(`${FN}/lib.ts`, ["confidence"]))!;
    expect(await impl({ rank: 0, iteration: 0 })).toBeCloseTo(1);
    expect(await impl({ rank: 3, iteration: 3, maxIterations: 3 })).toBeCloseTo(0.4);
  });

  it("applies a parameter default when the caller omits the slot", async () => {
    const h = await harness(files);
    h.userFunctions.operationFor(`${FN}/lib.ts`, ["confidence"]);
    await h.userFunctions.prepare();
    const impl = h.userFunctions.impls.get(userFunctionRef(`${FN}/lib.ts`, ["confidence"]))!;
    // `maxIterations` omitted — the default read off the parameter list stands in.
    expect(await impl({ rank: 3, iteration: 3 })).toBeCloseTo(0.4);
  });
});

describe("an embedded body in a state file", () => {
  it("compiles, types and runs, with slots declared the way a state declares them", async () => {
    const h = await harness({});
    const bundle = load(
      {
        root: {
          operation: {
            kind: "function",
            input: { severity: { kind: "json", schema: { type: "number" }, index: 0 } },
            body: "Math.max(0, 1 - 0.35 * severity)",
          },
          outputs: { score: { binding: ".operation.output.result" } },
        },
      },
      h,
    );
    const op = opOf(bundle);
    expect(op.kind).toBe("function");
    expect(op.input.severity).toMatchObject({ index: 0 });

    await h.userFunctions.prepare();
    const impl = h.userFunctions.impls.get(op.functionRef)!;
    expect(await impl({ severity: 2 })).toBeCloseTo(0.3);
  });

  it("runs a statement body that returns a record", async () => {
    const h = await harness({});
    const bundle = load(
      {
        root: {
          operation: {
            kind: "function",
            input: { severity: { kind: "json", schema: { type: "number" }, index: 0 } },
            body: "const high = severity >= 2;\nreturn { high, score: 1 - severity / 3 };",
          },
          outputs: { verdict: { binding: ".operation.output.result" } },
        },
      },
      h,
    );
    await h.userFunctions.prepare();
    const impl = h.userFunctions.impls.get(opOf(bundle).functionRef)!;
    expect(await impl({ severity: 2 })).toMatchObject({ high: true });
  });

  it("refuses a document that declares both a body and a function", async () => {
    const h = await harness({});
    // Exactly one thing says what an operation runs.
    const bundle = load(
      { root: { operation: { kind: "function", function: "x", body: "1" }, outputs: {} } },
      h,
    );
    expect(bundle.states.root?.operationError).toMatch(/both a 'body' and a 'function'/);
  });
});

describe("marshalling at the call boundary", () => {
  it("converts a Date in and back out again", async () => {
    const files = {
      [`${FN}/dates.ts`]: "export function dayAfter(at: Date): Date { return new Date(at.getTime() + 86400000); }",
    };
    const h = await harness(files);
    const resolved = h.userFunctions.operationFor(`${FN}/dates.ts`, ["dayAfter"]);
    // The WIRE type is a string; the function sees a Date. That is the whole of §7.5.3.
    expect((resolved.operation as FunctionOp<InlineFamily>).input.at).toMatchObject({
      schema: { type: "string", format: "date-time" },
    });

    await h.userFunctions.prepare();
    const impl = h.userFunctions.impls.get(resolved.ref)!;
    expect(await impl({ at: "2026-08-21T00:00:00.000Z" })).toBe("2026-08-22T00:00:00.000Z");
  });
});

describe("what the loader refuses", () => {
  it("reports a module callee when it was given no way to read a signature", async () => {
    const files = { [`${FN}/lib.ts`]: "export function helper(n: number) { return n; }" };
    const vfs = vfsOf(files);
    const symbols = await createSymbolIndex({ vfs });
    // Index but no `userFunctions`: the symbol is FOUND and cannot be typed, so the message says
    // that rather than "not an operation document" about a perfectly good file.
    expect(() =>
      loadBundle({ root: { outputs: { n: { binding: { expr: "helper(1)" } } } } }, "root", {
        defaultRoot: FN,
        vfs,
        symbols,
      }),
    ).toThrow(/no way to read a js\/ts signature/);
  });

  it("reports a body when it was given no way to compile one", async () => {
    const bundle = loadBundle(
      { root: { operation: { kind: "function", body: "1" }, outputs: {} } },
      "root",
      { defaultRoot: FN, vfs: vfsOf({}) },
    );
    expect(bundle.states.root?.operationError).toMatch(/no way to compile js\/ts/);
  });

  it("surfaces an unrepresentable parameter type as a load failure", async () => {
    const files = { [`${FN}/bad.ts`]: "export function weigh(n: bigint) { return Number(n); }" };
    const h = await harness(files);
    expect(() => load({ root: { outputs: { n: { binding: { expr: "weigh(1)" } } } } }, h)).toThrow(/bigint/);
  });
});

describe("resolution is not execution", () => {
  it("loads a workflow without running a line of the module", async () => {
    // The gap SPEC §7.5.5's approval gate lives in: a workflow can be loaded, type-checked and
    // validated before anybody has agreed to run any of it.
    const files = {
      [`${FN}/effect.ts`]: "globalThis.__loadRan = true;\nexport function go(n: number) { return n; }",
    };
    (globalThis as Record<string, unknown>).__loadRan = false;
    const h = await harness(files);
    load({ root: { outputs: { n: { binding: { expr: "go(1)" } } } } }, h);
    expect((globalThis as Record<string, unknown>).__loadRan).toBe(false);

    await h.userFunctions.prepare();
    expect((globalThis as Record<string, unknown>).__loadRan).toBe(false);

    const impl = h.userFunctions.impls.get(userFunctionRef(`${FN}/effect.ts`, ["go"]))!;
    await impl({ n: 1 });
    expect((globalThis as Record<string, unknown>).__loadRan).toBe(true);
  });
});
