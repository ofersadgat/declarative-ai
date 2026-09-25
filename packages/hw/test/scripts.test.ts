/**
 * Workflow scripts (SCRIPTS.md): a state machine written as code, compiled to hw states.
 *
 * What is pinned here is what the compile PROMISES — the documents it makes are ordinary states that
 * load and validate like authored ones, each `phase()` is a child state, control flow becomes the
 * mount's rules, a variable crosses a phase as a wire — and what running them does: `llm()` calls
 * dispatched and journaled at sites of their own, `llm<T>` typed from `T`, Claude's `agent()`,
 * `parallel` and `pipeline` semantics, and a re-run of a phase's code answered from its record.
 */
import { describe, expect, it } from "vitest";
import { MapSessionStore, withRecord, withSessionPosition, type JsonValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { validateBundle } from "../src/validate.js";
import { compileScript, ScriptCompileError, type CompiledScript } from "../src/scriptCompile.js";
import type { StateDef } from "../src/format.js";
import type { LoadedInstance } from "../src/load.js";
import { InMemoryPersistence, type EngineEvent } from "../src/ports.js";
import { FakePromptExecutor, newRegistry, ok, promptOf, type FakeCall, type Script } from "./fakes.js";

const ROOT = "/wf";

function vfsOf(files: Record<string, string>) {
  return {
    read: (p: string) => files[p],
    list: (dir: string) => Object.keys(files).filter((f) => f.startsWith(`${dir}/`)).map((f) => f.slice(dir.length + 1).split("/")[0]!),
  };
}

async function compile(source: string, extra: Record<string, string> = {}, stateId = "s"): Promise<CompiledScript> {
  const file = `${ROOT}/${stateId}.ts`;
  const files = { [file]: source, ...extra };
  return compileScript({ file, stateId, vfs: vfsOf(files), requirePath: [ROOT] });
}

async function run(source: string, inputs: Record<string, JsonValue>, script: Script, options: { extra?: Record<string, string>; loaded?: LoadedInstance } = {}) {
  const extra = options.extra ?? {};
  const compiled = await compile(source, extra);
  // The JSON states beside the script are the host's files, loaded with the compiled ones.
  const states: Record<string, StateDef> = { ...(compiled.documents as Record<string, StateDef>) };
  for (const [path, text] of Object.entries(extra)) {
    if (path.endsWith(".json")) states[path.slice(ROOT.length + 1, -".json".length)] = JSON.parse(text) as StateDef;
  }
  const bundle = loadBundle(states, "s");
  const fake = new FakePromptExecutor(script);
  const sessions = new MapSessionStore();
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle,
    registry: newRegistry(),
    prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
    sessions,
    validator: new SchemaValidator(),
    persistence,
    scripts: { vfs: vfsOf({ [`${ROOT}/s.ts`]: source, ...extra }), requirePath: [ROOT] },
  } as never);
  const result = options.loaded !== undefined ? await engine.loadRun(options.loaded) : await engine.run({ inputs });
  const events = persistence.events.map(({ event }) => event);
  return { result, calls: fake.calls, events, compiled, bundle };
}

/** The prompt text a script's call sent — bound as an input, so read off the rendered op. */
function textOf(call: FakeCall): string {
  return promptOf(call);
}

const entered = (events: EngineEvent[]) => events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered").map((e) => e.stateId);

const SWEEP = `
export const meta = {
  name: "Sweep review",
  description: "Find until a round finds nothing new, verify each, triage.",
};

type Finding = { id: string; claim: string };

export default async function sweep(ref: string, dimensions: string[] = ["bugs", "perf"]): Promise<{ confirmed: Finding[] }> {
  let seen: string[] = [];
  let confirmed: Finding[] = [];
  for (let round = 0; round < 3; round++) {
    phase("Find");
    const found = await parallel(dimensions.map((d) => () =>
      llm<{ findings: Finding[] }>({ prompt: \`Review \${ref} for \${d}. Skip: \${seen.join(",")}\` })));
    const fresh = found.filter((r) => r !== null).flatMap((r) => r!.findings).filter((f) => !seen.includes(f.id));
    if (fresh.length === 0) break;
    seen = [...seen, ...fresh.map((f) => f.id)];

    phase("Verify");
    const verdicts = await parallel(fresh.map((f) => () => llm<{ refuted: boolean }>(\`Refute: \${f.claim}\`)));
    confirmed = [...confirmed, ...fresh.filter((_, i) => verdicts[i] !== null && !verdicts[i]!.refuted)];
  }
  phase("Triage");
  log(\`confirmed \${confirmed.length}\`);
  return { confirmed };
}
`;

/** The sweep's model: round one finds two bugs and one perf issue, round two nothing new; b2 is refuted. */
const sweepModel: Script = (call) => {
  const p = textOf(call);
  if (p.startsWith("Review")) {
    const skip = p.slice(p.indexOf("Skip:") + 5).split(",").filter(Boolean);
    const all = p.includes("for bugs") ? [{ id: "b1", claim: "null deref" }, { id: "b2", claim: "off by one" }] : [{ id: "p1", claim: "n^2 loop" }];
    return ok({ findings: all.filter((f) => !skip.includes(f.id)) });
  }
  if (p.startsWith("Refute")) return ok({ refuted: p.includes("off by one") });
  return ok("?");
};

describe("compiling a script", () => {
  it("makes the state the script IS, and one child state per phase", async () => {
    const { documents } = await compile(SWEEP);
    expect(Object.keys(documents).sort()).toEqual(["s", "s/find", "s/triage", "s/verify"]);
    const root = documents.s!;
    expect(root.label).toBe("Sweep review");
    expect(root.description).toMatch(/^Find until/);
    expect(Object.keys(root.inputs ?? {})).toEqual(["ref", "dimensions"]);
    expect(root.inputs?.dimensions).toMatchObject({ optional: true, default: ["bugs", "perf"] });
    expect(Object.keys(root.outputs ?? {})).toEqual(["confirmed"]);
    expect(root.sequence).toEqual([]);
    expect(Object.keys(root.children ?? {}).sort()).toEqual(["find", "triage", "verify"]);
    expect(documents["s/find"]!.label).toBe("Find");
    expect(root.generated?.from).toBe(`${ROOT}/s.ts`);
  });

  it("wires only the variables a phase needs — a value used inside one phase never crosses", async () => {
    const { documents } = await compile(SWEEP);
    const find = Object.keys(documents["s/find"]!.inputs ?? {}).sort();
    expect(find).toEqual(["_entry", "confirmed", "dimensions", "ref", "round", "seen"]);
    const verify = Object.keys(documents["s/verify"]!.inputs ?? {}).sort();
    expect(verify).toEqual(["_entry", "confirmed", "dimensions", "fresh", "ref", "round", "seen"]);
    const triage = Object.keys(documents["s/triage"]!.inputs ?? {}).sort();
    expect(triage).toEqual(["_entry", "confirmed"]);
  });

  it("types a crossing variable from its TypeScript type", async () => {
    const { documents } = await compile(SWEEP);
    expect(documents["s/find"]!.inputs?.seen?.schema).toEqual({ type: "array", items: { type: "string" } });
  });

  it("produces documents that load and validate like authored ones", async () => {
    const { documents } = await compile(SWEEP);
    const bundle = loadBundle(documents as Record<string, StateDef>, "s");
    const report = validateBundle(bundle, {} as never) as { errors: unknown[] };
    expect(report.errors).toEqual([]);
  });

  it("refuses meta keys that describe structure, naming where the fact comes from", async () => {
    await expect(compile(`export const meta = { inputs: {} };\nexport default async function f() {}`)).rejects.toThrow(/meta.inputs is not a script property — a script's inputs are its exported function's parameters/);
    await expect(compile(`export const meta = { name: "a", label: "b" };\nreturn 1;`)).rejects.toThrow(/both `name` and `label`/);
  });

  it("refuses a meta that is not a pure literal", async () => {
    await expect(compile(`const x = "a";\nexport const meta = { name: "a" };`)).rejects.toThrow(/must be the first statement/);
    await expect(compile(`export const meta = { name: \`a\${1}\` };\nreturn 1;`)).rejects.toThrow(/pure literal/);
  });

  it("refuses a phase() the compiler cannot cut at", async () => {
    await expect(compile(`export default async function f(xs: string[]) { await parallel(xs.map(() => async () => { phase("A"); })); }`)).rejects.toThrow(/phase\(\) inside a function or a callback/);
    await expect(compile(`export default async function f(n: string) { phase(n); }`)).rejects.toThrow(/one string literal/);
  });

  it("refuses a variable with no wire form that lives across a phase", async () => {
    await expect(
      compile(`export default async function f() { const m = new Map<string, number>(); phase("A"); phase("B"); return m.size; }`),
    ).rejects.toThrow(/'m' lives across the phase 'A'.*Map/);
  });

  it("reports a compile error at the script's line", async () => {
    const error = await compile(`export const meta = {};\n\nexport default async function f(n: string) {\n  phase(n);\n}`).catch((e) => e);
    expect(error).toBeInstanceOf(ScriptCompileError);
    expect(error.message).toMatch(/s\.ts:4:3:/);
  });
});

describe("running a compiled script", () => {
  it("runs the phases as child states and returns what the function returned", async () => {
    const { result, events } = await run(SWEEP, { ref: "main" }, sweepModel);
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ confirmed: [{ id: "b1", claim: "null deref" }, { id: "p1", claim: "n^2 loop" }] });
    expect(entered(events)).toEqual(["s", "s/find", "s/verify", "s/find", "s/triage"]);
  });

  it("journals every call at a site of its own, and every log line", async () => {
    const { events } = await run(SWEEP, { ref: "main" }, sweepModel);
    const settled = events.filter((e): e is Extract<EngineEvent, { type: "script.call.settled" }> => e.type === "script.call.settled");
    // round 1: two reviews + three verifications; round 2: two reviews.
    expect(settled).toHaveLength(7);
    expect(new Set(settled.map((e) => `${e.instanceId}:${e.site}`)).size).toBe(7);
    const logs = events.filter((e) => e.type === "script.log");
    expect(logs).toMatchObject([{ message: "confirmed 2" }]);
  });

  it("asks the model for T's shape — llm<T> is the output contract", async () => {
    const { calls } = await run(SWEEP, { ref: "main" }, sweepModel);
    const refute = calls.find((c) => textOf(c).startsWith("Refute"))!;
    expect(refute.op.output.schema).toMatchObject({ type: "object", properties: { refuted: { type: "boolean" } } });
  });

  it("sends a prompt verbatim — braces in it are text, not a template", async () => {
    const source = `export default async function f() { return await llm("Return {{.inputs.x}} as JSON: { \\"a\\": 1 }"); }`;
    const { calls, result } = await run(source, {}, () => ok("fine"));
    expect(textOf(calls[0]!)).toBe('Return {{.inputs.x}} as JSON: { "a": 1 }');
    expect(result.outputs).toEqual({ result: "fine" });
  });

  it("wraps a non-object T for the wire and hands the script the plain value", async () => {
    const source = `export default async function f(command: string): Promise<boolean> { return await llm<boolean>({ prompt: "Is this destructive? " + command }); }`;
    const { calls, result } = await run(source, { command: "rm -rf /" }, () => ok({ value: true }));
    expect(calls[0]!.op.output.schema).toMatchObject({ type: "object", properties: { value: { type: "boolean" } } });
    expect(result.outputs).toEqual({ result: true });
  });

  it("types llm<T> from a named, imported type", async () => {
    const source = `import type { Person } from "./people";\nexport default async function f(bio: string) { return await llm<Person>("Who is this? " + bio); }`;
    const people = `export interface Person { name: string; /** In years. */ age: number }`;
    const { calls } = await run(source, { bio: "x" }, () => ok({ name: "Ada", age: 36 }), { extra: { [`${ROOT}/people.ts`]: people } });
    expect(calls[0]!.op.output.schema).toMatchObject({ type: "object", properties: { name: { type: "string" }, age: { type: "number" } }, required: ["name", "age"] });
  });

  it("gives agent() Claude's null on failure, and llm() a throw that fails the state", async () => {
    const agentSource = `export default async function f() { const a = await agent("try"); return { got: a === null ? "null" : "value" }; }`;
    const fail: Script = () => ({ error: { classification: "permanent", reason: "nope" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } }) as never;
    const agentRun = await run(agentSource, {}, fail);
    expect(agentRun.result.outcome).toBe("success");
    expect(agentRun.result.outputs).toEqual({ got: "null" });
    const llmRun = await run(`export default async function f() { return await llm("try"); }`, {}, fail);
    expect(llmRun.result.outcome).toBe("error");
  });

  it("runs a Claude script unchanged — meta, globals, args in, a top-level return out", async () => {
    const claude = `
export const meta = {
  name: 'find-flaky-tests',
  description: 'Find flaky tests and propose fixes',
  phases: [{ title: 'Scan', detail: 'grep test logs for retries' }, { title: 'Fix', detail: 'one agent per flaky test' }],
}
phase('Scan')
const flaky = await agent('grep CI logs for retry markers in ' + args.repo, { schema: { type: 'object', properties: { tests: { type: 'array', items: { type: 'string' } } }, required: ['tests'] } })
phase('Fix')
const fixes = await pipeline(flaky.tests, (t) => agent('propose a fix for ' + t, { phase: 'Fix' }))
return { fixes: fixes.filter(Boolean) }
`;
    const compiled = await compile(claude);
    expect(compiled.documents["s/scan"]?.description).toBe("grep test logs for retries");
    const model: Script = (call) => (textOf(call).startsWith("grep") ? ok({ tests: ["a", "b"] }) : ok(`fix ${textOf(call).slice(-1)}`));
    const { result, events } = await run(claude, { args: { repo: "r" } }, model);
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ result: { fixes: ["fix a", "fix b"] } });
    expect(entered(events)).toEqual(["s", "s/scan", "s/fix"]);
  });

  it("gives a body script its exports as outputs", async () => {
    const source = `const a = await llm("a");\nphase("Two");\nexport const first = a;\nexport const second = await llm("b");`;
    const { result } = await run(source, {}, (c) => ok(textOf(c).toUpperCase()));
    expect(result.outputs).toEqual({ first: "A", second: "B" });
  });

  it("runs imported code inline", async () => {
    const source = `import { shout } from "./lib";\nexport default async function f(x: string) { phase("One"); const y = shout(x); phase("Two"); return await llm(y); }`;
    const lib = `export function shout(s: string): string { return s.toUpperCase() + "!"; }`;
    const { calls } = await run(source, { x: "hi" }, () => ok("ok"), { extra: { [`${ROOT}/lib.ts`]: lib } });
    expect(textOf(calls[0]!)).toBe("HI!");
  });

  it("refuses Date.now() and Math.random() in the script's own code, and offers the recorded ones", async () => {
    const bad = await run(`export default async function f() { return Date.now(); }`, {}, () => ok(""));
    expect(bad.result.outcome).toBe("error");
    const good = await run(`import { now } from "@declarative-ai/hw/script";\nexport default async function f() { return now() > 0; }`, {}, () => ok(""));
    expect(good.result.outputs).toEqual({ result: true });
    expect(good.events.some((e) => e.type === "script.call.settled" && e.site === "now#0")).toBe(true);
  });

  it("answers a re-run of a phase's code from its record, and asks only what it had not", async () => {
    const source = `export default async function f() { const a = await llm("one"); const b = await llm("two"); return a + b; }`;
    const first = await run(source, {}, (c) => ok(textOf(c)));
    const settled = first.events.filter((e): e is Extract<EngineEvent, { type: "script.call.settled" }> => e.type === "script.call.settled");
    const root = first.events.find((e) => e.type === "instance.entered")!;
    // The run stopped after the first call: the root is live, its operation never completed.
    const loaded: LoadedInstance = {
      id: root.instanceId,
      stateId: "s",
      inputs: {},
      live: true,
      scriptCalls: [{ site: settled[0]!.site, value: "cached-one" }],
    };
    const resumed = await run(source, {}, (c) => ok(textOf(c)), { loaded });
    expect(resumed.calls.map(textOf)).toEqual(["two"]);
    expect(resumed.result.outputs).toEqual({ result: "cached-onetwo" });
  });

  it("makes three identical calls three draws — the ordinal keeps their sites apart", async () => {
    const source = `export default async function f() { const votes = await parallel([0, 1, 2].map(() => () => llm<{ ok: boolean }>("vote"))); return votes.length; }`;
    const { calls, events } = await run(source, {}, () => ok({ ok: true }));
    expect(calls).toHaveLength(3);
    const sites = events.filter((e) => e.type === "script.call.settled").map((e) => (e as { site: string }).site);
    expect(sites.map((s) => s.split("#")[1])).toEqual(["0", "1", "2"]);
  });
});

describe("calling states from a script", () => {
  const TRIAGE = JSON.stringify({
    inputs: { finding: { schema: { type: "string" } } },
    outputs: { severity: { schema: { type: "string" }, binding: { $expr: "startsWith(.inputs.finding, 'crash') ? 'high' : 'low'" } } },
  });

  it("runs an imported JSON state as a child instance and hands back its outputs", async () => {
    const source = `import triage from "./triage";\nexport default async function f(items: string[]) { const out = await Promise.all(items.map((finding) => triage({ finding }))); return { severities: out.map((o) => o.severity) }; }`;
    const { result, events, compiled } = await run(source, { items: ["crash on load", "typo"] }, () => ok(""), { extra: { [`${ROOT}/triage.json`]: TRIAGE } });
    expect(compiled.documents.s!.children).toEqual({ triage: { state: "triage", called: true } });
    expect(result.failure?.reason).toBeUndefined();
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ severities: ["high", "low"] });
    const calls = events.filter((e) => e.type === "instance.entered" && e.stateId === "triage");
    // Two calls of one state run side by side — neither supersedes the other — each marked as a call.
    expect(calls).toHaveLength(2);
    expect(calls.every((e) => (e as { calledAt?: string }).calledAt !== undefined)).toBe(true);
    expect(events.some((e) => e.type === "child.superseded")).toBe(false);
  });

  it("calls a state named in workflow() with a literal, and a script's whole return comes back whole", async () => {
    const child = `export const meta = { name: "child" };\nreturn { doubled: args.n * 2 };`;
    const source = `export const meta = { name: "parent" };\nconst r = await workflow("child", { n: 21 });\nreturn r.doubled;`;
    const childDocs = await compileScript({ file: `${ROOT}/child.ts`, stateId: "child", vfs: vfsOf({ [`${ROOT}/child.ts`]: child }), requirePath: [ROOT] });
    const parent = await compile(source);
    const bundle = loadBundle({ ...(parent.documents as Record<string, StateDef>), ...(childDocs.documents as Record<string, StateDef>) }, "s");
    const engine = new WorkflowEngine({ bundle, registry: newRegistry(), validator: new SchemaValidator(), persistence: new InMemoryPersistence() } as never);
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ result: 42 });
  });

  it("throws a called state's failure into the script, where it can be caught", async () => {
    const failing = JSON.stringify({ operation: { prompt: "explode" } });
    const source = `import bad from "./bad";\nexport default async function f() { try { await bad(); return "no"; } catch (e) { return "caught"; } }`;
    const model: Script = (c) =>
      textOf(c) === "explode" ? ({ error: { classification: "permanent", reason: "boom" }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } } as never) : ok("");
    const { result, events } = await run(source, {}, model, { extra: { [`${ROOT}/bad.json`]: failing } });
    expect(result.outputs).toEqual({ result: "caught" });
    // It ran, and failed — not a call that never reached a state.
    expect(events.some((e) => e.type === "instance.terminated" && e.stateId === "bad" && e.outcome === "error")).toBe(true);
  });

  it("refuses workflow() with a computed name", async () => {
    await expect(compile(`export default async function f(n: string) { return await workflow(n); }`)).rejects.toThrow(/names its state with a literal/);
  });
});

describe("the other modes", () => {
  it("runs a script as ONE state in 'state' mode, with phase() as Claude's display group", async () => {
    const source = `export const meta = { compile: "state" };\nphase("A");\nconst x = await llm("one");\nphase("B");\nreturn x + (await llm("two"));`;
    const { result, events, compiled } = await run(source, {}, (c) => ok(textOf(c)));
    expect(Object.keys(compiled.documents)).toEqual(["s"]);
    expect(result.outputs).toEqual({ result: "onetwo" });
    expect(events.filter((e) => e.type === "script.phase").map((e) => (e as { title: string }).title)).toEqual(["A", "B"]);
    const settled = events.filter((e) => e.type === "script.call.settled") as Array<{ phase?: string }>;
    expect(settled.map((e) => e.phase)).toEqual(["A", "B"]);
  });

  it("compiles nothing in 'function' mode — the module stays a module", async () => {
    const compiled = await compile(`export const meta = { compile: "function" };\nexport function f() { return 1; }`);
    expect(compiled.mode).toBe("function");
    expect(compiled.documents).toEqual({});
  });

  it("lets a function module call llm(), recorded under the state that called it", async () => {
    const FN = "/fn";
    const files = { [`${FN}/classify.ts`]: `import { llm } from "@declarative-ai/hw/script";\nexport default async function classify(text: string): Promise<string> { return await llm("Classify: " + text); }` };
    const vfs = vfsOf(files);
    const { createSymbolIndex } = await import("../src/moduleIndex.js");
    const { createUserFunctions } = await import("../src/userFunctions.js");
    const { requirePathFor } = await import("../src/moduleLoader.js");
    const symbols = await createSymbolIndex({ vfs });
    const userFunctions = await createUserFunctions({ vfs, requirePath: requirePathFor([FN]) });
    const bundle = loadBundle(
      { root: { operation: { function: "classify", args: { text: "a bug" } }, outputs: { label: { schema: { type: "string" }, binding: ".operation.output" } } } },
      "root",
      { defaultRoot: FN, vfs, symbols, userFunctions, documentCache: new Map() },
    );
    await userFunctions.prepare();
    const registry = newRegistry();
    for (const [ref, entry] of userFunctions.entries) registry.functions.set(ref, entry as never);
    const fake = new FakePromptExecutor((c) => ok(`label for ${textOf(c)}`));
    const persistence = new InMemoryPersistence();
    const engine = new WorkflowEngine({ bundle, registry, prompt: fake as never, validator: new SchemaValidator(), persistence } as never);
    const result = await engine.run({ inputs: {} });
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ label: "label for Classify: a bug" });
    const settled = persistence.events.map(({ event }) => event).filter((e) => e.type === "script.call.settled") as Array<{ site: string }>;
    expect(settled).toHaveLength(1);
    expect(settled[0]!.site).toMatch(/^user:.*classify\.ts#default@0\//);
  });
});

describe("scripts in a workflow directory", () => {
  it("loads a directory's scripts as the states they compile to, checks generated files against them, and refuses a stale one", async () => {
    const { mkdtemp, writeFile, mkdir, rm } = await import("node:fs/promises");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const { loadBundleFromDir } = await import("../src/loader.js");
    const { generatedFiles } = await import("../src/scriptCompile.js");
    const dir = await mkdtemp(join(tmpdir(), "hw-scripts-"));
    try {
      await mkdir(join(dir, "review"));
      await writeFile(join(dir, "review", "sweep.ts"), SWEEP);
      const bundle = await loadBundleFromDir(dir, "review/sweep");
      expect(Object.keys(bundle.states).sort()).toEqual(["review/sweep", "review/sweep/find", "review/sweep/triage", "review/sweep/verify"]);

      // The host writes the generated files beside the script; a load checks them and uses the script.
      const root = dir.split("\\").join("/");
      const compiled = await compileScript({ file: `${root}/review/sweep.ts`, stateId: "review/sweep", vfs: { read: (p) => (p === `${root}/review/sweep.ts` ? SWEEP : undefined), list: () => [] }, requirePath: [root] });
      for (const [path, text] of Object.entries(generatedFiles(compiled))) {
        await mkdir(join(dir, ...path.split("/").slice(0, -1)), { recursive: true });
        await writeFile(join(dir, ...path.split("/")), text);
      }
      await expect(loadBundleFromDir(dir, "review/sweep")).resolves.toBeDefined();

      // An edit to the script makes the files stale — the load says to regenerate, never picks one.
      await writeFile(join(dir, "review", "sweep.ts"), SWEEP.replace("Sweep review", "Sweep review 2"));
      await expect(loadBundleFromDir(dir, "review/sweep")).rejects.toThrow(/changed since this state was generated.*edit the script and regenerate/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("SCRIPTS.md §2", () => {
  it("compiles the example to the tree and rules the document shows, and it validates", async () => {
    const source = `import classify from "./triage";
import { dedupeByLocation } from "./lib/findings";
type Change = { ref: string };
type Finding = { id: string; claim: string };

export const meta = {
  name: "Sweep review",
  description: "Find issues along several dimensions until a round finds nothing new; verify; triage.",
  title: { binding: { $expr: "'Review ' + .inputs.change.ref" } },
  environment: { model: "anthropic/claude-opus-5-5" },
};

export default async function sweep(change: Change, dimensions: string[] = ["bugs", "perf"]) {
  let seen: string[] = [];
  let confirmed: Finding[] = [];
  for (let round = 0; round < 3; round++) {
    phase("Find");
    const found = await parallel(dimensions.map((d) => () =>
      agent<{ findings: Finding[] }>(\`Review \${change.ref} for \${d}. Skip anything in: \${seen.join(", ")}\`)));
    const fresh = dedupeByLocation(found.filter((r) => r !== null).flatMap((r) => r!.findings));
    if (fresh.length === 0) break;
    seen = [...seen, ...fresh.map((f) => f.id)];

    phase("Verify");
    const verdicts = await parallel(fresh.map((f) => () =>
      llm<{ refuted: boolean; reason: string }>({ prompt: \`Try to refute: \${f.claim}\`, model: "anthropic/claude-sonnet-5" })));
    confirmed = [...confirmed, ...fresh.filter((_, i) => verdicts[i] !== null && !verdicts[i]!.refuted)];
  }
  phase("Triage");
  return { confirmed: await Promise.all(confirmed.map((f) => classify({ finding: f }))) };
}
`;
    const findings = `export function dedupeByLocation<T extends { id: string }>(xs: T[]): T[] { return xs.filter((x, i) => xs.findIndex((y) => y.id === x.id) === i); }`;
    const triage = JSON.stringify({ inputs: { finding: {} }, outputs: { finding: { binding: ".inputs.finding" } } });
    const compiled = await compileScript({
      file: `${ROOT}/review/sweep.ts`,
      stateId: "review/sweep",
      vfs: vfsOf({ [`${ROOT}/review/sweep.ts`]: source, [`${ROOT}/review/lib/findings.ts`]: findings, [`${ROOT}/review/triage.json`]: triage }),
      requirePath: [ROOT],
    });
    const root = compiled.documents["review/sweep"]!;
    expect(Object.keys(root.children ?? {}).sort()).toEqual(["classify", "find", "triage", "verify"]);
    expect(root.children!.classify).toEqual({ state: "review/triage", called: true });
    expect(root.sequence).toEqual([]);
    const findRules = (root.children!.find as { transitions: Array<{ name: string; inputs?: Record<string, string> }> }).transitions;
    expect(findRules.map((r) => r.name).sort()).toEqual(["to_triage", "to_verify"]);
    expect(findRules.find((r) => r.name === "to_triage")!.inputs).toEqual({ _entry: ".children.find.output._entry", confirmed: ".children.find.output.confirmed" });
    const states: Record<string, StateDef> = { ...(compiled.documents as Record<string, StateDef>), "review/triage": JSON.parse(triage) as StateDef };
    const report = validateBundle(loadBundle(states, "review/sweep"), {} as never) as { errors: unknown[] };
    expect(report.errors).toEqual([]);
  });
});
