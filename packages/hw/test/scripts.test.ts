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
import { hostFunction, MapSessionStore, withRecord, withSessionPosition, type JsonValue } from "@declarative-ai/exec";
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

async function run(
  source: string,
  inputs: Record<string, JsonValue>,
  script: Script,
  options: { extra?: Record<string, string>; loaded?: LoadedInstance; functions?: Record<string, (inputs: Record<string, unknown>) => JsonValue> } = {},
) {
  const extra = options.extra ?? {};
  const compiled = await compile(source, extra);
  // The JSON states beside the script are the host's files, loaded with the compiled ones.
  const states: Record<string, StateDef> = { ...(compiled.documents as Record<string, StateDef>) };
  for (const [path, text] of Object.entries(extra)) {
    if (path.endsWith(".json")) states[path.slice(ROOT.length + 1, -".json".length)] = JSON.parse(text) as StateDef;
  }
  const bundle = loadBundle(states, "s");
  // Every bundle a script compiles to must validate as an authored one would.
  const report = validateBundle(bundle, {} as never) as { errors: Array<{ stateId: string; path: string; message: string }> };
  if (report.errors.length > 0) throw new Error(`the compiled bundle does not validate:\n${report.errors.map((e) => `${e.stateId} ${e.path}: ${e.message}`).join("\n")}`);
  const fake = new FakePromptExecutor(script);
  const registry = newRegistry();
  for (const [name, fn] of Object.entries(options.functions ?? {})) {
    registry.functions.set(name, hostFunction(async (i: Record<string, unknown>) => ok(fn(i)) as never, { interactive: false, readOnly: true, memoizable: false }));
  }
  const sessions = new MapSessionStore();
  const persistence = new InMemoryPersistence();
  const engine = new WorkflowEngine({
    bundle,
    registry,
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
  it("makes the state the script IS, one child state per phase, and one state per call inside each", async () => {
    const { documents } = await compile(SWEEP);
    expect(Object.keys(documents).filter((id) => id.split("/").length === 2).sort()).toEqual(["s/find", "s/triage", "s/verify"]);
    // Find fans a call out over the dimensions: a mount with `each`, its element, the element's call, the state after.
    expect(Object.keys(documents["s/find"]!.children ?? {}).sort()).toEqual(["after_0", "map_0"]);
    expect(documents["s/find"]!.children!.map_0).toMatchObject({ state: "s/find/map_0", async: true, failureValue: null });
    expect(documents["s/find/map_0/call_2"]!.operation).toMatchObject([{ prompt: "{{.inputs._call.prompt}}" }, { script: {} }]);
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
    // The element reads what it needs from outside it, and nothing it does not.
    expect(Object.keys(documents["s/find/map_0"]!.inputs ?? {}).sort()).toEqual(["d", "ref", "seen"]);
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
    ).rejects.toThrow(/'m' lives across a cut.*Map/);
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
    expect(entered(events).filter((id) => id.split("/").length <= 2)).toEqual(["s", "s/find", "s/verify", "s/find", "s/triage"]);
    // Every call was a state of its own: round 1 two reviews and three verifications, round 2 two reviews.
    expect(entered(events).filter((id) => /call_\d+$/.test(id))).toHaveLength(7);
  });

  it("journals every call at a site of its own, and every log line (\"phases\" mode)", async () => {
    const { events } = await run(SWEEP.replace('name: "Sweep review",', 'name: "Sweep review", compile: "phases",'), { ref: "main" }, sweepModel);
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
    expect(entered(events).filter((id) => id.split("/").length <= 2)).toEqual(["s", "s/scan", "s/fix"]);
  });

  it("gives a body script its exports as outputs", async () => {
    const source = `const a = await llm("a");\nphase("Two");\nexport const first = a;\nexport const second = await llm("b");`;
    const { result } = await run(source, {}, (c) => ok(textOf(c).toUpperCase()));
    expect(result.outputs).toEqual({ first: "A", second: "B" });
  });

  it("runs imported code inline", async () => {
    const source = `import { shout } from "./lib";\nexport default async function f(x: string) { phase("One"); const y = shout(x); phase("Two"); return await llm(y); }`;
    const lib = `export function shout(s: string): string { return s.toUpperCase() + "!"; }`;
    const { calls, compiled } = await run(source, { x: "hi" }, () => ok("ok"), { extra: { [`${ROOT}/lib.ts`]: lib } });
    expect(textOf(calls[0]!)).toBe("HI!");
    // The module it runs is named for a host to gate and freeze; a type-only import runs nothing and is not.
    expect(compiled.documents.s!.generated?.modules).toEqual([`${ROOT}/lib.ts`]);
    const typed = await compile(`import type { T } from "./types";
export default async function f(): Promise<T> { return { n: 1 }; }`, { [`${ROOT}/types.ts`]: `export type T = { n: number };` });
    expect(typed.documents.s!.generated?.modules).toBeUndefined();
    expect(Object.keys(typed.generated.inputs)).toContain(`${ROOT}/types.ts`);
  });

  it("refuses Date.now() and Math.random() in the script's own code, and offers the recorded ones", async () => {
    const bad = await run(`export default async function f() { return Date.now(); }`, {}, () => ok(""));
    expect(bad.result.outcome).toBe("error");
    const good = await run(`import { now } from "@declarative-ai/hw/script";\nexport default async function f() { return now() > 0; }`, {}, () => ok(""));
    expect(good.result.outputs).toEqual({ result: true });
    expect(good.events.some((e) => e.type === "script.call.settled" && e.site === "now#0")).toBe(true);
  });

  it("answers a re-run of a phase's code from its record, and asks only what it had not", async () => {
    const source = `export const meta = { compile: "phases" };\nexport default async function f() { const a = await llm("one"); const b = await llm("two"); return a + b; }`;
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
    const source = `export const meta = { compile: "state" };\nexport default async function f() { const votes = await parallel([0, 1, 2].map(() => () => llm<{ ok: boolean }>("vote"))); return votes.length; }`;
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
    // Called from a fan-out's element: the element mounts the state, once per call site.
    expect(compiled.documents["s/map_0"]!.children!.triage_1).toMatchObject({ state: "triage", called: true });
    expect(result.failure?.reason).toBeUndefined();
    expect(result.outcome).toBe("success");
    expect(result.outputs).toEqual({ severities: ["high", "low"] });
    const calls = events.filter((e) => e.type === "instance.entered" && e.stateId === "triage");
    // Two calls of one state run side by side — one per element of the fan-out — neither superseding the other.
    expect(calls).toHaveLength(2);
    expect(events.some((e) => e.type === "child.superseded")).toBe(false);
  });

  it("calls a state from a phase's code in \"phases\" mode, marking each entry as a call", async () => {
    const source = `import triage from "./triage";
export const meta = { compile: "phases" };
export default async function f(items: string[]) { const out = await Promise.all(items.map((finding) => triage({ finding }))); return { severities: out.map((o) => o.severity) }; }`;
    const { result, events } = await run(source, { items: ["crash", "typo"] }, () => ok(""), { extra: { [`${ROOT}/triage.json`]: TRIAGE } });
    expect(result.outputs).toEqual({ severities: ["high", "low"] });
    const calls = events.filter((e) => e.type === "instance.entered" && e.stateId === "triage");
    expect(calls).toHaveLength(2);
    expect(calls.every((e) => (e as { calledAt?: string }).calledAt !== undefined)).toBe(true);
  });

  it("calls a state named in workflow() with a literal, and a script's whole return comes back whole", async () => {
    const child = `export const meta = { name: "child" };\nreturn { doubled: args.n * 2 };`;
    const source = `export const meta = { name: "parent" };\nconst r = await workflow("child", { n: 21 });\nreturn r.doubled;`;
    const both = { [`${ROOT}/child.ts`]: child, [`${ROOT}/s.ts`]: source };
    const childDocs = await compileScript({ file: `${ROOT}/child.ts`, stateId: "child", vfs: vfsOf(both), requirePath: [ROOT] });
    const parent = await compile(source, { [`${ROOT}/child.ts`]: child });
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

  it("re-attaches a state its code had called when the run stopped — continuing a live one, reading a finished one", async () => {
    const source = `import triage from "./triage";
export const meta = { compile: "state" };
export default async function f() { const t = await triage({ finding: "crash here" }); return t.severity; }`;
    const extra = { [`${ROOT}/triage.json`]: TRIAGE };
    const first = await run(source, {}, echoing, { extra });
    const entered = first.events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered");
    const root = entered.find((e) => e.stateId === "s")!;
    const called = entered.find((e) => e.stateId === "triage")!;
    expect(called.calledAt).toMatch(/^workflow:triage:/);
    for (const live of [true, false]) {
      // The run stopped with the call made and its answer never settled: the instance is the caller's child, marked as called.
      const child: LoadedInstance = { id: called.instanceId, stateId: "triage", childKey: "triage", calledAt: called.calledAt!, inputs: { finding: "crash here" }, live, ...(live ? {} : { outcome: "success" as const }) };
      const loaded: LoadedInstance = { id: root.instanceId, stateId: "s", inputs: {}, live: true, children: [child] };
      const resumed = await run(source, {}, echoing, { extra, loaded });
      expect(resumed.result.outputs).toEqual({ result: "high" });
      // Not called again: no instance of it was entered, and a live one ended under the id it had.
      expect(resumed.events.some((e) => e.type === "instance.entered" && e.stateId === "triage")).toBe(false);
      expect(resumed.events.some((e) => e.type === "instance.terminated" && e.instanceId === called.instanceId)).toBe(live);
    }
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

  it("warns about a function that calls llm() — itself or through what it imports — used in a guard, which runs every round", async () => {
    const FN = "/fn";
    const files = {
      [`${FN}/helper.ts`]: `import { llm } from "@declarative-ai/hw/script";
export async function ask(q: string): Promise<string> { return await llm(q); }`,
      [`${FN}/is_bug.ts`]: `import { ask } from "./helper";
export default async function is_bug(text: string): Promise<boolean> { return (await ask(text)) === "yes"; }`,
      [`${FN}/is_short.ts`]: `export default function is_short(text: string): boolean { return text.length < 5; }`,
    };
    const vfs = vfsOf(files);
    const { createSymbolIndex } = await import("../src/moduleIndex.js");
    const { createUserFunctions } = await import("../src/userFunctions.js");
    const { requirePathFor } = await import("../src/moduleLoader.js");
    const symbols = await createSymbolIndex({ vfs });
    const userFunctions = await createUserFunctions({ vfs, requirePath: requirePathFor([FN]) });
    const bundle = loadBundle(
      {
        root: {
          inputs: { text: { schema: { type: "string" } } },
          transitions: [
            { when: "is_bug(.inputs.text)", to: "terminate.success" },
            { when: "is_short(.inputs.text)", to: "terminate.success" },
          ],
        },
      },
      "root",
      { defaultRoot: FN, vfs, symbols, userFunctions, documentCache: new Map() },
    );
    const report = validateBundle(bundle, { functions: userFunctions.entries as never });
    expect(report.errors).toEqual([]);
    const warned = report.warnings.filter((w) => /makes model calls/.test(w.message));
    expect(warned.map((w) => w.path)).toEqual(["transitions[0].when"]);
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
      expect(Object.keys(bundle.states).filter((id) => id.split("/").length === 3).sort()).toEqual(["review/sweep/find", "review/sweep/triage", "review/sweep/verify"]);

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
    expect(Object.keys(root.children ?? {}).sort()).toEqual(["find", "triage", "verify"]);
    expect(root.sequence).toEqual([]);
    const findRules = (root.children!.find as { transitions: Array<{ name: string; inputs?: Record<string, string> }> }).transitions;
    expect(findRules.map((r) => r.name).sort()).toEqual(["to_triage", "to_verify"]);
    expect(findRules.find((r) => r.name === "to_triage")!.inputs).toEqual({ _entry: ".children.find.output._entry", confirmed: ".children.find.output.confirmed" });
    // `Promise.all(confirmed.map((f) => classify(…)))` in Triage: a fan-out whose element mounts the called state.
    const elementId = Object.keys(compiled.documents).find((id) => /^review\/sweep\/triage\/map_\d+$/.test(id))!;
    expect(Object.values(compiled.documents[elementId]!.children ?? {}).some((c) => (c as { state?: string }).state === "review/triage")).toBe(true);
    const states: Record<string, StateDef> = { ...(compiled.documents as Record<string, StateDef>), "review/triage": JSON.parse(triage) as StateDef };
    const report = validateBundle(loadBundle(states, "review/sweep"), {} as never) as { errors: unknown[] };
    expect(report.errors).toEqual([]);
  });
});

/** A model that fails when the prompt says so, and otherwise echoes it. */
const echoing: Script = (call) =>
  textOf(call).startsWith("boom")
    ? ({ error: { classification: "permanent", reason: textOf(call) }, metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" } } as never)
    : ok(`said ${textOf(call)}`);

const callStates = (events: EngineEvent[]) => entered(events).filter((id) => /call_\d+$/.test(id));

describe("one state per call (\"calls\" mode)", () => {
  it("makes each call a state whose operation IS the call — literal options literal, computed ones read its inputs", async () => {
    const source = `export default async function f(which: string) {
  const a = await llm("first", { model: "anthropic/claude-sonnet-5", temperature: 0 });
  const b = await llm({ prompt: "second " + a, model: which });
  return b;
}`;
    const { documents } = await compile(source);
    expect(documents["s/call_0"]!.operation).toMatchObject([{ prompt: "{{.inputs._call.prompt}}", model: "anthropic/claude-sonnet-5", temperature: 0 }, { script: {} }]);
    expect((documents["s/call_1"]!.operation as unknown[])[0]).toMatchObject({ model: { $expr: ".inputs.model_1" } });
    // A computed knob is an input of its own, typed as the option is, so the wiring checker reads it.
    expect(documents["s/call_1"]!.inputs!.model_1).toMatchObject({ schema: { type: "string" } });
    const { calls, result } = await run(source, { which: "openai/gpt-5.6" }, echoing);
    expect(calls.map((c) => [textOf(c), c.name])).toEqual([["first", "anthropic/claude-sonnet-5"], ["second said first", "openai/gpt-5.6"]]);
    expect(result.outputs).toEqual({ result: "said second said first" });
  });

  it("keeps evaluation order: what is evaluated before a call is evaluated before it, across the cut", async () => {
    const source = `export default async function f() {
  const order: string[] = [];
  const mark = (s: string) => { order.push(s); return s; };
  const x = mark("a") + (await llm(mark("b")));
  mark("c");
  return { order, x };
}`;
    const { result } = await run(source, {}, echoing);
    expect(result.outputs).toEqual({ order: ["a", "b", "c"], x: "asaid b" });
  });

  it("lowers a call under &&, || and ?: to a branch — made only when JavaScript would make it", async () => {
    const source = `export default async function f(flag: boolean) {
  const a = flag && (await llm("and"));
  const b = flag || (await llm<string>("or")); // untyped, TypeScript would infer T from 'flag': boolean
  const c = flag ? await llm("then") : "skipped";
  return { a, b, c };
}`;
    const on = await run(source, { flag: true }, echoing);
    expect(on.calls.map(textOf)).toEqual(["and", "then"]);
    expect(on.result.outputs).toEqual({ a: "said and", b: true, c: "said then" });
    const off = await run(source, { flag: false }, echoing);
    expect(off.calls.map(textOf)).toEqual(["or"]);
    expect(off.result.outputs).toEqual({ a: false, b: "said or", c: "skipped" });
  });

  it("inlines a helper that makes a call, once per call of it — a generic one typed by its caller's T", async () => {
    const source = `async function ask(q: string) { const a = await llm(q); return a.toUpperCase(); }
async function typed<T>(q: string): Promise<T> { return await llm<T>(q); }
export default async function f() {
  const first = (await ask("x")) + (await ask("y"));
  const verdict = await typed<{ ok: boolean }>("judge");
  return { first, ok: verdict.ok };
}`;
    const { calls, result, events } = await run(source, {}, (c) => (textOf(c) === "judge" ? ok({ ok: true }) : echoing(c)));
    expect(callStates(events)).toHaveLength(3);
    expect(result.outputs).toEqual({ first: "SAID XSAID Y", ok: true });
    expect(calls[2]!.op.output.schema).toMatchObject({ type: "object", properties: { ok: { type: "boolean" } } });
  });

  it("refuses a recursive helper that makes a call", async () => {
    await expect(compile(`async function again(n: number): Promise<string> { return n > 0 ? await again(n - 1) : await llm("x"); }\nexport default async function f() { return await again(2); }`)).rejects.toThrow(/calls itself/);
  });

  it("catches a failed call in the state its catch compiled to, with the variables it had before the call", async () => {
    const source = `export default async function f() {
  let log = "start";
  try {
    const a = await llm("boom now");
    log = "unreached " + a;
  } catch (e) {
    log = log + " / caught: " + (e as Error).message;
  }
  try {
    await llm("fine");
    throw new Error("thrown by the code");
  } catch (e) {
    log = log + " / " + (e as Error).message;
  }
  return log;
}`;
    const { result, documents } = { ...(await run(source, {}, echoing)), documents: (await compile(source)).documents };
    expect(result.outputs).toEqual({ result: "start / caught: boom now / thrown by the code" });
    // The failure rule reads the call's own inputs — `.children.<call>.inputs` — since a failed state publishes no outputs.
    const rules = (documents.s!.children!.call_0 as { transitions: Array<{ name: string; inputs?: Record<string, string> }> }).transitions;
    expect(rules.find((r) => r.name === "catch")!.inputs).toMatchObject({ log: ".children.call_0.inputs.log", _error: ".children.call_0.failure" });
  });

  it("lowers a switch around calls, falling through as JavaScript does", async () => {
    const source = `export default async function f() {
  const kind = await llm("kind");
  let out = "";
  switch (kind) {
    case "said kind": out += await llm("matched");
    case "other": out += "|fell";
      break;
    default: out = "default";
  }
  return out;
}`;
    const { result } = await run(source, {}, echoing);
    expect(result.outputs).toEqual({ result: "said matched|fell" });
  });

  it("makes now() and random() calls like any other — each its own state, each asked afresh", async () => {
    const source = `import { now, random } from "@declarative-ai/hw/script";
export default async function f() { const t0 = now(); await llm("x"); const t1 = now(); const r = random(); return t1 >= t0 && r >= 0 && r < 1; }`;
    const { result, events } = await run(source, {}, echoing);
    expect(result.outputs).toEqual({ result: true });
    expect(callStates(events)).toHaveLength(4);
  });

  it("fans a map out over an each mount — parallel's failed items read as null, Promise.all's fail the fan-out", async () => {
    const parallelSource = `export default async function f(items: string[]) { return await parallel(items.map((x) => () => llm(x))); }`;
    const tolerant = await run(parallelSource, { items: ["ok", "boom one", "fine"] }, echoing);
    expect(tolerant.result.outputs).toEqual({ result: ["said ok", null, "said fine"] });
    const strictSource = `export default async function f(items: string[]) { return await Promise.all(items.map(async (x) => await llm(x))); }`;
    const strict = await run(strictSource, { items: ["ok", "boom one"] }, echoing);
    expect(strict.result.outcome).toBe("error");
  });

  it("runs a list of calls side by side, and a pipeline's stages in order per item", async () => {
    const fork = await run(`export default async function f() { return await Promise.all([llm("a"), llm("b")]); }`, {}, echoing);
    expect(fork.result.outputs).toEqual({ result: ["said a", "said b"] });
    const piped = await run(`export default async function f(xs: string[]) { return await pipeline(xs, (x) => llm("one " + x), (y) => llm("two " + y)); }`, { xs: ["p", "q"] }, echoing);
    expect(piped.result.outputs).toEqual({ result: ["said two said one p", "said two said one q"] });
  });

  it("refuses an element that writes what lives outside it", async () => {
    await expect(compile(`export default async function f(xs: string[]) { let n = 0; await parallel(xs.map((x) => async () => { n++; return await llm(x); })); return n; }`)).rejects.toThrow(/writes 'n', which lives outside it/);
  });

  it("lifts a branch on a call's result into the rules, where a person can read it", async () => {
    const source = `export default async function f() {
  const r = await llm<{ ok: boolean }>("judge");
  if (r.ok) return await llm("yes");
  return await llm("no");
}`;
    const { documents } = await compile(source);
    const rules = (documents.s!.children!.call_0 as { transitions: Array<{ when: string; to: string }> }).transitions;
    expect(rules.map((r) => [r.when, r.to])).toEqual([
      [".children.call_0.output.r.ok", "call_1"],
      ["!(.children.call_0.output.r.ok)", "call_2"],
    ]);
    const { calls } = await run(source, {}, (c) => (textOf(c) === "judge" ? ok({ ok: false }) : echoing(c)));
    expect(calls.map(textOf)).toEqual(["judge", "no"]);
  });

  it("puts a named type used twice in $defs", async () => {
    const source = `type Person = { name: string };\nexport default async function f() { return await llm<{ author: Person; reviewer: Person }>("who"); }`;
    const { documents } = await compile(source);
    const schema = ((documents["s/call_0"]!.operation as Array<{ output: { value: { schema: Record<string, unknown> } } }>)[0]!.output.value.schema) as { properties: Record<string, unknown>; $defs: Record<string, unknown> };
    expect(schema.properties.author).toEqual({ $ref: "#/$defs/Person" });
    expect(schema.$defs.Person).toMatchObject({ type: "object", properties: { name: { type: "string" } } });
  });

  it("maps every compiled state back to the script's line, and a failure in its code to the line that threw", async () => {
    const source = `export default async function f() {\n  const a = await llm("x");\n  const b: any = undefined;\n  return b.missing + a;\n}`;
    const compiled = await compile(source);
    expect(compiled.sourceMap["s/call_0"]).toEqual({ line: 2, column: 1 });
    expect(compiled.documents["s/call_0"]!.generated?.at).toEqual({ line: 2, column: 1 });
    const { result } = await run(source, {}, echoing);
    expect(result.outcome).toBe("error");
    expect(JSON.stringify(result)).toMatch(/s\.ts:4: /);
  });
});

describe("imports (SCRIPTS.md §10)", () => {
  it("imports a prompt: the template is the call's operation, its holes the call's inputs", async () => {
    const source = `import summarize from "./summarize.md";\nexport default async function f(topic: string) { return await summarize({ topic }); }`;
    const { documents } = await compile(source, { [`${ROOT}/summarize.md`]: "Summarize {{.inputs.topic}} in one line." });
    expect((documents["s/call_0"]!.operation as unknown[])[0]).toMatchObject({ prompt: "Summarize {{.inputs.topic}} in one line.", input: { topic: { binding: { $expr: ".inputs._call.topic" } } } });
    const { calls } = await run(source, { topic: "tides" }, echoing, { extra: { [`${ROOT}/summarize.md`]: "Summarize {{.inputs.topic}} in one line." } });
    expect(textOf(calls[0]!)).toBe("Summarize tides in one line.");
  });

  it("imports a registered function from $REGISTRY, and data with { type: \"json\" }", async () => {
    const source = `import { shout } from "$REGISTRY";\nimport config from "./config.json" with { type: "json" };\nexport default async function f() { return await shout({ text: config.greeting }); }`;
    const { result } = await run(source, {}, echoing, { extra: { [`${ROOT}/config.json`]: JSON.stringify({ greeting: "hi" }) }, functions: { shout: (i) => ({ loud: String(i.text).toUpperCase() }) } });
    expect(result.outputs).toEqual({ result: { loud: "HI" } });
  });

  it("imports code as an operation: a call of its own in \"calls\" mode, recorded in \"phases\" mode", async () => {
    const lib = `export function stamp(x: string): string { return "stamped " + x; }`;
    const source = (mode: string) => `import { stamp } from "./lib" with { as: "operation" };\nexport const meta = { compile: "${mode}" };\nexport default async function f() { return stamp("a"); }`;
    const calls = await run(source("calls"), {}, echoing, { extra: { [`${ROOT}/lib.ts`]: lib } });
    expect(calls.result.outputs).toEqual({ result: "stamped a" });
    expect(callStates(calls.events)).toHaveLength(1);
    const phases = await run(source("phases"), {}, echoing, { extra: { [`${ROOT}/lib.ts`]: lib } });
    expect(phases.result.outputs).toEqual({ result: "stamped a" });
    expect(phases.events.some((e) => e.type === "script.call.settled" && e.site === "recorded:stamp#0")).toBe(true);
    // Recorded means awaited: a synchronous function cannot make the call.
    const sync = `import { stamp } from "./lib" with { as: "operation" };
export const meta = { compile: "phases" };
export default async function f() { return ["a"].map((x) => stamp(x)); }`;
    await expect(compile(sync, { [`${ROOT}/lib.ts`]: lib })).rejects.toThrow(/s\.ts:3.*imported as an operation.*async function/);
  });

  it("bans Date.now() in code the script imports, too", async () => {
    const lib = `export function late(): number { return Date.now(); }`;
    const { result } = await run(`import { late } from "./lib";\nexport default async function f() { return late(); }`, {}, echoing, { extra: { [`${ROOT}/lib.ts`]: lib } });
    expect(result.outcome).toBe("error");
    expect(JSON.stringify(result)).toMatch(/Date\.now\(\) breaks replay/);
  });

  it("hands a generic helper its caller's T as a schema, where helpers are not inlined (\"phases\" mode)", async () => {
    const source = `export const meta = { compile: "phases" };\nasync function typed<T>(q: string): Promise<T> { return await llm<T>(q); }\nexport default async function f() { const v = await typed<{ ok: boolean }>("judge"); return v.ok; }`;
    const { calls, result } = await run(source, {}, () => ok({ ok: true }));
    expect(calls[0]!.op.output.schema).toMatchObject({ type: "object", properties: { ok: { type: "boolean" } } });
    expect(result.outputs).toEqual({ result: true });
  });
});
