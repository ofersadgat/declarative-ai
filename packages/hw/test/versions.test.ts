/**
 * A running program taking a newer version of a state it calls (JaiRA decision 0029 §3): the call is
 * the boundary, the new version is entered under a versioned id beside the old one, and the switch is
 * journaled — so a conversation picks up a lesson at its next message without disturbing what ran.
 */
import { describe, expect, it } from "vitest";
import { MapSessionStore, withRecord, withSessionPosition, type JsonValue } from "@declarative-ai/exec";
import { SchemaValidator } from "@declarative-ai/validate";
import { hostFunction } from "@declarative-ai/exec";
import { WorkflowEngine } from "../src/engine.js";
import { loadBundle } from "../src/loader.js";
import { compileScript } from "../src/scriptCompile.js";
import type { StateDef, WorkflowBundle } from "../src/format.js";
import { InMemoryPersistence, type EngineEvent } from "../src/ports.js";
import { DirectedTransitions } from "../src/directed.js";
import { baseStateIdOf, versionedSource, versionedStateId, versionHashOf, versionedStates, withVersions, type OfferedVersion } from "../src/versions.js";
import { FakePromptExecutor, newRegistry, ok, promptOf } from "./fakes.js";

const ROOT = "/wf";

function vfsOf(files: Record<string, string>) {
  return {
    read: (p: string) => files[p],
    list: (dir: string) => Object.keys(files).filter((f) => f.startsWith(`${dir}/`)).map((f) => f.slice(dir.length + 1).split("/")[0]!),
  };
}

const PROGRAM = `
import { input } from "@declarative-ai/hw/script";
import turn from "./turn";
export const meta = { name: "Chat" };
export default async function chat(first: string): Promise<{ replies: string[] }> {
  let text: string | null = first;
  let replies: string[] = [];
  while (text !== null) {
    const said = await turn(text, replies.length);
    replies = [...replies, said.reply];
    const next = await input();
    text = next !== null && next.idle !== true ? next.text : null;
  }
  return { replies };
}
`;

const TURN = (prefix: string, extra = "") => `
import { llm } from "@declarative-ai/hw/script";
export const meta = { name: "Turn" };
export default async function turn(text: string, n: number${extra}): Promise<{ reply: string }> {
  return { reply: await llm("${prefix}" + text) };
}
`;

async function compiled(files: Record<string, string>, ids: string[]): Promise<Record<string, StateDef>> {
  const states: Record<string, StateDef> = {};
  for (const id of ids) {
    const c = await compileScript({ file: `${ROOT}/${id}.ts`, stateId: id, vfs: vfsOf(files), requirePath: [ROOT] });
    Object.assign(states, c.documents);
  }
  return states;
}

async function bundleOf(files: Record<string, string>, ids: string[], root: string): Promise<WorkflowBundle> {
  return loadBundle(await compiled(files, ids), root);
}

async function talk(offer: (n: number) => Promise<OfferedVersion | undefined>, messages: Array<string | null>) {
  const files = { [`${ROOT}/s.ts`]: PROGRAM, [`${ROOT}/turn.ts`]: TURN("v1: ") };
  const bundle = await bundleOf(files, ["s", "turn"], "s");
  const fake = new FakePromptExecutor((call) => ok(`[${promptOf(call)}]`));
  const registry = newRegistry();
  const answers = [...messages];
  registry.functions.set("input", hostFunction(async () => ok((answers.shift() ?? null) as JsonValue) as never, { interactive: false, readOnly: true, memoizable: false }));
  const sessions = new MapSessionStore();
  const persistence = new InMemoryPersistence();
  let asked = 0;
  const engine = new WorkflowEngine({
    bundle,
    registry,
    prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
    sessions,
    validator: new SchemaValidator(),
    persistence,
    scripts: { vfs: vfsOf(files), requirePath: [ROOT] },
    // Asked about the called state only: the root's boundaries ask too, about the root.
    versions: (stateId: string) => (stateId === "turn" ? offer(asked++) : undefined),
  } as never);
  const result = await engine.run({ inputs: { first: "hello" } });
  const events = persistence.events.map(({ event }) => event);
  return { result, events, bundle, calls: fake.calls.map(promptOf) };
}

// These compile real scripts — a second or more each, several seconds on a machine running the suite.
describe("a called state's newer version (decision 0029 §3)", { timeout: 30_000 }, () => {
  it("is taken at the next call, entered under a versioned id, and journaled once", async () => {
    const v2 = await bundleOf({ [`${ROOT}/turn.ts`]: TURN("v2: ") }, ["turn"], "turn");
    // The first call runs the version the run started with; every call after is offered v2.
    const { result, events, calls } = await talk(async (n) => (n === 0 ? undefined : { bundle: v2, snapshot: "snap-v2" }), [{ text: "second" } as never, { text: "third" } as never, null]);
    expect(result.outcome).toBe("success");
    expect(calls).toEqual(["v1: hello", "v2: second", "v2: third"]);
    expect(result.outputs).toEqual({ replies: ["[v1: hello]", "[v2: second]", "[v2: third]"] });
    const adopted = events.filter((e): e is Extract<EngineEvent, { type: "version.adopted" }> => e.type === "version.adopted");
    const versioned = versionedStateId("turn", versionHashOf(v2, "turn"));
    expect(adopted).toEqual([expect.objectContaining({ stateId: "turn", version: versioned, snapshot: "snap-v2" })]);
    const entered = events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered").map((e) => e.stateId);
    expect(entered.filter((id) => baseStateIdOf(id) === "turn")).toEqual(["turn", versioned, versioned]);
  });

  it("is passed over when its inputs are not the call's, and the call runs what it was running", async () => {
    const v2 = await bundleOf({ [`${ROOT}/turn.ts`]: TURN("v2: ", ", mood: string") }, ["turn"], "turn");
    const { calls, events } = await talk(async () => ({ bundle: v2 }), [{ text: "second" } as never, null]);
    expect(calls).toEqual(["v1: hello", "v1: second"]);
    expect(events.some((e) => e.type === "version.adopted")).toBe(false);
    expect(events.find((e) => e.type === "version.declined")).toMatchObject({ reason: expect.stringMatching(/'mood' is required/) });
  });

  it("takes nothing when what is offered is what is running", async () => {
    const same = await bundleOf({ [`${ROOT}/turn.ts`]: TURN("v1: ") }, ["turn"], "turn");
    const { calls, events } = await talk(async () => ({ bundle: same }), [{ text: "second" } as never, null]);
    expect(calls).toEqual(["v1: hello", "v1: second"]);
    expect(events.some((e) => e.type === "version.adopted" || e.type === "version.declined")).toBe(false);
  });
});

describe("versions side by side", { timeout: 30_000 }, () => {
  it("names a version by its content, not by the id it runs under", async () => {
    const v1 = await bundleOf({ [`${ROOT}/turn.ts`]: TURN("v1: ") }, ["turn"], "turn");
    const versioned = versionedStateId("turn", versionHashOf(v1, "turn"));
    const moved = { states: versionedStates(v1, "turn", versioned), source: versionedSource(v1, "turn", versioned) };
    expect(Object.keys(moved.states).every((id) => id === versioned || id.startsWith(`${versioned}/`))).toBe(true);
    expect(versionHashOf(moved, versioned)).toBe(versionHashOf(v1, "turn"));
    expect(baseStateIdOf(`${versioned}/call_0`)).toBe("turn/call_0");
    // A replay lays the version back in beside the run's own states.
    const merged = withVersions({ states: { other: v1.states.turn! } }, [{ base: "turn", versioned, bundle: v1 }]);
    expect(Object.keys(merged.states)).toContain(versioned);
    expect(Object.keys(merged.states)).toContain("other");
  });
});

describe("a running workflow's newer version, at a state boundary (decision 0029 §3)", { timeout: 30_000 }, () => {
  const LOOP = (prefix: string, extra = "") => `
import { llm, input } from "@declarative-ai/hw/script";
export const meta = { name: "Session" };
export default async function session(first: string): Promise<{ replies: string[] }> {
  let text: string | null = first;
  let conv: { id: string } | null = null;
  let replies: string[] = [];
  while (text !== null) {
    const turn = await llm<string>({ prompt: "${prefix}" + text, session: conv, withSession: true, label: "reply" });
    conv = turn.session;
    replies = [...replies, turn.value];${extra}
    const next = await input({ prompt: "next?" });
    text = next !== null && next.idle !== true ? next.text : null;
  }
  return { replies };
}
`;

  async function converse(offer: (n: number, replies: number) => Promise<OfferedVersion | undefined>, messages: Array<JsonValue>) {
    const files = { [`${ROOT}/session.ts`]: LOOP("v1: ") };
    const bundle = await bundleOf(files, ["session"], "session");
    const fake = new FakePromptExecutor((call) => ok(`[${promptOf(call)}]`));
    const registry = newRegistry();
    const answers = [...messages];
    registry.functions.set("input", hostFunction(async () => ok((answers.shift() ?? null) as JsonValue) as never, { interactive: false, readOnly: true, memoizable: false }));
    const sessions = new MapSessionStore();
    const persistence = new InMemoryPersistence();
    let asked = 0;
    const engine = new WorkflowEngine({
      bundle,
      registry,
      prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
      sessions,
      validator: new SchemaValidator(),
      persistence,
      scripts: { vfs: vfsOf(files), requirePath: [ROOT] },
      versions: () => offer(asked++, fake.calls.length),
    } as never);
    const result = await engine.run({ inputs: { first: "hello" } });
    return { result, events: persistence.events.map(({ event }) => event), calls: fake.calls.map(promptOf) };
  }

  it("switches at input() — the wait both versions share — and the rest of the conversation runs the new one", async () => {
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: LOOP("v2: ") }, ["session"], "session");
    // Offered from the second boundary on: the first reply has already run under v1.
    const { result, events, calls } = await converse(async (n) => (n === 0 ? undefined : { bundle: v2, snapshot: "s2" }), [{ text: "second" }, { text: "third" }, null]);
    expect(result.outcome).toBe("success");
    expect(calls).toEqual(["v1: hello", "v2: second", "v2: third"]);
    // The old root ends with what the new one returned — everything it said, both versions' turns.
    expect(result.outputs).toEqual({ replies: ["[v1: hello]", "[v2: second]", "[v2: third]"] });
    const adopted = events.filter((e) => e.type === "version.adopted");
    expect(adopted).toEqual([expect.objectContaining({ stateId: "session", at: "input", snapshot: "s2" })]);
    const entered = events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered");
    const handed = entered.find((e) => e.childKey === "$version")!;
    expect(handed.stateId).toMatch(/^session@v[0-9a-f]+$/);
    // The new version starts AT the boundary: its first child is the wait, not its first reply.
    expect(entered.filter((e) => e.parentInstanceId === handed.instanceId).map((e) => e.childKey)[0]).toBe("input");
  });

  it("does not switch where the new version's state takes other inputs, and says why", async () => {
    // v2 counts the messages: a variable alive across the wait, so its `input` is not v1's.
    const counting = LOOP("v2: ")
      .replace("let replies: string[] = [];", "let replies: string[] = [];\n  let asked = 0;")
      .replace("const next = await input({ prompt: \"next?\" });", "const next = await input({ prompt: \"next?\" });\n    asked = asked + 1;")
      .replace("return { replies };", "return { replies: [...replies, String(asked)] };");
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: counting }, ["session"], "session");
    const { calls, events } = await converse(async () => ({ bundle: v2 }), [{ text: "second" }, null]);
    expect(calls).toEqual(["v1: hello", "v1: second"]);
    expect(events.find((e) => e.type === "version.declined")).toMatchObject({ reason: expect.stringMatching(/takes other inputs/) });
  });

  it("takes a second improvement too, one level deeper — v1 hands to v2, v2 to v3", async () => {
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: LOOP("v2: ") }, ["session"], "session");
    const v3 = await bundleOf({ [`${ROOT}/session.ts`]: LOOP("v3: ") }, ["session"], "session");
    // v2 is there once the first reply is made, v3 once the second is.
    const { result, events, calls } = await converse(async (_n, replies) => (replies < 1 ? undefined : replies < 2 ? { bundle: v2 } : { bundle: v3 }), [{ text: "second" }, { text: "third" }, null]);
    expect(result.outcome).toBe("success");
    expect(calls).toEqual(["v1: hello", "v2: second", "v3: third"]);
    const adopted = events.filter((e): e is Extract<EngineEvent, { type: "version.adopted" }> => e.type === "version.adopted");
    expect(adopted.map((e) => e.at)).toEqual(["input", "input"]);
    // The second was handed off by the first version, not by the root: each one level deeper.
    expect(adopted[0]!.from).toBe("session");
    expect(adopted[1]!.from).toBe(adopted[0]!.version);
    expect(result.outputs).toEqual({ replies: ["[v1: hello]", "[v2: second]", "[v3: third]"] });
  });

  it("takes nothing when what is offered is what is running", async () => {
    const same = await bundleOf({ [`${ROOT}/session.ts`]: LOOP("v1: ") }, ["session"], "session");
    const { calls, events } = await converse(async () => ({ bundle: same }), [{ text: "second" }, null]);
    expect(calls).toEqual(["v1: hello", "v1: second"]);
    expect(events.some((e) => e.type === "version.adopted" || e.type === "version.declined")).toBe(false);
  });

  /**
   * A conversation as a person has it: the wait PARKS until they write, and the improvement lands while
   * it waits — past the boundary it waits at. The host asks the run to take it there and then
   * (`takeNewerVersion`), and answers the `input()` the newer version asks again.
   */
  async function parked() {
    const files = { [`${ROOT}/session.ts`]: LOOP("v1: ") };
    const bundle = await bundleOf(files, ["session"], "session");
    const fake = new FakePromptExecutor((call) => ok(`[${promptOf(call)}]`));
    const registry = newRegistry();
    const waits: Array<{ answer: (value: JsonValue) => void; aborted: () => boolean }> = [];
    registry.functions.set(
      "input",
      hostFunction(
        (async (_inputs: unknown, ctx: { abortSignal?: AbortSignal }) =>
          new Promise((resolve) => {
            ctx.abortSignal?.addEventListener("abort", () => resolve({ error: { classification: "canceled", reason: "stepped past" } }), { once: true });
            waits.push({ answer: (value) => resolve(ok(value)), aborted: () => ctx.abortSignal?.aborted === true });
          })) as never,
        { interactive: true, readOnly: true, memoizable: false },
      ),
    );
    const sessions = new MapSessionStore();
    const persistence = new InMemoryPersistence();
    const directed = new DirectedTransitions();
    let offering: OfferedVersion | undefined;
    const engine = new WorkflowEngine({
      bundle,
      registry,
      prompt: withSessionPosition({ sessions }, withRecord({ records: sessions as never }, fake as never)) as never,
      sessions,
      validator: new SchemaValidator(),
      persistence,
      directed,
      scripts: { vfs: vfsOf(files), requirePath: [ROOT] },
      versions: () => offering,
    } as never);
    const running = engine.run({ inputs: { first: "hello" } });
    const until = async (what: () => boolean): Promise<void> => {
      for (let i = 0; i < 400 && !what(); i++) await new Promise((r) => setTimeout(r, 10));
      expect(what()).toBe(true);
    };
    return {
      waits,
      directed,
      until,
      offer: (v: OfferedVersion | undefined) => {
        offering = v;
      },
      running,
      events: () => persistence.events.map(({ event }) => event),
      calls: () => fake.calls.map(promptOf),
    };
  }

  it("switches FROM the input() it waits at when asked: the wait is asked again by the new version, which answers the message", async () => {
    // v2's reply reads a value made from the message, so the reply call is NOT the same boundary in both
    // versions: only the wait is. Without the switch at the wait, v1 would answer the next message.
    const shouting = LOOP("v2: ").replace('const turn = await llm<string>({ prompt: "v2: " + text,', 'const loud = text.toUpperCase();\n    const turn = await llm<string>({ prompt: "v2: " + loud,');
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: shouting }, ["session"], "session");
    const run = await parked();
    await run.until(() => run.waits.length === 1);
    run.offer({ bundle: v2, snapshot: "s2" });
    expect(await run.directed.takeNewerVersion()).toBe(true);
    // The old wait was stepped past, and the new version asks again.
    await run.until(() => run.waits.length === 2);
    expect(run.waits[0]!.aborted()).toBe(true);
    run.waits[1]!.answer({ text: "second" });
    await run.until(() => run.waits.length === 3);
    run.waits[2]!.answer(null);
    const result = await run.running;
    expect(result.outcome).toBe("success");
    expect(run.calls()).toEqual(["v1: hello", "v2: SECOND"]);
    expect(result.outputs).toEqual({ replies: ["[v1: hello]", "[v2: SECOND]"] });
    const events = run.events();
    expect(events.filter((e) => e.type === "version.adopted")).toEqual([expect.objectContaining({ stateId: "session", at: "input", from: "session", snapshot: "s2" })]);
    const entered = events.filter((e): e is Extract<EngineEvent, { type: "instance.entered" }> => e.type === "instance.entered");
    const waitedAt = entered.find((e) => e.childKey === "input" && e.stateId === "session/input")!;
    expect(events).toContainEqual(expect.objectContaining({ type: "instance.terminated", instanceId: waitedAt.instanceId, outcome: "skipped" }));
    const handed = entered.find((e) => e.childKey === "$version")!;
    expect(entered.filter((e) => e.parentInstanceId === handed.instanceId).map((e) => e.childKey)[0]).toBe("input");
  });

  it("stays at the wait when nothing newer fits it, and the waiting input() is the one to answer", async () => {
    const counting = LOOP("v2: ")
      .replace("let replies: string[] = [];", "let replies: string[] = [];\n  let asked = 0;")
      .replace('const next = await input({ prompt: "next?" });', 'const next = await input({ prompt: "next?" });\n    asked = asked + 1;')
      .replace("return { replies };", "return { replies: [...replies, String(asked)] };");
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: counting }, ["session"], "session");
    const run = await parked();
    await run.until(() => run.waits.length === 1);
    expect(await run.directed.takeNewerVersion()).toBe(false);
    run.offer({ bundle: v2 });
    expect(await run.directed.takeNewerVersion()).toBe(false);
    expect(run.events().find((e) => e.type === "version.declined")).toMatchObject({ reason: expect.stringMatching(/not at 'input'.*takes other inputs/) });
    expect(run.waits[0]!.aborted()).toBe(false);
    run.offer(undefined);
    run.waits[0]!.answer({ text: "second" });
    await run.until(() => run.waits.length === 2);
    run.waits[1]!.answer(null);
    expect((await run.running).outputs).toEqual({ replies: ["[v1: hello]", "[v1: second]"] });
  });

  it("switches at a boundary BELOW the root — an input() inside a phase — handing each composite on the way what the old one had", async () => {
    const PHASED = (prefix: string) => `
import { llm, input, phase } from "@declarative-ai/hw/script";
export const meta = { name: "Session" };
export default async function session(first: string): Promise<{ replies: string[] }> {
  let replies: string[] = [];
  phase("Talk");
  let text: string | null = first;
  while (text !== null) {
    const said = await llm<string>({ prompt: "${prefix}" + text, label: "reply" });
    replies = [...replies, said];
    const next = await input({ prompt: "next?" });
    text = next !== null && next.idle !== true ? next.text : null;
  }
  phase("Done");
  return { replies };
}
`;
    const files = { [`${ROOT}/session.ts`]: PHASED("v1: ") };
    const bundle = await bundleOf(files, ["session"], "session");
    const v2 = await bundleOf({ [`${ROOT}/session.ts`]: PHASED("v2: ") }, ["session"], "session");
    const fake = new FakePromptExecutor((call) => ok(`[${promptOf(call)}]`));
    const registry = newRegistry();
    const answers: JsonValue[] = [{ text: "second" }, null];
    registry.functions.set("input", hostFunction(async () => ok((answers.shift() ?? null) as JsonValue) as never, { interactive: false, readOnly: true, memoizable: false }));
    const persistence = new InMemoryPersistence();
    const engine = new WorkflowEngine({
      bundle,
      registry,
      prompt: fake as never,
      validator: new SchemaValidator(),
      persistence,
      scripts: { vfs: vfsOf(files), requirePath: [ROOT] },
      versions: () => (fake.calls.length >= 1 ? { bundle: v2 } : undefined),
    } as never);
    const result = await engine.run({ inputs: { first: "hello" } });
    const events = persistence.events.map(({ event }) => event);
    expect(result.outcome).toBe("success");
    expect(fake.calls.map(promptOf)).toEqual(["v1: hello", "v2: second"]);
    const adopted = events.filter((e): e is Extract<EngineEvent, { type: "version.adopted" }> => e.type === "version.adopted");
    expect(adopted).toHaveLength(1);
    // Inside the phase: the way down names the phase, then the wait.
    expect(adopted[0]!.at).toMatch(/^[^/]+\/input$/);
    expect(result.outputs).toEqual({ replies: ["[v1: hello]", "[v2: second]"] });
  });

  it("declines, in a workflow written by hand, a boundary after which the new version reads what an earlier state of this run produced — and takes the next", async () => {
    const fn = (name: string) => ({ operation: { function: name, args: {} }, outputs: { v: { schema: { type: "string" }, binding: ".operation.output.v" } } });
    const json = (tag: string): Record<string, StateDef> => ({
      flow: {
        children: {
          a: { state: "flow/a" },
          b: { state: "flow/b" },
          c: { state: "flow/c", inputs: { from_a: ".children.a.output.v" } },
        },
        sequence: ["a", "b", "c"],
        outputs: { last: { schema: { type: "string" }, binding: ".children.c.output.v" } },
      } as unknown as StateDef,
      "flow/a": fn("a") as unknown as StateDef,
      "flow/b": fn("b") as unknown as StateDef,
      "flow/c": { ...fn(`c${tag}`), inputs: { from_a: { schema: { type: "string" } } } } as unknown as StateDef,
    });
    const registry = newRegistry();
    const ran: string[] = [];
    for (const name of ["a", "b", "c1", "c2"]) registry.functions.set(name, hostFunction(async () => (ran.push(name), ok({ v: name })) as never, { interactive: false, readOnly: true, memoizable: false }));
    const persistence = new InMemoryPersistence();
    const v2 = loadBundle(json("2"), "flow");
    const engine = new WorkflowEngine({
      bundle: loadBundle(json("1"), "flow"),
      registry,
      validator: new SchemaValidator(),
      persistence,
      // Offered once `a` has run: from `b` on.
      versions: () => (ran.includes("a") ? { bundle: v2 } : undefined),
    } as never);
    const result = await engine.run({ inputs: {} });
    const events = persistence.events.map(({ event }) => event);
    expect(result.outcome).toBe("success");
    // At `b`, v2's `c` would read `a` — which ran in v1, and is not in v2's tree: declined, saying so.
    expect(events.find((e) => e.type === "version.declined")).toMatchObject({ reason: expect.stringMatching(/not at 'b': it reads what 'a' produced in this run/) });
    // At `c`, what it reads of `a` is handed to it resolved: taken.
    expect(events.filter((e) => e.type === "version.adopted")).toEqual([expect.objectContaining({ at: "c" })]);
    expect(ran).toEqual(["a", "b", "c2"]);
    expect(result.outputs).toEqual({ last: "c2" });
  });
});
