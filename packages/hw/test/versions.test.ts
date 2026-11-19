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
    versions: (_stateId: string) => offer(asked++),
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
