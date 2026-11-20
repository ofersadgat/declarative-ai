/**
 * A workflow script run as a plain program (SCRIPTS.md §17): the same file the engine compiles to
 * states, run directly under the standalone host — `llm()` to a model, conversations held in memory,
 * `input()` from the person, `workflow()` as a function call, `agent()` through an agent query.
 */
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createStandaloneHost, idleMs, runScript, type StandaloneLlmDefinition } from "../src/index.js";

// Never reported to an engine this machine happens to be running (§17.4): these runs are the plain kind.
process.env["DAI_SCRIPT_REPORT"] = "0";

function dir(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmpdir(), "dai-script-"));
  for (const [name, text] of Object.entries(files)) writeFileSync(join(root, name), text);
  return root;
}

const CHAT = `
import { llm, input } from "@declarative-ai/hw/script";
export const meta = { name: "Chat" };
export default async function chat(first: string): Promise<{ replies: string[]; positions: string[] }> {
  let msg: { text: string } | null = { text: first };
  let conv: { id: string } | null = null;
  let replies: string[] = [];
  let positions: string[] = [];
  while (msg !== null) {
    const turn = await llm<string>({ prompt: msg.text, session: conv, withSession: true });
    conv = turn.session;
    replies = [...replies, turn.value];
    positions = [...positions, conv === null ? "" : conv.id];
    const next = await input({ prompt: "next?" });
    msg = next !== null && next.idle !== true ? { text: next.text } : null;
  }
  return { replies, positions };
}
`;

describe("a script run as a program", () => {
  it("runs the loop the engine would compile, continuing one conversation in memory", async () => {
    const root = dir({ "chat.ts": CHAT });
    const sent: StandaloneLlmDefinition[] = [];
    const lines = ["second", "third", null];
    const result = await runScript(join(root, "chat.ts"), ["first"], {
      model: "test/model",
      execute: async (def) => (sent.push(def), { value: `reply ${sent.length}` }),
      read: async () => lines.shift() ?? null,
      write: () => undefined,
    });
    expect(result).toEqual({ replies: ["reply 1", "reply 2", "reply 3"], positions: ["s1@1", "s1@2", "s1@3"] });
    // The third call carries everything said before it, in order.
    expect(sent[2]!.messages.map((m) => `${m.role}:${m.content}`)).toEqual([
      "user:first",
      "assistant:reply 1",
      "user:second",
      "assistant:reply 2",
      "user:third",
    ]);
  });

  it("branches when a call continues an earlier position, leaving the original stream as it was", async () => {
    const host = createStandaloneHost({ model: "m/x", execute: async (def) => ({ value: `saw ${def.messages.length}` }), write: () => undefined });
    const a = (await host.llm({ prompt: "one", withSession: true }, undefined)) as { session: { id: string } };
    const b = (await host.llm({ prompt: "two", session: a.session, withSession: true }, undefined)) as { session: { id: string } };
    const c = (await host.llm({ prompt: "two again", session: a.session, withSession: true }, undefined)) as { session: { id: string }; value: string };
    expect(a.session.id).toBe("s1@1");
    expect(b.session.id).toBe("s1@2");
    expect(c.session.id).toBe("s2@1");
    expect(c.value).toBe("saw 3");
    expect(host.conversations.turnsAt("s2@1")!.map((t) => t.content)).toEqual(["one", "saw 1", "two again", "saw 3"]);
  });

  it("asks for T's shape, wrapping a non-object one for the wire and handing back the plain value", async () => {
    const host = createStandaloneHost({ model: "m/x", execute: async (def) => ({ value: (def.schema as { type?: string }).type === "object" ? { value: 7 } : 0 }), write: () => undefined });
    expect(await host.llm({ prompt: "n?" }, { type: "number" } as never)).toBe(7);
  });

  it("calls another script by path with named arguments, as the engine would mount it", async () => {
    const root = dir({
      "greet.ts": `export default async function greet(who: string, punct = "!"): Promise<string> { return "hello " + who + punct; }`,
      "main.ts": `import { workflow } from "@declarative-ai/hw/script";
export default async function main(): Promise<unknown> { return await workflow("./greet.ts", { who: "ada", punct: "?" }); }`,
    });
    expect(await runScript(join(root, "main.ts"), [], { cwd: root, write: () => undefined })).toBe("hello ada?");
  });

  it("runs the script as it stands at each call — an edited one takes over at the next call (non-durable handoff)", () => {
    // In a Node process of its own, as a real run is: vitest's loader keys a module by its path and
    // ignores the query that names its version, which Node's module cache does not.
    // Its imports of the other packages go to their source, as vitest's aliases send this file's:
    // their `exports` name `dist`, which CI does not build before it tests.
    const packages = join(__dirname, "..", "..");
    const paths = Object.fromEntries(
      readdirSync(packages).flatMap((pkg) => [
        [`@declarative-ai/${pkg}`, [join(packages, pkg, "src", "index.ts")]],
        [`@declarative-ai/${pkg}/*`, [join(packages, pkg, "src", "*.ts")]],
      ]),
    );
    const root = dir({
      "tsconfig.json": JSON.stringify({ compilerOptions: { paths } }),
      "greet.mjs": `export default async function greet(who) { return "v1 " + who; }`,
      "check.mts": `
import { writeFileSync } from "node:fs";
import { createStandaloneHost } from ${JSON.stringify(join(__dirname, "..", "src", "standalone.ts"))};
const lines: string[] = [];
const host = createStandaloneHost({ cwd: process.cwd(), write: (l) => lines.push(l) });
const out: unknown[] = [];
out.push(await host.workflow("./greet.mjs", { who: "ada" }));
out.push(await host.workflow("./greet.mjs", { who: "ada" }));
writeFileSync("greet.mjs", 'export default async function greet(who) { return "v2 " + who; }');
out.push(await host.workflow("./greet.mjs", { who: "ada" }));
console.log(JSON.stringify({ out, lines }));
`,
    });
    const tsx = pathToFileURL(createRequire(import.meta.url).resolve("tsx/esm")).href;
    const printed = execFileSync(process.execPath, ["--import", tsx, join(root, "check.mts")], { cwd: root, encoding: "utf8" });
    const { out, lines } = JSON.parse(printed.trim().split("\n").pop()!) as { out: string[]; lines: string[] };
    expect(out).toEqual(["v1 ada", "v1 ada", "v2 ada"]);
    expect(lines).toEqual([expect.stringMatching(/running a newer version/)]);
  });

  it("resolves input() idle when nobody answers in time, and null at the end of input", async () => {
    const host = createStandaloneHost({ read: () => new Promise(() => undefined), write: () => undefined });
    expect(await host.input!({ idle: "20ms" })).toEqual({ idle: true });
    const ended = createStandaloneHost({ read: async () => null, write: () => undefined });
    expect(await ended.input!({})).toBeNull();
    expect(idleMs("4h")).toBe(14_400_000);
    expect(() => idleMs("soon")).toThrow(/duration/);
  });

  it("runs agent() through the agent query, null when it fails — Claude's contract", async () => {
    const ok = createStandaloneHost({
      write: () => undefined,
      agentQuery: () =>
        (async function* () {
          yield { type: "result", result: { text: "done", costUsd: 0.5 } } as never;
        })() as never,
    });
    expect(await ok.agent("do it", {}, undefined)).toBe("done");
    expect(ok.budget.spent()).toBe(0.5);
    const bad = createStandaloneHost({
      write: () => undefined,
      agentQuery: () =>
        (async function* () {
          yield { type: "other", error: "the binary is missing" } as never;
        })() as never,
    });
    expect(await bad.agent("do it", {}, undefined)).toBeNull();
  });

  it("refuses a registered function plainly, since only an engine registers them", async () => {
    const host = createStandaloneHost({ write: () => undefined });
    await expect(host.call("summarize", [])).rejects.toThrow(/registered function/);
  });
});
