/**
 * A script run as a plain program REPORTS to an engine that takes reports (SCRIPTS.md §17.4): it finds
 * the endpoint an engine advertised, tells it each call and what it said, asks it about an agent's tool
 * uses, and says how the run ended — and it still runs its own calls. With no endpoint, or one that will
 * not take it, it is the plain program it always was.
 */
import { createServer, type Server, type Socket } from "node:net";
import { chmodSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { AgentQuery } from "@declarative-ai/agents-api";
import { advertiseEndpoint, endpointsDir, findEndpoints, REPORT_PROTOCOL, runScript } from "../src/index.js";

function dir(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmpdir(), "dai-report-"));
  for (const [name, text] of Object.entries(files)) writeFileSync(join(root, name), text);
  return root;
}

const CHAT = `
import { llm, input, phase } from "@declarative-ai/hw/script";
export const meta = { name: "Chat", environment: { tools: "$/permission-sets/ask-first" } };
export default async function chat(first: string): Promise<string[]> {
  phase("talking");
  let msg: { text: string } | null = { text: first };
  let conv: { id: string } | null = null;
  let replies: string[] = [];
  while (msg !== null) {
    const turn = await llm<string>({ prompt: msg.text, session: conv, withSession: true });
    conv = turn.session;
    replies = [...replies, turn.value];
    const next = await input({ prompt: "next?" });
    msg = next !== null && next.idle !== true ? { text: next.text } : null;
  }
  return replies;
}
`;

const WORKER = `
import { agent } from "@declarative-ai/hw/script";
export default async function work(task: string): Promise<unknown> {
  return agent(task);
}
`;

/** An endpoint that records what it is told and answers as `answer` says. */
function endpoint(answer: (method: string, params: Record<string, unknown>) => unknown = () => ({})): { server: Server; socket: string; heard: Array<{ method: string; params: Record<string, unknown> }> } {
  const heard: Array<{ method: string; params: Record<string, unknown> }> = [];
  const socket = join(mkdtempSync(join(tmpdir(), "dai-ep-")), "e.sock");
  const server = createServer((conn: Socket) => {
    let buffer = "";
    conn.setEncoding("utf8");
    conn.on("data", (chunk: string) => {
      buffer += chunk;
      for (let at = buffer.indexOf("\n"); at >= 0; at = buffer.indexOf("\n")) {
        const line = buffer.slice(0, at);
        buffer = buffer.slice(at + 1);
        const frame = JSON.parse(line) as { id?: number; method: string; params: Record<string, unknown> };
        heard.push({ method: frame.method, params: frame.params });
        if (frame.id === undefined) continue;
        try {
          conn.write(`${JSON.stringify({ id: frame.id, result: answer(frame.method, frame.params) })}\n`);
        } catch (e) {
          conn.write(`${JSON.stringify({ id: frame.id, error: { message: (e as Error).message } })}\n`);
        }
      }
    });
  });
  server.listen(socket);
  servers.push(server);
  return { server, socket, heard };
}

const servers: Server[] = [];
afterEach(() => {
  for (const server of servers.splice(0)) server.close();
});

const advert = (socket: string, token = "secret") => JSON.stringify({ name: "Engine", socket, token, protocol: REPORT_PROTOCOL, pid: process.pid });

describe("a run reported to an engine", () => {
  it("says hello with the script and its meta, tells each call and what it said, its phases and its input, and how it ended", async () => {
    const root = dir({ "chat.ts": CHAT });
    const { socket, heard } = endpoint((method, params) => {
      if (method === "hello") {
        if (params["token"] !== "secret") throw new Error("not you");
        return { run: "r1", name: "Engine", url: "engine://r1" };
      }
      return {};
    });
    const lines = ["second", null];
    let n = 0;
    const written: string[] = [];
    const result = await runScript(join(root, "chat.ts"), ["first"], {
      model: "test/model",
      execute: async () => ({ value: `reply ${++n}`, costUsd: 0.01 }),
      read: async () => lines.shift() ?? null,
      write: (line) => written.push(line),
      env: { DAI_SCRIPT_ENDPOINT: advert(socket) },
    });
    expect(result).toEqual(["reply 1", "reply 2"]);
    expect(written).toContain("▸ reported to Engine — engine://r1");
    expect(heard[0]).toMatchObject({ method: "hello", params: { protocol: REPORT_PROTOCOL, run: { file: join(root, "chat.ts"), args: ["first"], meta: { name: "Chat", environment: { tools: "$/permission-sets/ask-first" } } } } });
    expect(heard.map((h) => h.method)).toEqual(["hello", "note", "call.started", "call.settled", "note", "call.started", "call.settled", "note", "end"]);
    const calls = heard.filter((h) => h.method === "call.started").map((h) => h.params["call"]);
    // The second continues the conversation the first ended at, named as the script holds it.
    expect(calls).toEqual([
      expect.objectContaining({ id: "c1", kind: "llm", model: "test/model", prompt: "first", session: null }),
      expect.objectContaining({ id: "c2", kind: "llm", prompt: "second", session: "s1@1" }),
    ]);
    expect(heard.find((h) => h.method === "call.settled")!.params).toEqual({
      id: "c1",
      outcome: { value: "reply 1", session: "s1@1", messages: [{ role: "user", content: "first" }, { role: "assistant", content: "reply 1" }], costUsd: 0.01 },
    });
    expect(heard.filter((h) => h.method === "note").map((h) => h.params)).toEqual([
      { kind: "phase", title: "talking" },
      { kind: "input", prompt: "next?", answer: { text: "second" } },
      { kind: "input", prompt: "next?", answer: null },
    ]);
    expect(heard.at(-1)).toEqual({ method: "end", params: { value: ["reply 1", "reply 2"] } });
  });

  it("asks the engine about each tool an agent wants to use, and the agent hears its answer", async () => {
    const root = dir({ "work.ts": WORKER });
    const { socket, heard } = endpoint((method, params) => (method === "hello" ? { run: "r2" } : method === "approve" ? (params["toolName"] === "Bash" ? { allow: false, reason: "not here" } : { allow: true }) : {}));
    const answers: unknown[] = [];
    const agentQuery: AgentQuery = (opts) =>
      (async function* () {
        for (const toolName of ["Read", "Bash"]) answers.push(await opts.canUseTool!({ toolName, input: { command: "ls" } }, { signal: new AbortController().signal }));
        yield { type: "result", result: { text: "done", costUsd: 0 } } as never;
      })() as never;
    const result = await runScript(join(root, "work.ts"), ["tidy up"], { agentQuery, write: () => undefined, env: { DAI_SCRIPT_ENDPOINT: advert(socket) } });
    expect(result).toBe("done");
    expect(answers).toEqual([{ allow: true }, { allow: false, reason: "not here" }]);
    expect(heard.filter((h) => h.method === "approve").map((h) => h.params)).toEqual([
      { id: "c1", toolName: "Read", input: { command: "ls" } },
      { id: "c1", toolName: "Bash", input: { command: "ls" } },
    ]);
    expect(heard.find((h) => h.method === "call.started")!.params["call"]).toMatchObject({ id: "c1", kind: "agent", prompt: "tidy up" });
  });

  it("says a run that failed ended in its error", async () => {
    const root = dir({ "chat.ts": CHAT });
    const { socket, heard } = endpoint((method) => (method === "hello" ? { run: "r3" } : {}));
    await expect(
      runScript(join(root, "chat.ts"), ["first"], { model: "test/model", execute: async () => ({ error: "rate limited" }), write: () => undefined, env: { DAI_SCRIPT_ENDPOINT: advert(socket) } }),
    ).rejects.toThrow(/rate limited/);
    // The asked turn stays in the conversation, at the position the failed call ended at (SCRIPTS.md §9).
    expect(heard.find((h) => h.method === "call.settled")!.params).toEqual({ id: "c1", outcome: { error: "rate limited", session: "s1@1", messages: [{ role: "user", content: "first" }] } });
    expect(heard.at(-1)).toEqual({ method: "end", params: { error: "rate limited" } });
  });

  it("runs as a plain program when the endpoint will not take it, or none is listening", async () => {
    const root = dir({ "chat.ts": CHAT });
    const { socket, heard } = endpoint(() => {
      throw new Error("not you");
    });
    const run = (env: NodeJS.ProcessEnv) => runScript(join(root, "chat.ts"), ["first"], { model: "test/model", execute: async () => ({ value: "hi" }), read: async () => null, write: () => undefined, env });
    expect(await run({ DAI_SCRIPT_ENDPOINT: advert(socket, "wrong") })).toEqual(["hi"]);
    expect(heard.map((h) => h.method)).toEqual(["hello"]);
    expect(await run({ DAI_SCRIPT_ENDPOINT: advert(join(root, "nobody.sock")) })).toEqual(["hi"]);
    expect(await run({ DAI_SCRIPT_ENDPOINT: advert(socket), DAI_SCRIPT_REPORT: "0" })).toEqual(["hi"]);
    expect(heard).toHaveLength(1);
  });
});

describe("finding an endpoint", () => {
  it("reads the adverts in $DAI_ENDPOINTS, passing over one whose process is gone or that speaks another protocol", () => {
    const dir = mkdtempSync(join(tmpdir(), "dai-endpoints-"));
    writeFileSync(join(dir, "live.json"), JSON.stringify({ name: "Live", socket: "/tmp/a.sock", token: "t", protocol: REPORT_PROTOCOL, pid: process.pid }));
    writeFileSync(join(dir, "gone.json"), JSON.stringify({ name: "Gone", socket: "/tmp/b.sock", token: "t", protocol: REPORT_PROTOCOL, pid: 2 ** 22 + 12345 }));
    writeFileSync(join(dir, "newer.json"), JSON.stringify({ name: "Newer", socket: "/tmp/c.sock", token: "t", protocol: REPORT_PROTOCOL + 1, pid: process.pid }));
    writeFileSync(join(dir, "junk.json"), "{");
    expect(findEndpoints({ DAI_ENDPOINTS: dir }).map((a) => a.name)).toEqual(["Live"]);
    expect(findEndpoints({ DAI_ENDPOINTS: dir, DAI_SCRIPT_REPORT: "0" })).toEqual([]);
  });

  it("keeps adverts in this user's directory in the system's temporary one, not in the home directory", () => {
    const dir = endpointsDir({});
    expect(dir.startsWith(homedir())).toBe(false);
    if (process.platform === "win32") expect(dir).toBe(join(tmpdir(), "declarative-ai-endpoints"));
    else expect(dir).toBe(`/tmp/declarative-ai-endpoints-${process.getuid!()}`);
  });

  it("advertises in a directory it makes this user's alone, and a run finds what it advertised", () => {
    const dir = join(mkdtempSync(join(tmpdir(), "dai-endpoints-")), "endpoints");
    const file = advertiseEndpoint("engine", { name: "Engine", socket: "/tmp/e.sock", token: "t", protocol: REPORT_PROTOCOL, pid: process.pid, version: "1.2.3" }, { DAI_ENDPOINTS: dir });
    expect(file).toBe(join(dir, "engine.json"));
    expect(readdirSync(dir)).toEqual(["engine.json"]);
    if (process.platform !== "win32") {
      expect(statSync(dir).mode & 0o777).toBe(0o700);
      expect(statSync(file).mode & 0o777).toBe(0o600);
    }
    expect(findEndpoints({ DAI_ENDPOINTS: dir }).map((a) => a.name)).toEqual(["Engine"]);
    // Advertising again — as an engine does while it listens — puts back an advert that was cleaned away.
    rmSync(file);
    advertiseEndpoint("engine", { name: "Engine", socket: "/tmp/e.sock", token: "t", protocol: REPORT_PROTOCOL, pid: process.pid }, { DAI_ENDPOINTS: dir });
    expect(findEndpoints({ DAI_ENDPOINTS: dir }).map((a) => a.name)).toEqual(["Engine"]);
  });

  it.skipIf(process.platform === "win32")("trusts no advert in a directory others can write to, and advertises nothing there", () => {
    const dir = mkdtempSync(join(tmpdir(), "dai-endpoints-"));
    writeFileSync(join(dir, "planted.json"), JSON.stringify({ name: "Planted", socket: "/tmp/p.sock", token: "t", protocol: REPORT_PROTOCOL, pid: process.pid }));
    chmodSync(dir, 0o777);
    expect(findEndpoints({ DAI_ENDPOINTS: dir })).toEqual([]);
    expect(() => advertiseEndpoint("engine", { name: "Engine", socket: "/tmp/e.sock", token: "t", protocol: REPORT_PROTOCOL }, { DAI_ENDPOINTS: dir })).toThrow(/not this user's alone/);
  });
});
