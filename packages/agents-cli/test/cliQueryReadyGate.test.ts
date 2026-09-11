/**
 * The prompt waits for the bridge handshake.
 *
 * The race this closes: the CLI starts the model turn the moment it has input and resolves the
 * permission tool lazily — at the first permission decision, from the servers connected BY THEN. A
 * bridge that reports its handshake (`McpBridge.ready`) gets the prompt after it; one that cannot
 * gets it at spawn, exactly as before.
 */
import { describe, expect, it } from "vitest";
import { createCliAgentQuery } from "../src/cliQuery.js";
import type { AgentProcess, SpawnProcess } from "../src/process.js";

/**
 * A fake process that replays scripted stdout lines and records what reached its input.
 *
 * Its `exit` settles only once its output has been read — a process that has "exited" before it was
 * spawned would release the gate on its own (a dead child reads nothing, so the wait ends), which is
 * the right rule and the wrong fake.
 */
function fakeSpawn(lines: string[]): { spawn: SpawnProcess; stdins: (string | undefined)[]; written: string[] } {
  const stdins: (string | undefined)[] = [];
  const written: string[] = [];
  const spawn: SpawnProcess = (_argv, opts) => {
    stdins.push(opts.stdin);
    let exited: (code: number) => void = () => undefined;
    const exit = new Promise<number>((resolve) => {
      exited = resolve;
    });
    const proc: AgentProcess = {
      lines: (async function* () {
        for (const l of lines) yield l;
        exited(0);
      })(),
      kill: () => exited(0),
      exit,
      write: (line) => void written.push(line),
      endInput: () => undefined,
    };
    return proc;
  };
  return { spawn, stdins, written };
}

function gatedBridge(): { startBridge: () => Promise<{ url: string; ready: Promise<void>; close: () => Promise<void> }>; ready: () => void } {
  let markReady: () => void = () => undefined;
  const ready = new Promise<void>((resolve) => {
    markReady = resolve;
  });
  const startBridge = async () => ({ url: "http://127.0.0.1:9999/mcp/t", ready, close: async () => undefined });
  return { startBridge, ready: () => markReady() };
}

const approver = { canUseTool: async () => ({ allow: true as const }) };

describe("the prompt waits for the bridge handshake", () => {
  it("sends the prompt through the input channel AFTER ready, and nothing at spawn", async () => {
    const { spawn, stdins, written } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge, ready } = gatedBridge();
    const it = createCliAgentQuery({ spawn, startBridge })({ prompt: "go", ...approver })[Symbol.asyncIterator]();
    const first = it.next();
    // Spawned and gated: nothing has been written yet.
    await new Promise((r) => setTimeout(r, 10));
    expect(stdins[0]).toBeUndefined();
    expect(written).toEqual([]);
    ready();
    await first;
    expect(JSON.parse(written[0]!)).toMatchObject({ type: "user", message: { role: "user", content: "go" } });
    await it.return?.(undefined as never);
  });

  it("gives up waiting at the bound and sends the prompt anyway", async () => {
    const { spawn, written } = fakeSpawn(['{"type":"result","result":"done"}']);
    const { startBridge } = gatedBridge(); // never made ready
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn, startBridge, bridgeReadyTimeoutMs: 20 })({ prompt: "go", ...approver })) seen.push(m);
    expect(written).toHaveLength(1);
    expect(seen.at(-1)).toMatchObject({ type: "result" });
  });

  it("does not gate a bridge that reports no readiness — the prompt goes in at spawn, as before", async () => {
    const { spawn, stdins, written } = fakeSpawn(['{"type":"result","result":"done"}']);
    const startBridge = async () => ({ url: "http://127.0.0.1:9999/mcp", close: async () => undefined });
    for await (const _ of createCliAgentQuery({ spawn, startBridge })({ prompt: "go", ...approver })) void _;
    expect(JSON.parse(stdins[0]!)).toMatchObject({ message: { content: "go" } });
    expect(written).toEqual([]);
  });

  it("refuses, rather than hanging, when the seam pipes no input to gate on", async () => {
    const spawn: SpawnProcess = () => ({
      lines: (async function* () {})(),
      kill: () => undefined,
      exit: Promise.resolve(0),
    });
    const { startBridge } = gatedBridge();
    const seen = [];
    for await (const m of createCliAgentQuery({ spawn, startBridge })({ prompt: "go", ...approver })) seen.push(m);
    expect(seen).toEqual([{ type: "other", error: expect.stringMatching(/no input channel/) }]);
  });
});
