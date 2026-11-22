/**
 * A replayed conversation written as the agent's own session file (`nativeSeed.ts`).
 *
 * What it guards: the file has to be one a resumed `claude` sends to the Messages API without being
 * refused, so a cut conversation's loose ends are repaired, and a transcript not in the API's shape is
 * not written at all (the caller renders it instead, which always works).
 */
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { JsonValue } from "@declarative-ai/exec";
import { nativeSessionPath, seedLines, seedTurns, writeNativeSeed } from "../src/index.js";

const call = (id: string, name = "shell") => ({ type: "tool_use", id, name, input: {} });
const result = (id: string, text = "ok") => ({ type: "tool_result", tool_use_id: id, content: text });

describe("seedTurns", () => {
  it("keeps a transcript in the agent's own shape as it is", () => {
    const prior: JsonValue[] = [
      { role: "user", content: "push it" },
      { role: "assistant", content: [{ type: "thinking", thinking: "", signature: "sig" }] },
      { role: "assistant", content: [call("t1")] },
      { role: "user", content: [result("t1")] },
      { role: "assistant", content: [{ type: "text", text: "pushed" }] },
    ];
    expect(seedTurns(prior)).toEqual(prior);
  });

  it("answers a call a stop cut off, and drops a result whose call is not there", () => {
    const turns = seedTurns([
      { role: "user", content: [result("gone")] },
      { role: "assistant", content: [call("t1"), call("t2")] },
      { role: "user", content: [result("t1")] },
      { role: "user", content: "[Request interrupted by user for tool use]" },
      { role: "assistant", content: [call("t3")] },
    ]);
    expect(turns).toEqual([
      { role: "assistant", content: [call("t1"), call("t2")] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t2", content: expect.any(String), is_error: true }] },
      { role: "user", content: [result("t1")] },
      { role: "user", content: "[Request interrupted by user for tool use]" },
      { role: "assistant", content: [call("t3")] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "t3", content: expect.any(String), is_error: true }] },
    ]);
  });

  it("leaves out what the host said on the agent's behalf", () => {
    expect(seedTurns([{ role: "assistant", content: [call("ask")], calledBy: "t1" }, { role: "user", content: "hi" }])).toEqual([{ role: "user", content: "hi" }]);
  });

  it("refuses a transcript that is not in the API's shape, so the caller renders it instead", () => {
    expect(seedTurns([{ role: "assistant", content: [{ type: "tool-call", toolCallId: "x", toolName: "shell", input: {} }] }])).toBeUndefined();
    expect(seedTurns([{ role: "system", content: "be brief" }])).toBeUndefined();
  });
});

describe("seedLines", () => {
  it("is one chain, and gives consecutive assistant turns one message id", () => {
    let n = 0;
    const lines = seedLines(
      [
        { role: "user", content: "go" },
        { role: "assistant", content: [{ type: "thinking", thinking: "", signature: "s" }] as never },
        { role: "assistant", content: [call("t1")] as never },
        { role: "user", content: [result("t1")] as never },
        { role: "assistant", content: [{ type: "text", text: "done" }] as never },
      ],
      { sessionId: "S", cwd: "C:\\work", at: new Date(0), uuid: () => `u${++n}` },
    ) as Array<{ uuid: string; parentUuid: string | null; sessionId: string; type: string; message: { id?: string } }>;
    expect(lines.map((l) => [l.parentUuid, l.uuid])).toEqual([
      [null, "u1"],
      ["u1", "u2"],
      ["u2", "u4"],
      ["u4", "u5"],
      ["u5", "u6"],
    ]);
    expect(lines.every((l) => l.sessionId === "S")).toBe(true);
    const ids = lines.map((l) => l.message.id);
    expect(ids[1]).toBe(ids[2]);
    expect(ids[4]).toBeDefined();
    expect(ids[4]).not.toBe(ids[1]);
  });
});

describe("writeNativeSeed", () => {
  let root: string;
  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "native-seed-"));
  });
  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("writes the session where a resume in that directory reads it, and answers its id", async () => {
    const id = await writeNativeSeed([{ role: "user", content: "hello" }, { role: "assistant", content: [{ type: "text", text: "hi" }] }], "C:\\proj", { configDir: root });
    expect(id).toMatch(/^[0-9a-f-]{36}$/);
    const lines = (await readFile(nativeSessionPath(id!, "C:\\proj", root), "utf8")).trim().split("\n").map((l) => JSON.parse(l));
    expect(lines.map((l) => [l.type, l.cwd, l.sessionId])).toEqual([
      ["user", "C:\\proj", id],
      ["assistant", "C:\\proj", id],
    ]);
  });

  it("writes nothing for a transcript it cannot seed", async () => {
    expect(await writeNativeSeed([{ role: "system", content: "x" }], "C:\\proj", { configDir: root })).toBeUndefined();
    expect(await writeNativeSeed([], "C:\\proj", { configDir: root })).toBeUndefined();
  });
});
