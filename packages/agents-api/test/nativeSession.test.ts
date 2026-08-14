/**
 * Reading the agent's own session file back, and folding it down to what the stream never carried.
 *
 * The reader runs against a REAL temp directory rather than a fake fs, because the thing under test
 * is precisely the path arithmetic — the dashed cwd encoding, the per-project folder, the id-only
 * search — and a fake that reimplements it would prove the fake.
 */
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { encodeSessionCwd, nativeLinesOf, nativeSessionPath, nativeSessionReader, readNativeSidechains } from "../src/nativeSession.js";

describe("encodeSessionCwd", () => {
  it("spells a cwd the way the agent names its project folder — every non-alphanumeric becomes a dash", () => {
    // Verified against a real folder on disk: C:\Users\Ofer\.jaira\workflows is kept at
    // C--Users-Ofer--jaira-workflows — the drive colon, the separators AND the dot all dash.
    expect(encodeSessionCwd("C:\\Users\\Ofer\\.jaira\\workflows")).toBe("C--Users-Ofer--jaira-workflows");
    expect(encodeSessionCwd("/home/me/project")).toBe("-home-me-project");
  });
});

describe("nativeSessionReader", () => {
  let root: string;
  const id = "f8f054dc-0000-4ac7-8a14-c9a19a36209a";
  const cwd = "C:\\work\\proj";
  const lines = [{ type: "user", message: { role: "user", content: "hi" } }, { type: "ai-title", aiTitle: "T" }];

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "native-session-"));
    const folder = join(root, "projects", encodeSessionCwd(cwd));
    await mkdir(folder, { recursive: true });
    await writeFile(join(folder, `${id}.jsonl`), `${lines.map((l) => JSON.stringify(l)).join("\n")}\nnot json\n`, "utf8");
  });
  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("addresses the file by cwd, parses each line, and keeps an unparseable one as the raw string", async () => {
    const read = nativeSessionReader({ configDir: root });
    expect(await read(id, cwd)).toEqual([...lines, "not json"]);
  });

  it("finds the file by id alone when no cwd is given — the seam forwards only the id", async () => {
    const read = nativeSessionReader({ configDir: root });
    expect((await read(id)).length).toBe(3);
  });

  it("reads a missing conversation as [], the seam's documented shape for nothing", async () => {
    const read = nativeSessionReader({ configDir: root });
    expect(await read("no-such-id", cwd)).toEqual([]);
    expect(await read(id, "D:\\elsewhere")).toEqual([]);
    expect(await read("no-such-id")).toEqual([]);
  });

  it("refuses an id that could step out of the projects tree", async () => {
    const read = nativeSessionReader({ configDir: root });
    expect(await read("..", cwd)).toEqual([]);
    expect(await read("a/../../b", cwd)).toEqual([]);
  });

  it("computes the documented path", () => {
    expect(nativeSessionPath(id, cwd, root)).toBe(join(root, "projects", "C--work-proj", `${id}.jsonl`));
  });
});

describe("readNativeSidechains", () => {
  let root: string;
  const id = "11111111-2222-4333-8444-555555555555";
  const cwd = "C:\\work\\proj";
  const chainLine = { type: "assistant", isSidechain: true, agentId: "abc123", message: { role: "assistant", content: "aside" } };

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "native-side-"));
    const project = join(root, "projects", encodeSessionCwd(cwd));
    const dir = join(project, id, "subagents");
    await mkdir(dir, { recursive: true });
    // The main file, so id-only search can find the project folder.
    await writeFile(join(project, `${id}.jsonl`), `${JSON.stringify({ type: "user", message: {} })}\n`, "utf8");
    await writeFile(join(dir, "agent-abc123.jsonl"), `${JSON.stringify(chainLine)}\n`, "utf8");
    await writeFile(
      join(dir, "agent-abc123.meta.json"),
      JSON.stringify({ agentType: "Explore", description: "map things", toolUseId: "toolu_01X", spawnDepth: 1 }),
      "utf8",
    );
    // A spawn whose meta file is gone: the lines survive, only the join is lost.
    await writeFile(join(dir, "agent-nometa.jsonl"), `${JSON.stringify({ ...chainLine, agentId: "nometa" })}\n`, "utf8");
  });
  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("reads every subagent file, joined to its spawning tool call by the meta sidecar", async () => {
    const chains = await readNativeSidechains(id, cwd, root);
    expect(chains.map((c) => c.agentId)).toEqual(["abc123", "nometa"]);
    expect(chains[0]).toMatchObject({ toolUseId: "toolu_01X", meta: { agentType: "Explore" } });
    expect(chains[0]!.lines).toHaveLength(1);
    // Meta gone ⇒ the lines are kept and only the join key is absent — never the conversation.
    expect(chains[1]!.toolUseId).toBeUndefined();
  });

  it("finds the project folder from the id alone, through the main session file beside the subagents", async () => {
    const chains = await readNativeSidechains(id, undefined, root);
    expect(chains).toHaveLength(2);
  });

  it("reads a run that spawned nothing as [] — the ordinary case", async () => {
    expect(await readNativeSidechains("no-such-id", cwd, root)).toEqual([]);
  });
});

describe("nativeLinesOf", () => {
  const at = (s: number): string => new Date(60_000 + s * 1000).toISOString();
  const file = [
    { type: "queue-operation", operation: "enqueue", timestamp: at(0) },
    { type: "user", uuid: "u1", parentUuid: null, timestamp: at(1), message: { role: "user", content: "prompt" } },
    { type: "attachment", uuid: "a1", timestamp: at(2), attachment: { type: "skill_listing" } },
    { type: "assistant", uuid: "s1", parentUuid: "u1", timestamp: at(3), message: { role: "assistant", content: "…" } },
    { type: "user", uuid: "u2", timestamp: at(4), toolUseResult: { stdout: "ok" }, message: { role: "user", content: [] } },
    { type: "last-prompt", lastPrompt: "prompt" },
    { type: "ai-title", aiTitle: "Title" },
    { type: "assistant", uuid: "s2", timestamp: at(7), message: { role: "assistant", content: "done" } },
  ];

  it("keeps non-message lines whole and message ENVELOPES only — the body already rides the stream", () => {
    const kept = nativeLinesOf(file);
    expect(kept.map((k) => (k.line as { type?: string }).type)).toEqual([
      "queue-operation",
      "user",
      "attachment",
      "assistant",
      "user",
      "last-prompt",
      "ai-title",
      "assistant",
    ]);
    // Envelope, not body: threading and toolUseResult survive, `message` does not.
    const tool = kept[4]!.line as Record<string, unknown>;
    expect(tool["toolUseResult"]).toEqual({ stdout: "ok" });
    expect(tool["message"]).toBeUndefined();
    // The attachment is kept WHOLE — it exists nowhere else.
    expect((kept[2]!.line as { attachment?: unknown }).attachment).toEqual({ type: "skill_listing" });
  });

  it("pins each line to how many main-chain message lines preceded it in the file", () => {
    const kept = nativeLinesOf(file);
    expect(kept.map((k) => k.index)).toEqual([0, 0, 1, 1, 2, 3, 3, 3]);
  });

  it("cuts a resumed session's file at the call's start, inheriting stamps for unstamped lines", () => {
    // Everything before at(4) belongs to earlier records; last-prompt/ai-title carry no stamp of
    // their own and ride on the last one seen (at(4), then at(7)).
    const kept = nativeLinesOf(file, { sinceMs: 60_000 + 4000 });
    expect(kept.map((k) => (k.line as { type?: string }).type)).toEqual(["user", "last-prompt", "ai-title", "assistant"]);
    // File positions are kept, not renumbered — the record's reader pairs by role and order.
    expect(kept.map((k) => k.index)).toEqual([2, 3, 3, 3]);
  });

  it("keeps lines that precede any stamp — the head of a fresh file is the capture's own material", () => {
    const kept = nativeLinesOf([{ type: "summary", summary: "S" }, ...file], { sinceMs: 0 });
    expect((kept[0]!.line as { type?: string }).type).toBe("summary");
  });

  it("keeps a sidechain line's envelope without advancing the main-chain count", () => {
    const kept = nativeLinesOf([
      { type: "assistant", uuid: "s1", message: { role: "assistant", content: "main" } },
      { type: "assistant", uuid: "x1", isSidechain: true, message: { role: "assistant", content: "aside" } },
      { type: "assistant", uuid: "s2", message: { role: "assistant", content: "main2" } },
    ]);
    expect(kept.map((k) => k.index)).toEqual([0, 1, 1]);
    expect((kept[1]!.line as { message?: unknown }).message).toBeUndefined();
  });

  it("counts a SUBAGENT file's own lines when folding as a sidechain — there they ARE the chain", () => {
    const kept = nativeLinesOf(
      [
        { type: "user", uuid: "u1", isSidechain: true, message: { role: "user", content: "task" } },
        { type: "attachment", uuid: "a1", isSidechain: true, attachment: { type: "skill_listing" } },
        { type: "assistant", uuid: "s1", isSidechain: true, message: { role: "assistant", content: "…" } },
      ],
      { sidechain: true },
    );
    expect(kept.map((k) => k.index)).toEqual([0, 1, 1]);
    expect((kept[0]!.line as { message?: unknown }).message).toBeUndefined();
    expect((kept[1]!.line as { attachment?: unknown }).attachment).toEqual({ type: "skill_listing" });
  });
});
