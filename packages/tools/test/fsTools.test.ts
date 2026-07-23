import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { ExecServices } from "@declarative-ai/exec";
import {
  allTools,
  editFileTool,
  fsTools,
  globTool,
  grepTool,
  listDirTool,
  readFileTool,
  requireWorkspace,
  resolveInWorkspace,
  writeFileTool,
} from "../src";

let root: string;
beforeAll(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), "dai-tools-"));
});
afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});
const ctx = (): ExecServices => ({ workspace: { root } });

describe("resolveInWorkspace — path guard (SPEC §7.2)", () => {
  it("resolves an in-root relative path", () => {
    expect(resolveInWorkspace(root, "a/b.txt")).toBe(path.resolve(root, "a/b.txt"));
    expect(resolveInWorkspace(root, ".")).toBe(path.resolve(root));
  });

  it("throws on `..` escape and a sibling dir that merely shares the root's name prefix", () => {
    expect(() => resolveInWorkspace(root, "../evil")).toThrow(/escapes/);
    // `<root>-evil` starts with the root string but is NOT inside it — the separator check catches it.
    expect(() => resolveInWorkspace("/repo", "../repo-evil/x")).toThrow(/escapes/);
  });
});

describe("fs tools — round-trip on a real workspace", () => {
  it("write_file creates parent dirs and reports bytes; read_file returns the content", async () => {
    const wrote = (await writeFileTool.run({ path: "sub/hello.txt", content: "hi there" }, ctx())) as Record<string, unknown>;
    expect(wrote).toMatchObject({ written: true, bytes: 8 });
    const read = (await readFileTool.run({ path: "sub/hello.txt" }, ctx())) as Record<string, unknown>;
    expect(read.content).toBe("hi there");
  });

  it("write_file accepts empty content", async () => {
    await writeFileTool.run({ path: "empty.txt", content: "" }, ctx());
    const read = (await readFileTool.run({ path: "empty.txt" }, ctx())) as Record<string, unknown>;
    expect(read.content).toBe("");
  });

  it("list_dir lists files and directories (default the root)", async () => {
    await mkdir(path.join(root, "adir"), { recursive: true });
    await writeFile(path.join(root, "afile.txt"), "x");
    const out = (await listDirTool.run({}, ctx())) as { entries: Array<{ name: string; type: string }> };
    const byName = Object.fromEntries(out.entries.map((e) => [e.name, e.type]));
    expect(byName["adir"]).toBe("dir");
    expect(byName["afile.txt"]).toBe("file");
  });

  it("a tool path input that escapes the workspace is rejected", async () => {
    await expect(readFileTool.run({ path: "../../etc/passwd" }, ctx())).rejects.toThrow(/escapes/);
  });
});

describe("capabilities + workspace requirement", () => {
  it("read_file and list_dir are read-only; write_file is not", () => {
    expect(readFileTool.readOnly).toBe(true);
    expect(listDirTool.readOnly).toBe(true);
    expect(writeFileTool.readOnly).toBe(false);
  });

  it("fsTools + allTools expose the sets keyed by logical name", () => {
    expect(Object.keys(fsTools).sort()).toEqual(["edit_file", "list_dir", "read_file", "write_file"]);
    expect(Object.keys(allTools).sort()).toEqual(["edit_file", "glob", "grep", "list_dir", "read_file", "run_command", "write_file"]);
  });

  it("requireWorkspace throws with a clear message when no workspace is configured", () => {
    expect(() => requireWorkspace({})).toThrow(/needs a workspace/);
  });
});

describe("edit_file", () => {
  it("replaces a unique substring and reports one replacement", async () => {
    await writeFileTool.run({ path: "edit/a.txt", content: "alpha beta gamma" }, ctx());
    const out = (await editFileTool.run({ path: "edit/a.txt", old_string: "beta", new_string: "BETA" }, ctx())) as Record<string, unknown>;
    expect(out).toMatchObject({ replacements: 1 });
    const read = (await readFileTool.run({ path: "edit/a.txt" }, ctx())) as Record<string, unknown>;
    expect(read.content).toBe("alpha BETA gamma");
  });

  it("refuses a non-unique match without replace_all, but replaces all with it", async () => {
    await writeFileTool.run({ path: "edit/b.txt", content: "x x x" }, ctx());
    await expect(editFileTool.run({ path: "edit/b.txt", old_string: "x", new_string: "y" }, ctx())).rejects.toThrow(/occurs 3 times/);
    const out = (await editFileTool.run({ path: "edit/b.txt", old_string: "x", new_string: "y", replace_all: true }, ctx())) as Record<string, unknown>;
    expect(out).toMatchObject({ replacements: 3 });
    expect(((await readFileTool.run({ path: "edit/b.txt" }, ctx())) as Record<string, unknown>).content).toBe("y y y");
  });

  it("errors when old_string is absent from the file", async () => {
    await writeFileTool.run({ path: "edit/c.txt", content: "hello" }, ctx());
    await expect(editFileTool.run({ path: "edit/c.txt", old_string: "nope", new_string: "x" }, ctx())).rejects.toThrow(/not found/);
  });
});

describe("grep + glob", () => {
  beforeAll(async () => {
    await mkdir(path.join(root, "search/nested"), { recursive: true });
    await writeFile(path.join(root, "search/one.ts"), "const needle = 1;\nconst other = 2;\n");
    await writeFile(path.join(root, "search/nested/two.ts"), "// needle here too\n");
    await writeFile(path.join(root, "search/readme.md"), "no match here\n");
  });

  it("grep finds a pattern across the tree, reporting posix paths + 1-based line numbers", async () => {
    const out = (await grepTool.run({ pattern: "needle", path: "search" }, ctx())) as {
      matches: Array<{ file: string; line: number; text: string }>;
    };
    const files = out.matches.map((m) => m.file).sort();
    expect(files).toContain("search/one.ts");
    expect(files).toContain("search/nested/two.ts");
    expect(out.matches.find((m) => m.file === "search/one.ts")?.line).toBe(1);
  });

  it("grep honors regex flags and reports an invalid pattern clearly", async () => {
    const ci = (await grepTool.run({ pattern: "NEEDLE", path: "search", flags: "i" }, ctx())) as { matches: unknown[] };
    expect(ci.matches.length).toBeGreaterThan(0);
    await expect(grepTool.run({ pattern: "(unclosed", path: "search" }, ctx())).rejects.toThrow(/invalid grep pattern/);
  });

  it("glob matches by path with ** across dirs and * within a segment", async () => {
    const ts = (await globTool.run({ pattern: "search/**/*.ts" }, ctx())) as { files: string[] };
    expect(ts.files.sort()).toEqual(["search/nested/two.ts", "search/one.ts"]);
    const top = (await globTool.run({ pattern: "search/*.md" }, ctx())) as { files: string[] };
    expect(top.files).toEqual(["search/readme.md"]);
  });
});
