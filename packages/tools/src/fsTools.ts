/**
 * A starter Claude-Code-parity tool library (RUNTIMES-AND-PERMISSIONS.md §2, build order 3): cross-platform
 * fs {@link Tool}s that operate on `ctx.workspace`. These are the impls that turn the composed `llm` runtime
 * into a coding agent. Each is a plain core `Tool` (so it registers into `registry.tools` and is
 * permission-gated like any other). `read_file`/`list_dir` declare `readOnly` (in scope under the
 * `read-only`/`plan` profiles); `write_file` mutates. Shell/edit tools are follow-ups.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { Tool } from "@declarative-ai/core";
import { requireWorkspace, resolveInWorkspace } from "./workspace";

/** Require a string field (empty allowed — e.g. an empty file body). */
function str(input: Record<string, unknown>, key: string): string {
  const v = input[key];
  if (typeof v !== "string") throw new Error(`tool input '${key}' is required (string)`);
  return v;
}

/** Require a non-empty path field. */
function reqPath(input: Record<string, unknown>, key: string): string {
  const v = str(input, key);
  if (v.length === 0) throw new Error(`tool input '${key}' must be a non-empty path`);
  return v;
}

/** read_file — return the UTF-8 contents of a workspace file. Read-only. */
export const readFileTool: Tool = {
  description: "Read a UTF-8 text file from the workspace. Input: { path }.",
  inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
  capabilities: { readOnly: true },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const abs = resolveInWorkspace(ws.root, reqPath(input, "path"));
    return { content: await fs.readFile(abs, "utf8") };
  },
};

/** write_file — write UTF-8 contents to a workspace file, creating parent directories. Mutating. */
export const writeFileTool: Tool = {
  description: "Write a UTF-8 text file to the workspace, creating parent directories. Input: { path, content }.",
  inputSchema: {
    type: "object",
    properties: { path: { type: "string" }, content: { type: "string" } },
    required: ["path", "content"],
  },
  capabilities: { readOnly: false },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const abs = resolveInWorkspace(ws.root, reqPath(input, "path"));
    const content = str(input, "content");
    await fs.mkdir(path.dirname(abs), { recursive: true });
    await fs.writeFile(abs, content, "utf8");
    return { written: true, bytes: Buffer.byteLength(content, "utf8") };
  },
};

/** edit_file — replace an exact substring in a workspace file. Without `replace_all`, the match must be
 *  unique (0 or >1 occurrences is an error — the same contract as Claude Code's Edit). Mutating. */
export const editFileTool: Tool = {
  description:
    "Replace an exact substring in a workspace file. Input: { path, old_string, new_string, replace_all? }. Without replace_all, old_string must occur exactly once.",
  inputSchema: {
    type: "object",
    properties: {
      path: { type: "string" },
      old_string: { type: "string" },
      new_string: { type: "string" },
      replace_all: { type: "boolean" },
    },
    required: ["path", "old_string", "new_string"],
  },
  capabilities: { readOnly: false },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const abs = resolveInWorkspace(ws.root, reqPath(input, "path"));
    const oldStr = str(input, "old_string");
    const newStr = str(input, "new_string");
    if (oldStr.length === 0) throw new Error("'old_string' must be non-empty");
    const replaceAll = input.replace_all === true;
    const original = await fs.readFile(abs, "utf8");
    const count = original.split(oldStr).length - 1;
    if (count === 0) throw new Error("'old_string' not found in the file");
    if (!replaceAll && count > 1) throw new Error(`'old_string' occurs ${count} times; pass replace_all or make it unique`);
    const updated = replaceAll ? original.split(oldStr).join(newStr) : original.replace(oldStr, newStr);
    await fs.writeFile(abs, updated, "utf8");
    return { replacements: replaceAll ? count : 1 };
  },
};

/** list_dir — list the entries of a workspace directory (default the root). Read-only. */
export const listDirTool: Tool = {
  description: "List the entries of a workspace directory. Input: { path? } (default the workspace root).",
  inputSchema: { type: "object", properties: { path: { type: "string" } } },
  capabilities: { readOnly: true },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const rel = typeof input.path === "string" && input.path.length > 0 ? input.path : ".";
    const abs = resolveInWorkspace(ws.root, rel);
    const dirents = await fs.readdir(abs, { withFileTypes: true });
    return { entries: dirents.map((d) => ({ name: d.name, type: d.isDirectory() ? "dir" : "file" })) };
  },
};

/** The core fs tool set, keyed by the logical name a `runtime.tools` entry references. */
export const fsTools: Record<string, Tool> = {
  read_file: readFileTool,
  write_file: writeFileTool,
  edit_file: editFileTool,
  list_dir: listDirTool,
};
