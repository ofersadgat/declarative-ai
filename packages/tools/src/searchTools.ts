/**
 * Search tools (RUNTIMES-AND-PERMISSIONS.md §2): `grep` (content search) and `glob` (path match), both
 * read-only and pure-fs (no ripgrep/shell dependency, so they run identically on every platform). They
 * walk the workspace, skipping conventional heavy/generated directories, and report POSIX-style paths
 * relative to the workspace root regardless of the host separator.
 */
import fs from "node:fs/promises";
import path from "node:path";
import type { Tool } from "@declarative-ai/core";
import { requireWorkspace, resolveInWorkspace } from "./workspace";

/** Directory names skipped by both tools — generated/heavy trees an agent almost never wants to search. */
const DEFAULT_IGNORE: ReadonlySet<string> = new Set([".git", "node_modules", "dist", "build", ".next", "coverage"]);

/** Caps so a single call can't return an unbounded result to the model. */
const MAX_MATCHES = 200;
const MAX_FILES = 5000;

/** Recursively collect absolute file paths under `dir`, skipping ignored directories. */
async function walkFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  const rec = async (d: string): Promise<void> => {
    if (out.length >= MAX_FILES) return;
    let dirents;
    try {
      dirents = await fs.readdir(d, { withFileTypes: true });
    } catch {
      return; // unreadable dir — skip rather than fail the whole search
    }
    for (const ent of dirents) {
      if (out.length >= MAX_FILES) return;
      const abs = path.join(d, ent.name);
      if (ent.isDirectory()) {
        if (!DEFAULT_IGNORE.has(ent.name)) await rec(abs);
      } else if (ent.isFile()) {
        out.push(abs);
      }
    }
  };
  await rec(dir);
  return out;
}

/** Workspace-root-relative POSIX path (forward slashes on every platform). */
function toPosixRel(root: string, abs: string): string {
  return path.relative(root, abs).split(path.sep).join("/");
}

function reqString(input: Record<string, unknown>, key: string): string {
  const v = input[key];
  if (typeof v !== "string" || v.length === 0) throw new Error(`tool input '${key}' is required (non-empty string)`);
  return v;
}

/** Compile a glob (`**` across dirs, `*` within a segment, `?` one char) to a whole-path-anchored RegExp. */
function globToRegExp(glob: string): RegExp {
  let re = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i]!;
    if (c === "*") {
      if (glob[i + 1] === "*") {
        re += ".*"; // ** matches across directory separators
        i++;
        if (glob[i + 1] === "/") i++; // consume the slash after ** so `**/x` also matches a bare `x`
      } else {
        re += "[^/]*";
      }
    } else if (c === "?") {
      re += "[^/]";
    } else {
      re += c.replace(/[.+^${}()|[\]\\]/g, "\\$&"); // escape regex metachars; `/` passes through literally
    }
  }
  return new RegExp(`^${re}$`);
}

/** grep — search workspace file contents for a regex. Read-only. */
export const grepTool: Tool = {
  description:
    "Search workspace file contents for a regular expression. Input: { pattern, path?, flags? }. Returns up to 200 { file, line, text } matches.",
  inputSchema: {
    type: "object",
    properties: { pattern: { type: "string" }, path: { type: "string" }, flags: { type: "string" } },
    required: ["pattern"],
  },
  capabilities: { readOnly: true },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const rel = typeof input.path === "string" && input.path.length > 0 ? input.path : ".";
    const base = resolveInWorkspace(ws.root, rel);
    const flags = typeof input.flags === "string" ? input.flags : "";
    let re: RegExp;
    try {
      re = new RegExp(reqString(input, "pattern"), flags);
    } catch (e) {
      throw new Error(`invalid grep pattern/flags: ${(e as Error).message}`);
    }
    const matches: Array<{ file: string; line: number; text: string }> = [];
    for (const abs of await walkFiles(base)) {
      if (matches.length >= MAX_MATCHES) break;
      let content: string;
      try {
        content = await fs.readFile(abs, "utf8");
      } catch {
        continue;
      }
      if (content.indexOf(String.fromCharCode(0)) !== -1) continue; // skip binary (contains a NUL byte)
      const lines = content.split(/\r?\n/);
      for (let i = 0; i < lines.length; i++) {
        if (re.test(lines[i]!)) {
          matches.push({ file: toPosixRel(ws.root, abs), line: i + 1, text: lines[i]! });
          if (matches.length >= MAX_MATCHES) break;
        }
      }
    }
    return { matches, truncated: matches.length >= MAX_MATCHES };
  },
};

/** glob — list workspace files whose relative path matches a glob pattern. Read-only. */
export const globTool: Tool = {
  description: "List workspace files matching a glob pattern (** across dirs, * within a segment, ?). Input: { pattern }.",
  inputSchema: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] },
  capabilities: { readOnly: true },
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const re = globToRegExp(reqString(input, "pattern"));
    const files = (await walkFiles(ws.root)).map((abs) => toPosixRel(ws.root, abs)).filter((rel) => re.test(rel));
    return { files, truncated: files.length >= MAX_FILES };
  },
};

/** The search tool set, keyed by logical name. */
export const searchTools: Record<string, Tool> = {
  grep: grepTool,
  glob: globTool,
};
