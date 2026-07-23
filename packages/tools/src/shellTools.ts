/**
 * The shell tool (DESIGN §5.1, "Functions and tools"): `run_command` executes a command via the host shell in
 * the workspace and captures stdout / stderr / exit code. It is the most sensitive tool — `readOnly: false`,
 * so the `read-only`/`plan` profiles block it and `ask`/`deny` modes gate it. A non-zero exit is NOT a tool
 * failure: the model sees the code and output and decides what to do. Cross-platform via Node's `exec`
 * (cmd.exe on Windows, /bin/sh elsewhere), with a timeout and output caps so a hang or a firehose can't
 * stall or flood the loop. This is best-effort containment: the permission profile is the real gate — a
 * shell can always reach outside a `cwd`, so `run_command` should sit behind `ask`/`deny` in untrusted runs.
 */
import { exec } from "node:child_process";
import { promisify } from "node:util";
import type { Tool } from "@declarative-ai/exec";
import { requireWorkspace, resolveInWorkspace } from "./workspace";

const pexec = promisify(exec);

const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_OUTPUT = 30_000;
const MAX_BUFFER = 4 * 1024 * 1024;

function reqString(input: Record<string, unknown>, key: string): string {
  const v = input[key];
  if (typeof v !== "string" || v.length === 0) throw new Error(`tool input '${key}' is required (non-empty string)`);
  return v;
}

/** Truncate long output with a marker so a firehose can't flood the model's context. */
function cap(s: string): string {
  return s.length > MAX_OUTPUT ? `${s.slice(0, MAX_OUTPUT)}\n… [truncated ${s.length - MAX_OUTPUT} chars]` : s;
}

/** run_command — execute a shell command in the workspace, returning { exit_code, stdout, stderr }. Mutating. */
export const runCommandTool: Tool = {
  description:
    "Run a shell command in the workspace and capture stdout, stderr, and the exit code. Input: { command, cwd?, timeout_ms? }. A non-zero exit is returned, not thrown.",
  inputSchema: {
    type: "object",
    properties: { command: { type: "string" }, cwd: { type: "string" }, timeout_ms: { type: "number" } },
    required: ["command"],
  },
  readOnly: false,
  async run(input, ctx) {
    const ws = requireWorkspace(ctx);
    const command = reqString(input, "command");
    const cwd = typeof input.cwd === "string" && input.cwd.length > 0 ? resolveInWorkspace(ws.root, input.cwd) : ws.root;
    const timeout = typeof input.timeout_ms === "number" && input.timeout_ms > 0 ? input.timeout_ms : DEFAULT_TIMEOUT_MS;
    try {
      const { stdout, stderr } = await pexec(command, { cwd, timeout, maxBuffer: MAX_BUFFER, windowsHide: true });
      return { exit_code: 0, stdout: cap(stdout), stderr: cap(stderr), killed: false };
    } catch (e) {
      const err = e as { code?: number; killed?: boolean; stdout?: string; stderr?: string; message?: string };
      return {
        exit_code: typeof err.code === "number" ? err.code : 1,
        stdout: cap(err.stdout ?? ""),
        stderr: cap(err.stderr ?? err.message ?? ""),
        killed: err.killed === true, // true when the timeout fired
      };
    }
  },
};

/** The shell tool set, keyed by logical name. */
export const shellTools: Record<string, Tool> = {
  run_command: runCommandTool,
};
