import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { ExecServices } from "@declarative-ai/exec";
import { allTools, requireWorkspace, runCommandTool } from "../src";

// Drive the shell through tiny node scripts written into the workspace — deterministic and cross-platform
// (no shell-specific echo/exit syntax, no nested-quote fragility), needing only `node` on PATH.
let root: string;
beforeAll(async () => {
  root = await mkdtemp(path.join(os.tmpdir(), "dai-shell-"));
  await writeFile(path.join(root, "print.js"), "process.stdout.write('hi')");
  await writeFile(path.join(root, "exit3.js"), "process.exit(3)");
  await writeFile(path.join(root, "err.js"), "process.stderr.write('boom')");
  await writeFile(path.join(root, "sleep.js"), "setTimeout(() => {}, 5000)");
});
afterAll(async () => {
  // The timeout test kills a child mid-run; on Windows the temp dir can stay briefly locked (EBUSY), so
  // retry and then swallow — a leftover OS-temp dir must not fail the suite.
  try {
    await rm(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  } catch {
    /* best-effort cleanup */
  }
});
const ctx = (): ExecServices => ({ workspace: { root } });

describe("run_command", () => {
  it("runs a command in the workspace and captures stdout with exit 0", async () => {
    const out = (await runCommandTool.run({ command: "node print.js" }, ctx())) as Record<string, unknown>;
    expect(out.exit_code).toBe(0);
    expect(String(out.stdout)).toContain("hi");
    expect(out.killed).toBe(false);
  });

  it("returns a non-zero exit code instead of throwing", async () => {
    const out = (await runCommandTool.run({ command: "node exit3.js" }, ctx())) as Record<string, unknown>;
    expect(out.exit_code).toBe(3);
  });

  it("captures stderr (a stderr write alone still exits 0)", async () => {
    const out = (await runCommandTool.run({ command: "node err.js" }, ctx())) as Record<string, unknown>;
    expect(String(out.stderr)).toContain("boom");
  });

  it("kills a command that exceeds the timeout", async () => {
    const out = (await runCommandTool.run({ command: "node sleep.js", timeout_ms: 300 }, ctx())) as Record<string, unknown>;
    expect(out.killed).toBe(true);
    expect(out.exit_code).not.toBe(0);
  });

  it("is mutating (so read-only/plan profiles block it), is in allTools, and needs a workspace", () => {
    expect(runCommandTool.readOnly).toBe(false);
    expect(allTools["run_command"]).toBe(runCommandTool);
    expect(() => requireWorkspace({})).toThrow(/needs a workspace/);
  });
});
