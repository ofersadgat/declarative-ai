/**
 * Making a command spawnable on Windows, from the SPAWN seam's side.
 *
 * The sibling of `agents-api`'s `resolveAgentBinary`, and deliberately a different answer to a
 * different question: that one produces ONE path, because the Agent SDK spawns it itself and can be
 * handed only a string. This one produces a file plus an argument prefix, because a subprocess can be
 * launched under an interpreter — which is the better answer when it is available, since it follows an
 * npm shim to the entry the shim itself names rather than to a conventional location.
 *
 * Every case runs against a FAKE filesystem, which is the only way to test win32 resolution from a
 * machine that is not Windows.
 */
import { describe, expect, it } from "vitest";
import { resolveProgram, type ProgramDeps } from "../src/process.js";

const NPM = "C:\\Users\\me\\AppData\\Roaming\\npm";
/** The stable, documented npm shim format: its last line names the JS entry it runs. */
const SHIM = ['@ECHO off', 'GOTO start', ':find_dp0', 'SET dp0=%~dp0', ':start', '"%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js"   %*'].join("\r\n");

/** A fake filesystem: the listed paths exist, and `.cmd`/`.bat` entries read back as npm shims. */
const fs = (paths: Record<string, string | true>): ProgramDeps => ({
  platform: "win32",
  pathDirs: [NPM, "C:\\ProgramData\\chocolatey\\bin"],
  node: "C:\\Program Files\\nodejs\\node.exe",
  exists: (p) => p in paths,
  readText: (p) => (typeof paths[p] === "string" ? (paths[p] as string) : undefined),
});

describe("resolveProgram", () => {
  it("leaves a command alone off Windows — POSIX spawn resolves the PATH itself", () => {
    expect(resolveProgram("codex", { ...fs({}), platform: "linux" })).toEqual({ file: "codex", prefix: [] });
  });

  it("passes a command backed by a real .exe through UNCHANGED, because Windows appends PATHEXT", () => {
    expect(resolveProgram("claude", fs({ "C:\\ProgramData\\chocolatey\\bin\\claude.exe": true }))).toEqual({ file: "claude", prefix: [] });
  });

  it("passes a command already spelled with a binary extension through unchanged", () => {
    expect(resolveProgram("D:\\builds\\claude.exe", fs({}))).toEqual({ file: "D:\\builds\\claude.exe", prefix: [] });
  });

  it("FOLLOWS an npm shim to the JS entry it names, and runs it under the interpreter", () => {
    // Node refuses to spawn a `.cmd` with `shell: false` (EINVAL, since CVE-2024-27980), and a shell is
    // not an option — these argv carry a bearer token and a TOML inline table, neither of which
    // survives cmd.exe quoting. Reading the shim is a lookup, not a guess.
    expect(resolveProgram("codex", fs({ [`${NPM}\\codex.cmd`]: SHIM }))).toEqual({
      file: "C:\\Program Files\\nodejs\\node.exe",
      prefix: [`${NPM}\\node_modules\\@openai\\codex\\bin\\codex.js`],
    });
  });

  it("runs a JS entry under the interpreter — Windows cannot execute one directly", () => {
    // What `resolveAgentBinary` falls back to for an older `@anthropic-ai/claude-code`, and what a
    // caller may pin outright. Handing it to `spawn` as a program fails with EACCES/ENOEXEC.
    const entry = `${NPM}\\node_modules\\@anthropic-ai\\claude-code\\cli.js`;
    expect(resolveProgram(entry, fs({ [entry]: true }))).toEqual({ file: "C:\\Program Files\\nodejs\\node.exe", prefix: [entry] });
    expect(resolveProgram("tool.mjs", fs({})).prefix).toEqual(["tool.mjs"]);
    expect(resolveProgram("tool.cjs", fs({})).prefix).toEqual(["tool.cjs"]);
  });

  it("does not mistake a name merely CONTAINING js for a JS entry", () => {
    expect(resolveProgram("nodejs", fs({ [`${NPM}\\nodejs.exe`]: true }))).toEqual({ file: "nodejs", prefix: [] });
  });

  it("prefers a real executable over a shim when both are on the PATH", () => {
    const resolved = resolveProgram("claude", fs({ [`${NPM}\\claude.cmd`]: SHIM, "C:\\ProgramData\\chocolatey\\bin\\claude.exe": true }));
    expect(resolved).toEqual({ file: "claude", prefix: [] });
  });

  it("does NOT treat the extensionless Git-Bash script npm installs beside a shim as a hit", () => {
    // Windows cannot execute one, and treating it as found is exactly how `codex` resolved to something
    // unspawnable. Nothing usable here ⇒ hand the name back so the spawn failure at least names it.
    expect(resolveProgram("codex", fs({ [`${NPM}\\codex`]: true }))).toEqual({ file: "codex", prefix: [] });
  });

  it("hands the name back when nothing is found, so the failure names what was asked for", () => {
    expect(resolveProgram("codex", fs({}))).toEqual({ file: "codex", prefix: [] });
  });

  it("searches the directory a command NAMES, rather than the PATH", () => {
    expect(resolveProgram("D:\\tools\\codex", fs({ "D:\\tools\\codex.cmd": SHIM }))).toEqual({
      file: "C:\\Program Files\\nodejs\\node.exe",
      prefix: ["D:\\tools\\node_modules\\@openai\\codex\\bin\\codex.js"],
    });
  });
});
