/**
 * Making a NAMED binary launchable on Windows — a live defect, not a gap.
 *
 * Neither transport spawns through a shell, so `spawn("claude")` does no PATH/PATHEXT lookup and fails
 * as ENOENT with `claude.exe` sitting right there, while an npm `claude.cmd` shim fails with EINVAL
 * (Node ≥ 20.12) whether or not it is found. Both symptoms read as "the agent is not installed".
 *
 * Every case runs against a FAKE filesystem, which is the only way to test this from a machine that is
 * not Windows — and the reason `exists` is injected rather than reached for.
 */
import { describe, expect, it } from "vitest";
import { resolveAgentBinary, type BinaryDeps } from "../src/binary.js";

const NPM = "C:\\Users\\me\\AppData\\Roaming\\npm";
const PACKAGE_EXE = `${NPM}\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe`;
const PACKAGE_JS = `${NPM}\\node_modules\\@anthropic-ai\\claude-code\\cli.js`;

/** A fake filesystem: the listed paths exist and nothing else does. */
const fs = (...paths: string[]): BinaryDeps => ({
  platform: "win32",
  pathDirs: ["C:\\ProgramData\\chocolatey\\bin", NPM],
  exists: (path) => paths.includes(path),
});

describe("resolveAgentBinary", () => {
  it("leaves a command alone off Windows — POSIX spawn resolves the PATH itself", () => {
    // Doing the resolution anyway would only be a way to get it wrong: there are no launcher scripts to
    // follow and no PATHEXT to append.
    expect(resolveAgentBinary("claude", { platform: "linux", exists: () => false })).toEqual({ path: "claude" });
  });

  it("finds a real executable on the PATH, which is the case that costs nothing", () => {
    const resolved = resolveAgentBinary("claude", fs("C:\\ProgramData\\chocolatey\\bin\\claude.exe"));
    expect(resolved).toEqual({ path: "C:\\ProgramData\\chocolatey\\bin\\claude.exe" });
  });

  it("FOLLOWS an npm launcher shim to the package entry rather than handing it back", () => {
    // The whole point. Returning `claude.cmd` would trade an ENOENT for an EINVAL — strictly less
    // informative, and just as broken.
    const resolved = resolveAgentBinary("claude", fs(`${NPM}\\claude.cmd`, PACKAGE_EXE));
    expect(resolved).toEqual({ path: PACKAGE_EXE });
  });

  it("falls back to cli.js for an older package layout, which the SDK still knows how to run", () => {
    const resolved = resolveAgentBinary("claude", fs(`${NPM}\\claude.cmd`, PACKAGE_JS));
    expect(resolved).toEqual({ path: PACKAGE_JS });
  });

  it("prefers a real executable over a launcher when BOTH are on the PATH", () => {
    const resolved = resolveAgentBinary("claude", fs(`${NPM}\\claude.cmd`, PACKAGE_EXE, "C:\\ProgramData\\chocolatey\\bin\\claude.exe"));
    expect(resolved.path).toBe("C:\\ProgramData\\chocolatey\\bin\\claude.exe");
  });

  it("follows a .bat and a .ps1 shim too, not only .cmd", () => {
    expect(resolveAgentBinary("claude", fs(`${NPM}\\claude.bat`, PACKAGE_EXE)).path).toBe(PACKAGE_EXE);
    expect(resolveAgentBinary("claude", fs(`${NPM}\\claude.ps1`, PACKAGE_EXE)).path).toBe(PACKAGE_EXE);
  });

  it("resolves a command that NAMES a directory against that directory, not against the PATH", () => {
    expect(resolveAgentBinary(`${NPM}\\claude`, fs(`${NPM}\\claude.cmd`, PACKAGE_EXE)).path).toBe(PACKAGE_EXE);
  });

  it("takes an already-suffixed command at its word rather than appending a second extension", () => {
    expect(resolveAgentBinary(`${NPM}\\claude.cmd`, fs(`${NPM}\\claude.cmd`, PACKAGE_EXE)).path).toBe(PACKAGE_EXE);
    expect(resolveAgentBinary("D:\\builds\\claude.exe", fs("D:\\builds\\claude.exe")).path).toBe("D:\\builds\\claude.exe");
  });

  it("WARNS and passes the original through when a shim leads nowhere", () => {
    // A path we invented failing is undiagnosable; the caller's own string failing at least names what
    // was asked for. The warning is what turns an unexplained EINVAL into a sentence.
    const resolved = resolveAgentBinary("claude", fs(`${NPM}\\claude.cmd`));
    expect(resolved.path).toBe("claude");
    expect(resolved.warning).toContain("EINVAL");
  });

  it("WARNS and passes the original through when nothing is on the PATH at all", () => {
    const resolved = resolveAgentBinary("claude", fs());
    expect(resolved.path).toBe("claude");
    expect(resolved.warning).toContain("not found on the PATH");
  });
});
