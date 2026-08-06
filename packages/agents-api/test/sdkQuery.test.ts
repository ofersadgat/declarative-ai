/**
 * The SDK adapter's BOUNDARY MAPPING — the one part of this package no end-to-end test can reach.
 *
 * `@anthropic-ai/claude-agent-sdk` is an optional peer dependency, so `sdkAgentQuery` cannot be driven
 * here at all: the module is not installed and the whole point of the lazy specifier is that it need not
 * be. That gap is not academic. It is exactly where this adapter silently dropped `resume` and
 * `forkSession` while its executor declared `sessionResume: true` and `sessionFork: true` — a
 * declaration the session layer answers by SKIPPING replay, so every call started a cold conversation
 * and reported it as a successful resume. It also never read `session_id` back, so even a correct
 * resume would have had no handle to resume WITH on the next call.
 *
 * So the two mapping spots are pure functions, and these are their tests.
 */
import { describe, expect, it } from "vitest";
import { readSdkResult, sdkOptions } from "../src/sdkQuery.js";

describe("sdkOptions — the request the SDK is handed", () => {
  it("passes the caller's posture through under the SDK's own names", () => {
    expect(sdkOptions({ prompt: "x", cwd: "/repo", model: "sonnet", permissionMode: "plan", allowedTools: ["Read"], disallowedTools: ["Bash"] })).toEqual({
      cwd: "/repo",
      model: "sonnet",
      permissionMode: "plan",
      allowedTools: ["Read"],
      disallowedTools: ["Bash"],
    });
  });

  it("omits what the caller did not ask for, so the SDK keeps its own defaults", () => {
    expect(sdkOptions({ prompt: "x" })).toEqual({});
  });

  it("carries the session handle, so a resumed conversation continues instead of restarting", () => {
    // Without this the session layer's whole cheap path is a lie: it reads ZERO messages on the
    // strength of `sessionResume: true`, and the agent is asked to continue a conversation it was
    // never told about.
    expect(sdkOptions({ prompt: "go on", resume: "sess-abc" })).toEqual({ resume: "sess-abc" });
  });

  it("asks for a BRANCH when the session forked, which is what `sessionFork: true` promises", () => {
    expect(sdkOptions({ prompt: "go on", resume: "sess-abc", forkSession: true })).toEqual({ resume: "sess-abc", forkSession: true });
  });

  it("starts fresh when there is no handle, and never asks to fork nothing", () => {
    expect(sdkOptions({ prompt: "hi" })).not.toHaveProperty("resume");
    // A fork BRANCHES a conversation, so it says nothing without one to branch — and the executor
    // only ever sets the two together.
    expect(sdkOptions({ prompt: "hi", forkSession: true })).not.toHaveProperty("forkSession");
  });
});

describe("readSdkResult — the answer read back", () => {
  it("normalizes the terminal result, cost and all", () => {
    expect(readSdkResult({ type: "result", result: "done", total_cost_usd: 0.02 })).toEqual({ type: "result", result: { text: "done", costUsd: 0.02 } });
  });

  it("records the session the run ENDED in — a new id after a fork, the resumed one otherwise", () => {
    // Recording it is not optional: a fork that kept its parent's handle would put two branches into
    // one remote session, and a run whose id is dropped leaves the next call nothing to resume.
    expect(readSdkResult({ type: "result", result: "ZEPHYR", session_id: "51caeb77" })).toEqual({
      type: "result",
      result: { text: "ZEPHYR", costUsd: undefined, sessionId: "51caeb77" },
    });
  });

  it("treats a run the SDK reported as FAILED as an error, not as the agent's answer", () => {
    // Observed on the binary this SDK drives: `is_error` is independent of `subtype` AND of the exit
    // code, so reading only the discriminator reported "Not logged in · Please run /login" as a
    // successful answer — indistinguishable from a review that found nothing.
    expect(readSdkResult({ type: "result", subtype: "success", is_error: true, result: "Not logged in · Please run /login" })).toEqual({
      type: "other",
      error: "Not logged in · Please run /login",
    });
  });

  it("names the failure even when the SDK supplied no text to explain it", () => {
    expect(readSdkResult({ type: "result", is_error: true })).toEqual({ type: "other", error: expect.stringMatching(/reported a failed run/) });
  });

  it("passes everything else through as unremarkable stream traffic", () => {
    expect(readSdkResult({ type: "assistant" })).toEqual({ type: "other" });
  });
});
