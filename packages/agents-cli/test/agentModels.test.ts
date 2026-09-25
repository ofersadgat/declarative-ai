import { describe, expect, it } from "vitest";
import { acceptanceOf, ModelInfo } from "@declarative-ai/llm";
import { claudeModelRows, codexModelRows, probeClaudeModels, probeCodexModels } from "../src/agentModels.js";
import type { AgentProcess, SpawnProcess } from "../src/process.js";

/**
 * A fake binary that answers what is written to it: `respond` maps each request line to the lines the
 * binary writes back. Records the argv and every request.
 */
function conversing(respond: (request: Record<string, unknown>) => unknown[]): { spawn: SpawnProcess; argv: string[][]; requests: Record<string, unknown>[] } {
  const argv: string[][] = [];
  const requests: Record<string, unknown>[] = [];
  const spawn: SpawnProcess = (a) => {
    argv.push(a);
    const queue: string[] = [];
    let wake: (() => void) | undefined;
    let ended = false;
    const proc: AgentProcess = {
      lines: (async function* () {
        for (;;) {
          while (queue.length > 0) yield queue.shift()!;
          if (ended) return;
          await new Promise<void>((resolve) => (wake = resolve));
        }
      })(),
      kill: () => {
        ended = true;
        wake?.();
      },
      exit: Promise.resolve(0),
      write: (text: string) => {
        const request = JSON.parse(text) as Record<string, unknown>;
        requests.push(request);
        for (const reply of respond(request)) queue.push(JSON.stringify(reply));
        wake?.();
      },
      endInput: () => undefined,
    };
    return proc;
  };
  return { spawn, argv, requests };
}

// MEASURED 2026-09-24: the SDK-bundled claude 2.1.223's initialize `models` (trimmed).
const BUNDLED = [
  { value: "default", resolvedModel: "claude-opus-5[1m]", displayName: "Default", supportedEffortLevels: ["low", "medium", "high", "xhigh", "max"] },
  { value: "opus[1m]", resolvedModel: "claude-opus-5[1m]", displayName: "Opus (1M)", supportedEffortLevels: ["low", "medium", "high", "xhigh", "max"] },
  { value: "haiku", resolvedModel: "claude-haiku-4-5-20251001", displayName: "Haiku", supportedEffortLevels: null },
];

// MEASURED 2026-09-24: codex-cli 0.147.0's `model/list` (trimmed).
const CODEX = [
  {
    id: "gpt-5.6-sol",
    displayName: "GPT-5.6-Sol",
    hidden: false,
    isDefault: true,
    supportedReasoningEfforts: [
      { reasoningEffort: "low", description: "Fast responses with lighter reasoning" },
      { reasoningEffort: "high", description: "Greater reasoning depth for complex problems" },
      { reasoningEffort: "ultra", description: "Maximum reasoning with automatic task delegation" },
    ],
    defaultReasoningEffort: "low",
    inputModalities: ["text", "image"],
  },
  { id: "gpt-5.5", displayName: "GPT-5.5", hidden: true, isDefault: false, supportedReasoningEfforts: [{ reasoningEffort: "medium" }] },
];

describe("claudeModelRows — what claude says it runs", () => {
  it("files each entry under the id it resolves to AND the alias it is picked by, once each", () => {
    const rows = claudeModelRows(BUNDLED, "claude-code");
    expect(rows.map((r) => `${r.route}/${r.model}`)).toEqual([
      "claude-code/claude-opus-5[1m]",
      "claude-code/default",
      "claude-code/opus[1m]",
      "claude-code/claude-haiku-4-5-20251001",
      "claude-code/haiku",
    ]);
    // The canonical id is the resolved model's, so the API row prices it.
    expect(rows[1]?.canonicalId).toBe("claude-opus-5");
    expect(rows[4]?.canonicalId).toBe("claude-haiku-4-5");
  });

  it("offers the model's own levels and no budget; a model with none takes no reasoning", () => {
    const [opus, , , haiku] = claudeModelRows(BUNDLED);
    expect(acceptanceOf(opus?.parameters).efforts).toEqual(["low", "medium", "high", "xhigh", "max"]);
    expect(acceptanceOf(opus?.parameters).acceptsBudget).toBe(false);
    expect(acceptanceOf(haiku?.parameters).acceptsReasoning).toBe(false);
  });

  it("reads an older claude's bare aliases, which name only themselves", () => {
    // MEASURED: installed 2.1.142 answers no `resolvedModel`, and its sonnet lacks xhigh.
    const rows = claudeModelRows([{ value: "sonnet", supportedEffortLevels: ["low", "medium", "high", "max"] }]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ route: "claude-cli", model: "sonnet", canonicalId: "sonnet" });
    expect(acceptanceOf(rows[0]?.parameters).efforts).toEqual(["low", "medium", "high", "max"]);
  });
});

describe("probeClaudeModels — the initialize exchange, no prompt", () => {
  it("sends only initialize and reads the models off its answer", async () => {
    const fake = conversing((req) =>
      req["type"] === "control_request"
        ? [{ type: "control_response", response: { subtype: "success", request_id: (req as { request_id: string }).request_id, response: { models: BUNDLED } } }]
        : [],
    );
    const rows = await probeClaudeModels({ spawn: fake.spawn, route: "claude-code", command: "claude.exe" });
    expect(fake.argv[0]).toEqual(["claude.exe", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]);
    expect(fake.requests.map((r) => (r["request"] as { subtype: string }).subtype)).toEqual(["initialize"]);
    expect(rows.map((r) => r.model)).toContain("claude-opus-5[1m]");
  });

  it("fails naming claude's refusal, for the refresh to skip the source", async () => {
    const fake = conversing((req) => [{ type: "control_response", response: { subtype: "error", request_id: (req as { request_id: string }).request_id, error: "not logged in" } }]);
    await expect(probeClaudeModels({ spawn: fake.spawn })).rejects.toThrow("not logged in");
  });
});

describe("codexModelRows — what codex says it runs", () => {
  it("offers exactly codex's levels, with its default, and no budget", () => {
    const [sol] = codexModelRows(CODEX);
    const gate = acceptanceOf(sol?.parameters);
    expect(gate.efforts).toEqual(["low", "high", "ultra"]);
    expect(gate.acceptsBudget).toBe(false);
    expect(sol?.parameters?.["properties"]).toMatchObject({ reasoning: { properties: { effort: { default: "low" } } } });
    expect(sol).toMatchObject({ route: "codex-cli", model: "gpt-5.6-sol", canonicalId: "gpt-5-6-sol", modalities: { input: ["text", "image"], output: ["text"] } });
  });

  it("files the default model under `default` too, and marks a hidden one unavailable", () => {
    const rows = codexModelRows(CODEX);
    expect(rows.map((r) => r.model)).toEqual(["gpt-5.6-sol", "default", "gpt-5.5"]);
    expect(rows[2]?.available).toBe(false);
  });
});

describe("probeCodexModels — app-server JSON-RPC, page by page", () => {
  it("initializes, then lists every page, then stops", async () => {
    const fake = conversing((req) => {
      if (req["id"] === 1) return [{ id: 1, result: {} }];
      if (req["method"] !== "model/list") return [];
      const first = (req["params"] as { cursor?: string }).cursor === undefined;
      return [{ id: req["id"], result: first ? { data: [CODEX[0]], nextCursor: "p2" } : { data: [CODEX[1]], nextCursor: null } }];
    });
    const rows = await probeCodexModels({ spawn: fake.spawn });
    expect(fake.argv[0]).toEqual(["codex", "app-server"]);
    expect(fake.requests.map((r) => r["method"])).toEqual(["initialize", "initialized", "model/list", "model/list"]);
    expect(fake.requests[3]?.["params"]).toEqual({ cursor: "p2" });
    expect(rows.map((r) => r.model)).toEqual(["gpt-5.6-sol", "default", "gpt-5.5"]);
  });

  it("fails naming a refused listing", async () => {
    const fake = conversing((req) => (req["id"] === 1 ? [{ id: 1, result: {} }] : req["method"] === "model/list" ? [{ id: req["id"], error: { message: "not signed in" } }] : []));
    await expect(probeCodexModels({ spawn: fake.spawn })).rejects.toThrow("codex model/list: not signed in");
  });
});

describe("an agent's rows fit its reasoning and take the API's price", () => {
  it("clamps a level to the agent's own, and prices the agent row by the model's canonical id", () => {
    const catalog = new ModelInfo([...codexModelRows(CODEX), { route: "openai", model: "gpt-5.6-sol", inputPerMillion: 5, outputPerMillion: 40 }]);
    expect(catalog.paramAcceptance("codex-cli/gpt-5.6-sol").efforts).toEqual(["low", "high", "ultra"]);
    expect(catalog.computeCostUsd("codex-cli/default", 1_000_000, 0)).toBeCloseTo(5, 10);
  });
});

describe("codex's level descriptions", () => {
  it("travel with the levels, for a picker to show beside each one", () => {
    const [sol] = codexModelRows(CODEX);
    expect(acceptanceOf(sol?.parameters).effortDescriptions).toEqual({
      low: "Fast responses with lighter reasoning",
      high: "Greater reasoning depth for complex problems",
      ultra: "Maximum reasoning with automatic task delegation",
    });
  });
});
