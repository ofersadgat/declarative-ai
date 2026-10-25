/**
 * `blockOfPart` — a transport's message part, read into the record's block. The case here is the AI
 * SDK's: a tool result says it FAILED by its output's tag, not by a flag beside it.
 */
import { describe, expect, it } from "vitest";
import { blockOfPart } from "../src/entry.js";

describe("blockOfPart — a tool result", () => {
  it("keeps an AI SDK result's failure: a tool that threw arrives as `error-text`", () => {
    const part = { type: "tool-result", toolCallId: "c1", toolName: "edit", output: { type: "error-text", value: "Not run: the person denied this call." } };
    expect(blockOfPart(part)).toEqual({ type: "tool_result", toolUseId: "c1", isError: true, text: "Not run: the person denied this call." });
  });

  it("reads an ordinary result as one that worked", () => {
    expect(blockOfPart({ type: "tool-result", toolCallId: "c1", output: { type: "text", value: "ok" } })).toEqual({ type: "tool_result", toolUseId: "c1", text: "ok" });
  });
});
