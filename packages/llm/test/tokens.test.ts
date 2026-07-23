import { describe, expect, it } from "vitest";
import { estimateCallTokens } from "../src/tokens";

describe("estimateCallTokens", () => {
  it("estimates input from chars/4 across prompt + system, and output from the ceiling", () => {
    expect(estimateCallTokens("abcd".repeat(10), "efgh", 256)).toEqual({ inputTokens: 11, outputTokens: 256 });
  });

  it("falls back to a default output ceiling when the call declares none", () => {
    expect(estimateCallTokens("x", undefined, undefined)).toEqual({ inputTokens: 1, outputTokens: 512 });
  });
});
