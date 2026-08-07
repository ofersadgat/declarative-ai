/**
 * The transport-independent half of tool injection.
 *
 * These used to be `agents-cli`'s, reached only through `handleToolCall`, while the SDK path had no
 * equivalent at all — it handed JSON Schema to a factory that only takes Zod and threw before the agent
 * started. One implementation, tested once, is the point.
 */
import { describe, expect, it } from "vitest";
import type { SyncOutputValidator } from "@declarative-ai/exec";
import { injectedToolDescriptors, mcpToolName, runInjectedTool, textResult } from "../src/mcpTools.js";

const schema = { type: "object", properties: { path: { type: "string" } }, required: ["path"] } as never;

describe("injectedToolDescriptors — what the agent is told exists", () => {
  it("advertises the JSON Schema VERBATIM, because nothing downstream re-serializes it", () => {
    expect(injectedToolDescriptors({ read_file: { description: "read a file", inputSchema: schema, run: () => "" } })).toEqual([
      { name: "read_file", description: "read a file", inputSchema: schema },
    ]);
  });

  it("omits a description rather than inventing one", () => {
    expect(injectedToolDescriptors({ t: { inputSchema: schema, run: () => "" } })[0]).not.toHaveProperty("description");
  });

  it("says nothing when there is nothing injected", () => {
    expect(injectedToolDescriptors(undefined)).toEqual([]);
  });
});

describe("mcpToolName — the name an agent actually addresses", () => {
  it("qualifies a logical name with the server it is served from", () => {
    // The logical name alone names nothing the agent can call, which is why a logical-name `deny` entry
    // never matches an injected tool.
    expect(mcpToolName("read_file")).toBe("mcp__dai__read_file");
  });
});

describe("runInjectedTool — one call, whatever carried it", () => {
  it("runs the host impl and returns its value as text", async () => {
    const seen: unknown[] = [];
    const result = await runInjectedTool(
      { tools: { read_file: { inputSchema: schema, run: (input) => (seen.push(input), "ZEPHYR") } } },
      "read_file",
      { path: "a.txt" },
    );
    expect(seen).toEqual([{ path: "a.txt" }]);
    expect(result).toEqual(textResult("ZEPHYR"));
  });

  it("serializes a non-string value rather than stringifying it by coercion", async () => {
    const result = await runInjectedTool({ tools: { t: { inputSchema: schema, run: () => ({ ok: true }) } } }, "t", {});
    expect(result.content[0]!.text).toBe('{"ok":true}');
  });

  it("reports an unknown tool as a tool ERROR the agent reads, not as a transport fault", async () => {
    expect(await runInjectedTool({ tools: {} }, "nope", {})).toEqual(textResult("no tool 'nope' is available", true));
  });

  it("turns a THROWING impl into an error result — a tool failure is something the agent reacts to", async () => {
    const result = await runInjectedTool(
      {
        tools: {
          t: {
            inputSchema: schema,
            run: () => {
              throw new Error("disk on fire");
            },
          },
        },
      },
      "t",
      {},
    );
    expect(result).toEqual(textResult("tool 't' failed: disk on fire", true));
  });

  it("checks the arguments against the tool's OWN schema before an impl sees them", async () => {
    // No MCP server validates a tool call: the low-level `Server` advertises our schema and hands the
    // handler whatever arrived. So this is the only thing between an arbitrary payload and a host impl.
    let ran = false;
    const validator: SyncOutputValidator = { validateValue: () => ({ ok: false, errors: "path is required" }) };
    const result = await runInjectedTool({ tools: { t: { inputSchema: schema, run: () => (ran = true) as never } }, validator }, "t", {});
    expect(ran).toBe(false);
    expect(result).toEqual(textResult("tool 't' input is invalid: path is required", true));
  });

  it("treats a validator that THROWS as a refusal, never as a pass", async () => {
    let ran = false;
    const validator: SyncOutputValidator = {
      validateValue: () => {
        throw new Error("schema unreadable");
      },
    };
    const result = await runInjectedTool({ tools: { t: { inputSchema: schema, run: () => (ran = true) as never } }, validator }, "t", {});
    expect(ran).toBe(false);
    expect(result.isError).toBe(true);
    expect(result.content[0]!.text).toContain("schema unreadable");
  });

  it("enforces nothing when no validator was injected — the caller that owns the tools decides", async () => {
    const result = await runInjectedTool({ tools: { t: { inputSchema: schema, run: () => "ran" } } }, "t", { wrong: 1 });
    expect(result).toEqual(textResult("ran"));
  });
});
