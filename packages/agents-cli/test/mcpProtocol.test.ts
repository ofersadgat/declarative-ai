/**
 * The CLI permission-prompt contract.
 *
 * This is the security-critical surface of the CLI adapter: it is what makes a delegated agent ASK
 * before each gated tool-use instead of running on its own defaults. Its shapes were read off the
 * shipping CLI's own code rather than from a published spec, so they are pinned here — if a future CLI
 * changes the contract, these tests are the tripwire, and they say exactly which field moved.
 */
import { describe, expect, it } from "vitest";
import {
  APPROVAL_INPUT_SCHEMA,
  APPROVAL_TOOL,
  PERMISSION_PROMPT_TOOL,
  approvalResponseText,
  assertNoReservedToolNames,
  bridgePath,
  handleToolCall,
  injectedToolAllowEntries,
  isAuthorizedBridgeRequest,
  mcpConfigJson,
  mcpToolName,
  newBridgeToken,
  parseApprovalRequest,
  toolDescriptors,
} from "../src/mcpProtocol";

/** Decode what the CLI would actually read: the JSON string inside `content[0].text`. */
const decode = (result: { content: Array<{ text: string }> }) => JSON.parse(result.content[0]!.text) as Record<string, unknown>;

describe("naming and config", () => {
  it("qualifies a tool the way the CLI addresses it", () => {
    expect(mcpToolName("approve")).toBe("mcp__dai__approve");
    expect(mcpToolName("read_file", "other")).toBe("mcp__other__read_file");
    expect(PERMISSION_PROMPT_TOOL).toBe(`mcp__dai__${APPROVAL_TOOL}`);
  });

  it("declares an HTTP server with an explicit `type`", () => {
    // Documented gotcha: an entry with a `url` and no `type` is read as a STDIO server and errors out.
    expect(JSON.parse(mcpConfigJson("http://127.0.0.1:1234/mcp"))).toEqual({
      mcpServers: { dai: { type: "http", url: "http://127.0.0.1:1234/mcp" } },
    });
  });

  it("allow-lists injected tools under their MCP-qualified names", () => {
    // The logical names alone would allow nothing — the agent only ever sees the qualified form.
    expect(injectedToolAllowEntries({ read_file: {} as never, bash: {} as never })).toEqual(["mcp__dai__read_file", "mcp__dai__bash"]);
    expect(injectedToolAllowEntries(undefined)).toEqual([]);
  });
});

describe("who may talk to the bridge at all", () => {
  // The bridge executes host tools. Loopback is not authorization: every other process on the machine
  // is also on loopback, and the approver is on the FAR side of this port — it is the CLI's job to ask
  // it, so an unauthenticated caller reaching `tools/call` runs the tool with nobody consulted.
  const token = newBridgeToken();

  it("mints an unguessable 256-bit secret, fresh every time", () => {
    expect(token).toMatch(/^[0-9a-f]{64}$/);
    // Not a counter, not a timestamp: two bridges must not be able to address each other.
    expect(newBridgeToken()).not.toBe(newBridgeToken());
  });

  it("admits the run's own URL", () => {
    expect(isAuthorizedBridgeRequest({ url: bridgePath(token) }, token)).toBe(true);
    // A query string is not part of the secret.
    expect(isAuthorizedBridgeRequest({ url: `${bridgePath(token)}?x=1` }, token)).toBe(true);
  });

  it.each([
    ["the bare MCP path — the whole original hole", "/mcp"],
    ["a trailing slash instead of a token", "/mcp/"],
    ["another run's token", `/mcp/${newBridgeToken()}`],
    ["a truncated token", bridgePath(token).slice(0, -1)],
    ["a token with something appended", `${bridgePath(token)}extra`],
    ["a nested path under the real token", `${bridgePath(token)}/../mcp`],
    ["no URL at all", undefined],
  ])("refuses %s", (_case, url) => {
    expect(isAuthorizedBridgeRequest({ url }, token)).toBe(false);
  });

  it("refuses a request carrying ANY Origin, even with the right token — DNS rebinding", () => {
    // A page that resolves a name to 127.0.0.1 cannot READ the reply cross-origin, but by then the tool
    // has already run. The CLI never sends an Origin; a browser always does.
    expect(isAuthorizedBridgeRequest({ url: bridgePath(token), origin: "https://evil.example" }, token)).toBe(false);
    expect(isAuthorizedBridgeRequest({ url: bridgePath(token), origin: "null" }, token)).toBe(false);
  });
});

describe("the tools we advertise", () => {
  it("exposes the approval tool only when an approver is wired", () => {
    expect(toolDescriptors({}).map((t) => t.name)).toEqual([]);
    expect(toolDescriptors({ approve: () => {} }).map((t) => t.name)).toEqual([APPROVAL_TOOL]);
  });

  it("gives the approval tool a non-empty input schema — the CLI refuses one without", () => {
    const approval = toolDescriptors({ approve: () => {} })[0]!;
    expect(approval.inputSchema).toBe(APPROVAL_INPUT_SCHEMA);
    expect(Object.keys(APPROVAL_INPUT_SCHEMA.properties)).toEqual(["tool_name", "input", "tool_use_id"]);
  });

  it("passes an injected tool's JSON Schema through VERBATIM — no conversion, no re-serialization", () => {
    const inputSchema = { type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false } as never;
    const [tool] = toolDescriptors({ tools: { read_file: { description: "Read a file", inputSchema, run: () => null } } });
    expect(tool!.inputSchema).toBe(inputSchema);
    expect(tool!.description).toBe("Read a file");
  });
});

describe("reading the CLI's permission request", () => {
  it("reads the snake_case fields the CLI actually sends", () => {
    expect(parseApprovalRequest({ tool_name: "Bash", input: { command: "ls" }, tool_use_id: "tu_1" })).toEqual({
      toolName: "Bash",
      input: { command: "ls" },
      toolUseId: "tu_1",
    });
  });

  it("treats tool_use_id as optional", () => {
    expect(parseApprovalRequest({ tool_name: "Bash", input: {} })).toEqual({ toolName: "Bash", input: {} });
  });

  it("defaults a missing or non-object input to an empty object", () => {
    expect(parseApprovalRequest({ tool_name: "Bash" })?.input).toEqual({});
    expect(parseApprovalRequest({ tool_name: "Bash", input: "nope" })?.input).toEqual({});
  });

  it.each([undefined, null, "string", 42, [], {}, { toolName: "Bash" }, { tool_name: "" }, { tool_name: 5 }])(
    "rejects a malformed request (%s) rather than guessing a tool name",
    (args) => {
      expect(parseApprovalRequest(args)).toBeUndefined();
    },
  );
});

describe("the decision payload", () => {
  // THE detail that would fail silently: the CLI's allow schema requires `updatedInput`. The Agent
  // SDK's TypeScript type marks it optional, so inferring this from the `.d.ts` produces a payload the
  // CLI rejects — and a rejected decision is not an allow.
  it("ECHOES the original input into updatedInput on allow, because the CLI requires it", () => {
    const input = { command: "ls -la", timeout: 30 };
    expect(JSON.parse(approvalResponseText({ allow: true }, input))).toEqual({ behavior: "allow", updatedInput: input });
  });

  it("still emits updatedInput for an empty input", () => {
    expect(JSON.parse(approvalResponseText({ allow: true }, {}))).toEqual({ behavior: "allow", updatedInput: {} });
  });

  it("carries a required message on deny", () => {
    expect(JSON.parse(approvalResponseText({ allow: false, reason: "writes outside the workspace" }, {}))).toEqual({
      behavior: "deny",
      message: "writes outside the workspace",
    });
  });

  it("supplies a message even when the approver gave no reason — the field is not optional", () => {
    const denied = JSON.parse(approvalResponseText({ allow: false }, {})) as { behavior: string; message: string };
    expect(denied.behavior).toBe("deny");
    expect(typeof denied.message).toBe("string");
    expect(denied.message.length).toBeGreaterThan(0);
  });
});

describe("serving a permission ask", () => {
  const approve = (decision: boolean, reason?: string) => async () => (decision ? { allow: true as const } : { allow: false as const, reason });

  it("routes the ask to the approver and returns its allow", async () => {
    let seen: unknown;
    const spec = {
      approve: async (req: unknown) => {
        seen = req;
        return { allow: true as const };
      },
    };
    const result = await handleToolCall(spec, APPROVAL_TOOL, { tool_name: "Bash", input: { command: "ls" } });
    expect(seen).toEqual({ toolName: "Bash", input: { command: "ls" } });
    expect(decode(result)).toEqual({ behavior: "allow", updatedInput: { command: "ls" } });
  });

  it("returns the approver's denial with its reason", async () => {
    const result = await handleToolCall({ approve: approve(false, "not permitted") }, APPROVAL_TOOL, { tool_name: "Bash", input: {} });
    expect(decode(result)).toMatchObject({ behavior: "deny", message: "not permitted" });
  });

  // Fail CLOSED. Every path that cannot produce a genuine approval must deny.
  it("DENIES a malformed request rather than falling through to an allow", async () => {
    const result = await handleToolCall({ approve: approve(true) }, APPROVAL_TOOL, { garbage: true });
    expect(decode(result).behavior).toBe("deny");
  });

  it("DENIES when the approver itself throws — an exception is not consent", async () => {
    const spec = {
      approve: async () => {
        throw new Error("approval UI crashed");
      },
    };
    const result = await handleToolCall(spec, APPROVAL_TOOL, { tool_name: "Bash", input: {} });
    expect(decode(result).behavior).toBe("deny");
  });

  it("does not treat `approve` as a callable tool when no approver is wired", async () => {
    const result = await handleToolCall({}, APPROVAL_TOOL, { tool_name: "Bash", input: {} });
    expect(result.isError).toBe(true);
    expect(result.content[0]!.text).toMatch(/no tool 'approve' is available/);
  });
});

describe("`approve` is RESERVED, in both directions", () => {
  const colliding = { approve: { inputSchema: {} as never, run: () => "ran the host tool" } };

  it("refuses the registration LOUDLY rather than picking a winner", () => {
    // With an approver the gate shadows the host tool (the agent asks for a tool and gets a permission
    // verdict); without one the host tool is exposed under the very name `--permission-prompt-tool`
    // addresses. Both are silent, and neither is guessable from the caller's side.
    expect(() => toolDescriptors({ tools: colliding })).toThrow(/reserved/);
    expect(() => toolDescriptors({ tools: colliding, approve: () => {} })).toThrow(/reserved/);
    expect(() => assertNoReservedToolNames(colliding)).toThrow(/mcp__dai__approve/);
    expect(() => assertNoReservedToolNames({ grep: {} as never })).not.toThrow();
  });

  it("never dispatches the reserved name to a host impl, approver or not", async () => {
    const denied = await handleToolCall({ tools: colliding, approve: async () => ({ allow: false as const, reason: "no" }) }, APPROVAL_TOOL, {
      tool_name: "Bash",
      input: {},
    });
    expect(decode(denied).behavior).toBe("deny");
    const unwired = await handleToolCall({ tools: colliding }, APPROVAL_TOOL, {});
    expect(unwired.isError).toBe(true);
    expect(unwired.content[0]!.text).not.toMatch(/ran the host tool/);
  });
});

describe("injected tool arguments are CHECKED against the tool's own schema", () => {
  // The low-level MCP `Server` validates nothing — it advertises our schema and hands the handler
  // whatever arrived. Without this, `{"path":42,"junk":{"nested":true}}` reached an impl whose schema
  // said `required:["path","contents"]`, `path: string`, `additionalProperties:false`.
  const inputSchema = {
    type: "object",
    properties: { path: { type: "string" }, contents: { type: "string" } },
    required: ["path", "contents"],
    additionalProperties: false,
  } as never;

  /** A hand-rolled `OutputValidator` — the seam is three lines, which is exactly why this package can
   *  check arguments without ever depending on ajv. */
  const validator = {
    validateValue: (schema: Record<string, unknown>, value: unknown) => {
      const record = value as Record<string, unknown>;
      const props = (schema.properties ?? {}) as Record<string, { type?: string }>;
      const missing = ((schema.required ?? []) as string[]).filter((k) => !(k in record));
      if (missing.length) return { ok: false, errors: `missing ${missing.join(",")}` };
      if (schema.additionalProperties === false) {
        const extra = Object.keys(record).filter((k) => !(k in props));
        if (extra.length) return { ok: false, errors: `unexpected ${extra.join(",")}` };
      }
      for (const [k, spec] of Object.entries(props)) {
        if (k in record && spec.type === "string" && typeof record[k] !== "string") return { ok: false, errors: `${k} is not a string` };
      }
      return { ok: true };
    },
  };

  const specFor = (ran: { input?: unknown }) => ({
    validator,
    tools: {
      write_file: {
        inputSchema,
        run: (i: Record<string, unknown>) => {
          ran.input = i;
          return "ok";
        },
      },
    },
  });

  it.each([
    ["a wrong-typed property plus an undeclared one", { path: 42, junk: { nested: true } }],
    ["a missing required property", { path: "/tmp/x" }],
    ["an empty bag", {}],
    ["a non-object argument, which becomes an empty bag", "not-an-object"],
  ])("REFUSES %s without the impl ever running", async (_case, args) => {
    const ran: { input?: unknown } = {};
    const result = await handleToolCall(specFor(ran), "write_file", args);
    expect(ran.input).toBeUndefined();
    expect(result.isError).toBe(true);
    expect(result.content[0]!.text).toMatch(/input is invalid/);
  });

  it("still runs a call that matches the schema", async () => {
    const ran: { input?: unknown } = {};
    const result = await handleToolCall(specFor(ran), "write_file", { path: "/tmp/x", contents: "hi" });
    expect(ran.input).toEqual({ path: "/tmp/x", contents: "hi" });
    expect(result.isError).toBeUndefined();
  });

  it("fails CLOSED when the validator itself throws — it is the last thing before the impl", async () => {
    const ran: { input?: unknown } = {};
    const spec = {
      ...specFor(ran),
      validator: {
        validateValue: () => {
          throw new Error("validator exploded");
        },
      },
    };
    const result = await handleToolCall(spec, "write_file", { path: "/tmp/x", contents: "hi" });
    expect(ran.input).toBeUndefined();
    expect(result.isError).toBe(true);
  });

  it("leaves an unvalidated caller exactly as it was — the seam is optional, like every other", async () => {
    const ran: { input?: unknown } = {};
    const { validator: _dropped, ...unchecked } = specFor(ran);
    await handleToolCall(unchecked, "write_file", { path: 42 });
    expect(ran.input).toEqual({ path: 42 });
  });
});

describe("serving an injected tool", () => {
  it("runs the host implementation and returns its value", async () => {
    const spec = { tools: { add: { inputSchema: {} as never, run: (i: Record<string, unknown>) => ({ sum: (i.a as number) + (i.b as number) }) } } };
    const result = await handleToolCall(spec, "add", { a: 2, b: 3 });
    expect(JSON.parse(result.content[0]!.text)).toEqual({ sum: 5 });
    expect(result.isError).toBeUndefined();
  });

  it("returns a string result unwrapped", async () => {
    const spec = { tools: { echo: { inputSchema: {} as never, run: () => "hello" } } };
    expect((await handleToolCall(spec, "echo", {})).content[0]!.text).toBe("hello");
  });

  // A tool failure travels back to the MODEL as a result it reads and reacts to — it is not the
  // classified-failure channel, and it must not fault the transport.
  it("turns a THROWING tool into an isError result, not a transport fault", async () => {
    const spec = {
      tools: {
        boom: {
          inputSchema: {} as never,
          run: () => {
            throw new Error("disk full");
          },
        },
      },
    };
    const result = await handleToolCall(spec, "boom", {});
    expect(result.isError).toBe(true);
    expect(result.content[0]!.text).toMatch(/tool 'boom' failed: disk full/);
  });

  it("reports an unknown tool as an error result", async () => {
    const result = await handleToolCall({ tools: {} }, "nope", {});
    expect(result.isError).toBe(true);
  });

  it("passes a non-object argument bag through as an empty input rather than crashing", async () => {
    const spec = { tools: { peek: { inputSchema: {} as never, run: (i: unknown) => i as never } } };
    expect(JSON.parse((await handleToolCall(spec, "peek", "not-an-object")).content[0]!.text)).toEqual({});
  });
});
