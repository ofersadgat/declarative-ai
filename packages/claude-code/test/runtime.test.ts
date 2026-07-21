import { describe, expect, it } from "vitest";
import type { Approver, RuntimeOp, Tool } from "@declarative-ai/core";
import { createClaudeCodeRuntime } from "../src";
import type { AgentPermissionDecision, AgentQuery, AgentQueryOptions, AgentToolRequest } from "../src";

const op = (over: Partial<RuntimeOp> = {}): RuntimeOp => ({ prompt: "do it", config: {}, ...over });
const tool = (): Tool => ({ inputSchema: { type: "object" }, run: () => ({}) });

describe("createClaudeCodeRuntime — delegated agent adapter", () => {
  it("maps op + ctx onto the query and returns the terminal result (value/rawText/cost)", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "assistant" };
      yield { type: "result", result: { text: "done", costUsd: 0.02 } };
    };
    const outcome = await createClaudeCodeRuntime({ query })
      .run(op({ config: { permissionMode: "plan" }, tools: { read_file: tool(), write_file: tool() } }), { workspace: { root: "/repo" } })
      .outcome;

    expect(outcome.error).toBeUndefined();
    expect(outcome.value).toBe("done");
    expect(outcome.rawText).toBe("done");
    expect(outcome.metrics.cost).toBe(0.02);
    expect(captured?.prompt).toBe("do it");
    expect(captured?.cwd).toBe("/repo");
    expect(Object.keys(captured!.mcpTools!)).toEqual(["read_file", "write_file"]); // injected by default
    expect(captured?.allowedTools).toEqual([]); // none resolved to a native built-in
    expect(captured?.permissionMode).toBe("plan");
  });

  it("ignores an unknown permissionMode and omits allowedTools when the state declares none", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeRuntime({ query }).run(op({ config: { permissionMode: "bogus" } }), {}).outcome;
    expect(captured?.permissionMode).toBeUndefined();
    expect(captured?.allowedTools).toBeUndefined();
    expect(captured?.cwd).toBeUndefined();
  });

  it("routes the agent's tool-approval callback through ctx.approve (allow/deny mapped)", async () => {
    const decisions: AgentPermissionDecision[] = [];
    const query: AgentQuery = async function* (opts) {
      for (const req of [
        { toolName: "write_file", input: { path: "x" } },
        { toolName: "read_file", input: {} },
      ] satisfies AgentToolRequest[]) {
        decisions.push(await opts.canUseTool!(req, { signal: new AbortController().signal }));
      }
      yield { type: "result", result: { text: "ok" } };
    };
    const approve: Approver = (r) => ({ decision: r.tool === "write_file" ? "deny" : "allow", scope: "once" });
    await createClaudeCodeRuntime({ query }).run(op(), { approve }).outcome;

    expect(decisions[0]).toMatchObject({ allow: false });
    expect(decisions[1]).toEqual({ allow: true });
  });

  it("supplies no canUseTool when ctx has no approver", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeRuntime({ query }).run(op(), {}).outcome;
    expect(captured?.canUseTool).toBeUndefined();
  });

  it("maps an agent error message to a permanent failure", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "other", error: "boom" };
    };
    const outcome = await createClaudeCodeRuntime({ query }).run(op(), {}).outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/boom/);
  });

  it("fails when the stream yields no result message", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "assistant" };
    };
    const outcome = await createClaudeCodeRuntime({ query }).run(op(), {}).outcome;
    expect(outcome.error?.reason).toMatch(/no result/);
  });

  it("normalizes a thrown query to a permanent failure (never rejects)", async () => {
    const query: AgentQuery = async function* () {
      throw new Error("kaboom");
    };
    const outcome = await createClaudeCodeRuntime({ query }).run(op(), {}).outcome;
    expect(outcome.error?.classification).toBe("permanent");
    expect(outcome.error?.reason).toMatch(/kaboom/);
  });

  it("injects op.tools as mcpTools whose run routes to our Tool with the runtime ctx", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      // Simulate the agent invoking our injected tool over MCP.
      if (opts.mcpTools?.["read_file"]) await opts.mcpTools["read_file"].run({ path: "x" });
      yield { type: "result", result: { text: "ok" } };
    };
    let seenInput: unknown;
    let seenWorkspace: string | undefined;
    const readTool: Tool = {
      description: "read",
      inputSchema: { type: "object" },
      capabilities: { readOnly: true },
      run: (input, ctx) => {
        seenInput = input;
        seenWorkspace = ctx.workspace?.root;
        return { content: "hi" };
      },
    };
    await createClaudeCodeRuntime({ query }).run(op({ tools: { read_file: readTool } }), { workspace: { root: "/repo" } }).outcome;

    expect(Object.keys(captured!.mcpTools!)).toEqual(["read_file"]);
    expect(captured!.mcpTools!["read_file"]!.inputSchema).toEqual({ type: "object" });
    expect(seenInput).toEqual({ path: "x" }); // the agent's tool input reached our impl
    expect(seenWorkspace).toBe("/repo"); // our tool ran with the runtime's ctx (workspace threaded through)
  });

  it("injectTools:false passes no mcpTools (native allow-list mode)", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeRuntime({ query, injectTools: false }).run(op({ tools: { read_file: tool() } }), {}).outcome;
    expect(captured?.mcpTools).toBeUndefined();
    expect(captured?.allowedTools).toEqual(["read_file"]);
  });

  it("nativeTools resolves a logical tool to the agent's aliased built-in, injecting the rest", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeRuntime({ query, nativeTools: { read_file: { native: "Read" } } })
      .run(op({ tools: { read_file: tool(), bash: tool() } }), {}).outcome;
    expect(captured?.allowedTools).toEqual(["Read"]); // read_file → the agent's native "Read"
    expect(Object.keys(captured!.mcpTools!)).toEqual(["bash"]); // bash → injected (our impl)
  });

  it("cancel() yields a canceled outcome even if a late result arrives", async () => {
    const query: AgentQuery = async function* (opts) {
      await new Promise<void>((resolve) => {
        if (opts.abortSignal?.aborted) resolve();
        else opts.abortSignal?.addEventListener("abort", () => resolve(), { once: true });
      });
      yield { type: "result", result: { text: "late" } };
    };
    const handle = createClaudeCodeRuntime({ query }).run(op(), {});
    setTimeout(() => void handle.cancel(), 10);
    const outcome = await handle.outcome;
    expect(outcome.error?.classification).toBe("canceled");
  });
});
