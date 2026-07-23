import { describe, expect, it } from "vitest";
import { isOk, type ExecServices, type FunctionInputs, type JsonValue, type Tool } from "@declarative-ai/exec";
import type { Approver } from "@declarative-ai/permissions";
import { functionOp, runtimeOp } from "@declarative-ai/exec";
import { createClaudeCodeFunction, DELEGATED_CAPS } from "../src";
import type { AgentPermissionDecision, AgentQuery, AgentQueryOptions, AgentToolRequest } from "../src";

/** The op's bound inputs, as the engine hands them to the registered async function (§3.1). */
const inputs = (over: { prompt?: string; config?: JsonValue } = {}): FunctionInputs => ({
  prompt: over.prompt ?? "do it",
  config: over.config ?? {},
});
const tool = (): Tool => ({ inputSchema: { type: "object" }, readOnly: true, run: () => ({}) });
/** Run the adapter and split its `Result`. Errors are DATA now (§4.2): the adapter RESOLVES a
 *  classified failure instead of throwing, so there is nothing for the caller to guess at. */
async function runOrError(
  fn: ReturnType<typeof createClaudeCodeFunction>,
  i: FunctionInputs,
  ctx: ExecServices,
): Promise<{ text?: string; error?: { classification: string; reason: string } }> {
  const result = await fn.run(i, ctx);
  return isOk(result) ? { text: result.value } : { error: result.error };
}

describe("createClaudeCodeFunction — delegated agent as a registered async function", () => {
  it("registers delegated-agent capabilities on the ENTRY, not the op (§3.1)", () => {
    const fn = createClaudeCodeFunction();
    expect(fn.capabilities).toEqual(DELEGATED_CAPS);
    expect(fn.capabilities?.memoizable).toBe(false); // own-loop agent — never memoized
    expect(fn.capabilities?.mutatesWorkspace).toBe(true);
    expect(fn.capabilities?.policyEnforcement).toBe("callback");
  });

  it("the authoring builder lowers to a PLAIN FunctionOp — no extra field, no runtime marker", () => {
    const op = runtimeOp({ runtime: "claude-code", prompt: "fix the test", config: { permissionMode: "plan" } });
    expect(op.kind).toBe("function");
    expect(op.functionRef).toBe("claude-code");
    expect(Object.keys(op).sort()).toEqual(["functionRef", "input", "kind", "output"]);
    expect(op.input.prompt?.binding).toEqual({ text: "fix the test" });
    expect(op.input.config?.binding).toEqual({ json: { permissionMode: "plan" } });
  });

  it("maps inputs + ctx onto the query and returns the agent's text", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "assistant" };
      yield { type: "result", result: { text: "done", costUsd: 0.02 } };
    };
    const result = await createClaudeCodeFunction({ query }).run(inputs({ config: { permissionMode: "plan" } }), {
      workspace: { root: "/repo" },
      tools: { read_file: tool(), write_file: tool() },
    });

    expect(isOk(result) && result.value).toBe("done");
    // The agent's spend rides back on `Result.metrics` — the only channel a delegated agent has to
    // report what it cost, since it bills inside its own loop.
    expect(result.metrics).toMatchObject({ costUsd: 0.02, childCostUsd: 0.02, childLlmCalls: 1, costSource: "provider" });
    expect(captured?.prompt).toBe("do it");
    expect(captured?.cwd).toBe("/repo");
    expect(Object.keys(captured!.mcpTools!)).toEqual(["read_file", "write_file"]); // injected by default
    expect(captured?.allowedTools).toEqual([]); // none resolved to a native built-in
    expect(captured?.permissionMode).toBe("plan");
  });

  it("settles the agent's reported spend against an injected wallet", async () => {
    const settled: number[] = [];
    const reserved: number[] = [];
    const query: AgentQuery = async function* () {
      yield { type: "result", result: { text: "done", costUsd: 0.05 } };
    };
    await createClaudeCodeFunction({ query }).run(inputs(), {
      meter: {
        reserve: async (est) => {
          reserved.push(est);
          return { settle: async (actual) => void settled.push(actual) };
        },
        availableCostUsd: async () => 100,
      },
    });
    expect(reserved).toEqual([0.05]);
    expect(settled).toEqual([0.05]);
  });

  // The agent bills inside its own loop, so its charge arrives as a FACT. `reserve` returning null
  // (balance can't cover it) meant `reservation?.settle(...)` no-opped and the spend vanished — the
  // wallet then kept reporting headroom it did not have and admitted the next call on it.
  it("debits spend the wallet could not have reserved, rather than dropping it", async () => {
    const debited: number[] = [];
    const query: AgentQuery = async function* () {
      yield { type: "result", result: { text: "done", costUsd: 5 } };
    };
    const result = await createClaudeCodeFunction({ query }).run(inputs(), {
      meter: {
        reserve: async () => null, // over budget — the money is already spent regardless
        availableCostUsd: async () => 0,
        debit: async (actual) => void debited.push(actual),
      },
    });
    expect(debited).toEqual([5]);
    // The result survives: failing the op here would discard the agent's work over bookkeeping.
    expect(isOk(result) && result.value).toBe("done");
    expect(result.metrics?.costUsd).toBe(5);
  });

  it("reports cost on metrics even with no wallet wired at all", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "result", result: { text: "done", costUsd: 0.07 } };
    };
    const result = await createClaudeCodeFunction({ query }).run(inputs(), {});
    // With no meter, the spend used to vanish entirely — this is the only channel left.
    expect(result.metrics).toMatchObject({ costUsd: 0.07, childCostUsd: 0.07, childLlmCalls: 1 });
    expect(typeof result.metrics?.durationMs).toBe("number");
  });

  it("ignores an unknown permissionMode and omits allowedTools when no tools are supplied", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeFunction({ query }).run(inputs({ config: { permissionMode: "bogus" } }), {});
    expect(captured?.permissionMode).toBeUndefined();
    expect(captured?.allowedTools).toBeUndefined();
    expect(captured?.cwd).toBeUndefined();
  });

  it("routes the agent's tool-approval callback through ctx.approve (allow/deny mapped)", async () => {
    const decisions: AgentPermissionDecision[] = [];
    const query: AgentQuery = async function* (opts) {
      const requests: AgentToolRequest[] = [
        { toolName: "write_file", input: { path: "x" } },
        { toolName: "read_file", input: {} },
      ];
      for (const req of requests) {
        decisions.push(await opts.canUseTool!(req, { signal: new AbortController().signal }));
      }
      yield { type: "result", result: { text: "ok" } };
    };
    const approve: Approver = (r: { tool: string }) => ({ decision: r.tool === "write_file" ? "deny" : "allow", scope: "once" });
    await createClaudeCodeFunction({ query }).run(inputs(), { approve });

    expect(decisions[0]).toMatchObject({ allow: false });
    expect(decisions[1]).toEqual({ allow: true });
  });

  it("uses the config's sessionId as the approval scope key", async () => {
    const seen: string[] = [];
    const query: AgentQuery = async function* (opts) {
      await opts.canUseTool!({ toolName: "bash", input: {} }, { signal: new AbortController().signal });
      yield { type: "result", result: { text: "ok" } };
    };
    const approve: Approver = (r: { sessionId: string }) => {
      seen.push(r.sessionId);
      return { decision: "allow", scope: "once" };
    };
    await createClaudeCodeFunction({ query }).run(inputs({ config: { sessionId: "review-1" } }), { approve });
    expect(seen).toEqual(["review-1"]);
  });

  it("supplies no canUseTool when ctx has no approver", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeFunction({ query }).run(inputs(), {});
    expect(captured?.canUseTool).toBeUndefined();
  });

  it("resolves a classified failure on an agent error message", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "other", error: "boom" };
    };
    const { error } = await runOrError(createClaudeCodeFunction({ query }), inputs(), {});
    expect(error?.reason).toMatch(/boom/);
    expect(error?.classification).toBe("permanent");
  });

  it("resolves a classified failure when the stream yields no result message", async () => {
    const query: AgentQuery = async function* () {
      yield { type: "assistant" };
    };
    const { error } = await runOrError(createClaudeCodeFunction({ query }), inputs(), {});
    expect(error?.reason).toMatch(/no result/);
  });

  it("wraps a thrown query with adapter context in the failure reason", async () => {
    const query: AgentQuery = async function* () {
      throw new Error("kaboom");
    };
    const { error } = await runOrError(createClaudeCodeFunction({ query }), inputs(), {});
    expect(error?.reason).toMatch(/claude-code: .*kaboom/);
  });

  it("injects ctx.tools as mcpTools whose run routes to our Tool with the same ctx", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      // Simulate the agent invoking our injected tool over MCP.
      if (opts.mcpTools?.["read_file"]) await opts.mcpTools["read_file"].run({ path: "x" });
      yield { type: "result", result: { text: "ok" } };
    };
    let seenInput: JsonValue | undefined;
    let seenWorkspace: string | undefined;
    const readTool: Tool = {
      description: "read",
      inputSchema: { type: "object" },
      readOnly: true,
      run: (input: FunctionInputs, ctx: ExecServices) => {
        seenInput = input as JsonValue;
        seenWorkspace = ctx.workspace?.root;
        return { content: "hi" };
      },
    };
    await createClaudeCodeFunction({ query }).run(inputs(), { workspace: { root: "/repo" }, tools: { read_file: readTool } });

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
    await createClaudeCodeFunction({ query, injectTools: false }).run(inputs(), { tools: { read_file: tool() } });
    expect(captured?.mcpTools).toBeUndefined();
    expect(captured?.allowedTools).toEqual(["read_file"]);
  });

  it("carries a per-tool `deny` from the policy baseline to the agent as a DENY-LIST", async () => {
    // A `deny` needs no human, so waiting for an approval that will never be asked for is not
    // enforcement. Without this channel it could not be expressed to a delegated agent at all.
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeFunction({ query, injectTools: false }).run(inputs(), {
      tools: { read_file: tool(), bash: tool() },
      policy: { baseline: { tools: { bash: "deny", read_file: "ask" } } },
    });
    expect(captured?.disallowedTools).toEqual(["bash"]);
    // And a denied tool must not ALSO be pre-approved — `allowedTools` is the CLI's pre-approval list,
    // so a name on both is a contradiction the agent would resolve for us.
    expect(captured?.allowedTools).toEqual(["read_file"]);
  });

  it("denies an aliased tool under the NATIVE name the agent addresses", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeFunction({ query, nativeTools: { read_file: { native: "Read" } } }).run(inputs(), {
      tools: { read_file: tool() },
      policy: { baseline: { tools: { read_file: "deny" } } },
    });
    expect(captured?.disallowedTools).toEqual(["Read"]);
    expect(captured?.allowedTools).toEqual([]);
  });

  it("passes ctx.validator down so injected tool ARGUMENTS are checked at the agent boundary", async () => {
    // An agent's tool call arrives as untyped JSON and no MCP server validates it, so the adapter must
    // hand the seam on or the check documented on the bridge does not exist on the real path.
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    let hits = 0;
    const validator = {
      validateValue: () => {
        hits++;
        return { ok: true };
      },
    };
    await createClaudeCodeFunction({ query }).run(inputs(), { tools: { bash: tool() }, validator });
    // The ctx validator arrives WRAPPED by json's fail-closed `syncOnly` narrowing, so identity is not
    // the claim — DELEGATION is: the query-side validator consults the injected one.
    expect(captured?.validator).toBeDefined();
    expect(captured!.validator!.validateValue({} as never, 1 as never)).toEqual({ ok: true });
    expect(hits).toBe(1);
  });

  it("nativeTools resolves a logical tool to the agent's aliased built-in, injecting the rest", async () => {
    let captured: AgentQueryOptions | undefined;
    const query: AgentQuery = async function* (opts) {
      captured = opts;
      yield { type: "result", result: { text: "x" } };
    };
    await createClaudeCodeFunction({ query, nativeTools: { read_file: { native: "Read" } } }).run(inputs(), {
      tools: { read_file: tool(), bash: tool() },
    });
    expect(captured?.allowedTools).toEqual(["Read"]); // read_file → the agent's native "Read"
    expect(Object.keys(captured!.mcpTools!)).toEqual(["bash"]); // bash → injected (our impl)
  });

  it("an aborted ctx.abortSignal cancels even if a late result arrives", async () => {
    const controller = new AbortController();
    const query: AgentQuery = async function* (opts) {
      await new Promise<void>((resolve) => {
        if (opts.abortSignal?.aborted) resolve();
        else opts.abortSignal?.addEventListener("abort", () => resolve(), { once: true });
      });
      yield { type: "result", result: { text: "late" } };
    };
    setTimeout(() => controller.abort(), 10);
    const { error } = await runOrError(createClaudeCodeFunction({ query }), inputs(), { abortSignal: controller.signal });
    // An abort is `canceled`, not a failure of the unit — classified HERE now, as data (§4.2),
    // rather than reconstructed from `err.name` by the caller.
    expect(error?.classification).toBe("canceled");
  });

  it("is invocable as a plain FunctionOp through a typed def application", () => {
    // The adapter is named by `functionRef` like any other registered function — nothing about the op
    // shape distinguishes a delegated runtime from `parse` or `combine`.
    const op = functionOp(
      { name: "claude-code", input: { type: "object" }, output: { type: "string" }, impl: () => "" },
      { prompt: "go" },
    );
    expect(op.kind).toBe("function");
    expect(op.functionRef).toBe("claude-code");
  });
});
