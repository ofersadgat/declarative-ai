/**
 * Dispatch by model ROUTE — the sibling of `exec`'s dispatch by op KIND.
 *
 * The behaviour worth pinning is mostly about what this executor does NOT do: it does not parse the
 * id beyond the first `/`, it does not strip the prefix on the way down, and it does not know which
 * routes `llm` serves. Those are all `ModelRouter`'s business, and duplicating any of them here would
 * mean two places to update when a provider is added.
 */
import { describe, expect, it } from "vitest";
import { finishedHandle, isOk, promptOp, type ExecHandle, type ExecServices, type Executor, type InlineFamily, type Operation, type ResolvedValue } from "@declarative-ai/exec";
import type { CallEstimate, RateLimiter } from "@declarative-ai/exec";
import { isEmbeddedModel, type LlmMetrics } from "@declarative-ai/llm";
import { withRateLimit } from "../src/wrappers.js";
import { PromptRouterExecutor, routePrefixOf } from "../src/routerExecutor.js";

const op = (model?: string) =>
  promptOp({ user: "hi", ...(model !== undefined ? { config: { model } } : {}), output: { name: "a", schema: { type: "string" } } });

/** An executor that answers with its own name, so a test can see WHICH one was reached. */
function named(name: string, capabilities?: Partial<{ structuredOutput: boolean }>) {
  const seen: Array<Operation<InlineFamily>> = [];
  const executor: Executor<ExecServices, LlmMetrics, Operation<InlineFamily>, ResolvedValue> = {
    capabilities: { structuredOutput: true, mutatesWorkspace: false, policyEnforcement: "none", sessionResume: false, streaming: false, runtime: "node", interactive: false, readOnly: true, memoizable: true, ...capabilities } as never,
    metrics: { merge: (a) => a, empty: () => ({ durationMs: 0, costUsd: 0, costSource: "unknown" }) },
    start: (o, _ctx): ExecHandle<ResolvedValue, LlmMetrics> => {
      seen.push(o);
      return finishedHandle({ value: name as ResolvedValue, metrics: { durationMs: 0, costUsd: 0, costSource: "table" } }) as never;
    },
  };
  return { executor, seen };
}

describe("PromptRouterExecutor", () => {
  it("sends each prefix to the executor wired for it", async () => {
    const cli = named("cli");
    const provider = named("provider");
    const router = new PromptRouterExecutor({ routes: { "claude-cli": cli.executor }, fallback: provider.executor });

    expect(isOk(await router.start(op("claude-cli/sonnet"), {}).result) && (await router.start(op("claude-cli/sonnet"), {}).result).value).toBe("cli");
    const viaProvider = await router.start(op("anthropic/claude-sonnet-5"), {}).result;
    expect(isOk(viaProvider) && viaProvider.value).toBe("provider");
  });

  it("passes the id DOWN UNTOUCHED, prefix included", async () => {
    // Stripping it would hand the provider path an id its own parser rejects, and would make the
    // route invisible to everything downstream that reads the model — a memo key, a price table.
    const provider = named("provider");
    const router = new PromptRouterExecutor({ routes: {}, fallback: provider.executor });
    await router.start(op("anthropic/claude-sonnet-5"), {}).result;
    expect((provider.seen[0] as unknown as { config: { model: string } }).config.model).toBe("anthropic/claude-sonnet-5");
  });

  it("falls back for an id whose prefix names no route — `llm` owns those, and its error is the good one", async () => {
    const provider = named("provider");
    const router = new PromptRouterExecutor({ routes: { "claude-cli": named("cli").executor }, fallback: provider.executor });
    const result = await router.start(op("openrouter/openai/gpt-5"), {}).result;
    expect(isOk(result) && result.value).toBe("provider");
  });

  it("refuses — namefully — when nothing matches and there is no fallback", async () => {
    // Trying one anyway would report a missing API key for a model the caller never meant to send
    // anywhere, which is the least useful thing it could say.
    const router = new PromptRouterExecutor({ routes: { "claude-cli": named("cli").executor, anthropic: named("p").executor } });
    const result = await router.start(op("openrouter/gpt-5"), {}).result;
    expect(isOk(result)).toBe(false);
    expect(!isOk(result) && result.error.reason).toContain("openrouter/gpt-5");
    expect(!isOk(result) && result.error.reason).toContain("anthropic, claude-cli");
  });

  it("says so when the op names no model at all", async () => {
    const router = new PromptRouterExecutor({ routes: { "claude-cli": named("cli").executor } });
    const result = await router.start(op(), {}).result;
    expect(!isOk(result) && result.error.reason).toContain("no model is configured");
  });

  it("reports the capabilities of the executor that would ACTUALLY serve the op", () => {
    // Load-bearing: a wrapper gating on `structuredOutput` reads this, and an agent-served call and a
    // provider-served call disagree on exactly that field.
    const cli = named("cli", { structuredOutput: false });
    const provider = named("provider", { structuredOutput: true });
    const router = new PromptRouterExecutor({ routes: { "claude-cli": cli.executor }, fallback: provider.executor });
    expect(router.capabilitiesFor(op("claude-cli/sonnet")).structuredOutput).toBe(false);
    expect(router.capabilitiesFor(op("anthropic/x")).structuredOutput).toBe(true);
  });

  it("reads the prefix as everything before the FIRST slash, and nothing from a bare id", () => {
    expect(routePrefixOf("openrouter/openai/gpt-5")).toBe("openrouter");
    expect(routePrefixOf("claude-sonnet-5")).toBeUndefined();
    expect(routePrefixOf("/leading")).toBeUndefined();
    expect(routePrefixOf(undefined)).toBeUndefined();
  });
});

describe("the wrappers, once an agent can serve a PROMPT op", () => {
  /** Counts what it was asked to schedule, and always admits. */
  function counting() {
    let scheduled = 0;
    const limiter: RateLimiter = {
      schedule: <T,>(_est: CallEstimate, run: () => Promise<T>): Promise<T> => ((scheduled += 1), run()),
      reportOutcome: () => {},
    };
    return { limiter, scheduled: () => scheduled };
  }

  it("a scope predicate that cannot parse an agent id says 'not mine' instead of throwing", async () => {
    // `isEmbeddedModel` and friends THROW on an id whose prefix is not a SERVING route — by design, so
    // a typo'd provider id is caught. That became reachable the moment a prompt op could carry
    // `claude-cli/sonnet`, which names an executor rather than a route. Letting it out of `start` would
    // break the never-throws contract synchronously, before any handle exists to carry the failure.
    const inner = named("inner");
    const { limiter, scheduled } = counting();
    const limited = withRateLimit({ limiter, appliesTo: (m?: string) => m !== undefined && isEmbeddedModel(m) }, inner.executor as never);
    const result = await (limited as Executor).start(op("claude-cli/sonnet"), {}).result;
    expect(isOk(result) && result.value).toBe("inner");
    expect(scheduled()).toBe(0); // handed straight through, untouched
  });

  it("still governs the models its predicate DOES claim", async () => {
    const inner = named("inner");
    const { limiter, scheduled } = counting();
    const limited = withRateLimit({ limiter, appliesTo: (m?: string) => m !== undefined && isEmbeddedModel(m) }, inner.executor as never);
    await (limited as Executor).start(op("embedded/qwen"), {}).result;
    expect(scheduled()).toBe(1);
  });
});
