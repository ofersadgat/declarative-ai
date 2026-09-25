/**
 * What an agent binary says it runs — its catalog rows, asked of the binary itself (JaiRA decision 0009).
 *
 * Each agent route gets rows of its own (`claude-cli/…`, `codex-cli/…`), because a transport and the API
 * behind it accept different things for one model: the claude CLI takes `--effort max` and no budget, and
 * codex takes whatever levels ITS `model/list` names. A row carries what that transport takes and no
 * price — a call on it is priced by the API row with the same canonical id (`ModelInfo.pricedRow`).
 *
 * A binary's list is its PICKER MENU, not everything it can run: neither `claude` lists
 * `claude-opus-5-5`, and both run it by id. So a model missing here is unknown — fitted as asked — never
 * refused.
 *
 * Both probes start the binary for nothing else, ask, and close its input; neither sends a prompt, so no
 * model is called and nothing is spent.
 */
import { canonicalIdFor, orderedEfforts, parametersFromNames, type CatalogSource, type ModelInfoInterface, type ReasoningCapability } from "@declarative-ai/llm";
import { defaultSpawn, type AgentProcess, type SpawnProcess } from "./process.js";

export interface AgentModelsProbeOptions {
  /** The binary to ask. Default `claude` / `codex`. */
  command?: string;
  /** The route the rows are filed under. Default `claude-cli` / `codex-cli`. */
  route?: string;
  /** Process seam (tests inject a fake). */
  spawn?: SpawnProcess;
  /** How long the whole exchange may take. Default 20 s. */
  timeoutMs?: number;
  signal?: AbortSignal;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}

/** Start `argv`, write `requests`, and hand each JSON line read back to `onLine` until it returns a value. */
async function exchange<T>(argv: string[], requests: readonly unknown[], options: AgentModelsProbeOptions, onLine: (msg: Record<string, unknown>, write: (m: unknown) => void) => T | undefined): Promise<T> {
  const spawn = options.spawn ?? (await defaultSpawn());
  let child: AgentProcess | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const kill = (): void => child?.kill();
  try {
    child = spawn(argv, {
      ...(options.cwd !== undefined ? { cwd: options.cwd } : {}),
      ...(options.env !== undefined ? { env: options.env } : {}),
      keepInputOpen: true,
    });
    const c = child;
    if (c.write === undefined) throw new Error("the process seam offers no input channel");
    if (options.signal?.aborted) throw new Error("aborted");
    options.signal?.addEventListener("abort", kill, { once: true });
    timer = setTimeout(kill, options.timeoutMs ?? 20_000);
    const write = (m: unknown): void => c.write!(`${JSON.stringify(m)}\n`);
    for (const request of requests) write(request);
    for await (const line of c.lines) {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(line) as Record<string, unknown>;
      } catch {
        continue;
      }
      const done = onLine(msg, write);
      if (done !== undefined) return done;
    }
    throw new Error(`${argv[0]} ended without answering`);
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", kill);
    child?.endInput?.();
    child?.kill();
  }
}

// --- claude -------------------------------------------------------------------

/** One entry of claude's initialize `models` — the SDK's `ModelInfo`, the slice read here. */
interface ClaudeModelEntry {
  value?: unknown;
  resolvedModel?: unknown;
  displayName?: unknown;
  supportedEffortLevels?: unknown;
}

/**
 * claude's initialize `models` as rows under `route`.
 *
 * MEASURED 2026-09-24. The SDK's bundled 2.1.223 answers entries like `{value: "default", resolvedModel:
 * "claude-opus-5[1m]", supportedEffortLevels: ["low","medium","high","xhigh","max"]}`; the installed
 * 2.1.142 answers bare aliases (`{value: "sonnet", supportedEffortLevels: ["low","medium","high","max"]}`)
 * with no `resolvedModel`; haiku carries no levels on either. So:
 *
 *  - an entry becomes a row under the id it RESOLVES to, and another under the alias it is picked by
 *    (`default`, `sonnet`, `opus[1m]`) — both are what a config may name;
 *  - its levels become `reasoning.effort`'s enum, with no budget (the CLI has no budget flag);
 *  - no levels ⇒ no `reasoning` at all, so a request for one is dropped with a note rather than
 *    refused by the binary.
 */
export function claudeModelRows(models: unknown, route = "claude-cli"): ModelInfoInterface[] {
  if (!Array.isArray(models)) return [];
  const rows = new Map<string, ModelInfoInterface>();
  for (const entry of models as ClaudeModelEntry[]) {
    const value = typeof entry.value === "string" && entry.value.length > 0 ? entry.value : undefined;
    if (value === undefined) continue;
    const resolved = typeof entry.resolvedModel === "string" && entry.resolvedModel.length > 0 ? entry.resolvedModel : undefined;
    const levels = Array.isArray(entry.supportedEffortLevels) ? orderedEfforts(entry.supportedEffortLevels.filter((l): l is string => typeof l === "string")) : [];
    const reasoning: ReasoningCapability | undefined = levels.length > 0 ? { efforts: levels, budget: false } : undefined;
    const base = {
      route,
      provider: "Anthropic",
      source: "claude-models",
      // The resolved id names the model; an alias alone names only itself, and prices nothing.
      canonicalId: canonicalIdFor(resolved ?? value),
      parameters: parametersFromNames([], reasoning !== undefined ? { reasoning } : {}),
    };
    const label = typeof entry.displayName === "string" && entry.displayName.length > 0 ? entry.displayName : value;
    for (const model of [resolved, value]) {
      if (model === undefined || rows.has(model)) continue;
      rows.set(model, { ...base, model, label });
    }
  }
  return [...rows.values()];
}

/**
 * Ask claude for its models: `-p` with streaming input, the `initialize` control request, read its
 * answer's `models`. The same no-prompt exchange `probeClaudeUsage` makes.
 */
export async function probeClaudeModels(options: AgentModelsProbeOptions = {}): Promise<ModelInfoInterface[]> {
  const argv = [options.command ?? "claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"];
  return exchange(argv, [{ type: "control_request", request_id: "models_init", request: { subtype: "initialize" } }], options, (msg) => {
    const response = msg["response"] as { request_id?: unknown; subtype?: unknown; response?: { models?: unknown }; error?: unknown } | undefined;
    if (msg["type"] !== "control_response" || response?.request_id !== "models_init") return undefined;
    if (response.subtype !== "success") throw new Error(typeof response.error === "string" ? response.error : "claude refused to initialize");
    return claudeModelRows(response.response?.models, options.route ?? "claude-cli");
  });
}

/** A catalog source over {@link probeClaudeModels} — for `refreshModelCatalog`. Strict: a short list. */
export function claudeModelsSource(options: AgentModelsProbeOptions = {}): CatalogSource {
  return { name: `${options.route ?? "claude-cli"}-models`, fetchRows: () => probeClaudeModels(options) };
}

// --- codex --------------------------------------------------------------------

/** One entry of codex's `model/list` — the slice read here. */
interface CodexModelEntry {
  id?: unknown;
  model?: unknown;
  displayName?: unknown;
  hidden?: unknown;
  isDefault?: unknown;
  supportedReasoningEfforts?: unknown;
  defaultReasoningEffort?: unknown;
  inputModalities?: unknown;
}

/**
 * codex's `model/list` entries as rows under `route`.
 *
 * MEASURED 2026-09-24 (codex-cli 0.147.0): `{id: "gpt-5.6-terra", displayName: "GPT-5.6-Terra",
 * hidden: false, isDefault: false, supportedReasoningEfforts: [{reasoningEffort: "low", description: …},
 * …, {reasoningEffort: "ultra", …}], defaultReasoningEffort: "medium", inputModalities: ["text","image"]}`.
 * The levels are the ones codex passes as `model_reasoning_effort` — the API rejects anything else,
 * and codex does not check. The entry marked `isDefault` also answers for `default`, the id a config
 * names to mean "whatever codex uses".
 */
export function codexModelRows(entries: unknown, route = "codex-cli"): ModelInfoInterface[] {
  if (!Array.isArray(entries)) return [];
  const rows: ModelInfoInterface[] = [];
  for (const entry of entries as CodexModelEntry[]) {
    const id = typeof entry.id === "string" ? entry.id : typeof entry.model === "string" ? entry.model : undefined;
    if (id === undefined || id.length === 0) continue;
    const efforts = Array.isArray(entry.supportedReasoningEfforts)
      ? orderedEfforts(
          entry.supportedReasoningEfforts
            .map((e) => (e !== null && typeof e === "object" ? (e as { reasoningEffort?: unknown }).reasoningEffort : undefined))
            .filter((e): e is string => typeof e === "string"),
        )
      : [];
    const fallback = typeof entry.defaultReasoningEffort === "string" ? orderedEfforts([entry.defaultReasoningEffort])[0] : undefined;
    const inputs = Array.isArray(entry.inputModalities) ? entry.inputModalities.filter((m): m is string => typeof m === "string") : undefined;
    const row: ModelInfoInterface = {
      route,
      model: id,
      provider: "OpenAI",
      label: typeof entry.displayName === "string" && entry.displayName.length > 0 ? entry.displayName : id,
      source: "codex-models",
      canonicalId: canonicalIdFor(id),
      parameters: parametersFromNames(
        [],
        efforts.length > 0 ? { reasoning: { efforts, ...(fallback !== undefined ? { defaultEffort: fallback } : {}), budget: false } } : {},
      ),
      ...(inputs !== undefined && inputs.length > 0 ? { modalities: { input: inputs, output: ["text"] } } : {}),
      ...(entry.hidden === true ? { available: false } : {}),
    };
    rows.push(row);
    if (entry.isDefault === true) rows.push({ ...row, model: "default", label: `default (${row.label})` });
  }
  return rows;
}

/**
 * Ask codex for its models: `codex app-server` over JSON-RPC — `initialize`, `initialized`, then
 * `model/list` page by page (`nextCursor`). ✅ Answered by codex-cli 0.147.0.
 */
export async function probeCodexModels(options: AgentModelsProbeOptions = {}): Promise<ModelInfoInterface[]> {
  const argv = [options.command ?? "codex", "app-server"];
  const route = options.route ?? "codex-cli";
  const entries: unknown[] = [];
  let next = 2;
  return exchange(argv, [{ id: 1, method: "initialize", params: { clientInfo: { name: "declarative-ai", version: "0" } } }], options, (msg, write) => {
    if (msg["id"] === 1) {
      write({ method: "initialized" });
      write({ id: next, method: "model/list", params: {} });
      return undefined;
    }
    if (msg["id"] !== next) return undefined;
    if (msg["error"] !== undefined) {
      const error = msg["error"] as { message?: unknown };
      throw new Error(`codex model/list: ${typeof error.message === "string" ? error.message : "refused"}`);
    }
    const result = msg["result"] as { data?: unknown; nextCursor?: unknown } | undefined;
    if (Array.isArray(result?.data)) entries.push(...result.data);
    // Bounded, like any listing that might never say it is done.
    if (typeof result?.nextCursor === "string" && result.nextCursor.length > 0 && next < 20) {
      next += 1;
      write({ id: next, method: "model/list", params: { cursor: result.nextCursor } });
      return undefined;
    }
    return codexModelRows(entries, route);
  });
}

/** A catalog source over {@link probeCodexModels} — for `refreshModelCatalog`. */
export function codexModelsSource(options: AgentModelsProbeOptions = {}): CatalogSource {
  return { name: `${options.route ?? "codex-cli"}-models`, fetchRows: () => probeCodexModels(options) };
}
