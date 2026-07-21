import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import type { LanguageModel } from "ai";
import { createLogger } from "./logger";
import { installLongTimeoutDispatcher } from "./dispatcher";

const log = createLogger("engine.providers.router");

/**
 * Provider router (§5): a model id is REQUIRED to name its serving route as a `{route}/{model}`
 * prefix — `anthropic/…` → the native Anthropic provider, `openrouter/…` → OpenRouter. Routing is a
 * pure, EXPLICIT function of that prefix (no `startsWith("claude-")` sniffing), so the same underlying
 * model can be reached either natively (`anthropic/claude-opus-4-8`) or via OpenRouter
 * (`openrouter/anthropic/claude-opus-4.8`) with no ambiguity. The remainder after the first "/" is the
 * provider-native id — what the catalog / pricing / schema-profile layer keys on.
 *
 * `ModelRouter` is the seam §6.1's `RunCtx` depends on — a swappable interface, not a hardcoded call.
 */
export type ModelFamily = "anthropic" | "openrouter";
/** The two serving ROUTES a model id may name in its `{route}/…` prefix (same set as {@link ModelFamily}). */
export type ModelRoute = ModelFamily;

/** A model id's parsed serving route + the provider-native id that route serves. */
export interface ParsedModel {
  route: ModelRoute;
  /** The provider-native id (route prefix stripped) — the catalog / pricing / profile key. */
  providerId: string;
}

/**
 * Parse a route-prefixed model id `{route}/{model}` (route ∈ "anthropic" | "openrouter"). Throws a clear
 * error on a bare/unprefixed id — routing is explicit by contract, never guessed. Examples:
 * `anthropic/claude-sonnet-5` → `{ route:"anthropic", providerId:"claude-sonnet-5" }`;
 * `openrouter/openai/gpt-5` → `{ route:"openrouter", providerId:"openai/gpt-5" }`.
 */
export function parseModelRoute(modelId: string): ParsedModel {
  const slash = modelId.indexOf("/");
  const route = slash > 0 ? modelId.slice(0, slash) : "";
  if (route !== "anthropic" && route !== "openrouter") {
    throw new Error(
      `model "${modelId}" must be route-prefixed as "{route}/{model}" with route "anthropic" or "openrouter" ` +
        `(e.g. "anthropic/claude-sonnet-5", "openrouter/openai/gpt-5")`,
    );
  }
  return { route, providerId: modelId.slice(slash + 1) };
}

/** The provider-native id of a route-prefixed model id (route prefix stripped) — the id the catalog /
 *  pricing / schema-profile layer keys on. Throws on a bare/unprefixed id (see {@link parseModelRoute}). */
export function providerNativeId(modelId: string): string {
  return parseModelRoute(modelId).providerId;
}

/** The serving route (family) of a route-prefixed model id. Throws on a bare/unprefixed id. */
export function familyForModel(modelId: string): ModelFamily {
  return parseModelRoute(modelId).route;
}

/**
 * True iff a PROVIDER-NATIVE id (route prefix already stripped) is a native-Anthropic model — a bare
 * `claude-*` id with NO vendor prefix. This operates on the native-id space (catalog seed/import + schema
 * profile selection, where `anthropic/claude-…` means "Anthropic-via-OpenRouter" and is correctly NON-native);
 * for routing a user-facing id, use {@link parseModelRoute}/{@link familyForModel} instead.
 */
export function isAnthropicModel(modelId: string): boolean {
  return modelId.startsWith("claude-");
}

/** Per-call model-resolution options. Today: the schema adapter's enforce decision, which sets the
 *  OpenRouter constrained-decoding (strict) flag for THIS call (§5.1). */
export interface ResolveModelOptions {
  /** Request OpenRouter's strict structured-output mode for this call. Ignored by the native Anthropic
   *  provider (it decides strict via its own structured-output path). When omitted, falls back to the
   *  router-level `openRouterStrictStructuredOutputs` default. */
  strictStructuredOutput?: boolean;
}

export interface ModelRouter {
  resolveModel(modelId: string, opts?: ResolveModelOptions): LanguageModel;
  isAnthropic(modelId: string): boolean;
}

export interface ModelRouterOptions {
  anthropicApiKey?: string;
  openRouterApiKey?: string;
  /** Skip installing the long-timeout undici dispatcher (tests with mock models). */
  skipDispatcher?: boolean;
  /**
   * Enable OpenRouter usage accounting (`usage: { include: true }`) so each response carries
   * OpenRouter's ACTUAL charged cost (§5). This is the only billing-accurate price for
   * OpenRouter — its dynamic upstream routing + markup + (possibly) normalized token counts
   * make a token×rate estimate unreliable. Default ON; costs a small latency per call.
   */
  openRouterUsageAccounting?: boolean;
  /**
   * Send `response_format.json_schema.strict` to OpenRouter (§5.1). OpenRouter defaults this
   * to `true`, which makes strict-mode providers (e.g. Azure OpenAI) REJECT the request unless
   * the schema has `additionalProperties:false` on every object AND lists every property in
   * `required` — a contract our arbitrary candidate/meta schemas don't meet (they carry genuinely
   * optional fields like `targetScore`). Forcing all-required to satisfy strict mode would make
   * the model emit those fields spuriously, corrupting the data. So default OFF: the schema is
   * sent as advisory guidance and conformance is enforced on the way out by the §4 Ajv boundary.
   * Set true only for a fleet of strict-clean schemas.
   *
   * NB the native Anthropic path is NOT symmetric with this: on Claude 4.5+ models,
   * `@ai-sdk/anthropic` (≥3.0.8x, default `structuredOutputMode: "auto"`) sends the schema as
   * Anthropic's native structured outputs (`output_config.format` json_schema — genuinely
   * grammar-constrained), falling back to the unconstrained jsonTool emulation only on older
   * models. The §4 Ajv boundary stays as the final gate on every path.
   */
  openRouterStrictStructuredOutputs?: boolean;
}

/**
 * Build a router. Provider clients are created lazily on first use so that a process
 * that only ever calls Anthropic never needs an OpenRouter key (and vice versa), and
 * so importing this module never requires any key.
 */
export function createModelRouter(options: ModelRouterOptions = {}): ModelRouter {
  if (!options.skipDispatcher) installLongTimeoutDispatcher();

  let anthropic: ReturnType<typeof createAnthropic> | undefined;
  let openrouter: ReturnType<typeof createOpenRouter> | undefined;

  const resolveModel = (modelId: string, opts: ResolveModelOptions = {}): LanguageModel => {
    const { route, providerId } = parseModelRoute(modelId);
    log.debug("resolve model", { modelId, route, providerId, strict: opts.strictStructuredOutput });
    if (route === "anthropic") {
      anthropic ??= createAnthropic({
        apiKey: options.anthropicApiKey ?? process.env.ANTHROPIC_API_KEY,
      });
      return anthropic(providerId);
    }
    openrouter ??= createOpenRouter({
      apiKey: options.openRouterApiKey ?? process.env.OPENROUTER_API_KEY,
    });
    // Usage accounting => OpenRouter returns the real charged cost in
    // providerMetadata.openrouter.usage.cost, which generateStructured prefers over the
    // price-table estimate (the only way to get accurate OpenRouter pricing, §5).
    const usageAccounting = options.openRouterUsageAccounting ?? true;
    // The per-call enforce decision (§5.1) drives strict: the schema adapter computed whether THIS
    // schema fits the provider's constrained-decoder bounds. Falls back to the router-level default
    // (then off), where correctness still comes from the §4 Ajv check (see openRouterStrictStructuredOutputs).
    const strict = opts.strictStructuredOutput ?? options.openRouterStrictStructuredOutputs ?? false;
    return openrouter(providerId, {
      ...(usageAccounting ? { usage: { include: true } } : {}),
      structuredOutputs: { strict },
      // When we actually request strict, constrain routing to upstreams that support ALL request
      // params (`require_parameters`), so OpenRouter picks a structured-output-capable provider for
      // this model instead of routing to one that ignores `response_format` and silently degrades —
      // or 400s. Left off for advisory calls, where the schema is just a hint and any upstream is fine.
      ...(strict ? { provider: { require_parameters: true } } : {}),
    });
  };

  return {
    resolveModel,
    // Route-based (§5.1): the explicit `{route}/…` prefix decides native-vs-OpenRouter, so a claude model
    // reached via `openrouter/anthropic/claude-…` is correctly treated as non-native (route "openrouter").
    isAnthropic: (modelId: string) => parseModelRoute(modelId).route === "anthropic",
  };
}
