import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import type { LanguageModel } from "ai";
import { createLogger } from "./logger";
import { installLongTimeoutDispatcher } from "./dispatcher";

const log = createLogger("engine.providers.router");

/**
 * Provider router (§5): Anthropic models -> the native Anthropic provider; everything
 * else -> OpenRouter as the catch-all. The model id is part of a candidate's identity,
 * so routing is a pure function of it.
 *
 * `ProviderRouter` is the seam §6.1's `RunCtx` depends on — a swappable interface, not
 * a hardcoded call.
 */
export type ModelFamily = "anthropic" | "openrouter";

export function isAnthropicModel(modelId: string): boolean {
  return modelId.startsWith("claude-");
}

export function familyForModel(modelId: string): ModelFamily {
  return isAnthropicModel(modelId) ? "anthropic" : "openrouter";
}

/**
 * Whether a resolved model runs on Anthropic's native provider — which is what decides
 * if the structured-output schema patch is needed (§5.1). Derived from the model object
 * itself (its `.provider`), falling back to the model id; callers never pass a flag.
 * An Anthropic model reached *via OpenRouter* reports the OpenRouter provider and is
 * (correctly) treated as non-native.
 */
export function providerIsAnthropic(model: LanguageModel, modelId: string): boolean {
  const provider = (model as { provider?: unknown }).provider;
  if (typeof provider === "string" && provider.length > 0) {
    return provider.toLowerCase().includes("anthropic");
  }
  return isAnthropicModel(modelId);
}

/** Per-call model-resolution options. Today: the schema adapter's enforce decision, which sets the
 *  OpenRouter constrained-decoding (strict) flag for THIS call (§5.1). */
export interface ResolveModelOptions {
  /** Request OpenRouter's strict structured-output mode for this call. Ignored by the native Anthropic
   *  provider (it decides strict via its own structured-output path). When omitted, falls back to the
   *  router-level `openRouterStrictStructuredOutputs` default. */
  strictStructuredOutput?: boolean;
}

export interface ProviderRouter {
  resolveModel(modelId: string, opts?: ResolveModelOptions): LanguageModel;
  isAnthropic(modelId: string): boolean;
}

export interface RouterOptions {
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
export function createRouter(options: RouterOptions = {}): ProviderRouter {
  if (!options.skipDispatcher) installLongTimeoutDispatcher();

  let anthropic: ReturnType<typeof createAnthropic> | undefined;
  let openrouter: ReturnType<typeof createOpenRouter> | undefined;

  const resolveModel = (modelId: string, opts: ResolveModelOptions = {}): LanguageModel => {
    const family = familyForModel(modelId);
    log.debug("resolve model", { modelId, family, strict: opts.strictStructuredOutput });
    if (family === "anthropic") {
      anthropic ??= createAnthropic({
        apiKey: options.anthropicApiKey ?? process.env.ANTHROPIC_API_KEY,
      });
      return anthropic(modelId);
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
    return openrouter(modelId, {
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
    // Provider-based (not id-based): aligns with the schema-patch decision in generate.ts
    // so a claude model reached *via OpenRouter* is correctly treated as non-native (§5.1).
    isAnthropic: (modelId: string) => providerIsAnthropic(resolveModel(modelId), modelId),
  };
}
