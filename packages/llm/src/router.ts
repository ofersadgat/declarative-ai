import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { FetchFunction } from "@ai-sdk/provider-utils";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import type { LanguageModel } from "ai";
import { createLogger } from "@declarative-ai/log";
import { installLongTimeoutDispatcher } from "./dispatcher.js";
import { ManagedServer, readyGatedFetch, type ManagedServerSpec } from "./localServer.js";
import { EmbeddedModelStore, embeddedLanguageModel, type EmbeddedModelResolver } from "./embedded.js";

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
export type ModelFamily = "anthropic" | "openai" | "openrouter" | "local" | "embedded";
/** The serving ROUTES a model id may name in its `{route}/…` prefix (same set as {@link ModelFamily}). */
export type ModelRoute = ModelFamily;

/**
 * Every route, in one place, so adding one is a single edit and the parser's error message can name
 * the real set rather than a hand-maintained prose list.
 *
 * The four split on TWO independent questions, which is why there are four rather than two or three:
 *
 *  - **Who serves it** — `anthropic`/`openrouter` are remote provider fleets metered by someone else;
 *    `local`/`embedded` run on your own machine and cost no money.
 *  - **How it is reached** — `anthropic`/`openrouter`/`local` all speak HTTP (the last one to an
 *    OpenAI-compatible server, whether you started it or not), while `embedded` has no wire at all:
 *    the weights are loaded into this process.
 *
 * `local` deliberately does NOT distinguish a server you attached to from one this library spawned.
 * That is a deployment fact about lifecycle ownership, not a fact about the model, and the id names
 * the model.
 */
export const MODEL_ROUTES = ["anthropic", "openai", "openrouter", "local", "embedded"] as const satisfies readonly ModelRoute[];

/** A model id's parsed serving route + the provider-native id that route serves. */
export interface ParsedModel {
  route: ModelRoute;
  /** The provider-native id (route prefix stripped) — the catalog / pricing / profile key. */
  providerId: string;
}

function isRoute(value: string): value is ModelRoute {
  return (MODEL_ROUTES as readonly string[]).includes(value);
}

/**
 * Parse a route-prefixed model id `{route}/{model}` (route ∈ {@link MODEL_ROUTES}). Throws a clear
 * error on a bare/unprefixed id — routing is explicit by contract, never guessed. Examples:
 * `anthropic/claude-sonnet-5` → `{ route:"anthropic", providerId:"claude-sonnet-5" }`;
 * `openrouter/openai/gpt-5` → `{ route:"openrouter", providerId:"openai/gpt-5" }`;
 * `embedded/qwen2.5-32b-instruct-q4_k_m` → `{ route:"embedded", providerId:"qwen2.5-32b-instruct-q4_k_m" }`.
 */
export function parseModelRoute(modelId: string): ParsedModel {
  const slash = modelId.indexOf("/");
  const route = slash > 0 ? modelId.slice(0, slash) : "";
  if (!isRoute(route)) {
    throw new Error(
      `model "${modelId}" must be route-prefixed as "{route}/{model}" with route ${MODEL_ROUTES.map((r) => `"${r}"`).join(" | ")} ` +
        `(e.g. "anthropic/claude-sonnet-5", "openrouter/openai/gpt-5", "embedded/qwen2.5-7b-instruct-q4_k_m")`,
    );
  }
  return { route, providerId: modelId.slice(slash + 1) };
}

/**
 * True iff the model is served by a REMOTE provider fleet — someone else's capacity, someone else's
 * quota, and a bill. The complement of {@link isLocalModel}.
 *
 * This is the predicate a rate limiter is scoped by (`withRateLimit({ appliesTo: isRemoteModel })`):
 * provider rate headroom and local MEMORY residency are different scarcities gating different model
 * sets, and giving each layer its own set is what keeps two bounded resources from being acquired in
 * opposite orders. Throws on a bare/unprefixed id, like every other route reader.
 */
export function isRemoteModel(modelId: string): boolean {
  const { route } = parseModelRoute(modelId);
  return route === "anthropic" || route === "openai" || route === "openrouter";
}

/** True iff the model runs on YOUR hardware — an OpenAI-compatible server on this box (`local`) or
 *  weights loaded into this very process (`embedded`). The complement of {@link isRemoteModel}. */
export function isLocalModel(modelId: string): boolean {
  return !isRemoteModel(modelId);
}

/**
 * True iff the model's weights are loaded into THIS process — the only models whose memory residency
 * we actually control, and therefore the correct scope for a residency manager.
 *
 * Distinct from {@link isLocalModel} for a reason worth stating: a `local/` server also runs on your
 * hardware, but its memory is another process's business. It swaps on its own schedule (Ollama's
 * `keep_alive`) or pins one model for its lifetime (vLLM), so a manager that queued `local/` calls
 * behind residency decisions would be serializing work on the strength of bookkeeping it cannot
 * enforce — the `residencyControl: "none"` case, declared rather than pretended.
 *
 * The total, disjoint pairing is therefore `withModelManager({ appliesTo: isEmbeddedModel })` outside
 * `withRateLimit({ appliesTo: (id) => !isEmbeddedModel(id) })`: local servers have finite capacity and
 * are worth rate-limiting, they just have no residency for us to manage.
 */
export function isEmbeddedModel(modelId: string): boolean {
  return parseModelRoute(modelId).route === "embedded";
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
  /**
   * Per-call residency knobs for a locally-served model, read off the call's resolved config.
   *
   * They belong on the CALL rather than only on router construction because they are what makes two
   * calls against the same weights differ in resource terms, and because hw's `environment` chain is
   * how a subtree declares "run this under a 32k context" — a declaration with nowhere to arrive is a
   * declaration that silently does nothing. The router's own `embedded` config remains the default;
   * these override it per call.
   */
  embedded?: { contextSize?: number; gpuLayers?: number | "auto" | "max"; sequences?: number };
}

/**
 * The provider ROUTER seam: resolves a route-prefixed model id (`anthropic/…`, `openrouter/…`) to the
 * provider model handle this package calls with.
 *
 * It lives HERE now (DESIGN §2). The interface used to sit in core purely so
 * `ExecServices.modelRouter` could be typed, which meant the bottom package named an AI-SDK concept it
 * could not describe (an opaque `ModelHandle`) and llm had to re-narrow it. `@declarative-ai/promptop`
 * augments `ExecServices` with this real type instead — that package already depends on llm, so it can
 * name it (DESIGN §3.2).
 */
export interface ModelRouter {
  resolveModel(modelId: string, opts?: ResolveModelOptions): LanguageModel;
  isAnthropic(modelId: string): boolean;
  /**
   * Release whatever this router STARTED — today, managed local servers (§5).
   *
   * Optional, because a router over remote providers owns nothing: the absent seam is a no-op, as
   * everywhere else in this stack. It only becomes load-bearing once `local.serve` is configured, and
   * then it is the difference between a workflow ending and a `llama-server` holding 20 GB of VRAM
   * until the machine reboots.
   *
   * Idempotent, and it never stops a server this router merely ADOPTED — see `ManagedServer.close`.
   */
  close?(): Promise<void>;
  /**
   * Drop ONE model's weights, freeing its memory — the `unload` a {@link ResidencyManager} is wired to.
   *
   * It is exposed here, rather than the router taking a manager, so construction stays acyclic and the
   * wiring is the caller's:
   *
   * ```ts
   * const router = createModelRouter({ embedded: … });
   * const manager = new ResidencyManager({ unload: (id) => router.unloadModel(id) });
   * ```
   *
   * Takes a full `{route}/{model}` id, like everything else on this seam; a non-embedded id is a no-op,
   * because a remote model has no local memory to free.
   */
  unloadModel?(modelId: string): Promise<void>;
}

export interface ModelRouterOptions {
  anthropicApiKey?: string;
  openAiApiKey?: string;
  /**
   * Where the `openai` route points. Defaults to OpenAI itself.
   *
   * Overridable because the same protocol is what Azure OpenAI and every gateway in front of it
   * speak — pointing this at one is a config change rather than a new route.
   */
  openAiBaseURL?: string;
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
  /**
   * How a `local/…` model reaches its OpenAI-compatible server (Ollama, LM Studio, `llama-server`,
   * vLLM). Absent ⇒ the `local` route is refused, exactly as it was before this was wired.
   *
   * The FUNCTION form is the reason this is not just a `localBaseURL` string: running Ollama on
   * :11434 and LM Studio on :1234 at the same time is ordinary, and the route prefix alone cannot say
   * which one `local/qwen2.5-32b-instruct` means. Returning `undefined` for a model refuses it by
   * name rather than sending it to whichever server happened to be configured.
   *
   * No environment variable is read for this. Endpoints are deployment facts the caller holds, and a
   * library that guessed `localhost:11434` would silently succeed against the wrong process.
   */
  local?: LocalServerConfig | ((providerId: string) => LocalServerConfig | undefined);
  /**
   * How an `embedded/…` model's weights are loaded in-process. Absent ⇒ the route is refused.
   *
   * Needs the optional peer `node-llama-cpp`, imported only when an embedded model is actually
   * resolved — a consumer who never uses this route never loads it and never pays for its ~100 MB of
   * native binaries.
   */
  embedded?: EmbeddedModelResolver;
}

/** One OpenAI-compatible server a `local/…` model can be served by. */
export interface LocalServerConfig {
  /** Base URL INCLUDING the version path — `http://localhost:11434/v1` (Ollama),
   *  `http://localhost:1234/v1` (LM Studio), `http://localhost:8080/v1` (`llama-server`). */
  baseURL: string;
  /** Sent as `Authorization: Bearer …`. Most local servers ignore it; vLLM can be configured to
   *  require one. */
  apiKey?: string;
  /** Extra request headers. */
  headers?: Record<string, string>;
  /** Extra URL query parameters. */
  queryParams?: Record<string, string>;
  /** Provider label carried on the model handle and into diagnostics. Defaults to `"local"`. */
  name?: string;
  /**
   * Whether this server can honor a full `response_format: json_schema`. Default `true`.
   *
   * A CEILING, not a switch. The per-call decision is the schema profile's (`enforce`), which the
   * router receives as {@link ResolveModelOptions.strictStructuredOutput}; this caps it for a server
   * that cannot do constrained decoding no matter what the profile concluded. `llama-server` and vLLM
   * can; a bare completion shim in front of llama.cpp cannot.
   *
   * What it does NOT do is suppress `response_format` entirely — capped calls still send
   * `{type:"json_object"}`, because the provider treats this flag as json_schema-vs-json_object.
   * Sending NOTHING is the text tier's job, and that is driven by the profile
   * (`supportsStructuredOutput: false` ⇒ the schema goes in the prompt and Ajv is the only gate),
   * not from here.
   *
   * It lives on the server rather than a catalog row because it is a property of the PROCESS in front
   * of the weights: the same GGUF gets grammar-constrained decoding from one server and nothing from
   * another.
   */
  supportsStructuredOutputs?: boolean;
  /**
   * Ask for usage on streamed responses (`stream_options.include_usage`). Default `true`.
   *
   * Without it most OpenAI-compatible servers stream a final chunk carrying no `usage`, and every
   * call reports zero tokens. Local inference is free, so this is not about money — it is about
   * `LlmMetrics` telling the truth, and about a residency planner being able to see how much context
   * a call actually consumed.
   */
  includeUsage?: boolean;
  /**
   * How to START this server if it is not already running — the ATTACHED→MANAGED upgrade.
   *
   * Absent ⇒ attached: the server is expected to exist and nothing is spawned. Present ⇒ the router
   * probes `baseURL` on the first call and starts the process only if nothing answers, so a developer
   * who already has one running keeps theirs. Whatever the router started, and only that, is stopped by
   * {@link ModelRouter.close}.
   */
  serve?: ManagedServerSpec;
  /** Transport seam, for tests and for a caller that needs its own agent/proxy. A managed server wraps
   *  this so requests wait for readiness; an attached one passes it through untouched. */
  fetch?: FetchFunction;
}

/**
 * Build a router. Provider clients are created lazily on first use so that a process
 * that only ever calls Anthropic never needs an OpenRouter key (and vice versa), and
 * so importing this module never requires any key.
 */
export function createModelRouter(options: ModelRouterOptions = {}): ModelRouter {
  if (!options.skipDispatcher) installLongTimeoutDispatcher();

  let anthropic: ReturnType<typeof createAnthropic> | undefined;
  /** One client per strict-flag value — the flag is per CALL, so a single memo would freeze it. */
  const openaiClients = new Map<string, ReturnType<typeof createOpenAICompatible>>();
  let openrouter: ReturnType<typeof createOpenRouter> | undefined;
  /** Built on the first `embedded/` resolve, so a router that never sees one never touches the peer. */
  let embedded: EmbeddedModelStore | undefined;
  /**
   * One client per distinct SERVER, not per model — a resolver handing back the same `{ baseURL }` for
   * twenty models must not build twenty clients, and two models on genuinely different servers must
   * not share one. Keyed on every setting that changes what goes on the wire, so a config differing
   * only in `supportsStructuredOutputs` gets its own client instead of silently reusing the first
   * model's answer for the rest.
   */
  const localClients = new Map<string, ReturnType<typeof createOpenAICompatible>>();
  /**
   * Supervisors, keyed by `baseURL` rather than by the full client key: the PROCESS is per endpoint,
   * while clients also vary by the per-call structured-output flag. Keying these the same way would
   * start a second `llama-server` the first time a strict call followed an advisory one.
   */
  const supervisors = new Map<string, ManagedServer>();
  const supervisorFor = (server: LocalServerConfig): ManagedServer | undefined => {
    if (server.serve === undefined) return undefined;
    let supervisor = supervisors.get(server.baseURL);
    if (supervisor === undefined) {
      supervisor = new ManagedServer(server.serve, server.baseURL, server.fetch);
      supervisors.set(server.baseURL, supervisor);
    }
    return supervisor;
  };

  const localProvider = (server: LocalServerConfig, strict: boolean): ReturnType<typeof createOpenAICompatible> => {
    const name = server.name ?? "local";
    // The provider reads this as json_schema-VS-json_object, so it must follow the per-call `enforce`
    // decision rather than a static server setting: a profile that concluded "advisory" wants
    // `{type:"json_object"}`, and hard-coding `true` here sent every local call a strict json_schema —
    // the one shape an Ollama or LM Studio endpoint is most likely to reject. The server flag caps it.
    const structured = strict && (server.supportsStructuredOutputs ?? true);
    const usage = server.includeUsage ?? true;
    const key = JSON.stringify([name, server.baseURL, server.apiKey, server.headers, server.queryParams, structured, usage]);
    let client = localClients.get(key);
    if (client === undefined) {
      // A managed endpoint boots on the first REQUEST, not here: `resolveModel` is synchronous and
      // cannot await a process start, and gating the transport means a router configured with a server
      // it never actually calls starts nothing at all.
      const supervisor = supervisorFor(server);
      const transport = supervisor ? readyGatedFetch(supervisor, server.fetch) : server.fetch;
      client = createOpenAICompatible({
        name,
        baseURL: server.baseURL,
        ...(server.apiKey !== undefined ? { apiKey: server.apiKey } : {}),
        ...(server.headers !== undefined ? { headers: server.headers } : {}),
        ...(server.queryParams !== undefined ? { queryParams: server.queryParams } : {}),
        ...(transport !== undefined ? { fetch: transport } : {}),
        supportsStructuredOutputs: structured,
        includeUsage: usage,
      });
      localClients.set(key, client);
    }
    return client;
  };

  const resolveModel = (modelId: string, opts: ResolveModelOptions = {}): LanguageModel => {
    const { route, providerId } = parseModelRoute(modelId);
    log.debug("resolve model", { modelId, route, providerId, strict: opts.strictStructuredOutput });
    if (route === "anthropic") {
      anthropic ??= createAnthropic({
        apiKey: options.anthropicApiKey ?? process.env.ANTHROPIC_API_KEY,
      });
      return anthropic(providerId);
    }
    if (route === "openai") {
      // Built on `@ai-sdk/openai-compatible` rather than a dedicated OpenAI package, because OpenAI
      // IS the shape that package implements — it is the reference the "compatible" name refers to.
      // A second SDK dependency would buy provider-specific extras this project does not use, and
      // the local route already proves this client against a server speaking the same protocol.
      //
      // Keyed on the strict flag and cached per value, exactly as the local route is, because the
      // flag is PER CALL: `supportsStructuredOutputs` makes the client send
      // `response_format.json_schema.strict`, and a schema that could not be strictified reaches here
      // on the ADVISORY tier. Pinning it on would 400 every one of those — which is the tier's whole
      // purpose — and one memoized client would freeze whichever answer the first call happened to
      // want. `openRouterStrictStructuredOutputs` documents the same hazard and defaults it off.
      const strict = opts.strictStructuredOutput ?? false;
      const key = `openai:${strict}`;
      let client = openaiClients.get(key);
      if (client === undefined) {
        client = createOpenAICompatible({
          name: "openai",
          baseURL: options.openAiBaseURL ?? "https://api.openai.com/v1",
          apiKey: options.openAiApiKey ?? process.env.OPENAI_API_KEY,
          // OpenAI returns usage on the stream only when asked, and cost is computed from it.
          includeUsage: true,
          supportsStructuredOutputs: strict,
        });
        openaiClients.set(key, client);
      }
      return client(providerId);
    }
    if (route === "local") {
      const server = typeof options.local === "function" ? options.local(providerId) : options.local;
      if (!server) {
        throw new Error(
          `model "${modelId}" names the "local" route, but no OpenAI-compatible server is configured for it ` +
            `(set ModelRouterOptions.local to a { baseURL } or a resolver returning one)`,
        );
      }
      return localProvider(server, opts.strictStructuredOutput ?? false)(providerId);
    }
    if (route === "embedded") {
      // Refused explicitly rather than falling through to the OpenRouter branch below, which would send
      // a local model id to a remote provider — with an API key attached — and report the resulting 404
      // as if the model were the problem.
      if (options.embedded === undefined) {
        throw new Error(
          `model "${modelId}" names the "embedded" route, but no in-process weights are configured for it ` +
            `(set ModelRouterOptions.embedded to a { modelPath } or a resolver returning one)`,
        );
      }
      embedded ??= new EmbeddedModelStore(options.embedded);
      // Refuse an UNRESOLVED id here, synchronously, rather than letting it surface as a rejected load
      // on the first token: a typo'd model name is a wiring mistake and belongs at the same place every
      // other wiring mistake on this route lands.
      if (embedded.configFor(providerId) === undefined) {
        throw new Error(`model "${modelId}" names the "embedded" route, but no weights are configured for "${providerId}"`);
      }
      // The handle returns immediately; the GGUF is mapped on the first `doStream`. Loading here would
      // block a synchronous function on seconds of I/O.
      return embeddedLanguageModel(providerId, embedded, "embedded", opts.embedded);
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
    unloadModel: async (modelId: string): Promise<void> => {
      const { route, providerId } = parseModelRoute(modelId);
      if (route !== "embedded") return; // nothing local to free
      await embedded?.unload(providerId);
    },
    close: async (): Promise<void> => {
      // Every supervisor, in parallel, and the map is cleared FIRST so a close racing a resolve cannot
      // hand out a client gated on a supervisor that is already shutting down. Each `close` is itself a
      // no-op for a server we only adopted.
      const running = [...supervisors.values()];
      const models = embedded;
      supervisors.clear();
      localClients.clear();
      embedded = undefined;
      // Embedded weights matter MORE here than a managed server does: an undisposed GGUF holds its VRAM
      // for the life of the process, with no port to notice and no child to reap.
      await Promise.all([...running.map((s) => s.close()), ...(models ? [models.close()] : [])]);
    },
  };
}
