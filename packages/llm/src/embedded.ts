/**
 * EMBEDDED models (§5) — `node-llama-cpp` weights loaded into this process, behind a `LanguageModelV3`.
 *
 * Everything above `resolveModel` is unchanged by this route: the same `generateStructured`, the same
 * wrappers, the same metrics. What differs is that there is no wire. Three consequences shape the code:
 *
 *  - **`node-llama-cpp` is an OPTIONAL peer.** It ships ~100 MB of prebuilt native binaries, and a
 *    consumer who only ever calls Anthropic must not pay for that. It is reached by dynamic `import()`
 *    on the first `embedded/` resolve, and its absence is a clear install instruction rather than a
 *    module-not-found stack.
 *  - **Loading is SLOW and must not happen at resolve time.** `resolveModel` is synchronous; a GGUF
 *    takes seconds to map and can take minutes to fetch. `doStream` returns `PromiseLike`, so the load
 *    happens on the first call and is memoized — the lazy-handle pattern.
 *  - **Concurrency is bought with memory.** A context fixes its `sequences` count at creation and each
 *    sequence carries its own KV cache (measured: a 0.5B model was 374 MB of weights and 491 MB for four
 *    4096-token sequences). So a sequence is a POOLED resource here, acquired per call and returned on
 *    finish — not something a call can conjure.
 */
import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3FinishReason,
  LanguageModelV3Prompt,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
} from "@ai-sdk/provider";
import { createLogger } from "@declarative-ai/log";
import type { Placement, PlacementProbe, PlacementTier } from "./residency.js";
import type { ModelInfoInterface, WeightsLocation } from "./model-catalog.js";

const log = createLogger("engine.providers.embedded");

/** How ONE embedded model is loaded. The caller supplies the file: this library never picks a path or a
 *  cache directory (see the weights store for downloads). */
export interface EmbeddedModelConfig {
  /** Filesystem path to the GGUF. For a split model, the FIRST part — llama.cpp finds the rest. */
  modelPath: string;
  /** Context window to allocate. Defaults to the model's trained size, clamped by what fits. */
  contextSize?: number;
  /** Layers to offload. `"auto"` (the default) fits as many as the CURRENT free VRAM allows, which is
   *  the only correct answer on a GPU this process does not own. */
  gpuLayers?: number | "auto" | "max";
  /** How many generations this model may serve at once. Each costs its own KV cache. Default 1. */
  sequences?: number;
}

/** Resolve a provider-native model id (the part after `embedded/`) to how it should be loaded. */
export type EmbeddedModelResolver = EmbeddedModelConfig | ((providerId: string) => EmbeddedModelConfig | undefined);

/** The per-CALL residency knobs, layered over the router's config for this model. Never the model path:
 *  which weights a model id means is deployment, not something a call gets to redirect. */
export type EmbeddedOverride = Omit<EmbeddedModelConfig, "modelPath">;

/**
 * Layer a call's residency knobs over the router's configuration for that model.
 *
 * Absent keys are DROPPED rather than spread: every call carries all three fields and most are
 * `undefined`, so a plain `{...base, ...override}` would blank a configured `contextSize` with nothing —
 * the opposite of "this call did not ask". Exported because that asymmetry is the whole content of the
 * function and is worth testing directly.
 */
export function mergeEmbeddedConfig(base: EmbeddedModelConfig, override: EmbeddedOverride | undefined): EmbeddedModelConfig {
  if (override === undefined) return base;
  return { ...base, ...(Object.fromEntries(Object.entries(override).filter(([, v]) => v !== undefined)) as EmbeddedOverride) };
}

/** The slice of `node-llama-cpp` this adapter uses, so the dynamic import has a type without the
 *  package being a hard dependency. Structural, deliberately minimal. */
interface LlamaModule {
  getLlama(options?: Record<string, unknown>): Promise<LlamaHandle>;
  LlamaChatSession: new (options: { contextSequence: unknown; systemPrompt?: string }) => LlamaSession;
}
interface LlamaHandle {
  loadModel(options: Record<string, unknown>): Promise<LlamaLoadedModel>;
  createGrammarForJsonSchema(schema: Record<string, unknown>): Promise<LlamaGrammar>;
  getVramState(): Promise<{ total: number; used: number; free: number; unifiedSize: number }>;
  dispose?(): Promise<void>;
}
interface LlamaLoadedModel {
  createContext(options: Record<string, unknown>): Promise<LlamaCtx>;
  readonly gpuLayers: number;
  readonly trainContextSize: number;
  dispose(): Promise<void>;
}
interface LlamaCtx {
  getSequence(): LlamaSequence;
  readonly sequencesLeft: number;
  dispose(): Promise<void>;
}
interface LlamaSequence {
  readonly tokenMeter: { getState(): TokenMeterState; diff(s: TokenMeterState): TokenMeterState };
  dispose?(): void;
}
interface TokenMeterState {
  usedInputTokens: number;
  usedOutputTokens: number;
}
interface LlamaGrammar {
  parse(text: string): unknown;
}
interface LlamaSession {
  prompt(text: string, options: Record<string, unknown>): Promise<string>;
}

/**
 * The peer's name, assembled at runtime so no bundler can read it.
 *
 * `import("node-llama-cpp")` with a LITERAL specifier is statically analyzable, and bundlers follow it
 * even though nothing awaits it at module scope — so bundling any consumer of this package walked into
 * the native peer and failed on things that consumer had nothing to do with: top-level `await` in the
 * platform binary shims (illegal in a `cjs` output), unresolvable `@node-llama-cpp/{mac,linux}-*`
 * packages that were never installed on the building machine, and a `.node` binary with no configured
 * loader. Sixteen errors for a route the consumer had not used.
 *
 * A runtime-built specifier is invisible to that analysis while remaining a REAL module-scope `import()`
 * — which is the part that matters and that a `new Function("return import(s)")` trick gets wrong:
 * that form throws "a dynamic import callback was not specified" in any host that does not install one
 * (vitest, `vm` contexts), trading a build-time break for a runtime one. Verified against esbuild
 * `--bundle --platform=node --format=cjs`: zero errors, the peer absent from the output, and the call
 * emitted verbatim as `import(PEER)` rather than rewritten to `require()` — which also matters, since
 * `node-llama-cpp` is ESM-only with top-level await and could not be `require`d.
 *
 * The magic comments cover the bundlers that DO fold or scan variable specifiers.
 */
const PEER = ["node", "llama", "cpp"].join("-");

/** Load `node-llama-cpp`, or explain how to get it. */
let modulePromise: Promise<LlamaModule> | undefined;
export function loadLlamaModule(): Promise<LlamaModule> {
  modulePromise ??= import(/* webpackIgnore: true */ /* @vite-ignore */ PEER).then(
    (m) => m as unknown as LlamaModule,
    (cause: unknown) => {
      modulePromise = undefined; // a transient failure must not poison the process
      throw new Error(
        "the `embedded` route needs the optional peer dependency `node-llama-cpp` — install it with " +
          "`npm i node-llama-cpp` (it ships prebuilt binaries for win/mac/linux, including CUDA and Vulkan). " +
          `Original error: ${cause instanceof Error ? cause.message : String(cause)}`,
      );
    },
  );
  return modulePromise;
}

/** Flatten a V3 prompt into the system string + the single user turn `LlamaChatSession` takes.
 *
 *  Prior turns are rendered into the user turn rather than replayed through the session's own history,
 *  because the caller owns the transcript here: `withSession` materializes it and hands us the whole
 *  conversation every call, so a session that ALSO kept state would double it. */
export function flattenPrompt(prompt: LanguageModelV3Prompt): { system: string | undefined; user: string } {
  const systems: string[] = [];
  const turns: string[] = [];
  for (const message of prompt) {
    if (message.role === "system") {
      systems.push(message.content);
      continue;
    }
    const text = Array.isArray(message.content)
      ? message.content
          .map((part) => ("text" in part && typeof part.text === "string" ? part.text : ""))
          .filter((t) => t.length > 0)
          .join("\n")
      : "";
    if (text.length === 0) continue;
    turns.push(message.role === "user" ? text : `${message.role}: ${text}`);
  }
  return { system: systems.length > 0 ? systems.join("\n\n") : undefined, user: turns.join("\n\n") };
}

/** llama.cpp's stop reason → the SDK's unified vocabulary. */
export function mapFinishReason(stopReason: string | undefined): LanguageModelV3FinishReason {
  switch (stopReason) {
    case "maxTokens":
      return { unified: "length", raw: stopReason };
    case "eogToken":
    case "stopGenerationTrigger":
    case "customStopTrigger":
      return { unified: "stop", raw: stopReason };
    case "abort":
      return { unified: "other", raw: stopReason };
    default:
      return { unified: "stop", raw: stopReason };
  }
}

const usageOf = (d: TokenMeterState): LanguageModelV3Usage => ({
  inputTokens: { total: d.usedInputTokens, noCache: d.usedInputTokens, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: d.usedOutputTokens, text: d.usedOutputTokens, reasoning: undefined },
});

/**
 * One loaded model plus its context, and a SEQUENCE POOL over it.
 *
 * The pool is the part that is easy to get wrong. `context.getSequence()` throws "No sequences left"
 * once the declared count is exhausted, so a call arriving while every sequence is busy must WAIT
 * rather than fail — that is ordinary backpressure, not an error. Sequences are returned on finish,
 * including on abort and on failure, or the pool drains permanently and the model wedges after N calls.
 */
class LoadedModel {
  private readonly free: LlamaSequence[] = [];
  private readonly waiting: ((seq: LlamaSequence) => void)[] = [];

  private constructor(
    readonly llama: LlamaHandle,
    private readonly model: LlamaLoadedModel,
    private readonly context: LlamaCtx,
    readonly gpuLayers: number,
  ) {}

  static async open(config: EmbeddedModelConfig): Promise<LoadedModel> {
    const { getLlama } = await loadLlamaModule();
    const llama = await getLlama();
    const before = await llama.getVramState();
    const model = await llama.loadModel({
      modelPath: config.modelPath,
      gpuLayers: config.gpuLayers ?? "auto",
    });
    const sequences = Math.max(1, config.sequences ?? 1);
    const context = await model.createContext({
      ...(config.contextSize !== undefined ? { contextSize: config.contextSize } : {}),
      sequences,
    });
    const after = await llama.getVramState();
    log.debug("loaded embedded model", {
      modelPath: config.modelPath,
      gpuLayers: model.gpuLayers,
      sequences,
      vramUsedMb: Math.round((after.used - before.used) / 1024 / 1024),
      vramFreeMb: Math.round(after.free / 1024 / 1024),
    });
    const loaded = new LoadedModel(llama, model, context, model.gpuLayers);
    for (let i = 0; i < sequences; i++) loaded.free.push(context.getSequence());
    return loaded;
  }

  /** Take a sequence, waiting if they are all busy. */
  acquire(): Promise<LlamaSequence> {
    const ready = this.free.pop();
    if (ready !== undefined) return Promise.resolve(ready);
    return new Promise((resolve) => this.waiting.push(resolve));
  }

  /** Hand a sequence back — to the longest-waiting caller if there is one, else to the pool. */
  release(seq: LlamaSequence): void {
    const next = this.waiting.shift();
    if (next) next(seq);
    else this.free.push(seq);
  }

  async dispose(): Promise<void> {
    await this.context.dispose();
    await this.model.dispose();
  }
}

/**
 * The registry of loaded models for one router, and the thing its `close()` disposes.
 *
 * Loads are memoized by model id and shared: two ops naming the same weights get one copy in VRAM, which
 * is the entire reason a registry exists rather than a load per call.
 */
export class EmbeddedModelStore {
  private readonly loaded = new Map<string, Promise<LoadedModel>>();

  constructor(private readonly resolve: EmbeddedModelResolver) {}

  configFor(providerId: string): EmbeddedModelConfig | undefined {
    return typeof this.resolve === "function" ? this.resolve(providerId) : this.resolve;
  }

  /**
   * Load (or reuse) a model. `override` is the CALL's residency knobs, layered over the router's config.
   *
   * The memo is keyed on the model id alone, so the FIRST call's effective config wins for as long as
   * the weights stay resident — a later call asking for a bigger context reuses the loaded one rather
   * than silently loading a second copy beside it. Whether it should instead evict and reload is a
   * residency decision, and the residency manager is the thing that owns memory; making that choice
   * here would be this layer quietly allocating gigabytes behind the manager's back.
   */
  open(providerId: string, override?: EmbeddedOverride): Promise<LoadedModel> {
    let entry = this.loaded.get(providerId);
    if (entry === undefined) {
      const base = this.configFor(providerId);
      const config = base === undefined ? undefined : mergeEmbeddedConfig(base, override);
      if (config === undefined) {
        return Promise.reject(
          new Error(`no embedded model is configured for "${providerId}" (set ModelRouterOptions.embedded to a { modelPath } or a resolver)`),
        );
      }
      entry = LoadedModel.open(config).catch((err: unknown) => {
        // A failed load must not be cached: the usual causes (a missing file, a transient VRAM
        // shortage because something else held the GPU) are all things a later call could survive.
        this.loaded.delete(providerId);
        throw err;
      });
      this.loaded.set(providerId, entry);
    }
    return entry;
  }

  /**
   * Drop ONE model's weights, freeing its VRAM. Idempotent, and a no-op for a model that is not loaded.
   *
   * This is what a {@link ResidencyManager}'s `unload` is wired to — without it, `maxResident` is
   * bookkeeping that changes nothing, and the manager would happily "evict" a model whose gigabytes
   * stayed resident, then admit another on top of it.
   *
   * The manager guarantees zero leases before it calls this, which is the invariant that makes disposal
   * safe: a sequence in mid-generation would otherwise be freed underneath itself.
   */
  async unload(providerId: string): Promise<void> {
    const entry = this.loaded.get(providerId);
    if (entry === undefined) return;
    this.loaded.delete(providerId);
    // Awaited rather than abandoned: the next admission is sized against the memory this releases, so
    // returning before the free has happened is how two models end up resident at once.
    await entry.then(
      (model) => model.dispose(),
      () => undefined, // a load that failed has nothing to dispose
    );
  }

  async close(): Promise<void> {
    const entries = [...this.loaded.values()];
    this.loaded.clear();
    // `allSettled`: one model failing to dispose must not strand the others' VRAM.
    await Promise.allSettled(entries.map((e) => e.then((m) => m.dispose())));
  }
}

/**
 * The default {@link PlacementProbe}: predict where a model would land RIGHT NOW, using llama.cpp's own
 * resource estimator against this machine's live memory state.
 *
 * Two things make this trustworthy in a way hand-rolled arithmetic would not be. It reads the GGUF's
 * real tensor layout rather than guessing from file size, and it accounts for the KV cache at the
 * context and concurrency actually configured — which is what decides the tier at all (the same 32B
 * model sits entirely in VRAM at 4k context and spills at 32k). It also runs WITHOUT instantiating a
 * llama.cpp backend, so predicting costs no GPU.
 *
 * Returns `undefined` for anything it cannot measure, which admits the call unjudged — a probe that
 * guessed would be worse than one that abstains.
 */
export function embeddedPlacementProbe(store: EmbeddedModelStore, providerIdOf: (modelId: string) => string | undefined): PlacementProbe {
  return {
    async predict(modelId) {
      const providerId = providerIdOf(modelId);
      if (providerId === undefined) return undefined;
      const config = store.configFor(providerId);
      if (config === undefined) return undefined;
      let insights: GgufInsightsLike;
      let vram: { total: number; free: number };
      let ram: { total: number; free: number };
      try {
        const mod = (await loadLlamaModule()) as unknown as {
          getLlama(): Promise<LlamaHandle & { getRamState(): Promise<{ total: number; free: number }> }>;
          readGgufFileInfo(path: string): Promise<unknown>;
          GgufInsights: { from(info: unknown, llama?: unknown): Promise<GgufInsightsLike> };
        };
        const llama = await mod.getLlama();
        [vram, ram] = await Promise.all([llama.getVramState(), llama.getRamState()]);
        insights = await mod.GgufInsights.from(await mod.readGgufFileInfo(config.modelPath), llama);
      } catch {
        return undefined; // not measurable (peer absent, unreadable file) — admit unjudged
      }
      const resolved = await insights.configurationResolver.resolveAndScoreConfig(
        config.contextSize !== undefined ? { targetContextSize: config.contextSize } : {},
      );
      const v = resolved.resolvedValues;
      return classifyPlacement(v, insights.totalLayers, { vramFree: vram.free, ramFree: ram.free });
    },
  };
}

/**
 * GGML file-type enum → the quantization name people actually use.
 *
 * The GGUF header stores `general.file_type` as a number; every filename convention, model card and
 * catalog row says `Q4_K_M`. Reading the header rather than parsing the filename is what makes the row
 * trustworthy — a renamed file still reports what it really is.
 */
const GGML_FILE_TYPES: Record<number, string> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
  10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
  16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
  22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
  28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
};

/**
 * Build a CATALOG ROW for a GGUF by reading its header — the metadata generator (§5).
 *
 * Everything the runtime knows about a model is supposed to come from the catalog, and for a local
 * model nobody publishes that data: there is no `/api/v1/models` for a file on your disk. This reads it
 * from the weights themselves, so a row is generated rather than hand-written and cannot drift from
 * what the file actually is.
 *
 * `source` may be an `hf:<org>/<repo>/<file>` reference, and the read is a RANGE request against the
 * header — an 18 GB model's row can be produced in about a second without downloading it. That is what
 * makes it practical to catalog models before deciding to fetch them.
 *
 * The rates are ZERO, and that is a claim rather than a placeholder: local inference costs no money, so
 * a row saying `0` lets `costSource` report `"table"`. Without a row the same call reports `"unknown"`,
 * and this codebase keeps "free" and "we could not price it" apart deliberately.
 */
export async function catalogRowForGguf(options: {
  /** Provider-native id; the catalog key becomes `embedded/<model>`. Include the quantization, since
   *  each quant is its own row (different footprint, different quality, different context headroom). */
  model: string;
  /** Local path or `hf:` reference to read the header from. */
  source: string;
  /** Where the weights can be fetched from later, for the weights store. */
  downloads?: readonly WeightsLocation[];
}): Promise<ModelInfoInterface> {
  const mod = (await loadLlamaModule()) as unknown as {
    readGgufFileInfo(path: string): Promise<GgufFileInfoLike>;
    GgufInsights: { from(info: unknown): Promise<{ modelSize: number; totalLayers: number; trainContextSize?: number }> };
  };
  const info = await mod.readGgufFileInfo(options.source);
  // No `llama` handle passed: the insights layer explicitly does not need a llama.cpp backend for this,
  // so cataloguing a model never touches the GPU.
  const insights = await mod.GgufInsights.from(info);
  const general = info.metadata?.general ?? {};
  const fileType = typeof general.file_type === "number" ? GGML_FILE_TYPES[general.file_type] : undefined;
  const contextLength = insights.trainContextSize ?? info.architectureMetadata?.context_length;
  return {
    route: "embedded",
    model: options.model,
    // Local inference is free — a real claim, not a missing price. See the note above.
    inputPerMillion: 0,
    outputPerMillion: 0,
    openWeights: true,
    ...(fileType !== undefined ? { quantization: fileType } : {}),
    weightsMb: Math.round(insights.modelSize / 1024 / 1024),
    ...(contextLength !== undefined ? { contextLength } : {}),
    ...(typeof general.name === "string" ? { label: general.name } : {}),
    ...(options.downloads !== undefined ? { downloads: options.downloads } : {}),
    modalities: { input: ["text"], output: ["text"] },
    source: "gguf",
  };
}

/** The slice of `readGgufFileInfo`'s result this reads. */
interface GgufFileInfoLike {
  metadata?: { general?: { file_type?: number; name?: string } };
  architectureMetadata?: { context_length?: number };
}

/** The resolver's answer for one model+context, and this machine's free memory — everything the tier
 *  decision depends on. */
export interface ResolvedPlacement {
  gpuLayers: number;
  contextSize: number;
  totalVramUsage: number;
  totalRamUsage: number;
}

/**
 * Classify a predicted allocation into the three placement tiers.
 *
 * Pure, and exported, because this is the whole judgement — everything around it is I/O. Keeping it
 * separable is what lets the same arithmetic be checked against invented hardware (a 12 GB card, a
 * machine with no GPU) rather than only against whatever box the tests happen to run on.
 *
 * The ORDER of the tests is the substance:
 *
 *  - **Swap first.** A working set that does not fit free system RAM is the catastrophic case, and it
 *    is reached by exactly the same partial-offload path as an ordinary spill — so testing "did some
 *    layers come off the GPU" first would label a model that is about to thrash the pagefile as merely
 *    degraded, which is the one mistake the three-tier split exists to prevent.
 *  - **Then RAM.** Any spill off the GPU: layers left on the CPU, or a VRAM figure that does not fit
 *    what is free. Both conditions matter — a full-layer offload can still exceed free VRAM when
 *    something else took the card between predicting and loading.
 *  - **VRAM last**, as the case where neither of the above is true.
 */
export function classifyPlacement(
  resolved: ResolvedPlacement,
  totalLayers: number,
  free: { vramFree: number; ramFree: number },
): Placement {
  const tier: PlacementTier =
    resolved.totalRamUsage > free.ramFree
      ? "swap"
      : resolved.gpuLayers < totalLayers || resolved.totalVramUsage > free.vramFree
        ? "ram"
        : "vram";
  return {
    tier,
    gpuLayers: resolved.gpuLayers,
    totalLayers,
    contextSize: resolved.contextSize,
    vramBytes: resolved.totalVramUsage,
    ramBytes: resolved.totalRamUsage,
  };
}

/** The slice of `GgufInsights` the probe uses. */
interface GgufInsightsLike {
  readonly totalLayers: number;
  readonly configurationResolver: {
    resolveAndScoreConfig(options?: { targetContextSize?: number }): Promise<{
      resolvedValues: { gpuLayers: number; contextSize: number; totalRamUsage: number; totalVramUsage: number };
    }>;
  };
}

/** Build the `LanguageModelV3` for one embedded model id. The handle is returned SYNCHRONOUSLY; the
 *  weights load on the first `doStream`. */
export function embeddedLanguageModel(
  providerId: string,
  store: EmbeddedModelStore,
  providerName = "embedded",
  override?: EmbeddedOverride,
): LanguageModelV3 {
  const run = async (options: LanguageModelV3CallOptions): Promise<ReadableStream<LanguageModelV3StreamPart>> => {
    const model = await store.open(providerId, override);
    const { LlamaChatSession } = await loadLlamaModule();
    const { system, user } = flattenPrompt(options.prompt);

    // A json response format becomes a GBNF GRAMMAR — the decoder cannot emit a token the schema
    // forbids. That is why this route's schema profile claims strict enforcement honestly.
    const format = options.responseFormat;
    const grammar =
      format?.type === "json" && format.schema !== undefined
        ? await model.llama.createGrammarForJsonSchema(format.schema as Record<string, unknown>)
        : undefined;

    const seq = await model.acquire();
    const meter = seq.tokenMeter;
    const before = meter.getState();
    const session = new LlamaChatSession({ contextSequence: seq, ...(system !== undefined ? { systemPrompt: system } : {}) });

    return new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        const id = "0";
        controller.enqueue({ type: "stream-start", warnings: [] });
        controller.enqueue({ type: "text-start", id });
        let stopReason: string | undefined;
        try {
          await session.prompt(user, {
            ...(grammar !== undefined ? { grammar } : {}),
            ...(options.maxOutputTokens !== undefined ? { maxTokens: options.maxOutputTokens } : {}),
            ...(options.temperature !== undefined ? { temperature: options.temperature } : {}),
            ...(options.topP !== undefined ? { topP: options.topP } : {}),
            ...(options.topK !== undefined ? { topK: options.topK } : {}),
            ...(options.seed !== undefined ? { seed: options.seed } : {}),
            ...(options.stopSequences !== undefined ? { customStopTriggers: options.stopSequences } : {}),
            ...(options.abortSignal !== undefined ? { signal: options.abortSignal, stopOnAbortSignal: true } : {}),
            onTextChunk: (text: string) => controller.enqueue({ type: "text-delta", id, delta: text }),
            onResponseChunk: (c: { stopReason?: string }) => {
              if (c?.stopReason != null) stopReason = c.stopReason;
            },
          });
        } catch (err) {
          controller.enqueue({ type: "text-end", id });
          // An abort is reported through the stream rather than thrown: the caller already knows it
          // canceled, and the tokens generated up to that point are still real usage.
          const aborted = options.abortSignal?.aborted === true;
          if (!aborted) controller.enqueue({ type: "error", error: err });
          controller.enqueue({
            type: "finish",
            usage: usageOf(meter.diff(before)),
            finishReason: { unified: aborted ? "other" : "error", raw: aborted ? "abort" : "error" },
          });
          controller.close();
          return;
        } finally {
          // ALWAYS — on success, abort and failure alike. A sequence not returned is one the pool never
          // sees again, and after `sequences` such calls the model stops serving entirely.
          model.release(seq);
        }
        controller.enqueue({ type: "text-end", id });
        controller.enqueue({ type: "finish", usage: usageOf(meter.diff(before)), finishReason: mapFinishReason(stopReason) });
        controller.close();
      },
    });
  };

  return {
    specificationVersion: "v3",
    provider: providerName,
    modelId: providerId,
    supportedUrls: {}, // no wire, so no URL the model could fetch itself
    doStream: async (options) => ({ stream: await run(options) }),
    doGenerate: async (options) => {
      // The whole stack streams; `doGenerate` exists to satisfy the interface and is implemented by
      // draining our own stream rather than by a second generation path that could drift from it.
      const stream = await run(options);
      let text = "";
      let usage: LanguageModelV3Usage = { inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined }, outputTokens: { total: undefined, text: undefined, reasoning: undefined } };
      let finishReason: LanguageModelV3FinishReason = { unified: "other", raw: undefined };
      const reader = stream.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value.type === "text-delta") text += value.delta;
        else if (value.type === "finish") {
          usage = value.usage;
          finishReason = value.finishReason;
        } else if (value.type === "error") throw value.error;
      }
      return { content: text.length > 0 ? [{ type: "text" as const, text }] : [], finishReason, usage, warnings: [] };
    },
  };
}
