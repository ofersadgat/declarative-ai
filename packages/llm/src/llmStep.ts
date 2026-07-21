import type { BlobStore, FileInput, LlmCallConfig, OutputValidator, ProviderOptions, ToolDefinition } from "@declarative-ai/core";
import { jsonSchema, stepCountIs, type FilePart, type ModelMessage, type StopCondition, type SystemModelMessage, type TextPart, type ToolCallOptions, type ToolChoice, type ToolSet } from "ai";
import { generateStructured, promptAsMessages, promptText, type CallOutcome, type CallPromptInput, type JsonSchema } from "./generate";
import { ModelInfo } from "./model-catalog";
import { adaptReasoning } from "./reasoning";
import { providerNativeId, type ModelRouter } from "./router";
import { adaptSchemaCached, profileForModelId } from "./schema";

/** A runtime tool implementation — the `execute` for a declared FUNCTION tool, looked up by tool name.
 *  Injected via {@link CallDeps} (not serialized), mirroring how the model handle + validator are supplied. */
export type ToolExecutor = (input: unknown, options: ToolCallOptions) => unknown | Promise<unknown>;

/** Default cap on tool-loop steps when executable tools are present and the config names no `maxSteps`. */
const DEFAULT_MAX_STEPS = 8;

/** The prompt inputs + time budget layered onto a config to make a concrete call definition. The prompt
 *  is the full SDK-shaped {@link CallPromptInput} (string system/prompt, or structured/multimodal
 *  messages), so the definition is as expressive as the underlying call. */
type CallPrompt = CallPromptInput & {
  /** Per-call wall-clock budget (ms). Optional at the definition layer — the executor fills it from
   *  `spec.limits.timeoutMs` or the default before the call (where {@link StructuredCallParams} requires it). */
  timeoutMs?: number;
};

/**
 * The serializable definition of ONE LLM call, MINUS the output schema — and exactly `spec.definition`
 * for kind "llm-call". It IS core's canonical {@link LlmCallConfig} (the sampling-XOR-reasoning union:
 * `model` string + serializable decoding knobs, with reasoning OR sampling but never both) with the prompt
 * texts and a time budget added — the single serializable "how to call an LLM" descriptor, not a parallel
 * re-listing. It stays a UNION (not a flattened bag) so illegal "sampling + reasoning at once" states are
 * unrepresentable. Everything here is plain JSON — no live handle, no closures — which is what lets the
 * call become a durable WDK **step** (persisted inputs/outputs must serialize; hence `model` is a string
 * id). The schema is layered on by {@link StructuredCallParams} (in v1 from `spec.outputSchema`); the
 * model and validator are reconstructed *inside* the executor from `model`/`schema`, never crossing the
 * step boundary.
 */
export type LlmCallDefinition = LlmCallConfig & CallPrompt;

/**
 * ONE structured LLM call: an {@link LlmCallDefinition} with the output schema folded in and the time
 * budget resolved. Generic in `T`, the parsed output type — the `schema` DESCRIBES `T`, and the
 * resulting `CallOutcome<T>.value` is typed as `T`. (A JSON Schema can't be statically tied to `T`, so
 * `T` is the caller's assertion that this schema produces that shape; the §4 Ajv boundary enforces it
 * at runtime. Use `typedSchema<T>(doc)` to infer `T` from the schema.)
 */
export type StructuredCallParams<T = unknown> = LlmCallDefinition & {
  /** The output JSON Schema, OR omitted for a TEXT-output op (plain text, no structured output, §3.14). */
  schema?: JsonSchema<T>;
  /** Output-schema content id. Optional and informational only: validation compiles from the INLINE
   *  `schema` document (the extracted SchemaValidator caches by content hash itself). */
  schemaId?: string;
  /** Remaining wall-clock budget for this call's abort signal (§6.2) — REQUIRED at call time. */
  timeoutMs: number;
};

/**
 * The injected runtime ENVIRONMENT a call executes against (declarative-ai's "provider block" + creds):
 * everything that ISN'T the serializable declaration — provider resolution, boundary validation, tool
 * implementations, and (later phases) blob/session stores + observers. Every seam is OPTIONAL; the floor is
 * a `modelRouter` (a call that actually reaches a model errors at execution if it's absent).
 * Non-serializable by design — this never enters the content hash or the durable declaration.
 */
export interface LlmCallEnvironment {
  /** Model resolution (keys/endpoints/strict flags). Required to actually reach a model. */
  modelRouter?: ModelRouter;
  /** Boundary validator (an `@declarative-ai/services` SchemaValidator or any `OutputValidator`). */
  validator?: OutputValidator;
  /** `execute` implementations for declared FUNCTION tools, keyed by tool name. A declared tool with no
   *  implementation here is SINGLE-TURN (its call is returned in the outcome, not executed); supplying one
   *  turns the call into a bounded tool LOOP. Provider tools need no implementation. */
  toolExecutors?: Record<string, ToolExecutor>;
  /** External abort (caller cancel) — combined with the per-call `timeoutMs` signal. */
  abortSignal?: AbortSignal;
  /** Content-addressed blob store — resolves a `FileInput`'s `contentHash`/`path` reference to bytes/URL. */
  blobs?: BlobStore;
}

/** Runtime dependencies `executeStructuredCall` reconstructs the call from — an {@link LlmCallEnvironment}
 *  with `modelRouter` REQUIRED (the base call can't resolve a model without it). */
export type CallDeps = LlmCallEnvironment & { modelRouter: ModelRouter };

/**
 * The ergonomic bundle for a one-shot call: a full serializable declaration ({@link StructuredCallParams})
 * with its runtime {@link LlmCallEnvironment} attached under `env`. This is the ONLY place declaration and
 * environment co-exist; {@link executeRequest} splits them back apart — stripping `env` so a non-declarative
 * handle never reaches anything that serializes/hashes the declaration — and calls the base method.
 */
export type LlmCallRequest<T = unknown> = StructuredCallParams<T> & { env: LlmCallEnvironment };

/** Convenience over {@link executeStructuredCall}: split an {@link LlmCallRequest} into its declaration +
 *  environment and execute. The single place `env` is stripped from the config. */
export async function executeRequest<T = unknown>(req: LlmCallRequest<T>): Promise<CallOutcome<T>> {
  const { env, ...rest } = req;
  const config = rest as StructuredCallParams<T>;
  if (!env.modelRouter) throw new Error("executeRequest: env.modelRouter is required to execute a call");
  return executeStructuredCall(config, { ...env, modelRouter: env.modelRouter });
}

/** A pluggable executor for one structured call — the seam the WDK step swaps in (§6/§10.3).
 *  Generic per call: the output type `T` flows from the params' schema to the outcome's value. */
export type StructuredCallExecutor = <T = unknown>(params: StructuredCallParams<T>) => Promise<CallOutcome<T>>;

/**
 * The default (in-process) executor: resolve the model, rebuild the boundary validator from the
 * schema, and run the structured call. This is the body the WDK step also runs — the only
 * difference in production is that it runs *inside* a durable, replayable step (§10.3). Pure and
 * dependency-injected; it never imports the workflow runtime, so the engine stays substrate-agnostic.
 */
export async function executeStructuredCall<T = unknown>(params: StructuredCallParams<T>, deps: CallDeps): Promise<CallOutcome<T>> {
  // Model ids are route-prefixed `{route}/{model}` — the exact catalog / pricing / schema-profile key
  // (`resolveModel` re-parses the full id to pick the client). Validate the prefix up front so a
  // bare/unprefixed id is a permanent failure surfaced via the never-throw outcome, not a raised error.
  try {
    providerNativeId(params.model);
  } catch (e) {
    return failFast<T>(e instanceof Error ? e.message : String(e));
  }
  // §4 boundary validation is always against the ORIGINAL schema (the value `postProcess` reconstructs
  // back to), never the provider-adapted one we send.
  let validate: ((value: unknown) => void) | undefined;
  if (deps.validator && params.schema) {
    const validator = deps.validator;
    const originalSchema = params.schema;
    validate = (value: unknown) => {
      const result = validator.validateValue(originalSchema, value);
      if (!result.ok) throw new Error(`output failed schema validation: ${result.errors ?? "unknown error"}`);
    };
  }
  // Adapt the schema for the provider transport (§5.1), then resolve the model with the MATCHING strict
  // flag — so a schema that fits the constrained decoder gets `strict`, one that doesn't is sent as a
  // json_object hint, and a text-tier model gets no `response_format` at all. `outgoing` goes to the
  // model; `postProcess` reverses it back to the original shape.
  const profile = params.schema ? profileForModelId(params.model) : undefined;
  const adapt = params.schema && profile ? adaptSchemaCached(params.schema, profile) : undefined;
  const model = deps.modelRouter.resolveModel(params.model, { strictStructuredOutput: adapt?.enforce === "strict" });

  // §5.1 wire-mode shaping of the system prompt (the ONLY place the candidate prompt is touched, and only
  // for non-strict tiers):
  //  - TEXT tier (`enforce:"text"`): no `response_format` is sent, so DESCRIBE the schema in the prompt —
  //    that's how the model learns the shape it must emit as plain-text JSON.
  //  - json_object tier (`enforce:"advisory"`): honor the profile's `promptRequiresJSONSpecifier` contract
  //    (OpenAI-compatible upstreams 400 unless the messages contain "json"). `"force"` injects a directive;
  //    `true` fails fast (permanent) so the candidate isn't silently 400'd with the prompt left intact.
  let system = params.system;
  if (adapt?.enforce === "text" && adapt.outgoing) {
    system = appendToSystem(system, schemaPromptHint(adapt.outgoing));
  } else if (adapt?.enforce === "advisory" && profile?.promptRequiresJSONSpecifier) {
    const mentionsJson = /json/i.test(promptText(params));
    if (!mentionsJson) {
      if (profile.promptRequiresJSONSpecifier === "force") {
        system = appendToSystem(system, JSON_OBJECT_DIRECTIVE);
      } else {
        return failFast<T>(
          `model ${params.model} requires the word "json" in the prompt to use json_object mode (promptRequiresJSONSpecifier), but neither the system nor user prompt contained it`,
        );
      }
    }
  }

  // Filter the sampling params to those the model actually accepts (§5.1). A param no endpoint
  // supports — e.g. `temperature`/`top_p`/`top_k` on an OpenAI reasoning model — would otherwise be
  // dragged into OpenRouter's `require_parameters` routing constraint (which we set whenever strict is
  // on) and make EVERY endpoint a non-match → HTTP 404 "No endpoints found that can handle the
  // requested parameters". `paramAcceptance` reads the recorded capability (refreshed from the provider),
  // a reasoning-family fallback for a cold catalog, or accepts-everything for an unknown model (⇒ send
  // everything, the prior behavior). Cost/identity params (maxOutputTokens) are never filtered.
  const { accepts, acceptsReasoning } = ModelInfo.instance.paramAcceptance(params.model);

  // Narrow the sampling-XOR-reasoning union ONCE (§config-as-dimensions): a reasoning config carries the
  // neutral `reasoning` spec and no sampling knobs; a sampling config the reverse. `reasoning` is the
  // discriminant, so `"reasoning" in params` splits the union cleanly. `sampling` is the narrowed branch
  // we read the decoding knobs off (undefined for a reasoning config).
  const sampling = "reasoning" in params ? undefined : params;
  const reasoning = "reasoning" in params ? params.reasoning : undefined;

  // Reasoning: capability-gated like the sampling params (the OpenRouter param name is `reasoning`), then
  // ADAPTED from the neutral `ReasoningSpec` to the provider's `providerOptions` shape. Undefined when the
  // model rejects reasoning or none was requested → the call is byte-identical to a no-reasoning call. The
  // adapted reasoning is MERGED over the config's raw `providerOptions` passthrough (adapted reasoning wins
  // per provider key, since it's the first-class neutral request; the raw bag is the escape hatch).
  const adaptedReasoning = acceptsReasoning ? adaptReasoning(reasoning, { anthropic: deps.modelRouter.isAnthropic(params.model) }) : undefined;
  const providerOptions = mergeProviderOptions(params.providerOptions, adaptedReasoning);

  // Build the runtime tool set from the serializable declarations + injected `execute` impls, and bound
  // the loop when any function tool is executable (else it's single-turn: the call is returned, not run).
  const tools = buildToolSet(params.tools, deps.toolExecutors);
  const executable = (params.tools ?? []).some((t) => t.type !== "provider" && deps.toolExecutors?.[t.name] !== undefined);
  const stopWhen: StopCondition<ToolSet> | undefined = tools && executable ? stepCountIs(params.maxSteps ?? DEFAULT_MAX_STEPS) : undefined;

  // Lower any neutral file `attachments` into provider file parts merged into the user turn (resolving
  // contentHash/path refs via the blob store). Failure is a permanent outcome (never-throw contract).
  let lowered: { prompt?: string | ModelMessage[]; messages?: ModelMessage[] };
  try {
    lowered = await lowerAttachments(params, deps.blobs);
  } catch (e) {
    return failFast<T>(e instanceof Error ? e.message : String(e));
  }

  return generateStructured<T>({
    model,
    modelId: params.model,
    prompt: lowered.prompt,
    messages: lowered.messages,
    system,
    schema: adapt ? adapt.outgoing : params.schema,
    tools,
    toolChoice: params.toolChoice as ToolChoice<ToolSet> | undefined,
    stopWhen,
    postProcess: adapt?.postProcess,
    // TEXT tier sends no response_format — the schema is in the prompt (see above) and the JSON is parsed
    // out of the plain-text answer; every other tier attaches Output.object as before.
    attachStructuredOutput: adapt?.enforce !== "text",
    abortSignal: deps.abortSignal
      ? AbortSignal.any([deps.abortSignal, AbortSignal.timeout(params.timeoutMs)])
      : AbortSignal.timeout(params.timeoutMs),
    temperature: accepts("temperature") ? sampling?.temperature : undefined,
    topP: accepts("topP") ? sampling?.topP : undefined,
    topK: accepts("topK") ? sampling?.topK : undefined,
    presencePenalty: accepts("presencePenalty") ? sampling?.presencePenalty : undefined,
    frequencyPenalty: accepts("frequencyPenalty") ? sampling?.frequencyPenalty : undefined,
    seed: accepts("seed") ? params.seed : undefined,
    maxOutputTokens: params.maxOutputTokens,
    stopSequences: accepts("stopSequences") ? params.stopSequences : undefined,
    providerOptions,
    validate,
  });
}

/** Build the SDK `ToolSet` from the serializable declarations + injected `execute` impls. A FUNCTION tool
 *  becomes a `tool({ inputSchema: jsonSchema(...), execute? })` — with `execute` it runs in-loop, without
 *  it the model's call is returned single-turn. A PROVIDER tool becomes a `{type:"provider", id, args}`
 *  (server-side; the provider owns its schema, so a permissive placeholder is passed). Undefined ⇒ no tools. */
function buildToolSet(defs: ToolDefinition[] | undefined, executors: Record<string, ToolExecutor> | undefined): ToolSet | undefined {
  if (!defs || defs.length === 0) return undefined;
  const set: ToolSet = {};
  for (const d of defs) {
    if (d.type === "provider") {
      set[d.name] = {
        type: "provider",
        id: d.id,
        args: d.args,
        inputSchema: jsonSchema({ type: "object", additionalProperties: true } as never),
      } as ToolSet[string];
    } else {
      const execute = executors?.[d.name];
      set[d.name] = {
        ...(d.description !== undefined ? { description: d.description } : {}),
        inputSchema: jsonSchema(d.inputSchema as never),
        ...(execute ? { execute } : {}),
        ...(d.strict !== undefined ? { strict: d.strict } : {}),
      } as ToolSet[string];
    }
  }
  return set;
}

/** Resolve a `FileInput`'s data to something the SDK file part accepts: an inline base64 string, a `URL`,
 *  or bytes loaded from the blob store (for a contentHash/path reference). */
async function resolveFileData(d: FileInput["data"], blobs?: BlobStore): Promise<string | Uint8Array | URL> {
  if ("base64" in d) return d.base64;
  if ("url" in d) return new URL(d.url);
  if (!blobs) throw new Error("file input requires a blob store (env.blobs) to resolve a contentHash/path reference");
  const loaded = await blobs.load("contentHash" in d ? { contentHash: d.contentHash } : { path: d.path });
  if (loaded.bytes) return loaded.bytes;
  if (loaded.url) return new URL(loaded.url);
  throw new Error("blob store returned neither bytes nor url for the file reference");
}

/** Lower neutral `FileInput` attachments into an AI-SDK `FilePart`. */
async function toFilePart(att: FileInput, blobs?: BlobStore): Promise<FilePart> {
  return {
    type: "file",
    data: await resolveFileData(att.data, blobs),
    mediaType: att.mediaType,
    ...(att.filename !== undefined ? { filename: att.filename } : {}),
  };
}

/** Per-attachment-array cache of lowered file parts: the SAME `attachments` array object re-enters the
 *  pipeline on every repair turn / plan-then-execute flow, and re-lowering re-fetches the bytes from the
 *  blob store. A rejected lowering is evicted so a later call (e.g. with a store now available) retries. */
const loweredFilesCache = new WeakMap<FileInput[], Promise<FilePart[]>>();

/** Merge the declaration's file `attachments` into the user turn: a string prompt becomes one user message
 *  of `[text, ...files]`; a message/array prompt gets a trailing user message carrying the files. No
 *  attachments ⇒ the prompt/messages pass through untouched. */
async function lowerAttachments(
  params: StructuredCallParams,
  blobs?: BlobStore,
): Promise<{ prompt?: string | ModelMessage[]; messages?: ModelMessage[] }> {
  const atts = params.attachments;
  if (!atts || atts.length === 0) return { prompt: params.prompt, messages: params.messages };
  let filesP = loweredFilesCache.get(atts);
  if (!filesP) {
    filesP = Promise.all(atts.map((a) => toFilePart(a, blobs)));
    loweredFilesCache.set(atts, filesP);
    filesP.catch(() => loweredFilesCache.delete(atts));
  }
  const files = await filesP;
  if (params.messages || Array.isArray(params.prompt)) {
    return { messages: [...promptAsMessages(params), { role: "user", content: files }] };
  }
  const content: Array<TextPart | FilePart> =
    typeof params.prompt === "string" && params.prompt.length > 0 ? [{ type: "text", text: params.prompt }, ...files] : files;
  return { messages: [{ role: "user", content }] };
}

/** Merge the config's raw `providerOptions` passthrough with the reasoning adapted at the boundary,
 *  one level deep and per provider key (adapted reasoning wins on conflict). Either side may be absent. */
function mergeProviderOptions(
  raw: ProviderOptions | undefined,
  adapted: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (!raw) return adapted;
  if (!adapted) return { ...raw };
  const out: Record<string, unknown> = { ...raw };
  for (const [provider, opts] of Object.entries(adapted)) {
    const existing = out[provider];
    out[provider] =
      existing && typeof existing === "object" && opts && typeof opts === "object"
        ? { ...(existing as Record<string, unknown>), ...(opts as Record<string, unknown>) }
        : opts;
  }
  return out;
}

/** A short directive appended to the system prompt to satisfy an OpenAI-compatible json_object upstream's
 *  "the messages must contain the word 'json'" contract (§5.1 `promptRequiresJSONSpecifier:"force"`). */
const JSON_OBJECT_DIRECTIVE = "\n\nReturn your answer as a single valid JSON object.";

/** Describe the (adapted) output schema in prose for a TEXT-tier model that gets no `response_format` —
 *  the only signal it has about the shape to emit. Kept compact; the §4 Ajv boundary is still the gate. */
function schemaPromptHint(schema: Record<string, unknown>): string {
  return `\n\nRespond with ONLY a single valid JSON value conforming to this JSON Schema — no prose, no markdown fences:\n${JSON.stringify(schema)}`;
}

/** Append a text hint to the system prompt (§5.1 wire-mode shaping). A string (or absent) system is
 *  concatenated — inserting the separator only when there's existing content, dropping the leading
 *  separator when empty so the addition doesn't start with blank lines. A STRUCTURED system (message or
 *  message array) gets the hint as an APPENDED system message, so multimodal/structured prompts still work. */
function appendToSystem(
  system: string | SystemModelMessage | SystemModelMessage[] | undefined,
  add: string,
): string | SystemModelMessage | SystemModelMessage[] {
  if (system === undefined || typeof system === "string") {
    return system && system.trim() ? `${system}${add}` : add.replace(/^\n+/, "");
  }
  const hint: SystemModelMessage = { role: "system", content: add.replace(/^\n+/, "") };
  return Array.isArray(system) ? [...system, hint] : [system, hint];
}

/** Build a terminal PERMANENT-failure outcome without hitting the provider (§5.1 fail-fast). Mirrors the
 *  `CallOutcome` shape `generateStructured` returns on error: best-effort empty output + the failure. */
function failFast<T = unknown>(reason: string): CallOutcome<T> {
  return { rawText: "", finishReason: "error", metrics: { durationMs: 0 }, error: { classification: "permanent", reason } };
}
