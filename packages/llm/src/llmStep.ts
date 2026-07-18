import type { OutputValidator, ReasoningSpec } from "@ai-exec/core";
import { generateStructured, type CallOutcome } from "./generate";
import { defaultModelCatalog, SAMPLING_PARAM_NAMES } from "./model-catalog";
import { adaptReasoning } from "./reasoning";
import type { ProviderRouter } from "./router";
import { adaptSchema, profileForModelId } from "./schema";

/**
 * The serializable description of ONE structured LLM call (§6.1). Everything here is plain JSON —
 * no live model handle, no closures — which is exactly what lets the call become a durable WDK
 * **step**: the step persists its inputs/outputs, so they must serialize. The live model and the
 * validator are reconstructed *inside* the executor from `modelId`/`schema`, never passed across
 * the step boundary.
 */
export interface StructuredCallParams {
  modelId: string;
  prompt: string;
  system?: string;
  /** The output JSON Schema, OR omitted for a TEXT-output op (plain text, no structured output, §3.14). */
  schema?: Record<string, unknown>;
  /** Output-schema content id. Optional and informational only: validation compiles from the INLINE
   *  `schema` document (the extracted SchemaValidator caches by content hash itself). */
  schemaId?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxOutputTokens?: number;
  stopSequences?: string[];
  /** Provider-neutral reasoning request (effort/budget); adapted to provider `providerOptions` at the
   *  call boundary (§reasoning-as-dimension). Only sent to models that accept reasoning. */
  reasoning?: ReasoningSpec;
  /** Remaining wall-clock budget for this call's abort signal (§6.2). */
  timeoutMs: number;
}

/** Runtime dependencies an executor reconstructs the call from (module-level in the WDK step). */
export interface CallDeps {
  providers: ProviderRouter;
  /** Boundary validator (an `@ai-exec/services` SchemaValidator or any `OutputValidator`). */
  validator?: OutputValidator;
  /** External abort (caller cancel) — combined with the per-call `timeoutMs` signal. */
  abortSignal?: AbortSignal;
}

/** A pluggable executor for one structured call — the seam the WDK step swaps in (§6/§10.3). */
export type StructuredCallExecutor = (params: StructuredCallParams) => Promise<CallOutcome>;

/**
 * The default (in-process) executor: resolve the model, rebuild the boundary validator from the
 * schema, and run the structured call. This is the body the WDK step also runs — the only
 * difference in production is that it runs *inside* a durable, replayable step (§10.3). Pure and
 * dependency-injected; it never imports the workflow runtime, so the engine stays substrate-agnostic.
 */
export async function executeStructuredCall(params: StructuredCallParams, deps: CallDeps): Promise<CallOutcome> {
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
  const profile = params.schema ? profileForModelId(params.modelId) : undefined;
  const adapt = params.schema && profile ? adaptSchema(params.schema, profile) : undefined;
  const model = deps.providers.resolveModel(params.modelId, { strictStructuredOutput: adapt?.enforce === "strict" });

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
    const mentionsJson = /json/i.test(`${system ?? ""}\n${params.prompt}`);
    if (!mentionsJson) {
      if (profile.promptRequiresJSONSpecifier === "force") {
        system = appendToSystem(system, JSON_OBJECT_DIRECTIVE);
      } else {
        return failFast(
          `model ${params.modelId} requires the word "json" in the prompt to use json_object mode (promptRequiresJSONSpecifier), but neither the system nor user prompt contained it`,
        );
      }
    }
  }

  // Filter the sampling params to those the model actually accepts (§5.1). A param no endpoint
  // supports — e.g. `temperature`/`top_p`/`top_k` on an OpenAI reasoning model — would otherwise be
  // dragged into OpenRouter's `require_parameters` routing constraint (which we set whenever strict is
  // on) and make EVERY endpoint a non-match → HTTP 404 "No endpoints found that can handle the
  // requested parameters". `supportedParametersFor` returns the recorded capability (refreshed from
  // the provider), a reasoning-family fallback for a cold catalog, or `undefined` for an unknown model
  // (⇒ send everything, the prior behavior). Cost/identity params (maxOutputTokens) are never filtered.
  const supported = defaultModelCatalog.supportedParametersFor(params.modelId);
  const accepts = (key: keyof typeof SAMPLING_PARAM_NAMES): boolean =>
    supported === undefined || supported.includes(SAMPLING_PARAM_NAMES[key]);

  // Reasoning: capability-gated like the sampling params (the OpenRouter param name is `reasoning`), then
  // ADAPTED from the neutral `ReasoningSpec` to the provider's `providerOptions` shape. Undefined when the
  // model rejects reasoning or none was requested → the call is byte-identical to a no-reasoning call.
  const acceptsReasoning = supported === undefined || supported.includes("reasoning");
  const providerOptions = acceptsReasoning ? adaptReasoning(params.reasoning, { anthropic: deps.providers.isAnthropic(params.modelId) }) : undefined;

  return generateStructured({
    model,
    modelId: params.modelId,
    prompt: params.prompt,
    system,
    schema: adapt ? adapt.outgoing : params.schema,
    postProcess: adapt?.postProcess,
    // TEXT tier sends no response_format — the schema is in the prompt (see above) and the JSON is parsed
    // out of the plain-text answer; every other tier attaches Output.object as before.
    attachStructuredOutput: adapt?.enforce !== "text",
    abortSignal: deps.abortSignal
      ? AbortSignal.any([deps.abortSignal, AbortSignal.timeout(params.timeoutMs)])
      : AbortSignal.timeout(params.timeoutMs),
    temperature: accepts("temperature") ? params.temperature : undefined,
    topP: accepts("topP") ? params.topP : undefined,
    topK: accepts("topK") ? params.topK : undefined,
    maxOutputTokens: params.maxOutputTokens,
    stopSequences: accepts("stopSequences") ? params.stopSequences : undefined,
    providerOptions,
    validate,
  });
}

/** A short directive appended to the system prompt to satisfy an OpenAI-compatible json_object upstream's
 *  "the messages must contain the word 'json'" contract (§5.1 `promptRequiresJSONSpecifier:"force"`). */
const JSON_OBJECT_DIRECTIVE = "\n\nReturn your answer as a single valid JSON object.";

/** Describe the (adapted) output schema in prose for a TEXT-tier model that gets no `response_format` —
 *  the only signal it has about the shape to emit. Kept compact; the §4 Ajv boundary is still the gate. */
function schemaPromptHint(schema: Record<string, unknown>): string {
  return `\n\nRespond with ONLY a single valid JSON value conforming to this JSON Schema — no prose, no markdown fences:\n${JSON.stringify(schema)}`;
}

/** Append to the system prompt, inserting a separator only when there's existing content (and dropping
 *  the leading separator when the system prompt is empty, so the addition doesn't start with blank lines). */
function appendToSystem(system: string | undefined, add: string): string {
  return system && system.trim() ? `${system}${add}` : add.replace(/^\n+/, "");
}

/** Build a terminal PERMANENT-failure outcome without hitting the provider (§5.1 fail-fast). Mirrors the
 *  `CallOutcome` shape `generateStructured` returns on error: best-effort empty output + the failure. */
function failFast(reason: string): CallOutcome {
  return { rawText: "", finishReason: "error", metrics: { durationMs: 0 }, error: { classification: "permanent", reason } };
}
