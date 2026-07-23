/**
 * ONE structured LLM call: `executeLlmCall(definition, environment)` (DESIGN §4.1).
 *
 * Four call shapes became two plus an environment:
 *
 * | was                        | now                                                              |
 * | -------------------------- | ---------------------------------------------------------------- |
 * | `PromptOp`                 | kept — a declaration with unresolved bindings (`@declarative-ai/ops`) |
 * | `LlmCallDefinition`        | kept, and it now carries the output `schema`                      |
 * | `StructuredCallParams<T>`  | REMOVED — it differed by two optional fields and a required-ness  |
 * | `GenerateStructuredParams` | REMOVED — collapsed into `(def, env)`                             |
 *
 * The DECLARATION is serializable and hashable; the ENVIRONMENT holds live handles and closures. They
 * never merge into one bag again: that flattening is what made `schemaId` look necessary and what forced
 * the old lowering to smuggle the output schema through `spec.outputSchema` and cast the phantom away.
 */
import { jsonSchema, stepCountIs, type FilePart, type ModelMessage, type StopCondition, type TextPart, type ToolCallOptions, type ToolChoice, type ToolSet } from "ai";
import type { JsonValue, OutputValidator } from "@declarative-ai/json";
import type { LlmCallDefinition, ProviderOptions, ToolDefinition } from "./llmConfig";
import type { FileInput } from "./files";
import { generateStructured, type GenerateEnvironment } from "./generate";
import type { LlmCallResult } from "./output";
import { ModelInfo } from "./model-catalog";
import { promptAsMessages, promptText, type CallPromptInput } from "./prompt";
import { adaptReasoning } from "./reasoning";
import { providerNativeId, type ModelRouter } from "./router";
import { adaptSchemaCached, profileForModelId, type ProviderSchemaProfile } from "./schema";

/** A runtime tool implementation — the `execute` for a declared FUNCTION tool, looked up by tool name.
 *  Injected via the ENVIRONMENT (not serialized), mirroring how the model handle and validator are. */
export type ToolExecutor<I = Record<string, JsonValue>, O = JsonValue> = (input: I, options: ToolCallOptions) => O | Promise<O>;

/** Default cap on tool-loop steps when executable tools are present and the config names no `maxSteps`. */
const DEFAULT_MAX_STEPS = 8;

/** Default per-call timeout when neither the definition nor the caller names one (10 min). */
export const DEFAULT_TIMEOUT_MS = 10 * 60_000;

/**
 * The injected runtime ENVIRONMENT a call executes against: everything that ISN'T the serializable
 * declaration — provider resolution, boundary validation, tool implementations, cancellation. Every
 * seam is OPTIONAL; the floor is a `modelRouter`. Non-serializable by design — this never enters the
 * content hash or the durable declaration.
 */
export interface LlmCallEnvironment {
  /** Model resolution (keys/endpoints/strict flags). Required to actually reach a model. */
  modelRouter?: ModelRouter;
  /** Boundary validator — `json`'s three-line `OutputValidator` seam, so nothing here knows about ajv. */
  validator?: OutputValidator;
  /** `execute` implementations for declared FUNCTION tools, keyed by tool name. A declared tool with no
   *  implementation here is SINGLE-TURN (its call is returned in the outcome, not executed); supplying
   *  one turns the call into a bounded tool LOOP. Provider tools need no implementation. */
  toolExecutors?: Record<string, ToolExecutor>;
  /** External abort (caller cancel) — combined with the per-call `timeoutMs` signal. */
  abortSignal?: AbortSignal;
  /**
   * How this model's structured-output transport profile is resolved. Defaults to the built-in
   * catalog-backed `profileForModelId`. An environment seam because profile knowledge is DEPLOYMENT
   * state, not declaration content — a consumer that derives profiles from its own store (e.g. a DB of
   * observed `supported_parameters`) injects its resolver here instead of priming a global. Returning
   * `undefined` means "no profile known": the schema is sent unadapted.
   */
  schemaProfile?: (modelId: string) => ProviderSchemaProfile | undefined;
}

/** The environment with `modelRouter` REQUIRED (the base call can't resolve a model without it). */
export type CallDeps = LlmCallEnvironment & { modelRouter: ModelRouter };

/**
 * Execute one structured call.
 *
 * TEXT mode yields `string` — a text call produces text. It previously yielded `LlmCallResult<JsonValue>`,
 * which was simply wrong. The overloads discriminate on the ABSENCE of a schema, so the typing is a
 * property of the declaration rather than something the caller has to assert.
 */
export function executeLlmCall(
  def: LlmCallDefinition & { schema?: undefined },
  env: CallDeps,
  timeoutMs?: number,
): Promise<LlmCallResult<string>>;
export function executeLlmCall<T>(def: LlmCallDefinition<T>, env: CallDeps, timeoutMs?: number): Promise<LlmCallResult<T>>;
export function executeLlmCall<T = JsonValue>(
  def: LlmCallDefinition<T>,
  env: CallDeps,
  timeoutMs?: number,
): Promise<LlmCallResult<T>> {
  return runCall(def, env, timeoutMs);
}

async function runCall<T>(def: LlmCallDefinition<T>, env: CallDeps, timeoutArg?: number): Promise<LlmCallResult<T>> {
  // Model ids are route-prefixed `{route}/{model}` — the exact catalog / pricing / schema-profile key.
  // Validate the prefix up front so a bare/unprefixed id is a permanent failure surfaced via the
  // never-throw outcome, not a raised error.
  try {
    providerNativeId(def.model);
  } catch (e) {
    return failFast<T>(e instanceof Error ? e.message : String(e));
  }

  // Boundary validation is always against the ORIGINAL schema (the value `postProcess` reconstructs
  // back to), never the provider-adapted one we send.
  let validate: ((value: JsonValue) => void) | undefined;
  if (env.validator && def.schema) {
    const validator = env.validator;
    const originalSchema = def.schema;
    validate = (value: JsonValue) => {
      const result = validator.validateValue(originalSchema, value);
      if (!result.ok) throw new Error(`output failed schema validation: ${result.errors ?? "unknown error"}`);
    };
  }

  // Adapt the schema for the provider transport, then resolve the model with the MATCHING strict flag —
  // so a schema that fits the constrained decoder gets `strict`, one that doesn't is sent as a
  // json_object hint, and a text-tier model gets no `response_format` at all.
  const profile = def.schema ? (env.schemaProfile ?? profileForModelId)(def.model) : undefined;
  const adapt = def.schema && profile ? adaptSchemaCached(def.schema, profile) : undefined;
  const model = env.modelRouter.resolveModel(def.model, { strictStructuredOutput: adapt?.enforce === "strict" });

  // Wire-mode shaping of the system prompt (the ONLY place the authored prompt is touched, and only for
  // non-strict tiers):
  //  - TEXT tier (`enforce:"text"`): no `response_format` is sent, so DESCRIBE the schema in the prompt.
  //  - json_object tier (`enforce:"advisory"`): honor the profile's `promptRequiresJSONSpecifier`
  //    contract. `"force"` injects a directive; `true` fails fast (permanent) so the call isn't silently
  //    400'd with the prompt left intact.
  let system = def.system;
  if (adapt?.enforce === "text" && adapt.outgoing) {
    system = appendToSystem(system, schemaPromptHint(adapt.outgoing));
  } else if (adapt?.enforce === "advisory" && profile?.promptRequiresJSONSpecifier) {
    if (!/json/i.test(promptText(def))) {
      if (profile.promptRequiresJSONSpecifier === "force") {
        system = appendToSystem(system, JSON_OBJECT_DIRECTIVE);
      } else {
        return failFast<T>(
          `model ${def.model} requires the word "json" in the prompt to use json_object mode (promptRequiresJSONSpecifier), but neither the system nor user prompt contained it`,
        );
      }
    }
  }

  // Which decoding knobs the model actually accepts. A param no endpoint supports — e.g. `temperature`
  // on an OpenAI reasoning model — would otherwise be dragged into OpenRouter's `require_parameters`
  // routing constraint and make EVERY endpoint a non-match (HTTP 404 "No endpoints found").
  const { accepts, acceptsReasoning } = ModelInfo.instance.paramAcceptance(def.model);

  // Reasoning: capability-gated like the sampling params, then ADAPTED from the neutral `ReasoningSpec`
  // to the provider's `providerOptions` shape, MERGED over the config's raw passthrough (adapted
  // reasoning wins per provider key, since it's the first-class neutral request).
  const reasoning = "reasoning" in def ? def.reasoning : undefined;
  const adaptedReasoning = acceptsReasoning ? adaptReasoning(reasoning, { anthropic: env.modelRouter.isAnthropic(def.model) }) : undefined;
  const providerOptions = mergeProviderOptions(def.providerOptions, adaptedReasoning);

  // Build the runtime tool set from the serializable declarations + injected `execute` impls, and bound
  // the loop when any function tool is executable (else it's single-turn: the call is returned, not run).
  const tools = buildToolSet(def.tools, env.toolExecutors);
  const executable = (def.tools ?? []).some((t: ToolDefinition) => t.type !== "provider" && env.toolExecutors?.[t.name] !== undefined);
  const stopWhen: StopCondition<ToolSet> | undefined = tools && executable ? stepCountIs(def.maxSteps ?? DEFAULT_MAX_STEPS) : undefined;

  // Lower any file `attachments` into provider file parts merged into the user turn. Failure is a
  // permanent outcome (never-throw contract).
  let lowered: { prompt?: string | ModelMessage[]; messages?: ModelMessage[] };
  try {
    lowered = await lowerAttachments(def);
  } catch (e) {
    return failFast<T>(e instanceof Error ? e.message : String(e));
  }

  // The per-call budget: the caller's argument wins (a deadline clamp), else the declaration's own,
  // else the default. It is an ARGUMENT rather than a required declaration field because it is a
  // call-site concern (§5.1).
  const timeoutMs = timeoutArg ?? def.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  // The EFFECTIVE declaration: the authored one with wire-mode shaping and attachment lowering applied.
  // Both edits are to declaration FIELDS, which is why they belong here and not in the environment.
  const effective: LlmCallDefinition<T> = { ...def, system, prompt: lowered.prompt, messages: lowered.messages };

  const environment: GenerateEnvironment<T> = {
    model,
    outgoing: adapt ? (adapt.outgoing as typeof def.schema) : def.schema,
    postProcess: adapt?.postProcess,
    validate,
    tools,
    toolChoice: def.toolChoice as ToolChoice<ToolSet> | undefined,
    stopWhen,
    providerOptions,
    // TEXT tier sends no response_format — the schema is in the prompt and the JSON is parsed out of
    // the plain-text answer; every other tier attaches Output.object.
    attachStructuredOutput: adapt?.enforce !== "text",
    accepts: (param: string) => accepts(param as never),
    abortSignal: env.abortSignal ? AbortSignal.any([env.abortSignal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs),
  };
  return generateStructured<T>(effective, environment);
}

/**
 * The ergonomic bundle for a one-shot call: a declaration with its environment attached under `env`.
 * This is the ONLY place declaration and environment co-exist; {@link executeRequest} splits them back
 * apart — stripping `env` so a non-declarative handle never reaches anything that serializes or hashes
 * the declaration — and calls the base function.
 */
export type LlmCallRequest<T = JsonValue> = LlmCallDefinition<T> & { env: LlmCallEnvironment };

/** Convenience over {@link executeLlmCall}: split a request into declaration + environment and execute.
 *  Carries the SAME schema-absence overload — a schema-less request is a TEXT call and yields `string`
 *  (§5.2). Without it this was the one remaining path where a text call's `parsed` was a runtime string
 *  typed `JsonValue`: not unsound, but the claim was not expressed in the types. */
export function executeRequest(req: LlmCallRequest & { schema?: undefined }): Promise<LlmCallResult<string>>;
export function executeRequest<T = JsonValue>(req: LlmCallRequest<T>): Promise<LlmCallResult<T>>;
export async function executeRequest<T = JsonValue>(req: LlmCallRequest<T>): Promise<LlmCallResult<T>> {
  const { env, ...rest } = req;
  const def = rest as LlmCallDefinition<T>;
  if (!env.modelRouter) throw new Error("executeRequest: env.modelRouter is required to execute a call");
  return executeLlmCall<T>(def, { ...env, modelRouter: env.modelRouter });
}

/** Build the SDK `ToolSet` from the serializable declarations + injected `execute` impls. A FUNCTION
 *  tool becomes a `tool({ inputSchema, execute? })` — with `execute` it runs in-loop, without it the
 *  model's call is returned single-turn. A PROVIDER tool becomes a `{type:"provider", id, args}`
 *  (server-side; the provider owns its schema, so a permissive placeholder is passed). */
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

/**
 * Resolve a `FileInput`'s data to something the SDK file part accepts. There is no blob store to
 * consult (DESIGN §3.6, "There is no blob store"): the library takes bytes, a base64 string, or a URL, and a caller
 * that has a content hash or a filesystem path resolves it BEFORE calling. That is what keeps this
 * package free of `fetch` and `node:fs`.
 */
function resolveFileData(d: FileInput["data"]): string | Uint8Array | URL {
  if (d instanceof Uint8Array) return d;
  if ("base64" in d) return d.base64;
  return new URL(d.url);
}

/** Lower a `FileInput` attachment into an AI-SDK `FilePart`. */
function toFilePart(att: FileInput): FilePart {
  return {
    type: "file",
    data: resolveFileData(att.data),
    mediaType: att.mediaType,
    ...(att.filename !== undefined ? { filename: att.filename } : {}),
  };
}

/** Merge the declaration's file `attachments` into the user turn: a string prompt becomes one user
 *  message of `[text, ...files]`; a message/array prompt gets a trailing user message carrying the
 *  files. No attachments ⇒ the prompt/messages pass through untouched. */
async function lowerAttachments(def: CallPromptInput): Promise<{ prompt?: string | ModelMessage[]; messages?: ModelMessage[] }> {
  const atts = def.attachments;
  if (!atts || atts.length === 0) return { prompt: def.prompt, messages: def.messages };
  const files = atts.map(toFilePart);
  if (def.messages || Array.isArray(def.prompt)) {
    return { messages: [...promptAsMessages(def), { role: "user", content: files }] };
  }
  const content: Array<TextPart | FilePart> =
    typeof def.prompt === "string" && def.prompt.length > 0 ? [{ type: "text", text: def.prompt }, ...files] : files;
  return { messages: [{ role: "user", content }] };
}

/** Merge the config's raw `providerOptions` passthrough with the reasoning adapted at the boundary,
 *  one level deep and per provider key (adapted reasoning wins on conflict). Either side may be absent. */
function mergeProviderOptions(raw: ProviderOptions | undefined, adapted: ProviderOptions | undefined): Record<string, JsonValue> | undefined {
  if (!raw) return adapted;
  if (!adapted) return { ...raw };
  const out: Record<string, JsonValue> = { ...raw };
  for (const [provider, opts] of Object.entries(adapted)) {
    const existing = out[provider];
    out[provider] =
      existing && typeof existing === "object" && opts && typeof opts === "object"
        ? { ...(existing as Record<string, JsonValue>), ...(opts as Record<string, JsonValue>) }
        : (opts as JsonValue);
  }
  return out;
}

/** A short directive appended to the system prompt to satisfy an OpenAI-compatible json_object
 *  upstream's "the messages must contain the word 'json'" contract. */
const JSON_OBJECT_DIRECTIVE = "\n\nReturn your answer as a single valid JSON object.";

/** Describe the (adapted) output schema in prose for a TEXT-tier model that gets no `response_format` —
 *  the only signal it has about the shape to emit. The boundary validation is still the gate. */
function schemaPromptHint(schema: Record<string, unknown>): string {
  return `\n\nRespond with ONLY a single valid JSON value conforming to this JSON Schema — no prose, no markdown fences:\n${JSON.stringify(schema)}`;
}

/** Append a text hint to the system prompt. A string (or absent) system is concatenated — inserting the
 *  separator only when there's existing content. A STRUCTURED system (message or message array) gets the
 *  hint as an APPENDED system message, so multimodal/structured prompts still work. */
function appendToSystem(system: CallPromptInput["system"], add: string): CallPromptInput["system"] {
  if (system === undefined || typeof system === "string") {
    return system && system.trim() ? `${system}${add}` : add.replace(/^\n+/, "");
  }
  const hint = { role: "system" as const, content: add.replace(/^\n+/, "") };
  return Array.isArray(system) ? [...system, hint] : [system, hint];
}

/** Build a terminal PERMANENT-failure outcome without hitting the provider. Mirrors the shape
 *  `generateStructured` returns on error: best-effort empty output + the failure. */
function failFast<T>(reason: string): LlmCallResult<T> {
  return {
    error: { classification: "permanent", reason },
    value: { finishReason: "error" },
    metrics: { durationMs: 0, costUsd: 0, costSource: "unknown" },
  };
}
