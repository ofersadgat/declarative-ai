/**
 * Provider schema PROFILES (§5.1) — the config object that describes ONE structured-output transport's
 * JSON-Schema capabilities and limits. `adaptSchema` (./adapt) reads a profile and (a) transforms the
 * schema we send to the model and (b) returns a post-processor that reverses whatever it did, plus a
 * per-call `enforce` decision (strict constrained-decoding vs advisory hint).
 *
 * A profile describes the EFFECTIVE (provider + SDK) transport, so it is keyed by TRANSPORT, not just
 * provider. Where the SDK in front of us already cleans the schema (e.g. `@ai-sdk/anthropic` ≥3.0.73
 * sanitizes it on its native path — adds `additionalProperties:false`, rewrites `oneOf`→`anyOf`, strips
 * unsupported keywords), we express that as the transport NATIVELY SUPPORTING those things: the profile
 * declares `additionalProperties:"leave"`, keeps unions, strips no keywords — so the uniform adapter
 * leaves them alone and the SDK underneath does the work. There is no "sdkPreCleans" special case; the
 * difference between `anthropic:ai-sdk` (SDK cleans) and a hypothetical `anthropic:raw` (we clean) is
 * just different capability flags. OpenRouter cleans NOTHING (raw passthrough) and normalizes everything
 * to an OpenAI-compatible `response_format.json_schema` interface, so its upstreams SHARE one dialect
 * (OpenAI-strict) and differ only in `limits` / `supportsStructuredOutput`.
 */

import type { JsonValue, MutableSchema } from "@declarative-ai/json";

/** A JSON Schema node (an object schema) being TRANSFORMED — the ops mutable working form of the one
 *  `JsonSchema` document type (API.md, "The JSON vocabulary"), so an adapted schema flows straight back into the
 *  declaration vocabulary. Arrays/primitives are passed through untouched. */
export type SchemaNode = MutableSchema;

/**
 * How ONE JSON Schema keyword is handled by a provider:
 *  - `true`         — supported anywhere; keep it.
 *  - `false`        — unsupported; strip it.
 *  - `"nested-only"`— supported, but NOT at the root (e.g. OpenAI/Anthropic `anyOf`).
 *  - `"describe"`   — unsupported; fold the constraint into the node's `description` prose (the
 *                     `@ai-sdk/anthropic` strategy — preserves the hint without breaking the decoder).
 *  - object form    — `support` plus optional `pos` (position restriction) and `allowedValues`
 *                     (a whitelist; values outside it are dropped — e.g. the `format` list, or
 *                     `minItems ∈ {0,1}` on Anthropic).
 */
export type KeywordRule =
  | boolean
  | "nested-only"
  | "describe"
  | {
      support: "yes" | "strip" | "describe";
      pos?: "any" | "nested";
      allowedValues?: readonly JsonValue[];
    };

/**
 * Keyword support, in either form the design calls for:
 *  - ARRAY  — a whitelist: these keywords are supported, everything else is stripped.
 *  - OBJECT — explicit per-keyword rules; a keyword ABSENT from the map is allowed by default (so a
 *             provider that supports almost everything lists only its exceptions).
 */
export type KeywordSupport = readonly string[] | Record<string, KeywordRule>;

export interface ProviderSchemaProfile {
  /** Stable id, e.g. `"openai:strict"`, `"anthropic:ai-sdk"`, `"openrouter:strict"`, `"advisory"`. */
  id: string;
  /**
   * The structured-output tier the transport supports (§5.1) — the capability the routing + adapt
   * decision keys on. Three-valued because providers differ in KIND, not just presence:
   *  - `"schema"` — constrained decoding against a JSON Schema (json_schema / native structured outputs).
   *    The model's grammar is bound to the schema; `adaptSchema` attempts strict, falling back to
   *    json_object (advisory) if the specific schema doesn't fit the decoder's bounds.
   *  - `"object"` — JSON-object mode only (`response_format:{type:"json_object"}`): the model emits SOME
   *    JSON but conformance to a schema is NOT enforced by the decoder — the §4 Ajv boundary is the gate,
   *    and the schema rides along only as an advisory hint. Requires the word "json" in the prompt on
   *    OpenAI-compatible upstreams (see {@link promptRequiresJSONSpecifier}).
   *  - `false` — neither: no structured mode at all. The call is a PLAIN text completion (no
   *    `response_format`); the schema is described in the prompt and the JSON is parsed out of the text.
   * Derived from the model's `supported_parameters` at import (`structured_outputs`→`"schema"`,
   * else `response_format`→`"object"`, else `false`); native Anthropic overrides to `"schema"` (its SDK
   * carries structured output the OpenRouter param names don't describe).
   */
  supportsStructuredOutput: "schema" | "object" | false;
  /**
   * Whether this transport's json_object mode REQUIRES the literal word "json" in the messages (the
   * OpenAI-compatible contract — Alibaba/DashScope, OpenAI, and others 400 without it). Only consulted
   * when a call actually lands in json_object mode (`enforce:"advisory"`); irrelevant to strict/text.
   *  - falsy/absent — no requirement (the default; capability-routing usually keeps these off json_object).
   *  - `true`       — FAIL FAST: if neither system nor user prompt contains "json", the call is failed
   *                   locally (permanent) instead of hitting the provider's 400 — the prompt is left intact.
   *  - `"force"`    — if "json" is absent, APPEND a short JSON directive to the system prompt so the call
   *                   succeeds (the prompt is minimally augmented at call time; the stored op is untouched).
   */
  promptRequiresJSONSpecifier?: boolean | "force";
  /** How the provider expresses an ABSENT optional field — the 3-rung ladder:
   *  `"omit"` (truly optional, leave `required` alone) → `"nullable"` (force all required, mark optionals
   *  nullable so the model can answer `null`, drop the nulls on the way out) → `"none"` (force all
   *  required, no null — the lossy floor). */
  optionalSupport: "omit" | "nullable" | "none";
  /** How `null` is encoded when emulating optional via `"nullable"`. */
  nullable: "type-array" | "nullable-flag" | "anyOf-null";
  /** Object `additionalProperties` policy: force `false` (OpenAI/Anthropic), strip it (Gemini subset),
   *  or leave as-authored (advisory). */
  additionalProperties: "force-false" | "strip" | "leave";
  /** `oneOf`/`anyOf`/`allOf` strategy when WE handle unions (ignored when `sdkPreCleans`):
   *  `"flatten"` merges every variant into one object (+ `postProcess` reconstructs the matched variant),
   *  `"anyOf"` normalizes to nested `anyOf` and trusts the provider (no reconstruction). */
  unions: "flatten" | "anyOf";
  /** May a union appear at the ROOT? `false` ⇒ a root union is a non-fit for strict (→ advisory) unless
   *  flattening removed it. */
  rootUnion: boolean;
  /** May an ARRAY be the schema ROOT? `false` ⇒ the adapter wraps a root array in a single-property
   *  object (`{ items: [...] }`) on the way out and unwraps it on the way back (`postProcess`) — the
   *  OpenAI dialect and Anthropic tool inputs both require an object root, and OpenAI's plain JSON mode
   *  can only emit an object. The system-level type stays the bare array; the wrap is wire-only. */
  rootArray: boolean;
  /** Collapse `type: ["x","null"]` to its non-null member. Must be `false` on any profile that encodes
   *  optional via `nullable: "type-array"` (else it would undo the null). */
  collapseTypeArrays: boolean;
  /** How to represent an "any"/untyped node (`{}`): leave it `"native"` (only representable on providers
   *  that accept freeform — none of the strict ones, so it forces advisory), or `"encode-json-string"`
   *  (send a string field, `JSON.parse` it back) so even a closed-grammar provider can carry it. */
  anyType: "native" | "encode-json-string";
  /** `$ref` support. `"native"` keeps internal refs; `"inline"` would inline them (not yet implemented —
   *  treated as a non-fit). */
  refs: "native" | "inline";
  /** Whether the provider's constrained decoder accepts recursive schemas. `false` ⇒ a `$ref` is a
   *  non-fit for strict. */
  recursion: boolean;
  /** Max nesting depth the strict decoder accepts, COUNTED per `maxDepthCountStrategy` (undefined = no
   *  published cap). */
  maxDepth?: number;
  /** How `maxDepth` is counted — see {@link MaxDepthCountStrategy}. Defaults to `"all"` (conservative). */
  maxDepthCountStrategy?: MaxDepthCountStrategy;
  /** What to do when a schema exceeds `maxDepth` — see {@link MaxDepthStrategy}. Defaults to `"adapt"`. */
  maxDepthStrategy?: MaxDepthStrategy | null;
  /** Numeric strict-mode ceilings; exceeding any of them is a non-fit (→ advisory). */
  limits?: { maxProperties?: number; maxNameEnumChars?: number; maxEnumValues?: number };
  /** Per-keyword support (array whitelist or object rule map). */
  keywords: KeywordSupport;
}

/**
 * What `adaptSchema` does when a schema exceeds the transport's strict nesting-depth ceiling
 * (`maxDepth`). It governs ONLY the depth dimension — a non-depth strict-incompatibility (an `{}` any
 * node, a root union, a `maxProperties` overflow) follows its own path regardless of this value.
 *  - `"adapt"`            — send the full-depth schema as an advisory hint, strict off (today's behavior).
 *  - `"strict"`           — force strict at FULL depth anyway, overriding our own (deliberately
 *                           conservative) `maxDepth` estimate: send the native un-flattened schema and let
 *                           the provider be the arbiter. If it can't decode it, the CALL errors at runtime
 *                           — `adaptSchema` itself does not throw. (If a NON-depth blocker remains, still
 *                           advisory — forcing strict can't make a closed grammar represent `{}`.)
 *  - `"error"`            — throw at adapt time (trust the estimate; never attempt the call).
 *  - `"flatten"`          — losslessly flatten object chains to dotted keys (`./flatten`); if depth is
 *                           STILL over the cap after flattening (irreducible array nesting), THROW.
 *  - `"flatten-or-adapt"` — flatten as above; if depth is still over after flattening, fall back to `"adapt"`.
 * `undefined`/`null` ⇒ the default, `"adapt"` (so existing profiles are unchanged).
 */
export type MaxDepthStrategy = "adapt" | "strict" | "error" | "flatten" | "flatten-or-adapt";

/**
 * How a transport COUNTS "levels of nesting" toward {@link ProviderSchemaProfile.maxDepth}. Strict
 * decoders differ: OpenAI counts only nested OBJECTS — arrays and the terminal leaf are FREE (verified
 * live), so a 20-deep array chain is just "1 level". Counting arrays as levels (the conservative `"all"`)
 * therefore OVER-counts for those providers and forces needless advisory fallbacks on schemas they would
 * accept — which is why this is a per-transport choice, not a hardcoded rule.
 *  - `"objects"`    — only nested objects count (OpenAI's rule); arrays + the leaf are free.
 *  - `"containers"` — objects AND arrays count; the leaf is free.
 *  - `"all"`        — every nesting step counts (objects, arrays, leaf) — the conservative, never-under-
 *                     counts default for a transport whose rule we haven't measured.
 * A pure union node is TRANSPARENT under every mode (the chosen variant occupies its position).
 */
export type MaxDepthCountStrategy = "objects" | "containers" | "all";

/**
 * The enforcement decision `adaptSchema` returns for a single call (the wire mode):
 *  - `"strict"`   — constrained decoding (json_schema); `Output.object` sent with `strict` on.
 *  - `"advisory"` — json_object mode: `Output.object` sent WITHOUT strict; schema is a hint, Ajv gates.
 *  - `"text"`     — plain text completion: NO `Output.object`/`response_format`; the schema is described
 *                   in the prompt and the JSON is parsed out of the returned text.
 */
export type Enforcement = "strict" | "advisory" | "text";

/** A reason a schema fell back to advisory (or any other adaptation note), surfaced for diagnostics. */
export interface AdaptNote {
  code:
    | "no-structured-output"
    | "any-not-representable"
    | "recursion-unsupported"
    | "root-union-unsupported"
    | "root-array-wrapped"
    | "depth-exceeded"
    | "depth-flattened"
    | "depth-strict-forced"
    | "properties-exceeded"
    | "enums-exceeded"
    | "name-enum-chars-exceeded";
  detail: string;
}

export interface AdaptResult {
  /** The schema to send to the model (already provider-cleaned when `enforce === "strict"`). */
  outgoing: SchemaNode;
  /** Whether to request constrained decoding (`strict`) or send the schema as an advisory hint. */
  enforce: Enforcement;
  /** Reverse the model's answer back to the ORIGINAL schema's shape (union reconstruction, any-decode,
   *  nullable-optional drop). Identity when nothing lossy was applied. */
  postProcess: (value: JsonValue) => JsonValue;
  /** Why the call landed where it did (e.g. why it fell back to advisory). */
  notes: AdaptNote[];
}
