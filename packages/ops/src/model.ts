/**
 * The generic operation model (DESIGN §3.1, with the blob leaf kind at §3.7) — findmyprompt's
 * `PromptOp`/`FunctionOp` vocabulary made generic over the REFERENCE SUBSTRATE. findmyprompt leaves
 * are content ids into artifact stores; hw/JaiRA leaves are inline values. Both are instantiations of
 * one shape, parameterized by a "ref family".
 *
 * Semantics are findmyprompt's `src/engine/model/index.ts` unchanged; three things differ and only
 * these:
 *  1. Ref property names drop the `Id` suffix (`textId` → `text`, …) — the property no longer
 *     necessarily holds an id; the FAMILY decides what it holds.
 *  2. The `op` member is how a family NAMES a producer — a content id (`IdFamily`), or an embedded op /
 *     a declared child's local key (`InlineFamily`).
 *  3. `blob` is a leaf KIND alongside `text` and `json` (DESIGN §3.7). Binary data is a leaf
 *     value, so hydration is the family's business — the same as text and json. That is what lets
 *     `BlobStore`/`BlobRef` and the `x-artifact` marker disappear rather than be renamed: an id-family
 *     blob resolves through the SAME artifact store as text and json, and an inline-family blob IS the
 *     bytes (or a stream over them).
 *
 * This module has NO runtime dependencies and (by design) no execution machinery: content addressing
 * stays in findmyprompt's `artifacts/`, execution in `@declarative-ai/exec`.
 */
import type { Failure, JsonSchema, JsonValue, Result, SchemaDocument } from "@declarative-ai/json";
import type { Metrics } from "./metrics";

/** A content id (kind-prefixed sha256 hex) or a UUID domain-entity id. */
export type Id = string;

/**
 * The byte-stream shape a `blob` leaf may hold — structurally the WHATWG `ReadableStream<Uint8Array>`,
 * declared STRUCTURALLY here so `json` and `ops` need neither DOM nor `node:stream/web` types. A real
 * `ReadableStream<Uint8Array>` is assignable to it.
 *
 * Streaming is an input/output optimization, never part of a DEFINITION: an authored document is JSON,
 * so a stream only ever appears in a RUNTIME value. Where materialization is required — hashing for a
 * memo key, fan-out to two consumers, storing a `OperationRecord` — the stream would be replaced in the
 * resolved value by its `Uint8Array`: an idempotent, in-place upgrade of the runtime value. That
 * automatic materialization is NOT yet implemented (DESIGN §10.1) — those paths raise and the caller
 * materializes first.
 */
export interface ByteStream {
  getReader(): ByteStreamReader;
}

export interface ByteStreamReader {
  read(): Promise<{ done: boolean; value?: Uint8Array }>;
  cancel(reason?: unknown): Promise<void>;
  releaseLock(): void;
}

/** What an inline `blob` leaf holds: the bytes, or a stream over them. */
export type Bytes = Uint8Array | ByteStream;

/**
 * What a `Parameter` holds once it has been RESOLVED: JSON, the bytes of a `blob` leaf, or a container
 * of either. Blob is a leaf kind alongside text and json (DESIGN §3.7), so a function op
 * genuinely receives (and can produce) bytes. The container cases are what a `RefTree` resolves to: an
 * inline arrangement of refs whose leaves may mix json and blob.
 *
 * Was `SlotValue`, and it lived in the registry — but it is what a PARAMETER holds, so it belongs with
 * the model that defines parameters. The name matters too: it read as a sibling of `ValueRef` while
 * meaning its OPPOSITE. A `ValueRef` is kind-TAGGED and unresolved (`{json: x}` vs `{text: x}`); this
 * is the plain value you hold after resolving one.
 */
export type ResolvedValue = JsonValue | Bytes | ResolvedValue[] | { [key: string]: ResolvedValue };

// --- The ref family ----------------------------------------------------------

/**
 * The member types every op field is built from. These are generic BOUNDS
 * (`F extends RefFamily`), not value types — `unknown` here means "pinned by the
 * instantiation", and every instantiation pins them to precise types (§2.2 policy).
 */
export interface RefFamily {
  /** What a text-valued leaf holds (a prompt body, an error). */
  text: unknown;
  /** What a json-valued leaf holds (a config, a value). */
  json: unknown;
  /** What a BINARY leaf holds (an image, a pdf, a produced artifact). */
  blob: unknown;
  /** How a parameter carries its JSON Schema (an Id, or the schema document). */
  schema: unknown;
  /** What a result leaf holds (an id of, or an inline, OperationRecord). */
  result: unknown;
  /** How a producer names its operation (by id, embedded, or a local name). */
  op: unknown;
  /** What a `Parameter.binding` holds — `Ref<this>` in both standard families. */
  binding: unknown;
}

// --- References --------------------------------------------------------------
// Two orthogonal axes (unchanged from findmyprompt):
//   - a parameter's VALUE-TYPE      -> Parameter.kind (RefKind)
//   - where a value COMES FROM -> the Ref it is bound to

/**
 * A parameter's value type. KEPT INLINE rather than derived from the parameter's schema:
 * in the id family the `schema` member is itself an `Id`, so deriving the kind would need
 * a store round-trip before you know what you are resolving. With `blob` added it is a five-way
 * discriminator carrying real information.
 */
export type RefKind = "text" | "json" | "blob" | "prompt" | "function";

/** A leaf VALUE reference — the three data kinds a family hydrates. */
export type ValueRef<F extends RefFamily> = { text: F["text"] } | { json: F["json"] } | { blob: F["blob"] };

/**
 * An inline arrangement of refs: leaves are refs (`text`/`json`/`blob`/`result`) or PRIMITIVES,
 * branches are arrays or objects, nested arbitrarily. Resolving it yields the same shape with each
 * leaf replaced by its value — the run-specific collection/order of refs lives inline in an input
 * value.
 *
 * A primitive in a tree position is its own value (`{ refs: { greeting: "hi", n: 3 } }` is a literal
 * record). Only PRIMITIVES, never all of `JsonValue`: a literal object would be indistinguishable
 * from the `{ [key: string]: RefTree<F> }` branch, and the tree has no tag to tell them apart.
 */
export type RefTree<F extends RefFamily> =
  | ValueRef<F>
  | { result: F["result"] }
  | string
  | number
  | boolean
  | null
  | RefTree<F>[]
  | { [key: string]: RefTree<F> };

/** The named wrapper for an inline ref arrangement (was findmyprompt's `JsonRefs`). */
export type JsonRefs<F extends RefFamily> = { refs: RefTree<F> };

/** How the ID FAMILY names a producer operation: by content id. */
export type OperationRef = { promptId: Id } | { functionId: Id };

/**
 * A binding: a literal value, the memoized output of a past run, an inline arrangement of
 * refs, or a producer op applied at this site. The `parameters?` on a producer fill that op's
 * FREE input parameters (function application); they are part of the binding's identity.
 */
export type Ref<F extends RefFamily> =
  | ValueRef<F>
  | { result: F["result"] } // reuse an EXISTING OperationRecord's value
  | JsonRefs<F>
  | { op: F["op"]; parameters?: { [name: string]: Parameter<F> } }; // compute inline (a producer edge)

// --- Operations --------------------------------------------------------------
// There is NO Graph type. An Operation's `input` Parameters carry bindings, so the parameter
// tree IS the graph; a graph's external inputs are exactly the Parameters with no `binding`.

export interface Parameter<F extends RefFamily> {
  kind: RefKind;
  /** JSON Schema for the parameter (inline family: the schema document; id family: an Id). */
  schema?: F["schema"];
  /** Absent = free/external input parameter. */
  binding?: F["binding"];
  /**
   * Positional sort key for bare/tuple ingestion: the i-th positional value fills the parameter
   * with the i-th smallest `index`. Per op, all-set-or-all-unset and distinct. Wiring, not type.
   */
  index?: number;
}
// A `Parameter` carries no name: the NAME of a parameter lives in its container — a key in the
// `input` map, or `NamedParameter.name` for a standalone `output`.
//
// When `binding` is a producer edge, `kind` decides how it's used:
//   - kind in {text, json, blob}  -> RUN the producer; its `output` fills this parameter.
//   - kind in {prompt, function}  -> pass the op DEFINITION itself as the value (higher-order).
// An explicitly-passed value for a parameter OVERRIDES its binding; a run producer is filled from
// the CONSUMING op's inputs by name.

/** A `Parameter` plus the NAME it is bound under — used where one stands alone (an op's `output`). */
export interface NamedParameter<F extends RefFamily> extends Parameter<F> {
  name: string;
}

/**
 * One structured LLM call — exactly findmyprompt's semantics; LLM-only. Its `config` field is the
 * `LlmConfiguration` surface, but this SHAPE imports nothing from `@declarative-ai/llm`: it is text
 * fields + a config field + a schema. The `PromptOp → LlmCallDefinition` LOWERING is llm-specific and
 * lives in `@declarative-ai/promptop` (DESIGN §4.1).
 */
export interface PromptOp<F extends RefFamily> {
  kind: "prompt";
  system?: F["text"];
  user: F["text"];
  /** Typed by the `LlmConfiguration` schema. */
  config: F["json"];
  input: { [name: string]: Parameter<F> };
  /** `binding` never set on an output. */
  output: NamedParameter<F>;
}

/**
 * A registered function (sync or async) or, in the id family, a PARTIAL APPLICATION of
 * another op. Everything that isn't a bare LLM call folds in here: sub-workflows, composite
 * units, and delegated agent runtimes are registered (usually async) functions.
 */
export interface FunctionOp<F extends RefFamily> {
  kind: "function";
  /**
   * EITHER a registered function name OR (id family) another Operation's `Id`. Resolution
   * tries the function registry FIRST; on a miss the ref is read as an op id, making this op
   * a partial application: its `input` Parameters carry the fixed constants and pass the
   * remaining free parameters through, by name.
   */
  functionRef: string;
  input: { [name: string]: Parameter<F> };
  output: NamedParameter<F>;
}

/** The base union — findmyprompt's exact shape. */
export type Operation<F extends RefFamily> = PromptOp<F> | FunctionOp<F>;

// --- The two standard families -----------------------------------------------

/** The findmyprompt substrate: every leaf is a content id into an artifact store. */
export interface IdFamily {
  text: Id; // -> Text artifact
  json: Id; // -> Json artifact
  blob: Id; // -> Blob artifact, resolved by the SAME store as text/json
  schema: Id; // -> Json artifact holding a JSON Schema
  result: Id; // -> OperationRecord
  op: OperationRef;
  binding: Ref<IdFamily>;
}

/** The inline-family instantiations, NAMED — the payloads leaf executors and lowerings speak. A
 *  consumer that only ever runs resolved ops (a call seam, a WDK step payload) says `InlinePromptOp`
 *  instead of respelling the instantiation at every site. */
export type InlineOperation = Operation<InlineFamily>;
export type InlinePromptOp = PromptOp<InlineFamily>;
export type InlineFunctionOp = FunctionOp<InlineFamily>;

/** The hw/JaiRA substrate: every leaf is the value itself; producers are embedded ops or a
 *  DECLARED CHILD's local key (the local-name analog of an op id). */
export interface InlineFamily {
  text: string;
  json: JsonValue; // the value itself — NOT `unknown` (§2.2)
  blob: Bytes; // the bytes themselves, or a stream over them
  schema: JsonSchema; // the schema document itself
  // The resolved record itself. `R`/`M` are pinned here because a FAMILY names concrete types — that
  // is what a family IS — which is also why `ops` keeps a two-field `Metrics` floor: not for the
  // record's convenience, but because `InlineFamily` cannot leave a parameter open.
  result: OperationRecord<InlineFamily, ResolvedValue, Metrics>;
  op: Operation<InlineFamily> | string;
  binding: Ref<InlineFamily>;
}

// --- Parameter kinds ---------------------------------------------------------------

/**
 * The kind a parameter carries when the author didn't say, derived from its schema:
 *  - a string-typed parameter declaring BINARY/DOCUMENT content keywords is `blob` — JSON Schema's own
 *    `contentEncoding`/`contentMediaType`, never a bespoke marker (DESIGN §3.7);
 *  - any other string-typed parameter is `text`;
 *  - everything else is `json`.
 *
 * This is the one place the blob/text/json split is decided, so a parameter authored as
 * `{ type: "string", contentMediaType: "image/png", contentEncoding: "base64" }` and one authored as
 * `{ type: "string" }` cannot drift apart between the loader, the checker, and the engine.
 */
export function kindFor(schema: SchemaDocument | undefined): RefKind {
  if (schema?.type !== "string") return "json";
  return schema.contentEncoding !== undefined || schema.contentMediaType !== undefined ? "blob" : "text";
}

// --- Signatures --------------------------------------------------------------

/**
 * The I/O contract an op implements: one input type `I`, one output type `O`. Multi-field inputs are
 * just an `I` that is a json object.
 *
 * Generic in the ref family, like everything else in this model. `a second name for `Parameter`` is gone: it was
 * `Parameter<InlineFamily>` minus `binding` under a second name — which is why the schema bridge
 * already CAST a `Parameter` to it to read `.kind`. A signature parameter is a parameter that happens never
 * to be bound.
 *
 * A signature's schemas are read through the family's deref (`slotSchema(parameter, deref)`) rather than
 * assumed inline. The inline family's deref is the identity, so the "read it straight off the value"
 * property holds exactly where it held before — it is now a property of the FAMILY rather than an
 * assumption baked into the type.
 */
export interface Signature<F extends RefFamily> {
  input: Parameter<F>;
  output: NamedParameter<F>;
}

// --- Execution provenance (record shapes; execution itself is NOT here) -------

/** A FREE parameter filled at run time: a literal value ref, a prior run's output, an inline
 *  arrangement of refs, or a PLAIN STRING (an unhashed cache-policy token, read as a scope id). */
export interface ResolvedInput<F extends RefFamily> {
  name: string;
  value: ValueRef<F> | { result: F["result"] } | JsonRefs<F> | string;
  /** When `false`, excluded from the record's content hash (may grow/mutate after the row
   *  opens, or is run-level cache policy). Default (omitted/true) → part of the identity. */
  hashed?: boolean;
}

/**
 * The record of ONE operation having run — what you STORE, as distinct from what execution RETURNS
 * (`@declarative-ai/exec`'s `ExecResult<O>`) and what one provider call yields
 * (`@declarative-ai/llm`'s `LlmCallResult<T>`). The three are NOT duplicates: this one is the record of
 * a FILLED output parameter, so it adds `id`, `source`, `inputs`, and `createdBy`. In the id family
 * `id` is the content hash of (source, inputs); inline records carry no id unless a store interns them.
 *
 * Was `GenerationResult`, which read as an LLM record. It is not one: a pure function op's run records
 * here identically.
 *
 * Generic in what the call PRODUCED (`R`) and what it cost (`M`), which is what stopped it duplicating
 * two things it should have been composing with:
 *
 *  - `result` was `ValueResult<F> = ValueRef<F> & { error?: Failure }` — value-or-failure re-derived in
 *    kind-tagged form. It is now the SAME {@link Result} envelope the live call returned. The lost tag
 *    is not lost information: a stored value's kind is already on the producing op's output
 *    `Parameter.kind`, and this record names its `source` op, so the tag was a third copy.
 *  - `thinking?: ValueRef<F>` was an ad-hoc, lossy projection of a model's output sitting beside the
 *    value. With `R` generic it disappears into the payload: a prompt op's record is
 *    `OperationRecord<InlineFamily, LlmOutput<T>, LlmMetrics>`, so the trace is stored as part of what
 *    the call produced rather than as a parallel field that could drift from it.
 *
 * Serialization note: `id`, `source`, `inputs`, `createdBy`, and `metrics` are unchanged from
 * findmyprompt's row; `result` changes from `{json: x, error?}` to `{value: x, error?}`.
 */
export interface OperationRecord<F extends RefFamily, R, M> {
  id?: Id;
  /** Which op produced this. */
  source: F["op"];
  /** The op's FREE parameters, filled at run time. */
  inputs: ResolvedInput<F>[];
  /** What the call produced, or why it failed — the same envelope execution returned. */
  result: Result<R, Failure>;
  metrics: M;
  /** The principals who already own (were charged for) this record — memo-billing support. */
  createdBy?: string[];
}
