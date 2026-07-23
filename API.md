# API Reference

The complete public API of the `declarative-ai` packages, package by package, with intended usage. This is
the reference companion to the narrative docs:

- **[DESIGN.md](DESIGN.md)** — the architecture and the settled declarative model (read §1 first: the three
  layers, declare/plan/execute, composition, resolution, sessions; §2 for the package graph, §5.1 for the
  runtime/tool/permission model).
- **[SPEC.md](SPEC.md)** — the hierarchical-workflow formalism (normative for `@declarative-ai/hw`).
- **[README.md](README.md)** — runnable, copy-pasteable examples for each capability.

This file is the *what-and-why of every export*; the README is the *how-to*. Where an example would repeat
the README, this doc links to it instead.

## Contents

- [Orientation](#orientation)
- [`@declarative-ai/json`](#declarative-aijson)
  - [The JSON vocabulary](#the-json-vocabulary)
  - [Codecs and type names](#codecs-and-type-names)
  - [Schema templates, inference, and select-typing](#schema-templates-inference-and-select-typing)
  - [Hashing & identity](#hashing--identity)
  - [Error classification](#error-classification)
- [`@declarative-ai/ops`](#declarative-aiops)
  - [The operation model](#the-operation-model)
  - [The function registry](#the-function-registry)
  - [The `Signature` ⇄ schema bridge](#the-signature--schema-bridge)
  - [Op metadata](#op-metadata)
  - [The typed layer](#the-typed-layer)
- [`@declarative-ai/exec`](#declarative-aiexec)
  - [The execution contract](#the-execution-contract)
  - [The dispatching executor](#the-dispatching-executor)
  - [Composition](#composition)
  - [Generic wrappers](#generic-wrappers)
  - [Memoization](#memoization)
  - [Hydration — the family transition](#hydration--the-family-transition)
  - [Injected service seams](#injected-service-seams)
  - [The capability registry](#the-capability-registry)
  - [Sessions](#sessions)
  - [Handle scaffolding](#handle-scaffolding)
  - [Rate limiting](#rate-limiting)
  - [Retry](#retry)
  - [Deadline arithmetic](#deadline-arithmetic)
- [`@declarative-ai/validate`](#declarative-aivalidate)
  - [Schema subtyping](#schema-subtyping)
  - [The binding checker](#the-binding-checker)
  - [Schema validation](#schema-validation)
- [`@declarative-ai/permissions`](#declarative-aipermissions)
- [`@declarative-ai/llm`](#declarative-aillm)
  - [The LLM declaration](#the-llm-declaration)
  - [Configuration resolution](#configuration-resolution)
  - [File inputs and generated files](#file-inputs-and-generated-files)
  - [One-shot calls](#one-shot-calls)
  - [`plan` — the dry run](#plan--the-dry-run)
  - [Model router](#model-router)
  - [Model catalog](#model-catalog)
  - [Cost estimation](#cost-estimation)
  - [Schema & provider adaptation](#schema--provider-adaptation)
- [`@declarative-ai/promptop`](#declarative-aipromptop)
  - [The lowering](#the-lowering)
  - [The prompt executor](#the-prompt-executor)
  - [LLM-aware wrappers](#llm-aware-wrappers)
- [`@declarative-ai/tools`](#declarative-aitools)
- [`@declarative-ai/hw`](#declarative-aihw)
  - [The workflow executor](#the-workflow-executor)
  - [Bundles & identity](#bundles--identity)
  - [Operation dispatch & ports](#operation-dispatch--ports)
  - [The engine (lower level)](#the-engine-lower-level)
  - [State-file format types](#state-file-format-types)
  - [Binding desugaring](#binding-desugaring)
  - [Expression language & inference](#expression-language--inference)
- [`@declarative-ai/agents-api`](#declarative-aiagents-api)
- [`@declarative-ai/agents-cli`](#declarative-aiagents-cli)

---

## Orientation

**Packages.** Eleven, in the graph of [DESIGN.md](DESIGN.md) §2:

```text
                          json
                     ┌──────┴──────┐
                    llm           ops
                     │             │
                     │            exec
                     │   ┌─────┬───┴────┬────────────┬────────────┐
                     └ promptop │   validate    permissions   agents-api
                             tools      └───────┬──────┘            │
                                                hw             agents-cli
```

`json` is the floor and **nothing in it can be declined** — its only dependencies are `canonicalize` and
`@noble/hashes`, no `node:*` imports, edge-safe. `ops` adds `json-schema-to-ts`, types-only, and
**re-exports the whole of `json`**, so a consumer that speaks ops needs one import; `exec` in turn
re-exports the whole of `ops`. `llm` pulls in the `ai` SDK + provider packages and is otherwise
independent — `npm i @declarative-ai/llm` installs **no ajv**, and a structured call runs with `json + llm`
and nothing else (both asserted by test). `validate` is the only package carrying a heavy dependency;
`hw` inherits ajv through it. **`hw` does not depend on `promptop`** — it takes the prompt executor as a
plain `Executor`, so the AI SDK stays out of the workflow engine's graph. `agents-api` has an optional
peer on the Claude Agent SDK. Consumed as TypeScript source (`exports` → `src/index.ts`); consumers
bundle.

**Operations are the spine.** What flows *through* the execution contract is an
`Operation<F>` — a `PromptOp` (one structured LLM call) or a `FunctionOp` (a registered function). There is
no third kind: sub-workflows, composite units, and delegated agent runtimes are all registry entries, so a
delegated agent is a `FunctionOp` naming one, not a special payload type. Every
slot carries a JSON Schema and every wire is a typed binding, so an op graph is checkable as data.

**One execution seam.** `Executor.start(op, ctx)` is the whole contract. There is no separate execution
spec, no unit-kind taxonomy, and no prompt-specific runner interface: dispatch is
`op.kind === "prompt"` → the prompt executor, `"function"` → a registry lookup by `functionRef`. Because
that is one ordinary seam, wrapper composition reaches function ops as well as prompt ops.

**The three layers** (DESIGN §1.2). Everything below is one of:

- a **declaration** — pure serializable data (`Operation`, `LlmConfiguration`/`LlmCallDefinition`, tool
  *declarations*, schema), content-hashable, its hash is its identity;
- an **environment** — injected, secret-bearing, non-serializable seams (`LlmCallEnvironment` /
  `ExecServices`): model router, validator, session store, tool *executors*, clock. Every
  seam optional;
- **resolved transport** — internal only (`GenerateEnvironment`), never exported for use.

**Two ways to run a call.** The ergonomic direct path (`executeRequest` / `executeLlmCall`, returns an
`LlmCallResult<T>`, needs only `json + llm`), and the full contract path (`Executor.start(op, ctx)`, returns
an `ExecHandle` with an event stream, cancel, and a never-rejecting `result` promise). The contract path
is what you compose wrappers and registries around.

**Errors are data.** A function impl RESOLVES a `Result` — `{ value } | { error: Failure; value? }`, plus an
optional `metrics` report — rather than throwing, so a classified failure (a 429, a network blip) reaches
the retry machinery instead of being reconstructed from `err.name`.

**`unknown` is not part of the vocabulary.** It is legal in exactly two positions:
generic *bounds* (the `RefFamily` slots) and the interior of a boundary parser (wire responses,
`JSON.parse`) that immediately narrows it. Every exported type uses `JsonValue`, `JsonSchema`, a generic
parameter, or a precise domain type — including generic *defaults* (`ExecResult<O = ResolvedValue>`, not
`= unknown`).

**Model ids are route-prefixed** `{route}/{model}` — `anthropic/…` (native Anthropic) or `openrouter/…`
(everything else). A bare id is a fail-fast error; routing is never guessed.

**Results never throw for a unit failure.** Both `LlmCallResult` and `ExecResult` are always returned,
best-effort populated; a failure is on `.error` (and `isOk(r)` narrows to the success branch), and a
partial `value` is kept on the failure branch so it is diagnosable rather than empty.

---

## `@declarative-ai/json`

The bottom of the graph: the JSON value/document vocabulary, the wire projection of a decoded type
(`Jsonify`) and its codec/type-name registry, pure schema transforms (templates, inference, JSONPath
select-typing), canonical serialization + hashing, and the classified error + telemetry vocabulary all
three result types share. It knows nothing about operations, execution, providers, or validation.

### The JSON vocabulary

Six types, and the distinctions between them are load-bearing. Source: `json.ts`.

```ts
type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
type ReadonlyJsonValue = string | number | boolean | null
                       | readonly ReadonlyJsonValue[] | { readonly [key: string]: ReadonlyJsonValue };

type SchemaDocument = { readonly [key: string]: ReadonlyJsonValue };
type MutableSchema  = { [key: string]: JsonValue };
type JsonSchema<T = JsonValue> = SchemaDocument & { readonly [SchemaOutputType]?: T };   // phantom T

type Serializable = string | number | boolean | null | undefined
                  | readonly Serializable[] | { readonly [key: string]: Serializable };
type SerializableFields<T> = { [K in keyof T]: Serializable };

type Jsonify<T>;                                               // T's JSON projection (see below)
type SchemaOutput<S>;                                          // JsonSchema<T> → T (else JsonValue)
function typedSchema<T>(schema: SchemaDocument): JsonSchema<T>; // brand — identity at runtime
function collectRefs(node: unknown, out?: Set<string>): Set<string>;  // every `$ref` target reachable

interface ValidationResult { ok: boolean; errors?: string }

interface OutputValidator {   // the MINIMAL structural validation seam, declared once at the bottom.
  // MAY resolve async: a store-backed validator (content-addressed $refs) has reads to do — the
  // boundary consumers (executeLlmCall) await it, which costs a sync implementation nothing.
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult | Promise<ValidationResult>;
}
interface SyncOutputValidator {  // the SYNC refinement for mid-walk consumers (hw slot validation,
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult;  // the MCP input gate)
}
function syncOnly(v: OutputValidator): SyncOutputValidator;  // FAIL-CLOSED narrowing (a suspender is refused)
```

| Type | Use it when |
| --- | --- |
| `JsonValue` | a program **reads** a value: an op's `json` leaf, a resolved input, an `ExecResult`'s `value`, a tool's input/output. The default everywhere `unknown` used to sit. |
| `ReadonlyJsonValue` | a document is **consumed, not built** — the index type `SchemaDocument` is written in terms of, so `as const` literals are assignable without a cast. |
| `SchemaDocument` | schema-**reading** code: transformers, the subtype checker, `OutputValidator.validateValue`. It makes no claim about what the schema validates, and every `JsonSchema<T>` is assignable to it, so a typed schema flows into untyped machinery cast-free. |
| `MutableSchema` | schema-**writing** code: provider adaptation, union flattening, `$ref` inlining. Structurally both a `SchemaDocument` and a `JsonValue`; it differs only in dropping the `readonly` index so a transform can assign into it. |
| `JsonSchema<T>` | a schema that **carries its output type**. The phantom is a unique-symbol key — type-level only, never serialized, cannot collide with a schema keyword. It subsumes llm's old `Record<string, unknown> & { __out?: T }` trick. |
| `Serializable` / `SerializableFields<T>` | a **serialization edge** (`canonicalize`, `hashCanonical`). `Serializable` is `JsonValue` widened to tolerate `undefined` in optional members; `SerializableFields<T>` does the same check field-wise, which is needed because TypeScript grants an implicit index signature to type *aliases* but not to *interfaces* — so a nominal record like `LlmCallDefinition` fails a structural `Serializable` test even when every field passes. |

**The phantom carries the DECODED type; the DOCUMENT describes `Jsonify<T>`.**
`Jsonify<T>` is what `JSON.parse(JSON.stringify(value))` actually yields, at the type level: a type
declaring `toJSON()` projects to its return type, object members holding `undefined`/functions/symbols/
bigint are dropped and array elements holding them become `null`, tuples keep their arity, and optional/
readonly modifiers survive. Read `JsonSchema<T>` as "a schema *for* `T`, whose wire form is `Jsonify<T>`" —
the phantom deliberately does **not** carry `Jsonify<T>`, because TypeScript cannot infer backwards through
a conditional type and `JsonSchema<Jsonify<T>>` would make `T` uninferable at every call site. So compile
time works in `T` while the runtime check validates `Jsonify<T>`: you never ask a model to emit a
`DateTime`, you ask for an epoch, validate the epoch, and lift it with the type's registered codec.

`OutputValidator` is declared here — three lines, at the bottom — so every layer that needs a boundary
check names the SAME interface and none of them learns about ajv. `@declarative-ai/validate`'s
`SchemaValidator` implements it; `exec`, `llm`, and `hw` consume it.

`typedSchema<T>(doc)` brands a plain document by hand; the [typed layer](#the-typed-layer)'s `InferSchema`
derives `T` from an `as const` literal instead, which is the preferred route when the schema is a literal.

### Codecs and type names

Encode is derivable from a type (the `toJSON()` contract). **Decode is not derivable from any type** — it
needs a runtime function. And a codec holds closures, so it cannot be content-addressed or stored, while a
schema can. So a schema NAMES its type and the codec is resolved by that name, globally, once per type.
Source: `codec.ts`.

```ts
const X_TYPE = "x-type";                    // the schema keyword that names a value's DECODED type

interface TypeRegistry {}                   // AUGMENTABLE — the owning package declares its types
type TypeName    = keyof TypeRegistry & string;
type DecodedOf<N extends TypeName>;         // the decoded type registered under N
type WireOf<N extends TypeName>;            // its `json` member, else Jsonify of the decoded type

interface Codec<T = JsonValue, J = Jsonify<T>> {
  schema?: SchemaDocument;                  // the canonical document for this type, incl. its `x-type`
  encode(value: T): J;
  decode(json: J): T;
}

class CodecRegistry {
  register<N extends TypeName>(name: N, codec: Codec<DecodedOf<N>, WireOf<N>>): this;  // throws on a conflict
  has(name: string): boolean;
  get(name: string): Codec<JsonValue, JsonValue> | undefined;
  names(): string[];
}
const codecs: CodecRegistry;                // the process-global registry

type Decoded<S>;                            // type-level: walk a literal document, lift `x-type` nodes
function typeNameOf(schema: SchemaDocument | undefined): string | undefined;
function decodeWithSchema(schema: SchemaDocument | undefined, json: JsonValue): unknown;
function encodeWithSchema(schema: SchemaDocument | undefined, value: unknown): JsonValue;
```

```jsonc
{ "type": "number", "x-type": "DateTime" }
```

```ts
codecs.register("DateTime", { encode: (d) => d.getTime(), decode: (n) => new Date(n) });

declare module "@declarative-ai/json" {
  interface TypeRegistry { DateTime: { value: Date; json: number } }
}
```

Registration is **once per TYPE**, not per op, per parameter, or per use — which is what makes rich types
work in the id family, where the op is stored and the closures cannot be. Re-registering a name with a
different codec **throws**, because two codecs for one name would make a stored schema mean two different
things depending on load order. A slot naming an unregistered type stays raw JSON.

`decodeWithSchema`/`encodeWithSchema` are the runtime twins of `Decoded`/`Jsonify`: walk a value against a
document, applying each named node's codec, passing everything else through structurally. This is the
decode step that belongs at a call boundary — what `postProcess` on a structured call becomes once schemas
can name rich types.

**`x-type` is the one `x-` keyword with validation semantics.** Every other `x-` keyword carries
application metadata and is ignored by the subtype checker; this one is CONSTRAINING on the consumer side,
so a slot declaring a type name accepts only producers declaring the same name — a `DateTime` producer can
never silently hand an encoded epoch to a bare-number slot, or vice versa.

`Decoded<S>` is deliberately modest, in the same way `InferSchema` is: it interprets named leaves, objects
with `properties`/`required`, arrays with `items`, `const`/`enum`, and the primitive `type`s, degrading to
`JsonValue` — surfaced, never a silent `unknown`.

### Schema templates, inference, and select-typing

Three pure schema/data transforms with no model dependency, which is what keeps them at the bottom.
Sources: `template.ts`, `infer.ts`, `selectType.ts`.

```ts
// template.ts — parametric polymorphism for schemas
const PARAM_KEY = "$param";
function isParamHole(node: unknown): node is { $param: string };
function collectParams(node: unknown, out?: Set<string>): Set<string>;
function applyTemplate(template: SchemaDocument, bindings: Record<string, SchemaDocument>,
                       opts?: { partial?: boolean }): SchemaDocument;
function collapseHoles(template: SchemaDocument): SchemaDocument;      // unresolved holes → `{}` ("any")
function resolveTypes(template: SchemaDocument, typeMap: Record<string, SchemaDocument>): SchemaDocument;

// infer.ts — a schema from DATA
type JoinPolicy = "strict" | "widen";
function inferValueSchema(value: unknown): JsonSchema | undefined;
function joinSchemas(a, b, policy?: JoinPolicy): JsonSchema | undefined;
function inferFromValues(values: readonly unknown[], policy?: JoinPolicy): JsonSchema | undefined;

// selectType.ts — the schema→schema transform behind `select`/`project`
type PathSegment = { kind: "prop"; name: string } | { kind: "wildcard" } | { kind: "index"; index: number };
function parseJsonPath(path: string): PathSegment[];
function resolveSelectOutputSchema(inputSchema: SchemaDocument, path: string): JsonSchema;
```

A schema document may contain **parameter holes** — `{ "$param": "output" }` — standing for a schema
supplied later. `applyTemplate` fills them; `resolveTypes` fills what it can and collapses the rest to the
universal `{}`. Substitution is **single-pass and verbatim**: a binding that itself contains holes is
spliced in unchanged, which makes application total and cycle-free. `$param` and `x-type` are the same
SHAPE and coexist — `$param` binds type *variables* per call, `x-type` binds type *names* globally.

`infer` is "as wide as possible within the kind": two strings join to `string`, not an enum of the
literals, so the inferred type stays a large space an author can narrow later. `null`/`undefined` carry no
type information and unify with anything.

`resolveSelectOutputSchema` walks the input schema in parallel with a static JSONPath to *compute* the
output schema: `.prop` descends `properties`, `[*]` requires an array, descends `items`, and re-wraps the
result once per wildcard crossed, `[n]` descends to a single element. No JSONPath library does this
schema→schema transform, which is why it is here.

### Hashing & identity

RFC 8785 (JCS) canonicalization + sha256. Pure JS (no `node:crypto`) so hashing runs in any runtime — Node,
edge, and the Vercel Workflow runtime where Node modules are forbidden; digests are byte-identical to
`node:crypto`'s. Source: `hashing.ts`.

```ts
function canonicalize<T extends Serializable | SerializableFields<T>>(value: T): string;   // RFC 8785 (JCS)
function sha256Hex(payload: string): string;
function hashCanonical<T extends Serializable | SerializableFields<T>>(value: T): string;  // content identity
```

`hashCanonical` is a declaration's content identity; the `Serializable | SerializableFields<T>` bound is
the serialization edge (see [The JSON vocabulary](#the-json-vocabulary)) — it admits
precise domain records with optional fields while rejecting the values JCS would actually throw on.
`canonicalize` **throws** on values JCS cannot serialize (`undefined`, functions, bigint, circular) rather
than silently hashing nothing.

These live at the bottom because canonical JSON serialization is definitionally a json concern and its
consumers reach well past memoization: the hw loader's snapshot hash, llm's schema cache keys, the ajv
validator's inline-schema cache. With hashing here, [memoization](#memoization) is a few dozen
dependency-free lines in `exec`.

### Error classification

One shared error vocabulary every layer reasons over — `Failure` and `ErrorClass`, plus the
`Result`/`ResultWithMetrics` envelope they ride in — lives at the bottom so all three result types (exec's
`ExecResult`, llm's `LlmCallResult`, ops' `FunctionResult`) share one shape and one failure vocabulary
instead of each re-deriving it. Sources: `classification.ts`, `failure.ts`, `result.ts`, `encodedError.ts`.

```ts
type ErrorClass = "network-retriable" | "api-retriable" | "permanent"
                | "deadline" | "out-of-credits" | "canceled" | "policy-denied";
type ClassifiedErrorClass = "network-retriable" | "api-retriable" | "permanent";  // classifyError's subset

function classifyError(err: unknown): ErrorClass;   // walks the cause chain; default → permanent
```

| Function | Returns |
| --- | --- |
| `isNetworkError(err)` | `boolean` — connection reset/refused, DNS, undici socket. |
| `isRateLimit(err)` | `boolean` — HTTP 429. |
| `isTimeoutOrAbort(err)` | `boolean` — timeout / abort. |
| `retryAfterMs(err)` | `number \| undefined` — parsed `retry-after`. |
| `extractRateLimitInfo(err)` | `Record<string, string> \| undefined` — rate-limit headers. |
| `describeError(err)` | `string` — human-readable rendering. |
| `encodeError(e: EncodedError)` / `decodeError(text)` | round-trip an error to/from a stored string. |

`classifyError` reads an explicit retryable flag first (`isRetryable`/`retryable`), then HTTP 429/5xx, then
timeout/abort, then low-level network codes; unknown errors are **permanent** (never blindly retried). The
control-flow classes (`deadline`, `out-of-credits`, `canceled`, `policy-denied`) are carried on
`Failure` but never *produced* by `classifyError` — executors set them from their own control flow.

Also here: the value-or-failure envelope itself — `Result<S, E>`, `ResultWithMetrics<S, E, M>`, and `isOk`
— declared once so every layer's result is the same shape with its own payload, failure, and metrics
filled in. The metrics *records* live with whatever they measure, not here: the `Metrics` floor in
[`@declarative-ai/ops`](#declarative-aiops), `ExecMetrics` in [`exec`](#declarative-aiexec), and
`LlmMetrics`/`TokenCounts` (plus the `ReasoningSegment`/`ToolCall`/`ToolResult` trace) in
[`@declarative-ai/llm`](#declarative-aillm).

---

## `@declarative-ai/ops`

The typed operation spine: the generic `PromptOp`/`FunctionOp` model, the ONE function registry, the
`Signature` ⇄ schema bridge, op metadata, and a compile-time typed layer over all of it. Execution-free —
there is no `runOperation` here; ops are what *flows through* an executor, not a replacement for one. **It
re-exports the whole of [`@declarative-ai/json`](#declarative-aijson)**, so `import { JsonSchema } from
"@declarative-ai/ops"` is the normal way to reach that vocabulary.

### The operation model

The generic op vocabulary, ported from findmyprompt and made generic over the **reference substrate**.
Source: `model.ts`.

#### Ref families — the one idea to internalize

findmyprompt's op leaves are content ids into artifact stores; hw/JaiRA's leaves are inline values. These
are not two models — they are one model parameterized by a **ref family**, the seven slot types every op
field is built from:

```ts
interface RefFamily {
  text: unknown;      // what a text-valued leaf holds (a prompt body, an error)
  json: unknown;      // what a json-valued leaf holds (a config, a value)
  blob: unknown;      // what a BINARY leaf holds (an image, a pdf, a produced artifact)
  schema: unknown;    // how a slot carries its JSON Schema (an Id, or the document)
  result: unknown;    // what a result leaf holds (an id of, or an inline, OperationRecord)
  op: unknown;        // how a producer NAMES its operation (by id, embedded, or a local key)
  binding: unknown;   // what a Parameter's `binding` holds — Ref<this> in both standard families
}
```

Those `unknown`s are **generic bounds, not value types** — every instantiation pins them, which is why
this is the one sanctioned `unknown` position in the workspace. Property names are uniform across
families; only the *types* the properties hold change:

```ts
interface IdFamily {                                interface InlineFamily {
  text: Id;                                           text: string;
  json: Id;                                           json: JsonValue;
  blob: Id;                                           blob: Bytes;         // Uint8Array | ByteStream
  schema: Id;                                         schema: JsonSchema;
  result: Id;                                         result: OperationRecord<InlineFamily, ResolvedValue, Metrics>;
  op: OperationRef;                                   op: Operation<InlineFamily> | string;
  binding: Ref<IdFamily>;                             binding: Ref<InlineFamily>;
}                                                   }

type Id = string;                                       // content id or UUID
type OperationRef = { promptId: Id } | { functionId: Id };

// The inline-family instantiations, NAMED — what leaf executors and lowerings speak:
type InlineOperation = Operation<InlineFamily>;
type InlinePromptOp = PromptOp<InlineFamily>;
type InlineFunctionOp = FunctionOp<InlineFamily>;

type Bytes = Uint8Array | ByteStream;                   // ByteStream is the WHATWG ReadableStream shape,
                                                        // declared STRUCTURALLY so json/ops need no DOM types
```

**`blob` is a leaf KIND alongside `text` and `json`.** Binary data is a leaf value, so hydration is the
family's business — an id-family blob resolves through the SAME artifact store as text and json, and an
inline-family blob IS the bytes (or a stream over them). There is no separate blob store to inject and no
parallel artifact channel: a produced artifact is simply a blob-kind output slot. `kindFor` derives
`"blob"` from JSON Schema's own binary keywords (`contentEncoding`/`contentMediaType`), never a bespoke
marker.

Streaming is an input/output optimization, **never part of a definition**: an authored document is JSON, so
a stream only ever appears in a runtime value. **Materializing a stream into bytes is not implemented.**
Three paths need bytes — hashing for a memo key, fan-out to two consumers, storing a result inline — and
each raises with the remedy rather than draining the stream, so a caller holding a live stream materializes
it first. See [DESIGN.md](DESIGN.md) §10.1.

The `op` slot is the whole difference between the two: findmyprompt names a producer by content id, hw
names a declared child by its **local key**. hw therefore needs no widened binding union — wiring a child's
output is a *producer edge*, exactly how findmyprompt wires op to op.

#### References

```ts
type RefKind = "text" | "json" | "blob" | "prompt" | "function";

type ValueRef<F> = { text: F["text"] } | { json: F["json"] } | { blob: F["blob"] };
type RefTree<F>  = ValueRef<F> | { result: F["result"] } | RefTree<F>[] | { [key: string]: RefTree<F> };
type JsonRefs<F> = { refs: RefTree<F> };

type Ref<F extends RefFamily> =
  | ValueRef<F>                                              // a literal
  | { result: F["result"] }                                  // reuse an EXISTING OperationRecord
  | JsonRefs<F>                                              // an inline arrangement of refs
  | { op: F["op"]; parameters?: { [name: string]: Parameter<F> } };   // a producer edge
```

Two orthogonal axes: a slot's **value type** is its `Parameter.kind`; where the value **comes from** is the
`Ref` it is bound to. `{ result }` reuses an `OperationRecord` that already exists; `{ op }` is an edge the engine may
still have to *run*. `parameters?` on a producer edge fill that op's free slots (function application) and
are part of the binding's identity.

#### Parameters and operations

```ts
interface Parameter<F extends RefFamily> {
  kind: RefKind;
  schema?: F["schema"];   // JSON Schema for the slot
  binding?: F["binding"]; // ABSENT = a free/external input slot
  index?: number;         // positional sort key for tuple ingestion; per op all-set-or-all-unset
}
interface NamedParameter<F> extends Parameter<F> { name: string; }

interface PromptOp<F extends RefFamily> {
  kind: "prompt";
  system?: F["text"]; user: F["text"];
  config: F["json"];                              // the LlmConfiguration surface
  input: { [name: string]: Parameter<F> };
  output: NamedParameter<F>;                      // `binding` never set on an output
}
interface FunctionOp<F extends RefFamily> {
  kind: "function";
  functionRef: string;                            // a registry name, or (id family) another op's Id
  input: { [name: string]: Parameter<F> };
  output: NamedParameter<F>;
}
type Operation<F extends RefFamily> = PromptOp<F> | FunctionOp<F>;
```

**There is no `Graph` type.** An op's `input` parameters carry bindings, so the parameter tree *is* the
graph, and a graph's external inputs are exactly the parameters with no `binding`. A parameter carries no
name — the name lives in its container (a key in `input`, or `NamedParameter.name` for a standalone
output).

When a binding is a producer edge, `kind` decides how it is used: `text`/`json`/`blob` **run** the producer
and fill the slot with its output; `prompt`/`function` pass the op *definition* itself (higher-order). An
explicitly-passed value overrides a binding.

`FunctionOp.functionRef` resolves against the function registry **first**; in the id family a miss is read
as an op id, making the op a *partial application* — its parameters fix some constants and pass the
remaining free slots through by name.

```ts
function kindFor(schema: SchemaDocument | undefined): RefKind;
```

`kindFor` is the **one place the blob/text/json split is decided**, so the loader, the checker, and the
engine cannot drift: a string-typed slot declaring JSON Schema's own `contentEncoding`/`contentMediaType`
is `blob`, any other string-typed slot is `text`, everything else is `json`. There is no bespoke marker
keyword.

#### Signatures and provenance

```ts
interface SignatureSlot { kind: RefKind; schema?: JsonSchema; index?: number; }
interface Signature { input: SignatureSlot; output: SignatureSlot & { name: string }; }
```

A `Signature` is the I/O contract an op implements — one input type, one output type (multi-field inputs
are just an object-typed input). It carries its schemas **inline as values**, never as references, so
inference reads them straight off with no deref.

| Export | Purpose |
| --- | --- |
| `Metrics` | the metrics **floor** — `{ durationMs, startMs? }`, the only fields every measurement shares. A richer record (`ExecMetrics`, `LlmMetrics`) satisfies it structurally and adds its own; a consumer constrains `M extends Metrics` and stays ignorant of the rest. `MetricsAlgebra<M>` (`{ merge(a, b): M }`) is how two of a kind combine — registered by whatever produces `M`. |
| `PartialMetrics` | `Partial<Metrics>` — an in-flight row that exists before the work completes, so nothing is required yet. |
| `OperationRecord<F, R, M>` | the record of one op having **run** — what you **store**, as distinct from what execution *returns* (exec's `ExecResult`) and what one provider call yields (llm's `LlmCallResult`). Generic in what the call produced (`R`) and what it cost (`M`): `{ id?, source: F["op"], inputs: ResolvedInput<F>[], result: Result<R, Failure>, metrics, createdBy? }`. Its `result` is the **same `Result` envelope** the live call returned — a failed record still keeps its partial value, so it is diagnosable. A prompt op's record is `OperationRecord<InlineFamily, LlmOutput<T>, LlmMetrics>`, so the reasoning trace is part of the stored payload rather than a parallel `thinking` field that could drift from it. |
| `ResolvedInput<F>` | a free parameter filled at run time: `{ name, value, hashed? }`; `hashed: false` excludes it from the record's content hash. |

### The function registry

Where a `FunctionOp.functionRef` resolves. **One map of discriminated entries** — host code, sub-workflows,
and delegated runtime adapters alike — generic in the async context (`Ctx`) so `ops` need not import
`ExecServices`; findmyprompt instantiates `Ctx = RunCtx`, ai-exec `Ctx = ExecServices`. Source:
`registry.ts`.

```ts
type ResolvedValue  = JsonValue | Bytes | ResolvedValue[] | { [key: string]: ResolvedValue };
type FunctionInputs = Record<string, ResolvedValue>;

// The value-or-failure envelope from `json` — no metrics on it, and no defaults on S/E:
type Result<S, E> = { value: S } | { error: E; value?: S };   // success has NO `error` key
function isOk<S, E>(r: Result<S, E>): r is { value: S };
// A function op's result: that envelope with `E` pinned to the classified `Failure`, plus an OPTIONAL
// metrics report (most impls are pure glue with nothing to say):
type FunctionResult<O, M> = Result<O, Failure> & { metrics?: M };

type FunctionImpl<I = FunctionInputs, O = ResolvedValue, M = Metrics>            = (inputs: I) => FunctionResult<O, M>;
type AsyncFunctionImpl<I = FunctionInputs, O = ResolvedValue, M = Metrics, Ctx = unknown> =
  (inputs: I, ctx: Ctx) => Promise<FunctionResult<O, M>>;
interface AsyncFunctionOptions { stream?: boolean }

type RegisteredFunction<Ctx, M> =
  | { kind: "pure";    impl: FunctionImpl;             capabilities: PureCapabilities;    stream?: false;   description? }
  | { kind: "host";    impl: AsyncFunctionImpl<…,Ctx>; capabilities: HostCapabilities;    stream?: boolean; description? }
  | { kind: "runtime"; impl: AsyncFunctionImpl<…,Ctx>; capabilities: RuntimeCapabilities; stream?: boolean; description? };

/** The registry IS a Map from `functionRef` to entry. */
type FunctionRegistry<Ctx, M> = Map<string, RegisteredFunction<Ctx, M>>;

// Entry constructors — free functions, because building an entry is all the old `registerX` methods did:
function pureFunction<M>(impl, capabilities?: PureCapabilities): PureFunction<M>;
function hostFunction<Ctx, M>(impl, capabilities: HostCapabilities, opts?: AsyncFunctionOptions): HostFunction<Ctx, M>;
function runtimeFunction<Ctx, M>(impl, capabilities: RuntimeCapabilities, opts?: AsyncFunctionOptions): RuntimeFunction<Ctx, M>;

// The read-only VIEW a checker needs: validation never invokes an impl, so it must not have to name
// `Ctx` or `M` to ask whether a function is interactive. A registry is assignable to
// `ReadonlyMap<string, FunctionCapabilities>` and the `kind` discriminant survives the widening.
type FunctionCapabilities =
  | { kind: "pure"; capabilities: PureCapabilities }
  | { kind: "host"; capabilities: HostCapabilities }
  | { kind: "runtime"; capabilities: RuntimeCapabilities };

async function runFunction<Ctx, M>(entry, inputs: FunctionInputs, ctx: Ctx): Promise<FunctionResult<ResolvedValue, M>>;
function isStreaming(entry): boolean;
function liftThrowing<I, O, Ctx>(impl, context?: string);   // the documented `catch` fallback
function failureOf(e: unknown, context?: string): Failure;  // classify a thrown value
```

**The registry is a plain `Map`.** `get`/`has` are the map's own, `refs()` was `[...map.keys()]`, and the
`registerX` helpers only built a discriminated entry — so the entry constructors became free functions and
the container became the map it always was. `Map.set` returns the map, so chaining is unchanged:
`functions.set("summarize", pureFunction(impl)).set("ask", hostFunction(impl, HOST_CAPABILITIES))`.

`runFunction` **never throws.** The impl contract is that errors resolve as data, but nothing at
registration forces an impl through `liftThrowing` — so the `catch` fallback lives at the one place every
dispatch path goes through, rather than being a rule each caller has to remember.

A **`pure`** entry is deterministic glue (parse, combine, select) that runs inline — no model, no cost, no
ctx. A **`host`** entry is host code, including interactive UI. A **`runtime`** entry is a delegated agent
adapter, or anything else driving its own loop. `opts.stream` opts into two-phase in-flight visibility (a
partial record at open, filled on completion) plus spawn tracking; progressive output rides *that*, never
the return value.

**Capabilities are required and total per variant** — there is no "registered but uncharacterized" state,
so permission gating and search refusal read a definite value instead of falling through an `undefined`.
Registration always produces an entry, and the record an `Executor` advertises is the *same*
`RuntimeCapabilities` a `runtime` entry carries, so the two cannot drift.

```ts
interface PureCapabilities { memoizable: boolean }
interface HostCapabilities { interactive: boolean; readOnly: boolean; memoizable: boolean }
interface RuntimeCapabilities extends HostCapabilities {
  structuredOutput: boolean; mutatesWorkspace: boolean;
  policyEnforcement: "callback" | "config" | "none";
  sessionResume: boolean; streaming: boolean; runtime: "edge-safe" | "node";
}
type Capabilities = RuntimeCapabilities;          // what an Executor advertises — the same record

const PURE_CAPABILITIES: PureCapabilities;        // sensible totals for hand-registering
const HOST_CAPABILITIES: HostCapabilities;
const RUNTIME_CAPABILITIES: RuntimeCapabilities;
```

`Capabilities` is `RuntimeCapabilities` because an executor *is* what a `runtime` entry delegates to —
there is no separate executor capability record to drift.

**Errors are data.** An impl RESOLVES a `Result` rather than throwing, so a 429 raised inside a function
impl carries its classification to the retry machinery instead of being reconstructed from `err.name`
(which made every non-`AbortError` `permanent`). `metrics` is the impl's optional *report* of what the work
cost — most impls are pure glue with nothing to say, but a delegated agent spends real money inside its own
loop and is the only thing that knows how much.

```ts
function liftThrowing<I, O, Ctx>(impl: (inputs: I, ctx: Ctx) => O | Promise<O>, context?: string):
  (inputs: I, ctx: Ctx) => Promise<Result<O>>;
function failureOf(e: unknown, context?: string): Failure;
```

`liftThrowing` is the documented `catch` FALLBACK for impls that throw anyway, and `runFunction` applies the
same fallback at the one place every dispatch path goes through — so a throwing impl becomes a classified
failure rather than a rejection that escapes the caller's error handling. `runFunction` **never throws**.

**Consequence for validation.** hw's static validator wants to reject "an interactive function in a
search-only workflow" at authoring time, which means the checker reads the registry. Validation is
therefore a function of *(document, registry)*, not of the document alone. Deliberate, not accidental.

### The `Signature` ⇄ schema bridge

Ported from findmyprompt; it lands in `ops` rather than `json` because a `Signature` is part of the op
model, while the pure schema transforms it composes with (`applyTemplate`, `resolveTypes`) sit one layer
down. Source: `signatureSchema.ts`.

```ts
function slotSchema(slot: SignatureSlot): JsonSchema;            // inline schema, else the kind's broad type
function signatureInputSchema(signature: Signature): JsonSchema; // Schema<I>
function signatureOutputSchema(signature: Signature): JsonSchema; // Schema<O>
function bindSignatureTemplate(template: SchemaDocument, signature?: Signature): JsonSchema;
function asSignature(v: unknown): Signature | undefined;         // structural guard
const SIGNATURE_META_SCHEMA: JsonSchema;                         // the reflective meta-schema
```

`bindSignatureTemplate` builds the type map `{ input: Schema<I>, output: Schema<O> }` and resolves a
template against it — so a generic op computes its concrete output schema per call. With **no** signature
the map is empty, every variable is unbound and collapses to `{}` ("any"): the generic floor is *any*,
never an error.

`SIGNATURE_META_SCHEMA` is the witness that makes a signature a first-class typed datum: each slot's
`schema` is a `$param` hole (`input.schema` is the variable `I`, `output.schema` is `O`), so "a signature's
input/output type and its instances' input/output match" is a fact of the data rather than a render-time
transform.

### Op metadata

Annotations keyed **by** an op's identity, never **part of** it. Source: `metadata.ts`.

```ts
type OpRef = Id | object;                    // a content id, or the op OBJECT's identity
interface OpMetadata<V = JsonValue> {        // generic default is JsonValue, never unknown
  get(op: OpRef, key: string): V | undefined;
  set(op: OpRef, key: string, value: V): void;
}
class InMemoryOpMetadata<V = JsonValue> implements OpMetadata<V> {}   // Map over ids, WeakMap over op objects
```

This is findmyprompt's `ref_metadata` pattern as an interface (its table is the durable instantiation).
Typical keys: a resolved registry entry's capabilities (a cache, so permission gating doesn't re-look-up
per call), the checker's inferred schemas, provenance. Because metadata is keyed by identity and never part
of it, caching anything here **cannot change what an op is** — content ids and memo keys are unaffected.
The `WeakMap` half means inline-op annotations die with the op object.

### The typed layer

TS generics *over* the runtime schema typing, with zero runtime cost: the schemas stay the source of truth,
and this adds compile-time inference plus builders whose producer wiring the compiler checks. Source:
`typed.ts` (`json-schema-to-ts` is a types-only dependency).

```ts
interface Widened<T = JsonValue> { readonly [WidenedTag]?: T; }
type InferSchema<S>;    // FromSchema for compile-time literals; Widened where interpretation degrades
```

`InferSchema` interprets a schema document as a TypeScript type, stripping the `[x: string]: unknown` index
`FromSchema` adds to open object schemas. Where interpretation can't work — a `$ref` resolved at runtime
through a store, a schema built dynamically, or a composition past TypeScript's instantiation-depth ceiling
— it yields `Widened` rather than a silent `unknown`: the degradation is **named**, shows up in hovers, and
consuming it requires an explicit cast. Only the compile-time tier degrades; the pre-run schema checker
(where every schema is a concrete value) and the run-time boundary validation are unaffected.

**This layer is `InlineFamily`-only.** `FromSchema` requires a compile-time
literal and an `Id` is opaque to the compiler, so an id-family consumer gets runtime schema checking and
pre-run static validation, never compile-time inference. That is by design, but it is not "the typed layer
applies to both families".

#### Typed functions

```ts
interface FunctionDef<I = FunctionInputs, O = JsonValue, Ctx = void> {
  readonly name: string;
  readonly description?: string;            // surfaced when the def doubles as an agent tool
  readonly input: JsonSchema<I>;
  readonly output: JsonSchema<O>;
  readonly impl: (inputs: I, ctx: Ctx) => O | Promise<O>;
}

function defineFunction<const SI extends JSONSchema, const SO extends JSONSchema, Ctx = void>(spec: {
  name: string; description?: string; input: SI; output: SO;
  impl: (inputs: InferSchema<SI>, ctx: Ctx) => InferSchema<SO> | Promise<InferSchema<SO>>;
}): FunctionDef<InferSchema<SI>, InferSchema<SO>, Ctx>;

function registerFunctionDef<I, O, Ctx>(
  registry: FunctionRegistry<Ctx>,
  def: FunctionDef<I, O, Ctx> | FunctionDef<I, O, void>,
  opts?: { async?: false; capabilities?: PureCapabilities }
      | { async: true; capabilities: HostCapabilities; stream?: boolean },
): void;
```

The `impl`'s parameter types are **inferred from the input schema** and its return type is checked against
the output schema — one document, runtime truth and compile-time truth at once. `const` type parameters
make `as const` optional on the literals. `registerFunctionDef` also registers the plain string-keyed impl
so dynamic (`functionRef` by name) resolution keeps working: the typed handle is *additional*, never
required. A pure def (`Ctx = void`) registers as a `pure` entry; pass `opts.async` for a ctx-bearing one,
which registers as `host` with the capabilities you declare — registration **always** produces an entry.

A def's `impl` returns its value and may throw, while the registry's contract is a resolved `Result`. This
is exactly the seam `liftThrowing` covers: the exception becomes a classified failure, so a 429 raised
inside a def is `network-retriable` and the retry machinery can act on it. An impl that wants full control
resolves its own `Result` and registers directly on the registry.

#### Typed operations and builders

```ts
type TypedOperation<I, O, F extends RefFamily = InlineFamily> = Operation<F> & { /* phantom I, O */ };
type Producer<O, F extends RefFamily = InlineFamily> = Operation<F> & { /* phantom O */ };
type OperationOutput<T>;                    // extract a TypedOperation's phantom output type
type TypedBinding<T> = T | Producer<T>;     // a literal of that type, or a producer whose output matches

function free(kind?: RefKind, schema?: JsonSchema): Parameter<InlineFamily>;
function bound(value: TypedBinding<JsonValue>, kind?: RefKind, schema?: JsonSchema): Parameter<InlineFamily>;

function promptOp<const SO extends JSONSchema, I = Record<string, JsonValue>>(spec: {
  system?: string; user: string;
  config?: { [key: string]: JsonValue };
  input?: { [name: string]: Parameter<InlineFamily> };
  output: { name?: string; schema: SO };
}): TypedOperation<I, InferSchema<SO>> & PromptOp<InlineFamily>;

function functionOp<I extends Record<string, JsonValue>, O, Ctx>(
  def: FunctionDef<I, O, Ctx>,
  bindings?: { [K in keyof I]?: TypedBinding<I[K]> },
): TypedOperation<I, O> & FunctionOp<InlineFamily>;

function runtimeOp<const SO extends JSONSchema = { readonly type: "string" }>(spec: {
  runtime: string;                          // the registered adapter's function ref, e.g. "claude-code"
  prompt: TypedBinding<string>;
  system?: string;
  config?: { [key: string]: JsonValue };    // the authored runtime surface
  input?: { [name: string]: Parameter<InlineFamily> };
  output?: { name?: string; schema?: SO };
}): TypedOperation<Record<string, JsonValue>, InferSchema<SO>> & FunctionOp<InlineFamily>;
```

A `TypedOperation` is phantom-branded and **structurally still a plain `Operation<F>`** — serialization and
hashing see no difference, so a typed op and a deserialized one are the same value. `functionOp` checks
each binding against the def's inferred input types: a literal of the wrong type, or a producer whose
output type doesn't match the consumed slot, is a **compile error**; unbound declared names stay free
(external) inputs, and a binding for a name the schema doesn't declare still wires (dynamic inputs).
`free`/`bound` are the untyped escape hatch for dynamic construction — which bypasses these builders
entirely and relies on the runtime checkers. Both tiers check the *same* schemas, so they cannot drift.

`runtimeOp` is authoring sugar for a delegated-runtime invocation and emits **exactly a plain `FunctionOp`**
— `functionRef` names the adapter, the runtime surface is a bound `config` input, the prompt an ordinary
`text` input named `prompt`. No extra field, no refinement in the hashed form. Builders assign a slot's
`kind` with [`kindFor`](#the-operation-model), so a builder-made slot and a hand-written one agree.

---

## `@declarative-ai/exec`

The ONE execution seam and the generic machinery around it — handles, outcomes, the augmentable
`ExecServices` bundle, composition, memoization, rate limiting, deadlines, retry, sessions. It knows
nothing about LLMs, validation, permissions, or filesystems: those declare their own seams by augmenting
`ExecServices`. It has **no dependencies outside the workspace**, and it **re-exports the whole of
[`@declarative-ai/ops`](#declarative-aiops)** (and through it `json`), so a consumer that speaks execution
imports one name set.

### The execution contract

The uniform way to execute any operation — an LLM call, a hierarchical workflow, a delegated agent.
Source: `contract.ts`.

#### `ExecResult<O, M>`

The result of one execution — the value-or-failure envelope every layer shares, with its failure type
pinned to the classified `Failure` and its metrics to `ExecMetrics`. **Never thrown for a unit failure** —
always returned, and the failure branch may still carry a partial `value`.

```ts
type ExecResult<O = ResolvedValue, M extends ExecMetrics = ExecMetrics> = ResultWithMetrics<O, Failure, M>;

// The shared envelope, declared in `json` and re-exported here:
type Result<S, E>               = { value: S } | { error: E; value?: S };  // success has NO `error` key
type ResultWithMetrics<S, E, M> = Result<S, E> & { metrics: M };
function isOk<S, E>(r: Result<S, E>): r is { value: S };
```

**Read it as `isOk(r) ? r.value : r.error`.** The success branch has no `error` key at all — `if (r.error)`
does not compile (it would silently widen `value`), so `isOk(r)` and `"error" in r` are the narrowing
checks. A failed call still keeps its partial `value`, which is what makes a failure diagnosable.

**Execution returns only the op's output value.** What a *model* produced — `thinking`,
`toolCalls`, `finishReason` — is [`@declarative-ai/llm`](#declarative-aillm)'s `LlmOutput`, and it stops at
`promptop`: none of it rides on an `ExecResult`, because none of it is meaningful for a function op.

Generic in the produced value type: an executor threads the op's output type through, so a typed call site
gets a typed result. The default is `ResolvedValue` — JSON, or the bytes of a `blob` output slot — never
`unknown`.

**There is no `artifacts` channel.** A produced artifact is a `blob`-kind output slot like any other, not
a parallel output channel.

Supporting shapes:

| Type | Purpose |
| --- | --- |
| `ExecMetrics` | what EXECUTION measures — `{ durationMs, startMs?, childLlmCalls? }`. No tokens and no money: those belong to whatever ran. A richer `M` (llm's `LlmMetrics`) satisfies this floor structurally and adds its own fields; `mergeExecMetrics` / `EXEC_METRICS_ALGEBRA` combine two of them (duration sums, `startMs` is the first observation, `childLlmCalls` sum). |
| `Failure` (from `json`) | `{ classification: ErrorClass; reason: string; retryAfterMs?; rateLimited? }`. `reason` is the REAL underlying cause, never a bookkeeping message like "retries exhausted". The SAME value classifies an llm call, an execution, and a stored record. |

#### `Executor<R, M, Op>` and `ExecHandle<O, M>`

```ts
interface Executor<R = ExecServices, M extends ExecMetrics = ExecMetrics, Op = Operation<InlineFamily>> {
  readonly capabilities: Capabilities;      // the ops record — an executor IS what a `runtime` entry delegates to
  readonly metrics: MetricsAlgebra<M>;      // how its measurements combine (retry attempts, child→parent)
  capabilitiesFor?(op: Op): Capabilities;   // a DISPATCHER's per-op record (see below)
  start(op: Op, ctx: R): ExecHandle<ResolvedValue, M>;
}

interface ExecHandle<O = ResolvedValue, M extends ExecMetrics = ExecMetrics> {
  events: AsyncIterable<ExecEvent>;   // SINGLE-CONSUMER — attaching a second iterator throws
  result: Promise<ExecResult<O, M>>;  // resolves when done; NEVER rejects for a unit failure
  cancel(): Promise<void>;            // genuinely stops the operation, then settles `result`
}
```

**That is the whole contract.** `Executor` is **generic in `R`**, the environment `start` still requires.
The bare cores and registry-dispatched executors use the default — every seam optional. Composition
**narrows `R`**: a wrapper that reads a ctx seam adds it to `R`, so a composed stack's `start` demands
exactly the fields its wrappers consume — a missing one is a compile error (see [`compose`](#composition)).

Because the payload is an `Operation`, wrapper composition applies **uniformly to prompt and function ops
alike** — previously it could not reach anything dispatched through the function registry, so `withRetry`
and `withMemoize` simply stopped at that boundary.

**`Op` is the operation payload the stack accepts**, defaulting to the resolved inline op — the only thing
a leaf can run, and where every content-reading wrapper (pricing, repair, the leaf) is pinned. The layers
that need only the op's IDENTITY generalize: `withMemoize` keys any serializable op, and
[`withHydration`](#hydration--the-family-transition) is the transition that lets a stack accept another
family's ops (e.g. content-id ops) above it while running inline ops below it.

**`events` is single-consumer.** Events are *delivered* — each to exactly one iterator — not broadcast, so
a second `for await` over the same handle would steal events from the first; attaching twice throws rather
than silently splitting or hanging the stream. A caller that needs several observers drains once and fans
out itself.

**`cancel()` genuinely stops the operation.** It settles `result` — with a `canceled` failure unless the
work had already finished — and returns once it *has* settled, bounded by the handle rather than by
whatever the operation is parked on. It is equivalent to aborting `ctx.abortSignal`: both are the same
event, and both are honored.

**`capabilitiesFor(op)` is a dispatcher's per-op record.** A dispatcher's static `capabilities` is one
record for a whole registry; `capabilitiesFor` returns the capabilities of the entry *this* op dispatches
to (a prompt op's from the prompt executor, a function op's from its registry entry). Absent ⇒
`capabilities` is total for every op (a leaf executor). A wrapper gating on capabilities — `withMemoize`
does — reads this so it consults the entry that is actually running, not a record belonging to no
particular one; a wrapper must forward it (`forwardCapabilitiesFor(inner)`) or the stack degrades to the
static record.

#### `ExecEvent`

The normalized event stream (`handle.events`):

```ts
type ExecEvent =
  | { type: "progress"; message: string }
  | { type: "message"; role: string; content: JsonValue }
  | { type: "child_result"; ref: { label?: string }; metrics: ExecMetrics }
  | { type: "command_request"; command: string; parsed?: JsonValue }
  | { type: "command_result"; decision: "allowed"|"blocked"|"approved"|"denied" }
  | { type: "output_partial"; text: string };
```

> The bare prompt executor emits no events in v1 (its `events` is an empty stream); wrappers forward inner
> streams, and the hw executor emits `progress`/`child_result`.

Human-in-the-loop is **not** a separate engine seam: an interactive step is a `host` registry entry with
`capabilities.interactive` that drives its own renderer, and a tool-call approval is an `Approver`
(see [`@declarative-ai/permissions`](#declarative-aipermissions)).

### The dispatching executor

The `Executor` that turns an op into a call. Source: `operationExecutor.ts`.

```ts
interface OperationExecutorOptions {
  functions: FunctionRegistry<ExecServices, ExecMetrics>;   // the ONE registry of discriminated entries
  prompt?: Executor;                           // what a PromptOp dispatches to; absent ⇒ a permanent failure
  capabilities?: Capabilities;                 // override the advertised record
  metrics?: MetricsAlgebra<ExecMetrics>;       // how measurements combine; defaults to timing/counts
}
class OperationExecutor implements Executor {
  capabilitiesFor(op): Capabilities;           // the DISPATCHED entry's record — the prompt executor's, or the function entry's
}
function createOperationExecutor(options: OperationExecutorOptions): Executor;

function resolveLiteralInputs(op: Operation<InlineFamily>):
  { values: FunctionInputs } | { error: string };
function entryCapabilities(entry: RegisteredFunction<ExecServices, ExecMetrics>): Capabilities;  // an entry's own record, widened to total
```

Dispatch is exactly two cases — `op.kind === "prompt"` → the injected prompt executor, `"function"` → a
registry lookup by `functionRef`. Those two cases are the entire taxonomy — op kind and registry entry
already carry every distinction a third one would add. `prompt` is typed as a plain `Executor`, so this
package never learns that a `PromptOp` HAS an llm lowering.

It takes a **resolved** op: every input parameter is either free (filled by name) or bound to a literal
value ref. Resolving producer edges is the family's business — hw's engine walks its own scope, pending
joins and all — so a producer edge reaching here is a wiring bug and is reported as one rather than
silently skipped. The impl's own `metrics` report wins over the dispatch timing frame, while
`startMs`/`durationMs` stay the executor's so they measure the dispatch, not the impl's opinion.

### Composition

Stack cross-cutting behaviors around a core executor. Source: `contract.ts`.

#### `ExecutorWrapper<RIn, ROut>`

```ts
type ExecutorWrapper<RIn = ExecServices, ROut = RIn> = (inner: Executor<RIn>) => Executor<ROut>;
```

A composable behavior. A construction-injected wrapper leaves the requirement unchanged
(`ExecutorWrapper<R, R>`); a ctx-reading one adds its seam (`withDeadline()` →
`ExecutorWrapper<R, R & { deadline; stepStartMs }>`). The **generic** wrappers — whose behavior is a
property of execution itself — are [here](#generic-wrappers); the **llm-aware** ones live in
[`@declarative-ai/promptop`](#llm-aware-wrappers).

#### `compose` / `ComposableExecutor`

The **inside-out builder** — the recommended form, because it type-tracks requirements.

```ts
function compose<R>(core: Executor<R>): ComposableExecutor<R>;
class ComposableExecutor<R, M, Op> implements Executor<R, M, Op> {
  // Subsumes both an ExecutorWrapper (op type unchanged) and a family-transition
  // adapter like withHydration, which changes what the stack above it accepts.
  with<ROut, OpOut = Op>(wrap: (inner: Executor<R, M, Op>) => Executor<ROut, M, OpOut>): ComposableExecutor<ROut, M, OpOut>;
  start(op, ctx): ExecHandle;   // it IS an Executor — drops into a registry unchanged
}
```

`compose(core).with(a).with(b)` = `b(a(core))`, read core-first with each `.with` adding an **outer** layer.
Each wrapper that adds a ctx seam narrows `R`, so the final `.start` requires exactly the union of what the
stack consumes — forgetting one is a compile error. See [README example 3](README.md#3-the-contract-path--one-seam-a-composed-executor-stack).

#### `composeExecutors`

```ts
function composeExecutors(core: Executor, ...wrappers: ExecutorWrapper[]): Executor;
```

The loose variadic convenience (flat list, **no** requirement tracking). Handy; `compose(...).with(...)` is
clearer about ordering and compile-time-checked.

### Generic wrappers

The wrappers whose behavior is a property of execution itself. Source: `wrappers.ts`. Each is
**dual-mode**: called with a `config` object it returns the curried wrapper (for `compose(...).with(...)` /
`composeExecutors`); called with a trailing `inner` executor it applies immediately (direct nesting).

| Wrapper | Signature (config form) | What it does |
| --- | --- | --- |
| `withRetry` | `withRetry({ transient?, validation?, budget? }): ExecutorWrapper` | The unified re-attempt policy. `transient` (a cap, or `{ cap, baseBackoffMs?, maxBackoffMs?, waitMs?, random? }`) re-attempts a **network-retriable** failure with full-jitter backoff. `validation: { turns, feedback? }` re-attempts a **validation** (`api-retriable`) failure; `feedback: true` appends the concrete errors to the prompt op's `user` text (targeted repair — this is what the old `withRepair` was), `false` is a blind re-roll (default). `budget` (a `RetryBudget` — `{ allowMore(): boolean }`) is consulted before each re-attempt. Metrics accumulate through the executor's registered `metrics.merge`; a non-retriable failure/success stops, and a **futile** failure (budget-exhausted or deadline-floor) short-circuits *before* spending a backoff, since it cannot be re-attempted into success within this window. Both axes compose. |
| `withDeadline` | `withDeadline(config?): ExecutorWrapper` | Reads `{ deadline, stepStartMs }` from config **or** ctx. Below the start floor it short-circuits with a `deadline` failure and never starts the call; otherwise it clamps `ctx.timeoutMs` to the remaining window **and enforces it** — racing the in-flight call against the window and cancelling it with a `deadline` failure if it overruns, because the clamp alone is only advisory to an inner executor that may not read `timeoutMs`. |
| `withMemoize` | `withMemoize({ cache, identify? }): ExecutorWrapper` | See [Memoization](#memoization). |

Now that a function impl resolves a *classified* failure, `withRetry` reaches function ops too: a 429
raised inside a registered async function is retried here instead of being permanently failed. A function
op has no prompt to amend, so `feedback` degrades to a plain re-attempt rather than pretending to repair
something. The augmented **op** carries the repair hint, so an inner `withMemoize` keys on exactly what is
sent — there is no separate hash to keep in sync.

Helpers: `isExecutor(x)` (disambiguates a wrapper's optional trailing `inner` from its optional config) and
`curryOrApply(wrap, inner?)` (the dual-mode dispatch). `withRetry` aggregates attempts through the inner
executor's registered `metrics.merge` — `mergeExecMetrics` for a bare stack — so it never learns which of
`M`'s fields sum, take the latest, or are the first observation.

**Requirement tracking.** `withDeadline` adds the ctx seams it reads to what the composed `.start`
requires — unless you supply them at construction (`withDeadline({ deadline })` needs only `stepStartMs`;
`withDeadline({ deadline, stepStartMs })` needs neither). With the `compose(...).with(...)` builder this is
compile-time-checked.

**Order encodes semantics** (DESIGN §1.4): `withMemoize` outermost caches the final post-repair result;
`withRateLimit`/`withDeadline` innermost apply per attempt.

### Memoization

With `canonicalize`/`sha256Hex` at the bottom of the graph in `json`, this is a few dozen dependency-free
lines. Source: `memo.ts`.

```ts
function hashOperation(op: Operation<InlineFamily>): string;
function memoKey(params: { operationHash: string; workspaceTreeHash?: string; executorId?: string }): string;

interface MemoCache {
  get(key: string): Promise<ExecResult<ResolvedValue> | undefined> | ExecResult<ResolvedValue> | undefined;
  set(key: string, result: ExecResult<ResolvedValue>): Promise<void> | void;
}
class MapMemoCache implements MemoCache { constructor(maxEntries?: number); }   // opt-in LRU bound
interface MemoizeOptions<Op = Operation<InlineFamily>> {
  identify?(op: Op): string;                        // cheaper/canonical op identity (an id-family op: its own id)
  namespace?: string;                               // the executorId component — WHO answered
}

function withMemoize<R, Op = Operation<InlineFamily>>(config: { cache: MemoCache } & MemoizeOptions<Op>): ExecutorWrapper<R, R, ExecMetrics, Op>;
```

`withMemoize` is generic in `Op`: the default `identify` hashes the op's canonical serialized content —
valid for ANY serializable op shape — so it composes above a
[`withHydration`](#hydration--the-family-transition) transition and keys the CHEAP family op, never the
hydrated content.

**The identity component is the OPERATION's content hash.** That is the whole simplification the single
execution seam buys: a resolved op's `input` parameters carry their bindings, so the op already embeds the
values a run was given. There is no `definition` + `inputs` pair to hash separately and nothing asserting
serializability on the caller's behalf.

`hashOperation` replaces a blob leaf with the hash of its **bytes**, so a memo key is stable across two
runs handed the same image by different means; a live stream **throws** with the remedy rather than
silently keying on object identity.

`workspaceTreeHash` is folded in whenever the run HAS a workspace — merely *reading* one makes the answer
snapshot-dependent — and is REQUIRED for an executor declaring `mutatesWorkspace`: `withMemoize` refuses
(with a permanent failure) rather than cache under a key that silently means "any workspace".

Any `{ get, set }` is a `MemoCache` (a `Map`, or a durable store); both methods may be sync or async, and
`MapMemoCache(maxEntries?)` opts into an LRU bound. `identify` supplies a cheaper/canonical op identity
(e.g. `workflowIdentify(definition)`) instead of the default `hashOperation`; `namespace` sets the
**`executorId`** component of the key — WHO answered — so two executors sharing one cache (different model
routing, different registry, a real one and a stub) never collide on byte-identical ops. `withMemoize`
supplies a per-inner-executor token by default; pass an explicit stable string for a durable cache.
**Only successful results are cached.** A cache **hit reports zero** metrics (every numeric field zeroed,
`startMs` re-stamped), so an outer retry or budget fold does not re-charge the work that filled the cache;
and identical calls **in flight together** share one execution rather than each missing the not-yet-written
entry. `withMemoize` **throws at composition time** if it would wrap a `sessionResume` layer (a dispatcher,
whose static record is not the whole truth, defers that refusal into `start`, per op) — session state isn't
in the memo key, so a hit would replay a stale answer and skip the transcript update; compose `withSession`
**outside** it instead (sound, because that layer recomputes the sent op from the full transcript).

### Hydration — the family transition

Source: `hydrate.ts`.

```ts
type Hydrator<Op, R = ExecServices> = (op: Op, ctx: R) => Operation<InlineFamily> | Promise<Operation<InlineFamily>>;

function withHydration<Op, R, M>(
  resolve: Hydrator<Op, R>,
  options?: { capabilitiesFor?: (op: Op) => Capabilities },
): (inner: Executor<R, M, Operation<InlineFamily>>) => Executor<R, M, Op>;
```

An id-leaf family has the OPPOSITE cost profile from the inline family: identity is cheap (the op's
leaves are content ids; it may carry its own hash) and content is expensive (every leaf is a store
read). So the identity-only layers should run before the content exists, and the content-reading ones
after. `withHydration` is that boundary — NOT an `ExecutorWrapper` (it changes the op type the stack
accepts, which is the point), but it composes through the same `.with(...)`:

```ts
compose(leaf)                        // inline: reads content
  .with(withRateLimit({ limiter })) // inline: estimates off the prompt text
  .with(withBudget({ meter }))      // inline: reserve → settle around the real call
  .with(withHydration(resolve))     // ← id ops above, inline ops below
  .with(withMemoize({ cache, identify: (op) => op.id }))  // id: keys on the content id, free
  .with(withBudget({ meter, computeCost }))               // id: bills memo reuse
```

On a memo **hit** nothing below the transition runs — the store reads happen only on a miss. `resolve`
is the family's hydrator ("hydration is the family's business"): artifact loads, structural-sharing
folds, producer edges. A hydration fault comes back as a **classified failure** through the never-throws
handle (`failureOf` reads retriability off the error), and the optional `capabilitiesFor` keeps per-op
capability gating available above the transition without hydrating.

### Injected service seams

`ExecServices` is the environment an executor runs with — the seam bundle passed as `ctx` to `start`. All
fields optional; an absent service is a no-op (unthrottled, unmetered, unvalidated).

```ts
interface ExecServices {
  meter?: BudgetMeter;
  validator?: OutputValidator;        // json's three-line seam
  clock?: Clock;
  deadline?: DeadlineConfig;
  stepStartMs?: number;               // step-start origin for deadline arithmetic
  executor?: Executor;                // composite ops execute children through this
  tools?: Record<string, Tool>;       // executables the current operation may call mid-loop
  sessions?: SessionStore;            // run-scoped, logical-id-keyed
  workspace?: Workspace;              // { root, treeHash? } — a Session-owned resource
  timeoutMs?: number;                 // per-call wall-clock budget
  maxCostUsd?: number;                // per-call cost ceiling
  abortSignal?: AbortSignal;          // cancellation for the operation in flight
}
```

**This interface is AUGMENTABLE.** Splitting packages does not by itself stop `exec` from *naming* every
optional capability, so each optional package declares its own seam by declaration merging:

```ts
// in @declarative-ai/permissions
declare module "@declarative-ai/exec" {
  interface ExecServices { policy?: ExecPolicy; approve?: Approver }
}
// in @declarative-ai/promptop
declare module "@declarative-ai/exec" {
  interface ExecServices { modelRouter?: ModelRouter }
}
```

`exec` therefore does not know that permissions or model routing exist. The cost is that augmentation is
GLOBAL — two packages cannot declare conflicting seams, and go-to-definition lands in the owning package —
so it is used only where it earns that. `OutputValidator` and `Workspace` take the *other* §1.2 route
("declare the minimal structural interface you consume"): `OutputValidator` is `json`'s three-line
interface, because `promptop` needs the validator seam without depending on `validate` (which would drag
ajv into the LLM path); `Workspace`'s `root`/`treeHash` are plain fields every consumer needs, while the
FILESYSTEM is what `tools` owns.

The seam interfaces:

| Interface | Shape | Notes |
| --- | --- | --- |
| `OutputValidator` | `validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult \| Promise<ValidationResult>` | Declared in `json`; maybe-ASYNC (a store-backed validator has `$ref` reads to do; `executeLlmCall` awaits). The sync refinement `SyncOutputValidator` + the fail-closed `syncOnly` narrowing serve mid-walk consumers. Implemented by `@declarative-ai/validate`'s `SchemaValidator` (sync) / `asBoundaryValidator` (maybe-async, store-backed). |
| `RateLimiter` | `schedule<T>(est: CallEstimate, run): Promise<T>`; `reportOutcome({ rateLimited?; modelId? })` | Construction-injected into `withRateLimit` (not a ctx seam). `CallEstimate = { inputTokens; outputTokens; modelId? }`. |
| `BudgetMeter` | `reserve(estCostUsd): Promise<BudgetReservation \| null>`; `availableCostUsd(): Promise<number>`; `debit?(actualCostUsd)` | Per-call wallet reservation; `null` ⇒ balance can't cover it. `BudgetReservation.settle(actualCostUsd)` corrects the reserve. `debit` records spend that ALREADY happened and could not be reserved first — a delegated agent bills inside its own loop, so its charge arrives as a fact rather than a request. |
| `Clock` | `now(): number` | Injectable time source (tests, deterministic replay). |
| `DeadlineConfig` | `{ maxDurationMs; safetyMarginMs?; floorMs? }` | The window budget consumed by `withDeadline` (DESIGN §3.5). |
| `Workspace` | `{ root: string; treeHash?: string }` | The directory an operation's tools act within — `root` for every consumer, `treeHash` for memoization. No filesystem here; see [`@declarative-ai/tools`](#declarative-aitools). |

**`abortSignal` on the ctx.** A registered async function's *only* channel to its caller is the ctx — so a
delegated agent adapter reads `ctx.abortSignal`, and the prompt executor combines it with its own internal
cancel.

### The capability registry

The named things a state operation can reference. Source: `contract.ts`.

```ts
interface CapabilityRegistry<M extends ExecMetrics = ExecMetrics> {
  functions: FunctionRegistry<ExecServices, M>;  // host code, sub-workflows, AND delegated runtime adapters
  skills: Map<string, SkillTemplate>;            // named prompt templates
  tools: Map<string, Tool>;                      // agent-callable tools, by logical name
}
function newCapabilityRegistry<M>(functions?: FunctionRegistry<ExecServices, M>): CapabilityRegistry<M>;
type SkillTemplate = string;            // name → prompt template
```

**All three facets are plain `Map`s.** A dedicated registry interface plus a map-backed class would add
nothing over `Map.get`/`Map.set` — including the chaining, since `Map.set` returns the map — and there was
never a second implementation. `newCapabilityRegistry()` is just "three empty maps".

There is **no `runtimes` facet and no `prompt` facet**: a runtime invocation is a plain `FunctionOp`, so
delegated agent adapters register in `functions` like anything else and are distinguished only by their
entry's *capabilities*; and a `PromptOp` is dispatched to an `Executor` like everything else, so the llm
path is not privileged over any other. The `functions` facet **is** the ops
[`FunctionRegistry`](#the-function-registry), unextended. Permission gating and search refusal read the
**resolved entry** — never the op, which carries no runtime marker.

#### `Tool` and `NativeToolRef`

```ts
interface Tool<I = FunctionInputs, O = JsonValue> {
  readonly description?: string;      // what the tool does — shown to the model
  readonly inputSchema: JsonSchema<I>;  // JSON Schema for the input the model must produce
  readonly readOnly: boolean;         // what the read-only/plan profiles gate on
  run(input: I, ctx: ExecServices): O | Promise<O>;
}

interface NativeToolRef { readonly native: string; }   // a delegated agent's OWN built-in, by name
```

A **tool** is an impl plus the call-metadata a model needs to decide to call it: the same impl can be
surfaced as a graph `function` op and as an agent tool, with one schema and no drift. A tool's `run`
returns its value and **may throw** — a tool failure travels back to the MODEL as a result it reads and
reacts to, so it is deliberately *not* the classified-failure channel that `Result` is for.
`NativeToolRef` names a black-box agent's own tool, which we cannot execute ourselves; a tool rename
binding is therefore `Tool | NativeToolRef`.

Ready-made workspace tools ship in **[`@declarative-ai/tools`](#declarative-aitools)**.

### Sessions

```ts
interface SessionState<Msg = JsonValue> { messages?: Msg[]; providerSessionId?: string }
interface SessionStore<Msg = JsonValue> {
  get(logicalId: string): SessionState<Msg> | undefined | Promise<SessionState<Msg> | undefined>;
  put(logicalId: string, state: SessionState<Msg>): void | Promise<void>;
}
class MapSessionStore<Msg = JsonValue> implements SessionStore<Msg> {}   // in-memory
```

**Mutable, keyed by a logical id** — a conversation's transcript (client-managed) or a provider handle. A
logical id NEVER carries the provider handle in the portable declaration; it lives here, mapped from the
logical id. Backs [`withSession`](#llm-aware-wrappers); a workflow injects a run-scoped store via
`ctx.sessions`. Generic in the message shape so each consumer pins it (promptop: the AI-SDK `ModelMessage`;
hw: a `Turn`); the default is plain JSON, since a stored transcript is serializable by construction.

### Handle scaffolding

The never-throws contract is easy to break by accident — a wrapper body that rejects, a cancel that lands
before the inner call starts, an inner event stream that gets swallowed — so the scaffolding is in one
place. Source: `handles.ts`.

| Export | Purpose |
| --- | --- |
| `emptyEvents()` | an empty, already-completed `AsyncIterable<ExecEvent>` (for executors that emit none). |
| `failure(classification, reason)` | a zero-cost `ExecResult` failure of the given class. |
| `permanentFailure(reason)` / `canceledFailure(reason)` | the two common specializations. |
| `finishedHandle(result)` | a completed handle wrapping a ready `ExecResult` (short-circuiting wrappers). |
| `withMetrics(result, metrics)` | swap a result's metrics for a supplied record (an accumulated total, a cache hit's zeros). |
| `wrapHandle(body, options?)` | the shared scaffold for wrappers that start inner handles asynchronously: forwards every registered inner handle's events, flips the canceled flag **first** so a body that hasn't started yet can short-circuit, wires `options.signal` to cancellation, and normalizes a body throw into a permanent-failure result. |
| `WrapControl` | what `wrapHandle` hands the body: `canceled()`, `started(h)`, and the abort `signal`. |
| `EventQueue` | a simple event queue an executor pushes into while it runs (`push`/`close`/`iterate`). |

### Rate limiting

Concurrency pool + token buckets with AIMD. Source: `concurrency.ts`.

```ts
class AdaptiveRateController implements RateLimiter {
  constructor(opts?: AdaptiveRateControllerOptions);
  schedule<T>(est: CallEstimate, run: () => Promise<T>): Promise<T>;
  reportOutcome(outcome: { rateLimited?: boolean }): void;
  get concurrency(): number; get activeCount(): number;
}
interface AdaptiveRateControllerOptions {
  initialConcurrency?; maxConcurrency?; minConcurrency?;
  increaseEvery?;              // +1 concurrency after this many consecutive successes (default 8)
  tokensPerMinute?;           // legacy combined bucket
  modelLimits?: ModelLimitResolver;   // per-model RPM/ITPM/OTPM buckets
  now?; wait?;
}
```

The v1 controller: a concurrency pool + optional per-model token buckets, with **AIMD** on concurrency —
halve on a 429 (`reportOutcome({ rateLimited: true })`), +1 after sustained success. Header backoff
(`retry-after`) is the retry loop's job (see [Retry](#retry)); this is the orthogonal "how fast to go next
time" control. This is the concrete `RateLimiter` you inject into
[`withRateLimit`](#llm-aware-wrappers).

Building blocks and helpers:

| Export | Purpose |
| --- | --- |
| `ConcurrencyLimiter` | a bare async concurrency gate (`run`, `setLimit`, `currentLimit`, `activeCount`). |
| `TokenBucket` | a refilling token bucket (`remove(n)` waits for capacity). |
| `PassthroughRateLimiter` | a no-op `RateLimiter` (unthrottled). |
| `ProviderDispatchRateLimiter` | routes to a per-provider `RateLimiter`. |
| `ModelRateLimits` / `ModelLimitResolver` | per-model `{ rpm?, inputTpm?, outputTpm? }` and `modelId → { key, limits }`. |
| `prefixModelLimitResolver(map)` | build a resolver from a `{ prefix: limits }` map. |

The prompt-shaped `estimateCallTokens(prompt, system, maxOutputTokens)` that prices admission lives in
[`@declarative-ai/llm`](#cost-estimation) — the limiter itself is generic counting machinery and needs no
notion of a prompt.

### Retry

The retry *loop* lives in the [`withRetry`](#generic-wrappers) wrapper — the whole re-attempt policy
(transient backoff, validation repair, the budget gate, and the futility short-circuits) is one place, and
there is no separate `retryLoop` export. What `retry.ts` still exports is the reusable pieces underneath it.
Source: `retry.ts`.

```ts
interface RetryBudget { allowMore(): boolean; }   // the gate consulted before each re-attempt
function backoffDelayMs(
  attempt: number,
  retryAfterMs: number | undefined,
  opts: { baseBackoffMs: number; maxBackoffMs: number },
  random: () => number,
): number;
```

**Discipline** (enforced by `withRetry`): only `network-retriable` failures auto-retry; an `api-retriable`
output failure is a re-roll you must opt into (`validation: { turns }`) — silently re-rolling a stochastic
output until it passes biases scores. A server `retry-after` wins over exponential backoff; otherwise
full-jitter exponential (`backoffDelayMs`). A **futile** failure — a budget-exhausted or deadline-floor
reason — short-circuits *before* the backoff, since it cannot be re-attempted into success in this window.

Constants: `DEFAULT_BASE_BACKOFF_MS` (500), `DEFAULT_MAX_BACKOFF_MS` (60_000).

### Deadline arithmetic

The window-budget math the `withDeadline` wrapper is built on. Source: `deadline.ts`.

```ts
function deadlineDecision(stepStartMs: number, cfg: DeadlineConfig, now: number): DeadlineDecision;
function computeRemainingMs(stepStartMs: number, cfg: DeadlineConfig, now: number): number;
interface DeadlineDecision { proceed: boolean; remainingMs: number; }
const systemClock: Clock;                 // { now: () => Date.now() }
function isDeadlineFloor(reason: string): boolean;
```

`remainingMs = stepStartMs + maxDurationMs − safetyMargin − now`; `proceed` is `remainingMs >= floor`. The
window budget (DESIGN §3.5) is *not* a per-call duration — it's shared across every step of a serverless
invocation, and `withDeadline` is the only thing that reads it. Constants: `DEFAULT_SAFETY_MARGIN_MS`
(10_000), `DEFAULT_FLOOR_MS` (5_000), `DEADLINE_FLOOR_REASON`.

---

## `@declarative-ai/validate`

The schema CHECKING layer: structural subtyping, ONE generic binding checker, and one ajv wrapper. This is
the **only package carrying a heavy dependency** (ajv), and nothing below it imports it — which is the
point: a structured LLM call runs with `json + llm` and nothing else.

### Schema subtyping

`isSubschema(sub, sup)` answers *"is every value valid under `sub` necessarily valid under `sup`?"* — i.e.
can a producer whose output matches `sub` safely feed a consumer slot requiring `sup`. It is the
compatibility rule the binding checker runs on. Source: `subtype.ts`.

```ts
type Schema = Record<string, unknown>;
type ResolveRef = (refId: string) => Schema | undefined;
interface SubtypeResult { ok: boolean; reason?: string; }

function isSubschema(sub: Schema, sup: Schema, resolve?: ResolveRef, seen?: Set<string>): SubtypeResult;
```

Sound structural subtyping over the v1 keyword set: `type` gate (with `integer` ⊆ `number`), `enum`/`const`
⊆, object `required` coverage + recursive property compatibility + `additionalProperties: false` honored,
array `items` recursion, and widen-only numeric/length bounds. `allOf` **is** supported — flattened into
one effective schema, which is what makes schema evolution (`allOf: [{$ref: old}, {+field}]`) typecheck.
`$ref` resolution is injectable via `resolve`, so the id family derefs through a store and the inline
family passes documents directly.

Everything else is **conservatively rejected with a precise reason** rather than silently passed:
`anyOf`/`oneOf`/`not`/`if`/`then`/`else` on either side, and any unmodeled *constraining* keyword on the
consumer. Annotations (`title`, `description`, `default`, `examples`, `$id`, `$schema`, `$comment`,
`format`, and the blob content keywords `contentEncoding`/`contentMediaType` — which carry KIND information
that `kindFor` reads, not a constraint) and `x-`-prefixed extension keywords are ignored, not rejected.
**`x-type` is the one exception**: a slot declaring a type name accepts only producers declaring the same
name. Consumer-side `anyOf` is the documented next addition.

### The binding checker

ONE generic checker, parameterized by the ref family with injectable resolution, replacing findmyprompt's
`checker.ts` and hw's hand-rolled twin. Source: `checker.ts`.

```ts
interface CheckIssue { path: string; message: string }
interface CheckResult { errors: CheckIssue[]; warnings: CheckIssue[] }
interface CheckerHooks<F extends RefFamily> { /* family-specific resolution */ }

function checkOperation<F extends RefFamily>(…): CheckResult;
function checkBinding<F extends RefFamily>(…): CheckResult;
function producerSchemaOf<F extends RefFamily>(…): JsonSchema | undefined;

function schemaOfValue(v: JsonValue): JsonSchema;        // a literal as a `const`-constrained schema
function isUniversalSchema(s: SchemaDocument | undefined): boolean;   // `{}` accepts anything
```

The id family resolves refs through stores; the inline family passes documents directly. A literal binding
becomes a `const`-constrained schema, so a literal satisfies an `enum`-constrained consumer.

### Schema validation

Ajv at the boundaries. Source: `ajv.ts`.

```ts
class SchemaValidator implements OutputValidator {
  constructor(resolver?: SchemaResolver);
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult;          // inline schema
  compile(schemaId: string, schemaDoc?: unknown): Promise<ValidateFunction>;          // store-backed
  validate(schemaId: string, value: unknown, schemaDoc?: unknown): Promise<ValidationResult>;
  errorsText(fn: ValidateFunction): string;
}
interface SchemaResolver { getSchema(id: string): Promise<unknown | undefined>; }
function collectRefs(node: unknown, out?: Set<string>): Set<string>;
function asBoundaryValidator(v: SchemaValidator): OutputValidator;   // maybe-async: $ref docs go store-backed
```

Two modes: **inline** (`validateValue`, compiled + cached by the schema's content hash, SYNC — the
`SyncOutputValidator` the hw engine consumes) and **store-backed** (`compile`/`validate`, resolving
`$ref` ids lazily through an injected `SchemaResolver`). `asBoundaryValidator` lifts a validator to the
maybe-async boundary seam: a ref-bearing document compiles through the store-backed path, a ref-free one
answers synchronously — the validator an llm-call environment wants when schemas are id-family
artifacts. `collectRefs` gathers every `$ref` target in a document.

`SchemaValidator` simply *implements* `json`'s three-line `OutputValidator` seam — this package declares no
`ExecServices` augmentation, because `exec` already names the minimal structural interface it consumes and
never learns the concrete type.

---

## `@declarative-ai/permissions`

An agent's tool call is authorized by a **profile × mode**. Source: `permissions.ts`; see
[DESIGN.md](DESIGN.md) §5.1. Its only consumers are the workflow engine
and the delegated-agent adapters, which is why it is its own package rather than 265 lines sitting in a
`core` everything depends on. Tools are permission-wrapped by the hw
engine only when `EngineConfig.permissions.approve` is supplied.

```ts
type PermissionMode = "allow" | "deny" | "ask" | "smart";   // smart → a bound arg-inspecting policy
type PermissionProfile = "read-only" | "plan" | "full" | (string & {});
type ProfilePredicate = (tool: { name: string; readOnly: boolean }) => boolean;   // a custom profile
function inProfile(profile: PermissionProfile, tool: { name; readOnly }, custom?): boolean;

type PermissionScope = "once" | "session" | "workflow-run" | "always";   // in-memory lifetimes, widening
interface PermissionRequest { tool: string; input: JsonValue; sessionId: string; }
interface PermissionDecision { decision: "allow" | "deny"; scope: PermissionScope; }
type Approver = (req: PermissionRequest) => PermissionDecision | Promise<PermissionDecision>;
type SmartVerdict = "allow" | "deny" | "ask";
type SmartApprover = (req: PermissionRequest) => SmartVerdict | Promise<SmartVerdict>;
interface PermissionBaseline { default?: PermissionMode; tools?: Record<string, PermissionMode>; profile?: PermissionProfile; }

class PermissionLedger {                    // scope chain: session → workflow-run → process → baseline → ask
  constructor(opts?: { baseline?: PermissionBaseline; process?: Map<string, PermissionMode> });
  resolve(tool, sessionId, fallback?): PermissionMode;          // fallback = the per-state authored mode
  apply(tool, decision: PermissionDecision, sessionId): void;   // writes at the decision's scope
  resolveProfile(sessionId): PermissionProfile; setProfile(sessionId, p): void; seedProfile(sessionId, p): void;
}
function withPermission(tool: Tool, opts: { ledger; sessionId; toolName; approve: Approver;
                                            authoredMode?; smart?: SmartApprover; profiles? }): Tool;
function planExitTool(opts: { ledger; sessionId; approve: Approver }): Tool;   // plan → full on approval
type PermissionDenied = { denied: true; tool: string; reason: string };        // isPermissionDenied(v)
```

`plan` mode blocks mutating tools until the agent calls the injected `exit_plan` tool and a human approves.
`withPermission` gates the profile first (an out-of-scope tool is refused regardless of mode), then the
mode; a refusal is returned to the model as a `PermissionDenied` JSON result, not thrown.

#### `ExecPolicy` — the compiled policy on `ctx.policy`

```ts
interface ExecPolicy {
  baseline?: PermissionBaseline;                   // authored, durable: per-tool modes, default, profile
  profiles?: Record<string, ProfilePredicate>;     // custom profile predicates by name
  smart?: Record<string, SmartApprover>;           // per-tool arg-inspecting policies
  nativeTools?: string[];                          // DELEGATED adapters: the agent's own tools, allow-listed
}

// This package DECLARES its own seams — `exec` therefore does not know that permissions exist.
declare module "@declarative-ai/exec" {
  interface ExecServices { policy?: ExecPolicy; approve?: Approver }
}
```

`ExecServices.policy` is this type, not an opaque blob. How it is enforced follows the executing entry's
`policyEnforcement` capability: `"callback"` wraps each tool with `withPermission` and gates per call;
`"config"` translates the policy into the delegated agent's own permission config and routes its native
prompt back through `approve`; `"none"` means the unit takes no tool calls.

---

## `@declarative-ai/llm`

One structured LLM call, end to end, and the provider layer: the call itself, the declaration + its strict
parsing/resolution, `plan`, the model router, the catalog, and the schema/reasoning adaptation.

**The direct call path is `exec`-FREE.** `executeLlmCall(definition, environment)` needs nothing but this
package and `@declarative-ai/json` — the coupling that used to exist was packaging, not code (`services`
re-exported the ajv validator, so one `import { systemClock }` dragged ajv into llm's *module* graph).
Running a `PromptOp` through the `Executor` seam is [`@declarative-ai/promptop`](#declarative-aipromptop)'s
job.

**Generic defaults are `JsonValue`, not `unknown`** across this package (`LlmOutput<T = JsonValue>`,
`LlmCallResult<T = JsonValue>`, `LlmCallDefinition`, `LlmCallRequest`, `executeRequest`, `executeLlmCall`).
Schema documents are the ONE
`JsonSchema<T>` everywhere; this package's `JsonSchema` and `typedSchema` exports are **re-exports** of
that type, as `ReasoningSegment` and `TokenCounts` are. The residual `unknown`s live only inside
wire-parsing functions (usage extraction, catalog-source probing, stream-part narrowing) and never appear
in an exported type.

### The LLM declaration

The one canonical "how to call an LLM" type — pure serializable data, no provider/DB coupling, no AI-SDK
import. Source: `llmConfig.ts`.

#### `LlmConfiguration` and its variants

```ts
interface LlmConfiguration {                 // universal base — every model accepts these
  model: string;                             // route-prefixed id
  maxOutputTokens?; stopSequences?: string[]; seed?;
  providerOptions?: ProviderOptions;         // raw per-provider passthrough (escape hatch)
  tools?: ToolDefinition[]; toolChoice?: LlmToolChoice; maxSteps?;
  sessionId?; providerSessionId?;            // session ids (DESIGN §1.6)
  outputModalities?: string[];               // gated against modalities.output at plan time
}
interface SamplingConfiguration extends LlmConfiguration {   // sampling family
  temperature?; topP?; topK?; presencePenalty?; frequencyPenalty?;
}
interface ReasoningConfiguration extends LlmConfiguration {  // reasoning family
  reasoning: ReasoningSpec;                  // presence is the discriminant; rejects sampling knobs
}
type LlmCallConfig = SamplingConfiguration | ReasoningConfiguration;   // sampling XOR reasoning
```

A model is **sampling XOR reasoning** — never both. The union is discriminated by `reasoning` (present ⇒
reasoning). Once parsed, an illegal "both at once" state is unrepresentable.

| Type | Shape |
| --- | --- |
| `ReasoningSpec` | `{ effort?: "low"\|"medium"\|"high"; budgetTokens? }` — provider-neutral, adapted at the boundary; at least one field required. |
| `ToolDefinition` | a **function** tool `{ type?: "function"; name; description?; inputSchema; strict? }` **or** a **provider** tool `{ type: "provider"; name; id: "<provider>.<name>"; args }`. Serializable declaration only — the `execute` impl is injected. |
| `LlmToolChoice` | `"auto" \| "none" \| "required" \| { type: "tool"; toolName }`. |
| `ProviderOptions` | `Record<string, Record<string, JsonValue>>` — per-provider raw options, merged with adapted reasoning at the call boundary. |

#### Parsing (`parseLlmConfig`, `parseReasoningSpec`, `LlmConfigParseError`)

```ts
function parseLlmConfig(json: JsonValue): LlmCallConfig;    // strict; throws on malformed/unknown-key
function parseReasoningSpec(v: unknown): ReasoningSpec;
function isReasoningConfig(cfg: LlmCallConfig): cfg is ReasoningConfiguration;
class LlmConfigParseError extends Error {}
```

**Parse, don't validate.** `parseLlmConfig` turns a stored JSON blob into a concrete variant and **throws**
on a present-but-wrong-typed field, an unknown key, or a reasoning config that also carries sampling knobs.
Errors naming a signature key (`prompt`/`system`/…) hint that `resolveConfig` should have split it out.

#### `CallSignature` and `LlmCallDefinition`

```ts
interface CallPromptInput {
  system?: string | SystemModelMessage | SystemModelMessage[];
  prompt?: string | ModelMessage[];      // provide exactly one of prompt/messages
  messages?: ModelMessage[];
  attachments?: FileInput[];             // lowered to provider file parts at the boundary
}

type CallSignature<T = JsonValue> = CallPromptInput & {
  schema?: JsonSchema<T>;                // omitted ⇒ a TEXT-output call
  timeoutMs?: number;                    // per-call wall-clock budget
};

type LlmCallDefinition<T = JsonValue> = LlmCallConfig & CallSignature<T>;

function promptAsMessages(p: CallPromptInput): ModelMessage[];   // normalize to message-list form
function promptText(p: CallPromptInput): string;                 // extract all plain text
```

`CallSignature` is the name because that is what it is — **an input shape plus an output shape** — and it
mirrors `PromptOp` exactly: `input` params + `output` param, with `config` a sibling of both. The output
schema goes on the **prompt side, not the config**, for two reasons that are not stylistic: `resolveConfig`
merges configs as LAYERS and merging output schemas across a defaults/preset/inline stack is meaningless;
and findmyprompt's search point is `LlmParameters = LlmCallConfig & { systemPrompt, userPrompt }`, where the
optimizer searches decoding knobs and never the output type.

`LlmCallDefinition` is a **union** (not a flattened bag), so "sampling + reasoning at once" is
unrepresentable. Everything in it is plain JSON — no live handle, no closures — which is what lets the call
become a durable step. `schema` belongs *in* the definition because it is declarative and serializable:
leaving it out would force the lowering to smuggle the output schema alongside and cast the phantom away.
`timeoutMs` is a call-site concern (a clamp), so it is an argument rather than a required field.

### Configuration resolution

Compose config fragments into one valid declaration. Source: `llmConfig.ts`.

```ts
type ConfigLayer<T = JsonValue> = Partial<FlattenUnion<LlmCallConfig>> & Partial<CallSignature<T>>;

function resolveConfig<T = JsonValue>(layers: Array<ConfigLayer<T> | undefined>): ResolveResult<T>;

interface ResolveResult<T = JsonValue> {
  definition: LlmCallDefinition<T>;         // composed, strictly-parsed — config knobs AND signature
  warnings: string[];                       // e.g. a family switch that cleared the opposite knobs
}
```

A layer is a **typed** partial config that may carry signature fields alongside — no `Record` bag.
`FlattenUnion` is what lets one layer name sampling and reasoning knobs alike; the XOR is enforced when the
merged bag is parsed, not per layer.

Merge raw property bags **low→high** (`[engineDefault, workflowDefault, registry.get(ref), inline]` — later
wins per key), split out the **signature keys** so the config half parses strictly, then layer them back
on. The merge is **family-aware** (introducing `reasoning` clears accumulated sampling knobs, and
vice-versa, each with a warning). Identity is always the resolved content hash; registry ids are provenance
only. See [README example 7](README.md#7-config-resolution--compose-fragments-into-one-valid-declaration).

```ts
const SAMPLING_KEYS  = ["temperature","topP","topK","presencePenalty","frequencyPenalty"] as const;
const SIGNATURE_KEYS = ["system","prompt","messages","attachments","timeoutMs","schema"] as const;
```

**It returns the `definition` DIRECTLY** — one object, config knobs and signature together. Internally it
still splits the `SIGNATURE_KEYS` out so `parseLlmConfig` (which is strict, and would throw on
prompt-shaped keys) sees only a config bag, then layers them back on; that split never reaches a caller.
This is possible because `llmConfig` lives *in* `llm`, where it can name the AI-SDK prompt types
precisely.

#### `ConfigurationRegistry` / `MapConfigurationRegistry`

```ts
interface ConfigurationRegistry {
  get(id: string): ConfigLayer | undefined;
  idOf?(config: ConfigLayer): string | undefined;   // reverse lookup — provenance only
}
class MapConfigurationRegistry implements ConfigurationRegistry {
  set(id, config): this; get(id): ConfigLayer | undefined;
}
```

Named presets resolved into a resolution layer. `idOf` is best-effort provenance; identity is never the id.

> **Search-space helpers.** `LlmParameters`, `MakeMembersArrays<T>`, `AllKeys<T>`, `FlattenUnion<T>` are
> type-level utilities findmyprompt uses to build a search space over config dimensions. Not needed for
> normal execution.

### File inputs and generated files

Source: `files.ts`.

```ts
interface FileInput {
  mediaType: string;                     // IANA type: application/pdf, image/png, audio/mp3, …
  filename?: string;
  data: Uint8Array | { base64: string } | { url: string };   // exactly one source
}

interface GeneratedFile { mediaType: string; bytes: Uint8Array }
```

**Sources are the caller's problem.** The library takes bytes, a base64 string, or a URL the provider
fetches itself; URLs, filesystem paths, and content hashes are resolved by the caller *before* the API is
called. There is no reference form to resolve and no store to inject — binary data is a leaf value, so
hydration is the ref family's business. That is what keeps `json`, `ops`, and `llm` free of `fetch` and
`node:fs`.

A `GeneratedFile` lands in a **`blob`-kind output slot**, not in a parallel `artifacts` channel on the
outcome.

> On streaming: `DataContent` in the AI SDK is `string | Uint8Array | ArrayBuffer | Buffer` — there is no
> `ReadableStream`, so the attachment path materializes regardless. Streaming inputs pay off for FUNCTION
> ops (pipe to a file, a hash, a subprocess, an upload) and for outputs. Recorded here so "stream to an LLM
> without materializing" is not written down as a motivating example it cannot currently be.

### One-shot calls

The ergonomic direct path — no contract, no wrappers, no `exec`. Source: `call.ts`, `generate.ts`.

```ts
function executeLlmCall(def: LlmCallDefinition & { schema?: undefined }, env: CallDeps, timeoutMs?: number):
  Promise<LlmCallResult<string>>;
function executeLlmCall<T>(def: LlmCallDefinition<T>, env: CallDeps, timeoutMs?: number):
  Promise<LlmCallResult<T>>;

type LlmCallRequest<T = JsonValue> = LlmCallDefinition<T> & { env: LlmCallEnvironment };
function executeRequest<T = JsonValue>(req: LlmCallRequest<T>): Promise<LlmCallResult<T>>;

const DEFAULT_TIMEOUT_MS = 10 * 60_000;   // when neither the definition nor the caller names one
```

**Two call shapes plus an environment**: a call is a `PromptOp` (the op-graph form) or an
`LlmCallDefinition` (the direct form), each passed alongside a separate environment. The DECLARATION is
serializable and hashable; the ENVIRONMENT holds live handles and closures. They are never flattened into
one bag — flattening is what makes an informational `schemaId` look necessary and what forces a lowering
to smuggle the output schema past the type system.

**Text mode yields `string`.** The overloads discriminate on the ABSENCE of a schema, so the typing is a
property of the declaration rather than something the caller asserts — a text call produces text, not
`JsonValue`.

`timeoutMs` is an **argument**, not a required declaration field: it is a call-site concern (a deadline
clamp). Resolution is `argument ?? def.timeoutMs ?? DEFAULT_TIMEOUT_MS`.

`executeRequest` is the **only** place a declaration and its environment co-exist; it strips `env` and calls
`executeLlmCall`. See [README example 1](README.md#1-a-one-shot-structured-call).

The environment:

```ts
interface LlmCallEnvironment {
  modelRouter?: ModelRouter;                           // required to actually reach a model
  validator?: OutputValidator;                         // json's three-line seam — nothing here knows ajv
  toolExecutors?: Record<string, ToolExecutor>;        // function-tool impls, keyed by name
  abortSignal?: AbortSignal;
  schemaProfile?: (modelId: string) => ProviderSchemaProfile | undefined;  // transport-profile resolution;
                                                       // defaults to the catalog-backed profileForModelId
}
type CallDeps = LlmCallEnvironment & { modelRouter: ModelRouter };   // modelRouter required

type ToolExecutor<I = Record<string, JsonValue>, O = JsonValue> =
  (input: I, options: ToolCallOptions) => O | Promise<O>;
```

Every seam is optional; the floor is a `modelRouter`. A declared tool **with** a `toolExecutors[name]` runs
a bounded loop, **without** one is single-turn (the model's call is returned in the result, unexecuted).
Non-serializable by design — this never enters the content hash or the durable declaration.

The result — the shared envelope, this layer's payload (`LlmOutput`), this layer's metrics (`LlmMetrics`):

```ts
type LlmCallResult<T = JsonValue> = ResultWithMetrics<LlmOutput<T>, LlmFailure, LlmMetrics>;
//    = ({ value: LlmOutput<T> } | { error: LlmFailure; value?: LlmOutput<T> }) & { metrics: LlmMetrics }

interface LlmOutput<T = JsonValue> {   // what the CALL produced — the payload under `.value`
  value?: T;               // the output value: the parsed/decoded structure, or in TEXT mode the text.
                           // Absent only when nothing usable came back — the raw text the model DID
                           // produce then rides the failure (`LlmFailure.rawOutput`), not the payload.
  thinking?: ReasoningSegment[];
  toolCalls?: ToolCall[]; toolResults?: ToolResult[];
  files?: GeneratedFile[]; // FILES the model generated — they land in a `blob`-kind output slot
  finishReason: string;
  providerSessionId?: string;  // a handle to resume from, when the provider is stateful
}

interface LlmMetrics extends TokenCounts {   // satisfies exec's ExecMetrics + BudgetMetrics structurally
  durationMs: number; startMs?: number;
  costUsd: number;                            // required — "free" and "unknown" are different claims
  costSource: "provider" | "table" | "unknown";
  rawUsage?: JsonValue;                       // the provider's exact usage object, so costUsd is recomputable
}
```

The whole model trace rides in `LlmOutput` under `.value` — read the structured answer as
`isOk(r) ? r.value.parsed : r.error`. `LlmMetrics` satisfies `exec`'s `ExecMetrics` (timing) and its
`BudgetMetrics` (`costUsd`/`costSource`) structurally, without importing either, so a prompt executor's
result folds into an `ExecResult` and its metrics into a budget fold with no re-classification. There is
deliberately no parallel `artifacts` channel. `mergeLlmMetrics` sums two calls (tokens and money add, the
start is the first observation, `costSource` keeps the more authoritative of the two).

The lower transport layer and the prompt/schema helpers:

```ts
function generateStructured<T = JsonValue>(def: LlmCallDefinition<T>, env: GenerateEnvironment<T>):
  Promise<LlmCallResult<T>>;

interface GenerateEnvironment<T = JsonValue> {   // the RESOLVED TRANSPORT — internal, never persisted
  model: LanguageModel;                          // the resolved provider handle
  outgoing?: JsonSchema<T>;                      // the provider-ADAPTED schema actually sent
  postProcess?: (value: JsonValue) => JsonValue; // reverse it back to the ORIGINAL schema's shape
  validate?: (value: JsonValue, originalSchema: JsonSchema<T>) => void;
  tools?: ToolSet; toolChoice?; stopWhen?;
  providerOptions?: Record<string, JsonValue>;
  attachStructuredOutput?: boolean;              // false for a TEXT-tier model
  abortSignal?: AbortSignal;
}

function extractTokenCounts(usage: unknown): TokenCounts;

export { typedSchema };                                      // re-export — the ONE JsonSchema<T> brand
export type { JsonSchema, ReasoningSegment, TokenCounts };   // re-exports, so shapes cannot drift
```

`GenerateEnvironment` holds live handles and closures — genuinely different from a serializable
declaration — which is why it is a separate argument rather than fields merged into one bag: the `(def,
env)` split is applied at this layer too, so no decoding knob is re-listed. `postProcess` is also the seam
a `Jsonify`→decoded lift belongs at: validate the wire form, then decode.

`typedSchema<T>` threads the output type from a call's `schema` through to `LlmOutput<T>.parsed` (the brand
is phantom — runtime identity — and the Ajv boundary enforces conformance). For an `as const` schema
literal, prefer the typed layer's `InferSchema` and let `T` be derived.

Also here: **`structured.ts`** — `parseOutputSchema`, `patchSchemaForAnthropic`, `findDiscriminators`, and
`reconstructOutput`, the model-independent structured-output transforms (non-mutating: the ORIGINAL schema
survives, we send the adapted one and reconstruct against the original), and **`dispatcher.ts`** —
`installLongTimeoutDispatcher(opts?)`, the node-only undici dispatcher that keeps a long generation from
being killed by socket timeouts (idempotent; auto-installed by `createModelRouter`).

### `plan` — the dry run

Everything knowable before execution, from the local catalog — no network, no spend. Source: `plan.ts`.

```ts
function plan(def: LlmCallDefinition & { schema?: JsonSchema }): CallPlan;
interface CallPlan {
  provider: { family: ModelFamily; modelId: string };
  contentHash: string;                    // identity / memo key
  unsupportedParams: string[];            // sampling params the model would drop
  modalities?: Modalities;
  structuredOutput?: Enforcement;         // "strict" | "advisory" | "text"
  estimate: { inputTokens: number; outputTokens: number; costUsd?: number };
  issues: string[];                       // human-readable fit problems
}
```

`plan` uses the **same** acceptance gate and cost/catalog lookups `executeLlmCall` uses, so the
dry-run can never drift from what execution sends. It gates media inputs per media-type against
`modalities.input` and requested `outputModalities` against `modalities.output`. See
[README example 2](README.md#2-plan--the-dry-run-no-network-no-spend).

### Model router

Explicit `{route}/{model}` routing over native Anthropic + OpenRouter. Source: `router.ts`.

```ts
function createModelRouter(options?: ModelRouterOptions): ModelRouter;
interface ModelRouter {
  resolveModel(modelId: string, opts?: ResolveModelOptions): LanguageModel;   // the AI-SDK type
  isAnthropic(modelId: string): boolean;
}
interface ModelRouterOptions {
  anthropicApiKey?; openRouterApiKey?; skipDispatcher?;
  openRouterUsageAccounting?;             // real charged cost per response (default ON)
  openRouterStrictStructuredOutputs?;     // send strict json_schema (default OFF; Ajv is the gate)
}
interface ResolveModelOptions { strictStructuredOutput?: boolean; }
```

The `ModelRouter` interface lives **here**, in the package that can describe what it returns. It used to
sit in `core` purely so `ExecServices.modelRouter` could be typed — which meant the bottom package named an
AI-SDK concept it could not describe (an opaque `ModelHandle`) and llm had to re-narrow it. Now
`@declarative-ai/promptop` augments `ExecServices` with `modelRouter?: ModelRouter` instead, so `exec`
never learns the concept at all.

`createModelRouter` creates provider clients **lazily** — a process that only calls Anthropic never needs
an OpenRouter key. Reads `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` from the environment by default, so it
is **optional** at every seam (`options.router` → `ctx.modelRouter` → this env-key default).

Route parsing helpers:

| Export | Purpose |
| --- | --- |
| `type ModelFamily = "anthropic" \| "openrouter"` / `ModelRoute` | the two serving routes. |
| `parseModelRoute(modelId): { route; providerId }` | parse `{route}/{model}`; **throws** on a bare id. |
| `providerNativeId(modelId): string` | the route-stripped provider id (the schema-profile family key). |
| `familyForModel(modelId): ModelFamily` | the serving route. |
| `isAnthropicModel(nativeId): boolean` | true for a bare `claude-*` **native** id (native-id space only). |
| `installLongTimeoutDispatcher(opts?)` | node-only undici dispatcher for long-running calls (auto-installed by `createModelRouter` unless `skipDispatcher`). |

### Model catalog

Local pricing, capability, and modality data. Source: `model-catalog.ts`, `model-catalog-source.ts`.

Catalog behavior lives on the **`ModelInfo` class**; the process-wide catalog is the lazily-built
`ModelInfo.instance` (get/set to hydrate/override at startup). Rows are keyed on the full
`{route}/{model}` id and matched **exactly** — there is no longest-prefix fallback, so a dated/variant id
must be present as its own row.

```ts
class ModelInfo<const Rows extends readonly ModelInfoInterface[] = readonly ModelInfoInterface[]> {
  static get instance(): ModelInfo;                          // lazily built from DEFAULT_MODELS
  static set instance(inst: ModelInfo): void;                // hydrate/override the process-wide catalog
  constructor(seed: Rows, opts?: PricingOptions);

  computeCost(model, usage: UsageForCost): number | null;    // cache-aware
  computeCostUsd(model, inputTokens, outputTokens): number | null;
  hasPricing(model): boolean;
  modalities(model): Modalities | undefined;
  paramAcceptance(model): ParamAcceptance;                   // { accepts(key); acceptsReasoning }
  supportedParameters(model): string[] | undefined;
  requiredParameters(model): string[] | undefined;
  schemaProfile(model): ProviderSchemaProfile | undefined;
  lookup(model): ModelInfoInterface | undefined;
  upsert(row): void; remove(model): void; load(rows): void; list(): ModelInfoInterface[];
}
```

`ModelInfo` is **generic over its seed rows**: construct with a literal array and the `model` param of every
method is typed to that seed's exact `${route}/${model}` keys — passing an unseeded model **fails to
compile**. Construct with a plain `ModelInfoInterface[]` (or use the runtime-hydrated `ModelInfo.instance`)
and the methods accept any `string`. So callers holding a compile-time-known model get checked; runtime
string ids (config-driven) still work.

| Export | Purpose |
| --- | --- |
| `ModelInfo` | the catalog class (behavior + static `instance` singleton); generic over the seed keys. |
| `DEFAULT_MODELS` | the committed snapshot (`GENERATED_MODELS`, an `as const` tuple) the runtime uses by default. |
| `KnownModelKey` | `ModelKeyOf<typeof GENERATED_MODELS>` — the union of every `${route}/${model}` in the snapshot. |
| `CORE_SEED_MODELS` / `modelsSeed()` | the hand-maintained core seed (generator fallback) / a fresh copy of `DEFAULT_MODELS`. |
| `ModelInfoInterface` / `RateSet` / `Modalities` | a catalog row (`route` + `model` + pricing rates + `modalities` + capability). |
| `ModelKeyOf<Rows>` | the `${route}/${model}` key union a catalog's methods are typed to. |
| `keyForModel(row): string` | the `${route}/${model}` key for a row. |
| `UsageForCost` / `PricingOptions` | cache-split usage input to `computeCost`. |
| `ParamAcceptance` | `{ accepts(key): boolean; acceptsReasoning: boolean }` — the single gate `plan` and execute share. |
| `SAMPLING_PARAM_NAMES` | map of config-knob → provider param name. |
| `isReasoningModel(modelId): boolean` | OpenAI reasoning-family heuristic (cold-start capability fallback). |
| `deriveIdentity` / `canonicalIdFor` / `displayProviderFor` | id-normalization helpers. |

The **catalog-source** module (`model-catalog-source.ts`) refreshes pricing/capabilities from the network
(node-only): `makeAnthropicDocsSource`, `makeOpenRouterSource`, `parseAnthropicDocsPricing`,
`parseOpenRouterModels`, `validatePricingRows`, `sanitizePricingRows`, plus the `PricingSource`/`FetchText`
seams and the `*_URL` constants. Only needed if you refresh the catalog yourself.

**Refreshing the snapshot.** `DEFAULT_MODELS` is a committed, strongly-typed snapshot in
`src/model-catalog-data.generated.ts`. Regenerate it with **`npm run update:model-info`**
(`scripts/updateModelInfo.ts`): it seeds a catalog with `CORE_SEED_MODELS`, runs the live refresh
(Anthropic docs + OpenRouter — needs network), and writes the sorted rows back as
`export const GENERATED_MODELS = [ … ] as const satisfies readonly ModelInfoInterface[]`. The `as const`
is what preserves the literal keys so `KnownModelKey` and the strong constructor typing work — a plain
`.json` import would widen them to `string`. A source that fails to fetch/validate is skipped (the core
seed stands). Add `--seed-only` to regenerate offline from just the core seed.

### Cost estimation

Pre-call TOKEN estimates. Source: `costEstimate.ts`. For USD cost, pass the token counts to
`ModelInfo.instance.computeCostUsd(...)` directly (there is no cost wrapper here).

| Export | Purpose |
| --- | --- |
| `estimateCallTokens(prompt, system, maxOutputTokens): CallTokenEstimate` | the input/output split rate pre-admission is priced on (chars/4 + the output ceiling). It lives here rather than beside the limiter because it estimates a PROMPT's footprint; the limiter is generic counting machinery. |
| `estimateInputTokens(...texts): number` | chars/4 proxy over prompt texts. |
| `estimateOutputTokens(...)` / `OutputTokenStats` / `noteOutputTokens(...)` | rolling output-size estimate from observed history. |
| `DEFAULT_HOLD_OUTPUT_MULTIPLIER`, `MIN_USEFUL_OUTPUT_TOKENS` | tuning constants. |

### Schema & provider adaptation

Advanced — how a JSON Schema is adapted to each provider's structured-output tier. Most callers never touch
this directly (`executeLlmCall`/`plan` use it internally); it's public for consumers that need the
enforcement decision. Source: `schema/`, `structured.ts`, `reasoning.ts`, `providerConfig.ts`.

```ts
type Enforcement = "strict" | "advisory" | "text";
type SchemaNode = MutableSchema;                  // the one mutable-schema working form
function profileForModelId(modelId: string): ProviderSchemaProfile;
function adaptSchema(original: SchemaDocument, profile: ProviderSchemaProfile): AdaptResult;
function adaptSchemaCached(original: SchemaDocument, profile: ProviderSchemaProfile): AdaptResult;  // { enforce, outgoing, postProcess, notes }
```

`Enforcement` is how a schema would be enforced for a model: `"strict"` (grammar-constrained), `"advisory"`
(json_object hint + Ajv), or `"text"` (schema described in the prompt + Ajv). Also exported: the built-in
`ProviderSchemaProfile`s (`OPENAI_STRICT`, `OPENROUTER_STRICT`, `ANTHROPIC_AI_SDK`, `ANTHROPIC_RAW`,
`JSON_OBJECT`, `ADVISORY`), the `PROFILE_REGISTRY`, `profileForCaps`/`profileIdForCaps`,
`flattenForDepth`, `adaptReasoning`, and the
provider-config schemas (`configSchemaFor`, `ANTHROPIC_CONFIG_SCHEMA`, `OPENROUTER_CONFIG_SCHEMA`).

---

## `@declarative-ai/promptop`

Running a `PromptOp` through the `Executor` seam. The op SHAPE stays in
`ops` (text slots + a config slot + a schema, importing nothing from `llm`); the LOWERING to an
`LlmCallDefinition` is llm-specific and lives here, together with the `Executor` that applies it and the
three wrappers that need llm knowledge.

Conceptually the layering is `llm ← promptop ← exec`; in dependency terms `promptop` depends on `exec`,
because the prompt executor IMPLEMENTS the interface `exec` defines. Both readings hold, and nothing in
`exec` knows `PromptOp` exists.

### The lowering

Source: `lowering.ts`.

```ts
interface LoweringOptions {
  defaults?: ConfigLayer;              // provider-wide defaults: model, sampling, a system/messages preamble
  configs?: ConfigurationRegistry;     // an op's `config.configRef` resolves a preset merged UNDER the inline config
  tools?: Record<string, Tool>;        // each becomes a FUNCTION tool DECLARATION on the definition
}

function configLayerOf(config: JsonValue): { configRef?: string; inline: ConfigLayer };
function lowerPromptOp(op: PromptOp<InlineFamily>, options?: LoweringOptions): LlmCallDefinition;
```

The lowering is a plain function and the thing that applies it is an ordinary `Executor`, which is why
there is no prompt-specific runner interface: the "runner" was only ever a lowering with an executor
wrapped around it.

| Op field | Becomes |
| --- | --- |
| `op.user` | **the** prompt. With a config-layer `messages` preamble it is appended as the final user turn; otherwise it is the `prompt`. |
| `op.system` | the system instruction (wins over a config-layer `system`). |
| `op.config` | the `LlmConfiguration` surface, read as the inline config layer. A `configRef` string key resolves a preset from `options.configs`. |
| `op.output.schema` | the definition's `schema` — the same document the pre-run checker type-checked the wiring against, so the two layers cannot drift. |
| `options.tools` / `ctx.tools` | appended as function-tool **declarations**, with their executors threaded separately — which turns the call into a bounded agent loop. |

Config resolution is `defaults ← configs.get(configRef) ← the op's inline config`, merged family-aware and
strict-parsed. A config-layer `prompt` is an **error** (the prompt is the op's `user` text). It **throws**
on a malformed config; the executor turns that into a `permanent`-classified `ExecResult`, honoring the
never-throws contract.

### The prompt executor

The MINIMAL core: lower, run, map onto an `ExecResult`, and nothing else. Source: `executor.ts`.

```ts
// The op-level call, NO projection: lowering + executeLlmCall, returning the FULL LlmCallResult
// (value, thinking, finishReason, metrics). For a consumer that PERSISTS what the model produced
// (an OperationRecord's R is the payload); lowering faults resolve `permanent` — never throws.
function executePromptOp(op: InlinePromptOp, env: CallDeps, options?: LoweringOptions & { runner?: CallRunner }): Promise<LlmCallResult>;

function createPromptExecutor(options?: PromptExecutorOptions): Executor;
class PromptExecutor implements Executor {}

interface PromptExecutorOptions extends LoweringOptions {
  router?: ModelRouter;       // else the typed ctx.modelRouter, else a lazy env-key router
  runner?: CallRunner;        // the injectable call seam; defaults to the real executeLlmCall pipeline
}
type CallRunner = (def: LlmCallDefinition, env: CallDeps, timeoutMs?: number) => Promise<LlmCallResult>;

// This package declares llm's seam on ExecServices — `exec` never names an opaque ModelHandle it
// cannot describe.
declare module "@declarative-ai/exec" {
  interface ExecServices { modelRouter?: ModelRouter }
}
```

Capabilities: `structuredOutput: true, sessionResume: false, streaming: true, interactive: false,
readOnly: true, mutatesWorkspace: false, policyEnforcement: "none", memoizable: true, runtime: edge-safe`.

The core does exactly one call → one `ExecResult`, honors caller cancel, and **refuses** any unconsumed
wrapper field — a `ctx.deadline` with no `withDeadline`, a `sessionId`/`providerSessionId` with no
`withSession` — as a loud permanent failure rather than silent degradation. Cross-cutting concerns are the
composable wrappers stacked around it.

A model-generated FILE lands in a `blob`-kind output slot: when `op.output.kind === "blob"`, the
`ExecResult`'s `value` is the generated file's bytes. That is what "a produced artifact is an output slot,
not a parallel channel" means in practice.

### LLM-aware wrappers

The three wrappers that need llm knowledge — a token estimate off a prompt, model pricing, transcript
folding. The generic ones live in [`exec`](#generic-wrappers). Source: `wrappers.ts`. Each is **dual-mode**
in the same way.

| Wrapper | Signature (config form) | What it does |
| --- | --- | --- |
| `withRateLimit` | `withRateLimit({ limiter }): ExecutorWrapper` | Admit the call through the injected `RateLimiter` (concurrency slot + rate headroom) and feed the outcome back (drives AIMD). A cancel while queued prevents it from ever starting. |
| `withBudget` | `withBudget(config?): ExecutorWrapper` | The ONE billing wrapper, two modes. **Reserve mode** (default): reserve against `ctx.meter` before the call — clamping the output ceiling to what the balance affords, refusing when it cannot cover a useful minimum — then **settle** the actual cost after (a failed call still settles); feeds observed output tokens back so the next reserve in the run is better priced. **Post-charge mode** (`computeCost` present): run the inner executor, then debit `computeCost(op, result)` and fold it into the reported `costUsd` — the mode an OUTER instance above `withMemoize` uses to bill memo reuse off the hit's annotation, applying to every op kind. |
| `withSession` | `withSession(config?): ExecutorWrapper` | Resolve the op's logical `sessionId` against a `SessionStore` (from config or `ctx.sessions`): prepend the stored transcript, run, fold the reply back on success, stamp `outcome.session.id`. The session fields are **consumed** (stripped from the op sent inward), and the sent op carries the full transcript, so an inner `withMemoize` keys on the real content. Refuses `providerSessionId` outright — no current executor can thread a provider-side handle. |

`withSession` must sit **outside** `withMemoize` (which throws at composition time if it would wrap a
session layer). See [README example 3](README.md#3-the-contract-path--one-seam-a-composed-executor-stack).

---

## `@declarative-ai/tools`

Workspace-backed agent tools — the impls that make a composed prompt executor a coding agent. They operate
on `ctx.workspace` with a path-escape guard, and they are what keeps `node:*` out of `exec`.

```ts
const allTools: Record<string, Tool>;      // every tool, keyed by logical name
const fsTools: Record<string, Tool>;       // read_file / write_file / edit_file / list_dir
const searchTools: Record<string, Tool>;   // grep / glob
const shellTools: Record<string, Tool>;    // run_command

const readFileTool: Tool; const writeFileTool: Tool; const editFileTool: Tool; const listDirTool: Tool;
const grepTool: Tool; const globTool: Tool; const runCommandTool: Tool;

function requireWorkspace(ctx: ExecServices): Workspace;              // throws a clear error when absent
function resolveInWorkspace(root: string, rel: string): string;       // throws if the path ESCAPES the root
```

`read_file` / `list_dir` / `grep` / `glob` declare `readOnly: true`; `write_file` / `edit_file` /
`run_command` do not — which is exactly what the `read-only`/`plan` permission profiles gate on. Register
them into `registry.tools` and reference by logical name from a state's `environment.tools`.

`resolveInWorkspace` compares prefixes on the *normalized* root plus a path separator, so `/repo-evil`
never counts as inside `/repo` (SPEC §7.2 — "may not access files outside the project").

---

## `@declarative-ai/hw`

The hierarchical-workflow formalism (see [SPEC.md](SPEC.md) for the semantics): loader/desugarer,
type-checking validator, evaluator engine, and the executor exposing a workflow run as an `Executor`.

Since the ops redesign a state's operation **is** an `Operation<InlineFamily>` and its wiring **is**
`Parameter` bindings. The `runtime`/`function` blocks and the `WiringValue` expression strings are gone;
what an author writes is *sugar* the loader lowers to base `Ref<InlineFamily>` cases, so the checker,
hasher, and engine only ever see literals and producer edges. The expression DSL survives exactly where
control flow needs it: transition guards, `limits`, and the `{ expr }` binding leaf.

### The workflow executor

Run a state-file bundle as an `Executor`. Source: `executor.ts`.

```ts
function createWorkflowExecutor(options: WorkflowExecutorOptions): WorkflowExecutor;
class WorkflowExecutor implements Executor {}

interface WorkflowExecutorOptions {
  definition: HierarchicalWorkflowDefinition;    // the authored bundle THIS executor runs
  registry: CapabilityRegistry;                  // functions / skills / tools
  prompt?: Executor;                             // what a PromptOp inside the workflow dispatches to
  persistence?: Persistence;
}
interface HierarchicalWorkflowDefinition { rootId: string; states: Record<string, StateDef>; }
```

**The bundle is held at construction**, not re-supplied per run — which is what it always was in practice:
a workflow's identity is its snapshot, not a payload the caller repeats. A run is started by a `FunctionOp`
whose bound inputs are the workflow's declared inputs. The executor loads + validates the bundle, wires
abort/timeout, runs the engine, folds child cost/calls into
its metrics, and maps termination to an `ExecResult` (`deadline`/`canceled`/`permanent` classifications for the
failure cases). Capabilities: `structuredOutput: true, interactive: true, memoizable: true,
sessionResume: false, streaming: true, readOnly: false, mutatesWorkspace: false, policyEnforcement: "none",
runtime: edge-safe`. See [README example 8](README.md#8-hierarchical-workflows).

`prompt` is typed as a plain `Executor`, so hw never learns that a `PromptOp` HAS an llm lowering — which
is what keeps the AI SDK out of this package's dependency graph.

Identity helpers:

```ts
function workflowIdentify(definition: HierarchicalWorkflowDefinition): (op: Operation<InlineFamily>) => string;
function workflowMemoKey(definition: HierarchicalWorkflowDefinition, op: Operation<InlineFamily>,
                         workspaceTreeHash?: string): string;
```

`workflowIdentify` folds the bundle's SNAPSHOT hash (SPEC §12) with the op's own hash (which carries the
run's resolved inputs) — which is why `withMemoize` never has to brute-force-canonicalize an opaque bundle.
To memoize a workflow run: `withMemoize({ cache, identify: workflowIdentify(definition) })`.

### Bundles & identity

Load, desugar, and validate a state-file map. Source: `loader.ts`, `validate.ts`.

```ts
function loadBundle(files: Record<string, unknown>, rootId: string): WorkflowBundle;
function loadBundleFromDir(dir: string, rootId: string): Promise<WorkflowBundle>;   // node-only convenience
function snapshotHash(bundle: WorkflowBundle): string;         // the versioning identity (SPEC §12)
function stateIdFromPath(relPath: string): string;             // path -> state id (drop suffix)
class WorkflowLoadError extends Error { readonly stateId?: string; }

function validateBundle(bundle: WorkflowBundle, env?: ValidationEnvironment): ValidationReport;
interface ValidationReport { errors: ValidationIssue[]; warnings: ValidationIssue[]; }
interface ValidationIssue { stateId: string; path: string; message: string; }
interface ValidationEnvironment {
  functions?: Pick<FunctionRegistry<never>, "get" | "has">;  // the registry the bundle will run against
  strict?: boolean;      // an unregistered `functionRef` is an ERROR rather than a warning (lint/CI)
  interactive?: boolean; // assert a NON-interactive context: an interactive entry is then an error
}
```

`loadBundle` parses each state file, **desugars** it (see [Binding desugaring](#binding-desugaring)), and
restricts the bundle to the transitive closure reachable from `rootId` — so the snapshot hash never varies
with unrelated files lying around the workflow dir. A declared `id` that disagrees with the path-derived
one is a load error; a missing child is left to the validator, so every problem is reported at once.

`snapshotHash` hashes the bundle's **`source`** (the states as authored, pre-desugaring), which is why
`WorkflowBundle` keeps it: the snapshot identity is what the author wrote, so improving the lowering never
invalidates a stored snapshot. A derived (previously absent) `id` is stripped before hashing.

`validateBundle` is the **tier-2 check** — once a workflow exists as values every schema is concrete, so the
wiring can be fully type-checked before anything runs. Three checks replace "the expression parses":

1. **Binding compatibility** — every producer's output schema must be an
   [`isSubschema`](#schema-subtyping) of the consuming slot's schema. Conservative: an unmodeled keyword or
   a union rejects with a precise reason, never a silent pass.
2. **Expression typing** — every `{ expr }` leaf and every `when` guard is inferred; a guard that does not
   infer to `boolean` is an error (strict, no truthiness coercion), and a declared `schema` on an expr leaf
   is an *assertion* checked against the inferred type.
3. **Reachability** — a reference to a producer not provably run on every path to its evaluation point is an
   error, so `T | undefined` never propagates silently. Every `sequence` member counts as proven whether or
   not it is `async` (an in-flight producer resolves to `PENDING`, which is a runtime *park*, not a value
   that can be permanently missing); a child reachable only through a conditional transition does not. A
   declared `default` — or, for an output, `optional: true` — is the explicit opt-out.

**Validation is a function of *(document, registry)*.** That is the deliberate consequence of capabilities
being required and total on every registry entry: a `functionRef` naming nothing registered, and "an
interactive function in a search-only workflow", are only decidable by reading the registry. `strict` is off
by default, and that default is load-bearing — `validateBundle` checks the WHOLE document, but a state the
run never enters never needs its function, and *not* registering a function is the documented way a search
context refuses a human gate. A lint/CI surface turns it on; the pre-run gate does not.

### Operation dispatch & ports

A state's operation dispatches **by op kind** through the typed
[`CapabilityRegistry`](#the-capability-registry) (`@declarative-ai/exec`) — no per-state binding table:

- a **`PromptOp`** is dispatched to the injected `prompt` `Executor`
  ([`@declarative-ai/promptop`](#the-prompt-executor)), typed as a plain `Executor`;
- a **`FunctionOp`** runs through `registry.functions` — host code, interactive UI, sub-workflows, composite
  units, and delegated agent adapters alike, since all of them are registry entries;
- a `user` slot carrying the `skill:` prefix resolves its template through `registry.skills`;
- a state's `environment.tools` (logical names) resolve through `registry.tools`, and the executables are
  handed to the operation (an unregistered name is a permanent failure). A composed prompt operation feeds
  them into its bounded tool loop; a **delegated** adapter (entry capability
  `policyEnforcement: "callback"`) is handed **raw** tools — no `withPermission` wrap — so its own gate
  isn't double-applied.

Interactive UI is a `host` registry entry with `capabilities.interactive`: it drives its renderer
internally, so there is no separate interaction seam on the engine.

The ports (apps implement these). Source: `ports.ts`.

| Export | Purpose |
| --- | --- |
| `Persistence` | `record(event: EngineEvent, atMs: number): void` — the durable run-record sink (SPEC §10.2). |
| `InMemoryPersistence` | the bundled buffering implementation (embedding & tests); exposes `events`. |
| `EngineEvent` | the run-record event union: `instance.entered`, `instance.blocked`, `operation.started`, `operation.completed`, `operation.failed`, `transition.taken`, `child.superseded`, `instance.terminated`. |
| `OperationKind` | `"prompt" \| "function"` — the two operation types (the event `op` field). |
| `ArtifactRef` / `isArtifactRef` | `{ artifact: true; name; format?; content?; path? }` — an artifact value flowing through workflow inputs/outputs. |

### The engine (lower level)

The evaluator engine underneath the executor. Most callers use the executor; use the engine directly for
custom hosting. Source: `engine.ts`.

```ts
class WorkflowEngine {
  constructor(config: EngineConfig);
  run(options: WorkflowRunOptions): Promise<WorkflowRunResult>;
}
interface EngineConfig {
  bundle: WorkflowBundle;
  registry: CapabilityRegistry;    // functions / skills / tools
  prompt?: Executor;               // what a PromptOp dispatches to; absent => a prompt state fails
  validator?: OutputValidator; persistence?: Persistence; services?: ExecServices; clock?: Clock;
  onEvent?: (event: EngineEvent) => void;
  // Tool-call permissions (DESIGN.md §5.1). `approve` collects a human decision on `ask`;
  // absent => a state's tools run UNGUARDED. `baseline` is the workflow-wide default; `process` is the
  // host-owned overlay carrying `always` decisions across runs; `smart` maps a tool to its arg-inspecting
  // policy; `profiles` supplies custom profile predicates by name.
  permissions?: { approve?: Approver; baseline?: PermissionBaseline; process?: Map<string, PermissionMode>;
                  smart?: Record<string, SmartApprover>; profiles?: Record<string, ProfilePredicate> };
  // Per-session workspace resolver (DESIGN.md §5.1): a state's `environment.session` -> its
  // workspace, for fan-out isolation. undefined => the run-level `services.workspace`.
  workspaceFor?: (sessionId: string) => Workspace | undefined;
}
interface WorkflowRunOptions { inputs: Record<string, unknown>; abortSignal?: AbortSignal; }
interface WorkflowRunResult {
  outcome: TerminationOutcome; outputs?: Record<string, ResolvedValue>; failure?: Failure;
  artifacts: ArtifactRef[]; metrics: { childLlmCalls: number; childCost: number; durationMs: number };
}
```

The engine no longer renders a bespoke operation payload: it resolves the state operation's bindings against
the run, then dispatches the resolved op by kind. Conversation preambles, sessions, and permission gating
attach exactly where they did before.

### State-file format types

The typed shape of a state file (see [SPEC.md](SPEC.md) for authoring semantics). Source: `format.ts`, which
also re-exports the op vocabulary (`Operation`, `PromptOp`, `FunctionOp`, `Parameter`, `NamedParameter`,
`Ref`, `RefKind`, `InlineFamily`) so authors import one set of names.

#### The authored state

```ts
interface StateDef {
  id?: string;                                   // = the path-derived id; may be omitted
  label?: string; description?: string;
  inputs?:  Record<string, ParameterDecl>;       // config knobs are inputs with a `default` (no separate params)
  outputs?: Record<string, NamedParameterDecl>;
  operation?: OperationDecl;                     // ONE operation (a state with only children is a composite)
  environment?: EnvironmentDecl;                 // session / tools / conversation / permissions
  children?: Record<string, ChildDecl>;
  sequence?: string[];
  transitions?: TransitionDecl[];
  limits?: LimitsDecl;
}
```

A state has **one `operation`** and a sibling **`environment`**. That split is the point: the operation is
what the state *is* (and is part of its identity); the environment is *how it runs* (session, tools,
conversation preamble, permissions) and is not.

```ts
type OperationDecl = PromptOpDecl | FunctionOpDecl;

interface PromptOpDecl {
  kind: "prompt";
  prompt?: { template?: string; skill?: string };   // exactly one — both fill the PromptOp `user` slot
  system?: string;
  config?: Record<string, JsonValue>;               // the LlmConfiguration surface (model, sampling, configRef)
  input?: Record<string, ParameterDecl>;            // render variables are bound inputs → `{{inputs.*}}`
  output?: NamedParameterDecl;
}
interface FunctionOpDecl {
  kind: "function";
  function: string;                                 // registry name — host function OR runtime adapter
  config?: Record<string, JsonValue>;               // the authored surface, bound as the op's `config` input
  input?: Record<string, ParameterDecl>;
  output?: NamedParameterDecl;
}

interface EnvironmentDecl {
  session?: string;                                 // logical session id; absent => the run's default session
  tools?: string[];                                 // logical names resolved through registry.tools
  conversation?: { mode: ConversationMode; artifacts?: string[] };
  permissions?: { profile?: PermissionProfile; default?: PermissionMode; tools?: Record<string, PermissionMode> };
}
type ConversationMode = "full_history" | "summary" | "fresh" | "selected_artifacts";
```

A `claude-code` invocation, a sub-workflow, and an ordinary host function are all `FunctionOpDecl` — the
resolved registry **entry's** capabilities distinguish them, never the declaration.

#### Slots and wiring

```ts
interface ParameterDecl {
  kind?: RefKind;             // defaulted: a `type: "string"` schema => "text", else "json"
  schema?: JsonSchema;
  binding?: BindingDecl;      // may still be sugar
  index?: number;
  default?: JsonValue;        // a free slot's fallback — and the explicit reachability opt-out
  optional?: boolean;         // SPEC §4.1: slots are required by default
  description?: string;
}
interface NamedParameterDecl extends ParameterDecl { name?: string; }

interface ChildDecl { state: string; inputs?: Record<string, BindingDecl>; async?: boolean; }
interface TransitionDecl { to: string; when?: string; }        // `when` must INFER to boolean
interface LimitsDecl { max_iterations?: number; timeout?: number; }
```

`ParameterDecl`/`NamedParameterDecl` replace the old `FieldSchema`/`OutputFieldSchema` — the same
information (schema, optionality, default, description) in the op vocabulary, with `OutputFieldSchema.from`
now being the output slot's `binding`. There is no `passthrough` keyword and no `type: "artifact"`: an
unconstrained slot (no `schema`) *with* a binding **is** a passthrough, and an **artifact slot is a
`blob`-kind slot** — `{ kind: "blob", schema: { type: "string", contentMediaType: "markdown" } }` — with
the media type naming the content format. There is no bespoke artifact marker: the content keywords are
ordinary JSON Schema, which the subtype checker treats as annotations because they carry KIND information
rather than a constraint.

> **Note (loader gap).** The engine reads `slot.kind === "blob"`, and `ops`' `kindFor` would derive that
> from `contentEncoding`/`contentMediaType` — but `loader.ts` still has its own `kindOf` that only
> distinguishes `text`/`json`. So an authored artifact slot must currently spell out `kind: "blob"`; the
> schema keywords alone do not yet imply it.

#### The loaded form

```ts
interface LoadedState extends Omit<StateDef, "operation" | "inputs" | "outputs" | "children"> {
  id: string;
  inputs?:  Record<string, Parameter<InlineFamily>>;
  outputs?: Record<string, NamedParameter<InlineFamily>>;
  operation?: Operation<InlineFamily>;                 // a real op — the authored sugar is gone
  children?: Record<string, LoadedChild>;
  slotMeta?: Record<string, SlotMeta>;                 // keyed "<section>.<name>"
}
interface SlotMeta { default?: JsonValue; optional?: boolean; description?: string; }
interface LoadedChild { state: string; inputs?: Record<string, Ref<InlineFamily>>; async?: boolean; }

interface WorkflowBundle {
  rootId: string;
  states: Record<string, LoadedState>;
  source?: Record<string, StateDef>;      // the states AS AUTHORED — what snapshotHash hashes
}
```

`SlotMeta` is the per-slot authoring metadata the op model deliberately doesn't carry (defaults,
optionality, docs), kept **alongside** the op so it can never affect the op's identity; the engine reads it
when filling free slots.

| Export | Purpose |
| --- | --- |
| `TerminationOutcome` | `"success" \| "error" \| "canceled" \| "timeout"`. |
| `RunStatus` | the state-run status enum (SPEC §10.1). |
| `TERMINATE_TARGETS` | the terminal transition targets (`terminate.success`, …). |
| `REF_NAMESPACES` | `inputs`, `outputs`, `children`, `artifacts`, `conversations` — the **data** namespaces authored bindings address, and which `{ expr }` leaves may read. |
| `GUARD_NAMESPACES` | `run`, `limits` — control-flow scalars reachable from guards only, never from a reference binding. |
| `CONTEXT_NAMESPACES` | `[...REF_NAMESPACES, ...GUARD_NAMESPACES]`. The old `function.*` namespace is **gone**: a function state's result is an ordinary state output, so guards read `outputs.*` / `children.<key>.outputs.*` uniformly. |

### Binding desugaring

What an author may write in a binding slot, and what the loader lowers it to. Source: `loader.ts`,
`format.ts`.

```ts
type BindingDecl =
  | Ref<InlineFamily>                             // the base cases: {text} | {json} | {result} | {refs} | {op}
  | { child: string; output?: string }
  | { input: string }
  | { expr: string }
  | { artifact: string }
  | { conversation: string; message?: number };

function desugarBinding(binding: BindingDecl, where: string, stateId: string): Ref<InlineFamily>;
function desugarOperation(decl: OperationDecl, stateId: string,
                          outputs?: Record<string, NamedParameterDecl>): Operation<InlineFamily>;
function desugarState(id: string, def: StateDef): LoadedState;
```

Every sugar lowers to a **producer edge on a well-known resolver function** — after this pass there is no
special wiring case left for the checker or the engine to know about:

```ts
const RESOLVER_REFS = {
  expr:         "expr.eval",        // evaluate a DSL expression; output schema = the inferred type
  select:       "select",           // project one property off a producer's object output
  scope:        "scope.get",        // read a declared inputs.* value by name
  artifact:     "artifact.get",     // read a session-owned artifact by name
  conversation: "conversation.get", // read a session transcript, or one message of it
} as const;
const RESOLVER_REF_VALUES: readonly string[];     // all of them, for registry seeding + validator checks
```

| Authored | Lowers to |
| --- | --- |
| `{ child: "plan" }` | `{ op: "plan" }` — a producer edge naming the declared child by its local key. |
| `{ child: "plan", output: "steps" }` | that edge wrapped in a `select` projection (hw states lower to single-object-output ops, so a named output *is* a property select). |
| `{ input: n }` | a `scope.get` producer — the model's by-name free-slot fill, made explicit. |
| `{ expr: "…" }` | an `expr.eval` producer. Semantically an expression **is** a pure `FunctionOp` whose output schema is its inferred type, which is why binding type-checking applies to expr leaves with no special case. |
| `{ artifact: n }` / `{ conversation: s, message? }` | an `artifact.get` / `conversation.get` producer reading session-owned resources. |

A `PromptOpDecl`'s render variables are authored directly as **bound input slots** (the operation's
resolved inputs ARE the template's `{{inputs.*}}` scope — there is no separate `params`), and a
`FunctionOpDecl`'s `config` lowers into a bound `config` input — the "authored surface rides bound inputs"
move that keeps the op shape exactly findmyprompt's. A `skill` prompt is marked in the same `user` slot
with a prefix rather than an extra op field:

```ts
const SKILL_PREFIX = "skill:";
const skillRef: (name: string) => string;          // `skill:<name>`
const skillNameOf: (user: string) => string | undefined;
```

Two more modules back the checker and the engine — `inferExpr.ts` and `resolve.ts` — and both **are**
re-exported from the package root, because each is reusable against a custom engine or a lint surface:

```ts
// inferExpr.ts — expression TYPE inference (what makes `{ expr }` not a typing hole)
type ExprScope = Record<string, JsonSchema>;                       // root name -> that namespace's schema
interface InferResult { schema: JsonSchema; unresolved: string[][]; }
function inferExpression(expr: Expr, scope: ExprScope): InferResult;
function joinSchemas(a: JsonSchema, b: JsonSchema): JsonSchema;    // least schema accepting both branches
function isBooleanSchema(s: JsonSchema): boolean;                  // what a `when` guard must satisfy
function isUniversalSchema(s: JsonSchema | undefined): boolean;
const ANY_SCHEMA: JsonSchema;                                      // `{}` — "any value"

// resolve.ts — turning bindings into values against a run
type Resolved = { value: ResolvedValue } | Pending | { error: string };
interface ResolutionScope {
  exprContext: Record<string, unknown>;
  childOutputs(key: string): JsonValue | Pending | undefined;      // already ran => the memo hit
  scopeValue(name: string): JsonValue | undefined;                 // a declared input value, by name
  artifact(name: string): JsonValue | undefined;
  conversation(session: string, message?: number): JsonValue | undefined;
}
function resolveRef(ref: Ref<InlineFamily>, scope: ResolutionScope): Resolved;
function resolveInputs(input: Record<string, Parameter<InlineFamily>>, scope: ResolutionScope):
  { values: FunctionInputs } | Pending | { error: string };
function isResolvedValue(r: Resolved): r is { value: JsonValue };
function isResolveError(r: Resolved): r is { error: string };
```

Member access projects property schemas, each operator has a fixed signature (comparison → boolean,
arithmetic → number, logical → the join of its branches), and an unresolved reference is *reported*, not
silently widened. `resolveInputs` resolves every **bound** parameter and leaves free slots to the caller
(which fills them by name); `PENDING` short-circuits — the state parks until the producers resolve, which
is the dataflow join.

### Expression language & inference

The pure transition/guard expression language (SPEC §6). Source: `expr.ts`.

```ts
function evaluateExpression(src: string, context: Record<string, unknown>): ExprValue;
function parseExpression(src: string): Expr;
function evaluate(expr: Expr, context: Record<string, unknown>): ExprValue;   // value may be PENDING
function referencesOf(expr: Expr): string[][];      // the paths an expression reads
class ExprError extends Error {}
const PENDING: unique symbol;                        // an async child's not-yet-resolved reference
function isPending(v: unknown): v is Pending;
```

JavaScript evaluation semantics with one deviation — property access on `undefined`/missing yields
`undefined` (implicit optional chaining) — and no side effects (no calls, imports, loops, or I/O). A
reference to a started-but-unfinished async child is `PENDING`, which skips a transition for that round
(SPEC §10.4). `referencesOf` powers the dataflow join (waiting on the inputs an expression reads).

The DSL is no longer the wiring default — it is a *binding kind* (`{ expr }`) plus the guard language — and
it is type-checked, not merely parsed: see `inferExpression` under
[Binding desugaring](#binding-desugaring).

---

## `@declarative-ai/agents-api`

**Delegated** agents reached through an in-process SDK: an agent that runs its own loop, so we configure it
(a prompt, a workspace `cwd`, an allowed-tools list, a permission mode) and route its native tool-approval
callback back through *our* approver, keeping the human-gate UX uniform across runtimes. `claude-code`
split by INVOCATION MECHANISM; the CLI-driven sibling is
[`@declarative-ai/agents-cli`](#declarative-aiagents-cli), and this package owns the normalized
`AgentQuery` seam both share. Source: `runtime.ts`, `seam.ts`, `sdkQuery.ts`.

There is no `Runtime` interface and no normalized runtime-op payload, so this is **a `runtime` registry
entry like any other**:

```ts
function createClaudeCodeFunction(options?: ClaudeCodeFunctionOptions): {
  capabilities: RuntimeCapabilities;
  run: (inputs: FunctionInputs, ctx: ExecServices) => Promise<Result<string>>;
};

interface ClaudeCodeFunctionOptions {
  query?: AgentQuery;                            // the agent seam; default: the lazily-loaded SDK
  capabilities?: RuntimeCapabilities;            // override the advertised entry capabilities
  injectTools?: boolean;                         // MCP-inject ctx.tools so the agent calls OUR impls (default true)
  nativeTools?: Record<string, NativeToolRef>;   // per-name override: use the agent's own built-in instead
}

const DELEGATED_CAPS: RuntimeCapabilities = {
  interactive: true, readOnly: false,
  mutatesWorkspace: true, memoizable: false, structuredOutput: false,
  policyEnforcement: "callback", sessionResume: false, streaming: true, runtime: "node",
};

interface ClaudeCodeConfig {                     // the authored surface, bound as the op's `config` input
  permissionMode?: AgentPermissionMode;
  sessionId?: string;                            // approval scope key for ctx.approve (default "delegated")
}

class ClaudeCodeError extends Error { constructor(message: string, readonly canceled = false); }
```

Register and author it:

```ts
const agent = createClaudeCodeFunction();
registry.functions.set("claude-code", runtimeFunction(agent.run, agent.capabilities));
const op = runtimeOp({ runtime: "claude-code", prompt: "…", config: { permissionMode: "plan" } });
// => exactly { kind: "function", functionRef: "claude-code", input: { prompt, config }, output }
```

The op shape carries **no runtime marker at all** — permission gating and search refusal read the resolved
registry entry's `DELEGATED_CAPS`. The function reads its `prompt` input as the agent instruction and its
`config` input as the authored surface; `ctx.workspace.root` is the cwd, `ctx.tools` the resolved tool set,
`ctx.approve` the human gate, `ctx.abortSignal` cancellation, and `ctx.meter` the wallet. Its output value
is the agent's answer **text**.

Two behaviors worth noting:

- **It resolves a `Result`, not an exception.** The adapter still throws internally — an agent SDK is an
  exception-shaped world — but the throw is classified at the seam, so a 429 or an abort inside the agent's
  own loop becomes `network-retriable`/`canceled` rather than a blanket `permanent`. A `ClaudeCodeError`
  raised for cancellation carries `name: "AbortError"`.
- **Spend is reported, not reserved.** A delegated agent spends inside its own loop, so the adapter reports
  what it cost on the `Result`'s `metrics` — `cost`, `childCost`, `childLlmCalls: 1`, `costSource: "provider"`
  — which is the channel `Result.metrics` exists for. Without it an agent's spend simply vanished once
  dispatch moved to the function registry.

Tool resolution: each logical tool is either **MCP-injected** (our impl, ctx-bound — so a `bash`/`read_file`
behaves identically to a composed prompt operation with the same tools) or **native** (the agent's own
built-in, aliased via `nativeTools[name].native`). A tool is native when `injectTools: false` or it has a
`nativeTools` entry; everything else is injected. The engine hands a delegated runtime **raw** tools and
authorization flows through `canUseTool` → `ctx.approve`, so injected tools are never double-gated.

The injectable seam (`seam.ts`) — implement it to test or swap the backend; the default lazily loads the
optional `@anthropic-ai/claude-agent-sdk` peer:

```ts
type AgentQuery = (opts: AgentQueryOptions) => AsyncIterable<AgentStreamMessage>;

type AgentPermissionMode = "default" | "plan" | "acceptEdits" | "bypassPermissions";
interface AgentToolRequest { toolName: string; input: FunctionInputs; }
type AgentPermissionDecision = { allow: true } | { allow: false; reason?: string };
type AgentPermissionCallback =
  (req: AgentToolRequest, opts: { signal: AbortSignal }) => Promise<AgentPermissionDecision>;

interface InjectedTool {
  description?: string;
  inputSchema: JsonSchema;
  run: (input: FunctionInputs) => JsonValue | Promise<JsonValue>;
}

interface AgentQueryOptions {
  prompt: string; cwd?: string;
  allowedTools?: string[];                       // native names the agent may use
  mcpTools?: Record<string, InjectedTool>;       // our impls, injected over MCP
  permissionMode?: AgentPermissionMode;
  canUseTool?: AgentPermissionCallback;
  abortSignal?: AbortSignal;
}
interface AgentResult { text: string; costUsd?: number; }
interface AgentStreamMessage { type: "result" | "assistant" | "other"; result?: AgentResult; error?: string; }
```

`sdkAgentQuery` is the default implementation mapping this onto the SDK's `query()`. Keeping the adapter
written against this interface rather than the SDK's is what lets the package be built and tested with
neither the SDK installed nor an API key.

---

## `@declarative-ai/agents-cli`

The same adapter over a **CLI subprocess**. It reuses `createClaudeCodeFunction` wholesale and swaps only
the `query` seam, so a workflow authored against one adapter runs against the other unchanged.

```ts
function createCliAgentFunction(options?: CliAgentFunctionOptions): ReturnType<typeof createClaudeCodeFunction>;
interface CliAgentFunctionOptions extends Omit<ClaudeCodeFunctionOptions, "query">, CliAgentOptions {}

const CLI_DELEGATED_CAPS: RuntimeCapabilities;      // = DELEGATED_CAPS — policyEnforcement: "callback"
const CLI_CONFIG_ONLY_CAPS: RuntimeCapabilities;    // = DELEGATED_CAPS + policyEnforcement: "config"

function createCliAgentQuery(config?: CliAgentOptions): AgentQuery;
interface CliAgentOptions {
  command?: string;               // the executable; default "claude"
  args?: string[];                // extra argv appended after the generated flags
  spawn?: SpawnProcess;           // the process seam; default a lazy `node:child_process` spawn
  startBridge?: StartMcpBridge;   // the MCP-bridge seam; default a loopback HTTP server
}

type SpawnProcess = (argv: string[], opts: { cwd?: string }) => AgentProcess;
interface AgentProcess { lines: AsyncIterable<string>; kill(): void; exit: Promise<number> }

function needsBridge(opts: AgentQueryOptions): boolean;
function cliArgv(opts: AgentQueryOptions, config?: CliAgentOptions, bridgeUrl?: string): string[];
```

**Its enforcement model is `callback`, the same guarantee as the SDK adapter's** — the mechanism differs,
not the promise. The CLI cannot call back into our process directly, so the adapter stands up a small
**MCP bridge** the agent reaches over `--mcp-config`, and passes `--permission-prompt-tool` so the CLI
*asks* before each gated tool-use rather than deciding on its own. Host-implemented tools ride the same
bridge, which is why `injectTools` means the same thing here as there. `CLI_CONFIG_ONLY_CAPS` is the honest
record for a caller that deliberately runs **without** an approver, on an up-front posture alone
(`permissionMode` / `allowedTools`) — no bridge, no callback.

The bridge is stood up **before** the process is spawned, and a failure to start **refuses the run**:
running the agent anyway would leave it under its own defaults while the caller believes its approver is in
force, and silence is exactly the failure mode this path exists to remove.

The MCP protocol pieces are exported too, so a host can stand up its own bridge:

| Export | Purpose |
| --- | --- |
| `MCP_SERVER_NAME` / `APPROVAL_TOOL` / `PERMISSION_PROMPT_TOOL` / `mcpToolName(tool, server?)` | the server + tool naming the CLI is pointed at. |
| `mcpConfigJson(url, server?)` | the `--mcp-config` payload. |
| `APPROVAL_INPUT_SCHEMA`, `ApprovalRequest`, `parseApprovalRequest`, `approvalResponseText`, `malformedApprovalResponseText` | the approval tool's wire contract. |
| `McpToolDescriptor` / `toolDescriptors(spec)` / `injectedToolAllowEntries(tools)` / `handleToolCall(...)` | exposing host tools over the bridge, and the `--allowedTools` entries their MCP-qualified names need. |
| `McpBridge` / `McpBridgeSpec` / `StartMcpBridge` / `defaultStartMcpBridge` / `SDK_MISSING` | the bridge seam and its default loopback-HTTP implementation. |
