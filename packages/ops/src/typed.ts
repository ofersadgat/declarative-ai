/**
 * The typed layer (API.md, "The typed layer"): TS generics OVER the runtime JSON-Schema typing, with
 * zero runtime cost. Runtime schemas stay the source of truth on every slot; this layer adds
 * compile-time inference (`FromSchema` over `as const` documents) and typed op builders whose
 * producer wiring is checked by the compiler.
 *
 * Known inference limits (API.md, "The typed layer"): only compile-time-LITERAL schemas can be interpreted. A
 * runtime-resolved `$ref`, a dynamically-built schema, or a composition past TypeScript's
 * instantiation depth degrades to {@link Widened} — surfaced, never a silent `unknown`.
 * Untyped/dynamic construction (deserialized ops, an optimizer mutating op graphs) bypasses
 * these builders entirely and relies on the pre-run/runtime schema checkers; both layers
 * check the SAME schemas, so they cannot drift.
 *
 * **This layer is `InlineFamily`-only** (API.md, "The typed layer"). `FromSchema` requires a compile-time
 * literal, and an `Id` is opaque to the compiler — so an id-family consumer gets runtime schema
 * checking and pre-run static validation, never compile-time inference. That is by design, but it must
 * be said out loud: "the typed layer" does not apply to both families.
 */
import type { FromSchema, JSONSchema } from "json-schema-to-ts";
import type { JsonSchema, JsonValue, ReadonlyJsonValue, SchemaDocument } from "@declarative-ai/json";
import type {
  FunctionOp,
  InlineFamily,
  NamedParameter,
  Operation,
  Parameter,
  PromptOp,
  Ref,
  RefFamily,
  RefKind,
} from "./model";
import { kindFor } from "./model";
import type { FunctionInputs, FunctionRegistry, HostCapabilities, PureCapabilities } from "./registry";
import { failureOf, hostFunction, liftThrowing, pureFunction, PURE_CAPABILITIES } from "./registry";

declare const WidenedTag: unique symbol;
declare const OpInputType: unique symbol;
declare const OpOutputType: unique symbol;

/**
 * The degradation marker (§4): this schema was not a compile-time literal the type system
 * could interpret (runtime `$ref`, dynamically-built document, or an instantiation-depth
 * blowout). Deliberately NOT `unknown`: it names the situation in hovers and errors, and
 * consuming it requires an explicit cast — the runtime schema checker is the guarantee here.
 */
export interface Widened<T = JsonValue> {
  readonly [WidenedTag]?: T;
}

/** Strip the `[x: string]: unknown` index signature `FromSchema` adds to OPEN object schemas
 *  (no `additionalProperties: false`), recursively — the declared properties are the inferred
 *  surface; dynamic extras ride the runtime schema check, not an `unknown` index. */
type CleanIndex<T> = T extends readonly unknown[]
  ? { [K in keyof T]: CleanIndex<T[K]> }
  : T extends object
    ? { [K in keyof T as string extends K ? never : number extends K ? never : K]: CleanIndex<T[K]> }
    : T;

/**
 * Type-level interpretation of a schema document: `FromSchema` for compile-time literals,
 * {@link Widened} where interpretation degrades (§4).
 */
export type InferSchema<S> = S extends JSONSchema
  ? [FromSchema<S>] extends [never]
    ? Widened
    : unknown extends FromSchema<S>
      ? Widened
      : CleanIndex<FromSchema<S>>
  : Widened;

type MaybePromise<T> = T | Promise<T>;

/**
 * A typed function definition (§4): the schemas are the RUNTIME truth, `I`/`O` the inferred
 * compile-time truth of the same documents. `Ctx` is the async context an orchestrating impl
 * receives (`void` for pure functions — the impl just ignores the argument).
 *
 * `R` is the impl's ACTUAL return type, kept as a parameter rather than collapsed into
 * `MaybePromise<O>`, so "is this impl async?" survives to the registration site. Without it, an
 * async impl registered on the `pure` path type-checked and then stored the PROMISE as the value —
 * a thenable flowing on into `ResolvedValue`, memo keys, and `acceptOpOutputs`, with the error
 * channel gone (a rejecting impl produced a successful result carrying a rejected promise). It
 * defaults to `MaybePromise<O>`, so every existing `FunctionDef<I, O, Ctx>` still names this type.
 */
export interface FunctionDef<I = FunctionInputs, O = JsonValue, Ctx = void, R extends MaybePromise<O> = MaybePromise<O>> {
  readonly name: string;
  /** What the function does — surfaced when the def doubles as an agent tool. */
  readonly description?: string;
  readonly input: JsonSchema<I>;
  readonly output: JsonSchema<O>;
  readonly impl: (inputs: I, ctx: Ctx) => R;
}

/**
 * Define a typed function: `impl` parameter types are INFERRED from the `input` schema, and
 * the return type is checked against the `output` schema (§4's `defineFunction` example).
 * Const type parameters make `as const` optional on the schema literals.
 *
 * `R` is inferred from the impl — a sync impl gets `O`, an `async` one `Promise<O>` — which is what
 * lets {@link registerFunctionDef} refuse the second on its sync path.
 */
export function defineFunction<
  const SI extends JSONSchema,
  const SO extends JSONSchema,
  Ctx = void,
  R extends MaybePromise<InferSchema<SO>> = MaybePromise<InferSchema<SO>>,
>(spec: {
  name: string;
  description?: string;
  input: SI;
  output: SO;
  impl: (inputs: InferSchema<SI>, ctx: Ctx) => R;
}): FunctionDef<InferSchema<SI>, InferSchema<SO>, Ctx, R> {
  return {
    name: spec.name,
    description: spec.description,
    // The literal schema IS the runtime document; the cast only re-brands its phantom.
    input: spec.input as unknown as JsonSchema<InferSchema<SI>>,
    output: spec.output as unknown as JsonSchema<InferSchema<SO>>,
    impl: spec.impl,
  };
}

/**
 * Register a def's plain string-keyed impl so dynamic (`functionRef` by name) resolution still works —
 * the typed handle is ADDITIONAL, not required (§4).
 *
 * A def's `impl` returns its value and may THROW; the registry's contract is a resolved
 * {@link FunctionResult}. This is the seam where {@link liftThrowing} — §4.2's documented `catch`
 * fallback — applies: the exception becomes a CLASSIFIED failure, so a 429 raised inside a def is
 * `network-retriable` and the retry machinery can act on it. An impl that wants full control resolves
 * its own result and puts an entry in the map directly.
 *
 * Registration always produces an ENTRY: a pure def (`Ctx = void`) registers as `pure`, a ctx-bearing
 * one as `host` with the capabilities the caller declares. A def reports no metrics, so `M` is the
 * caller's — the entry simply never sets the field.
 *
 * The two overloads make the async/pure mismatch UNSPEAKABLE rather than merely detected: the pure
 * signature demands `R extends O`, which an `async` impl's `Promise<O>` does not satisfy, so
 * `registerFunctionDef(registry, asyncDef)` is a compile error instead of a registration that returns
 * the promise itself as the value. The runtime guard below covers what the compiler cannot see — a def
 * built dynamically, or one arriving through a widening cast.
 */
/** Async: the impl may return a promise, and the `host` entry awaits it. */
export function registerFunctionDef<I, O, Ctx, M>(
  registry: FunctionRegistry<Ctx, M>,
  def: FunctionDef<I, O, Ctx> | FunctionDef<I, O, void>,
  opts: { async: true; capabilities: HostCapabilities; stream?: boolean },
): void;
/** Pure: the impl must be SYNC (`R extends O`), because a `pure` entry resolves its value in place. */
export function registerFunctionDef<I, O, Ctx, M, R extends O>(
  registry: FunctionRegistry<Ctx, M>,
  def: FunctionDef<I, O, void, R>,
  opts?: { async?: false; capabilities?: PureCapabilities },
): void;
export function registerFunctionDef<I, O, Ctx, M>(
  registry: FunctionRegistry<Ctx, M>,
  def: FunctionDef<I, O, Ctx> | FunctionDef<I, O, void>,
  opts?: { async?: false; capabilities?: PureCapabilities } | { async: true; capabilities: HostCapabilities; stream?: boolean },
): void {
  // The widening cast is sound: the runtime schema check at the boundary upholds `I` before
  // the impl runs, and `O` is JSON by the def's output schema.
  const impl = def.impl as unknown as (inputs: FunctionInputs, ctx: Ctx | undefined) => MaybePromise<JsonValue>;
  const lifted = liftThrowing<FunctionInputs, JsonValue, Ctx | undefined>(impl, `function '${def.name}'`);
  if (opts?.async === true) {
    registry.set(def.name, hostFunction<Ctx, M>((inputs, ctx) => lifted(inputs, ctx), opts.capabilities, { stream: opts.stream }));
    return;
  }
  // A pure def must be SYNC to fit the pure variant; `liftThrowing` returns a promise, so the sync
  // path catches directly rather than going through it.
  registry.set(
    def.name,
    pureFunction<M>((inputs) => {
      try {
        const value = impl(inputs, undefined);
        // The runtime half of the guard, for a def the compiler never saw as async (dynamic
        // construction, a widening cast). A thenable is NOT a value: letting it through hands a Promise
        // to `ResolvedValue`, to the memo key, and to `acceptOpOutputs`, and silently discards the
        // impl's error channel — a rejecting impl would resolve as a SUCCESS carrying a rejected
        // promise. Fail loudly, and name the fix.
        if (isThenable(value)) {
          // Observe the thenable's own outcome and drop it. We are refusing this value, and nothing will
          // ever await it — an unhandled rejection from an impl that also throws would take the host
          // process down instead of being reported as the failure below.
          void Promise.resolve(value).catch(() => undefined);
          return {
            error: failureOf(
              new Error(`impl returned a Promise but the function is registered as 'pure' — register it with { async: true } and host capabilities`),
              `function '${def.name}'`,
            ),
          };
        }
        return { value: value as JsonValue };
      } catch (e) {
        return { error: failureOf(e, `function '${def.name}'`) };
      }
    }, opts?.capabilities ?? PURE_CAPABILITIES),
  );
}

/** Duck-typed on `then`, which is exactly what `await` keys on — a hand-rolled thenable is as much of
 *  a problem here as a real `Promise`. */
function isThenable(v: unknown): v is PromiseLike<unknown> {
  return v !== null && (typeof v === "object" || typeof v === "function") && typeof (v as { then?: unknown }).then === "function";
}

// --- Typed operations --------------------------------------------------------

/** An op whose external input record type is `I` and output type is `O` — phantom-branded,
 *  structurally still a plain `Operation<F>` (serialization/hashing see no difference). */
export type TypedOperation<I, O, F extends RefFamily = InlineFamily> = Operation<F> & {
  readonly [OpInputType]?: I;
  readonly [OpOutputType]?: O;
};

/** An op (typed or not) usable as a PRODUCER for a slot of type `O`. A phantom-typed op with
 *  a mismatching output is a compile error; an untyped op is accepted and left to the pre-run
 *  schema checker (§4 tier 2). */
export type Producer<O, F extends RefFamily = InlineFamily> = Operation<F> & {
  readonly [OpOutputType]?: O;
};

/** Extract the phantom output type of a {@link TypedOperation}. */
export type OperationOutput<T> = T extends { readonly [OpOutputType]?: infer O } ? Exclude<O, undefined> : JsonValue;

/** What may be bound to a typed slot of value type `T`: a literal of that type, or a
 *  producer op whose output type matches. */
export type TypedBinding<T> = T | Producer<T>;

// --- Builders ----------------------------------------------------------------

/** A JSON Schema document as builders accept it (a compile-time literal or a plain document). */
type SchemaDoc = { readonly [key: string]: ReadonlyJsonValue | undefined };

/** Read a builder-supplied document as the canonical {@link SchemaDocument} the model's `kindFor`
 *  takes. The two differ only in tolerating `undefined` keyword values, which JSON drops anyway. */
const asDoc = (schema: SchemaDoc | undefined): SchemaDocument | undefined => schema as SchemaDocument | undefined;

function isOperation(v: unknown): v is Operation<InlineFamily> {
  if (!v || typeof v !== "object" || Array.isArray(v)) return false;
  const kind = (v as { kind?: unknown }).kind;
  if (kind === "prompt") return "user" in v && "output" in v;
  if (kind === "function") return "functionRef" in v && "output" in v;
  return false;
}

/** Desugar a binding value: a producer op becomes a producer edge, a literal becomes a
 *  text/blob/json literal ref per the slot's kind. */
function toRef(value: TypedBinding<JsonValue>, kind: RefKind): Ref<InlineFamily> {
  if (isOperation(value)) return { op: value };
  if (kind === "text") return { text: String(value) };
  // A blob-kind literal must keep its `{ blob }` tag: consumers discriminate on the Ref union key
  // (e.g. `producerSchemaOf`), so falling through to `{ json }` would mislabel it as arbitrary JSON
  // and lose the blob schema identity the model treats as load-bearing.
  // The runtime value is a base64 string today (`FromSchema` infers `string` for a blob-kind slot),
  // not `Bytes` — hence the `unknown` hop; the tag is what matters to downstream discrimination.
  if (kind === "blob") return { blob: value as unknown as InlineFamily["blob"] };
  return { json: value as JsonValue };
}

/** A FREE (external) input slot. */
export function free(kind: RefKind = "json", schema?: JsonSchema): Parameter<InlineFamily> {
  return schema === undefined ? { kind } : { kind, schema };
}

/** A BOUND input slot: a literal or a producer op (typed producers are compile-checked at the
 *  builder call sites; this helper is the untyped/dynamic escape hatch). */
export function bound(value: TypedBinding<JsonValue>, kind: RefKind = "json", schema?: JsonSchema): Parameter<InlineFamily> {
  const p: Parameter<InlineFamily> = { kind, binding: toRef(value, kind) };
  if (schema !== undefined) p.schema = schema;
  return p;
}

/**
 * Build a typed `PromptOp<InlineFamily>` (§4). The output type is inferred from the `output`
 * schema literal; `I` may be supplied explicitly for typed call sites (defaults to the
 * dynamic base).
 */
export function promptOp<const SO extends JSONSchema, I = Record<string, JsonValue>>(spec: {
  system?: string;
  user: string;
  /** The LlmConfiguration surface — typed at the seam that executes it (§6). */
  config?: { [key: string]: JsonValue };
  input?: { [name: string]: Parameter<InlineFamily> };
  output: { name?: string; schema: SO };
}): TypedOperation<I, InferSchema<SO>> & PromptOp<InlineFamily> {
  const outSchema = spec.output.schema as SchemaDoc as JsonSchema;
  const op: PromptOp<InlineFamily> = {
    kind: "prompt",
    ...(spec.system !== undefined ? { system: spec.system } : {}),
    user: spec.user,
    config: (spec.config ?? {}) as JsonValue,
    input: spec.input ?? {},
    output: { name: spec.output.name ?? "output", kind: kindFor(asDoc(outSchema)), schema: outSchema },
  };
  return op as TypedOperation<I, InferSchema<SO>> & PromptOp<InlineFamily>;
}

/**
 * Apply a typed function def as a `FunctionOp<InlineFamily>` (§4). Bindings are checked
 * against the def's INFERRED input types: a literal of the wrong type, or a producer whose
 * output type doesn't match the consumed slot, is a COMPILE error. Unbound names stay free
 * (external) inputs.
 */
export function functionOp<I extends Record<string, JsonValue>, O, Ctx>(
  def: FunctionDef<I, O, Ctx>,
  bindings?: { [K in keyof I]?: TypedBinding<I[K]> },
): TypedOperation<I, O> & FunctionOp<InlineFamily> {
  const inputDoc = def.input as SchemaDoc;
  const props = (inputDoc.properties ?? {}) as { [name: string]: SchemaDoc };
  const input: { [name: string]: Parameter<InlineFamily> } = {};
  for (const [name, prop] of Object.entries(props)) {
    const kind = kindFor(asDoc(prop));
    const p: Parameter<InlineFamily> = { kind, schema: prop as JsonSchema };
    const bindingValue = bindings?.[name as keyof I];
    if (bindingValue !== undefined) p.binding = toRef(bindingValue as TypedBinding<JsonValue>, kind);
    input[name] = p;
  }
  // Bindings for names the schema doesn't declare still wire (dynamic/dictionary inputs).
  for (const [name, value] of Object.entries(bindings ?? {})) {
    if (input[name] || value === undefined) continue;
    input[name] = { kind: "json", binding: toRef(value as TypedBinding<JsonValue>, "json") };
  }
  const outSchema = def.output as SchemaDoc as JsonSchema;
  const op: FunctionOp<InlineFamily> = {
    kind: "function",
    functionRef: def.name,
    input,
    output: { name: "output", kind: kindFor(asDoc(outSchema)), schema: outSchema },
  };
  return op as TypedOperation<I, O> & FunctionOp<InlineFamily>;
}

/**
 * Authoring sugar for a RUNTIME INVOCATION (§3.1): emits a PLAIN `FunctionOp` — no extra
 * field, no refinement — whose `functionRef` names the registered adapter, with the authored
 * runtime surface bound as the `config` input and the prompt as an ordinary `text` input.
 */
export function runtimeOp<const SO extends JSONSchema = { readonly type: "string" }>(spec: {
  /** The registered runtime adapter's function ref (e.g. "claude-code"). */
  runtime: string;
  prompt: TypedBinding<string>;
  system?: string;
  /** The authored runtime surface: permission baseline, tool allow-list, permission mode, … */
  config?: { [key: string]: JsonValue };
  /** Additional wired inputs beyond `prompt`/`config`. */
  input?: { [name: string]: Parameter<InlineFamily> };
  output?: { name?: string; schema?: SO };
}): TypedOperation<Record<string, JsonValue>, InferSchema<SO>> & FunctionOp<InlineFamily> {
  const outSchema = (spec.output?.schema ?? { type: "string" }) as SchemaDoc as JsonSchema;
  const input: { [name: string]: Parameter<InlineFamily> } = {
    prompt: { kind: "text", binding: toRef(spec.prompt, "text"), schema: { type: "string" } },
    config: { kind: "json", binding: { json: (spec.config ?? {}) as JsonValue } },
    ...(spec.system !== undefined ? { system: { kind: "text", binding: { text: spec.system }, schema: { type: "string" } } } : {}),
    ...spec.input,
  };
  const op: FunctionOp<InlineFamily> = {
    kind: "function",
    functionRef: spec.runtime,
    input,
    output: { name: spec.output?.name ?? "output", kind: kindFor(asDoc(outSchema)), schema: outSchema },
  };
  return op as TypedOperation<Record<string, JsonValue>, InferSchema<SO>> & FunctionOp<InlineFamily>;
}
