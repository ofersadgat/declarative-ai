/**
 * Turning a user's js/ts into something the engine can call (SPEC §7.5).
 *
 * Every other module in this feature builds one piece: `moduleIndex` finds a symbol, `signature`
 * types it, `moduleLoader` runs it, `marshal` converts at the boundary. This is where they meet the
 * rest of the system — and the meeting point is deliberately the one that already exists.
 *
 * ## A user function becomes a REGISTRY ENTRY
 *
 * The engine dispatches a `FunctionOp` by looking `functionRef` up in the `CapabilityRegistry`. So a
 * user function is registered like any other, under a generated ref, and the operation the loader
 * produces names that ref. Nothing downstream — the checker, the memo key, permission gating, the
 * run record — learns that this entry came from a `.ts` file rather than from the host, which is
 * exactly the property SPEC §7.5 opens with: a callee is an operation, and what differs is only
 * where the name resolves and where the body comes from.
 *
 * That also means the capability record is REQUIRED and total, as §3.3 demands. A user function is
 * `pure: false` on the memoizable axis by SPEC §7.5.6 — not memoized — and that is declared here
 * rather than left for something downstream to infer from an `undefined`.
 *
 * ## Everything here is synchronous
 *
 * `loadBundle` is sync, so resolving a callee must be. The compiler is loaded once by
 * {@link createUserFunctions}, and every call after that — parse, check, transpile, execute — is an
 * ordinary function call. See `loadCompiler` for why that split is available at all.
 */
import type { InlineFamily, JsonValue, Operation, Parameter } from "@declarative-ai/exec";
import type { JsonSchema } from "@declarative-ai/json";
import { synthesizeBodyWith, type SynthesizedBody } from "./functionBody.js";
import type { ParameterDecl } from "./format.js";
import { loadCompiler } from "./moduleExports.js";
import { prepareModules, type LoadedModules, type PrepareOptions } from "./moduleLoader.js";
import { marshalIn, marshalOut } from "./marshal.js";
import { extractSignatureWith, loadSignatureContext, type ExtractedSignature, type SignatureContext } from "./signature.js";
import { selectProperty, type Vfs } from "./reference.js";

export class UserFunctionError extends Error {}

/** How a user function is named in the registry. Distinctive so a collision is obvious, not silent. */
export function userFunctionRef(file: string, property: readonly string[]): string {
  return `user:${file}${property.length > 0 ? `#${property.join(".")}` : ""}`;
}

/** What one resolved user function is, once its signature has been read. */
export interface ResolvedUserFunction {
  /** The registry name the operation's `functionRef` carries. */
  ref: string;
  /** The operation a call site binds its arguments against. */
  operation: Operation<InlineFamily>;
  signature: ExtractedSignature;
  /** Slots that fell back to the universal schema — surfaced, never swallowed. */
  warnings: readonly string[];
}

export interface UserFunctionOptions extends Omit<PrepareOptions, "sources"> {
  vfs: Vfs;
  /** Sources with no file behind them — how a synthesized embedded body enters. */
  sources?: Record<string, string>;
  onWarn?: (message: string) => void;
}

/**
 * The synchronous facade the loader uses.
 *
 * One `await` at construction (the compiler and the lib reader); everything after is a call.
 */
export interface UserFunctions {
  /** The operation a MODULE symbol denotes — `moduleIndex` found the file and property. */
  operationFor(file: string, property: readonly string[]): ResolvedUserFunction;
  /** The operation an EMBEDDED body denotes (SPEC §7.5.1, form 2). */
  operationForBody(name: string, body: string, input: Readonly<Record<string, ParameterDecl>>): ResolvedUserFunction;
  /** Every impl registered so far, by ref — what a host merges into its `CapabilityRegistry`. */
  readonly impls: ReadonlyMap<string, UserFunctionImpl>;
  /**
   * Compile every function resolved so far, so they can actually run.
   *
   * Separate from resolution on purpose, and the boundary is the one §7.5.5 draws: resolving and
   * type-checking READ files, running them is a decision somebody has to have made. A host calls
   * this after its approval gate, never before.
   */
  prepare(): Promise<LoadedModules>;
}

/** One callable, with the marshalling boundary already wrapped around it. */
export type UserFunctionImpl = (args: Readonly<Record<string, unknown>>) => Promise<JsonValue>;

export async function createUserFunctions(options: UserFunctionOptions): Promise<UserFunctions> {
  const context = await loadSignatureContext();
  const ts = await loadCompiler();
  return new Facade(context, ts, options);
}

class Facade implements UserFunctions {
  readonly impls = new Map<string, UserFunctionImpl>();
  /** Resolved functions by ref, so one symbol is type-checked once however many states call it. */
  private readonly resolved = new Map<string, ResolvedUserFunction>();
  /** Synthesized bodies by their pseudo-path, so a body compiles once. */
  private readonly synthetic: Record<string, string> = {};
  private prepared: LoadedModules | undefined;

  constructor(
    private readonly context: SignatureContext,
    private readonly ts: Awaited<ReturnType<typeof loadCompiler>>,
    private readonly options: UserFunctionOptions,
  ) {}

  operationFor(file: string, property: readonly string[]): ResolvedUserFunction {
    return this.resolve(file, property, userFunctionRef(file, property));
  }

  operationForBody(name: string, body: string, input: Readonly<Record<string, ParameterDecl>>): ResolvedUserFunction {
    // An embedded body becomes a module with a pseudo-path, so from here down there is ONE pipeline.
    // The path is derived from the body's own text, which makes two identical bodies one compilation
    // and keeps the name stable across loads (a counter would not be).
    const key = `<body>/${name}.${fingerprint(body)}.ts`;
    let synthesized: SynthesizedBody | undefined;
    if (this.synthetic[key] === undefined) {
      synthesized = synthesizeBodyWith(this.ts, name, body, input, name);
      this.synthetic[key] = synthesized.source;
      // A new source invalidates whatever was prepared, since the closure has grown.
      this.prepared = undefined;
    }
    return this.resolve(key, ["default"], userFunctionRef(key, ["default"]));
  }

  private resolve(file: string, property: readonly string[], ref: string): ResolvedUserFunction {
    const cached = this.resolved.get(ref);
    if (cached !== undefined) return cached;

    const signature = extractSignatureWith(this.context, file, property, {
      vfs: this.options.vfs,
      requirePath: this.options.requirePath,
      ...(this.options.roots !== undefined ? { roots: this.options.roots } : {}),
      sources: { ...this.options.sources, ...this.synthetic },
    });
    for (const warning of signature.warnings) this.options.onWarn?.(warning);

    const operation = operationOf(ref, signature);
    this.impls.set(ref, this.implFor(file, property, signature));

    const result: ResolvedUserFunction = { ref, operation, signature, warnings: signature.warnings };
    this.resolved.set(ref, result);
    return result;
  }

  /**
   * The impl the registry dispatches to: marshal in, call, marshal out, and let a throw travel as a
   * rejection the caller classifies.
   *
   * Preparation is DEFERRED to first call rather than done at resolve time, because resolving is what
   * happens while a workflow is being loaded and loading must not execute anything (SPEC §7.5.4).
   */
  private implFor(file: string, property: readonly string[], signature: ExtractedSignature): UserFunctionImpl {
    return async (args) => {
      const modules = this.ensurePrepared(file);
      const namespace = modules.execute(file);
      const target = selectProperty(namespace, property, userFunctionRef(file, property));
      if (typeof target !== "function") {
        throw new UserFunctionError(`${userFunctionRef(file, property)} is not callable at run time`);
      }

      const ordered = signature.parameters.map((parameter) => {
        const supplied = args[parameter.name];
        const value = supplied === undefined ? parameter.default : (supplied as JsonValue);
        if (value === undefined) {
          if (parameter.optional) return undefined;
          throw new UserFunctionError(`${userFunctionRef(file, property)}: required input '${parameter.name}' was not supplied`);
        }
        return marshalIn(value as JsonValue, parameter.schema);
      });

      // `await` unconditionally: a function may be async (SPEC §7.5.6) and a synchronous one resolves
      // immediately, so one path serves both rather than a branch on the return value's shape.
      const returned: unknown = await (target as (...a: unknown[]) => unknown)(...ordered);
      return marshalOut(returned, signature.returns);
    };
  }

  /** Transpile the closure once, lazily, and reuse it. */
  private ensurePrepared(file: string): LoadedModules {
    if (this.prepared !== undefined && this.prepared.emitted.has(file)) return this.prepared;
    throw new UserFunctionError(
      `'${file}' has not been prepared — call prepare() before invoking a user function`,
    );
  }

  /**
   * Compile every function resolved so far.
   *
   * Separate from resolution on purpose, and the boundary is the same one §7.5.5 draws: resolving and
   * type-checking read files, running them is a decision somebody has to have made. A host calls this
   * after its approval gate.
   */
  async prepare(): Promise<LoadedModules> {
    const entries = [...this.resolved.values()].map((r) => r.ref.slice("user:".length).split("#")[0]!);
    const prepared = await prepareModules([...new Set(entries)], {
      ...this.options,
      sources: { ...this.options.sources, ...this.synthetic },
    });
    this.prepared = prepared;
    return prepared;
  }
}

/** A short, stable name for a body's text — enough to keep two different bodies apart. */
function fingerprint(body: string): string {
  let hash = 2166136261;
  for (let i = 0; i < body.length; i++) {
    hash ^= body.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

/**
 * The `Operation` a call site binds against.
 *
 * This is the whole point of extraction: `index` comes from the parameter's POSITION, so
 * `bindPositionally` binds `confidence(rank, iteration)` correctly with nothing declared anywhere,
 * and `schema` is the wire type every §6.2 check runs against.
 */
export function operationOf(ref: string, signature: ExtractedSignature): Operation<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const parameter of signature.parameters) {
    const slot: Parameter<InlineFamily> = { kind: kindOf(parameter.schema), index: parameter.index };
    if (Object.keys(parameter.schema).length > 0) slot.schema = parameter.schema;
    if (parameter.default !== undefined) slot.binding = { json: parameter.default };
    input[parameter.name] = slot;
  }
  const output: Operation<InlineFamily>["output"] = { name: "result", kind: kindOf(signature.returns) };
  if (Object.keys(signature.returns).length > 0) output.schema = signature.returns;
  return { kind: "function", functionRef: ref, input, output };
}

/** A slot's `kind`, from its wire type. Text where the schema says string; json otherwise. */
function kindOf(schema: JsonSchema): "text" | "json" {
  return (schema as { type?: unknown }).type === "string" && (schema as { format?: unknown }).format === undefined
    ? "text"
    : "json";
}
