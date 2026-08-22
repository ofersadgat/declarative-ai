/**
 * Reading a function's SIGNATURE out of its TypeScript (SPEC §7.5.2).
 *
 * A module declares no signature in JSON, because there is nowhere for one to disagree: a parameter
 * list already carries almost everything a `ParameterDecl` holds.
 *
 * ```text
 * name         the parameter name
 * index        its POSITION — so positional binding needs no annotation
 * schema       its type, converted to the wire schema (`wireType.ts`)
 * optional     `?`, or a `| undefined` member
 * default      a parameter default — `function f(limit = 3)` declares the slot's default
 * description  a JSDoc `@param` tag
 * ```
 *
 * ## This is the one place that needs a `Program`
 *
 * `moduleExports` reads which symbols a file contributes with a bare parse, because contributed
 * symbols are syntactic. A *type* is not: resolving an imported interface, instantiating a generic,
 * or flattening a conditional all need the checker, and the checker needs module resolution and every
 * transitively imported file. So the expensive machinery runs here, for the one symbol that was
 * actually called, rather than for every symbol on the path.
 *
 * ## The resolver is the same one `require` uses
 *
 * `ts.CompilerHost.resolveModuleNames` is overridden with `resolveSpecifier` — the function the
 * module loader resolves imports with. This is the single most load-bearing constraint in the
 * feature: two resolvers would let a signature be checked against one `helper.ts` and executed
 * against another, and nothing anywhere would report it.
 */
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type * as TS from "typescript";
import { loadCompiler } from "./moduleExports.js";
import { dirOf, resolveSpecifier, type ModuleResolveOptions } from "./moduleLoader.js";
import { typeToWireSlot, WireTypeError } from "./wireType.js";

export class SignatureError extends Error {}

/** One parameter, in the shape a `ParameterDecl` is built from. */
export interface ExtractedParameter {
  name: string;
  /** Position in the parameter list — the `index` a positional call binds by. */
  index: number;
  schema: JsonSchema;
  optional: boolean;
  /** A literal parameter default, where the author wrote one this can read. */
  default?: JsonValue;
  /** The JSDoc `@param` text, where there is one. */
  description?: string;
}

export interface ExtractedSignature {
  parameters: readonly ExtractedParameter[];
  /** The wire schema of what the function returns, with a `Promise` already unwrapped. */
  returns: JsonSchema;
  /** Slots that fell back to the universal schema, and why — never silent. */
  warnings: readonly string[];
}

export interface SignatureOptions extends ModuleResolveOptions {
  /** In-memory sources by pseudo-path, as `prepareModules` takes — how a synthesized body arrives. */
  sources?: Readonly<Record<string, string>>;
  /** Overrides merged over the defaults below. `strict` is deliberately hard to turn off. */
  compilerOptions?: TS.CompilerOptions;
}

/**
 * The compiler configuration a user function is read under.
 *
 * ⚠️ `strict` is load-bearing rather than a preference. SPEC §7.5.2's rule is "`any` → untyped"; an
 * unannotated parameter under `noImplicitAny: false` is IMPLICITLY `any`, so a lax configuration
 * turns the whole feature off with no error anywhere. `isolatedModules` is set because emit is
 * per-file, so a construct that will not survive that is better refused at check time.
 */
export const DEFAULT_COMPILER_OPTIONS: TS.CompilerOptions = {
  strict: true,
  noImplicitAny: true,
  isolatedModules: true,
  esModuleInterop: true,
  skipLibCheck: true,
  // `NodeNext` would demand a `package.json` beside every function, which `~/.jaira/functions` has
  // no reason to hold. Resolution is overridden below in any case; this only sets the dialect.
  moduleResolution: 99 satisfies number as TS.ModuleResolutionKind, // Bundler
  module: 99 satisfies number as TS.ModuleKind, // ESNext
  target: 9 satisfies number as TS.ScriptTarget, // ES2022
};

/**
 * The compiler plus a way to read TypeScript's own lib files — everything extraction needs that
 * cannot be obtained synchronously.
 *
 * Both are loaded ONCE, and afterwards `ts.createProgram` and the whole check are ordinary
 * synchronous calls. That is what lets a sync `loadBundle` resolve a module callee: the same split
 * `loadCompiler` makes for the symbol index, for the same reason.
 */
export interface SignatureContext {
  ts: typeof TS;
  /** Read a file from the real disk — used ONLY for lib files. */
  readLib: (file: string) => string | undefined;
}

let context: Promise<SignatureContext> | undefined;

/** Load the compiler and the lib reader. The one await signature extraction needs. */
export function loadSignatureContext(): Promise<SignatureContext> {
  context ??= (async (): Promise<SignatureContext> => {
    const ts = await loadCompiler();
    const { readFileSync, existsSync } = await import("node:fs");
    return {
      ts,
      readLib: (file) => {
        try {
          return existsSync(file) ? readFileSync(file, "utf8") : undefined;
        } catch {
          return undefined;
        }
      },
    };
  })();
  return context;
}

/**
 * Read the signature of the symbol at `property` inside `file`.
 *
 * `property` is the path `moduleExports` recorded — `["default"]` for a default-export function,
 * `["default", "text", "slug"]` for a symbol inside an object export. Walking it here rather than
 * re-deriving it is what keeps the index and the checker agreeing about which value is meant.
 */
export async function extractSignature(
  file: string,
  property: readonly string[],
  options: SignatureOptions,
): Promise<ExtractedSignature> {
  return extractSignatureWith(await loadSignatureContext(), file, property, options);
}

/** {@link extractSignature} with the context already loaded — the synchronous core. */
export function extractSignatureWith(
  context: SignatureContext,
  file: string,
  property: readonly string[],
  options: SignatureOptions,
): ExtractedSignature {
  const { ts } = context;
  const compilerOptions = { ...DEFAULT_COMPILER_OPTIONS, ...options.compilerOptions };
  const program = createProgram(context, file, compilerOptions, options);
  const checker = program.getTypeChecker();

  const source = program.getSourceFile(file);
  if (source === undefined) throw new SignatureError(`'${file}' could not be read for type checking`);

  const moduleSymbol = checker.getSymbolAtLocation(source);
  if (moduleSymbol === undefined) {
    throw new SignatureError(`'${file}' exports nothing — it is not a module`);
  }

  let type = checker.getTypeOfSymbolAtLocation(moduleSymbol, source);
  for (const step of property) {
    const member = checker.getPropertyOfType(type, step);
    if (member === undefined) {
      throw new SignatureError(`'${file}' has no '${property.join(".")}' — nothing named '${step}' at that point`);
    }
    const at = member.valueDeclaration ?? member.declarations?.[0] ?? source;
    type = checker.getTypeOfSymbolAtLocation(member, at);
  }

  const where = `${file}${property.length > 0 ? `#${property.join(".")}` : ""}`;
  const signatures = type.getCallSignatures();
  const signature = signatures[0];
  if (signature === undefined) {
    // Not an error at this layer: a non-callable export is DATA (SPEC §7.5.2), and it is the caller
    // asking for a signature that has made the category mistake.
    throw new SignatureError(`${where} is not callable — a non-callable export is data, not an operation`);
  }
  if (signatures.length > 1) {
    throw new SignatureError(`${where} is overloaded; a callee must have one signature for arguments to bind against`);
  }

  const warnings: string[] = [];
  const parameters: ExtractedParameter[] = [];
  signature.getParameters().forEach((symbol, index) => {
    const declaration = symbol.valueDeclaration;
    if (declaration !== undefined && ts.isParameter(declaration) && declaration.dotDotDotToken !== undefined) {
      throw new SignatureError(`${where}: a rest parameter has no fixed slot to bind an argument to`);
    }
    if (declaration !== undefined && ts.isParameter(declaration) && !ts.isIdentifier(declaration.name)) {
      throw new SignatureError(`${where}: a destructured parameter names no single slot`);
    }
    const name = symbol.getName();
    const at = declaration ?? source;
    const slotWhere = `${where}: parameter '${name}'`;

    // A CANCELLATION SIGNAL is the one carve-out (SPEC §7.5.6): it carries no workflow data, so it is
    // absent from the wire signature entirely rather than being a slot nobody can fill.
    const declaredType = checker.getTypeOfSymbolAtLocation(symbol, at);
    if (isAbortSignal(checker, declaredType)) return;

    let slot;
    try {
      slot = typeToWireSlot(ts, checker, declaredType, slotWhere);
    } catch (e) {
      if (e instanceof WireTypeError) throw new SignatureError(e.message);
      throw e;
    }
    warnings.push(...slot.warnings);

    const optional =
      slot.optional ||
      (declaration !== undefined && ts.isParameter(declaration) && (declaration.questionToken !== undefined || declaration.initializer !== undefined));

    const parameter: ExtractedParameter = { name, index, schema: slot.schema, optional };
    const literal = declaration !== undefined && ts.isParameter(declaration) ? literalValue(ts, declaration.initializer) : undefined;
    if (literal !== undefined) parameter.default = literal;
    const documentation = ts.displayPartsToString(symbol.getDocumentationComment(checker)).trim();
    if (documentation.length > 0) parameter.description = documentation;
    parameters.push(parameter);
  });

  const returned = awaited(ts, checker, signature.getReturnType());
  let returns;
  try {
    returns = typeToWireSlot(ts, checker, returned, `${where}: return`);
  } catch (e) {
    if (e instanceof WireTypeError) throw new SignatureError(e.message);
    throw e;
  }
  warnings.push(...returns.warnings);

  return { parameters, returns: returns.schema, warnings };
}

/** `Promise<T>` → `T`. A function may be async; what a caller sees is what it resolves with. */
function awaited(ts: typeof TS, checker: TS.TypeChecker, type: TS.Type): TS.Type {
  if (type.getSymbol()?.getName() !== "Promise") return type;
  const argument = checker.getTypeArguments(type as TS.TypeReference)[0];
  return argument ?? type;
}

/** The trailing `AbortSignal` of SPEC §7.5.6 — recognised by name, and dropped from the signature. */
function isAbortSignal(checker: TS.TypeChecker, type: TS.Type): boolean {
  const name = type.getSymbol()?.getName();
  if (name === "AbortSignal") return true;
  // `AbortSignal | undefined`, which is how an optional one is written.
  return type.isUnion() && type.types.some((t) => t.getSymbol()?.getName() === "AbortSignal");
}

/**
 * A parameter default, where it is a LITERAL this can read.
 *
 * Deliberately narrow. A default that is a call, an identifier, or anything else is computed at run
 * time by the function itself and is none of the wiring's business — reporting it as the slot's
 * `default` would be claiming a value nobody can check. `optional` is still set either way, so the
 * slot behaves correctly; only the declared value is skipped.
 */
function literalValue(ts: typeof TS, node: TS.Expression | undefined): JsonValue | undefined {
  if (node === undefined) return undefined;
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
  if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
  if (node.kind === ts.SyntaxKind.NullKeyword) return null;
  if (ts.isPrefixUnaryExpression(node) && node.operator === ts.SyntaxKind.MinusToken && ts.isNumericLiteral(node.operand)) {
    return -Number(node.operand.text);
  }
  if (ts.isArrayLiteralExpression(node)) {
    const out: JsonValue[] = [];
    for (const element of node.elements) {
      const value = literalValue(ts, element);
      if (value === undefined) return undefined; // one unreadable entry makes the whole default unreadable
      out.push(value);
    }
    return out;
  }
  if (ts.isObjectLiteralExpression(node)) {
    const out: Record<string, JsonValue> = {};
    for (const member of node.properties) {
      if (!ts.isPropertyAssignment(member)) return undefined;
      const key = ts.isIdentifier(member.name) || ts.isStringLiteral(member.name) ? member.name.text : undefined;
      if (key === undefined) return undefined;
      const value = literalValue(ts, member.initializer);
      if (value === undefined) return undefined;
      out[key] = value;
    }
    return out;
  }
  return undefined;
}

/**
 * A `Program` over the workflow's own filesystem, resolving modules the way `require` will.
 *
 * The lib files are the one thing read from the REAL filesystem: `Date`, `Array` and the rest live
 * inside the `typescript` package, and a checker without them cannot type `Date` at all — which is
 * the type the marshalling table exists for.
 */
function createProgram(
  context: SignatureContext,
  entry: string,
  compilerOptions: TS.CompilerOptions,
  options: SignatureOptions,
): TS.Program {
  const { ts, readLib } = context;
  const libDir = ts.getDefaultLibFilePath(compilerOptions).replace(/\\/g, "/").replace(/\/[^/]+$/, "");

  const read = (fileName: string): string | undefined => {
    const own = options.sources?.[fileName] ?? options.vfs.read(fileName);
    if (own !== undefined) return own;
    // Only lib files fall through to the real disk. A workflow's own files come from the `Vfs`, so a
    // test's in-memory tree stays in memory and a host's sandboxed view stays sandboxed.
    if (!fileName.replace(/\\/g, "/").startsWith(libDir)) return undefined;
    return readLib(fileName);
  };

  const sourceFiles = new Map<string, TS.SourceFile>();
  const host: TS.CompilerHost = {
    getSourceFile: (fileName, languageVersion) => {
      const cached = sourceFiles.get(fileName);
      if (cached !== undefined) return cached;
      const text = read(fileName);
      if (text === undefined) return undefined;
      const created = ts.createSourceFile(fileName, text, languageVersion, true);
      sourceFiles.set(fileName, created);
      return created;
    },
    getDefaultLibFileName: (o) => ts.getDefaultLibFilePath(o).replace(/\\/g, "/"),
    writeFile: () => {
      /* type checking only — nothing is emitted through this host */
    },
    getCurrentDirectory: () => "/",
    getCanonicalFileName: (f) => f.replace(/\\/g, "/"),
    useCaseSensitiveFileNames: () => true,
    getNewLine: () => "\n",
    fileExists: (fileName) => read(fileName) !== undefined,
    readFile: read,
    // THE shared resolver. See the module note: two resolvers is the bug that type-checks.
    resolveModuleNames: (moduleNames, containingFile) =>
      moduleNames.map((name) => {
        const resolved = resolveSpecifier(name, dirOf(containingFile), options);
        return resolved === undefined
          ? undefined
          : { resolvedFileName: resolved, extension: extensionOf(ts, resolved), isExternalLibraryImport: resolved.includes("/node_modules/") };
      }),
  };

  return ts.createProgram([entry], compilerOptions, host);
}

function extensionOf(ts: typeof TS, file: string): TS.Extension {
  const lower = file.toLowerCase();
  if (lower.endsWith(".d.ts")) return ts.Extension.Dts;
  if (lower.endsWith(".ts")) return ts.Extension.Ts;
  if (lower.endsWith(".tsx")) return ts.Extension.Tsx;
  if (lower.endsWith(".jsx")) return ts.Extension.Jsx;
  return ts.Extension.Js;
}
