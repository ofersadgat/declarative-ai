/**
 * Compiling a workflow script to hw states (SCRIPTS.md).
 *
 * A script is a state written as code. This module reads it and produces the STATE DOCUMENTS a person
 * could have written by hand — plain `StateDef`s `loadBundle` loads like any other — so nothing
 * downstream (the loader, the validator, the engine, the journal, the board) learns a new format. The
 * one thing the documents carry that authored JSON rarely does is `operation.script`: the code a state
 * runs, which the engine executes as a recorded operation (`$script`, `engine.ts`).
 *
 * ## The shape of the compile
 *
 *  1. **`meta`** is read as a pure literal: the state's standard properties (SCRIPTS.md §4).
 *  2. **The signature** comes from the exported function, or is `args` in / the exports out when the
 *     body is the function (§5).
 *  3. **In `"calls"` mode every non-deterministic call becomes a statement of its own** (§7): a call
 *     buried in an expression is lifted out with everything evaluated before it held in temporaries,
 *     and `&&`, `||`, `??` and `?:` around one become branches — so evaluation order is kept exactly.
 *     A helper that makes a call is inlined where it is called. `parallel`, `pipeline` and
 *     `Promise.all` over a callback that makes one are a FAN-OUT, compiled on their own.
 *  4. **The body is cut into basic blocks** at every `phase()` and, in `"calls"` mode, every call. A
 *     forward pass works out which phase can be current at each block.
 *  5. **Each cut starts a state.** Everything is a MACHINE — the script's root, each phase, each
 *     fan-out's element — a composite state whose own operation is the code it starts with and whose
 *     children are one state per call it makes: `[the call, the code after it]`, or a mount for a
 *     called state or a fan-out. Control flow becomes the rules on those children.
 *  6. **Variables that live across a cut** move into one record, `$v`: each state takes the variables
 *     it needs as inputs and hands on, as outputs, the ones the states after it need — ordinary
 *     backward liveness over the blocks.
 *
 * ## What a cut costs
 *
 * JavaScript's block scoping does not survive being cut into `switch` cases, so a variable declared
 * DIRECTLY in a structure that contains a cut — the function body, a loop body that makes a call — is
 * hoisted into `$v`. Everything inside a statement that contains no cut is left as written, scoping
 * and all. A variable declared in a hoisted position and captured by a closure that outlives one loop
 * iteration sees the variable's latest value rather than the iteration's; that is the one semantic
 * difference hoisting makes.
 */
import { sha256Hex } from "@declarative-ai/exec";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type * as TS from "typescript";
import {
  CONFIG_FIELD_SCHEMAS,
  SCRIPT_CONTROL,
  SCRIPT_RETURN,
  type ChildDecl,
  type EnvironmentDecl,
  type NamedParameterDecl,
  type ParameterDecl,
  type StateDef,
  type TitleDecl,
  type TransitionDecl,
} from "./format.js";
import { dirOf, resolveSpecifier, type ModuleResolveOptions } from "./moduleLoader.js";
import { DATA_EXTENSIONS, parseReferencedFile } from "./reference.js";
import { HOOK_GLOBALS_PATH, HOOK_MODULE, HOOK_MODULE_PATH } from "./scriptHooks.js";
import { awaited, createScriptProgram, loadSignatureContext, readSignature, type SignatureContext } from "./signature.js";
import { typeToWireSchema, typeToWireSlot, WireTypeError } from "./wireType.js";

/**
 * How a script becomes something hw runs (SCRIPTS.md §3), coarsest to finest: a callable, ONE state,
 * a state per phase, a state per call.
 */
export type ScriptMode = "function" | "state" | "phases" | "calls";
export const SCRIPT_MODES: readonly ScriptMode[] = ["function", "state", "phases", "calls"];

/** A compile failure, located at the line of the script that caused it. */
export class ScriptCompileError extends Error {
  constructor(
    message: string,
    readonly file: string,
    readonly line?: number,
    readonly column?: number,
  ) {
    super(line === undefined ? `${file}: ${message}` : `${file}:${line}:${column ?? 1}: ${message}`);
  }
}

/** A script's `meta` — the standard properties of the state it is (SCRIPTS.md §4). */
export interface ScriptMeta {
  name?: string;
  label?: string;
  description?: string;
  whenToUse?: string;
  title?: TitleDecl;
  environment?: EnvironmentDecl;
  limits?: Record<string, JsonValue>;
  compile?: ScriptMode;
  /** Claude's display list — descriptions of the compiled phases, declaring nothing (§6.5). */
  phases?: Array<{ title: string; detail?: string; model?: string }>;
}

const META_KEYS: ReadonlySet<string> = new Set(["name", "label", "description", "whenToUse", "title", "environment", "limits", "compile", "phases"]);

/** Keys a state file has that a script's `meta` may not — each with where the fact comes from instead. */
const REFUSED_META: Readonly<Record<string, string>> = {
  inputs: "a script's inputs are its exported function's parameters, or `args`",
  outputs: "a script's outputs are what its function returns, or what it exports",
  children: "a script's children are its phases and the calls it makes",
  sequence: "a script's order is its code",
  transitions: "a script's transitions are its control flow",
  operation: "a script's operation is its body",
  id: "a script's id is its path",
};

/** Where the generated documents came from — every file the compile read, by content hash (SCRIPTS.md §12). */
export interface GeneratedProvenance {
  from: string;
  inputs: Record<string, string>;
}

/** A line of the script a generated document answers to — the source map (SCRIPTS.md §12). */
export interface SourceAt {
  line: number;
  column: number;
}

export interface CompileScriptOptions extends ModuleResolveOptions {
  /** The script's path — where its imports resolve, and what the documents record as their source. */
  file: string;
  /** The id of the state the script IS. Everything it compiles to hangs below `<stateId>/`. */
  stateId: string;
  /** The source, when the caller already read it. Else read through `vfs`. */
  source?: string;
  /** The mode for a script whose `meta` names none — the location's default (SCRIPTS.md §3). */
  defaultMode?: ScriptMode;
  compilerOptions?: TS.CompilerOptions;
  /**
   * The state id a file is, when it is one — how an imported state and a `workflow("…")` target are
   * named (SCRIPTS.md §10). Default: the file's path under the workflow root the script itself sits
   * in, which `file` and `stateId` together say.
   */
  stateIdOf?: (file: string) => string | undefined;
}

export interface CompiledScript {
  mode: ScriptMode;
  meta: ScriptMeta;
  /** The state documents, by state id. Empty for a `"function"`-mode module, which stays a module. */
  documents: Record<string, StateDef>;
  warnings: string[];
  generated: GeneratedProvenance;
  /** Where in the script each generated state comes from, by state id. */
  sourceMap: Record<string, SourceAt>;
}

/**
 * Whether a module's source is written as a workflow script — its first statement other than an
 * `import` or a type is `export const meta = …` (SCRIPTS.md §3). The cheap syntactic test a loader probes with;
 * a module under a workflows root is a script whether or not it passes (its mode may still say
 * `"function"`).
 */
export function hasScriptMeta(ts: typeof TS, file: string, source: string): boolean {
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, scriptKindOf(ts, file));
  const first = sf.statements.find((s) => !ts.isImportDeclaration(s) && !isTypeOnly(ts, s));
  return first !== undefined && metaDeclaration(ts, first) !== undefined;
}

/** Compile one script. The one await is loading the compiler; see {@link compileScriptWith}. */
export async function compileScript(options: CompileScriptOptions): Promise<CompiledScript> {
  return compileScriptWith(await loadSignatureContext(), options);
}

/** {@link compileScript} with the compiler already loaded — synchronous, as a loader needs. */
export function compileScriptWith(context: SignatureContext, options: CompileScriptOptions): CompiledScript {
  return new Compiler(context, options).compile();
}

// --- internals --------------------------------------------------------------------------------------

/** What an import binds (SCRIPTS.md §10). */
type ImportKind = "code" | "state" | "prompt" | "registry" | "data" | "operation";

interface ImportedName {
  kind: ImportKind;
  declaration: TS.ImportDeclaration;
  /** A called state's id. */
  stateId?: string;
  /** A called state's inputs by name, when they are known — and whether its only one is `args`. */
  inputNames?: string[];
  argsInput?: boolean;
  /** Each input's declared type, where the called state declares one. */
  inputSchemas?: Record<string, JsonSchema>;
  /** The output a called script's non-record return lands in. */
  whole?: string;
  /** A registry entry's name. */
  ref?: string;
  /** A prompt import's template. */
  template?: string;
  /** A data import's value. */
  data?: JsonValue;
}

/** What kind of state a call becomes (SCRIPTS.md §7). */
type CallKind = "prompt" | "script" | "mount" | "map";

/** One non-deterministic call site — a cut, in `"calls"` mode. */
interface CallSite {
  id: number;
  kind: CallKind;
  node: TS.Node;
  label: string;
  /** The hoisted temporary the result lands in. */
  target: string;
  /** The `_call` value: the call's arguments, evaluated at the end of the state that makes it. */
  call: TS.Expression;
  resultSchema?: JsonSchema;
  /** `prompt`: the prompt operation's fields, written as a state would write them. */
  op?: Record<string, unknown>;
  /** `prompt`: the result is `{ value }`, a non-object `T` wrapped for the wire. */
  unwrap?: boolean;
  /** `script`: the call, as code over `$v._call`. */
  script?: string;
  /**
   * Values the call reads as TYPED variables of their own, by what reads them: a called state's inputs
   * by name, a computed option of a prompt operation by field. Typed where they are made, so the wire
   * into a typed slot checks.
   */
  inputVars?: Record<string, string>;
  /** `mount`: the called state. */
  stateId?: string;
  inputNames?: string[];
  argsInput?: boolean;
  whole?: string;
  /** `map`: the element machine, and whether a failed element reads as `null` (Claude's `parallel`). */
  element?: number;
  failureValue?: boolean;
  /** The `try` the call sits in, when it does. */
  handler?: Handler;
}

/** A lowered `try`: its catch block, and the variable its `catch (e)` binds. */
interface Handler {
  id: number;
  block: number;
  param?: string;
}

/** A fan-out's element — the machine one item runs. */
interface ElementSpec {
  id: number;
  node: TS.Node;
  /** Its parameters, by where each is fed from. */
  params: Array<{ name: string; from: "item" | "index" }>;
  statements: TS.Statement[];
  entry?: number;
  site?: number;
}

type Terminator =
  | { kind: "goto"; to: number }
  | { kind: "branch"; cond: TS.Expression; then: number; else: number }
  | { kind: "phase"; phase: string; to: number }
  | { kind: "call"; site: number; to: number }
  | { kind: "return"; value: TS.Expression | undefined; fallthrough: boolean };

interface Block {
  id: number;
  /** Which machine's code this is: 0 the script, `k` element `k`. */
  machine: number;
  /** Rewritten statements, in order, each with the script line it came from. */
  code: Array<{ statement: TS.Statement; line: number }>;
  term?: Terminator;
  /** Jumps a rewritten `break`/`continue` inside `code` makes — edges the terminator does not show. */
  jumps: number[];
  /** The `try` this block's code runs in, when a lowered one. */
  handler?: Handler;
  /** Hoisted variables this block reads before it writes them — its upward-exposed uses. */
  uses: Set<string>;
  /** Hoisted variables this block assigns outright. */
  defs: Set<string>;
  line: number;
}

interface LoweredLoop {
  breakTo: number;
  /** `-1` for a `switch` or a labeled block: `continue` passes through it. */
  continueTo: number;
  labels: readonly string[];
}

interface Hoisted {
  name: string;
  schema?: JsonSchema;
  /** Why the variable has no wire form, when it has none — an error only if it crosses a cut. */
  unrepresentable?: string;
}

/** Names that are the continuation's, never a variable's. */
const CONTROL_NAMES: ReadonlySet<string> = new Set([...Object.values(SCRIPT_CONTROL), "_call", "_result", "_error", "_items"]);

/** A fan-out's items as a state hands them on — what the element mount's `each` wire reads, and it reads a list. */
const ITEMS_OUTPUT = "_items";
const ITEMS_SCHEMA = { type: "array", items: {} } as unknown as JsonSchema;

/** The `_call`/`_result`/`_error` a state receives besides its variables. */
const CALL_INPUT = "_call";
const RESULT_INPUT = "_result";
const ERROR_INPUT = "_error";

class Compiler {
  readonly ts: typeof TS;
  private readonly file: string;
  private readonly source: string;
  sf!: TS.SourceFile;
  checker!: TS.TypeChecker;
  private program!: TS.Program;
  private readonly warnings: string[] = [];
  mode: ScriptMode = "calls";

  /** Hoisted variables by symbol, and by the unique name each lives under in `$v`. */
  readonly hoisted = new Map<TS.Symbol, Hoisted>();
  readonly hoistedByName = new Map<string, Hoisted>();
  private synthetic = 0;
  /** Helpers — function declarations and static constants — re-declared at the top of every segment. */
  private readonly helpers: TS.Statement[] = [];
  private readonly helperDeclarations = new Set<TS.VariableDeclaration>();
  private readonly helperRefs = new Set<string>();

  readonly blocks: Block[] = [];
  private cur!: Block;
  private machine = 0;
  private loops: LoweredLoop[] = [];
  private handlers: Handler[] = [];
  private handlerCount = 0;
  private pendingLabels: string[] = [];

  /** Body mode: what the fallthrough returns — named exports, or the default export's variable. */
  private readonly exportNames: string[] = [];
  private wholeReturn = false;
  private defaultExport: string | undefined;
  private sawReturn = false;
  private bodyMode = false;
  /** The workflow root the script's own id hangs off — where a called state's id is read from. */
  private readonly root: string;

  /** What each imported local name is (SCRIPTS.md §10). */
  private readonly imported = new Map<string, ImportedName>();
  /** The import declarations a segment re-emits: code, and code imported `as: "operation"`. */
  private readonly codeImports: TS.ImportDeclaration[] = [];

  /** Every call site, by id; and the statement that marks each in the rewritten body. */
  readonly sites: CallSite[] = [];
  private readonly markers = new Map<TS.Statement, CallSite>();
  readonly elements: ElementSpec[] = [];
  /** Helpers being inlined, so a recursive one is refused rather than expanded forever. */
  private readonly inlining: TS.Node[] = [];
  /** The type arguments of the helper being inlined, by type parameter — `llm<T>` inside it. */
  private readonly substitutions: Array<Map<TS.Symbol, TS.Type>> = [];
  /** Memo: whether a helper (transitively) makes a call. */
  private readonly helperCuts = new Map<TS.Node, boolean>();
  /**
   * The states a `"phases"`/`"state"` script calls from its code — imported, or named in `workflow()` —
   * by the key each is mounted under on the root, so the bundle holds them (SCRIPTS.md §10).
   */
  readonly coarseCalls = new Map<string, string>();
  /** Generic helpers whose `llm<T>` needs the caller's `T` (phases/state modes), by declaration. */
  private readonly schemaParams = new Map<TS.Node, string[]>();

  constructor(
    private readonly context: SignatureContext,
    private readonly options: CompileScriptOptions,
  ) {
    this.ts = context.ts;
    this.file = options.file;
    const source = options.source ?? options.vfs.read(options.file);
    if (source === undefined) throw new ScriptCompileError("could not be read", options.file);
    this.source = source;
    const posix = options.file.replace(/\\/g, "/");
    const stem = posix.replace(/\.[^./]+$/, "");
    this.root = stem.endsWith(`/${options.stateId}`) ? stem.slice(0, stem.length - options.stateId.length - 1) : dirOf(posix);
  }

  compile(): CompiledScript {
    const { ts } = this;
    this.program = createScriptProgram(this.context, this.file, {
      vfs: this.options.vfs,
      requirePath: this.options.requirePath,
      ...(this.options.roots !== undefined ? { roots: this.options.roots } : {}),
      sources: { [this.file]: this.source },
      // A Claude script is plain JavaScript; its types are inferred, never checked (SCRIPTS.md §13).
      compilerOptions: { allowJs: true, checkJs: false, ...this.options.compilerOptions },
    });
    this.checker = this.program.getTypeChecker();
    const sf = this.program.getSourceFile(this.file);
    if (sf === undefined) throw new ScriptCompileError("could not be read for compiling", this.file);
    this.sf = sf;

    const statements = [...sf.statements];
    const rest = statements.filter((s) => !ts.isImportDeclaration(s));

    // `meta`, when there is one, comes first — Claude's rule, and what makes a script recognizable
    // without running anything. Types have no run-time effect, so they may sit above it too.
    let meta: ScriptMeta = {};
    const metaIndex = rest.findIndex((s) => metaDeclaration(ts, s) !== undefined);
    if (metaIndex >= 0 && rest.slice(0, metaIndex).some((s) => !isTypeOnly(ts, s))) {
      throw this.error(rest[metaIndex]!, "`export const meta` must be the first statement after the imports (and any types)");
    }
    if (metaIndex >= 0) meta = this.readMeta(metaDeclaration(ts, rest[metaIndex]!)!);
    const body = metaIndex >= 0 ? rest.filter((_, i) => i !== metaIndex) : rest;

    this.mode = meta.compile ?? this.options.defaultMode ?? "calls";
    const generated = this.provenance();
    if (this.mode === "function") return { mode: this.mode, meta, documents: {}, warnings: this.warnings, generated, sourceMap: {} };

    for (const s of statements) if (ts.isImportDeclaration(s)) this.classifyImport(s);

    const entry = this.entryFunction(body);
    this.bodyMode = entry === undefined;
    const prelude: TS.Statement[] = [];
    let bodyStatements: TS.Statement[];
    let inputs: Record<string, ParameterDecl>;
    let returns: JsonSchema | undefined;
    if (entry !== undefined) {
      for (const s of body) if (s !== entry.statement) prelude.push(s);
      const signature = this.checker.getSignatureFromDeclaration(entry.fn);
      if (signature === undefined) throw this.error(entry.fn, "has no signature the checker can read");
      const read = readSignature(ts, this.checker, sf, signature, this.file);
      this.warnings.push(...read.warnings.filter((w) => !w.includes(": return")));
      inputs = {};
      for (const p of read.parameters) {
        inputs[p.name] = {
          ...(Object.keys(p.schema).length > 0 ? { schema: p.schema } : {}),
          ...(p.optional ? { optional: true } : {}),
          ...(p.default !== undefined ? { default: p.default } : {}),
          ...(p.description !== undefined ? { description: p.description } : {}),
        };
      }
      for (const param of entry.fn.parameters) {
        const symbol = this.checker.getSymbolAtLocation(param.name);
        if (symbol !== undefined) this.hoist(symbol, param.name);
      }
      returns = read.returns;
      bodyStatements = [...this.entryBody(entry.fn)];
    } else {
      // The body is the function: its one input is Claude's `args` (SCRIPTS.md §5).
      for (const s of body) if (isTypeOnly(ts, s)) prelude.push(s);
      bodyStatements = body.filter((s) => !isTypeOnly(ts, s));
      inputs = { args: { optional: true, description: "What the caller passed — Claude's `args`." } };
      const argsSymbol = this.globalSymbol("args");
      if (argsSymbol !== undefined) this.hoistNamed(argsSymbol, "args", undefined);
    }

    this.checkPhaseCalls([...bodyStatements, ...prelude], entry?.fn);
    if (this.mode !== "calls") this.findSchemaParams([...bodyStatements, ...prelude]);

    // In "calls" mode every call becomes a statement of its own, and every fan-out an element.
    if (this.mode === "calls") bodyStatements = this.anfList(bodyStatements);

    // Every hoisted variable is decided BEFORE anything is rewritten, so a helper declared above a
    // variable it reads still reads it through `$v`.
    for (const s of bodyStatements) this.collectHoisting(s);

    // Lower the body into blocks — machine 0 — and then each element, which the body may have queued.
    this.machine = 0;
    this.cur = this.newBlock(lineOf(this.sf, bodyStatements[0] ?? sf));
    this.lowerList(bodyStatements);
    this.end({ kind: "return", value: undefined, fallthrough: true });
    for (let i = 0; i < this.elements.length; i++) this.lowerElement(this.elements[i]!);

    if (this.bodyMode && this.sawReturn && (this.exportNames.length > 0 || this.defaultExport !== undefined)) {
      throw new ScriptCompileError("a script returns a value OR exports its outputs — it does both", this.file);
    }
    // A body's return is its exports: the record they make, typed by the variables they are.
    if (this.bodyMode && this.exportNames.length > 0) {
      const properties: Record<string, JsonValue> = {};
      for (const name of this.exportNames) properties[name] = (this.hoistedByName.get(name)?.schema ?? {}) as JsonValue;
      returns = { type: "object", properties } as unknown as JsonSchema;
    }

    const built = new MachineBuilder(this, {
      stateId: this.options.stateId,
      meta,
      inputs,
      returns,
      generated,
      prelude,
      imports: this.codeImports,
      rootOutputs: (spelling, optional) => this.rootOutputs(returns, spelling, optional),
    }).build();
    this.checkUnusedPhaseDescriptions(meta, built.phases);
    return { mode: this.mode, meta, documents: built.documents, warnings: this.warnings, generated, sourceMap: built.sourceMap };
  }

  // --- imports -------------------------------------------------------------------------------------

  /**
   * What an import binds (SCRIPTS.md §10), decided by what it resolves to: a STATE (a `.json`/`.yaml`
   * document, or a script with a `meta`), a PROMPT (`.md`), a REGISTRY entry (`$REGISTRY`), DATA
   * (`with { type: "json" }`), code imported `with { as: "operation" }`, or plain CODE.
   */
  private classifyImport(declaration: TS.ImportDeclaration): void {
    const { ts } = this;
    const clause = declaration.importClause;
    if (!ts.isStringLiteral(declaration.moduleSpecifier)) return;
    const specifier = declaration.moduleSpecifier.text;
    if (clause === undefined || clause.isTypeOnly || specifier === HOOK_MODULE) {
      if (clause !== undefined && !clause.isTypeOnly) this.codeImports.push(declaration);
      return;
    }
    const attributes = importAttributes(ts, declaration);
    const locals = importedLocals(ts, clause);

    if (specifier === "$REGISTRY") {
      for (const { local, imported } of locals) this.imported.set(local, { kind: "registry", declaration, ref: imported ?? local });
      return;
    }
    if (attributes.type === "json") {
      const file = this.relative(specifier);
      const text = file !== undefined ? this.options.vfs.read(file) : undefined;
      if (text === undefined) throw this.error(declaration, `'${specifier}' could not be read`);
      if (clause.name === undefined) throw this.error(declaration, "a JSON import binds its value as a default import");
      this.imported.set(clause.name.text, { kind: "data", declaration, data: JSON.parse(text) as JsonValue });
      return;
    }
    if (specifier.toLowerCase().endsWith(".md")) {
      const file = this.relative(specifier);
      const text = file !== undefined ? this.options.vfs.read(file) : undefined;
      if (text === undefined) throw this.error(declaration, `'${specifier}' could not be read`);
      if (clause.name === undefined) throw this.error(declaration, "a prompt import binds its template as a default import");
      this.imported.set(clause.name.text, { kind: "prompt", declaration, template: text });
      return;
    }
    const state = this.stateFileOf(specifier);
    if (state !== undefined) {
      if (clause.namedBindings !== undefined || clause.name === undefined) {
        throw this.error(declaration, `'${specifier}' is a state: import it as a default — \`import name from "${specifier}"\` — which binds a function that runs it`);
      }
      const stateId = this.stateIdOfFile(state, declaration);
      this.imported.set(clause.name.text, { kind: "state", declaration, stateId, ...this.calledShape(state) });
      this.coarseCalls.set(sanitizeKey(clause.name.text), stateId);
      return;
    }
    if (attributes.as === "operation") {
      for (const { local } of locals) this.imported.set(local, { kind: "operation", declaration });
      this.codeImports.push(declaration);
      return;
    }
    this.codeImports.push(declaration);
  }

  private relative(specifier: string): string | undefined {
    if (specifier.startsWith(".")) return normalizePath(`${dirOf(this.file.replace(/\\/g, "/"))}/${specifier}`);
    if (specifier.startsWith("/")) return specifier;
    return resolveSpecifier(specifier, dirOf(this.file.replace(/\\/g, "/")), this.options);
  }

  /** The state file a specifier names, if it names one. */
  private stateFileOf(specifier: string): string | undefined {
    const options = this.options;
    const base = specifier.startsWith(".") ? normalizePath(`${dirOf(this.file.replace(/\\/g, "/"))}/${specifier}`) : undefined;
    if (base !== undefined) {
      for (const ext of DATA_EXTENSIONS) if (options.vfs.read(`${base}.${ext}`) !== undefined) return `${base}.${ext}`;
      if (/\.(json|ya?ml)$/i.test(base) && options.vfs.read(base) !== undefined) return base;
    }
    const module = resolveSpecifier(specifier, dirOf(this.file.replace(/\\/g, "/")), options);
    if (module === undefined || module === this.file) return undefined;
    const text = options.vfs.read(module);
    if (text === undefined || !hasScriptMeta(this.ts, module, text)) return undefined;
    // A script that declares itself a library is code, not a state.
    return /compile\s*:\s*["']function["']/.test(text) ? undefined : module;
  }

  /** What a called state takes and gives back — its input names, and where a script's whole return is. */
  private calledShape(file: string): Pick<ImportedName, "inputNames" | "argsInput" | "whole" | "inputSchemas"> {
    const text = this.options.vfs.read(file) ?? "";
    if (/\.(json|ya?ml)$/i.test(file)) {
      const doc = parseReferencedFile(file, text) as { inputs?: Record<string, { schema?: JsonSchema }> } | undefined;
      const names = Object.keys(doc?.inputs ?? {});
      const inputSchemas: Record<string, JsonSchema> = {};
      for (const name of names) {
        const schema = doc!.inputs![name]!.schema;
        if (schema !== undefined) inputSchemas[name] = schema;
      }
      return { inputNames: names, inputSchemas };
    }
    // A script: its exported function's parameters, or Claude's `args`.
    const sf = this.program.getSourceFile(file);
    const moduleSymbol = sf !== undefined ? this.checker.getSymbolAtLocation(sf) : undefined;
    const exported = moduleSymbol !== undefined ? this.checker.getExportsOfModule(moduleSymbol) : [];
    const fn = exported.find((e) => e.getName() === "default") ?? exported.find((e) => e.getName() !== "meta");
    const type = fn !== undefined && sf !== undefined ? this.checker.getTypeOfSymbolAtLocation(fn, sf) : undefined;
    const signature = type?.getCallSignatures()[0];
    if (signature === undefined) return { inputNames: ["args"], argsInput: true, whole: "result" };
    const returned = awaited(this.ts, this.checker, signature.getReturnType());
    const record = (returned.flags & this.ts.TypeFlags.Object) !== 0 && this.checker.getPropertiesOfType(returned).length > 0 && !this.checker.isArrayType(returned);
    const inputSchemas: Record<string, JsonSchema> = {};
    try {
      for (const p of readSignature(this.ts, this.checker, sf!, signature, file).parameters) if (Object.keys(p.schema).length > 0) inputSchemas[p.name] = p.schema;
    } catch {
      /* an unreadable signature types nothing, which the wire then asserts */
    }
    return { inputNames: signature.getParameters().map((p) => p.getName()), inputSchemas, ...(record ? {} : { whole: "result" }) };
  }

  private stateIdOfFile(file: string, at: TS.Node): string {
    const id = this.options.stateIdOf?.(file) ?? (file.startsWith(`${this.root}/`) ? file.slice(this.root.length + 1).replace(/\.[^./]+$/, "") : undefined);
    if (id === undefined) throw this.error(at, `'${file}' is not under the workflow root '${this.root}', so it names no state`);
    return id;
  }

  /**
   * The state a `workflow(ref)` names — a LITERAL, since a computed name can be neither mounted nor
   * frozen (SCRIPTS.md §10). A relative ref is read from the script's directory, a bare one from the
   * workflow root; Claude's `{ scriptPath }` is a path.
   */
  workflowTarget(call: TS.CallExpression): { stateId: string } & Pick<ImportedName, "inputNames" | "argsInput" | "whole" | "inputSchemas"> {
    const { ts } = this;
    const arg = call.arguments[0];
    let ref: string | undefined;
    if (arg !== undefined && (ts.isStringLiteral(arg) || ts.isNoSubstitutionTemplateLiteral(arg))) ref = arg.text;
    else if (arg !== undefined && ts.isObjectLiteralExpression(arg)) {
      const path = arg.properties.find((p): p is TS.PropertyAssignment => ts.isPropertyAssignment(p) && ts.isIdentifier(p.name) && p.name.text === "scriptPath");
      if (path !== undefined && (ts.isStringLiteral(path.initializer) || ts.isNoSubstitutionTemplateLiteral(path.initializer))) ref = path.initializer.text;
    }
    if (ref === undefined) throw this.error(call, "workflow() names its state with a literal — a computed name can be neither declared nor frozen; import the state instead");
    const clean = ref.replace(/\.(json|ya?ml|ts|js)$/, "");
    const stateId = clean.startsWith(".") ? this.stateIdOfFile(normalizePath(`${dirOf(this.file.replace(/\\/g, "/"))}/${clean}`), call) : clean.replace(/^\/+/, "");
    // Where the file is, when it is under the root: what it takes and gives back.
    let shape: Pick<ImportedName, "inputNames" | "argsInput" | "whole" | "inputSchemas"> = {};
    for (const ext of [...DATA_EXTENSIONS, "ts", "js"]) {
      const file = `${this.root}/${stateId}.${ext}`;
      if (this.options.vfs.read(file) !== undefined) {
        shape = this.calledShape(file);
        break;
      }
    }
    return { stateId, ...shape };
  }

  // --- meta ----------------------------------------------------------------------------------------

  private readMeta(declaration: TS.VariableDeclaration): ScriptMeta {
    if (declaration.initializer === undefined) throw this.error(declaration, "`meta` has no value");
    const value = this.literal(declaration.initializer, "meta");
    if (value === null || typeof value !== "object" || Array.isArray(value)) throw this.error(declaration, "`meta` must be an object literal");
    const meta = value as Record<string, JsonValue>;
    for (const key of Object.keys(meta)) {
      const refused = REFUSED_META[key];
      if (refused !== undefined) throw this.error(declaration, `meta.${key} is not a script property — ${refused}`);
      if (!META_KEYS.has(key)) throw this.error(declaration, `meta.${key} is not a property a script's meta takes (${[...META_KEYS].join(", ")})`);
    }
    if (meta.name !== undefined && meta.label !== undefined) {
      throw this.error(declaration, "meta names both `name` and `label` — they are one field (Claude's spelling and hw's)");
    }
    if (meta.compile !== undefined && !SCRIPT_MODES.includes(meta.compile as ScriptMode)) {
      throw this.error(declaration, `meta.compile is '${String(meta.compile)}' — one of ${SCRIPT_MODES.join(", ")}`);
    }
    return meta as unknown as ScriptMeta;
  }

  /** A PURE LITERAL — Claude's rule for `meta`, and what lets it be read without running anything. */
  private literal(node: TS.Expression, where: string): JsonValue {
    const value = this.staticValue(node, false);
    if (value === undefined) throw this.error(node, `${where} must be a pure literal — no identifiers, calls, spreads or interpolation`);
    return value;
  }

  /**
   * The value an expression has at COMPILE time, when it has one: a literal, or — where `constants` —
   * a `const` whose initializer does, which is how Claude's top-level `FINDINGS` schema reaches a
   * call's output contract. `undefined` when it has none.
   */
  staticValue(node: TS.Expression, constants: boolean, seen: Set<TS.Node> = new Set()): JsonValue | undefined {
    const { ts } = this;
    const e = unwrap(ts, node);
    if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return e.text;
    if (ts.isNumericLiteral(e)) return Number(e.text);
    if (e.kind === ts.SyntaxKind.TrueKeyword) return true;
    if (e.kind === ts.SyntaxKind.FalseKeyword) return false;
    if (e.kind === ts.SyntaxKind.NullKeyword) return null;
    if (ts.isPrefixUnaryExpression(e) && e.operator === ts.SyntaxKind.MinusToken && ts.isNumericLiteral(e.operand)) return -Number(e.operand.text);
    if (ts.isArrayLiteralExpression(e)) {
      const out: JsonValue[] = [];
      for (const x of e.elements) {
        if (ts.isOmittedExpression(x)) return undefined;
        if (ts.isSpreadElement(x)) {
          if (!constants) return undefined;
          const spread = this.staticValue(x.expression, constants, seen);
          if (!Array.isArray(spread)) return undefined;
          out.push(...spread);
          continue;
        }
        const v = this.staticValue(x, constants, seen);
        if (v === undefined) return undefined;
        out.push(v);
      }
      return out;
    }
    if (ts.isObjectLiteralExpression(e)) {
      const out: Record<string, JsonValue> = {};
      for (const member of e.properties) {
        if (ts.isSpreadAssignment(member) && constants) {
          const spread = this.staticValue(member.expression, constants, seen);
          if (spread === null || typeof spread !== "object" || Array.isArray(spread)) return undefined;
          Object.assign(out, spread);
          continue;
        }
        if (ts.isShorthandPropertyAssignment(member) && constants) {
          const v = this.constantOf(member.name, seen);
          if (v === undefined) return undefined;
          out[member.name.text] = v;
          continue;
        }
        if (!ts.isPropertyAssignment(member)) return undefined;
        const name = member.name;
        const key = ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name) ? name.text : undefined;
        if (key === undefined) return undefined;
        const v = this.staticValue(member.initializer, constants, seen);
        if (v === undefined) return undefined;
        out[key] = v;
      }
      return out;
    }
    if (constants && ts.isIdentifier(e)) return this.constantOf(e, seen);
    if (constants && ts.isPropertyAccessExpression(e)) {
      const object = this.staticValue(e.expression, constants, seen);
      return object !== null && typeof object === "object" && !Array.isArray(object) ? (object as Record<string, JsonValue>)[e.name.text] : undefined;
    }
    return undefined;
  }

  private constantOf(identifier: TS.Identifier, seen: Set<TS.Node>): JsonValue | undefined {
    const { ts } = this;
    const imported = this.imported.get(identifier.text);
    if (imported?.kind === "data") return imported.data;
    const symbol = this.checker.getSymbolAtLocation(identifier);
    const declaration = symbol?.valueDeclaration;
    if (declaration === undefined || !ts.isVariableDeclaration(declaration) || declaration.initializer === undefined) return undefined;
    if ((declaration.parent.flags & ts.NodeFlags.Const) === 0 || seen.has(declaration)) return undefined;
    seen.add(declaration);
    return this.staticValue(declaration.initializer, true, seen);
  }

  // --- the entry function --------------------------------------------------------------------------

  /** The exported function whose signature is the state's (SCRIPTS.md §5), when there is one. */
  private entryFunction(body: readonly TS.Statement[]): { fn: TS.FunctionLikeDeclaration; statement: TS.Statement } | undefined {
    const { ts } = this;
    const found: Array<{ fn: TS.FunctionLikeDeclaration; statement: TS.Statement; isDefault: boolean }> = [];
    for (const s of body) {
      const exported = hasModifier(ts, s, ts.SyntaxKind.ExportKeyword);
      const isDefault = hasModifier(ts, s, ts.SyntaxKind.DefaultKeyword);
      if (ts.isFunctionDeclaration(s) && exported && s.body !== undefined) found.push({ fn: s, statement: s, isDefault });
      else if (ts.isVariableStatement(s) && exported) {
        for (const d of s.declarationList.declarations) {
          const init = d.initializer && unwrap(ts, d.initializer);
          if (init !== undefined && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) found.push({ fn: init, statement: s, isDefault: false });
        }
      } else if (ts.isExportAssignment(s) && !s.isExportEquals) {
        const init = unwrap(ts, s.expression);
        if (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) found.push({ fn: init, statement: s, isDefault: true });
      }
    }
    const byDefault = found.find((f) => f.isDefault);
    if (byDefault !== undefined) return byDefault;
    if (found.length === 1) return found[0];
    if (found.length > 1) {
      throw new ScriptCompileError(
        `exports ${found.length} functions and none is the default — which one is the state? Export it as the default, or set meta.compile to "function" for a library`,
        this.file,
      );
    }
    return undefined;
  }

  private entryBody(fn: TS.FunctionLikeDeclaration): readonly TS.Statement[] {
    const { ts } = this;
    if (fn.body === undefined) throw this.error(fn, "has no body");
    if (ts.isBlock(fn.body)) return fn.body.statements;
    // `async (x) => expr` — the expression is what it returns.
    return [ts.factory.createReturnStatement(fn.body as TS.Expression)];
  }

  // --- hoisting ------------------------------------------------------------------------------------

  hoist(symbol: TS.Symbol, at: TS.Node): Hoisted {
    return this.hoistNamed(symbol, symbol.getName(), at);
  }

  private hoistNamed(symbol: TS.Symbol, base: string, at: TS.Node | undefined): Hoisted {
    const existing = this.hoisted.get(symbol);
    if (existing !== undefined) return existing;
    let name = base;
    for (let n = 1; this.hoistedByName.has(name) || CONTROL_NAMES.has(name); n++) name = `${base}_${n}`;
    const entry: Hoisted = { name };
    if (at !== undefined) {
      try {
        const type = this.checker.getTypeOfSymbolAtLocation(symbol, at);
        const slot = typeToWireSlot(this.ts, this.checker, type, `variable '${base}'`);
        if (Object.keys(slot.schema).length > 0) entry.schema = slot.schema;
      } catch (e) {
        if (!(e instanceof WireTypeError)) throw e;
        entry.unrepresentable = e.message;
      }
    }
    this.hoisted.set(symbol, entry);
    this.hoistedByName.set(name, entry);
    return entry;
  }

  syntheticVar(base: string, schema: JsonSchema | undefined): Hoisted {
    let name = `${base}_${this.synthetic++}`;
    while (this.hoistedByName.has(name)) name = `${base}_${this.synthetic++}`;
    const entry: Hoisted = { name, ...(schema !== undefined && Object.keys(schema).length > 0 ? { schema } : {}) };
    this.hoistedByName.set(name, entry);
    return entry;
  }

  private globalSymbol(name: string): TS.Symbol | undefined {
    const globals = this.program.getSourceFile(HOOK_GLOBALS_PATH);
    if (globals === undefined) return undefined;
    for (const s of globals.statements) {
      if (this.ts.isVariableStatement(s)) {
        for (const d of s.declarationList.declarations) {
          if (this.ts.isIdentifier(d.name) && d.name.text === name) return this.checker.getSymbolAtLocation(d.name);
        }
      }
    }
    return undefined;
  }

  // --- hooks ---------------------------------------------------------------------------------------

  /** Which hook an identifier names, if any — through a global or an import of the hook module. */
  hookOf(node: TS.Identifier): string | undefined {
    let symbol = this.checker.getSymbolAtLocation(node);
    if (symbol === undefined) return undefined;
    if (symbol.flags & this.ts.SymbolFlags.Alias) symbol = this.checker.getAliasedSymbol(symbol);
    const declaration = symbol.declarations?.[0];
    const file = declaration?.getSourceFile().fileName;
    if (file !== HOOK_GLOBALS_PATH && file !== HOOK_MODULE_PATH) return undefined;
    const name = symbol.getName();
    return name === "args" ? undefined : name;
  }

  /** A hook named as a GLOBAL (Claude's spelling) — rewritten to the hook module's export. */
  isGlobalHook(n: TS.Identifier): boolean {
    const symbol = this.checker.getSymbolAtLocation(n);
    return symbol !== undefined && (symbol.flags & this.ts.SymbolFlags.Alias) === 0 && symbol.declarations?.[0]?.getSourceFile().fileName === HOOK_GLOBALS_PATH && symbol.getName() !== "args";
  }

  /** What an identifier a script CALLS is, when the script imported it (SCRIPTS.md §10). */
  importedAs(node: TS.Identifier): ImportedName | undefined {
    const imported = this.imported.get(node.text);
    if (imported === undefined) return undefined;
    // Only the import itself — a local of the same name shadows it.
    const symbol = this.checker.getSymbolAtLocation(node);
    return symbol !== undefined && (symbol.flags & this.ts.SymbolFlags.Alias) !== 0 ? imported : undefined;
  }

  private isPhaseCall(node: TS.Node): node is TS.CallExpression {
    return this.ts.isCallExpression(node) && this.ts.isIdentifier(node.expression) && this.hookOf(node.expression) === "phase";
  }

  /** `phase("X");` or `await phase("X");` as a statement of its own — the one spelling a cut takes. */
  private phaseStatement(s: TS.Statement): string | undefined {
    const { ts } = this;
    if (!ts.isExpressionStatement(s)) return undefined;
    let e = s.expression;
    if (ts.isAwaitExpression(e)) e = e.expression;
    if (!this.isPhaseCall(e)) return undefined;
    // In "state" mode a phase is Claude's display group, and its title may be anything.
    if (this.mode === "state") return "";
    const arg = e.arguments[0];
    if (arg === undefined || !(ts.isStringLiteral(arg) || ts.isNoSubstitutionTemplateLiteral(arg)) || e.arguments.length !== 1) {
      throw this.error(e, "phase() takes one string literal — a phase is a state, and a computed name has no state to be");
    }
    if (arg.text.length === 0) throw this.error(e, "phase(\"\") names no phase");
    return arg.text;
  }

  /**
   * Every `phase()` call must be one the compiler can cut at — a statement of its own, in the entry
   * function (or the script body), outside every callback (SCRIPTS.md §6.1). In `"state"` mode a phase
   * is Claude's display group and may be called from anywhere.
   */
  private checkPhaseCalls(statements: readonly TS.Statement[], entry: TS.FunctionLikeDeclaration | undefined): void {
    if (this.mode === "state") return;
    const { ts } = this;
    const visit = (node: TS.Node, insideFunction: boolean): void => {
      if (this.isPhaseCall(node)) {
        if (insideFunction) {
          throw this.error(node, "phase() inside a function or a callback — only the script's own control flow can move to another phase (a `parallel` thunk or a `pipeline` stage cannot); use `{ phase }` on the call to group it instead");
        }
        const parent = ts.isAwaitExpression(node.parent) ? node.parent.parent : node.parent;
        if (!ts.isExpressionStatement(parent)) throw this.error(node, "phase() must be a statement of its own: `phase(\"X\");`");
      }
      const boundary = isFunctionBoundary(ts, node) && node !== entry;
      ts.forEachChild(node, (child) => visit(child, insideFunction || boundary));
    };
    for (const s of statements) visit(s, false);
  }

  // --- calls: what is one --------------------------------------------------------------------------

  /** A helper's declaration, when an identifier a script calls is a script-local function. */
  private helperOf(callee: TS.Expression): TS.FunctionLikeDeclaration | undefined {
    const { ts } = this;
    const e = unwrap(ts, callee);
    if (ts.isArrowFunction(e) || ts.isFunctionExpression(e)) return e;
    if (!ts.isIdentifier(e)) return undefined;
    const symbol = this.checker.getSymbolAtLocation(e);
    const declaration = symbol?.valueDeclaration;
    if (declaration === undefined || declaration.getSourceFile() !== this.sf) return undefined;
    if (ts.isFunctionDeclaration(declaration) && declaration.body !== undefined) return declaration;
    if (ts.isVariableDeclaration(declaration) && declaration.initializer !== undefined) {
      const init = unwrap(ts, declaration.initializer);
      if ((ts.isArrowFunction(init) || ts.isFunctionExpression(init)) && (declaration.parent.flags & ts.NodeFlags.Const) !== 0) return init;
    }
    return undefined;
  }

  /** Whether a helper makes a call, through however many helpers it calls itself. */
  private helperMakesCall(fn: TS.FunctionLikeDeclaration): boolean {
    const known = this.helperCuts.get(fn);
    if (known !== undefined) return known;
    this.helperCuts.set(fn, false); // a recursive reference answers "not yet", which the refusal below catches
    const result = fn.body !== undefined && this.hasCut(fn.body, true);
    this.helperCuts.set(fn, result);
    return result;
  }

  /**
   * Whether a call expression is a CUT — a non-deterministic call, a fan-out over one, or a helper
   * that makes one (SCRIPTS.md §7).
   */
  isCutCall(call: TS.CallExpression): boolean {
    const { ts } = this;
    const callee = unwrap(ts, call.expression);
    if (ts.isIdentifier(callee)) {
      const hook = this.hookOf(callee);
      if (hook === "llm" || hook === "agent" || hook === "now" || hook === "random" || hook === "workflow") return true;
      if (hook === "parallel" || hook === "pipeline") return call.arguments.some((a) => this.hasCut(a, true));
      const imported = this.importedAs(callee);
      if (imported !== undefined) return imported.kind === "state" || imported.kind === "prompt" || imported.kind === "registry" || imported.kind === "operation";
    }
    if (ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === "Promise" && callee.name.text === "all") {
      return call.arguments.some((a) => this.hasCut(a, true));
    }
    const helper = this.helperOf(call.expression);
    return helper !== undefined && this.helperMakesCall(helper);
  }

  /**
   * Whether a node makes a call — outside the functions it declares, unless `intoFunctions`, which is
   * how a fan-out's callback and a helper's body are asked.
   */
  hasCut(node: TS.Node, intoFunctions = false): boolean {
    if (this.mode !== "calls") return false;
    const { ts } = this;
    let found = false;
    const visit = (n: TS.Node): void => {
      if (found) return;
      if (this.markers.has(n as TS.Statement)) {
        found = true;
        return;
      }
      if (ts.isCallExpression(n) && this.isCutCall(n)) {
        found = true;
        return;
      }
      if (!intoFunctions && isFunctionBoundary(ts, n) && n !== node) return;
      ts.forEachChild(n, visit);
    };
    visit(node);
    return found;
  }

  /** Whether a statement contains a CUT the lowering has to cut at — a `phase()` or a call. */
  private containsCut(node: TS.Node): boolean {
    if (this.mode === "state") return false;
    const { ts } = this;
    let found = false;
    const visit = (n: TS.Node): void => {
      if (found) return;
      if (this.markers.has(n as TS.Statement) || this.isPhaseCall(n)) {
        found = true;
        return;
      }
      if (isFunctionBoundary(ts, n) && n !== node) return;
      ts.forEachChild(n, visit);
    };
    visit(node);
    return found;
  }

  // --- calls: lifting each one into a statement of its own ("calls" mode) -----------------------------

  private anfList(statements: readonly TS.Statement[]): TS.Statement[] {
    return statements.flatMap((s) => this.anfStatement(s));
  }

  /**
   * One statement with every call in it lifted out: the calls become marker statements in evaluation
   * order, each followed by what used their results (SCRIPTS.md §7).
   */
  private anfStatement(s: TS.Statement): TS.Statement[] {
    const { ts } = this;
    const f = ts.factory;
    if (!this.hasCut(s)) return [s];
    const block = (statements: TS.Statement[]): TS.Statement => (statements.length === 1 && ts.isBlock(statements[0]!) ? statements[0]! : f.createBlock(statements, true));
    if (ts.isBlock(s)) return [f.updateBlock(s, this.anfList(s.statements))];
    if (ts.isVariableStatement(s)) {
      const out: TS.Statement[] = [];
      for (const d of s.declarationList.declarations) {
        let initializer = d.initializer;
        if (initializer !== undefined && this.hasCut(initializer)) {
          const r = this.anfExpr(initializer);
          out.push(...r.pre);
          initializer = r.value;
        }
        const declaration = f.updateVariableDeclaration(d, d.name, d.exclamationToken, d.type, initializer);
        out.push(f.createVariableStatement(ts.getModifiers(s), f.createVariableDeclarationList([declaration], s.declarationList.flags)));
      }
      return out;
    }
    if (ts.isExpressionStatement(s)) {
      const r = this.anfExpr(s.expression);
      return [...r.pre, ...(isTempRef(ts, r.value) ? [] : [f.updateExpressionStatement(s, r.value)])];
    }
    if (ts.isReturnStatement(s)) {
      const r = this.anfExpr(s.expression!);
      return [...r.pre, f.updateReturnStatement(s, r.value)];
    }
    if (ts.isThrowStatement(s)) {
      const r = this.anfExpr(s.expression);
      return [...r.pre, f.updateThrowStatement(s, r.value)];
    }
    if (ts.isExportAssignment(s)) {
      const r = this.anfExpr(s.expression);
      return [...r.pre, f.updateExportAssignment(s, ts.getModifiers(s), r.value)];
    }
    if (ts.isIfStatement(s)) {
      const r = this.anfExpr(s.expression);
      const then = block(this.anfStatement(s.thenStatement));
      const otherwise = s.elseStatement !== undefined ? block(this.anfStatement(s.elseStatement)) : undefined;
      return [...r.pre, f.updateIfStatement(s, r.value, then, otherwise)];
    }
    if (ts.isWhileStatement(s)) {
      const body = block(this.anfStatement(s.statement));
      if (!this.hasCut(s.expression)) return [f.updateWhileStatement(s, s.expression, body)];
      // `while (C) S` with a call in C: the condition is re-evaluated at the top of every pass.
      const r = this.anfExpr(s.expression);
      const guard = f.createIfStatement(f.createPrefixUnaryExpression(ts.SyntaxKind.ExclamationToken, f.createParenthesizedExpression(r.value)), f.createBreakStatement());
      return [f.createWhileStatement(f.createTrue(), f.createBlock([...r.pre, guard, body], true))];
    }
    if (ts.isDoStatement(s)) {
      if (this.hasCut(s.expression)) throw this.error(s.expression, "a call in a `do…while` condition cannot be cut — compute it at the end of the body into a variable, and test that");
      return [f.updateDoStatement(s, block(this.anfStatement(s.statement)), s.expression)];
    }
    if (ts.isForStatement(s)) {
      if ((s.condition !== undefined && this.hasCut(s.condition)) || (s.incrementor !== undefined && this.hasCut(s.incrementor))) {
        throw this.error(s, "a call in a `for` loop's condition or step cannot be cut — write the loop as a `while` and make the call in its body");
      }
      const pre: TS.Statement[] = [];
      let initializer = s.initializer;
      if (initializer !== undefined && this.hasCut(initializer)) {
        pre.push(...this.anfStatement(ts.isVariableDeclarationList(initializer) ? f.createVariableStatement(undefined, initializer) : f.createExpressionStatement(initializer)));
        initializer = undefined;
      }
      return [...pre, f.updateForStatement(s, initializer, s.condition, s.incrementor, block(this.anfStatement(s.statement)))];
    }
    if (ts.isForOfStatement(s) || ts.isForInStatement(s)) {
      const r = this.anfExpr(s.expression);
      const body = block(this.anfStatement(s.statement));
      return [...r.pre, ts.isForOfStatement(s) ? f.updateForOfStatement(s, s.awaitModifier, s.initializer, r.value, body) : f.updateForInStatement(s, s.initializer, r.value, body)];
    }
    if (ts.isLabeledStatement(s)) {
      const inner = this.anfStatement(s.statement);
      const last = inner.pop()!;
      return [...inner, f.updateLabeledStatement(s, s.label, last)];
    }
    if (ts.isSwitchStatement(s)) {
      const r = this.anfExpr(s.expression);
      const clauses = s.caseBlock.clauses.map((c) => {
        if (ts.isCaseClause(c) && this.hasCut(c.expression)) throw this.error(c.expression, "a call in a `case` label cannot be cut — compute it before the `switch`");
        return ts.isCaseClause(c) ? f.updateCaseClause(c, c.expression, this.anfList(c.statements)) : f.updateDefaultClause(c, this.anfList(c.statements));
      });
      return [...r.pre, f.updateSwitchStatement(s, r.value, f.updateCaseBlock(s.caseBlock, clauses))];
    }
    if (ts.isTryStatement(s)) {
      const tryBlock = f.updateBlock(s.tryBlock, this.anfList(s.tryBlock.statements));
      const catchClause = s.catchClause !== undefined ? f.updateCatchClause(s.catchClause, s.catchClause.variableDeclaration, f.updateBlock(s.catchClause.block, this.anfList(s.catchClause.block.statements))) : undefined;
      const finallyBlock = s.finallyBlock !== undefined ? f.updateBlock(s.finallyBlock, this.anfList(s.finallyBlock.statements)) : undefined;
      return [f.updateTryStatement(s, tryBlock, catchClause, finallyBlock)];
    }
    throw this.error(s, "a call inside this statement cannot be cut — make the call in a statement of its own");
  }

  /** An expression with every call in it lifted out: what to run first, and the value that is left. */
  private anfExpr(e: TS.Expression): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    if (!this.hasCut(e)) return { pre: [], value: e };
    if (ts.isParenthesizedExpression(e)) {
      const r = this.anfExpr(e.expression);
      return { pre: r.pre, value: f.updateParenthesizedExpression(e, r.value) };
    }
    if (ts.isAsExpression(e) || ts.isSatisfiesExpression(e) || ts.isNonNullExpression(e) || ts.isTypeAssertionExpression(e)) {
      return this.anfExpr(e.expression);
    }
    if (ts.isAwaitExpression(e)) {
      const inner = unwrap(ts, e.expression);
      if (ts.isCallExpression(inner) && this.isCutCall(inner)) return this.cut(inner);
      const r = this.anfExpr(e.expression);
      return { pre: r.pre, value: f.updateAwaitExpression(e, r.value) };
    }
    if (ts.isCallExpression(e)) {
      if (this.isCutCall(e)) return this.cut(e);
      const pre: TS.Statement[] = [];
      const argsCut = e.arguments.some((a) => this.hasCut(a));
      let callee = e.expression;
      if (ts.isPropertyAccessExpression(callee)) {
        const r = this.anfExpr(callee.expression);
        pre.push(...r.pre);
        callee = f.updatePropertyAccessExpression(callee, argsCut ? this.hold(r.value, pre) : r.value, callee.name);
      } else if (this.hasCut(callee) || argsCut) {
        const r = this.anfExpr(callee);
        pre.push(...r.pre);
        callee = (argsCut ? this.hold(r.value, pre) : r.value) as TS.LeftHandSideExpression;
      }
      const args = this.anfSequence(e.arguments);
      return { pre: [...pre, ...args.pre], value: f.updateCallExpression(e, callee as TS.LeftHandSideExpression, e.typeArguments, args.values) };
    }
    if (ts.isNewExpression(e)) {
      const parts = this.anfSequence([e.expression, ...(e.arguments ?? [])]);
      return { pre: parts.pre, value: f.updateNewExpression(e, parts.values[0]!, e.typeArguments, parts.values.slice(1)) };
    }
    if (ts.isBinaryExpression(e)) return this.anfBinary(e);
    if (ts.isConditionalExpression(e)) {
      const c = this.anfExpr(e.condition);
      if (!this.hasCut(e.whenTrue) && !this.hasCut(e.whenFalse)) return { pre: c.pre, value: f.updateConditionalExpression(e, c.value, e.questionToken, e.whenTrue, e.colonToken, e.whenFalse) };
      const t = this.syntheticVar("t", undefined);
      const a = this.anfExpr(e.whenTrue);
      const b = this.anfExpr(e.whenFalse);
      const branch = f.createIfStatement(c.value, f.createBlock([...a.pre, assign(ts, t.name, a.value)], true), f.createBlock([...b.pre, assign(ts, t.name, b.value)], true));
      return { pre: [...c.pre, branch], value: vRef(ts, t.name) };
    }
    if (ts.isPrefixUnaryExpression(e)) {
      const r = this.anfExpr(e.operand);
      return { pre: r.pre, value: f.updatePrefixUnaryExpression(e, r.value as TS.UnaryExpression) };
    }
    if (ts.isPostfixUnaryExpression(e)) {
      const r = this.anfExpr(e.operand);
      return { pre: r.pre, value: f.updatePostfixUnaryExpression(e, r.value as TS.LeftHandSideExpression) };
    }
    if (ts.isTypeOfExpression(e) || ts.isVoidExpression(e) || ts.isDeleteExpression(e)) {
      const r = this.anfExpr(e.expression);
      const value = ts.isTypeOfExpression(e) ? f.updateTypeOfExpression(e, r.value) : ts.isVoidExpression(e) ? f.updateVoidExpression(e, r.value) : f.updateDeleteExpression(e, r.value);
      return { pre: r.pre, value };
    }
    if (ts.isPropertyAccessExpression(e)) {
      const r = this.anfExpr(e.expression);
      return { pre: r.pre, value: f.updatePropertyAccessExpression(e, r.value, e.name) };
    }
    if (ts.isElementAccessExpression(e)) {
      const parts = this.anfSequence([e.expression, e.argumentExpression]);
      return { pre: parts.pre, value: f.updateElementAccessExpression(e, parts.values[0]!, parts.values[1]!) };
    }
    if (ts.isArrayLiteralExpression(e)) {
      const parts = this.anfSequence(e.elements);
      return { pre: parts.pre, value: f.updateArrayLiteralExpression(e, parts.values) };
    }
    if (ts.isObjectLiteralExpression(e)) return this.anfObject(e);
    if (ts.isTemplateExpression(e)) {
      const parts = this.anfSequence(e.templateSpans.map((span) => span.expression));
      return { pre: parts.pre, value: f.updateTemplateExpression(e, e.head, e.templateSpans.map((span, i) => f.updateTemplateSpan(span, parts.values[i]!, span.literal))) };
    }
    if (ts.isSpreadElement(e)) {
      const r = this.anfExpr(e.expression);
      return { pre: r.pre, value: f.updateSpreadElement(e, r.value) };
    }
    throw this.error(e, "a call inside this expression cannot be cut — make the call in a statement of its own");
  }

  /** `&&`, `||` and `??` around a call become branches; an assignment keeps JavaScript's order. */
  private anfBinary(e: TS.BinaryExpression): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    const op = e.operatorToken.kind;
    const K = ts.SyntaxKind;
    if (op === K.AmpersandAmpersandToken || op === K.BarBarToken || op === K.QuestionQuestionToken) {
      const left = this.anfExpr(e.left);
      if (!this.hasCut(e.right)) return { pre: left.pre, value: f.updateBinaryExpression(e, left.value, e.operatorToken, e.right) };
      const t = this.syntheticVar("t", undefined);
      const right = this.anfExpr(e.right);
      const held = vRef(ts, t.name);
      const test = op === K.AmpersandAmpersandToken ? held : op === K.BarBarToken ? f.createPrefixUnaryExpression(K.ExclamationToken, held) : f.createBinaryExpression(held, K.EqualsEqualsToken, f.createNull());
      return {
        pre: [...left.pre, assign(ts, t.name, left.value), f.createIfStatement(test, f.createBlock([...right.pre, assign(ts, t.name, right.value)], true))],
        value: vRef(ts, t.name),
      };
    }
    if (op === K.CommaToken) {
      const left = this.anfExpr(e.left);
      const right = this.anfExpr(e.right);
      return { pre: [...left.pre, f.createExpressionStatement(left.value), ...right.pre], value: right.value };
    }
    if (op === K.EqualsToken) {
      const pre: TS.Statement[] = [];
      let target = e.left;
      // The compiler's own `$v.x` needs no holding: nothing a call does can move it.
      if (ts.isPropertyAccessExpression(target) && !isTempRef(ts, target) && this.hasCut(e.right)) {
        const object = this.anfExpr(target.expression);
        pre.push(...object.pre);
        target = f.updatePropertyAccessExpression(target, this.hold(object.value, pre), target.name);
      } else if (ts.isElementAccessExpression(target) && this.hasCut(e.right)) {
        const parts = this.anfSequence([target.expression, target.argumentExpression]);
        pre.push(...parts.pre);
        target = f.updateElementAccessExpression(target, this.hold(parts.values[0]!, pre), this.hold(parts.values[1]!, pre));
      }
      const right = this.anfExpr(e.right);
      return { pre: [...pre, ...right.pre], value: f.updateBinaryExpression(e, target, e.operatorToken, right.value) };
    }
    if (op >= K.FirstCompoundAssignment && op <= K.LastCompoundAssignment) {
      if (op === K.AmpersandAmpersandEqualsToken || op === K.BarBarEqualsToken || op === K.QuestionQuestionEqualsToken) {
        // `x ||= await llm(…)` is `x || (x = await llm(…))`.
        const base = op === K.AmpersandAmpersandEqualsToken ? K.AmpersandAmpersandToken : op === K.BarBarEqualsToken ? K.BarBarToken : K.QuestionQuestionToken;
        return this.anfExpr(f.createBinaryExpression(e.left, base, f.createParenthesizedExpression(f.createAssignment(e.left, e.right))));
      }
      // `x += await llm(…)` reads `x` BEFORE the call.
      const pre: TS.Statement[] = [];
      const t = this.syntheticVar("t", undefined);
      pre.push(assign(ts, t.name, e.left));
      const right = this.anfExpr(e.right);
      return { pre: [...pre, ...right.pre], value: f.createAssignment(e.left, f.createBinaryExpression(vRef(ts, t.name), compoundBase(ts, op), right.value)) };
    }
    const parts = this.anfSequence([e.left, e.right]);
    return { pre: parts.pre, value: f.updateBinaryExpression(e, parts.values[0]!, e.operatorToken, parts.values[1]!) };
  }

  private anfObject(e: TS.ObjectLiteralExpression): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    const values: TS.Expression[] = [];
    for (const p of e.properties) {
      if (ts.isPropertyAssignment(p)) {
        if (ts.isComputedPropertyName(p.name)) values.push(p.name.expression);
        values.push(p.initializer);
      } else if (ts.isSpreadAssignment(p)) values.push(p.expression);
    }
    const parts = this.anfSequence(values);
    let i = 0;
    const properties = e.properties.map((p) => {
      if (ts.isPropertyAssignment(p)) {
        const name = ts.isComputedPropertyName(p.name) ? f.updateComputedPropertyName(p.name, parts.values[i++]!) : p.name;
        return f.updatePropertyAssignment(p, name, parts.values[i++]!);
      }
      if (ts.isSpreadAssignment(p)) return f.updateSpreadAssignment(p, parts.values[i++]!);
      return p;
    });
    return { pre: parts.pre, value: f.updateObjectLiteralExpression(e, properties) };
  }

  /**
   * Expressions evaluated left to right, some of which make calls: each one evaluated BEFORE a later
   * call is held in a temporary, so the call does not move ahead of it (SCRIPTS.md §7).
   */
  private anfSequence(expressions: readonly TS.Expression[]): { pre: TS.Statement[]; values: TS.Expression[] } {
    const { ts } = this;
    let lastCut = -1;
    expressions.forEach((x, i) => {
      if (this.hasCut(x)) lastCut = i;
    });
    const pre: TS.Statement[] = [];
    const values: TS.Expression[] = [];
    expressions.forEach((x, i) => {
      const spread = ts.isSpreadElement(x);
      const r = this.anfExpr(spread ? x.expression : x);
      pre.push(...r.pre);
      let value = r.value;
      if (i < lastCut && !isStable(ts, value)) value = this.hold(value, pre);
      values.push(spread ? ts.factory.createSpreadElement(value) : value);
    });
    return { pre, values };
  }

  /** Evaluate an expression now, into a temporary, and stand the temporary in for it. */
  private hold(value: TS.Expression, pre: TS.Statement[]): TS.Expression {
    if (isStable(this.ts, value)) return value;
    const t = this.syntheticVar("t", undefined);
    pre.push(assign(this.ts, t.name, value));
    return vRef(this.ts, t.name);
  }

  // --- calls: what each becomes -------------------------------------------------------------------------

  /**
   * A value a call reads, held in a TYPED variable of its own: typed by TypeScript where the checker
   * knows the type, and otherwise by the slot that receives it — an assertion the engine then enforces
   * when the value is handed on.
   */
  private typedValue(value: TS.Expression, fallback: JsonSchema | undefined, pre: TS.Statement[], base: string): string {
    let schema: JsonSchema | undefined;
    if (isTempRef(this.ts, value)) schema = this.hoistedByName.get((value as TS.PropertyAccessExpression).name.text)?.schema;
    else if (value.pos >= 0) {
      try {
        schema = typeToWireSchema(this.ts, this.checker, this.checker.getTypeAtLocation(value), "an argument").schema;
      } catch {
        schema = undefined;
      }
    }
    if (schema === undefined || Object.keys(wireForm(schema)).length === 0) schema = fallback;
    const held = this.syntheticVar(base, schema);
    pre.push(assign(this.ts, held.name, value));
    return held.name;
  }

  /** `{ x }`'s value: the VARIABLE `x` — which the name alone does not resolve to. */
  private shorthandValue(p: TS.ShorthandPropertyAssignment): TS.Expression {
    const symbol = this.checker.getShorthandAssignmentValueSymbol(p);
    const hoisted = symbol !== undefined ? this.hoisted.get(symbol) : undefined;
    return hoisted !== undefined ? vRef(this.ts, hoisted.name) : p.name;
  }

  /** A call as the marker that cuts at it, with its arguments lifted first (SCRIPTS.md §7). */
  private cut(call: TS.CallExpression): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    const callee = unwrap(ts, call.expression);

    const hook = ts.isIdentifier(callee) ? this.hookOf(callee) : undefined;
    const imported = ts.isIdentifier(callee) ? this.importedAs(callee) : undefined;
    const isPromiseAll = ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === "Promise" && callee.name.text === "all";
    if (hook === undefined && imported === undefined && !isPromiseAll) {
      const helper = this.helperOf(call.expression);
      if (helper !== undefined) return this.inline(call, helper);
    }
    if (hook === "parallel" || hook === "pipeline" || isPromiseAll) return this.fanOut(call, hook ?? "all");

    const args = this.anfSequence(call.arguments);
    const pre = [...args.pre];
    // A call made as code is handed its arguments as `_call.args` — `_call` is always an object.
    const array = f.createObjectLiteralExpression([f.createPropertyAssignment("args", f.createArrayLiteralExpression(args.values))]);
    const typed = call.typeArguments?.[0] !== undefined ? this.typeArgumentSchema(call) : hook === "llm" || hook === "agent" ? this.inferredOutput(call) : undefined;
    const withOutput = (schema: JsonSchema | undefined): string => (schema !== undefined ? `.withOutput(${JSON.stringify(schema)})` : "");

    let partial: Omit<CallSite, "id" | "target" | "node">;
    if (hook === "llm") {
      const plan = this.promptPlan(args.values, typed, call, pre);
      partial = plan ?? { kind: "script", label: this.labelOf(args.values, "llm"), call: array, script: `$rt.llm${withOutput(typed)}(...$v._call.args)`, resultSchema: typed ?? ({ type: "string" } as JsonSchema) };
    } else if (hook === "agent") {
      const schema = typed ?? this.optionSchema(args.values[1]);
      partial = { kind: "script", label: this.labelOf(args.values, "agent"), call: array, script: `$rt.agent${withOutput(typed)}(...$v._call.args)`, ...(schema !== undefined ? { resultSchema: schema } : {}) };
    } else if (hook === "now" || hook === "random") {
      partial = { kind: "script", label: `${hook}()`, call: array, script: `$rt.${hook}()`, resultSchema: { type: "number" } as JsonSchema };
    } else if (hook === "workflow") {
      const target = this.workflowTarget(call);
      partial = this.mountPlan(target.stateId, target, args.values[1], pre);
    } else if (imported?.kind === "state") {
      partial = this.mountPlan(imported.stateId!, imported, args.values[0], pre);
    } else if (imported?.kind === "registry") {
      partial = { kind: "script", label: imported.ref!, call: array, script: `$rt.call(${JSON.stringify(imported.ref)}, ...$v._call.args)` };
    } else if (imported?.kind === "operation") {
      partial = { kind: "script", label: (callee as TS.Identifier).text, call: array, script: `${(callee as TS.Identifier).text}(...$v._call.args)` };
    } else if (imported?.kind === "prompt") {
      partial = this.templatePlan(imported.template!, args.values[0], typed, (callee as TS.Identifier).text);
    } else {
      throw this.error(call, "this call cannot be cut");
    }
    const site = this.addSite(call, partial);
    return { pre: [...pre, this.marker(site)], value: vRef(ts, site.target) };
  }

  private addSite(node: TS.Node, partial: Omit<CallSite, "id" | "target" | "node">): CallSite {
    const target = this.syntheticVar("r", partial.resultSchema);
    const site: CallSite = { id: this.sites.length, target: target.name, node, ...partial };
    this.sites.push(site);
    return site;
  }

  private marker(site: CallSite): TS.Statement {
    const marker = this.ts.factory.createExpressionStatement(this.ts.factory.createIdentifier(`__cut_${site.id}`));
    this.markers.set(marker, site);
    return marker;
  }

  /**
   * An `llm()` call as the PROMPT OPERATION it is, when its options can be read at compile time:
   * literal options become literal fields of the operation, computed ones read the state's `_call`
   * input — so the generated state shows its model and its prompt as any authored one does. `undefined`
   * when the options are not an object the compiler can read, and the call runs as code instead.
   */
  private promptPlan(args: readonly TS.Expression[], typed: JsonSchema | undefined, call: TS.CallExpression, pre: TS.Statement[]): Omit<CallSite, "id" | "target" | "node"> | undefined {
    const { ts } = this;
    const f = ts.factory;
    const [a0, a1] = args;
    let prompt: TS.Expression | undefined;
    let options: TS.ObjectLiteralExpression | undefined;
    if (a0 !== undefined && ts.isObjectLiteralExpression(unwrap(ts, a0)) && a1 === undefined) options = unwrap(ts, a0) as TS.ObjectLiteralExpression;
    else if (a0 !== undefined && (a1 === undefined || ts.isObjectLiteralExpression(unwrap(ts, a1)))) {
      prompt = a0;
      options = a1 !== undefined ? (unwrap(ts, a1) as TS.ObjectLiteralExpression) : undefined;
    } else return undefined;

    const op: Record<string, unknown> = {};
    const inputVars: Record<string, string> = {};
    const held: TS.Statement[] = [];
    let output: Record<string, unknown> | undefined;
    let label: string | undefined;
    // A computed option is a typed variable the call state takes as an input, and the field reads it.
    const knob = (name: string, value: TS.Expression): void => {
      const literal = this.staticValue(value, true);
      if (literal !== undefined) {
        op[name] = literal;
        return;
      }
      const v = this.typedValue(value, CONFIG_FIELD_SCHEMAS[name] ?? (name === "system" ? ({ type: "string" } as JsonSchema) : undefined), held, name);
      inputVars[name] = v;
      op[name] = CONFIG_FIELD_SCHEMAS[name] !== undefined || name === "system" ? { $expr: `.inputs.${v}` } : { $binding: { $expr: `.inputs.${v}` } };
    };
    for (const member of options?.properties ?? []) {
      const valueOf = ts.isPropertyAssignment(member) ? member.initializer : ts.isShorthandPropertyAssignment(member) ? this.shorthandValue(member) : undefined;
      const nameNode = ts.isPropertyAssignment(member) || ts.isShorthandPropertyAssignment(member) ? member.name : undefined;
      const name = nameNode !== undefined && (ts.isIdentifier(nameNode) || ts.isStringLiteral(nameNode)) ? nameNode.text : undefined;
      if (valueOf === undefined || name === undefined) return undefined;
      if (name === "prompt") prompt = valueOf;
      else if (name === "output") {
        const value = this.staticValue(valueOf, true);
        if (value === null || typeof value !== "object" || Array.isArray(value)) throw this.error(valueOf, "an llm() call's `output` must be known when compiling — a literal, or a constant");
        output = value as Record<string, unknown>;
      } else if (name === "label") {
        const value = this.staticValue(valueOf, true);
        if (typeof value === "string") label = value;
      } else if (name === "phase") continue;
      else if (name === "failureValue") return undefined;
      else if (name === "config") {
        const config = unwrap(ts, valueOf);
        if (!ts.isObjectLiteralExpression(config)) return undefined;
        for (const k of config.properties) {
          if (!ts.isPropertyAssignment(k) || !(ts.isIdentifier(k.name) || ts.isStringLiteral(k.name))) return undefined;
          knob(k.name.text, k.initializer);
        }
      } else if (["tools", "session", "workspace", "permissions", "conversation"].includes(name)) {
        const value = this.staticValue(valueOf, true);
        if (value === undefined) return undefined;
        op[name] = value;
      } else knob(name, valueOf);
    }
    if (prompt === undefined) return undefined;
    pre.push(...held);
    const written = output?.schema as JsonSchema | undefined;
    if (typed !== undefined && written !== undefined) this.checkAgreement(typed, written, call);
    const schema = written ?? typed;
    // A plain string is text: the model says it, and nothing wraps it as `{ value }`.
    const text = (schema === undefined || JSON.stringify(schema) === '{"type":"string"}') && output?.kind !== "json";
    const object = !text && schema !== undefined && (schema as { type?: unknown }).type === "object";
    op.prompt = `{{.inputs.${CALL_INPUT}.prompt}}`;
    op.output = text ? { text: { kind: "text" } } : object || schema === undefined ? { value: { kind: "json", ...(schema !== undefined ? { schema } : {}) } } : { value: { schema } };
    return {
      kind: "prompt",
      label: label ?? this.labelOf([prompt], "llm"),
      call: f.createObjectLiteralExpression([f.createPropertyAssignment("prompt", prompt)]),
      op,
      inputVars,
      unwrap: !text && !object && schema !== undefined,
      resultSchema: (text ? { type: "string" } : (schema ?? {})) as JsonSchema,
    };
  }

  /** A prompt IMPORT's call: the template itself is the operation's prompt, its holes the call's inputs. */
  private templatePlan(template: string, inputs: TS.Expression | undefined, returned: JsonSchema | undefined, name: string): Omit<CallSite, "id" | "target" | "node"> {
    const { ts } = this;
    const typed = returned !== undefined && JSON.stringify(returned) === '{"type":"string"}' ? undefined : returned;
    const f = ts.factory;
    const literal = inputs !== undefined ? unwrap(ts, inputs) : undefined;
    const keys =
      literal !== undefined && ts.isObjectLiteralExpression(literal)
        ? literal.properties.flatMap((p) => ((ts.isPropertyAssignment(p) || ts.isShorthandPropertyAssignment(p)) && (ts.isIdentifier(p.name) || ts.isStringLiteral(p.name)) ? [p.name.text] : []))
        : [...template.matchAll(/\{\{\s*\.inputs\.([A-Za-z_$][\w$]*)/g)].map((m) => m[1]!);
    const input: Record<string, unknown> = {};
    for (const key of new Set(keys)) input[key] = { binding: { $expr: `.inputs.${CALL_INPUT}.${key}` } };
    const object = typed !== undefined && (typed as { type?: unknown }).type === "object";
    return {
      kind: "prompt",
      label: name,
      call: inputs ?? f.createObjectLiteralExpression([]),
      op: { prompt: template, input, output: typed === undefined ? { text: { kind: "text" } } : object ? { value: { kind: "json", schema: typed } } : { value: { schema: typed } } },
      unwrap: typed !== undefined && !object,
      resultSchema: typed ?? ({ type: "string" } as JsonSchema),
    };
  }

  /** A called state (SCRIPTS.md §10) — a mount, entered with the call's arguments by name. */
  private mountPlan(stateId: string, shape: Pick<ImportedName, "inputNames" | "argsInput" | "whole" | "inputSchemas">, args: TS.Expression | undefined, pre: TS.Statement[]): Omit<CallSite, "id" | "target" | "node"> {
    const { ts } = this;
    const f = ts.factory;
    const literal = args !== undefined ? unwrap(ts, args) : undefined;
    let inputNames = shape.inputNames;
    // Each argument is a typed variable of its own, which the rule hands the called state by name.
    const inputVars: Record<string, string> = {};
    if (shape.argsInput) {
      if (args !== undefined) inputVars.args = this.typedValue(args, shape.inputSchemas?.args, pre, "args");
    } else if (literal !== undefined && ts.isObjectLiteralExpression(literal)) {
      inputNames = [];
      for (const p of literal.properties) {
        if (!(ts.isPropertyAssignment(p) || ts.isShorthandPropertyAssignment(p)) || !(ts.isIdentifier(p.name) || ts.isStringLiteral(p.name))) {
          throw this.error(p, "a called state takes its arguments as `name: value` pairs");
        }
        const value = ts.isPropertyAssignment(p) ? p.initializer : this.shorthandValue(p);
        inputVars[p.name.text] = this.typedValue(value, shape.inputSchemas?.[p.name.text], pre, p.name.text);
        inputNames.push(p.name.text);
      }
    } else if (args !== undefined) {
      // A computed object: each input the state declares, read off it.
      const whole = this.typedValue(args, undefined, pre, "args");
      for (const name of inputNames ?? []) inputVars[name] = this.typedValue(f.createPropertyAccessExpression(vRef(ts, whole), name), shape.inputSchemas?.[name], pre, name);
    }
    return {
      kind: "mount",
      label: stateId,
      call: f.createObjectLiteralExpression([]),
      inputVars,
      stateId,
      inputNames: inputNames ?? [],
      ...(shape.argsInput ? { argsInput: true } : {}),
      ...(shape.whole !== undefined ? { whole: shape.whole } : {}),
    };
  }

  /**
   * A FAN-OUT (SCRIPTS.md §7): `parallel(xs.map(x => () => …))`, `Promise.all(xs.map(async x => …))`,
   * `pipeline(xs, …stages)`, or `parallel`/`Promise.all` over a list of calls. Each item runs as an
   * ELEMENT — a machine of its own, mounted with `each` — and the result is their results in order.
   * Claude's `parallel` and `pipeline` read a failed item as `null`; `Promise.all` fails with it.
   */
  private fanOut(call: TS.CallExpression, how: string): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    const tolerant = how === "parallel" || how === "pipeline";
    const pre: TS.Statement[] = [];
    let items: TS.Expression;
    let spec: ElementSpec;
    const id = this.elements.length;

    if (how === "pipeline") {
      const [source, ...stages] = call.arguments;
      if (source === undefined) throw this.error(call, "pipeline() takes the items first");
      const r = this.anfExpr(source);
      pre.push(...r.pre);
      items = r.value;
      // One item's chain: `v = item; v = await s1(v, item, index); …; return v`.
      const item = this.syntheticVar("item", undefined);
      const index = this.syntheticVar("index", undefined);
      const value = this.syntheticVar("value", undefined);
      const statements: TS.Statement[] = [assign(ts, value.name, vRef(ts, item.name))];
      for (const stage of stages) {
        statements.push(assign(ts, value.name, f.createAwaitExpression(f.createCallExpression(stage, undefined, [vRef(ts, value.name), vRef(ts, item.name), vRef(ts, index.name)]))));
      }
      statements.push(f.createReturnStatement(vRef(ts, value.name)));
      spec = { id, node: call, params: [{ name: item.name, from: "item" }, { name: index.name, from: "index" }], statements };
    } else {
      const source = call.arguments[0];
      if (source === undefined) throw this.error(call, `${how === "all" ? "Promise.all" : "parallel"}() needs a list`);
      const list = unwrap(ts, source);
      if (ts.isCallExpression(list) && ts.isPropertyAccessExpression(list.expression) && list.expression.name.text === "map") {
        const r = this.anfExpr(list.expression.expression);
        pre.push(...r.pre);
        items = r.value;
        spec = this.elementOf(id, list.arguments[0], how === "parallel", call);
      } else if (ts.isArrayLiteralExpression(list)) {
        // A list of calls: element `i` runs the `i`-th — `if (i === 0) return await e0; …`.
        items = f.createArrayLiteralExpression(list.elements.map((_, i) => f.createNumericLiteral(i)));
        const index = this.syntheticVar("i", undefined);
        const statements = list.elements.map((element, i) => {
          let body: TS.Expression = element;
          const u = unwrap(ts, element);
          if (how === "parallel" && (ts.isArrowFunction(u) || ts.isFunctionExpression(u))) body = f.createCallExpression(u, undefined, []);
          return f.createIfStatement(f.createBinaryExpression(vRef(ts, index.name), ts.SyntaxKind.EqualsEqualsEqualsToken, f.createNumericLiteral(i)), f.createReturnStatement(f.createAwaitExpression(body))) as TS.Statement;
        });
        spec = { id, node: call, params: [{ name: index.name, from: "item" }], statements };
      } else {
        throw this.error(call, `${how === "all" ? "Promise.all" : "parallel"}() over a list made elsewhere cannot be compiled — map over the items where they are awaited, or list the calls`);
      }
    }
    this.elements.push(spec);
    const site = this.addSite(call, {
      kind: "map",
      label: how === "all" ? "Promise.all" : how,
      call: f.createObjectLiteralExpression([f.createPropertyAssignment("items", f.createArrayLiteralExpression([f.createSpreadElement(items)]))]),
      element: id,
      ...(tolerant ? { failureValue: true } : {}),
      resultSchema: { type: "array" } as JsonSchema,
    });
    spec.site = site.id;
    return { pre: [...pre, this.marker(site)], value: vRef(ts, site.target) };
  }

  /** A `.map` callback as an element: its parameters, and its body — through Claude's thunk, for `parallel`. */
  private elementOf(id: number, callback: TS.Expression | undefined, thunked: boolean, at: TS.Node): ElementSpec {
    const { ts } = this;
    const f = ts.factory;
    if (callback === undefined) throw this.error(at, "map() needs a callback");
    const fn = unwrap(ts, callback);
    if (ts.isIdentifier(fn)) {
      // `xs.map(verify)` — the element calls the helper with the item and its index.
      const item = this.syntheticVar("item", undefined);
      const index = this.syntheticVar("index", undefined);
      let body: TS.Expression = f.createCallExpression(fn, undefined, [vRef(ts, item.name), vRef(ts, index.name)]);
      if (thunked) body = f.createCallExpression(body, undefined, []);
      return { id, node: callback, params: [{ name: item.name, from: "item" }, { name: index.name, from: "index" }], statements: [f.createReturnStatement(f.createAwaitExpression(body))] };
    }
    if (!ts.isArrowFunction(fn) && !ts.isFunctionExpression(fn)) throw this.error(callback, "a fan-out's callback must be a function written in place, or a helper's name");
    let body: TS.ConciseBody = fn.body;
    if (thunked) {
      // Claude's `x => () => agent(…)`: the element is the thunk's body.
      const inner = ts.isBlock(body) ? undefined : unwrap(ts, body);
      if (inner !== undefined && (ts.isArrowFunction(inner) || ts.isFunctionExpression(inner))) body = inner.body;
    }
    const params: ElementSpec["params"] = [];
    fn.parameters.forEach((p, i) => {
      if (!ts.isIdentifier(p.name) || i > 1) throw this.error(p, "a fan-out's callback takes the item and, optionally, its index");
      const symbol = this.checker.getSymbolAtLocation(p.name);
      if (symbol === undefined) throw this.error(p, "has no symbol");
      params.push({ name: this.hoist(symbol, p.name).name, from: i === 0 ? "item" : "index" });
    });
    this.checkElementWrites(fn);
    const statements = ts.isBlock(body) ? [...body.statements] : [f.createReturnStatement(body)];
    return { id, node: callback, params, statements };
  }

  /** An element runs apart from its siblings, so it may READ what lives outside it, and never write it. */
  private checkElementWrites(fn: TS.Node): void {
    const { ts } = this;
    const outside = (id: TS.Identifier): boolean => {
      const symbol = this.checker.getSymbolAtLocation(id);
      const declaration = symbol?.valueDeclaration;
      return declaration !== undefined && declaration.getSourceFile() === this.sf && !(declaration.pos >= fn.pos && declaration.end <= fn.end) && ts.isVariableDeclaration(declaration);
    };
    const visit = (n: TS.Node): void => {
      const target =
        ts.isBinaryExpression(n) && n.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && n.operatorToken.kind <= ts.SyntaxKind.LastAssignment
          ? n.left
          : (ts.isPrefixUnaryExpression(n) || ts.isPostfixUnaryExpression(n)) && (n.operator === ts.SyntaxKind.PlusPlusToken || n.operator === ts.SyntaxKind.MinusMinusToken)
            ? n.operand
            : undefined;
      if (target !== undefined && ts.isIdentifier(target) && outside(target)) {
        throw this.error(n, `a fan-out's element writes '${target.text}', which lives outside it — each item runs apart from the others, so return the value instead`);
      }
      ts.forEachChild(n, visit);
    };
    visit(fn);
  }

  /**
   * A helper that makes a call, INLINED where it is called: its parameters bound to the arguments, its
   * `return` a jump out of a labeled block (SCRIPTS.md §7). Its `T` is the caller's.
   */
  private inline(call: TS.CallExpression, fn: TS.FunctionLikeDeclaration): { pre: TS.Statement[]; value: TS.Expression } {
    const { ts } = this;
    const f = ts.factory;
    if (this.inlining.includes(fn)) throw this.error(call, "a helper that makes a call calls itself — recursion cannot be compiled to states; write it as a loop");
    const args = this.anfSequence(call.arguments);
    const pre = [...args.pre];
    const ret = this.syntheticVar("ret", undefined);
    const label = `inline_${this.synthetic++}`;
    const statements: TS.Statement[] = [];
    fn.parameters.forEach((p, i) => {
      if (p.dotDotDotToken !== undefined) throw this.error(p, "a helper that makes a call cannot take rest parameters");
      let value: TS.Expression = args.values[i] ?? f.createIdentifier("undefined");
      if (p.initializer !== undefined) {
        const held = this.hold(value, statements);
        value = f.createConditionalExpression(f.createBinaryExpression(held, ts.SyntaxKind.EqualsEqualsEqualsToken, f.createIdentifier("undefined")), undefined, p.initializer, undefined, held);
      }
      statements.push(f.createVariableStatement(undefined, f.createVariableDeclarationList([f.createVariableDeclaration(p.name, undefined, undefined, value)], ts.NodeFlags.Let)));
    });
    const body = fn.body!;
    if (ts.isBlock(body)) for (const s of body.statements) statements.push(replaceReturns(ts, s, ret.name, label));
    else statements.push(assign(ts, ret.name, body as TS.Expression));
    const substitution = this.typeSubstitution(call, fn);
    this.inlining.push(fn);
    this.substitutions.push(substitution);
    try {
      pre.push(...this.anfStatement(f.createLabeledStatement(label, f.createBlock(statements, true))));
    } finally {
      this.inlining.pop();
      this.substitutions.pop();
    }
    return { pre, value: vRef(ts, ret.name) };
  }

  /** The type arguments a call gives a generic helper — written, or read back off what it returns. */
  private typeSubstitution(call: TS.CallExpression, fn: TS.FunctionLikeDeclaration): Map<TS.Symbol, TS.Type> {
    const { ts } = this;
    const out = new Map<TS.Symbol, TS.Type>();
    (fn.typeParameters ?? []).forEach((tp, i) => {
      const symbol = this.checker.getSymbolAtLocation(tp.name);
      if (symbol === undefined) return;
      const written = call.typeArguments?.[i];
      if (written !== undefined) {
        out.set(symbol, this.resolveType(this.checker.getTypeFromTypeNode(written)));
        return;
      }
      // Inferred: when the helper returns `T` or `Promise<T>`, the call's own type says what `T` is.
      let declared = fn.type;
      if (declared !== undefined && ts.isTypeReferenceNode(declared) && declared.typeName.getText(this.sf) === "Promise") declared = declared.typeArguments?.[0];
      if (declared !== undefined && ts.isTypeReferenceNode(declared) && this.checker.getSymbolAtLocation(declared.typeName) === symbol) {
        out.set(symbol, awaited(ts, this.checker, this.checker.getTypeAtLocation(call)));
      }
    });
    return out;
  }

  /** A type with the helper being inlined's type parameters replaced by the caller's. */
  private resolveType(type: TS.Type): TS.Type {
    if ((type.flags & this.ts.TypeFlags.TypeParameter) !== 0) {
      const symbol = type.getSymbol();
      for (let i = this.substitutions.length - 1; i >= 0; i--) {
        const found = symbol !== undefined ? this.substitutions[i]!.get(symbol) : undefined;
        if (found !== undefined) return found;
      }
    }
    return type;
  }

  /**
   * The output contract `llm<T>` names — `T` converted by the one TS → wire path (SCRIPTS.md §9), each
   * named type a `$defs` entry when `T` is an object, and `T` resolved through the helper being
   * inlined when it is that helper's own type parameter.
   */
  typeArgumentSchema(call: TS.CallExpression, node: TS.TypeNode = call.typeArguments![0]!): JsonSchema {
    const type = this.resolveType(this.checker.getTypeFromTypeNode(node));
    try {
      const where = `the type argument '${node.getText(this.sf)}'`;
      const withDefs = typeToWireSchema(this.ts, this.checker, type, where, { defs: true });
      const result = (withDefs.schema as { type?: unknown }).type === "object" ? withDefs : typeToWireSchema(this.ts, this.checker, type, where);
      for (const w of result.warnings) this.warnings.push(`${this.where(node)}: ${w}`);
      return result.schema;
    } catch (e) {
      if (e instanceof WireTypeError) throw this.error(node, e.message);
      throw e;
    }
  }

  /**
   * The `T` TypeScript INFERRED for an `llm()`/`agent()` written without one — from what the call's
   * result is assigned to, or the other side of a `||` (SCRIPTS.md §9). The script's own types say
   * that is what the call returns, so it is the contract the model is held to. Text (`string`), or a
   * type nothing constrains, asks for text as an unwritten `T` does.
   */
  inferredOutput(call: TS.CallExpression): JsonSchema | undefined {
    const { ts } = this;
    if (call.pos < 0) return undefined;
    const signature = this.checker.getResolvedSignature(call);
    if (signature === undefined) return undefined;
    let type = awaited(ts, this.checker, signature.getReturnType());
    if (type.isUnion()) {
      const present = type.types.filter((t) => (t.flags & (ts.TypeFlags.Null | ts.TypeFlags.Undefined)) === 0);
      if (present.length === 1) type = present[0]!;
    }
    if ((type.flags & (ts.TypeFlags.String | ts.TypeFlags.StringLiteral | ts.TypeFlags.Any | ts.TypeFlags.Unknown | ts.TypeFlags.TypeParameter)) !== 0) return undefined;
    try {
      return this.schemaOfType(type, call);
    } catch {
      return undefined;
    }
  }

  private checkAgreement(typed: JsonSchema, written: JsonSchema, at: TS.Node): void {
    // Structural equality is the cheap check at compile time; the hook repeats `isSubschema` both ways
    // when the call is made, where it is exact.
    if (JSON.stringify(stripDefs(typed)) !== JSON.stringify(stripDefs(written))) {
      this.warnings.push(`${this.where(at)}: the type argument and output.schema are written differently — they must describe the same values`);
    }
  }

  /** `agent(p, { schema })` — the contract, when the compiler can read it. */
  private optionSchema(options: TS.Expression | undefined): JsonSchema | undefined {
    if (options === undefined) return undefined;
    const value = this.staticValue(options, true);
    return value !== null && typeof value === "object" && !Array.isArray(value) ? ((value as { schema?: JsonSchema }).schema ?? undefined) : undefined;
  }

  /** What a call state is called on the board: its `label`, or the start of its prompt. */
  private labelOf(args: readonly TS.Expression[], hook: string): string {
    const { ts } = this;
    const first = args[0] !== undefined ? unwrap(ts, args[0]) : undefined;
    const second = args[1] !== undefined ? unwrap(ts, args[1]) : undefined;
    const options = first !== undefined && ts.isObjectLiteralExpression(first) ? first : second !== undefined && ts.isObjectLiteralExpression(second) ? second : undefined;
    const labelled = options?.properties.find((p): p is TS.PropertyAssignment => ts.isPropertyAssignment(p) && ts.isIdentifier(p.name) && p.name.text === "label");
    const label = labelled !== undefined ? this.staticValue(labelled.initializer, true) : undefined;
    if (typeof label === "string") return label;
    const promptNode = first !== undefined && ts.isObjectLiteralExpression(first) ? first.properties.find((p): p is TS.PropertyAssignment => ts.isPropertyAssignment(p) && ts.isIdentifier(p.name) && p.name.text === "prompt")?.initializer : first;
    const text = promptNode !== undefined ? promptText(ts, promptNode) : undefined;
    return text !== undefined ? `${hook}: ${text.length > 48 ? `${text.slice(0, 47)}…` : text}` : hook;
  }

  // --- generic helpers in the coarse modes -----------------------------------------------------------

  /**
   * In `"phases"`/`"state"` mode a helper is not inlined, so a generic helper's `llm<T>` cannot know
   * its `T` when compiling. It is handed the schema instead: the helper gains a trailing parameter per
   * type parameter its calls use, and every call of it passes the caller's `T` converted (SCRIPTS.md §9).
   */
  private findSchemaParams(statements: readonly TS.Statement[]): void {
    const { ts } = this;
    const visit = (n: TS.Node): void => {
      if (ts.isCallExpression(n) && n.typeArguments?.[0] !== undefined && ts.isIdentifier(n.expression)) {
        const hook = this.hookOf(n.expression);
        const arg = n.typeArguments[0];
        if ((hook === "llm" || hook === "agent") && ts.isTypeReferenceNode(arg)) {
          const symbol = this.checker.getSymbolAtLocation(arg.typeName);
          const declaration = symbol?.declarations?.[0];
          if (declaration !== undefined && ts.isTypeParameterDeclaration(declaration) && isFunctionBoundary(ts, declaration.parent)) {
            const owner = declaration.parent;
            const names = this.schemaParams.get(owner) ?? [];
            if (!names.includes(declaration.name.text)) names.push(declaration.name.text);
            this.schemaParams.set(owner, names);
          }
        }
      }
      ts.forEachChild(n, visit);
    };
    for (const s of statements) visit(s);
  }

  // --- hoisting ------------------------------------------------------------------------------------

  /**
   * The pre-pass: every declaration at a HOISTED position — directly in the body, or directly in a
   * structure that contains a cut — becomes a `$v` variable, unless it is a helper or a constant.
   * Mirrors {@link lower}'s walk exactly, since the two must agree about which positions are cut.
   */
  private collectHoisting(s: TS.Statement): void {
    const { ts } = this;
    if (ts.isVariableStatement(s)) return this.collectVariables(s.declarationList, hasModifier(ts, s, ts.SyntaxKind.ExportKeyword));
    if (!this.containsCut(s)) return;
    if (ts.isBlock(s)) return s.statements.forEach((x) => this.collectHoisting(x));
    if (ts.isLabeledStatement(s)) return this.collectHoisting(s.statement);
    if (ts.isIfStatement(s)) {
      this.collectHoisting(s.thenStatement);
      if (s.elseStatement !== undefined) this.collectHoisting(s.elseStatement);
      return;
    }
    if (ts.isWhileStatement(s) || ts.isDoStatement(s)) return this.collectHoisting(s.statement);
    if (ts.isForStatement(s)) {
      if (s.initializer !== undefined && ts.isVariableDeclarationList(s.initializer)) this.collectVariables(s.initializer, false);
      return this.collectHoisting(s.statement);
    }
    if (ts.isForOfStatement(s)) {
      if (ts.isVariableDeclarationList(s.initializer)) {
        for (const d of s.initializer.declarations) for (const id of bindingIdentifiers(ts, d.name)) this.hoistIdentifier(id);
      }
      return this.collectHoisting(s.statement);
    }
    if (ts.isSwitchStatement(s)) {
      for (const clause of s.caseBlock.clauses) clause.statements.forEach((x) => this.collectHoisting(x));
      return;
    }
    if (ts.isTryStatement(s)) {
      s.tryBlock.statements.forEach((x) => this.collectHoisting(x));
      const binding = s.catchClause?.variableDeclaration?.name;
      if (binding !== undefined && ts.isIdentifier(binding)) this.hoistIdentifier(binding);
      s.catchClause?.block.statements.forEach((x) => this.collectHoisting(x));
    }
  }

  private collectVariables(list: TS.VariableDeclarationList, exported: boolean): void {
    const { ts } = this;
    const isConst = (list.flags & ts.NodeFlags.Const) !== 0;
    for (const d of list.declarations) {
      if (isConst && !exported && ts.isIdentifier(d.name) && d.initializer !== undefined && this.isStatic(d.initializer)) {
        this.helperDeclarations.add(d);
        continue;
      }
      for (const id of bindingIdentifiers(ts, d.name)) this.hoistIdentifier(id);
    }
  }

  private hoistIdentifier(identifier: TS.Identifier): void {
    const symbol = this.checker.getSymbolAtLocation(identifier);
    if (symbol === undefined) throw this.error(identifier, `'${identifier.text}' has no symbol the compiler can hoist`);
    this.hoist(symbol, identifier);
  }

  /** An initializer that is the same in every segment — so it is re-evaluated, never carried. */
  private isStatic(node: TS.Expression): boolean {
    const { ts } = this;
    const e = unwrap(ts, node);
    if (ts.isArrowFunction(e) || ts.isFunctionExpression(e) || ts.isClassExpression(e)) return !this.hasCut(e, true);
    if (ts.isStringLiteral(e) || ts.isNumericLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return true;
    if (e.kind === ts.SyntaxKind.TrueKeyword || e.kind === ts.SyntaxKind.FalseKeyword || e.kind === ts.SyntaxKind.NullKeyword) return true;
    if (ts.isPrefixUnaryExpression(e)) return this.isStatic(e.operand);
    if (ts.isTemplateExpression(e)) return e.templateSpans.every((span) => this.isStatic(span.expression));
    if (ts.isArrayLiteralExpression(e)) return !this.isMutated(node) && e.elements.every((x) => (ts.isSpreadElement(x) ? this.isStatic(x.expression) : ts.isOmittedExpression(x) || this.isStatic(x)));
    if (ts.isObjectLiteralExpression(e)) {
      if (this.isMutated(node)) return false;
      return e.properties.every((p) =>
        ts.isPropertyAssignment(p) ? !ts.isComputedPropertyName(p.name) && this.isStatic(p.initializer) : ts.isSpreadAssignment(p) ? this.isStatic(p.expression) : ts.isMethodDeclaration(p),
      );
    }
    // The compiler's own `$v.…` — a call's result, a temporary — is a value of this run, never static.
    if (isTempRef(ts, e)) return false;
    if (ts.isIdentifier(e)) {
      if (e.pos < 0) return false;
      const symbol = this.checker.getSymbolAtLocation(e);
      // A hoisted variable changes; anything else an identifier can name here (an import, a helper,
      // a module-level declaration, a global) is the same in every segment.
      return symbol === undefined || !this.hoisted.has(symbol);
    }
    if (ts.isPropertyAccessExpression(e)) return this.isStatic(e.expression);
    return false;
  }

  /**
   * Whether the `const` an initializer belongs to is ever CHANGED — a method called on it, a member
   * assigned, or handed to code that might: a constant object re-declared in every state must be one
   * nobody writes, or each state would start from a fresh copy.
   */
  private isMutated(initializer: TS.Expression): boolean {
    const { ts } = this;
    const declaration = initializer.parent;
    if (declaration === undefined || !ts.isVariableDeclaration(declaration) || !ts.isIdentifier(declaration.name)) return true;
    const symbol = this.checker.getSymbolAtLocation(declaration.name);
    if (symbol === undefined) return true;
    let mutated = false;
    const visit = (n: TS.Node): void => {
      if (mutated) return;
      if (ts.isIdentifier(n) && n !== declaration.name && this.checker.getSymbolAtLocation(n) === symbol) {
        let at: TS.Node = n;
        while (ts.isPropertyAccessExpression(at.parent) || ts.isElementAccessExpression(at.parent) || ts.isParenthesizedExpression(at.parent)) {
          const parent: TS.Node = at.parent;
          // `x.method(…)` — a call on it may change it.
          if (ts.isPropertyAccessExpression(parent) && ts.isCallExpression(parent.parent) && parent.parent.expression === parent && at === n) {
            mutated = true;
            return;
          }
          at = parent;
        }
        const parent = at.parent;
        if (ts.isBinaryExpression(parent) && parent.left === at && parent.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && parent.operatorToken.kind <= ts.SyntaxKind.LastAssignment) mutated = true;
        else if ((ts.isPrefixUnaryExpression(parent) || ts.isPostfixUnaryExpression(parent)) && at !== n) mutated = true;
        else if (ts.isDeleteExpression(parent)) mutated = true;
        else if (ts.isCallExpression(parent) && parent.arguments.includes(at as TS.Expression)) {
          // Handed to a function: only a hook, which reads it, keeps it constant.
          const callee = parent.expression;
          if (!(ts.isIdentifier(callee) && this.hookOf(callee) !== undefined) && !(ts.isPropertyAccessExpression(callee) && ts.isIdentifier(callee.expression) && callee.expression.text === "JSON")) mutated = true;
        }
      }
      ts.forEachChild(n, visit);
    };
    visit(this.sf);
    return mutated;
  }

  // --- lowering ------------------------------------------------------------------------------------

  private newBlock(line: number): Block {
    const block: Block = { id: this.blocks.length, machine: this.machine, code: [], jumps: [], uses: new Set(), defs: new Set(), line };
    const handler = this.handlers[this.handlers.length - 1];
    if (handler !== undefined) block.handler = handler;
    this.blocks.push(block);
    return block;
  }

  private end(term: Terminator): void {
    if (this.cur.term !== undefined) return;
    this.cur.term = term;
    const reads = new Set<string>();
    if (term.kind === "branch") this.collectRefs(term.cond, reads);
    if (term.kind === "call") {
      const site = this.sites[term.site]!;
      this.collectRefs(site.call, reads);
      for (const v of Object.values(site.inputVars ?? {})) reads.add(v);
    }
    if (term.kind === "return" && term.value !== undefined) this.collectRefs(term.value, reads);
    if (term.kind === "return" && term.fallthrough) {
      for (const name of this.exportNames) reads.add(name);
      if (this.defaultExport !== undefined) reads.add(this.defaultExport);
    }
    this.touch(reads, []);
  }

  /** Record, in order, what the current block reads and then assigns. */
  private touch(reads: Iterable<string>, writes: Iterable<string>): void {
    for (const r of reads) if (!CONTROL_NAMES.has(r) && !this.cur.defs.has(r)) this.cur.uses.add(r);
    for (const w of writes) if (!CONTROL_NAMES.has(w)) this.cur.defs.add(w);
  }

  private lowerList(statements: readonly TS.Statement[]): void {
    for (const s of statements) this.lower(s);
  }

  private lower(s: TS.Statement): void {
    const { ts } = this;
    const f = ts.factory;
    const line = this.lineOf(s);
    if (ts.isEmptyStatement(s)) return;
    if (isTypeOnly(ts, s)) return; // types have no run-time effect
    const site = this.markers.get(s);
    if (site !== undefined) {
      // A CALL: the state making it ends here, and the next state starts with its result.
      const handler = this.handlers[this.handlers.length - 1];
      if (handler !== undefined) site.handler = handler;
      const next = this.newBlock(line);
      this.end({ kind: "call", site: site.id, to: next.id });
      this.cur = next;
      this.emit(assign(ts, site.target, vRef(ts, RESULT_INPUT)), s, [site.target], this.lineOf(site.node));
      return;
    }
    if (ts.isFunctionDeclaration(s) || ts.isClassDeclaration(s)) {
      this.addHelper(s);
      return;
    }
    if (ts.isExportAssignment(s)) {
      // `export default <value>` in a body: the default output.
      const v = this.syntheticVar("default", undefined);
      this.defaultExport = v.name;
      this.emit(assign(ts, v.name, s.expression), s, [v.name]);
      return;
    }
    if (ts.isVariableStatement(s)) {
      this.lowerVariables(s.declarationList, s, hasModifier(ts, s, ts.SyntaxKind.ExportKeyword));
      return;
    }
    const phase = this.phaseStatement(s);
    if (phase !== undefined && this.mode !== "state") {
      const next = this.newBlock(line);
      this.end({ kind: "phase", phase, to: next.id });
      this.cur = next;
      return;
    }
    if (!this.containsCut(s)) {
      this.verbatim(s);
      return;
    }
    const labels = this.pendingLabels;
    this.pendingLabels = [];
    if (ts.isBlock(s)) return this.lowerList(s.statements);
    if (ts.isLabeledStatement(s)) {
      if (ts.isBlock(s.statement)) {
        // A labeled BLOCK — how an inlined helper returns: `break <label>` leaves it.
        const exit = this.newBlock(line);
        this.loops.push({ breakTo: exit.id, continueTo: -1, labels: [...labels, s.label.text] });
        this.lowerList(s.statement.statements);
        this.loops.pop();
        this.end({ kind: "goto", to: exit.id });
        this.cur = exit;
        return;
      }
      this.pendingLabels = [...labels, s.label.text];
      return this.lower(s.statement);
    }
    if (ts.isIfStatement(s)) {
      const then = this.newBlock(line);
      const otherwise = s.elseStatement !== undefined ? this.newBlock(line) : undefined;
      const join = this.newBlock(line);
      this.end({ kind: "branch", cond: s.expression, then: then.id, else: (otherwise ?? join).id });
      this.cur = then;
      this.lower(s.thenStatement);
      this.end({ kind: "goto", to: join.id });
      if (otherwise !== undefined) {
        this.cur = otherwise;
        this.lower(s.elseStatement!);
        this.end({ kind: "goto", to: join.id });
      }
      this.cur = join;
      return;
    }
    if (ts.isWhileStatement(s)) {
      const head = this.newBlock(line);
      const body = this.newBlock(line);
      const exit = this.newBlock(line);
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      this.end({ kind: "branch", cond: s.expression, then: body.id, else: exit.id });
      this.loop({ breakTo: exit.id, continueTo: head.id, labels }, body, s.statement, head.id);
      this.cur = exit;
      return;
    }
    if (ts.isDoStatement(s)) {
      const body = this.newBlock(line);
      const cond = this.newBlock(line);
      const exit = this.newBlock(line);
      this.end({ kind: "goto", to: body.id });
      this.loop({ breakTo: exit.id, continueTo: cond.id, labels }, body, s.statement, cond.id);
      this.cur = cond;
      this.end({ kind: "branch", cond: s.expression, then: body.id, else: exit.id });
      this.cur = exit;
      return;
    }
    if (ts.isForStatement(s)) {
      if (s.initializer !== undefined) {
        if (ts.isVariableDeclarationList(s.initializer)) this.lowerVariables(s.initializer, s, false);
        else this.verbatim(f.createExpressionStatement(s.initializer), s);
      }
      const head = this.newBlock(line);
      const body = this.newBlock(line);
      const cont = this.newBlock(line);
      const exit = this.newBlock(line);
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      if (s.condition !== undefined) this.end({ kind: "branch", cond: s.condition, then: body.id, else: exit.id });
      else this.end({ kind: "goto", to: body.id });
      this.loop({ breakTo: exit.id, continueTo: cont.id, labels }, body, s.statement, cont.id);
      this.cur = cont;
      if (s.incrementor !== undefined) this.verbatim(f.createExpressionStatement(s.incrementor), s);
      this.end({ kind: "goto", to: head.id });
      this.cur = exit;
      return;
    }
    if (ts.isForOfStatement(s)) {
      if (s.awaitModifier !== undefined) throw this.error(s, "a `for await` loop that is cut cannot be compiled — iterate an array");
      const iterableType = this.checker.getTypeAtLocation(s.expression.pos >= 0 ? s.expression : s);
      let schema: JsonSchema | undefined;
      try {
        schema = typeToWireSchema(ts, this.checker, iterableType, "the iterated value").schema;
      } catch (e) {
        if (!(e instanceof WireTypeError)) throw e;
        throw this.error(s.expression, `a loop that is cut keeps what it iterates across its states, so it must have a wire form: ${e.message}`);
      }
      const items = this.syntheticVar("items", Object.keys(schema).length > 0 ? withoutUnions(schema as JsonValue) as JsonSchema : undefined);
      const index = this.syntheticVar("index", { type: "integer" } as JsonSchema);
      this.emit(assign(ts, items.name, f.createArrayLiteralExpression([f.createSpreadElement(s.expression)])), s, [items.name]);
      this.emit(assign(ts, index.name, f.createNumericLiteral(0)), s, [index.name]);
      const head = this.newBlock(line);
      const body = this.newBlock(line);
      const cont = this.newBlock(line);
      const exit = this.newBlock(line);
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      const cond = f.createBinaryExpression(vRef(ts, index.name), ts.SyntaxKind.LessThanToken, f.createPropertyAccessExpression(vRef(ts, items.name), "length"));
      this.end({ kind: "branch", cond, then: body.id, else: exit.id });
      this.cur = body;
      const element = f.createElementAccessExpression(vRef(ts, items.name), vRef(ts, index.name));
      if (ts.isVariableDeclarationList(s.initializer)) this.assignDeclaration(s.initializer.declarations[0]!.name, element, s);
      else this.verbatim(f.createExpressionStatement(f.createAssignment(s.initializer as TS.Expression, element)), s);
      this.loop({ breakTo: exit.id, continueTo: cont.id, labels }, undefined, s.statement, cont.id);
      this.cur = cont;
      this.emit(f.createExpressionStatement(f.createPostfixIncrement(vRef(ts, index.name))), s, [index.name]);
      this.end({ kind: "goto", to: head.id });
      this.cur = exit;
      return;
    }
    if (ts.isSwitchStatement(s)) {
      // A `switch` that is cut: its value held, a chain of tests, and its clauses in order — falling
      // through as JavaScript's do.
      const held = this.syntheticVar("switch", undefined);
      this.emit(assign(ts, held.name, s.expression), s, [held.name]);
      const exit = this.newBlock(line);
      const clauseBlocks = s.caseBlock.clauses.map((c) => this.newBlock(this.lineOf(c)));
      const defaultIndex = s.caseBlock.clauses.findIndex((c) => ts.isDefaultClause(c));
      s.caseBlock.clauses.forEach((c, i) => {
        if (!ts.isCaseClause(c)) return;
        const next = this.newBlock(line);
        this.end({ kind: "branch", cond: f.createBinaryExpression(vRef(ts, held.name), ts.SyntaxKind.EqualsEqualsEqualsToken, f.createParenthesizedExpression(c.expression)), then: clauseBlocks[i]!.id, else: next.id });
        this.cur = next;
      });
      this.end({ kind: "goto", to: defaultIndex >= 0 ? clauseBlocks[defaultIndex]!.id : exit.id });
      this.loops.push({ breakTo: exit.id, continueTo: -1, labels, });
      (this.loops[this.loops.length - 1] as LoweredLoop & { isSwitch?: boolean }).isSwitch = true;
      s.caseBlock.clauses.forEach((c, i) => {
        this.cur = clauseBlocks[i]!;
        this.lowerList(c.statements);
        this.end({ kind: "goto", to: clauseBlocks[i + 1]?.id ?? exit.id });
      });
      this.loops.pop();
      this.cur = exit;
      return;
    }
    if (ts.isTryStatement(s)) {
      if (s.finallyBlock !== undefined) throw this.error(s.finallyBlock, "a `finally` around a call or a phase() cannot be compiled — put what it does after the `try`, in both the path that succeeds and the `catch`");
      const clause = s.catchClause!;
      const binding = clause.variableDeclaration?.name;
      const param = binding !== undefined && ts.isIdentifier(binding) ? this.hoisted.get(this.checker.getSymbolAtLocation(binding)!)?.name : undefined;
      // The catch block is created OUTSIDE the handler: a throw in it goes to the enclosing `try`.
      const catchBlock = this.newBlock(this.lineOf(clause));
      const join = this.newBlock(line);
      const handler: Handler = { id: this.handlerCount++, block: catchBlock.id, ...(param !== undefined ? { param } : {}) };
      this.handlers.push(handler);
      const start = this.newBlock(line);
      this.end({ kind: "goto", to: start.id });
      this.cur = start;
      this.lowerList(s.tryBlock.statements);
      this.end({ kind: "goto", to: join.id });
      this.handlers.pop();
      this.cur = catchBlock;
      this.lowerList(clause.block.statements);
      this.end({ kind: "goto", to: join.id });
      this.cur = join;
      return;
    }
    if (ts.isForInStatement(s)) throw this.error(s, "a `for…in` loop that is cut cannot be compiled — iterate `Object.keys(…)` with `for…of`");
    throw this.error(s, "a phase() or a call cannot be cut here — make it a statement of its own, directly in the script's control flow");
  }

  /** Lower a loop body under its jump targets, ending in a jump to `after`. */
  private loop(loop: LoweredLoop, body: Block | undefined, statement: TS.Statement, after: number): void {
    if (body !== undefined) this.cur = body;
    this.loops.push(loop);
    this.lower(statement);
    this.loops.pop();
    this.end({ kind: "goto", to: after });
  }

  private lowerVariables(list: TS.VariableDeclarationList, at: TS.Node, exported: boolean): void {
    const { ts } = this;
    for (const d of list.declarations) {
      const original = (d as { original?: TS.Node }).original as TS.VariableDeclaration | undefined;
      if (this.helperDeclarations.has(d) || (original !== undefined && this.helperDeclarations.has(original))) {
        // A helper or a constant: re-declared in every segment rather than carried across cuts.
        this.addHelper(ts.factory.createVariableStatement(undefined, ts.factory.createVariableDeclarationList([d], ts.NodeFlags.Const)));
        continue;
      }
      if (exported) for (const name of boundNames(ts, d.name)) this.exportNames.push(this.hoisted.get(this.checker.getSymbolAtLocation(bindingIdentifiers(ts, d.name).find((i) => i.text === name)!)!)?.name ?? name);
      this.assignDeclaration(d.name, d.initializer ?? ts.factory.createIdentifier("undefined"), at);
    }
  }

  /** `const <pattern> = value` at a hoisted position → an assignment into `$v`. */
  private assignDeclaration(name: TS.BindingName, value: TS.Expression, at: TS.Node): void {
    const { ts } = this;
    const f = ts.factory;
    for (const identifier of bindingIdentifiers(ts, name)) this.hoistIdentifier(identifier);
    const target = this.assignmentTarget(name);
    const written = bindingIdentifiers(ts, name).map((i) => this.hoisted.get(this.checker.getSymbolAtLocation(i)!)!.name);
    this.emit(f.createExpressionStatement(f.createAssignment(target, value)), at, written);
  }

  private assignmentTarget(name: TS.BindingName): TS.Expression {
    const { ts } = this;
    const f = ts.factory;
    if (ts.isIdentifier(name)) return vRef(ts, this.hoisted.get(this.checker.getSymbolAtLocation(name)!)!.name);
    if (ts.isObjectBindingPattern(name)) {
      return f.createObjectLiteralExpression(
        name.elements.map((e) => {
          const target = this.assignmentTarget(e.name);
          if (e.dotDotDotToken !== undefined) return f.createSpreadAssignment(target);
          const key = e.propertyName ?? (ts.isIdentifier(e.name) ? e.name : undefined);
          if (key === undefined) throw this.error(e, "a nested pattern needs a property name");
          const withDefault = e.initializer !== undefined ? f.createAssignment(target, e.initializer) : target;
          return f.createPropertyAssignment(key as TS.PropertyName, withDefault);
        }),
      );
    }
    return f.createArrayLiteralExpression(
      name.elements.map((e) => {
        if (ts.isOmittedExpression(e)) return e;
        const target = this.assignmentTarget(e.name);
        if (e.dotDotDotToken !== undefined) return f.createSpreadElement(target);
        return e.initializer !== undefined ? f.createAssignment(target, e.initializer) : target;
      }),
    );
  }

  private addHelper(s: TS.Statement): void {
    const rewritten = this.rewrite(s, false);
    for (const name of rewritten.refs) this.helperRefs.add(name);
    this.helpers.push(rewritten.node as TS.Statement);
  }

  /**
   * A statement as written, rewritten for the segment. What it names counts as READ — the
   * conservative answer for arbitrary code — except an assignment the compiler made itself
   * (`$v.x = …`), whose target is WRITTEN.
   */
  private verbatim(s: TS.Statement, original?: TS.Node, writes: readonly string[] = [], line?: number): void {
    const { ts } = this;
    let target: string | undefined;
    if (ts.isExpressionStatement(s) && ts.isBinaryExpression(s.expression) && s.expression.operatorToken.kind === ts.SyntaxKind.EqualsToken) {
      const left = s.expression.left;
      if (ts.isPropertyAccessExpression(left) && ts.isIdentifier(left.expression) && left.expression.text === "$v" && left.pos < 0) target = left.name.text;
    }
    if (target !== undefined) {
      const assignment = s as TS.ExpressionStatement & { expression: TS.BinaryExpression };
      const right = this.rewrite(this.ts.factory.createExpressionStatement(assignment.expression.right), true, original);
      const rewritten = this.ts.factory.createExpressionStatement(this.ts.factory.createAssignment(assignment.expression.left, (right.node as TS.ExpressionStatement).expression));
      this.cur.code.push({ statement: rewritten, line: line ?? this.lineOf(original ?? s) });
      this.touch(right.refs, [...writes, target]);
      this.cur.jumps.push(...right.jumps);
      return;
    }
    const rewritten = this.rewrite(s, true, original);
    this.cur.code.push({ statement: rewritten.node as TS.Statement, line: line ?? this.lineOf(original ?? s) });
    this.touch(rewritten.refs, writes);
    this.cur.jumps.push(...rewritten.jumps);
    if (rewritten.returns) this.sawReturn = true;
  }

  /** A synthesized assignment whose value is source — its target is WRITTEN, its value read. */
  private emit(s: TS.Statement, original: TS.Node, writes: readonly string[], line?: number): void {
    this.verbatim(s, original, writes, line);
  }

  /** A fan-out's element, lowered as a machine of its own (SCRIPTS.md §7). */
  private lowerElement(spec: ElementSpec): void {
    this.machine = spec.id + 1;
    this.loops = [];
    this.handlers = [];
    this.pendingLabels = [];
    const statements = this.mode === "calls" ? this.anfList(spec.statements) : spec.statements;
    for (const s of statements) this.collectHoisting(s);
    this.cur = this.newBlock(this.lineOf(spec.node));
    spec.entry = this.cur.id;
    this.lowerList(statements);
    this.end({ kind: "return", value: undefined, fallthrough: false });
  }

  // --- the rewrite ---------------------------------------------------------------------------------

  /**
   * Rewrite a statement for a segment: hoisted variables become `$v.<name>`, a hook becomes the hook
   * module's export, `llm<T>(…)` becomes `llm.withOutput(<schema>)(…)`, and — where `controlFlow` —
   * a `return` becomes the continuation that ends the state, and a `break`/`continue` that leaves the
   * statement becomes a jump to the block it targets.
   */
  rewrite(node: TS.Node, controlFlow: boolean, original?: TS.Node): { node: TS.Node; refs: Set<string>; jumps: number[]; returns: boolean } {
    const { ts } = this;
    const refs = new Set<string>();
    const jumps: number[] = [];
    let returns = false;
    const loops = this.loops;
    const compiler = this;

    const transformer: TS.TransformerFactory<TS.Node> = (context) => {
      const f = context.factory;
      /** Jump targets defined INSIDE the statement being rewritten, which a `break` there keeps. */
      const local: Array<{ kind: "loop" | "switch" | "label"; label?: string }> = [];
      let functionDepth = 0;

      const jump = (target: number): TS.Statement => {
        jumps.push(target);
        return f.createBlock([f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createNumericLiteral(target))), f.createContinueStatement(f.createIdentifier("$dispatch"))]);
      };

      const visit = (n: TS.Node): TS.Node | undefined => {
        if (ts.isTypeNode(n) && !ts.isExpressionWithTypeArguments(n)) return n;
        if (isFunctionBoundary(ts, n)) {
          functionDepth++;
          const saved = local.splice(0);
          let out = ts.visitEachChild(n, visit, context);
          local.push(...saved);
          functionDepth--;
          const names = compiler.schemaParams.get(n);
          if (names !== undefined) out = withSchemaParams(ts, out, names);
          return out;
        }
        if (ts.isPropertyAccessExpression(n) && ts.isIdentifier(n.expression) && n.expression.text === "$v" && n.pos < 0) {
          refs.add(n.name.text);
          return n;
        }
        if (ts.isIdentifier(n)) return compiler.rewriteIdentifier(n, refs, f);
        if (ts.isShorthandPropertyAssignment(n)) {
          const symbol = compiler.checker.getShorthandAssignmentValueSymbol(n);
          const hoisted = symbol !== undefined ? compiler.hoisted.get(symbol) : undefined;
          if (hoisted !== undefined) {
            refs.add(hoisted.name);
            return f.createPropertyAssignment(n.name.text, vRef(ts, hoisted.name));
          }
          const hook = compiler.hookOf(n.name);
          if (hook !== undefined && compiler.isGlobalHook(n.name)) return f.createPropertyAssignment(n.name.text, hookRef(ts, hook));
          return n;
        }
        if (ts.isCallExpression(n) && ts.isIdentifier(n.expression) && n.pos >= 0) {
          const hook = compiler.hookOf(n.expression);
          if (hook === "workflow") {
            const target = compiler.workflowTarget(n);
            compiler.coarseCalls.set(sanitizeKey(target.stateId.split("/").pop()!), target.stateId);
            return f.createCallExpression(visit(n.expression) as TS.Expression, undefined, [f.createStringLiteral(target.stateId), ...n.arguments.slice(1).map((a) => visit(a) as TS.Expression)]);
          }
          const inferred = (hook === "llm" || hook === "agent") && n.typeArguments === undefined ? compiler.inferredOutput(n) : undefined;
          if (inferred !== undefined) {
            const callee = visit(n.expression) as TS.Expression;
            const typed = f.createCallExpression(f.createPropertyAccessExpression(callee, "withOutput"), undefined, [jsonExpression(f, inferred as JsonValue)]);
            return f.createCallExpression(typed, undefined, n.arguments.map((a) => visit(a) as TS.Expression));
          }
          if ((hook === "llm" || hook === "agent") && n.typeArguments?.[0] !== undefined) {
            const callee = visit(n.expression) as TS.Expression;
            const schema = compiler.schemaExpression(n, f);
            const typed = f.createCallExpression(f.createPropertyAccessExpression(callee, "withOutput"), undefined, [schema]);
            return f.createCallExpression(typed, undefined, n.arguments.map((a) => visit(a) as TS.Expression));
          }
          const imported = compiler.importedAs(n.expression);
          if (imported?.kind === "operation") {
            // Code imported `as: "operation"`: its call is RECORDED, so a replay reads it back rather than running it again.
            // Recording is asynchronous, so the call is AWAITED where it stands — which a synchronous
            // function it stands in cannot do.
            let scope: TS.Node | undefined = n.parent;
            while (scope !== undefined && !ts.isFunctionLike(scope)) scope = scope.parent;
            const isAsync = scope === undefined || (ts.getCombinedModifierFlags(scope as TS.Declaration) & ts.ModifierFlags.Async) !== 0;
            if (!isAsync) throw compiler.error(n, `'${n.expression.text}' is imported as an operation, whose call is recorded and so awaited; call it from an async function`);
            const call = f.createCallExpression(n.expression, undefined, n.arguments.map((a) => visit(a) as TS.Expression));
            const record = f.createCallExpression(hookRef(ts, "recorded"), undefined, [f.createStringLiteral(n.expression.text), f.createArrowFunction(undefined, undefined, [], undefined, undefined, call)]);
            return f.createParenthesizedExpression(f.createAwaitExpression(record));
          }
          const helper = compiler.helperOf(n.expression);
          const names = helper !== undefined ? compiler.schemaParams.get(helper) : undefined;
          if (helper !== undefined && names !== undefined) {
            const args = n.arguments.map((a) => visit(a) as TS.Expression);
            const substitution = compiler.typeSubstitutionOf(n, helper);
            const schemas = names.map((name) => {
              const tp = helper.typeParameters?.find((p) => p.name.text === name);
              const symbol = tp !== undefined ? compiler.checker.getSymbolAtLocation(tp.name) : undefined;
              const type = symbol !== undefined ? substitution.get(symbol) : undefined;
              return type !== undefined ? jsonExpression(f, compiler.schemaOfType(type, n) as JsonValue) : f.createIdentifier("undefined");
            });
            while (args.length < helper.parameters.length) args.push(f.createIdentifier("undefined"));
            return f.createCallExpression(visit(n.expression) as TS.Expression, undefined, [...args, ...schemas]);
          }
        }
        if (controlFlow && functionDepth === 0) {
          if (ts.isReturnStatement(n)) {
            returns = true;
            const value = n.expression !== undefined ? (visit(n.expression) as TS.Expression) : undefined;
            return f.createReturnStatement(continuation(f, SCRIPT_RETURN, undefined, value));
          }
          if (ts.isBreakStatement(n) || ts.isContinueStatement(n)) {
            const label = n.label?.text;
            const isBreak = ts.isBreakStatement(n);
            const kept = label !== undefined ? local.some((l) => l.label === label) : local.some((l) => l.kind === "loop" || (isBreak && l.kind === "switch"));
            if (kept) return n;
            const candidates = [...loops].reverse();
            const target =
              label !== undefined
                ? candidates.find((l) => l.labels.includes(label))
                : isBreak
                  ? candidates.find((l) => l.continueTo >= 0 || (l as LoweredLoop & { isSwitch?: boolean }).isSwitch === true)
                  : candidates.find((l) => l.continueTo >= 0);
            if (target === undefined || (!isBreak && target.continueTo < 0)) throw compiler.error(original ?? n, `\`${isBreak ? "break" : "continue"}${label ? ` ${label}` : ""}\` has no loop to leave`);
            return jump(isBreak ? target.breakTo : target.continueTo);
          }
          if (ts.isLabeledStatement(n)) {
            local.push({ kind: "label", label: n.label.text });
            const out = ts.visitEachChild(n, visit, context);
            local.pop();
            return out;
          }
          if (ts.isIterationStatement(n, false) || ts.isSwitchStatement(n)) {
            local.push({ kind: ts.isSwitchStatement(n) ? "switch" : "loop" });
            const out = ts.visitEachChild(n, visit, context);
            local.pop();
            return out;
          }
        }
        return ts.visitEachChild(n, visit, context);
      };
      return (root) => visit(root) as TS.Node;
    };

    const result = ts.transform(node, [transformer]);
    const out = result.transformed[0]!;
    result.dispose();
    return { node: out, refs, jumps, returns };
  }

  rewriteIdentifier(n: TS.Identifier, refs: Set<string>, f: TS.NodeFactory): TS.Node {
    const { ts } = this;
    if (n.pos < 0) return n; // the compiler's own `$v`, `$rt`, `$pc`
    const parent = n.parent as TS.Node | undefined;
    // Only a REFERENCE is rewritten — never a name being declared, a property, or a label.
    if (parent !== undefined) {
      if ((ts.isPropertyAccessExpression(parent) && parent.name === n) || (ts.isQualifiedName(parent) && parent.right === n)) return n;
      if ((ts.isPropertyAssignment(parent) || ts.isMethodDeclaration(parent) || ts.isPropertyDeclaration(parent) || ts.isGetAccessor(parent) || ts.isSetAccessor(parent)) && parent.name === n) return n;
      if ((ts.isBindingElement(parent) && parent.propertyName === n) || ts.isLabeledStatement(parent) || ts.isBreakOrContinueStatement(parent)) return n;
      if ((ts.isVariableDeclaration(parent) || ts.isParameter(parent) || ts.isFunctionDeclaration(parent) || ts.isClassDeclaration(parent) || ts.isBindingElement(parent) || ts.isFunctionExpression(parent)) && parent.name === n) return n;
    }
    const symbol = this.checker.getSymbolAtLocation(n);
    if (symbol !== undefined) {
      const hoisted = this.hoisted.get(symbol);
      if (hoisted !== undefined) {
        refs.add(hoisted.name);
        return vRef(ts, hoisted.name);
      }
      // A type parameter a generic helper was handed as a schema (`"phases"`/`"state"` mode).
    }
    if (this.isGlobalHook(n)) return hookRef(ts, this.hookOf(n)!);
    return n;
  }

  /** The schema an `llm<T>` names, as code: a literal, or — for a generic helper's `T` — its schema parameter. */
  schemaExpression(call: TS.CallExpression, f: TS.NodeFactory): TS.Expression {
    const { ts } = this;
    const node = call.typeArguments![0]!;
    if (ts.isTypeReferenceNode(node)) {
      const symbol = this.checker.getSymbolAtLocation(node.typeName);
      const declaration = symbol?.declarations?.[0];
      if (declaration !== undefined && ts.isTypeParameterDeclaration(declaration) && this.schemaParams.has(declaration.parent)) {
        return f.createIdentifier(`$schema_${declaration.name.text}`);
      }
    }
    return jsonExpression(f, this.typeArgumentSchema(call) as JsonValue);
  }

  /** {@link typeSubstitution}, for a call site the rewrite reaches (a generic helper, not inlined). */
  typeSubstitutionOf(call: TS.CallExpression, fn: TS.FunctionLikeDeclaration): Map<TS.Symbol, TS.Type> {
    return this.typeSubstitution(call, fn);
  }

  schemaOfType(type: TS.Type, at: TS.Node): JsonSchema {
    try {
      const withDefs = typeToWireSchema(this.ts, this.checker, type, "a helper's type argument", { defs: true });
      return (withDefs.schema as { type?: unknown }).type === "object" ? withDefs.schema : typeToWireSchema(this.ts, this.checker, type, "a helper's type argument").schema;
    } catch (e) {
      if (e instanceof WireTypeError) throw this.error(at, e.message);
      throw e;
    }
  }

  collectRefs(node: TS.Node, into: Set<string>): void {
    const visit = (n: TS.Node): void => {
      if (this.ts.isIdentifier(n) && n.pos >= 0) {
        const symbol = this.checker.getSymbolAtLocation(n);
        const hoisted = symbol !== undefined ? this.hoisted.get(symbol) : undefined;
        if (hoisted !== undefined) into.add(hoisted.name);
      } else if (this.ts.isPropertyAccessExpression(n) && this.ts.isIdentifier(n.expression) && n.expression.text === "$v") {
        into.add(n.name.text);
      }
      this.ts.forEachChild(n, visit);
    };
    visit(node);
  }

  // --- documents: the parts only the compiler knows -----------------------------------------------------

  /** Helpers, rewritten, re-declared in every segment. */
  get helperStatements(): readonly TS.Statement[] {
    return this.helpers;
  }

  get helperReads(): ReadonlySet<string> {
    return this.helperRefs;
  }

  get scriptFile(): string {
    return this.file;
  }

  /** What the fallthrough returns: the exports (body mode), the default export, or nothing. */
  fallthroughValue(f: TS.NodeFactory, machine: number): TS.Expression | undefined {
    const { ts } = this;
    if (machine !== 0) return undefined;
    if (this.exportNames.length > 0) return f.createObjectLiteralExpression(this.exportNames.map((n) => f.createPropertyAssignment(n, vRef(ts, n))));
    if (this.defaultExport !== undefined) return vRef(ts, this.defaultExport);
    return undefined;
  }

  /** The preamble every segment carries for what the script imported (SCRIPTS.md §10). */
  importStubs(): string[] {
    const lines: string[] = [];
    for (const [name, imported] of this.imported) {
      if (imported.kind === "state") lines.push(`const ${name} = (inputs?: unknown): Promise<any> => $rt.workflow(${JSON.stringify(imported.stateId)}, inputs);`);
      else if (imported.kind === "registry") lines.push(`const ${name} = (...args: unknown[]): Promise<any> => $rt.call(${JSON.stringify(imported.ref)}, ...args);`);
      else if (imported.kind === "data") lines.push(`const ${name}: any = ${JSON.stringify(imported.data)};`);
      else if (imported.kind === "prompt") {
        const template = JSON.stringify(imported.template);
        lines.push(`const ${name}: any = Object.assign((inputs?: Record<string, unknown>) => $rt.llm({ prompt: "", template: ${template}, inputs: inputs ?? {} }), { withOutput: (schema: unknown) => (inputs?: Record<string, unknown>) => $rt.llm.withOutput(schema)({ prompt: "", template: ${template}, inputs: inputs ?? {} }) });`);
      }
    }
    return lines;
  }

  /** The rootOutputs of the root machine — what the function returns, read off the machine's `_return`. */
  rootOutputs(returns: JsonSchema | undefined, returned: (path: string) => string | { $expr: string }, optional = false): Record<string, NamedParameterDecl> {
    const outputs: Record<string, NamedParameterDecl> = {};
    const slot = (schema: JsonSchema | undefined, path: string, isOptional: boolean): NamedParameterDecl =>
      ({
        ...(schema !== undefined && Object.keys(wireForm(schema)).length > 0 ? { schema: wireForm(schema) } : {}),
        ...(isOptional || optional ? { optional: true } : {}),
        binding: returned(path),
      }) as NamedParameterDecl;
    if (this.bodyMode) {
      if (this.exportNames.length > 0) {
        for (const name of this.exportNames) outputs[name] = slot(this.hoistedByName.get(name)?.schema, `.${name}`, false);
        return outputs;
      }
      if (this.defaultExport !== undefined || this.sawReturn) {
        outputs.result = slot(undefined, "", true);
        this.wholeReturn = true;
      }
      return outputs;
    }
    const schema = returns as { type?: unknown; properties?: Record<string, JsonSchema>; required?: string[] } | undefined;
    if (schema === undefined || schema.type === "null") return outputs;
    if (schema.type === "object" && schema.properties !== undefined) {
      for (const [name, property] of Object.entries(schema.properties)) outputs[name] = slot(property, `.${name}`, !(schema.required ?? []).includes(name));
      return outputs;
    }
    outputs.result = slot(returns, "", false);
    this.wholeReturn = true;
    return outputs;
  }

  get returnsWhole(): boolean {
    return this.wholeReturn;
  }

  private checkUnusedPhaseDescriptions(meta: ScriptMeta, phases: readonly string[]): void {
    for (const described of meta.phases ?? []) {
      if (this.mode !== "state" && !phases.includes(described.title)) {
        this.warnings.push(`${this.file}: meta.phases names '${described.title}', which no phase() call reaches`);
      }
    }
  }

  private provenance(): GeneratedProvenance {
    const inputs: Record<string, string> = {};
    for (const sf of this.program.getSourceFiles()) {
      const name = sf.fileName;
      if (name === HOOK_GLOBALS_PATH || name === HOOK_MODULE_PATH || this.program.isSourceFileDefaultLibrary(sf) || name.includes("/node_modules/")) continue;
      inputs[name] = sha256Hex(sf.text);
    }
    // A prompt or data import is read, not compiled — it is an input all the same.
    for (const s of this.sf.statements) {
      if (!this.ts.isImportDeclaration(s) || !this.ts.isStringLiteral(s.moduleSpecifier)) continue;
      const specifier = s.moduleSpecifier.text;
      if (!/\.(md|json|ya?ml)$/i.test(specifier) && importAttributes(this.ts, s).type !== "json") continue;
      const file = this.relative(specifier);
      const text = file !== undefined ? this.options.vfs.read(file) : undefined;
      if (file !== undefined && text !== undefined) inputs[file] = sha256Hex(text);
    }
    return { from: this.file, inputs };
  }

  // --- errors and lines ----------------------------------------------------------------------------

  lineOf(node: TS.Node): number {
    let n: TS.Node | undefined = node;
    while (n !== undefined && n.pos < 0) n = (n as { original?: TS.Node }).original ?? undefined;
    if (n === undefined) return this.cur?.line ?? 1;
    return this.sf.getLineAndCharacterOfPosition(n.getStart(this.sf)).line + 1;
  }

  columnOf(node: TS.Node): number {
    return node.pos < 0 ? 1 : this.sf.getLineAndCharacterOfPosition(node.getStart(this.sf)).character + 1;
  }

  where(node: TS.Node): string {
    return `${this.file}:${this.lineOf(node)}:${this.columnOf(node)}`;
  }

  error(node: TS.Node, message: string): ScriptCompileError {
    // A synthesized node has no position; report it at the file.
    if (node.pos < 0) return new ScriptCompileError(message, this.file, this.lineOf(node));
    const at = this.sf.getLineAndCharacterOfPosition(node.getStart(this.sf));
    return new ScriptCompileError(message, this.file, at.line + 1, at.character + 1);
  }

  get warningsOut(): string[] {
    return this.warnings;
  }
}

// --- machines: blocks → states --------------------------------------------------------------------------

/** The region label of the script's own code before any phase, and of fan-out element `k`. */
const ROOT = "\u0000root";
const elementLabel = (k: number): string => `\u0000element:${k}`;

interface BuildSpec {
  stateId: string;
  meta: ScriptMeta;
  inputs: Record<string, ParameterDecl>;
  returns: JsonSchema | undefined;
  generated: GeneratedProvenance;
  prelude: readonly TS.Statement[];
  imports: readonly TS.ImportDeclaration[];
  rootOutputs: (returned: (path: string) => string | { $expr: string }, optional: boolean) => Record<string, NamedParameterDecl>;
}

/** One STATE a composite holds: its head (the composite's own operation), or a child. */
interface MNode {
  key: string;
  kind: "head" | "call" | "after" | "catch";
  composite: Composite;
  entries: number[];
  site?: CallSite;
  handler?: Handler;
  region: Set<number>;
  in: Set<string>;
  out: Set<string>;
  exits: MExit[];
  label: string;
  line: number;
  /** The rules its exits became, where lifting found readable guards. */
  lifted?: TransitionDecl[];
}

type MExit =
  | { kind: "node"; node: MNode }
  | { kind: "mount"; mount: Mount }
  | { kind: "composite"; target: Composite; entry: number }
  | { kind: "leave"; next: string; entry?: number };

/** A mount one state exits INTO — a called state or a fan-out — and the state its result goes to. */
interface Mount {
  key: string;
  site: CallSite;
  from: MNode;
  after: MNode;
  element?: Composite;
}

/** A MACHINE: the script's root, a phase, or a fan-out's element — a composite state (SCRIPTS.md §7). */
interface Composite {
  id: string;
  key: string;
  label: string;
  kind: "root" | "phase" | "element";
  region: string;
  head: MNode;
  nodes: MNode[];
  mounts: Mount[];
  phases: Composite[];
  in: Set<string>;
  out: Set<string>;
  /** The composite's exits, when it is a phase: where it can hand on to. */
  leaves: Map<string, number | undefined>;
  element?: ElementSpec;
  line: number;
  description?: string;
}

class MachineBuilder {
  private readonly ts: typeof TS;
  private readonly sets = new Map<number, Set<string>>();
  private readonly composites = new Map<string, Composite>();
  private readonly liveIn = new Map<number, Set<string>>();
  private readonly phaseKeys = new Map<string, string>();
  private readonly documents: Record<string, StateDef> = {};
  private readonly sourceMap: Record<string, SourceAt> = {};
  private printer!: TS.Printer;

  constructor(
    private readonly c: Compiler,
    private readonly spec: BuildSpec,
  ) {
    this.ts = c.ts;
  }

  build(): { documents: Record<string, StateDef>; sourceMap: Record<string, SourceAt>; phases: string[] } {
    this.printer = this.ts.createPrinter({ removeComments: false });
    this.phaseSets();
    const phases = this.phaseOrder();
    this.liveness();
    const root = this.composite(ROOT, this.spec.stateId, "", this.spec.meta.name ?? this.spec.meta.label ?? this.spec.stateId.split("/").pop()!, "root", [0], 1);
    for (const phase of phases) {
      const key = this.phaseKeys.get(phase)!;
      const entries = this.c.blocks.filter((b) => b.term?.kind === "phase" && b.term.phase === phase && this.sets.has(b.id)).map((b) => (b.term as { to: number }).to);
      const detail = this.spec.meta.phases?.find((p) => p.title === phase)?.detail;
      const composite = this.composite(phase, `${this.spec.stateId}/${key}`, key, phase, "phase", [...new Set(entries)], this.c.blocks[entries[0]!]?.line ?? 1);
      if (detail !== undefined) composite.description = detail;
      root.phases.push(composite);
    }
    for (const element of this.c.elements) {
      if (element.site === undefined || element.entry === undefined) continue;
      const label = `each: ${this.c.sites[element.site]!.label}`;
      this.composite(elementLabel(element.id), "", "", label, "element", [element.entry], this.c.blocks[element.entry]!.line, element);
    }
    // Nodes: one per call (per region it is reached in), per result of a mount, per catch.
    for (const block of this.c.blocks) {
      if (block.term?.kind !== "call") continue;
      const site = this.c.sites[block.term.site]!;
      for (const region of this.sets.get(block.id) ?? []) {
        const composite = this.composites.get(region)!;
        const line = this.c.lineOf(site.node);
        if (site.kind === "prompt" || site.kind === "script") this.node(composite, `call_${site.id}`, "call", [block.term.to], site.label, line, site);
        else this.node(composite, `after_${site.id}`, "after", [block.term.to], `after ${site.label}`, line, site);
        if (site.handler !== undefined) this.node(composite, `catch_${site.handler.id}`, "catch", [site.handler.block], "catch", this.c.blocks[site.handler.block]!.line, undefined, site.handler);
      }
    }
    // Element composites live under the composite the fan-out is made in.
    for (const composite of [...this.composites.values()]) {
      for (const node of [composite.head, ...composite.nodes]) node.region = this.region(node);
    }
    for (const composite of this.composites.values()) for (const node of [composite.head, ...composite.nodes]) node.exits = this.exitsOf(node);
    this.flows(root);
    for (const composite of this.composites.values()) for (const node of [composite.head, ...composite.nodes]) this.lift(node);
    this.checkWires();
    this.emitComposite(root, undefined);
    return { documents: this.documents, sourceMap: this.sourceMap, phases };
  }

  // --- regions ---------------------------------------------------------------------------------------

  /** Which region — the script's root, a phase, an element — can be current at the start of each block. */
  private phaseSets(): void {
    const add = (block: number, labels: Iterable<string>): boolean => {
      const set = this.sets.get(block) ?? new Set<string>();
      const before = set.size;
      for (const l of labels) set.add(l);
      this.sets.set(block, set);
      return set.size !== before;
    };
    const work: number[] = [];
    if (this.c.blocks.length > 0) {
      add(0, [ROOT]);
      work.push(0);
    }
    for (const element of this.c.elements) {
      if (element.entry === undefined) continue;
      add(element.entry, [elementLabel(element.id)]);
      work.push(element.entry);
    }
    while (work.length > 0) {
      const id = work.pop()!;
      const block = this.c.blocks[id]!;
      const current = this.sets.get(id)!;
      const flow = (to: number, labels: Iterable<string>): void => {
        if (add(to, labels)) work.push(to);
      };
      for (const j of block.jumps) flow(j, current);
      if (block.handler !== undefined) flow(block.handler.block, current);
      const term = block.term;
      if (term === undefined) continue;
      if (term.kind === "goto") flow(term.to, current);
      else if (term.kind === "branch") {
        flow(term.then, current);
        flow(term.else, current);
      } else if (term.kind === "call") flow(term.to, current);
      else if (term.kind === "phase") flow(term.to, [term.phase]);
    }
  }

  /** The phases, in the order control first reaches them — and their keys. */
  private phaseOrder(): string[] {
    const order: string[] = [];
    for (const block of this.c.blocks) {
      if (block.term?.kind === "phase" && this.sets.has(block.id) && !order.includes(block.term.phase)) order.push(block.term.phase);
    }
    const taken = new Map<string, string>();
    for (const phase of order) {
      const key = phase.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "phase";
      const clash = taken.get(key);
      if (clash !== undefined) throw new ScriptCompileError(`phases '${clash}' and '${phase}' both become the state '${key}' — rename one`, this.c.scriptFile);
      taken.set(key, phase);
      this.phaseKeys.set(phase, key);
    }
    return order;
  }

  private composite(region: string, id: string, key: string, label: string, kind: Composite["kind"], entries: number[], line: number, element?: ElementSpec): Composite {
    const composite = { id, key, label, kind, region, nodes: [], mounts: [], phases: [], in: new Set(), out: new Set(), leaves: new Map(), line, ...(element !== undefined ? { element } : {}) } as unknown as Composite;
    composite.head = { key: "", kind: "head", composite, entries, region: new Set(), in: new Set(), out: new Set(), exits: [], label, line };
    this.composites.set(region, composite);
    return composite;
  }

  private node(composite: Composite, key: string, kind: MNode["kind"], entries: number[], label: string, line: number, site?: CallSite, handler?: Handler): MNode {
    const existing = composite.nodes.find((n) => n.key === key);
    if (existing !== undefined) return existing;
    const node: MNode = { key, kind, composite, entries, region: new Set(), in: new Set(), out: new Set(), exits: [], label, line, ...(site !== undefined ? { site } : {}), ...(handler !== undefined ? { handler } : {}) };
    composite.nodes.push(node);
    return node;
  }

  /** The blocks a state runs: reachable from its entries, in its region, without crossing a cut. */
  private region(node: MNode): Set<number> {
    const label = node.composite.region;
    const region = new Set<number>();
    const work = [...node.entries];
    while (work.length > 0) {
      const id = work.pop()!;
      if (region.has(id) || !this.sets.get(id)?.has(label)) continue;
      region.add(id);
      const block = this.c.blocks[id]!;
      work.push(...block.jumps);
      if (block.handler !== undefined) work.push(block.handler.block);
      const term = block.term;
      if (term?.kind === "goto") work.push(term.to);
      else if (term?.kind === "branch") work.push(term.then, term.else);
      else if (term?.kind === "phase" && term.phase === label) work.push(term.to);
    }
    return region;
  }

  private exitsOf(node: MNode): MExit[] {
    const composite = node.composite;
    const exits: MExit[] = [];
    const seen = new Set<string>();
    const push = (key: string, exit: MExit): void => {
      if (seen.has(key)) return;
      seen.add(key);
      exits.push(exit);
    };
    for (const id of [...node.region].sort((a, b) => a - b)) {
      const term = this.c.blocks[id]!.term;
      if (term === undefined || term.kind === "return") {
        push("$return", { kind: "leave", next: SCRIPT_RETURN });
        continue;
      }
      if (term.kind === "call") {
        const site = this.c.sites[term.site]!;
        if (site.kind === "prompt" || site.kind === "script") push(`call_${site.id}`, { kind: "node", node: composite.nodes.find((n) => n.key === `call_${site.id}`)! });
        else push(`mount_${site.id}`, { kind: "mount", mount: this.mount(node, site) });
      } else if (term.kind === "phase" && term.phase !== composite.region) {
        const key = this.phaseKeys.get(term.phase)!;
        if (composite.kind === "root") push(key, { kind: "composite", target: this.composites.get(term.phase)!, entry: term.to });
        else {
          push(key, { kind: "leave", next: key, entry: term.to });
          composite.leaves.set(key, term.to);
        }
      }
    }
    return exits;
  }

  /** The mount a state exits into for a call site — one per state that can make the call. */
  private mount(from: MNode, site: CallSite): Mount {
    const composite = from.composite;
    const existing = composite.mounts.find((m) => m.site === site && m.from === from);
    if (existing !== undefined) return existing;
    const base = site.kind === "map" ? "map" : sanitizeKey(site.stateId!.split("/").pop()!);
    const count = composite.mounts.filter((m) => m.site === site).length;
    const key = `${base}_${site.id}${count > 0 ? `_${count}` : ""}`;
    const after = composite.nodes.find((n) => n.key === `after_${site.id}`)!;
    const mount: Mount = { key, site, from, after };
    if (site.kind === "map") mount.element = this.composites.get(elementLabel(site.element!))!;
    composite.mounts.push(mount);
    return mount;
  }

  // --- what crosses --------------------------------------------------------------------------------

  /**
   * Backward liveness over the blocks. A cut — a call, a `phase()` — is an edge like any other, and it
   * is exactly where a live value has to CROSS into another state.
   */
  private liveness(): void {
    const successors = (b: Block): number[] => {
      const out = [...b.jumps];
      if (b.handler !== undefined) out.push(b.handler.block);
      const t = b.term;
      if (t?.kind === "goto" || t?.kind === "phase" || t?.kind === "call") out.push(t.to);
      else if (t?.kind === "branch") out.push(t.then, t.else);
      return out;
    };
    for (const b of this.c.blocks) this.liveIn.set(b.id, new Set([...b.uses, ...this.c.helperReads]));
    // A fan-out READS, at the call, whatever its element reads from outside it.
    const elementReads = (b: Block): string[] => {
      if (b.term?.kind !== "call") return [];
      const site = this.c.sites[b.term.site]!;
      if (site.kind !== "map") return [];
      const element = this.c.elements[site.element!]!;
      if (element.entry === undefined) return [];
      const params = new Set(element.params.map((p) => p.name));
      return [...this.liveIn.get(element.entry)!].filter((v) => !params.has(v));
    };
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of [...this.c.blocks].reverse()) {
        const into = this.liveIn.get(b.id)!;
        for (const v of elementReads(b)) {
          if (!into.has(v)) {
            into.add(v);
            changed = true;
          }
        }
        for (const next of successors(b)) {
          for (const v of this.liveIn.get(next)!) {
            if (!b.defs.has(v) && !into.has(v)) {
              into.add(v);
              changed = true;
            }
          }
        }
      }
    }
  }

  private liveAt(blocks: Iterable<number>, without: Array<string | undefined> = []): Set<string> {
    const out = new Set<string>();
    for (const b of blocks) for (const v of this.liveIn.get(b) ?? []) out.add(v);
    for (const w of without) if (w !== undefined) out.delete(w);
    return out;
  }

  /** Every state's inputs and outputs, and every composite's. */
  private flows(root: Composite): void {
    const all = [...this.composites.values()];
    for (const composite of all) {
      for (const node of [composite.head, ...composite.nodes]) {
        if (node.kind === "head") node.in = composite.kind === "root" ? new Set() : this.liveAt(node.entries);
        else if (node.kind === "call" || node.kind === "after") {
          node.in = this.liveAt(node.entries, [node.site!.target]);
          if (node.kind === "call") for (const v of Object.values(node.site!.inputVars ?? {})) node.in.add(v);
        }
        else node.in = this.liveAt(node.entries, [node.handler!.param]);
      }
      composite.in = new Set(composite.head.in);
    }
    // A call's failure resumes at its `catch` with the call's own inputs.
    for (const composite of all) {
      for (const node of composite.nodes) {
        if (node.kind !== "call" || node.site!.handler === undefined) continue;
        const handler = composite.nodes.find((n) => n.key === `catch_${node.site!.handler!.id}`);
        for (const v of handler?.in ?? []) node.in.add(v);
      }
    }
    // Outputs: what each exit's target takes in.
    for (const composite of all) {
      for (const node of [composite.head, ...composite.nodes]) {
        for (const exit of node.exits) {
          if (exit.kind === "node") for (const v of exit.node.in) node.out.add(v);
          else if (exit.kind === "mount") {
            for (const v of exit.mount.after.in) node.out.add(v);
            for (const v of Object.values(exit.mount.site.inputVars ?? {})) node.out.add(v);
            if (exit.mount.element !== undefined) for (const v of this.captured(exit.mount.element)) node.out.add(v);
            const handler = exit.mount.site.handler !== undefined ? composite.nodes.find((n) => n.key === `catch_${exit.mount.site.handler!.id}`) : undefined;
            for (const v of handler?.in ?? []) node.out.add(v);
          } else if (exit.kind === "composite") for (const v of exit.target.in) node.out.add(v);
          else if (exit.kind === "leave" && exit.next !== SCRIPT_RETURN) {
            const target = [...this.composites.values()].find((x) => x.key === exit.next && x.kind === "phase");
            for (const v of target?.in ?? []) {
              node.out.add(v);
              composite.out.add(v);
            }
          }
        }
      }
    }
    void root;
  }

  /** What an element reads from outside it — wired in from the state that makes the fan-out. */
  private captured(element: Composite): Set<string> {
    const params = new Set(element.element!.params.map((p) => p.name));
    return new Set([...element.head.in].filter((v) => !params.has(v)));
  }

  /** A value that crosses a cut must have a wire form (SCRIPTS.md §8). */
  private checkWires(): void {
    for (const composite of this.composites.values()) {
      for (const node of [composite.head, ...composite.nodes]) {
        if (node.kind === "head" && composite.kind === "root") continue;
        for (const v of node.in) {
          const hoisted = this.c.hoistedByName.get(v);
          if (hoisted?.unrepresentable !== undefined) {
            throw new ScriptCompileError(`the variable '${v}' lives across a cut (into '${node.label}'), so it travels as a wire — but ${hoisted.unrepresentable}. Keep it between two calls, or make it plain data`, this.c.scriptFile, node.line);
          }
        }
      }
    }
  }

  // --- readable guards -------------------------------------------------------------------------------

  /**
   * LIFT a state's exits into guards a person could have written (SCRIPTS.md §6.2): when its code is a
   * tree of branches whose conditions are expressions over its outputs — boolean-typed, and not touched
   * by any code between the branch and the exit — each exit's rule is the conjunction of the conditions
   * on its path, instead of the continuation's `_next`.
   */
  private lift(node: MNode): void {
    if (node.entries.length !== 1) return;
    const composite = node.composite;
    type Tree = { kind: "leaf"; exit: MExit } | { kind: "branch"; cond: TS.Expression; then: Tree; else: Tree };
    const failure = Symbol("unliftable");
    const exitOf = (term: Terminator | undefined): MExit | undefined => {
      if (term === undefined || term.kind === "return") return node.exits.find((e) => e.kind === "leave" && e.next === SCRIPT_RETURN);
      if (term.kind === "call") {
        const site = this.c.sites[term.site]!;
        return node.exits.find((e) => (e.kind === "node" && e.node.site === site) || (e.kind === "mount" && e.mount.site === site));
      }
      if (term.kind === "phase") {
        const key = this.phaseKeys.get(term.phase);
        return node.exits.find((e) => (e.kind === "composite" && e.target.key === key) || (e.kind === "leave" && e.next === key));
      }
      return undefined;
    };
    const touchedBelow = new Map<Tree, Set<string>>();
    const decide = (start: number, visited: Set<number>): Tree => {
      let id = start;
      const touched = new Set<string>();
      for (;;) {
        if (visited.has(id)) throw failure;
        visited.add(id);
        const block = this.c.blocks[id]!;
        if (block.handler !== undefined || block.jumps.length > 0) throw failure;
        for (const { statement } of block.code) this.c.collectRefs(statement, touched);
        const term = block.term;
        if (term?.kind === "goto" || (term?.kind === "phase" && term.phase === composite.region)) {
          id = term.to;
          continue;
        }
        if (term?.kind === "branch") {
          const then = decide(term.then, new Set(visited));
          const otherwise = decide(term.else, new Set(visited));
          const below = new Set([...(touchedBelow.get(then) ?? []), ...(touchedBelow.get(otherwise) ?? [])]);
          const tree: Tree = { kind: "branch", cond: term.cond, then, else: otherwise };
          touchedBelow.set(tree, new Set([...below, ...touched]));
          // The condition must still mean at the exit what it meant at the branch.
          const reads = new Set<string>();
          this.c.collectRefs(term.cond, reads);
          for (const r of reads) if (below.has(r)) throw failure;
          return tree;
        }
        const exit = exitOf(term);
        if (exit === undefined) throw failure;
        const leaf: Tree = { kind: "leaf", exit };
        touchedBelow.set(leaf, touched);
        return leaf;
      }
    };
    let tree: Tree;
    try {
      tree = decide(node.entries[0]!, new Set());
    } catch (e) {
      if (e === failure) return;
      throw e;
    }
    if (tree.kind === "leaf") return;
    const prefix = this.prefixOf(node);
    const paths: Array<{ guards: string[]; exit: MExit }> = [];
    const reads = new Set<string>();
    const walk = (t: Tree, guards: string[]): boolean => {
      if (t.kind === "leaf") {
        paths.push({ guards, exit: t.exit });
        return true;
      }
      if (!this.isBoolean(t.cond)) return false;
      const translated = this.translate(t.cond, prefix, reads);
      if (translated === undefined) return false;
      return walk(t.then, [...guards, translated]) && walk(t.else, [...guards, `!(${translated})`]);
    };
    if (!walk(tree, [])) return;
    for (const r of reads) {
      if (this.c.hoistedByName.get(r)?.unrepresentable !== undefined) return;
      node.out.add(r);
    }
    node.lifted = paths.map((p, i) => ({ ...this.ruleFor(node, p.exit), name: `${this.ruleName(p.exit)}${paths.filter((q) => q.exit === p.exit).length > 1 ? `_${i}` : ""}`, when: p.guards.join(" && ") }));
  }

  private isBoolean(cond: TS.Expression): boolean {
    const { ts } = this;
    const e = unwrap(ts, cond);
    if (ts.isBinaryExpression(e)) {
      const k = e.operatorToken.kind;
      if ([ts.SyntaxKind.EqualsEqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken, ts.SyntaxKind.LessThanToken, ts.SyntaxKind.LessThanEqualsToken, ts.SyntaxKind.GreaterThanToken, ts.SyntaxKind.GreaterThanEqualsToken].includes(k)) return true;
      if (k === ts.SyntaxKind.AmpersandAmpersandToken || k === ts.SyntaxKind.BarBarToken) return this.isBoolean(e.left) && this.isBoolean(e.right);
    }
    if (ts.isPrefixUnaryExpression(e) && e.operator === ts.SyntaxKind.ExclamationToken) return this.isBoolean(e.operand);
    if (e.pos < 0) return false;
    const type = this.c.checker.getTypeAtLocation(e);
    return (type.flags & (ts.TypeFlags.Boolean | ts.TypeFlags.BooleanLiteral)) !== 0;
  }

  /** A JavaScript condition as an hw expression over a state's outputs — a safe subset, or `undefined`. */
  private translate(e: TS.Expression, prefix: string, reads: Set<string>): string | undefined {
    const { ts } = this;
    const K = ts.SyntaxKind;
    if (ts.isParenthesizedExpression(e)) {
      const inner = this.translate(e.expression, prefix, reads);
      return inner === undefined ? undefined : `(${inner})`;
    }
    if (ts.isAsExpression(e) || ts.isNonNullExpression(e)) return this.translate(e.expression, prefix, reads);
    if (ts.isPropertyAccessExpression(e) && ts.isIdentifier(e.expression) && e.expression.text === "$v" && e.expression.pos < 0) {
      reads.add(e.name.text);
      return `${prefix}.${e.name.text}`;
    }
    if (ts.isIdentifier(e)) {
      if (e.pos < 0) return undefined;
      const symbol = this.c.checker.getSymbolAtLocation(e);
      const hoisted = symbol !== undefined ? this.c.hoisted.get(symbol) : undefined;
      if (hoisted === undefined) return undefined;
      reads.add(hoisted.name);
      return `${prefix}.${hoisted.name}`;
    }
    if (ts.isNumericLiteral(e)) return e.text;
    if (ts.isStringLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return `'${e.text.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
    if (e.kind === K.TrueKeyword) return "true";
    if (e.kind === K.FalseKeyword) return "false";
    if (e.kind === K.NullKeyword) return "null";
    if (ts.isPropertyAccessExpression(e)) {
      const object = this.translate(e.expression, prefix, reads);
      return object === undefined || !/^[A-Za-z_][\w]*$/.test(e.name.text) ? undefined : `${object}.${e.name.text}`;
    }
    if (ts.isElementAccessExpression(e)) {
      const object = this.translate(e.expression, prefix, reads);
      const index = unwrap(ts, e.argumentExpression);
      if (object === undefined || !(ts.isNumericLiteral(index) || ts.isStringLiteral(index))) return undefined;
      return `${object}[${ts.isNumericLiteral(index) ? index.text : `'${index.text}'`}]`;
    }
    if (ts.isPrefixUnaryExpression(e) && (e.operator === K.ExclamationToken || e.operator === K.MinusToken)) {
      const operand = this.translate(e.operand, prefix, reads);
      return operand === undefined ? undefined : `${e.operator === K.ExclamationToken ? "!" : "-"}${operand}`;
    }
    if (ts.isBinaryExpression(e)) {
      const ops: Partial<Record<number, string>> = {
        [K.EqualsEqualsEqualsToken]: "===",
        [K.ExclamationEqualsEqualsToken]: "!==",
        [K.LessThanToken]: "<",
        [K.LessThanEqualsToken]: "<=",
        [K.GreaterThanToken]: ">",
        [K.GreaterThanEqualsToken]: ">=",
        [K.AmpersandAmpersandToken]: "&&",
        [K.BarBarToken]: "||",
        [K.PlusToken]: "+",
        [K.MinusToken]: "-",
        [K.AsteriskToken]: "*",
        [K.SlashToken]: "/",
      };
      const op = ops[e.operatorToken.kind];
      if (op === undefined) return undefined;
      const left = this.translate(e.left, prefix, reads);
      const right = this.translate(e.right, prefix, reads);
      return left === undefined || right === undefined ? undefined : `${left} ${op} ${right}`;
    }
    return undefined;
  }

  // --- rules -----------------------------------------------------------------------------------------

  /** How a state's outputs are read in the composite that holds it. */
  private prefixOf(node: MNode): string {
    return node.kind === "head" ? ".operation.output" : `.children.${node.key}.output`;
  }

  private keyOf(exit: MExit): string {
    if (exit.kind === "node") return exit.node.key;
    if (exit.kind === "mount") return exit.mount.key;
    if (exit.kind === "composite") return exit.target.key;
    return exit.next;
  }

  private ruleName(exit: MExit): string {
    return exit.kind === "leave" && exit.next === SCRIPT_RETURN ? "return" : `to_${this.keyOf(exit)}`;
  }

  /** The rule an exit is — its target and what it hands over; the guard is the caller's. */
  private ruleFor(node: MNode, exit: MExit): TransitionDecl {
    const pre = this.prefixOf(node);
    const vars = (names: Iterable<string>): Record<string, string> => Object.fromEntries([...names].sort().map((v) => [v, `${pre}.${v}`]));
    if (exit.kind === "node") {
      return { to: exit.node.key, inputs: { ...vars(exit.node.in), ...(exit.node.kind === "call" ? { [CALL_INPUT]: `${pre}.${CALL_INPUT}` } : {}) } } as unknown as TransitionDecl;
    }
    if (exit.kind === "mount") {
      const site = exit.mount.site;
      if (site.kind === "map") return { to: exit.mount.key } as TransitionDecl;
      const inputs = Object.fromEntries(Object.entries(site.inputVars ?? {}).map(([name, v]) => [name, `${pre}.${v}`]));
      return { to: exit.mount.key, inputs } as unknown as TransitionDecl;
    }
    if (exit.kind === "composite") {
      return { to: exit.target.key, inputs: { [SCRIPT_CONTROL.entry]: `${pre}.${SCRIPT_CONTROL.entry}`, ...vars(exit.target.in) } } as unknown as TransitionDecl;
    }
    return { to: "terminate.success" } as TransitionDecl;
  }

  /** A state's rules: its exits, guarded by the continuation's `_next` — or lifted — and its failure. */
  private rulesOf(node: MNode): TransitionDecl[] {
    const pre = this.prefixOf(node);
    const rules: TransitionDecl[] = node.lifted ?? node.exits.map((exit) => ({ name: this.ruleName(exit), when: `${pre}.${SCRIPT_CONTROL.next} === '${this.keyOf(exit)}'`, ...this.ruleFor(node, exit) }) as TransitionDecl);
    if (node.kind === "call" && node.site!.handler !== undefined) {
      const handler = node.composite.nodes.find((n) => n.key === `catch_${node.site!.handler!.id}`)!;
      const self = `.children.${node.key}`;
      rules.push({
        name: "catch",
        // The CALL failed — not the code after it, which catches its own throws.
        when: `${self}.outcome === 'error' && ${self}.operation[0].outcome === 'error'`,
        to: handler.key,
        inputs: { [ERROR_INPUT]: `${self}.failure`, ...Object.fromEntries([...handler.in].sort().map((v) => [v, `${self}.inputs.${v}`])) },
      } as unknown as TransitionDecl);
    }
    return rules;
  }

  /** The rules on a mount: its result handed to the state after it, and — in a `try` — its failure to the `catch`. */
  private mountRules(mount: Mount): TransitionDecl[] {
    const pre = this.prefixOf(mount.from);
    const self = `.children.${mount.key}`;
    const result = mount.site.kind === "map" ? `${self}.output.${SCRIPT_CONTROL.returned}` : mount.site.whole !== undefined ? `${self}.output.${mount.site.whole}` : `${self}.output`;
    const rules: TransitionDecl[] = [
      { name: "then", to: mount.after.key, inputs: { [RESULT_INPUT]: result, ...Object.fromEntries([...mount.after.in].sort().map((v) => [v, `${pre}.${v}`])) } } as unknown as TransitionDecl,
    ];
    if (mount.site.handler !== undefined) {
      const handler = mount.from.composite.nodes.find((n) => n.key === `catch_${mount.site.handler!.id}`)!;
      rules.push({
        name: "catch",
        when: `${self}.outcome === 'error'`,
        to: handler.key,
        inputs: { [ERROR_INPUT]: `${self}.failure`, ...Object.fromEntries([...handler.in].sort().map((v) => [v, `${pre}.${v}`])) },
      } as unknown as TransitionDecl);
    }
    return rules;
  }

  // --- documents -------------------------------------------------------------------------------------

  private emitComposite(composite: Composite, parent: Composite | undefined): void {
    if (composite.kind === "element") composite.id = `${parent!.id}/${composite.key}`;
    const spec = this.spec;
    const children: Record<string, ChildDecl> = {};
    for (const node of composite.nodes) {
      children[node.key] = { transitions: this.rulesOf(node) } as ChildDecl;
      this.emitNode(node);
    }
    for (const mount of composite.mounts) {
      if (mount.element !== undefined) {
        if (mount.element.key === "") mount.element.key = mount.key;
        const element = mount.element;
        const from = this.prefixOf(mount.from);
        const inputs: Record<string, unknown> = {};
        for (const p of element.element!.params) inputs[p.name] = p.from === "item" ? { $expr: `${from}.${ITEMS_OUTPUT}`, each: true } : ".each.index";
        for (const v of this.captured(element)) inputs[v] = `${from}.${v}`;
        if (!this.documents[`${composite.id}/${element.key}`]) this.emitComposite(element, composite);
        children[mount.key] = {
          state: element.id,
          async: true,
          inputs,
          ...(mount.site.failureValue ? { failureValue: null } : {}),
          transitions: this.mountRules(mount),
        } as unknown as ChildDecl;
      } else {
        children[mount.key] = { state: mount.site.stateId!, called: true, transitions: this.mountRules(mount) } as ChildDecl;
      }
    }
    for (const phase of composite.phases) {
      const rules: TransitionDecl[] = [];
      for (const [next] of phase.leaves) {
        const target = composite.phases.find((p) => p.key === next);
        if (target === undefined) continue;
        rules.push({
          name: `to_${next}`,
          when: `.children.${phase.key}.output.${SCRIPT_CONTROL.next} === '${next}'`,
          to: next,
          inputs: { [SCRIPT_CONTROL.entry]: `.children.${phase.key}.output.${SCRIPT_CONTROL.entry}`, ...Object.fromEntries([...target.in].sort().map((v) => [v, `.children.${phase.key}.output.${v}`])) },
        } as unknown as TransitionDecl);
      }
      if (this.canReturn(phase)) rules.push({ name: "return", when: `.children.${phase.key}.output.${SCRIPT_CONTROL.next} === '${SCRIPT_RETURN}'`, to: "terminate.success" } as TransitionDecl);
      children[phase.key] = { transitions: rules } as ChildDecl;
      this.emitComposite(phase, composite);
    }

    // In the coarse modes a called state is entered by the code, not a rule: it is mounted on the root
    // only so the bundle holds it and the validator sees it.
    if (composite.kind === "root" && this.c.mode !== "calls") {
      for (const [key, stateId] of this.c.coarseCalls) if (children[key] === undefined) children[key] = { state: stateId, called: true } as ChildDecl;
    }
    const hasChildren = Object.keys(children).length > 0;
    const at: SourceAt = { line: composite.line, column: 1 };
    const doc: Record<string, unknown> = {};
    let outputs: Record<string, NamedParameterDecl>;
    let inputs: Record<string, ParameterDecl>;
    if (composite.kind === "root") {
      const returning = this.returners(composite);
      outputs = spec.rootOutputs((path) => this.chain(returning, `${SCRIPT_CONTROL.returned}${path}`), returning.length > 1 || returning[0]?.pre !== ".operation.output");
      inputs = spec.inputs;
      Object.assign(doc, rootProperties(spec.meta, spec.stateId));
    } else {
      doc.label = composite.label;
      if (composite.description !== undefined) doc.description = composite.description;
      inputs = composite.kind === "phase" ? { [SCRIPT_CONTROL.entry]: { schema: { type: "integer" } as JsonSchema, optional: true } } : {};
      // An element's parameters are fed one item of a list each, so they are typed by what they hold.
      const params = new Set(composite.element?.params.map((p) => p.name) ?? []);
      for (const p of params) inputs[p] = { optional: true };
      for (const v of [...composite.in].sort()) if (!params.has(v)) inputs[v] = this.slotOf(v);
      const leaving = this.leavers(composite);
      outputs = {};
      const fields = composite.kind === "phase" ? [SCRIPT_CONTROL.next, SCRIPT_CONTROL.entry, SCRIPT_CONTROL.returned, ...[...composite.out].sort()] : [SCRIPT_CONTROL.returned];
      for (const field of fields) {
        outputs[field] = { ...(this.hoistedSchema(field, composite) ?? {}), optional: true, binding: this.chain(leaving, field) } as NamedParameterDecl;
      }
    }
    Object.assign(doc, {
      inputs,
      ...(Object.keys(outputs).length > 0 ? { outputs } : {}),
      operation: { script: { code: this.module(composite.head), file: this.c.scriptFile }, output: this.continuationOutput(composite.head.out, composite) },
      ...(hasChildren ? { children, sequence: [], transitions: this.rulesOf(composite.head) } : {}),
      generated: { ...spec.generated, at, ...(composite.kind === "root" && this.c.returnsWhole ? { whole: "result" } : {}) },
    });
    if (composite.kind === "root") {
      if (spec.meta.environment !== undefined) doc.environment = spec.meta.environment;
      if (spec.meta.limits !== undefined) doc.limits = spec.meta.limits;
    }
    this.documents[composite.id] = doc as StateDef;
    this.sourceMap[composite.id] = at;
  }

  /** A child state: a call — the call, then the code after it — or the code after a mount, or a catch. */
  private emitNode(node: MNode): void {
    const id = `${node.composite.id}/${node.key}`;
    const at: SourceAt = { line: node.line, column: 1 };
    const inputs: Record<string, ParameterDecl> = {};
    if (node.kind === "call") inputs[CALL_INPUT] = { optional: true, description: "The call's arguments." };
    if (node.kind === "after") inputs[RESULT_INPUT] = { optional: true, description: "What the call returned." };
    if (node.kind === "catch") inputs[ERROR_INPUT] = { optional: true, description: "Why the call failed." };
    for (const v of [...node.in].sort()) inputs[v] = this.slotOf(v);
    const tail = { script: { code: this.module(node), file: this.c.scriptFile }, output: this.continuationOutput(node.out, node.composite) };
    let doc: Record<string, unknown>;
    if (node.kind === "call") {
      const site = node.site!;
      const call =
        site.kind === "prompt"
          ? { ...site.op }
          : { script: { code: this.callModule(site), file: this.c.scriptFile }, output: { value: { kind: "json", ...(site.resultSchema !== undefined && !hasDefs(site.resultSchema) ? { schema: withoutUnions(site.resultSchema as JsonValue) } : {}) } } };
      // A prompt operation's object result also carries the engine's `session` (SPEC §6.1) — not the
      // call's answer, so the code is handed the answer without it, unless the call asked for one.
      const declaresSession = JSON.stringify(site.resultSchema ?? {}).includes('"session"');
      const object = (site.resultSchema as { type?: unknown } | undefined)?.type === "object";
      const result =
        site.kind === "prompt" && site.unwrap
          ? ".operation[0].output.value"
          : site.kind === "prompt" && object && !declaresSession
            ? { $expr: "omit(.operation[0].output, ['session'])" }
            : ".operation[0].output";
      const outputs: Record<string, NamedParameterDecl> = {};
      for (const field of [SCRIPT_CONTROL.next, SCRIPT_CONTROL.entry, SCRIPT_CONTROL.returned, CALL_INPUT, ITEMS_OUTPUT, ...[...node.out].sort()]) {
        outputs[field] = { ...(this.hoistedSchema(field, node.composite) ?? {}), ...(field === SCRIPT_CONTROL.next ? {} : { optional: true }), binding: `.operation[1].output.${field}` } as NamedParameterDecl;
      }
      doc = { label: node.label, inputs, outputs, operation: [call, { ...tail, input: { [RESULT_INPUT]: { binding: result } } }] };
    } else {
      doc = { label: node.label, inputs, outputs: this.nodeOutputs(node.out, node.composite), operation: tail };
    }
    doc.generated = { ...this.spec.generated, at };
    this.documents[id] = doc as StateDef;
    this.sourceMap[id] = at;
  }

  private nodeOutputs(vars: ReadonlySet<string>, composite: Composite): Record<string, NamedParameterDecl> {
    const outputs: Record<string, NamedParameterDecl> = {
      [SCRIPT_CONTROL.next]: { schema: { type: "string" } as JsonSchema, description: "Where this state hands on to, or '$return'." },
      [SCRIPT_CONTROL.entry]: { schema: { type: "integer" } as JsonSchema, optional: true },
      [SCRIPT_CONTROL.returned]: { ...(this.returnSchema(composite) ?? {}), optional: true, description: "What the script returned, when it returned here." },
      [CALL_INPUT]: { optional: true, description: "The arguments of the call this state hands on to." },
      [ITEMS_OUTPUT]: { schema: ITEMS_SCHEMA, optional: true, description: "The items of the fan-out this state hands on to." },
    };
    for (const v of [...vars].sort()) outputs[v] = this.slotOf(v);
    return outputs;
  }

  /** What `_return` carries: the script's return type — or, in a fan-out's element, whatever the element returns. */
  private returnSchema(composite: Composite): { schema: JsonSchema } | undefined {
    if (composite.kind === "element") return undefined;
    const returns = this.spec.returns;
    if (returns === undefined) return undefined;
    const widened = wireForm(returns);
    return Object.keys(widened).length > 0 ? { schema: widened } : undefined;
  }

  private hoistedSchema(field: string, composite: Composite): { schema: JsonSchema } | undefined {
    if (field === SCRIPT_CONTROL.next) return { schema: { type: "string" } as JsonSchema };
    if (field === SCRIPT_CONTROL.entry) return { schema: { type: "integer" } as JsonSchema };
    if (field === SCRIPT_CONTROL.returned) return this.returnSchema(composite);
    if (field === ITEMS_OUTPUT) return { schema: ITEMS_SCHEMA };
    if (CONTROL_NAMES.has(field)) return undefined;
    const schema = this.wireSchemaOf(field);
    return schema !== undefined ? { schema } : undefined;
  }

  private slotOf(v: string): ParameterDecl {
    const schema = this.wireSchemaOf(v);
    return { ...(schema !== undefined ? { schema } : {}), optional: true };
  }

  /**
   * A variable's type as it travels between states. A UNION inside it is widened to "anything": the
   * wiring checker does not reason about unions yet (§6.2), and both ends of every wire the compiler
   * makes are the same variable, so nothing is lost that the checker could have used.
   */
  private wireSchemaOf(v: string): JsonSchema | undefined {
    const schema = this.c.hoistedByName.get(v)?.schema;
    if (schema === undefined || hasDefs(schema)) return undefined;
    const widened = wireForm(schema);
    return Object.keys(widened).length > 0 ? widened : undefined;
  }

  /** The continuation a state's code returns, typed — what `.operation.output` reads (SCRIPTS.md §6.2). */
  private continuationOutput(vars: ReadonlySet<string>, composite: Composite): Record<string, ParameterDecl> {
    const properties: Record<string, JsonValue> = {
      [SCRIPT_CONTROL.next]: { type: "string" },
      [SCRIPT_CONTROL.entry]: { type: "integer" },
      [SCRIPT_CONTROL.returned]: (this.returnSchema(composite)?.schema ?? {}) as JsonValue,
      [CALL_INPUT]: {},
      [ITEMS_OUTPUT]: ITEMS_SCHEMA as JsonValue,
    };
    for (const v of [...vars].sort()) properties[v] = (this.wireSchemaOf(v) ?? {}) as JsonValue;
    return { continuation: { kind: "json", schema: { type: "object", properties, required: [SCRIPT_CONTROL.next] } as unknown as JsonSchema } };
  }

  /** The states of a composite that can end it — and the `_next` values they end it with. */
  private leavers(composite: Composite): Array<{ pre: string; nexts: string[] }> {
    const out: Array<{ pre: string; nexts: string[] }> = [];
    for (const node of [composite.head, ...composite.nodes]) {
      const nexts = node.exits.filter((e): e is Extract<MExit, { kind: "leave" }> => e.kind === "leave").map((e) => e.next);
      if (nexts.length > 0) out.push({ pre: this.prefixOf(node), nexts });
    }
    return out;
  }

  /** The states — and, for the root, the phases — that can return from the script. */
  private returners(root: Composite): Array<{ pre: string; nexts: string[] }> {
    const out = this.leavers(root).filter((l) => l.nexts.includes(SCRIPT_RETURN)).map((l) => ({ pre: l.pre, nexts: [SCRIPT_RETURN] }));
    for (const phase of root.phases) if (this.canReturn(phase)) out.push({ pre: `.children.${phase.key}.output`, nexts: [SCRIPT_RETURN] });
    return out;
  }

  private canReturn(composite: Composite): boolean {
    return [composite.head, ...composite.nodes].some((n) => n.exits.some((e) => e.kind === "leave" && e.next === SCRIPT_RETURN));
  }

  /** A field of whichever state ended a composite: the one whose `_next` says it ended it. */
  private chain(candidates: Array<{ pre: string; nexts: string[] }>, field: string): string | { $expr: string } {
    const nested = field.includes(".");
    if (candidates.length === 0) return nested ? { $expr: `.operation.output.${field}` } : `.operation.output.${field}`;
    if (candidates.length === 1) return nested ? { $expr: `${candidates[0]!.pre}.${field}` } : `${candidates[0]!.pre}.${field}`;
    let expr = `${candidates[candidates.length - 1]!.pre}.${field}`;
    for (const candidate of candidates.slice(0, -1).reverse()) {
      const cond = candidate.nexts.map((n) => `${candidate.pre}.${SCRIPT_CONTROL.next} === '${n}'`).join(" || ");
      expr = `(${cond}) ? ${candidate.pre}.${field} : ${expr}`;
    }
    return { $expr: expr };
  }

  // --- code ------------------------------------------------------------------------------------------

  private print(node: TS.Node): string {
    return this.printer.printNode(this.ts.EmitHint.Unspecified, node, this.c.sf);
  }

  /** The module preamble every segment shares: the imports, the hooks, what the script imported, its static code. */
  private preamble(heading: string): string[] {
    const lines: string[] = [`// Generated from ${this.c.scriptFile} — ${heading}. Edit the script, not this.`];
    for (const i of this.spec.imports) lines.push(this.print(i));
    lines.push(`import * as $rt from ${JSON.stringify(HOOK_MODULE)};`);
    lines.push(...this.c.importStubs());
    for (const s of this.spec.prelude) lines.push(this.print(this.c.rewrite(s, false).node));
    return lines;
  }

  /** The code a state runs: its region's blocks as a `switch`, entered at its entry. */
  private module(node: MNode): string {
    const { ts } = this;
    const f = ts.factory;
    const composite = node.composite;
    const lines = this.preamble(node.kind === "head" ? `${composite.kind} '${composite.label}'` : `'${node.label}'`);
    const entry = node.entries.length === 1 ? ` = ${node.entries[0]}` : "";
    lines.push(`export default async function $segment($v: any, $pc: number${entry}): Promise<any> {`);
    for (const h of this.c.helperStatements) lines.push(this.print(h));
    lines.push(`  let $ln = ${node.line};`);
    if (node.kind === "catch" && node.handler?.param !== undefined) {
      lines.push(`  $v.${node.handler.param} = Object.assign(new Error(String($v._error?.reason ?? "the call failed")), { classification: $v._error?.classification });`);
    }
    lines.push("  try {");
    lines.push("  $dispatch: for (;;) {");
    lines.push("    switch ($pc) {");
    for (const id of [...node.region].sort((a, b) => a - b)) {
      const block = this.c.blocks[id]!;
      lines.push(`      case ${id}: {`);
      const guarded = block.handler !== undefined && node.region.has(block.handler.block);
      if (guarded) lines.push("      try {");
      for (const { statement, line } of block.code) {
        lines.push(`        $ln = ${line};`);
        lines.push(this.print(statement));
      }
      for (const s of this.termCode(node, block, f)) lines.push(this.print(s));
      if (guarded) {
        const handler = block.handler!;
        lines.push(`      } catch ($err) { ${handler.param !== undefined ? `$v.${handler.param} = $err; ` : ""}$pc = ${handler.block}; continue $dispatch; }`);
      }
      lines.push("      }");
    }
    lines.push(`      default: throw new Error("'${node.label.replace(/["\\]/g, "")}' has no entry " + $pc);`);
    lines.push("    }");
    lines.push("  }");
    lines.push(`  } catch ($e: any) { if ($e && typeof $e === "object" && typeof $e.message === "string" && !$e.$located) { $e.message = ${JSON.stringify(`${this.c.scriptFile}:`)} + $ln + ": " + $e.message; $e.$located = true; } throw $e; }`);
    lines.push("}");
    return lines.join("\n");
  }

  private termCode(node: MNode, block: Block, f: TS.NodeFactory): TS.Statement[] {
    const { ts } = this;
    const composite = node.composite;
    const term = block.term;
    const leave = (next: string, extra: TS.ObjectLiteralElementLike[]): TS.Statement[] => [
      f.createReturnStatement(f.createObjectLiteralExpression([f.createSpreadAssignment(f.createIdentifier("$v")), f.createPropertyAssignment(SCRIPT_CONTROL.next, f.createStringLiteral(next)), ...extra])),
    ];
    if (term === undefined || term.kind === "return") {
      const value = term?.kind === "return" ? (term.value !== undefined ? (this.c.rewrite(f.createExpressionStatement(term.value), false).node as TS.ExpressionStatement).expression : term.fallthrough ? this.c.fallthroughValue(f, block.machine) : undefined) : undefined;
      return leave(SCRIPT_RETURN, value !== undefined ? [f.createPropertyAssignment(SCRIPT_CONTROL.returned, value)] : []);
    }
    if (term.kind === "goto") return gotoStatements(f, term.to);
    if (term.kind === "branch") {
      const cond = (this.c.rewrite(f.createExpressionStatement(term.cond), false).node as TS.ExpressionStatement).expression;
      return [
        f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createConditionalExpression(cond, undefined, f.createNumericLiteral(term.then), undefined, f.createNumericLiteral(term.else)))),
        f.createContinueStatement(f.createIdentifier("$dispatch")),
      ];
    }
    if (term.kind === "phase") {
      if (term.phase === composite.region) return gotoStatements(f, term.to);
      return leave(this.phaseKeys.get(term.phase)!, [f.createPropertyAssignment(SCRIPT_CONTROL.entry, f.createNumericLiteral(term.to))]);
    }
    const site = this.c.sites[term.site]!;
    const args = (this.c.rewrite(f.createExpressionStatement(site.call), false).node as TS.ExpressionStatement).expression;
    const key = site.kind === "prompt" || site.kind === "script" ? `call_${site.id}` : composite.mounts.find((m) => m.site === site && m.from === node)!.key;
    void ts;
    // A fan-out hands on its items as a list of their own, which the `each` wire reads.
    if (site.kind === "map") return leave(key, [f.createPropertyAssignment(ITEMS_OUTPUT, f.createPropertyAccessExpression(f.createParenthesizedExpression(args), "items"))]);
    return leave(key, [f.createPropertyAssignment(CALL_INPUT, args)]);
  }

  /** The code a call state's CALL runs, when the call is not a prompt operation: the one call, over `_call`. */
  private callModule(site: CallSite): string {
    const lines = this.preamble(`the call '${site.label}'`);
    lines.push(`export default async function $segment($v: any): Promise<any> {`);
    lines.push(`  return await (${site.script});`);
    lines.push("}");
    return lines.join("\n");
  }
}

// --- helpers --------------------------------------------------------------------------------------------

function vRef(ts: typeof TS, name: string): TS.PropertyAccessExpression {
  return ts.factory.createPropertyAccessExpression(ts.factory.createIdentifier("$v"), name);
}

function hookRef(ts: typeof TS, name: string): TS.PropertyAccessExpression {
  return ts.factory.createPropertyAccessExpression(ts.factory.createIdentifier("$rt"), name);
}

function assign(ts: typeof TS, name: string, value: TS.Expression): TS.Statement {
  return ts.factory.createExpressionStatement(ts.factory.createAssignment(vRef(ts, name), value));
}

function gotoStatements(f: TS.NodeFactory, to: number): TS.Statement[] {
  return [f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createNumericLiteral(to))), f.createContinueStatement(f.createIdentifier("$dispatch"))];
}

/** `{ ...$v, _next, _entry?, _return? }` — how a state's code ends it (SCRIPTS.md §6.2). */
function continuation(f: TS.NodeFactory, next: string, entry: number | undefined, value: TS.Expression | undefined): TS.Expression {
  const properties: TS.ObjectLiteralElementLike[] = [f.createSpreadAssignment(f.createIdentifier("$v"))];
  properties.push(f.createPropertyAssignment(SCRIPT_CONTROL.next, f.createStringLiteral(next)));
  if (entry !== undefined) properties.push(f.createPropertyAssignment(SCRIPT_CONTROL.entry, f.createNumericLiteral(entry)));
  if (value !== undefined) properties.push(f.createPropertyAssignment(SCRIPT_CONTROL.returned, value));
  return f.createObjectLiteralExpression(properties);
}

function jsonExpression(f: TS.NodeFactory, value: JsonValue): TS.Expression {
  if (value === null) return f.createNull();
  if (typeof value === "string") return f.createStringLiteral(value);
  if (typeof value === "number") return value < 0 ? f.createPrefixUnaryExpression(40 /* MinusToken */, f.createNumericLiteral(-value)) : f.createNumericLiteral(value);
  if (typeof value === "boolean") return value ? f.createTrue() : f.createFalse();
  if (Array.isArray(value)) return f.createArrayLiteralExpression(value.map((v) => jsonExpression(f, v)));
  return f.createObjectLiteralExpression(Object.entries(value).map(([k, v]) => f.createPropertyAssignment(f.createStringLiteral(k), jsonExpression(f, v as JsonValue))));
}

/**
 * A type as a WIRE between generated states carries it: unions and tuples widened, since the wiring
 * checker models neither yet (§6.2). Both ends of every generated wire are the same value, so nothing
 * is lost the checker could have used.
 */
function wireForm(schema: JsonSchema): JsonSchema {
  const widen = (value: JsonValue): JsonValue => {
    if (Array.isArray(value)) return value.map(widen);
    if (value === null || typeof value !== "object") return value;
    if ("anyOf" in value || "oneOf" in value || "allOf" in value) return {};
    const out: Record<string, JsonValue> = {};
    for (const [k, v] of Object.entries(value)) {
      if (k === "minItems" || k === "maxItems") continue;
      out[k] = k === "items" && Array.isArray(v) ? {} : widen(v as JsonValue);
    }
    return out;
  };
  return widen(schema as JsonValue) as JsonSchema;
}

/** A schema with every union widened to the universal schema — see `wireSchemaOf`. */
function withoutUnions(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(withoutUnions);
  if (value === null || typeof value !== "object") return value;
  if ("anyOf" in value || "oneOf" in value || "allOf" in value) return {};
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, withoutUnions(v as JsonValue)]));
}

function hasDefs(schema: JsonSchema): boolean {
  return JSON.stringify(schema).includes("\"$ref\"");
}

function stripDefs(schema: JsonSchema): JsonValue {
  const { $defs: _d, ...rest } = schema as Record<string, JsonValue>;
  return rest;
}

function metaDeclaration(ts: typeof TS, s: TS.Statement): TS.VariableDeclaration | undefined {
  if (!ts.isVariableStatement(s) || !hasModifier(ts, s, ts.SyntaxKind.ExportKeyword)) return undefined;
  if ((s.declarationList.flags & ts.NodeFlags.Const) === 0) return undefined;
  const d = s.declarationList.declarations[0];
  return d !== undefined && ts.isIdentifier(d.name) && d.name.text === "meta" ? d : undefined;
}

function hasModifier(ts: typeof TS, node: TS.Node, kind: TS.SyntaxKind): boolean {
  return ts.canHaveModifiers(node) && (ts.getModifiers(node) ?? []).some((m) => m.kind === kind);
}

function unwrap(ts: typeof TS, e: TS.Expression): TS.Expression {
  while (ts.isParenthesizedExpression(e) || ts.isAsExpression(e) || ts.isSatisfiesExpression(e) || ts.isNonNullExpression(e)) e = e.expression;
  return e;
}

function isTypeOnly(ts: typeof TS, s: TS.Statement): boolean {
  return ts.isInterfaceDeclaration(s) || ts.isTypeAliasDeclaration(s) || (ts.isModuleDeclaration(s) && hasModifier(ts, s, ts.SyntaxKind.DeclareKeyword));
}

function isFunctionBoundary(ts: typeof TS, n: TS.Node): boolean {
  return ts.isFunctionDeclaration(n) || ts.isFunctionExpression(n) || ts.isArrowFunction(n) || ts.isMethodDeclaration(n) || ts.isGetAccessor(n) || ts.isSetAccessor(n) || ts.isConstructorDeclaration(n) || ts.isClassDeclaration(n) || ts.isClassExpression(n);
}

function bindingIdentifiers(ts: typeof TS, name: TS.BindingName): TS.Identifier[] {
  if (ts.isIdentifier(name)) return [name];
  const out: TS.Identifier[] = [];
  for (const e of name.elements) if (!ts.isOmittedExpression(e)) out.push(...bindingIdentifiers(ts, e.name));
  return out;
}

function boundNames(ts: typeof TS, name: TS.BindingName): string[] {
  return bindingIdentifiers(ts, name).map((i) => i.text);
}

function scriptKindOf(ts: typeof TS, file: string): TS.ScriptKind {
  return file.toLowerCase().endsWith(".js") ? ts.ScriptKind.JS : ts.ScriptKind.TS;
}

/** A value the compiler may read twice without holding it: a literal, or its own temporary. */
function isStable(ts: typeof TS, e: TS.Expression): boolean {
  const u = unwrap(ts, e);
  return ts.isStringLiteral(u) || ts.isNumericLiteral(u) || ts.isNoSubstitutionTemplateLiteral(u) || u.kind === ts.SyntaxKind.TrueKeyword || u.kind === ts.SyntaxKind.FalseKeyword || u.kind === ts.SyntaxKind.NullKeyword || isTempRef(ts, u);
}

function isTempRef(ts: typeof TS, e: TS.Expression): boolean {
  return ts.isPropertyAccessExpression(e) && ts.isIdentifier(e.expression) && e.expression.text === "$v" && e.pos < 0;
}

function compoundBase(ts: typeof TS, op: TS.SyntaxKind): TS.BinaryOperator {
  const K = ts.SyntaxKind;
  const table: Partial<Record<number, TS.BinaryOperator>> = {
    [K.PlusEqualsToken]: K.PlusToken,
    [K.MinusEqualsToken]: K.MinusToken,
    [K.AsteriskEqualsToken]: K.AsteriskToken,
    [K.AsteriskAsteriskEqualsToken]: K.AsteriskAsteriskToken,
    [K.SlashEqualsToken]: K.SlashToken,
    [K.PercentEqualsToken]: K.PercentToken,
    [K.LessThanLessThanEqualsToken]: K.LessThanLessThanToken,
    [K.GreaterThanGreaterThanEqualsToken]: K.GreaterThanGreaterThanToken,
    [K.GreaterThanGreaterThanGreaterThanEqualsToken]: K.GreaterThanGreaterThanGreaterThanToken,
    [K.AmpersandEqualsToken]: K.AmpersandToken,
    [K.BarEqualsToken]: K.BarToken,
    [K.CaretEqualsToken]: K.CaretToken,
  };
  return table[op] ?? K.PlusToken;
}

/** The literal text a prompt starts with, when the compiler can read it — for a call state's label. */
function promptText(ts: typeof TS, e: TS.Expression): string | undefined {
  const u = unwrap(ts, e);
  if (ts.isStringLiteral(u) || ts.isNoSubstitutionTemplateLiteral(u)) return u.text;
  if (ts.isTemplateExpression(u)) return `${u.head.text}${u.templateSpans.map((s) => `{${s.expression.getText?.() ?? "…"}}${s.literal.text}`).join("")}`;
  if (ts.isBinaryExpression(u) && u.operatorToken.kind === ts.SyntaxKind.PlusToken) return promptText(ts, u.left);
  return undefined;
}

/** `with { type: "json" }` and `with { as: "operation" }` — an import's attributes, by key. */
function importAttributes(ts: typeof TS, declaration: TS.ImportDeclaration): Record<string, string> {
  const out: Record<string, string> = {};
  const attributes = (declaration as { attributes?: TS.ImportAttributes }).attributes;
  for (const element of attributes?.elements ?? []) {
    const key = ts.isIdentifier(element.name) ? element.name.text : element.name.text;
    if (ts.isStringLiteral(element.value)) out[key] = element.value.text;
  }
  return out;
}

/** The local names an import clause binds, each with the name it imports. */
function importedLocals(ts: typeof TS, clause: TS.ImportClause): Array<{ local: string; imported?: string }> {
  const out: Array<{ local: string; imported?: string }> = [];
  if (clause.name !== undefined) out.push({ local: clause.name.text, imported: "default" });
  const bindings = clause.namedBindings;
  if (bindings !== undefined && ts.isNamedImports(bindings)) {
    for (const element of bindings.elements) out.push({ local: element.name.text, imported: element.propertyName?.text ?? element.name.text });
  }
  return out;
}

/** A helper's body with its `return`s turned into a jump out of the labeled block it is inlined as. */
function replaceReturns(ts: typeof TS, statement: TS.Statement, ret: string, label: string): TS.Statement {
  const transformer: TS.TransformerFactory<TS.Node> = (context) => {
    const f = context.factory;
    const visit = (n: TS.Node): TS.Node => {
      if (isFunctionBoundary(ts, n)) return n;
      if (ts.isReturnStatement(n)) return f.createBlock([...(n.expression !== undefined ? [assign(ts, ret, n.expression)] : []), f.createBreakStatement(label)], true);
      return ts.visitEachChild(n, visit, context);
    };
    return (root) => visit(root);
  };
  const result = ts.transform(statement, [transformer]);
  const out = result.transformed[0]! as TS.Statement;
  result.dispose();
  return out;
}

/** A generic helper with a schema parameter per type parameter its `llm<T>` needs (SCRIPTS.md §9). */
function withSchemaParams(ts: typeof TS, fn: TS.Node, names: readonly string[]): TS.Node {
  const f = ts.factory;
  const extra = names.map((n) => f.createParameterDeclaration(undefined, undefined, `$schema_${n}`, f.createToken(ts.SyntaxKind.QuestionToken), f.createKeywordTypeNode(ts.SyntaxKind.AnyKeyword)));
  if (ts.isFunctionDeclaration(fn)) return f.updateFunctionDeclaration(fn, fn.modifiers, fn.asteriskToken, fn.name, fn.typeParameters, [...fn.parameters, ...extra], fn.type, fn.body);
  if (ts.isArrowFunction(fn)) return f.updateArrowFunction(fn, fn.modifiers, fn.typeParameters, [...fn.parameters, ...extra], fn.type, fn.equalsGreaterThanToken, fn.body);
  if (ts.isFunctionExpression(fn)) return f.updateFunctionExpression(fn, fn.modifiers, fn.asteriskToken, fn.name, fn.typeParameters, [...fn.parameters, ...extra], fn.type, fn.body);
  return fn;
}

function sanitizeKey(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "state";
}

function lineOf(sf: TS.SourceFile, node: TS.Node): number {
  return node.pos < 0 ? 1 : sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
}

/** POSIX separators, `.` dropped, `..` applied. */
function normalizePath(path: string): string {
  const absolute = path.startsWith("/");
  const drive = /^[a-zA-Z]:/.exec(path)?.[0];
  const out: string[] = [];
  for (const segment of (drive !== undefined ? path.slice(2) : path).split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") out.pop();
    else out.push(segment);
  }
  const joined = out.join("/");
  return drive !== undefined ? `${drive}/${joined}` : absolute ? `/${joined}` : joined;
}

/** The state properties `meta` sets — `name` read as `label` (SCRIPTS.md §4). */
function rootProperties(meta: ScriptMeta, stateId: string): Record<string, unknown> {
  const label = meta.name ?? meta.label ?? stateId.split("/").pop()!;
  return {
    label,
    ...(meta.description !== undefined ? { description: meta.description } : {}),
    ...(meta.whenToUse !== undefined ? { whenToUse: meta.whenToUse } : {}),
    ...(meta.title !== undefined ? { title: meta.title } : {}),
  };
}

/** Where a segment's code is prepared: beside the script, so its relative imports resolve as written. */
export function segmentPath(file: string, code: string): string {
  const base = file.replace(/\\/g, "/").split("/").pop()!.replace(/\.[^.]+$/, "");
  return `${dirOf(file.replace(/\\/g, "/"))}/.${base}.${sha256Hex(code).slice(0, 12)}.segment.ts`;
}

// --- the generated files ------------------------------------------------------------------------------

/**
 * The compiled states as FILES — `<state id>.json` beside the script, as a host writes them
 * (SCRIPTS.md §12), each carrying its provenance. Paths are relative to the workflow root.
 */
export function generatedFiles(compiled: CompiledScript): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [id, document] of Object.entries(compiled.documents)) out[`${id}.json`] = `${JSON.stringify(document, null, 2)}\n`;
  return out;
}

/**
 * Why a generated document no longer matches the script it came from — a file the compile read has
 * changed — or `undefined` when it still does (SCRIPTS.md §12). The script is the source: a stale or
 * hand-edited generated file is an error that says to edit the script, never a silent pick of one.
 */
export function staleGenerated(document: StateDef, current: CompiledScript): string | undefined {
  const recorded = document.generated?.inputs ?? {};
  const now = current.generated.inputs;
  for (const file of new Set([...Object.keys(recorded), ...Object.keys(now)])) {
    if (recorded[file] !== now[file]) {
      return `'${file}' changed since this state was generated from '${current.generated.from}' — edit the script and regenerate, rather than this file`;
    }
  }
  return undefined;
}
