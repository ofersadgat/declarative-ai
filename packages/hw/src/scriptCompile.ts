/**
 * Compiling a workflow script to hw states (SCRIPTS.md).
 *
 * A script is a state written as code. This module reads it and produces the STATE DOCUMENTS a person
 * could have written by hand — plain `StateDef`s that `loadBundle` loads like any other — so nothing
 * downstream (the loader, the validator, the engine, the journal, the board) learns a new format. The
 * one thing the documents carry that authored JSON rarely does is `operation.script`: the code a state
 * runs, which the engine executes as a recorded operation (`$script`, `engine.ts`).
 *
 * ## The shape of the compile
 *
 *  1. **`meta`** is read as a pure literal: the state's standard properties (SCRIPTS.md §4).
 *  2. **The signature** comes from the exported function, or is `args` in / the exports out when the
 *     body is the function (§5).
 *  3. **The body is cut at its `phase()` statements** (§6). The function becomes a set of basic
 *     blocks; each `phase("X")` ends one, and a forward pass over the blocks works out which phases
 *     can be current at each. A phase's STATE is every block that can run while it is current.
 *  4. **Variables that live across a cut** move into one record, `$v`, which is how a value survives
 *     the state boundary: each phase state takes the variables it needs as inputs and hands on, as
 *     outputs, the ones the phases after it need.
 *  5. **Each exit is a transition.** A phase state ends by returning a continuation — which phase
 *     next, at which entry, with which variables — and its mount's rules route on it.
 *
 * Inside one phase the code runs as it was written: its `llm()` calls are made through the engine one
 * at a time, each recorded at a site of its own (§11), so a restart part-way through a phase replays
 * the calls it already made.
 *
 * ## What a cut costs, and why it is restricted
 *
 * JavaScript's block scoping does not survive being cut into `switch` cases, so a variable declared
 * DIRECTLY in a structure that contains a `phase()` — the function body, a loop body that changes
 * phase — is hoisted into `$v`. Everything inside a statement that contains no `phase()` is left as
 * written, scoping and all. A variable declared in a hoisted position and captured by a closure that
 * outlives one loop iteration sees the variable's latest value rather than the iteration's; that is
 * the one semantic difference hoisting makes, and it only arises for a loop that changes phase.
 */
import { sha256Hex } from "@declarative-ai/exec";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type * as TS from "typescript";
import {
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
import { DATA_EXTENSIONS } from "./reference.js";
import { HOOK_GLOBALS_PATH, HOOK_MODULE, HOOK_MODULE_PATH } from "./scriptHooks.js";
import { createScriptProgram, loadSignatureContext, readSignature, type SignatureContext } from "./signature.js";
import { typeToWireSchema, typeToWireSlot, WireTypeError } from "./wireType.js";

/** How a script becomes something hw runs (SCRIPTS.md §3). */
export type ScriptMode = "states" | "state" | "function";
export const SCRIPT_MODES: readonly ScriptMode[] = ["states", "state", "function"];

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
  children: "a script's children are its phases and the states it calls",
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

export interface CompileScriptOptions extends ModuleResolveOptions {
  /** The script's path — where its imports resolve from, and what the documents record as their source. */
  file: string;
  /** The id of the state the script IS. Phases compile to `<stateId>/<key>`. */
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

/** The root "phase" — the code that runs before any `phase()` call, in the state the script IS. */
const ROOT = "\u0000root";

type Terminator =
  | { kind: "goto"; to: number }
  | { kind: "branch"; cond: TS.Expression; then: number; else: number }
  | { kind: "phase"; phase: string; to: number }
  | { kind: "return"; value: TS.Expression | undefined; fallthrough: boolean };

interface Block {
  id: number;
  /** Rewritten statements, in order. */
  code: TS.Statement[];
  term?: Terminator;
  /** Jumps a rewritten `break`/`continue` inside `code` makes — edges the terminator does not show. */
  jumps: number[];
  /** Hoisted variables this block reads before it writes them — its upward-exposed uses. */
  uses: Set<string>;
  /** Hoisted variables this block assigns outright. */
  defs: Set<string>;
}

interface LoweredLoop {
  breakTo: number;
  continueTo: number;
  labels: readonly string[];
}

interface Hoisted {
  name: string;
  schema?: JsonSchema;
  /** Why the variable has no wire form, when it has none — an error only if it crosses a phase. */
  unrepresentable?: string;
}

class Compiler {
  private readonly ts: typeof TS;
  private readonly file: string;
  private readonly source: string;
  private sf!: TS.SourceFile;
  private checker!: TS.TypeChecker;
  private program!: TS.Program;
  private readonly warnings: string[] = [];
  private mode: ScriptMode = "states";

  /** Hoisted variables by symbol, and by the unique name each lives under in `$v`. */
  private readonly hoisted = new Map<TS.Symbol, Hoisted>();
  private readonly hoistedByName = new Map<string, Hoisted>();
  /** Synthetic variables a lowered `for…of` keeps its place in. */
  private synthetic = 0;
  /** Helpers — function declarations and static constants — re-declared at the top of every segment. */
  private readonly helpers: TS.Statement[] = [];
  /** `const` declarations the pre-pass decided are helpers rather than variables. */
  private readonly helperDeclarations = new Set<TS.VariableDeclaration>();
  private readonly helperRefs = new Set<string>();

  private readonly blocks: Block[] = [];
  private cur!: Block;
  private readonly loops: LoweredLoop[] = [];
  private pendingLabels: string[] = [];

  /** Body mode: what the fallthrough returns — named exports, or the default export's variable. */
  private readonly exportNames: string[] = [];
  /** The function's WHOLE return is the one `result` output — what a caller of the state gets back. */
  private wholeReturn = false;
  private defaultExport: string | undefined;
  private sawReturn = false;
  private bodyMode = false;
  /** The workflow root the script's own id hangs off — where a called state's id is read from. */
  private readonly root: string;
  /** States the script calls — by import or by `workflow("…")` — as the mounts they become, by key. */
  private readonly called = new Map<string, string>();
  /** Imported states, by the local name the import binds. */
  private readonly stateImports = new Map<string, string>();

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
    const imports = statements.filter((s): s is TS.ImportDeclaration => ts.isImportDeclaration(s) && !this.importsState(s));
    const rest = statements.filter((s) => !ts.isImportDeclaration(s));

    // `meta`, when there is one, comes first — Claude's rule, and what makes a script recognizable
    // without running anything.
    let meta: ScriptMeta = {};
    // Types have no run-time effect, so they may sit above it too.
    const metaIndex = rest.findIndex((s) => metaDeclaration(ts, s) !== undefined);
    if (metaIndex >= 0 && rest.slice(0, metaIndex).some((s) => !isTypeOnly(ts, s))) {
      throw this.error(rest[metaIndex]!, "`export const meta` must be the first statement after the imports (and any types)");
    }
    if (metaIndex >= 0) meta = this.readMeta(metaDeclaration(ts, rest[metaIndex]!)!);
    const body = metaIndex >= 0 ? rest.filter((_, i) => i !== metaIndex) : rest;

    this.mode = meta.compile ?? this.options.defaultMode ?? "states";
    const generated = this.provenance();
    if (this.mode === "function") return { mode: this.mode, meta, documents: {}, warnings: this.warnings, generated };

    const entry = this.entryFunction(body);
    this.bodyMode = entry === undefined;
    const prelude: TS.Statement[] = [];
    let bodyStatements: readonly TS.Statement[];
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
      bodyStatements = this.entryBody(entry.fn);
    } else {
      // The body is the function: its one input is Claude's `args` (SCRIPTS.md §5).
      for (const s of body) if (isTypeOnly(ts, s)) prelude.push(s);
      bodyStatements = body.filter((s) => !isTypeOnly(ts, s));
      inputs = { args: { optional: true, description: "What the caller passed — Claude's `args`." } };
      const argsSymbol = this.globalSymbol("args");
      if (argsSymbol !== undefined) this.hoistNamed(argsSymbol, "args", undefined);
    }

    this.checkPhaseCalls([...bodyStatements, ...prelude], entry?.fn);
    // Every hoisted variable is decided BEFORE anything is rewritten, so a helper declared above a
    // variable it reads still reads it through `$v`.
    for (const s of bodyStatements) this.collectHoisting(s);

    // Lower the body into blocks.
    this.cur = this.newBlock();
    this.lowerList(bodyStatements);
    this.end({ kind: "return", value: undefined, fallthrough: true });

    if (this.bodyMode && this.sawReturn && (this.exportNames.length > 0 || this.defaultExport !== undefined)) {
      throw new ScriptCompileError("a script returns a value OR exports its outputs — it does both", this.file);
    }

    const phaseSets = this.phaseSets();
    const phases = this.phaseOrder(phaseSets);
    const keys = this.phaseKeys(phases);
    const graph = this.phaseGraph(phaseSets);
    const live = this.liveness(phaseSets, graph, phases);

    const outputs = this.rootOutputs(returns, phases, keys);
    const generatedHere = { ...generated };
    const documents: Record<string, StateDef> = {};
    const root: StateDef = {
      ...stateProperties(meta, this.options.stateId),
      inputs,
      ...(Object.keys(outputs).length > 0 ? { outputs } : {}),
      operation: { script: { code: this.segmentModule(ROOT, phaseSets, keys, imports, prelude), file: this.file }, output: this.continuationOutput(live.out.get(ROOT)!, returns) },
      ...(meta.environment !== undefined ? { environment: meta.environment } : {}),
      ...(meta.limits !== undefined ? { limits: meta.limits as StateDef["limits"] } : {}),
      generated: generatedHere,
    } as StateDef;

    const calledMounts: Record<string, ChildDecl> = {};
    for (const [key, id] of this.called) {
      if (keys.has(key) || [...keys.values()].includes(key)) throw new ScriptCompileError(`the called state '${key}' has the same key as a phase — rename the import`, this.file);
      calledMounts[key] = { state: id, called: true } as ChildDecl;
    }
    if (Object.keys(calledMounts).length > 0) {
      root.children = { ...calledMounts };
      root.sequence = [];
    }
    if (phases.length > 0) {
      const children: Record<string, ChildDecl> = { ...calledMounts };
      for (const phase of phases) {
        const key = keys.get(phase)!;
        children[key] = { transitions: this.exitRules(`.children.${key}.output`, phase, graph, live, keys) } as ChildDecl;
        const detail = meta.phases?.find((p) => p.title === phase)?.detail;
        documents[`${this.options.stateId}/${key}`] = {
          label: phase,
          ...(detail !== undefined ? { description: detail } : {}),
          inputs: this.phaseInputs(live.in.get(phase)!),
          outputs: this.phaseOutputs(live.out.get(phase)!, returns),
          operation: { script: { code: this.segmentModule(phase, phaseSets, keys, imports, prelude), file: this.file }, output: this.continuationOutput(live.out.get(phase)!, returns) },
          generated,
        } as StateDef;
      }
      root.children = children;
      root.sequence = [];
      root.transitions = this.exitRules(".operation.output", ROOT, graph, live, keys);
    }
    if (outputs.result !== undefined && Object.keys(outputs).length === 1 && this.wholeReturn) {
      (root.generated as GeneratedProvenance & { whole?: string }).whole = "result";
    }
    documents[this.options.stateId] = root;
    this.checkUnusedPhaseDescriptions(meta, phases);
    return { mode: this.mode, meta, documents, warnings: this.warnings, generated };
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
    const { ts } = this;
    if (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isSatisfiesExpression(node) || ts.isTypeAssertionExpression(node)) {
      return this.literal(node.expression, where);
    }
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
    if (ts.isNumericLiteral(node)) return Number(node.text);
    if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
    if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
    if (node.kind === ts.SyntaxKind.NullKeyword) return null;
    if (ts.isPrefixUnaryExpression(node) && node.operator === ts.SyntaxKind.MinusToken && ts.isNumericLiteral(node.operand)) return -Number(node.operand.text);
    if (ts.isArrayLiteralExpression(node)) {
      return node.elements.map((e, i) => {
        if (ts.isSpreadElement(e) || ts.isOmittedExpression(e)) throw this.error(e, `${where}[${i}]: a spread or a hole is not a literal`);
        return this.literal(e, `${where}[${i}]`);
      });
    }
    if (ts.isObjectLiteralExpression(node)) {
      const out: Record<string, JsonValue> = {};
      for (const member of node.properties) {
        if (!ts.isPropertyAssignment(member)) throw this.error(member, `${where}: only \`key: value\` members are literals — not a spread, a shorthand or a method`);
        const name = member.name;
        const key = ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name) ? name.text : undefined;
        if (key === undefined) throw this.error(name, `${where}: a computed key is not a literal`);
        out[key] = this.literal(member.initializer, `${where}.${key}`);
      }
      return out;
    }
    throw this.error(node, `${where} must be a pure literal — no identifiers, calls, spreads or interpolation`);
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

  private hoist(symbol: TS.Symbol, at: TS.Node): Hoisted {
    return this.hoistNamed(symbol, symbol.getName(), at);
  }

  private hoistNamed(symbol: TS.Symbol, base: string, at: TS.Node | undefined): Hoisted {
    const existing = this.hoisted.get(symbol);
    if (existing !== undefined) return existing;
    let name = base;
    for (let n = 1; this.hoistedByName.has(name) || (Object.values(SCRIPT_CONTROL) as string[]).includes(name); n++) name = `${base}_${n}`;
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

  private syntheticVar(base: string, schema: JsonSchema | undefined): Hoisted {
    let name = `${base}_${this.synthetic++}`;
    while (this.hoistedByName.has(name)) name = `${base}_${this.synthetic++}`;
    const entry: Hoisted = { name, ...(schema !== undefined ? { schema } : {}) };
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

  // --- called states -------------------------------------------------------------------------------

  /**
   * Whether an import names a STATE rather than code (SCRIPTS.md §10): a state document beside it, or
   * a script — a module with a `meta`. A module without one is code, and runs inline. An imported
   * state binds a callable `(inputs) => outputs` and becomes a mount of the script's state.
   */
  private importsState(declaration: TS.ImportDeclaration): boolean {
    const { ts } = this;
    const clause = declaration.importClause;
    if (clause === undefined || clause.isTypeOnly || !ts.isStringLiteral(declaration.moduleSpecifier)) return false;
    const specifier = declaration.moduleSpecifier.text;
    if (specifier === HOOK_MODULE) return false;
    const target = this.stateFileOf(specifier);
    if (target === undefined) return false;
    if (clause.namedBindings !== undefined || clause.name === undefined) {
      throw this.error(declaration, `'${specifier}' is a state: import it as a default — \`import name from "${specifier}"\` — which binds a function that runs it`);
    }
    const id = this.stateIdOfFile(target, declaration);
    this.stateImports.set(clause.name.text, id);
    this.mount(clause.name.text, id);
    return true;
  }

  /** The state file a specifier names, if it names one. */
  private stateFileOf(specifier: string): string | undefined {
    const options = this.options;
    const base = specifier.startsWith(".") ? normalizePath(`${dirOf(this.file.replace(/\\/g, "/"))}/${specifier}`) : undefined;
    if (base !== undefined) {
      for (const ext of DATA_EXTENSIONS) if (options.vfs.read(`${base}.${ext}`) !== undefined) return `${base}.${ext}`;
    }
    const module = resolveSpecifier(specifier, dirOf(this.file.replace(/\\/g, "/")), options);
    if (module === undefined || module === this.file) return undefined;
    const text = options.vfs.read(module);
    if (text === undefined || !hasScriptMeta(this.ts, module, text)) return undefined;
    // A script that declares itself a library is code, not a state.
    return /compile\s*:\s*["']function["']/.test(text) ? undefined : module;
  }

  private stateIdOfFile(file: string, at: TS.Node): string {
    const id = this.options.stateIdOf?.(file) ?? (file.startsWith(`${this.root}/`) ? file.slice(this.root.length + 1).replace(/\.[^./]+$/, "") : undefined);
    if (id === undefined) throw this.error(at, `'${file}' is not under the workflow root '${this.root}', so it names no state`);
    return id;
  }

  /** A called state as a mount of the script's state: keyed by the name it is called by. */
  private mount(key: string, id: string): void {
    const existing = this.called.get(key);
    if (existing !== undefined && existing !== id) throw new ScriptCompileError(`two called states are both '${key}' ('${existing}', '${id}') — import one under another name`, this.file);
    this.called.set(key, id);
  }

  /**
   * The state a `workflow(ref)` names — a LITERAL, since a computed name can be neither mounted nor
   * frozen (SCRIPTS.md §10). A relative ref is read from the script's directory, a bare one from the
   * workflow root; Claude's `{ scriptPath }` is a path.
   */
  private workflowTarget(call: TS.CallExpression): string {
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
    const id = clean.startsWith(".") ? this.stateIdOfFile(normalizePath(`${dirOf(this.file.replace(/\\/g, "/"))}/${clean}`), call) : clean.replace(/^\/+/, "");
    this.mount(id.split("/").pop()!, id);
    return id;
  }

  // --- hooks ---------------------------------------------------------------------------------------

  /** Which hook an identifier names, if any — through a global or an import of the hook module. */
  private hookOf(node: TS.Identifier): string | undefined {
    let symbol = this.checker.getSymbolAtLocation(node);
    if (symbol === undefined) return undefined;
    if (symbol.flags & this.ts.SymbolFlags.Alias) symbol = this.checker.getAliasedSymbol(symbol);
    const declaration = symbol.declarations?.[0];
    const file = declaration?.getSourceFile().fileName;
    if (file !== HOOK_GLOBALS_PATH && file !== HOOK_MODULE_PATH) return undefined;
    const name = symbol.getName();
    return name === "args" ? undefined : name;
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
    if (this.mode !== "states") return "";
    const arg = e.arguments[0];
    if (arg === undefined || !(ts.isStringLiteral(arg) || ts.isNoSubstitutionTemplateLiteral(arg)) || e.arguments.length !== 1) {
      throw this.error(e, "phase() takes one string literal — a phase is a state, and a computed name has no state to be");
    }
    if (arg.text.length === 0) throw this.error(e, "phase(\"\") names no phase");
    return arg.text;
  }

  /**
   * Every `phase()` call must be one the compiler can cut at — a statement of its own, in the entry
   * function (or the script body), outside every callback (SCRIPTS.md §6.1). Only in `"states"` mode;
   * in `"state"` mode a phase is Claude's display group and may be called from anywhere.
   */
  private checkPhaseCalls(statements: readonly TS.Statement[], entry: TS.FunctionLikeDeclaration | undefined): void {
    if (this.mode !== "states") return;
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

  private containsPhase(node: TS.Node): boolean {
    if (this.mode !== "states") return false;
    let found = false;
    const visit = (n: TS.Node): void => {
      if (found) return;
      if (this.isPhaseCall(n)) {
        found = true;
        return;
      }
      this.ts.forEachChild(n, visit);
    };
    visit(node);
    return found;
  }

  // --- lowering ------------------------------------------------------------------------------------

  /**
   * The pre-pass: every declaration at a HOISTED position — directly in the body, or directly in a
   * structure that contains a `phase()` — becomes a `$v` variable, unless it is a helper or a constant.
   * Mirrors {@link lower}'s walk exactly, since the two must agree about which positions are cut.
   */
  private collectHoisting(s: TS.Statement): void {
    const { ts } = this;
    if (ts.isVariableStatement(s)) return this.collectVariables(s.declarationList, hasModifier(ts, s, ts.SyntaxKind.ExportKeyword));
    if (!this.containsPhase(s)) return;
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

  private newBlock(): Block {
    const block: Block = { id: this.blocks.length, code: [], jumps: [], uses: new Set(), defs: new Set() };
    this.blocks.push(block);
    return block;
  }

  private end(term: Terminator): void {
    if (this.cur.term === undefined) this.cur.term = term;
    const reads = new Set<string>();
    if (term.kind === "branch") this.collectRefs(term.cond, reads);
    if (term.kind === "return" && term.value !== undefined) this.collectRefs(term.value, reads);
    if (term.kind === "return" && term.fallthrough) {
      for (const name of this.exportNames) reads.add(name);
      if (this.defaultExport !== undefined) reads.add(this.defaultExport);
    }
    this.touch(reads, []);
  }

  /** Record, in order, what the current block reads and then assigns. */
  private touch(reads: Iterable<string>, writes: Iterable<string>): void {
    for (const r of reads) if (!this.cur.defs.has(r)) this.cur.uses.add(r);
    for (const w of writes) this.cur.defs.add(w);
  }

  private lowerList(statements: readonly TS.Statement[]): void {
    for (const s of statements) this.lower(s);
  }

  private lower(s: TS.Statement): void {
    const { ts } = this;
    if (ts.isEmptyStatement(s)) return;
    if (isTypeOnly(ts, s)) return; // types have no run-time effect
    if (ts.isFunctionDeclaration(s) || ts.isClassDeclaration(s)) {
      this.addHelper(s);
      return;
    }
    if (ts.isExportAssignment(s)) {
      // `export default <value>` in a body: the default output.
      const v = this.syntheticVar("default", undefined);
      this.defaultExport = v.name;
      this.emit(ts.factory.createExpressionStatement(ts.factory.createAssignment(vRef(ts, v.name), s.expression)), s, [v.name]);
      return;
    }
    if (ts.isVariableStatement(s)) {
      this.lowerVariables(s.declarationList, s, hasModifier(ts, s, ts.SyntaxKind.ExportKeyword));
      return;
    }
    const phase = this.phaseStatement(s);
    if (phase !== undefined && this.mode === "states") {
      const next = this.newBlock();
      this.end({ kind: "phase", phase, to: next.id });
      this.cur = next;
      return;
    }
    if (!this.containsPhase(s)) {
      this.verbatim(s);
      return;
    }
    const labels = this.pendingLabels;
    this.pendingLabels = [];
    if (ts.isBlock(s)) return this.lowerList(s.statements);
    if (ts.isLabeledStatement(s)) {
      this.pendingLabels = [...labels, s.label.text];
      return this.lower(s.statement);
    }
    if (ts.isIfStatement(s)) {
      const then = this.newBlock();
      const otherwise = s.elseStatement !== undefined ? this.newBlock() : undefined;
      const join = this.newBlock();
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
      const head = this.newBlock();
      const body = this.newBlock();
      const exit = this.newBlock();
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      this.end({ kind: "branch", cond: s.expression, then: body.id, else: exit.id });
      this.loop({ breakTo: exit.id, continueTo: head.id, labels }, body, s.statement, head.id);
      this.cur = exit;
      return;
    }
    if (ts.isDoStatement(s)) {
      const body = this.newBlock();
      const cond = this.newBlock();
      const exit = this.newBlock();
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
        else this.verbatim(ts.factory.createExpressionStatement(s.initializer), s);
      }
      const head = this.newBlock();
      const body = this.newBlock();
      const cont = this.newBlock();
      const exit = this.newBlock();
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      if (s.condition !== undefined) this.end({ kind: "branch", cond: s.condition, then: body.id, else: exit.id });
      else this.end({ kind: "goto", to: body.id });
      this.loop({ breakTo: exit.id, continueTo: cont.id, labels }, body, s.statement, cont.id);
      this.cur = cont;
      if (s.incrementor !== undefined) this.verbatim(ts.factory.createExpressionStatement(s.incrementor), s);
      this.end({ kind: "goto", to: head.id });
      this.cur = exit;
      return;
    }
    if (ts.isForOfStatement(s)) {
      if (s.awaitModifier !== undefined) throw this.error(s, "a `for await` loop that changes phase cannot be cut — iterate an array");
      const iterableType = this.checker.getTypeAtLocation(s.expression);
      let schema: JsonSchema | undefined;
      try {
        schema = typeToWireSchema(ts, this.checker, iterableType, "the iterated value").schema;
      } catch (e) {
        if (!(e instanceof WireTypeError)) throw e;
        throw this.error(s.expression, `a loop that changes phase keeps what it iterates across the phases, so it must have a wire form: ${e.message}`);
      }
      const items = this.syntheticVar("items", Object.keys(schema).length > 0 ? schema : undefined);
      const index = this.syntheticVar("index", { type: "integer" } as JsonSchema);
      const f = ts.factory;
      // `$v.items = [...expr]; $v.index = 0;`
      this.emit(f.createExpressionStatement(f.createAssignment(vRef(ts, items.name), f.createArrayLiteralExpression([f.createSpreadElement(s.expression)]))), s, [items.name]);
      this.emit(f.createExpressionStatement(f.createAssignment(vRef(ts, index.name), f.createNumericLiteral(0))), s, [index.name]);
      const head = this.newBlock();
      const body = this.newBlock();
      const cont = this.newBlock();
      const exit = this.newBlock();
      this.end({ kind: "goto", to: head.id });
      this.cur = head;
      const cond = f.createBinaryExpression(vRef(ts, index.name), ts.SyntaxKind.LessThanToken, f.createPropertyAccessExpression(vRef(ts, items.name), "length"));
      this.end({ kind: "branch", cond, then: body.id, else: exit.id });
      this.cur = body;
      const element = f.createElementAccessExpression(vRef(ts, items.name), vRef(ts, index.name));
      this.touch([items.name, index.name], []);
      if (ts.isVariableDeclarationList(s.initializer)) {
        const d = s.initializer.declarations[0]!;
        this.assignDeclaration(d.name, element, s);
      } else {
        this.verbatim(f.createExpressionStatement(f.createAssignment(s.initializer as TS.Expression, element)), s);
      }
      this.loop({ breakTo: exit.id, continueTo: cont.id, labels }, undefined, s.statement, cont.id);
      this.cur = cont;
      this.touch([index.name], []);
      this.emit(f.createExpressionStatement(f.createPostfixIncrement(vRef(ts, index.name))), s, [index.name]);
      this.end({ kind: "goto", to: head.id });
      this.cur = exit;
      return;
    }
    if (ts.isTryStatement(s)) throw this.error(s, "a `try` around a phase() cannot be cut yet — catch inside one phase, or let the failure end the state");
    if (ts.isSwitchStatement(s)) throw this.error(s, "a `switch` around a phase() cannot be cut yet — write it as `if`/`else`");
    if (ts.isForInStatement(s)) throw this.error(s, "a `for…in` loop around a phase() cannot be cut — iterate `Object.keys(…)` with `for…of`");
    throw this.error(s, "phase() must be a statement of its own, directly in the script's control flow");
  }

  /** Lower a loop body under its jump targets, ending in a jump to `continueTo`. */
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
      if (d.initializer !== undefined && this.containsPhase(d.initializer)) {
        throw this.error(d, "phase() must be a statement of its own, not part of a declaration");
      }
      if (this.helperDeclarations.has(d)) {
        // A helper or a constant: re-declared in every segment rather than carried across phases.
        this.addHelper(ts.factory.createVariableStatement(undefined, ts.factory.createVariableDeclarationList([d], ts.NodeFlags.Const)));
        continue;
      }
      if (exported) for (const name of boundNames(ts, d.name)) this.exportNames.push(name);
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

  /** An initializer that is the same in every segment — so it is re-evaluated, never carried. */
  private isStatic(node: TS.Expression): boolean {
    const { ts } = this;
    const e = unwrap(ts, node);
    if (ts.isArrowFunction(e) || ts.isFunctionExpression(e) || ts.isClassExpression(e)) return !this.containsPhase(e);
    if (ts.isStringLiteral(e) || ts.isNumericLiteral(e) || ts.isNoSubstitutionTemplateLiteral(e)) return true;
    if (e.kind === ts.SyntaxKind.TrueKeyword || e.kind === ts.SyntaxKind.FalseKeyword || e.kind === ts.SyntaxKind.NullKeyword) return true;
    if (ts.isPrefixUnaryExpression(e)) return this.isStatic(e.operand);
    if (ts.isTemplateExpression(e)) return e.templateSpans.every((span) => this.isStatic(span.expression));
    if (ts.isArrayLiteralExpression(e)) return e.elements.every((x) => (ts.isSpreadElement(x) ? this.isStatic(x.expression) : ts.isOmittedExpression(x) || this.isStatic(x)));
    if (ts.isObjectLiteralExpression(e)) {
      return e.properties.every((p) =>
        ts.isPropertyAssignment(p) ? !ts.isComputedPropertyName(p.name) && this.isStatic(p.initializer) : ts.isSpreadAssignment(p) ? this.isStatic(p.expression) : ts.isMethodDeclaration(p),
      );
    }
    if (ts.isIdentifier(e)) {
      const symbol = this.checker.getSymbolAtLocation(e);
      // A hoisted variable changes; anything else an identifier can name here (an import, a helper,
      // a module-level declaration, a global) is the same in every segment.
      return symbol === undefined || !this.hoisted.has(symbol);
    }
    if (ts.isPropertyAccessExpression(e)) return this.isStatic(e.expression);
    return false;
  }

  private addHelper(s: TS.Statement): void {
    const rewritten = this.rewrite(s, false);
    for (const name of rewritten.refs) this.helperRefs.add(name);
    this.helpers.push(rewritten.node as TS.Statement);
  }

  /**
   * A statement as written, rewritten for the segment. Everything it names counts as READ — the
   * conservative answer for arbitrary code, which liveness can only over-approximate.
   */
  private verbatim(s: TS.Statement, original?: TS.Node, writes: readonly string[] = []): void {
    const rewritten = this.rewrite(s, true, original);
    this.cur.code.push(rewritten.node as TS.Statement);
    this.touch(rewritten.refs, writes);
    this.cur.jumps.push(...rewritten.jumps);
    if (rewritten.returns) this.sawReturn = true;
  }

  /** A synthesized assignment whose value is source — its target is WRITTEN, its value read. */
  private emit(s: TS.Statement, original: TS.Node, writes: readonly string[]): void {
    this.verbatim(s, original, writes);
  }

  // --- the rewrite ---------------------------------------------------------------------------------

  /**
   * Rewrite a statement for a segment: hoisted variables become `$v.<name>`, a hook becomes the hook
   * module's export, `llm<T>(…)` becomes `llm.withOutput(<schema>)(…)`, and — where `controlFlow` —
   * a `return` becomes the continuation that ends the state, and a `break`/`continue` that leaves the
   * statement becomes a jump to the block it targets.
   */
  private rewrite(node: TS.Node, controlFlow: boolean, original?: TS.Node): { node: TS.Node; refs: Set<string>; jumps: number[]; returns: boolean } {
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
        return f.createBlock([
          f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createNumericLiteral(target))),
          f.createContinueStatement(f.createIdentifier("$dispatch")),
        ]);
      };

      const visit = (n: TS.Node): TS.Node | undefined => {
        if (ts.isTypeNode(n) && !ts.isExpressionWithTypeArguments(n)) return n;
        if (isFunctionBoundary(ts, n)) {
          functionDepth++;
          const saved = local.splice(0);
          const out = ts.visitEachChild(n, visit, context);
          local.push(...saved);
          functionDepth--;
          return out;
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
        if (ts.isCallExpression(n) && ts.isIdentifier(n.expression) && compiler.hookOf(n.expression) === "workflow") {
          const id = compiler.workflowTarget(n);
          return f.createCallExpression(visit(n.expression) as TS.Expression, undefined, [f.createStringLiteral(id), ...n.arguments.slice(1).map((a) => visit(a) as TS.Expression)]);
        }
        if (ts.isCallExpression(n) && n.typeArguments !== undefined && n.typeArguments.length > 0 && ts.isIdentifier(n.expression)) {
          const hook = compiler.hookOf(n.expression);
          if (hook === "llm" || hook === "agent") {
            const schema = compiler.typeArgumentSchema(n);
            const callee = visit(n.expression) as TS.Expression;
            const typed = f.createCallExpression(f.createPropertyAccessExpression(callee, "withOutput"), undefined, [jsonExpression(f, schema as JsonValue)]);
            return f.createCallExpression(typed, undefined, n.arguments.map((a) => visit(a) as TS.Expression));
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
            const target = label !== undefined ? [...loops].reverse().find((l) => l.labels.includes(label)) : loops[loops.length - 1];
            if (target === undefined) throw compiler.error(original ?? n, `\`${isBreak ? "break" : "continue"}${label ? ` ${label}` : ""}\` has no loop to leave`);
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

  private rewriteIdentifier(n: TS.Identifier, refs: Set<string>, f: TS.NodeFactory): TS.Node {
    const { ts } = this;
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
    }
    if (this.isGlobalHook(n)) return hookRef(ts, this.hookOf(n)!);
    return n;
  }

  /** A hook named as a GLOBAL (Claude's spelling) — rewritten to the hook module's export. */
  private isGlobalHook(n: TS.Identifier): boolean {
    const symbol = this.checker.getSymbolAtLocation(n);
    return symbol !== undefined && (symbol.flags & this.ts.SymbolFlags.Alias) === 0 && symbol.declarations?.[0]?.getSourceFile().fileName === HOOK_GLOBALS_PATH && symbol.getName() !== "args";
  }

  /** The output contract `llm<T>` names — `T` converted by the one TS → wire path (SCRIPTS.md §9). */
  private typeArgumentSchema(call: TS.CallExpression): JsonSchema {
    const node = call.typeArguments![0]!;
    const type = this.checker.getTypeFromTypeNode(node);
    try {
      const { schema, warnings } = typeToWireSchema(this.ts, this.checker, type, `the type argument '${node.getText(this.sf)}'`);
      for (const w of warnings) this.warnings.push(`${this.where(node)}: ${w}`);
      return schema;
    } catch (e) {
      if (e instanceof WireTypeError) throw this.error(node, e.message);
      throw e;
    }
  }

  private collectRefs(node: TS.Node, into: Set<string>): void {
    const visit = (n: TS.Node): void => {
      if (this.ts.isIdentifier(n)) {
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

  // --- the phase graph -----------------------------------------------------------------------------

  /** Which phases can be current at the start of each block — a forward pass to a fixpoint. */
  private phaseSets(): Map<number, Set<string>> {
    const sets = new Map<number, Set<string>>();
    const add = (block: number, phases: Iterable<string>): boolean => {
      const set = sets.get(block) ?? new Set<string>();
      const before = set.size;
      for (const p of phases) set.add(p);
      sets.set(block, set);
      return set.size !== before;
    };
    add(0, [ROOT]);
    const work = [0];
    while (work.length > 0) {
      const id = work.pop()!;
      const block = this.blocks[id]!;
      const current = sets.get(id)!;
      const flow = (to: number, phases: Iterable<string>): void => {
        if (add(to, phases)) work.push(to);
      };
      for (const j of block.jumps) flow(j, current);
      const term = block.term;
      if (term === undefined) continue;
      if (term.kind === "goto") flow(term.to, current);
      else if (term.kind === "branch") {
        flow(term.then, current);
        flow(term.else, current);
      } else if (term.kind === "phase") flow(term.to, [term.phase]);
    }
    return sets;
  }

  /** The phases, in the order control first reaches them. */
  private phaseOrder(sets: Map<number, Set<string>>): string[] {
    const order: string[] = [];
    for (const block of this.blocks) {
      if (block.term?.kind === "phase" && sets.has(block.id) && !order.includes(block.term.phase)) order.push(block.term.phase);
    }
    return order;
  }

  private phaseKeys(phases: readonly string[]): Map<string, string> {
    const keys = new Map<string, string>();
    const taken = new Map<string, string>();
    for (const phase of phases) {
      const key = phase.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "phase";
      const clash = taken.get(key);
      if (clash !== undefined) throw new ScriptCompileError(`phases '${clash}' and '${phase}' both become the state '${key}' — rename one`, this.file);
      taken.set(key, phase);
      keys.set(phase, key);
    }
    return keys;
  }

  /** For each phase, where it can go next — the phases it transitions to, by entry block — and whether it can return. */
  private phaseGraph(sets: Map<number, Set<string>>): Map<string, { to: Map<string, Set<number>>; returns: boolean }> {
    const graph = new Map<string, { to: Map<string, Set<number>>; returns: boolean }>();
    const node = (p: string) => {
      let n = graph.get(p);
      if (n === undefined) graph.set(p, (n = { to: new Map(), returns: false }));
      return n;
    };
    node(ROOT);
    for (const block of this.blocks) {
      const phases = sets.get(block.id);
      if (phases === undefined) continue;
      for (const p of phases) {
        const n = node(p);
        const term = block.term;
        if (term?.kind === "phase" && term.phase !== p) {
          const entries = n.to.get(term.phase) ?? new Set<number>();
          entries.add(term.to);
          n.to.set(term.phase, entries);
        }
        if (term?.kind === "return" || block.code.some((s) => containsReturn(this.ts, s))) n.returns = true;
      }
    }
    return graph;
  }

  /**
   * Which variables each phase state takes in and hands on.
   *
   * Ordinary backward liveness over the blocks — a variable is live into a block if the block reads
   * it before assigning it, or it is live out and the block does not assign it — with one twist: a
   * `phase()` edge is an edge like any other, and it is exactly where a live value has to CROSS. So a
   * phase state takes in what is live at its entries, and hands on what the entries it moves to take.
   * Helpers read variables whenever they are called, so what they read is live everywhere.
   */
  private liveness(
    sets: Map<number, Set<string>>,
    graph: Map<string, { to: Map<string, Set<number>> }>,
    phases: readonly string[],
  ): { in: Map<string, Set<string>>; out: Map<string, Set<string>> } {
    const successors = (b: Block): number[] => {
      const out = [...b.jumps];
      const t = b.term;
      if (t?.kind === "goto" || t?.kind === "phase") out.push(t.to);
      else if (t?.kind === "branch") out.push(t.then, t.else);
      return out;
    };
    const liveIn = new Map<number, Set<string>>(this.blocks.map((b) => [b.id, new Set([...b.uses, ...this.helperRefs])]));
    let changed = true;
    while (changed) {
      changed = false;
      for (const b of [...this.blocks].reverse()) {
        const into = liveIn.get(b.id)!;
        for (const next of successors(b)) {
          for (const v of liveIn.get(next)!) {
            if (!b.defs.has(v) && !into.has(v)) {
              into.add(v);
              changed = true;
            }
          }
        }
      }
    }
    const live = { in: new Map<string, Set<string>>(), out: new Map<string, Set<string>>() };
    for (const p of [ROOT, ...phases]) {
      live.in.set(p, new Set());
      live.out.set(p, new Set());
    }
    for (const b of this.blocks) {
      if (b.term?.kind !== "phase" || !sets.has(b.id)) continue;
      for (const v of liveIn.get(b.term.to)!) live.in.get(b.term.phase)!.add(v);
    }
    for (const p of [ROOT, ...phases]) {
      for (const next of graph.get(p)?.to.keys() ?? []) for (const v of live.in.get(next)!) live.out.get(p)!.add(v);
    }
    // A value that crosses into a phase state must have a wire form (SCRIPTS.md §8).
    for (const p of phases) {
      for (const v of live.in.get(p)!) {
        const hoisted = this.hoistedByName.get(v);
        if (hoisted?.unrepresentable !== undefined) {
          throw new ScriptCompileError(`the variable '${v}' lives across the phase '${p}', so it travels as a wire — but ${hoisted.unrepresentable}. Keep it inside one phase, or make it plain data`, this.file);
        }
      }
    }
    return live;
  }

  // --- documents -----------------------------------------------------------------------------------

  private phaseInputs(vars: ReadonlySet<string>): Record<string, ParameterDecl> {
    const inputs: Record<string, ParameterDecl> = { [SCRIPT_CONTROL.entry]: { schema: { type: "integer" } as JsonSchema, optional: true, description: "Where in the phase's code this entry begins." } };
    for (const v of [...vars].sort()) inputs[v] = this.slotOf(v);
    return inputs;
  }

  private phaseOutputs(vars: ReadonlySet<string>, returns: JsonSchema | undefined): Record<string, NamedParameterDecl> {
    const returned = returns !== undefined ? (withoutUnions(returns as JsonValue) as JsonSchema) : undefined;
    const outputs: Record<string, NamedParameterDecl> = {
      [SCRIPT_CONTROL.next]: { schema: { type: "string" } as JsonSchema, description: "The phase this one hands on to, or '$return'." },
      [SCRIPT_CONTROL.entry]: { schema: { type: "integer" } as JsonSchema, optional: true },
      [SCRIPT_CONTROL.returned]: {
        ...(returned !== undefined && Object.keys(returned).length > 0 ? { schema: returned } : {}),
        optional: true,
        description: "What the script returned, when this phase returned.",
      },
    };
    for (const v of [...vars].sort()) outputs[v] = this.slotOf(v);
    return outputs;
  }

  private slotOf(v: string): ParameterDecl {
    const schema = this.wireSchemaOf(v);
    return { ...(schema !== undefined ? { schema } : {}), optional: true };
  }

  /**
   * A variable's type as it travels between phase states. A UNION inside it is widened to "anything":
   * the wiring checker does not reason about unions yet (§6.2), and both ends of every wire the
   * compiler makes are the same variable, so nothing is lost that the checker could have used.
   */
  private wireSchemaOf(v: string): JsonSchema | undefined {
    const schema = this.hoistedByName.get(v)?.schema;
    if (schema === undefined) return undefined;
    const widened = withoutUnions(schema as JsonValue) as JsonSchema;
    return Object.keys(widened).length > 0 ? widened : undefined;
  }

  /** The continuation a phase's code returns, typed — what `.operation.output` reads (SCRIPTS.md §6.2). */
  private continuationOutput(vars: ReadonlySet<string>, returns: JsonSchema | undefined): Record<string, ParameterDecl> {
    const properties: Record<string, JsonValue> = {
      [SCRIPT_CONTROL.next]: { type: "string" },
      [SCRIPT_CONTROL.entry]: { type: "integer" },
      [SCRIPT_CONTROL.returned]: (returns !== undefined ? withoutUnions(returns as JsonValue) : {}) as JsonValue,
    };
    for (const v of [...vars].sort()) properties[v] = (this.wireSchemaOf(v) ?? {}) as JsonValue;
    return { continuation: { kind: "json", schema: { type: "object", properties, required: [SCRIPT_CONTROL.next] } as unknown as JsonSchema } };
  }

  /** The rules that route one phase's continuation: one per phase it can go to, and one for its return. */
  private exitRules(
    output: string,
    phase: string,
    graph: Map<string, { to: Map<string, Set<number>>; returns: boolean }>,
    live: { in: Map<string, Set<string>> },
    keys: Map<string, string>,
  ): TransitionDecl[] {
    const rules: TransitionDecl[] = [];
    const node = graph.get(phase);
    for (const next of node?.to.keys() ?? []) {
      const key = keys.get(next)!;
      const inputs: Record<string, string> = { [SCRIPT_CONTROL.entry]: `${output}.${SCRIPT_CONTROL.entry}` };
      for (const v of [...live.in.get(next)!].sort()) inputs[v] = `${output}.${v}`;
      rules.push({ name: `to_${key}`, when: `${output}.${SCRIPT_CONTROL.next} === '${key}'`, to: key, inputs } as TransitionDecl);
    }
    if (node?.returns || phase === ROOT) {
      rules.push({ name: "return", when: `${output}.${SCRIPT_CONTROL.next} === '${SCRIPT_RETURN}'`, to: "terminate.success" } as TransitionDecl);
    }
    return rules;
  }

  /**
   * The root's outputs: what the function returns, read off whichever state returned it — the root's
   * own operation, or the phase that was current at the `return`.
   */
  private rootOutputs(returns: JsonSchema | undefined, phases: readonly string[], keys: Map<string, string>): Record<string, NamedParameterDecl> {
    const returned = (path: string): string => {
      let expr = `.operation.output.${SCRIPT_CONTROL.returned}${path}`;
      for (const phase of [...phases].reverse()) {
        const at = `.children.${keys.get(phase)!}.output`;
        expr = `${at}.${SCRIPT_CONTROL.next} === '${SCRIPT_RETURN}' ? ${at}.${SCRIPT_CONTROL.returned}${path} : ${expr}`;
      }
      return expr;
    };
    const outputs: Record<string, NamedParameterDecl> = {};
    const slot = (schema: JsonSchema | undefined, path: string, optional: boolean): NamedParameterDecl => ({
      ...(schema !== undefined && Object.keys(schema).length > 0 ? { schema } : {}),
      // With phases, which state returned is a run-time fact the reachability check cannot prove, so the
      // slot opts out of it (SPEC §6.2); the script's own return type is what guarantees the value.
      ...(optional || phases.length > 0 ? { optional: true } : {}),
      binding: phases.length > 0 ? { $expr: returned(path) } : `.operation.output.${SCRIPT_CONTROL.returned}${path}`,
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
      for (const [name, property] of Object.entries(schema.properties)) {
        outputs[name] = slot(property, `.${name}`, !(schema.required ?? []).includes(name));
      }
      return outputs;
    }
    outputs.result = slot(returns, "", false);
    this.wholeReturn = true;
    return outputs;
  }

  /** The module a phase state runs: the imports, the static code, and the phase's blocks as a switch. */
  private segmentModule(phase: string, sets: Map<number, Set<string>>, keys: Map<string, string>, imports: readonly TS.ImportDeclaration[], prelude: readonly TS.Statement[]): string {
    const { ts } = this;
    const printer = ts.createPrinter({ removeComments: false });
    const print = (n: TS.Node): string => printer.printNode(ts.EmitHint.Unspecified, n, this.sf);
    const lines: string[] = [];
    lines.push(`// Generated from ${this.file} — phase ${phase === ROOT ? "(root)" : `'${phase}'`}. Edit the script, not this.`);
    for (const i of imports) lines.push(print(i));
    lines.push(`import * as $rt from ${JSON.stringify(HOOK_MODULE)};`);
    lines.push(DETERMINISM_GUARD);
    // An imported state is a function that runs it (SCRIPTS.md §10).
    for (const [name, id] of this.stateImports) lines.push(`const ${name} = (inputs?: unknown): Promise<any> => $rt.workflow(${JSON.stringify(id)}, inputs);`);
    for (const s of prelude) lines.push(print(this.rewrite(s, false).node));
    lines.push("export default async function $segment($v: any, $pc: number): Promise<any> {");
    for (const h of this.helpers) lines.push(print(h));
    lines.push("  $dispatch: for (;;) {");
    lines.push("    switch ($pc) {");
    const f = ts.factory;
    for (const block of this.blocks) {
      if (!sets.get(block.id)?.has(phase)) continue;
      lines.push(`      case ${block.id}: {`);
      for (const s of block.code) lines.push(print(s));
      const term = block.term;
      let tail: TS.Statement[] = [];
      if (term?.kind === "goto") tail = gotoStatements(f, term.to);
      else if (term?.kind === "branch") {
        const cond = this.rewrite(f.createExpressionStatement(term.cond), false).node as TS.ExpressionStatement;
        tail = [
          f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createConditionalExpression(cond.expression, undefined, f.createNumericLiteral(term.then), undefined, f.createNumericLiteral(term.else)))),
          f.createContinueStatement(f.createIdentifier("$dispatch")),
        ];
      } else if (term?.kind === "phase") {
        tail = term.phase === phase ? gotoStatements(f, term.to) : [f.createReturnStatement(continuation(f, keys.get(term.phase)!, term.to, undefined, true))];
      } else if (term?.kind === "return") {
        tail = [f.createReturnStatement(continuation(f, SCRIPT_RETURN, undefined, this.fallthroughValue(f)))];
      }
      for (const s of tail) lines.push(print(s));
      lines.push("      }");
    }
    lines.push(`      default: throw new Error("phase ${phase === ROOT ? "(root)" : phase} has no entry " + $pc);`);
    lines.push("    }");
    lines.push("  }");
    lines.push("}");
    return lines.join("\n");
  }

  /** What falling off the end returns: the exports (body mode), the default export, or nothing. */
  private fallthroughValue(f: TS.NodeFactory): TS.Expression | undefined {
    const { ts } = this;
    if (this.exportNames.length > 0) {
      return f.createObjectLiteralExpression(this.exportNames.map((n) => f.createPropertyAssignment(n, vRef(ts, n))));
    }
    if (this.defaultExport !== undefined) return vRef(ts, this.defaultExport);
    return undefined;
  }

  private checkUnusedPhaseDescriptions(meta: ScriptMeta, phases: readonly string[]): void {
    for (const described of meta.phases ?? []) {
      if (this.mode === "states" && !phases.includes(described.title)) {
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
    return { from: this.file, inputs };
  }

  // --- errors --------------------------------------------------------------------------------------

  private where(node: TS.Node): string {
    const at = this.sf.getLineAndCharacterOfPosition(node.getStart(this.sf));
    return `${this.file}:${at.line + 1}:${at.character + 1}`;
  }

  error(node: TS.Node, message: string): ScriptCompileError {
    // A synthesized node has no position; report it at the file.
    if (node.pos < 0) return new ScriptCompileError(message, this.file);
    const at = this.sf.getLineAndCharacterOfPosition(node.getStart(this.sf));
    return new ScriptCompileError(message, this.file, at.line + 1, at.character + 1);
  }
}

/**
 * What a script's own code sees as `Date` and `Math`: Claude's determinism bans (SCRIPTS.md §11). A
 * phase re-runs from the top of its code after a restart, so a clock or a coin read inline would come
 * back different the second time; `now()` and `random()` are the recorded versions.
 */
const DETERMINISM_GUARD = `const Math: any = new Proxy(globalThis.Math, { get(t: any, p: any) { if (p === "random") return () => { throw new Error("Math.random() breaks replay — use random() from @declarative-ai/hw/script"); }; return Reflect.get(t, p); } });
const Date: any = new Proxy(globalThis.Date, {
  construct(t: any, a: any[]) { if (a.length === 0) throw new Error("new Date() with no argument breaks replay — use now() from @declarative-ai/hw/script"); return Reflect.construct(t, a); },
  apply() { throw new Error("Date() breaks replay — use now() from @declarative-ai/hw/script"); },
  get(t: any, p: any) { if (p === "now") return () => { throw new Error("Date.now() breaks replay — use now() from @declarative-ai/hw/script"); }; return Reflect.get(t, p); },
});`;

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

/** A schema with every union widened to the universal schema — see `wireSchemaOf`. */
function withoutUnions(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(withoutUnions);
  if (value === null || typeof value !== "object") return value;
  if ("anyOf" in value || "oneOf" in value || "allOf" in value) return {};
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, withoutUnions(v as JsonValue)]));
}

function vRef(ts: typeof TS, name: string): TS.PropertyAccessExpression {
  return ts.factory.createPropertyAccessExpression(ts.factory.createIdentifier("$v"), name);
}

function hookRef(ts: typeof TS, name: string): TS.PropertyAccessExpression {
  return ts.factory.createPropertyAccessExpression(ts.factory.createIdentifier("$rt"), name);
}

function gotoStatements(f: TS.NodeFactory, to: number): TS.Statement[] {
  return [f.createExpressionStatement(f.createAssignment(f.createIdentifier("$pc"), f.createNumericLiteral(to))), f.createContinueStatement(f.createIdentifier("$dispatch"))];
}

/** `{ _next, _entry?, _return?, ...$v }` — how a phase's code ends its state (SCRIPTS.md §6.2). */
function continuation(f: TS.NodeFactory, next: string, entry: number | undefined, value: TS.Expression | undefined, carryVars = false): TS.Expression {
  const properties: TS.ObjectLiteralElementLike[] = [];
  if (carryVars) properties.push(f.createSpreadAssignment(f.createIdentifier("$v")));
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
  while (ts.isParenthesizedExpression(e) || ts.isAsExpression(e) || ts.isSatisfiesExpression(e)) e = e.expression;
  return e;
}

function isTypeOnly(ts: typeof TS, s: TS.Statement): boolean {
  return ts.isInterfaceDeclaration(s) || ts.isTypeAliasDeclaration(s) || (ts.isModuleDeclaration(s) && hasModifier(ts, s, ts.SyntaxKind.DeclareKeyword));
}

function isFunctionBoundary(ts: typeof TS, n: TS.Node): boolean {
  return ts.isFunctionDeclaration(n) || ts.isFunctionExpression(n) || ts.isArrowFunction(n) || ts.isMethodDeclaration(n) || ts.isGetAccessor(n) || ts.isSetAccessor(n) || ts.isConstructorDeclaration(n) || ts.isClassDeclaration(n) || ts.isClassExpression(n);
}

function containsReturn(ts: typeof TS, s: TS.Node): boolean {
  let found = false;
  const visit = (n: TS.Node): void => {
    if (found || isFunctionBoundary(ts, n)) return;
    if (ts.isReturnStatement(n)) {
      found = true;
      return;
    }
    ts.forEachChild(n, visit);
  };
  visit(s);
  return found;
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

/** The state properties `meta` sets — `name` read as `label` (SCRIPTS.md §4). */
function stateProperties(meta: ScriptMeta, stateId: string): Partial<StateDef> {
  const label = meta.name ?? meta.label ?? stateId.split("/").pop()!;
  return {
    label,
    ...(meta.description !== undefined ? { description: meta.description } : {}),
    ...(meta.whenToUse !== undefined ? { whenToUse: meta.whenToUse } : {}),
    ...(meta.title !== undefined ? { title: meta.title } : {}),
  } as Partial<StateDef>;
}

/** Where a phase segment's code is prepared: beside the script, so its relative imports resolve as written. */
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
