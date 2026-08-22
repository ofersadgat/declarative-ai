/**
 * Transpiling and running a user module (SPEC §7.5.4, §7.5.6).
 *
 * ## Two halves, and why they are separate
 *
 * `prepareModules` is ASYNC: it resolves the import closure and transpiles it, both of which need
 * the compiler, which is imported on demand. `LoadedModules.execute` is SYNC, because `require` is
 * sync and because the engine calling a function is not a place to start awaiting a compiler.
 *
 * The split does more than satisfy a signature. What `prepareModules` hands back — the emitted
 * JavaScript for every file in the closure — is exactly what SPEC §7.5.5's freeze copies into the
 * snapshot. Preparing and running are separate because *approving* and running are, and the artifact
 * in between is the thing that gets frozen.
 *
 * ## CommonJS, and a require path that is not Node's
 *
 * Emit is CJS because `require` is synchronous and interceptable where the ESM loader hooks are
 * neither. Resolution is not Node's ambient algorithm: a specifier is a REFERENCE (SPEC §7.5.4), so
 * `$JAIRA/lib/review` means in an import what it means in a state file, and a bare specifier is
 * searched along a path derived from the workflow's own.
 *
 * Two things this is NOT:
 *
 *  - **A sandbox.** Node builtins resolve ahead of the path, so `require('fs')` works whatever the
 *    path holds. Controlling resolution makes imports PREDICTABLE; SPEC §7.5.5's approval is what
 *    makes running them acceptable, and nothing here should be mistaken for a boundary.
 *  - **A dependency manager.** A bare specifier finds `<entry>/helper.js` before it looks in any
 *    `node_modules`, so "bare" and "npm package" are different questions — which is why every rule
 *    that treats `node_modules` differently keys on the RESOLVED path.
 */
import { sha256Hex } from "@declarative-ai/exec";
import type * as TS from "typescript";
import { MODULE_EXTENSIONS, ReferenceError_, type Vfs } from "./reference.js";

/** The TypeScript compiler, imported on first use — see the note in `moduleExports`. */
let compiler: Promise<typeof TS> | undefined;
function typescript(): Promise<typeof TS> {
  compiler ??= import("typescript").then((m) => (m as unknown as { default?: typeof TS }).default ?? (m as unknown as typeof TS));
  return compiler;
}

export class ModuleLoadError extends Error {}

/**
 * The require path a search `path` produces: each entry, then that entry's `node_modules`.
 *
 * So one declared path governs both what a state resolves documents against and what a function
 * resolves imports against, and an author writes "where things are" once.
 */
export function requirePathFor(searchPath: readonly string[]): string[] {
  const out: string[] = [];
  for (const entry of searchPath) {
    const dir = entry.replace(/\\/g, "/").replace(/\/+$/, "");
    out.push(dir, `${dir}/node_modules`);
  }
  return out;
}

export interface ModuleResolveOptions {
  vfs: Vfs;
  /** Ordered directories a BARE specifier is searched along — normally {@link requirePathFor}. */
  requirePath: readonly string[];
  /** Roots a `$VAR/…` specifier may name, as `ReferenceOptions.roots`. */
  roots?: Readonly<Record<string, string>>;
}

/**
 * Suffixes and index files a specifier is probed with, in order.
 *
 * The `.js → .ts` rewrite is not a convenience: under TypeScript's own ESM rules an import names the
 * EMITTED file, so a `.ts` module writing `import './helper.js'` is the correct spelling and the one
 * `tsc` demands — this package's own sources are written that way throughout. A resolver without it
 * would make a module's imports unwritable in the dialect the module is written in.
 *
 * The exact spelling is tried FIRST, so a real `helper.js` sitting beside a `helper.ts` still wins
 * when it is the one that was named.
 */
function candidatesFor(base: string): string[] {
  const out = [base];
  const stripped = base.replace(/\.(js|mjs|cjs)$/i, "");
  if (stripped !== base) for (const ext of MODULE_EXTENSIONS) out.push(`${stripped}.${ext}`);
  for (const ext of MODULE_EXTENSIONS) out.push(`${base}.${ext}`);
  for (const ext of MODULE_EXTENSIONS) out.push(`${base}/index.${ext}`);
  return out;
}

/** POSIX separators, `.` dropped, `..` applied. */
function normalize(path: string): string {
  const drive = /^([a-zA-Z]:)(.*)$/.exec(path.replace(/\\/g, "/"));
  const rest = drive ? drive[2]! : path.replace(/\\/g, "/");
  const absolute = rest.startsWith("/");
  const out: string[] = [];
  for (const segment of rest.split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") {
      out.pop();
      continue;
    }
    out.push(segment);
  }
  const joined = out.join("/");
  if (drive) return `${drive[1]}/${joined}`;
  return absolute ? `/${joined}` : joined;
}

/** The directory part of an absolute file path. */
export function dirOf(file: string): string {
  const posix = file.replace(/\\/g, "/");
  const cut = posix.lastIndexOf("/");
  return cut > 0 ? posix.slice(0, cut) : "/";
}

/**
 * Resolve one import specifier to an absolute file, or `undefined` if nothing answers it.
 *
 * The four spellings of SPEC §7.5.4, and they are the reference grammar's rather than Node's — which
 * is the whole point: a function reaching for a shared library should not have to know it is inside a
 * module rather than beside one.
 */
export function resolveSpecifier(specifier: string, fromDir: string, options: ModuleResolveOptions): string | undefined {
  const exists = (path: string): boolean => options.vfs.read(path) !== undefined;
  const firstExisting = (base: string): string | undefined => candidatesFor(normalize(base)).find(exists);

  // `$JAIRA/lib/x` — the same root variable a state file and a `$ref` resolve.
  if (specifier.startsWith("$")) {
    const match = /^\$([A-Z_][A-Z0-9_]*)?(?:\/(.*))?$/.exec(specifier);
    if (!match) throw new ReferenceError_(`malformed root variable in import '${specifier}'`);
    const name = match[1] ?? "JAIRA";
    const root = options.roots?.[name];
    if (root === undefined) {
      throw new ReferenceError_(`unknown root '$${match[1] ?? ""}' in import '${specifier}'`);
    }
    return firstExisting(`${root}/${match[2] ?? ""}`);
  }
  // `./helper`, `../lib/helper` — against the requiring file, as expected.
  if (/^\.\.?(\/|$)/.test(specifier)) return firstExisting(`${fromDir}/${specifier}`);
  // `/opt/shared/x`, `C:/x`
  if (specifier.startsWith("/") || /^[a-zA-Z]:[/\\]/.test(specifier)) return firstExisting(specifier);
  // Bare — searched along the require path. Finds a plain file in a path entry BEFORE any
  // `node_modules`, which is why "bare" does not mean "dependency".
  for (const entry of options.requirePath) {
    const found = firstExisting(`${entry}/${specifier}`);
    if (found !== undefined) return found;
  }
  return undefined;
}

export interface PrepareOptions extends ModuleResolveOptions {
  /**
   * Whether a file may be loaded at all (SPEC §7.5.5). Absent ⇒ every file may.
   *
   * Refusal is an ERROR rather than a miss: a module that imports something unapproved has not
   * failed to find it, it has been told no, and quietly resolving elsewhere would be worse than
   * stopping.
   */
  approved?: (file: string) => boolean;
  /** Emitted JavaScript by CONTENT HASH — the same hash approval and the symbol index key on. */
  cache?: Map<string, string>;
  /** Extra sources not on the filesystem, by pseudo-path — how a synthesized embedded body enters. */
  sources?: Readonly<Record<string, string>>;
}

/** A prepared closure: the emitted code, and a synchronous way to run it. */
export interface LoadedModules {
  /** Absolute file → emitted CommonJS. What SPEC §7.5.5's freeze copies. */
  readonly emitted: ReadonlyMap<string, string>;
  /** Run a module and return its export namespace. Cached, so one module runs once. */
  execute(file: string): Record<string, unknown>;
}

/**
 * Resolve and transpile the import closure of `entries`.
 *
 * Nothing is executed here. That matters for more than tidiness: preparing is what an approval gate
 * happens *around*, and a design where discovering a module's imports required running it would have
 * no moment at which to ask.
 */
export async function prepareModules(entries: readonly string[], options: PrepareOptions): Promise<LoadedModules> {
  const ts = await typescript();
  const emitted = new Map<string, string>();
  /** file → (specifier → resolved file), so `execute`'s `require` needs no resolver at run time. */
  const wiring = new Map<string, Map<string, string>>();

  const sourceOf = (file: string): string | undefined => options.sources?.[file] ?? options.vfs.read(file);

  const queue = [...entries];
  while (queue.length > 0) {
    const file = queue.shift()!;
    if (emitted.has(file)) continue;
    if (options.approved !== undefined && !options.approved(file)) {
      throw new ModuleLoadError(`'${file}' is not approved to run`);
    }
    const source = sourceOf(file);
    if (source === undefined) throw new ModuleLoadError(`'${file}' could not be read`);

    const key = sha256Hex(`${file}\u0000${source}`);
    let js = options.cache?.get(key);
    if (js === undefined) {
      const result = ts.transpileModule(source, {
        fileName: file,
        reportDiagnostics: true,
        compilerOptions: {
          module: ts.ModuleKind.CommonJS,
          target: ts.ScriptTarget.ES2022,
          // Emit is per-file, so the constructs whole-program emit would be needed for are refused at
          // this boundary rather than producing code that silently means something else.
          isolatedModules: true,
          esModuleInterop: true,
          inlineSourceMap: true,
          inlineSources: true,
        },
      });
      const failure = result.diagnostics?.[0];
      if (failure !== undefined) {
        throw new ModuleLoadError(`'${file}' does not compile — ${ts.flattenDiagnosticMessageText(failure.messageText, " ")}`);
      }
      js = result.outputText;
      options.cache?.set(key, js);
    }
    emitted.set(file, js);

    // The closure is walked from the SOURCE's imports rather than by scanning the emitted code: a
    // `require(` in a string literal is not an import, and the AST knows the difference.
    const specifiers = importSpecifiers(ts, file, source);
    const resolved = new Map<string, string>();
    for (const specifier of specifiers) {
      if (isBuiltin(specifier)) continue; // resolved by the host at run time, never bundled
      const target = resolveSpecifier(specifier, dirOf(file), options);
      if (target === undefined) {
        throw new ModuleLoadError(
          `'${file}' imports '${specifier}', which matches nothing on the require path (${options.requirePath.join(", ")})`,
        );
      }
      resolved.set(specifier, target);
      queue.push(target);
    }
    wiring.set(file, resolved);
  }

  return { emitted, execute: (file) => execute(file, emitted, wiring) };
}

/** Node's own modules, which resolve ahead of the require path and are never transpiled. */
function isBuiltin(specifier: string): boolean {
  return specifier.startsWith("node:") || BUILTIN_NAMES.has(specifier);
}

/**
 * A conservative list rather than `module.builtinModules`, which would need a Node import at load.
 * A miss only means the specifier is looked for on the path, where it will fail with a message that
 * names it — never a silent wrong answer.
 */
const BUILTIN_NAMES: ReadonlySet<string> = new Set([
  "assert", "buffer", "child_process", "crypto", "events", "fs", "http", "https", "os", "path",
  "process", "stream", "string_decoder", "timers", "tty", "url", "util", "worker_threads", "zlib",
]);

/** Every module specifier a source imports: `import`, re-`export … from`, and `require(<literal>)`. */
function importSpecifiers(ts: typeof TS, file: string, source: string): string[] {
  const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true);
  const out: string[] = [];
  const visit = (node: TS.Node): void => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
      out.push(node.moduleSpecifier.text);
    } else if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "require" &&
      node.arguments.length === 1 &&
      ts.isStringLiteral(node.arguments[0]!)
    ) {
      out.push((node.arguments[0] as TS.StringLiteral).text);
    }
    node.forEachChild(visit);
  };
  visit(parsed);
  return out;
}

/**
 * Run one prepared module, and everything it requires, in a CommonJS scope.
 *
 * The module cache is populated with the (empty) exports object BEFORE the body runs, which is what
 * makes an import cycle terminate with a partial namespace instead of recursing — ordinary CJS
 * semantics, and worth doing on purpose rather than discovering.
 */
function execute(
  entry: string,
  emitted: ReadonlyMap<string, string>,
  wiring: ReadonlyMap<string, Map<string, string>>,
  loaded: Map<string, Record<string, unknown>> = new Map(),
): Record<string, unknown> {
  const cached = loaded.get(entry);
  if (cached !== undefined) return cached;

  const js = emitted.get(entry);
  if (js === undefined) throw new ModuleLoadError(`'${entry}' was never prepared`);

  const exports: Record<string, unknown> = {};
  const module = { exports };
  loaded.set(entry, exports);

  const requireFrom = (specifier: string): unknown => {
    if (isBuiltin(specifier)) {
      // Not a hole in anything: SPEC §7.5.4 says outright that the require path is resolution and
      // not containment, and the host is free to supply no builtins at all.
      throw new ModuleLoadError(
        `'${entry}' requires the host module '${specifier}', which this loader does not provide`,
      );
    }
    const target = wiring.get(entry)?.get(specifier);
    if (target === undefined) throw new ModuleLoadError(`'${entry}' requires '${specifier}', which was not prepared`);
    return execute(target, emitted, wiring, loaded);
  };

  // `//# sourceURL` is what puts the module's own path in a stack trace. Without it every failure
  // inside a user function is reported against `<anonymous>`, which is the difference between a
  // usable error and a shrug.
  const wrapper = new Function(
    "exports",
    "require",
    "module",
    "__filename",
    "__dirname",
    `${js}\n//# sourceURL=${entry}`,
  ) as (
    exports: Record<string, unknown>,
    require: (s: string) => unknown,
    module: { exports: Record<string, unknown> },
    filename: string,
    dirname: string,
  ) => void;

  wrapper(exports, requireFrom, module, entry, dirOf(entry));

  // A module that REPLACED `module.exports` rather than adding to it — `module.exports = fn` — is
  // the CJS idiom, and the cache has to end up holding what the module actually produced.
  const produced = module.exports;
  loaded.set(entry, produced);
  return produced;
}
