/**
 * What a js/ts module CONTRIBUTES to the search path (SPEC §7.5.2).
 *
 * A document contributes exactly one symbol — its own path — which is why a document can be located
 * by matching a reference against a directory listing. A module contributes a SET, and for an object
 * export the filename contributes nothing at all, so the only way to know what is in a module is to
 * read it. That is this file's whole job: source in, symbol table out.
 *
 * ## The two coordinates
 *
 * Every entry maps a **fully-qualified symbol** — what an expression writes — to a **property path**
 * — where the value sits once the module is loaded. They differ, and conflating them is the bug this
 * module exists to avoid:
 *
 * ```text
 * export default { text: { slug } }   symbol `text.slug`   property ["default", "text", "slug"]
 * export const text = { slug }        symbol `text.slug`   property ["text", "slug"]
 * export default function () {}       symbol <filename>    property ["default"]
 * export function helper() {}         symbol `helper`      property ["helper"]
 * ```
 *
 * The property path is relative to the module's **export namespace** — `default` plus the named
 * exports — which is exactly the shape a CommonJS `module.exports` has after transpilation, so the
 * loader reads a value out with the same `selectProperty` a JSON document's property goes through.
 *
 * ## Why a parse and not a `Program`
 *
 * Contributed symbols are SYNTACTIC: they are the keys an author wrote, not the type of anything.
 * A full `Program` needs module resolution, a tsconfig, and every transitively imported file, and it
 * answers a question this does not ask. Only the *signature* of a symbol actually called needs the
 * checker (SPEC §7.5.2's parameter table), and that happens later, for the one symbol that was
 * reached — not for every symbol on the path.
 */
import type * as TS from "typescript";

/** A module's contribution: fully-qualified symbol → property path in the export namespace. */
export type SymbolTable = ReadonlyMap<string, readonly string[]>;

/**
 * The TypeScript parser, imported on FIRST USE and never at module load.
 *
 * `typescript` is several megabytes and only a workflow that defines functions in a module pays for
 * it — the same reason `loadBundleFromDir` reaches for `node:fs` dynamically. Importing it at the
 * top would put the compiler in the dependency graph of every consumer of this package, including
 * the ones that never see a `.ts` file.
 */
let compiler: Promise<typeof TS> | undefined;
function typescript(): Promise<typeof TS> {
  compiler ??= import("typescript").then((m) => (m as unknown as { default?: typeof TS }).default ?? (m as unknown as typeof TS));
  return compiler;
}

/**
 * Load the parser and hand it back — the ONE await a lazy index needs.
 *
 * Reading a module is synchronous once the compiler is in hand: `createSourceFile` is a plain
 * function, and everything {@link moduleSymbolsWith} does is a walk over its result. Only the
 * dynamic `import` is async. Separating the two is what lets a directory be indexed on first ask,
 * inside a synchronous resolver, rather than requiring every directory to be enumerated in advance
 * (DESIGN §7.1) — which cannot be done at all, because a state's search `path` is discovered while
 * the tree is being loaded.
 */
export function loadCompiler(): Promise<typeof TS> {
  return typescript();
}

/** `functions/confidence.ts` → `confidence`. The name a default-export FUNCTION is called by. */
export function moduleStem(path: string): string {
  const base = path.replace(/\\/g, "/").split("/").pop() ?? path;
  return base.replace(/\.(ts|js|mts|cts|mjs|cjs|tsx|jsx)$/i, "");
}

/**
 * Read the symbols a module contributes.
 *
 * `path` is used for two things and neither is resolution: the stem names a default-export function,
 * and the extension tells the parser which dialect it is reading.
 */
export async function moduleSymbols(path: string, source: string): Promise<SymbolTable> {
  return moduleSymbolsWith(await typescript(), path, source);
}

/** {@link moduleSymbols} with the compiler already in hand — the synchronous core. */
export function moduleSymbolsWith(ts: typeof TS, path: string, source: string): SymbolTable {
  const file = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, /* setParentNodes */ true);
  const out = new Map<string, readonly string[]>();

  /** Top-level `const`/`function`/`class` values, so `export default table` can be followed once. */
  const locals = new Map<string, TS.Node>();
  for (const statement of file.statements) {
    if (ts.isVariableStatement(statement)) {
      for (const decl of statement.declarationList.declarations) {
        if (ts.isIdentifier(decl.name) && decl.initializer) locals.set(decl.name.text, decl.initializer);
      }
    } else if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) && statement.name) {
      locals.set(statement.name.text, statement);
    }
  }

  /** Record `symbol` at `property`, then recurse into an object literal's keys. */
  const contribute = (symbol: readonly string[], property: readonly string[], value: TS.Node | undefined): void => {
    if (symbol.length > 0) out.set(symbol.join("."), property);
    const resolved = value !== undefined && ts.isIdentifier(value) ? locals.get(value.text) : value;
    if (resolved === undefined || !ts.isObjectLiteralExpression(resolved)) return;
    for (const member of resolved.properties) {
      const key = keyOf(ts, member);
      if (key === undefined) continue;
      // A spread contributes whatever it spreads, which is a value question rather than a syntactic
      // one — so it is skipped rather than guessed at.
      const init = ts.isPropertyAssignment(member)
        ? member.initializer
        : ts.isShorthandPropertyAssignment(member)
          ? member.name
          : undefined;
      contribute([...symbol, key], [...property, key], init);
    }
  };

  const exported = (node: TS.Node): boolean =>
    ts.canHaveModifiers(node) && (ts.getModifiers(node)?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword) ?? false);
  const isDefault = (node: TS.Node): boolean =>
    ts.canHaveModifiers(node) && (ts.getModifiers(node)?.some((m) => m.kind === ts.SyntaxKind.DefaultKeyword) ?? false);

  let sawExport = false;
  for (const statement of file.statements) {
    // `export default <expr>`
    if (ts.isExportAssignment(statement) && !statement.isExportEquals) {
      sawExport = true;
      const value = statement.expression;
      const target = ts.isIdentifier(value) ? (locals.get(value.text) ?? value) : value;
      if (ts.isObjectLiteralExpression(target)) {
        // The object itself has no name to be called by: only its keys are symbols.
        contribute([], ["default"], target);
      } else {
        // A function, a class, anything else — named by the FILE, which is the one case where the
        // filename is the symbol.
        out.set(moduleStem(path), ["default"]);
      }
      continue;
    }
    // `export default function foo() {}` / `export default class Foo {}`
    if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) && exported(statement) && isDefault(statement)) {
      sawExport = true;
      out.set(moduleStem(path), ["default"]);
      continue;
    }
    // `export function helper() {}` / `export class Helper {}`
    if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) && exported(statement) && statement.name) {
      sawExport = true;
      contribute([statement.name.text], [statement.name.text], undefined);
      continue;
    }
    // `export const text = { slug }`
    if (ts.isVariableStatement(statement) && exported(statement)) {
      sawExport = true;
      for (const decl of statement.declarationList.declarations) {
        if (!ts.isIdentifier(decl.name)) continue; // a destructuring export names no single value
        contribute([decl.name.text], [decl.name.text], decl.initializer);
      }
      continue;
    }
    // `export { a, b as c }` — the local's shape, published under the exported name.
    if (ts.isExportDeclaration(statement) && statement.exportClause && ts.isNamedExports(statement.exportClause)) {
      sawExport = true;
      for (const element of statement.exportClause.elements) {
        const local = (element.propertyName ?? element.name).text;
        const name = element.name.text;
        contribute([name], [name], locals.get(local));
      }
      continue;
    }
  }

  // "If there is no export, then the local declarations are exported" — a script of plain helpers is
  // a library too, and requiring ceremony to say so would make the commonest scratch file unusable.
  if (!sawExport) {
    for (const [name, value] of locals) {
      contribute([name], [name], ts.isFunctionDeclaration(value) || ts.isClassDeclaration(value) ? undefined : value);
    }
  }

  return out;
}

/** The literal key a property assignment writes, or `undefined` for a computed or spread one. */
function keyOf(ts: typeof TS, member: TS.ObjectLiteralElementLike): string | undefined {
  const name = (member as { name?: TS.PropertyName }).name;
  if (name === undefined) return undefined;
  if (ts.isIdentifier(name) || ts.isStringLiteral(name)) return name.text;
  if (ts.isNumericLiteral(name)) return name.text;
  // A computed key is not knowable from the syntax, and guessing would contribute a symbol that does
  // not exist — worse than contributing nothing, because the miss would be silent.
  return undefined;
}
