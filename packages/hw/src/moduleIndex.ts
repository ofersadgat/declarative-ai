/**
 * The module symbol index — what answers `ReferenceOptions.symbols` (SPEC §7.5.2).
 *
 * `moduleExports` reads ONE module. This assembles those readings into the per-directory lookup the
 * resolver consults, and it exists for one reason: "does anything in this directory provide
 * `text.slug`" cannot be answered by a stat once a module's *contents* supply symbols, and parsing
 * every module on the path at every lookup is not viable.
 *
 * ## Built ahead, queried synchronously
 *
 * `resolveReference` is sync, and parsing is not — the compiler is imported on demand. So the index
 * is BUILT once, asynchronously, and hands back a synchronous lookup. That split is why loading a
 * bundle does not become async: the expensive half happens before the loader runs, and the loader
 * asks a map.
 *
 * ## Approved files only
 *
 * `approved` is not an optimization. If an unapproved module could contribute symbols, then indexing
 * one is how a workflow discovers it exists, and a file dropped into a directory could shadow a
 * symbol before anybody agreed to run it. SPEC §7.5.5's "an unknown file is an unapproved file" has
 * to hold for symbols as well as for loads, and this is where it holds for symbols.
 */
import { sha256Hex } from "@declarative-ai/exec";
import { MODULE_EXTENSIONS, type SymbolIndex, type SymbolLookup, type Vfs } from "./reference.js";
import type * as TS from "typescript";
import { loadCompiler, moduleSymbolsWith, type SymbolTable } from "./moduleExports.js";

/** The compiler namespace, named once so `indexDirectory` can take it. */
type TypeScriptModule = typeof TS;

/** Where a symbol lives: the module contributing it, and the property path inside its namespace. */
interface SymbolSite {
  file: string;
  property: readonly string[];
}

export interface ModuleIndexOptions {
  /** The filesystem the directories are read through — the same seam references resolve against. */
  vfs: Vfs;
  /**
   * Whether this module may contribute at all (SPEC §7.5.5). Absent ⇒ every module may, which is
   * what an in-memory bundle and the tests want; a real host passes its approval store.
   */
  approved?: (file: string) => boolean;
  /**
   * Parsed tables by CONTENT HASH, carried across builds by the caller.
   *
   * The same hash §7.5.5 computes for approval, so a file is hashed once and the result serves
   * integrity and this. A hit means the bytes are unchanged, which is exactly when a re-parse would
   * produce what is already here.
   */
  cache?: Map<string, SymbolTable>;
  onWarn?: (message: string) => void;
}

/**
 * A symbol index that reads each directory on FIRST ASK — one await, then synchronous forever.
 *
 * This is the form a loader actually needs, and the reason is not performance. A state's search
 * `path` is discovered while the tree is being loaded (`resolutionEnvironment`, and a state may
 * declare its own), so the set of directories to index **cannot be enumerated in advance**. An eager
 * index would either miss whatever a per-state `path` names, silently, or force a two-pass load.
 *
 * The single `await` is the compiler import. Reading a directory is synchronous once it is in hand,
 * which is what lets this satisfy `ReferenceOptions.symbols` — a sync seam called from a sync
 * resolver called from a sync `loadBundle`.
 */
export async function createSymbolIndex(options: ModuleIndexOptions): Promise<SymbolIndex> {
  const ts = await loadCompiler();
  const byDir = new Map<string, Map<string, SymbolSite>>();
  return (dir, symbol) => {
    let symbols = byDir.get(dir);
    if (symbols === undefined) {
      symbols = indexDirectory(ts, dir, options);
      byDir.set(dir, symbols);
    }
    return lookup(symbols, symbol);
  };
}

/**
 * Read every module under `dirs` up front.
 *
 * The eager form, for the caller that genuinely wants everything read now — the freeze (§7.5.5),
 * which has to verify every reachable file before anything runs and therefore gains nothing from
 * deferring. Ordinary loading wants {@link createSymbolIndex}.
 */
export async function buildModuleIndex(dirs: readonly string[], options: ModuleIndexOptions): Promise<SymbolIndex> {
  const ts = await loadCompiler();
  const byDir = new Map<string, Map<string, SymbolSite>>();
  for (const dir of dirs) {
    if (byDir.has(dir)) continue; // one path may name a directory twice; index it once
    byDir.set(dir, indexDirectory(ts, dir, options));
  }
  return (dir, symbol) => lookup(byDir.get(dir), symbol);
}

/** One directory's symbols. Synchronous — see {@link loadCompiler} for why that matters. */
function indexDirectory(ts: TypeScriptModule, dir: string, options: ModuleIndexOptions): Map<string, SymbolSite> {
  const symbols = new Map<string, SymbolSite>();
  // Sorted, so which of two files contributing one symbol wins is a property of their NAMES rather
  // than of directory-listing order. Nondeterminism here would be a resolution that changes between
  // machines for no reason anybody could see.
  for (const entry of [...options.vfs.list(dir)].sort()) {
    if (!MODULE_EXTENSIONS.some((ext) => entry.toLowerCase().endsWith(`.${ext}`))) continue;
    const file = `${dir}/${entry}`;
    if (options.approved !== undefined && !options.approved(file)) continue;
    const source = options.vfs.read(file);
    if (source === undefined) continue;

    const key = sha256Hex(source);
    let table = options.cache?.get(key);
    if (table === undefined) {
      table = moduleSymbolsWith(ts, file, source);
      options.cache?.set(key, table);
    }

    for (const [symbol, property] of table) {
      const existing = symbols.get(symbol);
      if (existing !== undefined) {
        // Within one directory there is no path order to appeal to, so this is a genuine ambiguity
        // rather than the deliberate layering the search path expresses.
        options.onWarn?.(
          `'${symbol}' is contributed by both '${existing.file}' and '${file}' — ` +
            `'${existing.file}' wins, so renaming either changes what this symbol means`,
        );
        continue;
      }
      symbols.set(symbol, { file, property });
    }
  }
  return symbols;
}

/** One directory's answer: the site, or what it holds NEARBY so the failure can say something. */
function lookup(symbols: Map<string, SymbolSite> | undefined, symbol: readonly string[]): SymbolLookup {
  if (symbols === undefined) return { found: false };
  const wanted = symbol.join(".");
  const site = symbols.get(wanted);
  if (site !== undefined) return { found: true, file: site.file, property: site.property };

  // Everything under the same head, grouped by the file offering it. A near miss is nearly always a
  // typo or a rename that missed a caller, and naming the neighbours is the whole diagnosis.
  const head = symbol[0];
  if (head === undefined) return { found: false };
  const near = new Map<string, string[]>();
  for (const [candidate, where] of symbols) {
    if (candidate !== head && !candidate.startsWith(`${head}.`)) continue;
    // The intermediate object nodes are not what anybody meant to call; only leaves are suggestions.
    if (candidate === head) continue;
    const list = near.get(where.file);
    if (list === undefined) near.set(where.file, [candidate]);
    else list.push(candidate);
  }
  if (near.size === 0) return { found: false };
  return { found: false, near: [...near].map(([file, list]) => ({ file, symbols: list })) };
}
