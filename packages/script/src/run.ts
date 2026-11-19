/**
 * Run a workflow script as a program (SCRIPTS.md §17): import it, call its default export with the
 * arguments given, under the standalone host — `dai-script my-workflow.ts "hello"` is
 * `node my-workflow.ts` with the hooks wired.
 */
import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { installStandaloneHost, type ScriptHost } from "@declarative-ai/hw/script";
import { createStandaloneHost, defaultExportOf, type StandaloneOptions } from "./standalone.js";

export interface RunScriptOptions extends StandaloneOptions {
  /** Use this host rather than building one — a test's, or a JaiRA-reporting one. */
  host?: ScriptHost;
}

/**
 * Import `file` and call its default export with `args`, under a standalone host built from `options`
 * (or `options.host`), installed for the duration; the previous one is put back afterwards.
 */
export async function runScript(file: string, args: readonly unknown[] = [], options: RunScriptOptions = {}): Promise<unknown> {
  const path = isAbsolute(file) ? file : resolve(options.cwd ?? process.cwd(), file);
  const host = options.host ?? createStandaloneHost({ cwd: options.cwd ?? process.cwd(), ...options });
  const previous = installStandaloneHost(host);
  try {
    const main = defaultExportOf(await (options.importer ?? ((url: string) => import(url)))(pathToFileURL(path).href));
    if (typeof main !== "function") throw new Error(`${file} has no default export — a workflow script exports the function it runs`);
    return await (main as (...a: unknown[]) => unknown)(...args);
  } finally {
    installStandaloneHost(previous);
  }
}
