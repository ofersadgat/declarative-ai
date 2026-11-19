/**
 * Run a workflow script as a program (SCRIPTS.md §17): import it, call its default export with the
 * arguments given, under the standalone host — `dai-script my-workflow.ts "hello"` is
 * `node my-workflow.ts` with the hooks wired.
 */
import { isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { installStandaloneHost, type ScriptHost } from "@declarative-ai/hw/script";
import { discoverReport, type ReportedRunInfo } from "./report.js";
import { createStandaloneHost, defaultExportOf, type StandaloneOptions } from "./standalone.js";

export interface RunScriptOptions extends StandaloneOptions {
  /** Use this host rather than building one — a test's. */
  host?: ScriptHost;
  /**
   * Whether to report the run to an engine that takes reports (SCRIPTS.md §17.4) — `true` (default)
   * finds one (`findEndpoints`), `false` never does. A `report` already given is used as it is.
   */
  reporting?: boolean;
  /** Where discovery reads `$DAI_HOME`, `$DAI_SCRIPT_ENDPOINT` and `$DAI_SCRIPT_REPORT`. Default: `process.env`. */
  env?: NodeJS.ProcessEnv;
}

/**
 * Import `file` and call its default export with `args`, under a standalone host built from `options`
 * (or `options.host`), installed for the duration; the previous one is put back afterwards.
 */
export async function runScript(file: string, args: readonly unknown[] = [], options: RunScriptOptions = {}): Promise<unknown> {
  const cwd = options.cwd ?? process.cwd();
  const path = isAbsolute(file) ? file : resolve(cwd, file);
  const mod = await (options.importer ?? ((url: string) => import(url)))(pathToFileURL(path).href);
  const main = defaultExportOf(mod);
  if (typeof main !== "function") throw new Error(`${file} has no default export — a workflow script exports the function it runs`);
  // Reported to an engine that takes reports, when one is listening (§17.4) — and run as a plain
  // program either way.
  const meta = (mod as { meta?: unknown } | null)?.meta;
  const report =
    options.report ??
    (options.host === undefined && options.reporting !== false
      ? await discoverReport({ file: path, args, cwd, pid: process.pid, ...(meta !== null && typeof meta === "object" ? { meta: meta as ReportedRunInfo["meta"] } : {}) }, options.env)
      : undefined);
  const write = options.write ?? ((line: string) => void process.stderr.write(`${line}\n`));
  if (report !== undefined && options.report === undefined) write(`▸ reported to ${report.name}${report.url !== undefined ? ` — ${report.url}` : ""}`);
  const host = options.host ?? createStandaloneHost({ cwd, ...options, ...(report !== undefined ? { report } : {}) });
  const previous = installStandaloneHost(host);
  try {
    const value = await (main as (...a: unknown[]) => unknown)(...args);
    await report?.end({ value });
    return value;
  } catch (e) {
    await report?.end({ error: (e as Error)?.message ?? String(e) });
    throw e;
  } finally {
    installStandaloneHost(previous);
    if (options.report === undefined) report?.close();
  }
}

/**
 * The host `import "@declarative-ai/script"` installs for a script run with plain `node` (SCRIPTS.md
 * §17.4): a standalone host that, at the run's FIRST call, looks for an engine that takes reports and,
 * finding one, reports to it from then on — the run named by `process.argv`, ended when the process is
 * about to exit (one that exits on its own, or fails, ends without saying so, which the engine reads as
 * such). With none, it is the standalone host as it always was.
 */
export function createDiscoveringHost(options: StandaloneOptions & { env?: NodeJS.ProcessEnv; reporting?: boolean } = {}): ScriptHost {
  let ready: Promise<ScriptHost> | undefined;
  let plain: ScriptHost | undefined;
  const cwd = options.cwd ?? process.cwd();
  const plainHost = (): ScriptHost => (plain ??= createStandaloneHost({ cwd, ...options }));
  const host = (): Promise<ScriptHost> =>
    (ready ??= (async () => {
      if (options.reporting === false) return plainHost();
      const file = process.argv[1] !== undefined ? resolve(cwd, process.argv[1]) : cwd;
      const report = await discoverReport({ file, args: process.argv.slice(2), cwd, pid: process.pid }, options.env).catch(() => undefined);
      if (report === undefined) return plainHost();
      (options.write ?? ((line: string) => void process.stderr.write(`${line}\n`)))(`▸ reported to ${report.name}${report.url !== undefined ? ` — ${report.url}` : ""}`);
      process.once("beforeExit", () => {
        void report.end({ value: null }).finally(() => report.close());
      });
      return (plain = createStandaloneHost({ cwd, ...options, report }));
    })());
  return {
    llm: async (call, schema) => (await host()).llm(call, schema),
    agent: async (prompt, opts, schema) => (await host()).agent(prompt, opts, schema),
    input: async (opts) => {
      const h = await host();
      if (h.input === undefined) throw new Error("this host reads no input");
      return h.input(opts);
    },
    workflow: async (ref, args) => (await host()).workflow(ref, args),
    call: async (ref, args) => (await host()).call(ref, args),
    recorded: async (name, fn) => (await host()).recorded(name, fn),
    // What does not wait goes to the host there is: reported once the run has found its endpoint.
    phase: (title) => (plain ?? plainHost()).phase(title),
    log: (message) => (plain ?? plainHost()).log(message),
    now: () => Date.now(),
    random: () => Math.random(),
    budget: {
      total: null,
      spent: () => (plain ?? plainHost()).budget.spent(),
      remaining: () => Number.POSITIVE_INFINITY,
    },
  };
}
