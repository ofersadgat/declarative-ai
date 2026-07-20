/**
 * Scoped, hierarchical logger for @declarative-ai/llm — ported from findmyprompt's `lib/logger`
 * (behavior-identical), stripped of the app's server coupling: hierarchical dot-scopes, level
 * filtering, structured fields, tagged-template messages, and a **stderr** default sink (so scripts
 * can still pipe clean JSON on stdout).
 *
 * The sink is swappable via `setLogSink`, and the level policy via `setLevelPolicy` — this is the
 * seam a consumer uses to inject richer logging (e.g. findmyprompt installs a DB batch-flush sink
 * that mirrors records to its `logs` table and correlates them to a run; JaiRA installs its own).
 *
 * Runtime-agnostic (no `process`/DOM assumptions): under Node the default sink writes one line per
 * record to stderr (keeping stdout clean for piping); in a browser or other non-Node runtime it
 * routes to the matching `console` method. The default level is likewise environment-aware — `info`
 * under Node, `warn` elsewhere so a bundled library doesn't flood the devtools console — and is
 * overridable via `LOG_LEVEL` (a Node env var or a `globalThis.LOG_LEVEL` global) or `setMinLevel`.
 * This file has zero dependency on any consumer.
 */
export type LogLevel = "debug" | "info" | "warn" | "error";

const LEVEL_RANK: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

/** One parsed stack frame — the structured form a sink may store. */
export interface StackFrame {
  file: string;
  line: number;
  fn: string;
}

export interface LogRecord {
  level: LogLevel;
  scope: string; // hierarchical dot-path, e.g. "engine.providers.generate"
  message: string;
  fields?: Record<string, unknown>;
  err?: unknown;
  time: number; // epoch ms
  /** Optional classification tag (set per-logger via `createLogger(scope, { tag })`), for
   *  tag-level policy overrides + downstream filtering. */
  tag?: string;
  /** Per-logger structured context merged onto every record (set via `createLogger`/`child`); on an
   *  error record it also carries `{ error: { name, message } }`. */
  metadata?: Record<string, unknown>;
  /** Parsed stack frames: the full error stack on an error record, else the caller frames. */
  frames?: StackFrame[];
}

/** Per-logger context attached to every record it emits (merged down through `child`). */
export interface LoggerOptions {
  tag?: string;
  metadata?: Record<string, unknown>;
}

export type LogSink = (record: LogRecord) => void;

// --- Runtime level policy ---------------------------------------------------
// A pluggable resolver — a consumer can install one (e.g. backed by a runtime config table) so
// per-category min-level + sampling can be tuned without a redeploy. Unset → the global `minLevel`,
// sampling off. Errors bypass sampling.

export interface ResolvedLevel {
  minLevel: LogLevel;
  /** Probability in [0,1] a non-error at/above minLevel is kept. 1 = keep all. */
  samplingRate: number;
}
/** Extra match context the resolver may key on (a record's tag + callsite), beyond the scope. The
 *  `scope` stays the first positional arg so a plain `(scope) => …` policy keeps working. */
export interface LevelMatchContext {
  tag?: string;
  callsite?: string;
}
export type LevelPolicy = (scope: string, ctx?: LevelMatchContext) => ResolvedLevel | undefined;

let levelPolicy: LevelPolicy | undefined;
let sampleRng: () => number = Math.random;
// Capturing the callsite BEFORE the level gate (so a callsite-keyed policy can match) costs a stack
// parse on every call — including dropped ones. We only pay it when a callsite override actually
// exists; the policy installer flips this flag (default off keeps the hot path free).
let needCallsiteForPolicy = false;

/** Install the runtime level policy (a consumer's config-backed resolver). */
export function setLevelPolicy(policy: LevelPolicy | undefined): void {
  levelPolicy = policy;
}
export function resetLevelPolicy(): void {
  levelPolicy = undefined;
  needCallsiteForPolicy = false;
}
/** The policy installer declares whether any callsite-keyed override exists (refreshed with the config). */
export function setPolicyNeedsCallsite(needed: boolean): void {
  needCallsiteForPolicy = needed;
}
/** Inject the sampling RNG (tests pass a deterministic one). */
export function setSampleRng(fn: () => number): void {
  sampleRng = fn;
}

// --- Tagged-template `log` --------------------------------------------------
// `logger.info(log`Picked up ${id} (attempt=${n})`)` — renders the message AND captures the
// interpolated values as structured `fields` (v0, v1, …) for a structured viewer's `data` column.

export interface TemplateLog {
  __tmpl: true;
  message: string;
  fields: Record<string, unknown>;
}
export function log(strings: TemplateStringsArray, ...values: unknown[]): TemplateLog {
  let message = "";
  const fields: Record<string, unknown> = {};
  strings.forEach((s, i) => {
    message += s;
    if (i < values.length) {
      message += String(values[i]);
      fields[`v${i}`] = values[i];
    }
  });
  return { __tmpl: true, message, fields };
}
function isTemplateLog(x: unknown): x is TemplateLog {
  return typeof x === "object" && x !== null && (x as { __tmpl?: unknown }).__tmpl === true;
}

/** Parse a V8 stack string into structured frames (drops the leading message line). Caps at 20 frames. */
export function parseStackFrames(stack: string): StackFrame[] {
  const frames: StackFrame[] = [];
  for (const line of stack.split("\n")) {
    // V8 (Node / Chrome): `    at fn (file:line:col)` or `    at file:line:col`.
    let m = line.match(/^\s*at\s+(?:(.+?)\s+\()?(.+?):(\d+):(\d+)\)?\s*$/);
    if (!m) {
      // Firefox / Safari: `fn@file:line:col` or `@file:line:col`.
      m = line.match(/^\s*(?:([^@\s]+)@)?(.+?):(\d+):(\d+)\s*$/);
      if (!m || !m[2]) continue;
    }
    frames.push({ fn: m[1] ?? "<anonymous>", file: m[2]!, line: Number(m[3]) });
    if (frames.length >= 20) break;
  }
  return frames;
}

/** Caller frames for a non-error record — this module's own frames stripped off. */
function captureFrames(): StackFrame[] {
  const stack = new Error().stack;
  if (!stack) return [];
  return parseStackFrames(stack)
    .filter((f) => !/logger[\\/]index/.test(f.file) && !/[\\/]logger\.[jt]s/.test(f.file))
    .slice(0, 12);
}

function frameToCallsite(f: StackFrame | undefined): string | undefined {
  return f ? `${f.file}:${f.line}` : undefined;
}

function errToObject(err: unknown): unknown {
  if (err instanceof Error) {
    return { name: err.name, message: err.message, stack: err.stack };
  }
  return err;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "[unserializable]";
  }
}

/** True when a Node-style stderr stream is writable (false in browsers, edge/worker runtimes, etc.). */
const canWriteStderr =
  typeof process !== "undefined" &&
  process.stderr != null &&
  typeof process.stderr.write === "function";

function isLogLevel(v: unknown): v is LogLevel {
  return v === "debug" || v === "info" || v === "warn" || v === "error";
}

/** Read a configured level environment-agnostically: `process.env.LOG_LEVEL` under Node, else a
 *  `globalThis.LOG_LEVEL` override (settable from a browser console or bootstrap). */
function readConfiguredLevel(): LogLevel | undefined {
  const fromProcess = typeof process !== "undefined" ? process.env?.LOG_LEVEL : undefined;
  if (isLogLevel(fromProcess)) return fromProcess;
  const fromGlobal = (globalThis as { LOG_LEVEL?: unknown }).LOG_LEVEL;
  return isLogLevel(fromGlobal) ? fromGlobal : undefined;
}

/** Default sink. Node → one line per record on stderr (keeps stdout clean for piping); browser and
 *  other non-Node runtimes → the matching `console` method, same one-line format. */
const consoleSink: LogSink = (r) => {
  const line = formatRecord(r);
  if (canWriteStderr) {
    process.stderr.write(`${line}\n`);
    return;
  }
  if (typeof console === "undefined") return;
  const method =
    r.level === "error" ? console.error
    : r.level === "warn" ? console.warn
    : r.level === "debug" ? (console.debug ?? console.log)
    : (console.info ?? console.log);
  method.call(console, line);
};

let sink: LogSink = consoleSink;
// Environment-aware default: Node logs `info` (server/CLI convention, stderr keeps stdout clean),
// while a browser or other embedding stays at `warn` so a bundled library doesn't flood the devtools
// console. Either is overridable via LOG_LEVEL (env or `globalThis`) or `setMinLevel`.
let minLevel: LogLevel = readConfiguredLevel() ?? (canWriteStderr ? "info" : "warn");

/** Swap the sink (a consumer's DB-backed batch sink installs here). */
export function setLogSink(next: LogSink): void {
  sink = next;
}

/** Restore the default stderr sink (used when uninstalling a custom sink, e.g. between tests). */
export function resetLogSink(): void {
  sink = consoleSink;
}

/** Format a LogRecord the way the default stderr sink would — for sinks that also tee to console. */
export function formatRecord(r: LogRecord): string {
  const extra: Record<string, unknown> = { ...(r.fields ?? {}) };
  if (r.err !== undefined) extra.err = errToObject(r.err);
  const tail = Object.keys(extra).length > 0 ? ` ${safeJson(extra)}` : "";
  return `[${r.level}] [${r.scope}] ${r.message}${tail}`;
}

/** Normalize an error/value into a JSON-storable object (name/message/stack for Errors). */
export function errorToJson(err: unknown): unknown {
  return errToObject(err);
}

/** Set the global minimum level (errors are never dropped by callers; see sampling above). */
export function setMinLevel(level: LogLevel): void {
  minLevel = level;
}

export interface Logger {
  debug(message: string | TemplateLog, fields?: Record<string, unknown>): void;
  info(message: string | TemplateLog, fields?: Record<string, unknown>): void;
  warn(message: string | TemplateLog, fields?: Record<string, unknown>): void;
  error(message: string | TemplateLog, fields?: Record<string, unknown>, err?: unknown): void;
  /** A nested scope: `createLogger("a").child("b")` logs under "a.b". Inherits the parent's tag +
   *  metadata; `opts` overrides the tag and/or merges additional metadata. */
  child(subScope: string, opts?: LoggerOptions): Logger;
}

/** Per-logger metadata + any error info, or undefined when there's nothing to store. */
function buildMetadata(base: Record<string, unknown> | undefined, err: unknown): Record<string, unknown> | undefined {
  const m: Record<string, unknown> = { ...(base ?? {}) };
  if (err instanceof Error) m.error = { name: err.name, message: err.message };
  else if (err !== undefined) m.error = err;
  return Object.keys(m).length > 0 ? m : undefined;
}

export function createLogger(scope: string, opts: LoggerOptions = {}): Logger {
  const emit = (level: LogLevel, rawMessage: string | TemplateLog, fields?: Record<string, unknown>, err?: unknown): void => {
    // A tagged-template message contributes both the rendered string and its captured values.
    const message = isTemplateLog(rawMessage) ? rawMessage.message : rawMessage;
    const merged = isTemplateLog(rawMessage) ? { ...rawMessage.fields, ...fields } : fields;

    // Runtime policy overrides the global min-level + adds sampling; else global default.
    // Tag is free (per-logger); the callsite is captured pre-gate ONLY when a callsite override exists.
    const callsite = needCallsiteForPolicy ? frameToCallsite(captureFrames()[0]) : undefined;
    const resolved = levelPolicy?.(scope, { tag: opts.tag, callsite });
    const effectiveMin = resolved?.minLevel ?? minLevel;
    if (LEVEL_RANK[level] < LEVEL_RANK[effectiveMin]) return;
    if (level !== "error") {
      const rate = resolved?.samplingRate ?? 1;
      if (rate < 1 && sampleRng() >= rate) return; // sampled out (errors always bypass sampling)
    }

    // Frames: the full error stack on an error, else the caller frames.
    const frames = err instanceof Error && err.stack ? parseStackFrames(err.stack) : captureFrames();
    sink({ level, scope, message, fields: merged, err, time: Date.now(), tag: opts.tag, metadata: buildMetadata(opts.metadata, err), frames });
  };
  return {
    debug: (m, f) => emit("debug", m, f),
    info: (m, f) => emit("info", m, f),
    warn: (m, f) => emit("warn", m, f),
    error: (m, f, e) => emit("error", m, f, e),
    child: (sub, subOpts) =>
      createLogger(`${scope}.${sub}`, {
        tag: subOpts?.tag ?? opts.tag,
        metadata: { ...opts.metadata, ...subOpts?.metadata },
      }),
  };
}
