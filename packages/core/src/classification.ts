/**
 * Error classification for the budget-gated retry loop (§10.4). This module owns ONLY
 * classification + real-error capture; the actual retry (advance `runId`,
 * budget-checked) is the eval loop's, not the provider's or the SDK's — every layer's
 * auto-retry is disabled (`maxRetries: 0`). The kept idea from alitheia `retry.ts` is
 * the classification; the `withRetry` loop is deliberately NOT ported.
 */

/**
 * Failure classification — the basis for every retry decision:
 *  - `"network-retriable"` — a transport-level failure (429 / 5xx / timeout / connection). Retrying the
 *    same request can succeed once the blip clears; honor `retry-after` backoff. This is the ONLY class
 *    an eval loop should auto-retry — it must never re-roll a candidate's output (that would bias scores).
 *  - `"api-retriable"` — the API RESPONDED but the result is unusable in a way a FRESH stochastic run can
 *    fix: schema-validation reject, truncation, unparseable/empty output. Retried only on explicit
 *    caller opt-in, never silently.
 *  - `"permanent"` — deterministic; re-running cannot help (bad input, bad config, unresolved op).
 *
 * Outcome-level classes carried on `ExecFailure` but never produced by `classifyError`
 * (executors set them from their own control flow):
 *  - `"deadline"` — the unit hit its time budget; the surrounding window/run decides to yield.
 *  - `"out-of-credits"` — a budget/wallet reservation was refused; retrying cannot succeed until top-up.
 *  - `"canceled"` — an explicit caller cancel (abort signal); not a failure of the unit itself.
 *  - `"policy-denied"` — the safety policy blocked a required action; deterministic for this policy.
 */
export type ErrorClass =
  | "network-retriable"
  | "api-retriable"
  | "permanent"
  | "deadline"
  | "out-of-credits"
  | "canceled"
  | "policy-denied";

/** The subset `classifyError` can produce (transport/response analysis only). */
export type ClassifiedErrorClass = "network-retriable" | "api-retriable" | "permanent";

interface ErrorLike {
  /** AI SDK `APICallError` exposes the flag as `isRetryable`; some libs use `retryable`. We honor both. */
  isRetryable?: unknown;
  retryable?: unknown;
  status?: unknown;
  statusCode?: unknown;
  code?: unknown;
  name?: unknown;
  message?: unknown;
  cause?: unknown;
  responseHeaders?: unknown;
  headers?: unknown;
}

function statusOf(err: ErrorLike): number {
  const s = typeof err.status === "number" ? err.status : 0;
  const sc = typeof err.statusCode === "number" ? err.statusCode : 0;
  return s || sc;
}

/**
 * Classify an error.
 *
 * Walks the `cause` chain (the AI SDK wraps provider errors), and at each level:
 *   1. Explicit retryable boolean wins — `isRetryable` (AI SDK `APICallError`) or `retryable`.
 *   2. HTTP 429 or 5xx -> network-retriable.
 *   3. Timeout / abort (deadline cutoff, stream cutoff, §6.2) -> network-retriable.
 *   4. Low-level network failure (connection reset/refused, DNS, undici socket) -> network-retriable.
 * Default -> permanent (unknown errors are NOT blindly retried, §10.4).
 *
 * NB: a network failure (e.g. `ECONNRESET`, "fetch failed") arrives from the AI SDK as an
 * `APICallError` with `isRetryable: true` and NO `statusCode` — without (1) reading
 * `isRetryable` and (4) the network-code probe, such a transient blip would fall through to
 * permanent and prematurely end a candidate's retries.
 */
export function classifyError(err: unknown): ErrorClass {
  let cur: unknown = err;
  for (let depth = 0; cur != null && typeof cur === "object" && depth < 6; depth++) {
    const e = cur as ErrorLike;
    const flag = retryableFlag(e);
    if (flag !== undefined) return flag ? "network-retriable" : "permanent";
    const status = statusOf(e);
    if (status === 429) return "network-retriable";
    if (status >= 500 && status < 600) return "network-retriable";
    if (isTimeoutOrAbort(e)) return "network-retriable";
    if (isNetworkError(e)) return "network-retriable";
    cur = e.cause;
  }
  return "permanent";
}

/** Explicit retryable signal at one error level: `isRetryable` (AI SDK) or `retryable`. */
function retryableFlag(e: ErrorLike): boolean | undefined {
  if (typeof e.isRetryable === "boolean") return e.isRetryable;
  if (typeof e.retryable === "boolean") return e.retryable;
  return undefined;
}

/** Node/undici connection-level error codes — these are transient by nature. */
const NETWORK_CODE = /\b(ECONNRESET|ECONNREFUSED|ECONNABORTED|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|ENETUNREACH|ENETDOWN|EHOSTUNREACH|EPIPE|UND_ERR(?:_[A-Z_]+)?)\b/i;

/**
 * A low-level network failure (no HTTP status): connection reset/refused, DNS lookup
 * failure, undici socket error, or the AI SDK's wrapped "fetch failed". Transient (§10.4).
 */
export function isNetworkError(err: unknown): boolean {
  if (err == null || typeof err !== "object") return false;
  const e = err as ErrorLike;
  const code = typeof e.code === "string" ? e.code : "";
  const name = typeof e.name === "string" ? e.name : "";
  const msg = typeof e.message === "string" ? e.message : "";
  if (NETWORK_CODE.test(code) || NETWORK_CODE.test(name) || NETWORK_CODE.test(msg)) return true;
  return /fetch failed|socket hang ?up|network (?:error|failure)|connection (?:error|closed|reset|refused)|terminated|other side closed/i.test(
    msg,
  );
}

export function isRateLimit(err: unknown): boolean {
  if (err == null || typeof err !== "object") return false;
  return statusOf(err as ErrorLike) === 429;
}

export function isTimeoutOrAbort(err: unknown): boolean {
  if (err == null || typeof err !== "object") return false;
  const e = err as ErrorLike;
  const name = typeof e.name === "string" ? e.name : "";
  const msg = typeof e.message === "string" ? e.message.toLowerCase() : "";
  return (
    name === "AbortError" ||
    name === "TimeoutError" ||
    (name === "APICallError" && msg.includes("timeout")) ||
    msg.includes("timeout") ||
    msg.includes("aborted")
  );
}

/** Header keys that name *which* rate-limit dimension tripped (§6.2/§10.4). */
const RATE_LIMIT_HEADER_KEYS = [
  "retry-after",
  "anthropic-ratelimit-requests-remaining",
  "anthropic-ratelimit-requests-reset",
  "anthropic-ratelimit-input-tokens-remaining",
  "anthropic-ratelimit-input-tokens-reset",
  "anthropic-ratelimit-output-tokens-remaining",
  "anthropic-ratelimit-output-tokens-reset",
];

function headersOf(err: ErrorLike): Record<string, string> | undefined {
  const raw = err.responseHeaders ?? err.headers;
  if (raw == null) return undefined;
  if (raw instanceof Headers) return Object.fromEntries(raw.entries());
  if (typeof raw === "object") return raw as Record<string, string>;
  return undefined;
}

/**
 * Pull the real rate-limit signal off a 429 so an all-429 exhaustion records *which*
 * dimension blocked each attempt + its `retry-after`, never a generic "retries
 * exhausted" (§10.4). Returns the subset of known keys that are present.
 */
export function extractRateLimitInfo(err: unknown): Record<string, string> | undefined {
  // Walk the `cause` chain — the AI SDK wraps the provider error, so the headers may sit a
  // level or two down (same reason `classifyError` walks it).
  let cur: unknown = err;
  for (let depth = 0; cur != null && typeof cur === "object" && depth < 6; depth++) {
    const headers = headersOf(cur as ErrorLike);
    if (headers) {
      const lower: Record<string, string> = {};
      for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = String(v);
      const out: Record<string, string> = {};
      for (const key of RATE_LIMIT_HEADER_KEYS) {
        if (key in lower) out[key] = lower[key]!;
      }
      if (Object.keys(out).length > 0) return out;
    }
    cur = (cur as ErrorLike).cause;
  }
  return undefined;
}

/** `retry-after` in milliseconds, if present (seconds or HTTP-date both handled). */
export function retryAfterMs(err: unknown): number | undefined {
  const info = extractRateLimitInfo(err);
  const raw = info?.["retry-after"];
  if (raw == null) return undefined;
  const secs = Number(raw);
  if (Number.isFinite(secs)) return secs * 1000;
  const date = Date.parse(raw);
  if (Number.isFinite(date)) return Math.max(0, date - Date.now());
  return undefined;
}

/**
 * A compact, diagnostic, human-readable description of the *real* underlying error —
 * what gets stored as the error `Text` so a failure is fully costed and explained
 * (§10.4: "capture the real error, always").
 */
export function describeError(err: unknown): string {
  if (err == null) return "unknown error";
  if (typeof err === "string") return err;
  if (typeof err !== "object") return String(err);
  const e = err as ErrorLike;
  const parts: string[] = [];
  const status = statusOf(e);
  if (status) parts.push(`HTTP ${status}`);
  if (typeof e.name === "string" && e.name && e.name !== "Error") parts.push(e.name);
  if (typeof e.message === "string" && e.message) parts.push(e.message);
  const rl = extractRateLimitInfo(err);
  if (rl) parts.push(`rate-limit[${Object.entries(rl).map(([k, v]) => `${k}=${v}`).join(", ")}]`);
  if (parts.length === 0 && e.cause) parts.push(describeError(e.cause));
  return parts.join(": ") || "unknown error";
}
