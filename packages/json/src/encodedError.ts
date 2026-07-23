import type { ErrorClass } from "./classification";

/**
 * Durable, re-derivable error payloads (extracted from findmyprompt
 * `src/engine/execution/errors.ts`, generalized to the full `ErrorClass` union).
 * Stored as JSON so a resumed run re-reads the classification and reaches the same
 * retry/advance decision it did live; the human `reason` rides along.
 */
export interface EncodedError {
  classification: ErrorClass;
  reason: string;
  /** Server-advised `retry-after` (ms), preserved so a resumed run backs off the same way. */
  retryAfterMs?: number;
}

const CLASSES: ReadonlySet<string> = new Set([
  "network-retriable",
  "api-retriable",
  "permanent",
  "deadline",
  "out-of-credits",
  "canceled",
  "policy-denied",
] satisfies ErrorClass[]);

export function encodeError(err: EncodedError): string {
  return JSON.stringify(err);
}

export function decodeError(text: string): EncodedError | undefined {
  try {
    const o = JSON.parse(text) as { classification?: unknown; reason?: unknown; retryAfterMs?: unknown };
    if (typeof o.classification === "string" && CLASSES.has(o.classification)) {
      return {
        classification: o.classification as ErrorClass,
        reason: typeof o.reason === "string" ? o.reason : "",
        retryAfterMs: typeof o.retryAfterMs === "number" && Number.isFinite(o.retryAfterMs) ? o.retryAfterMs : undefined,
      };
    }
  } catch {
    // not our structured shape
  }
  return undefined;
}
