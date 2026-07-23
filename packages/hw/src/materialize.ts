/**
 * Stream MATERIALIZATION for `blob` leaves (SPEC §7.3; DESIGN §3.7).
 *
 * A `blob` leaf holds `Bytes = Uint8Array | ByteStream` (ops `model.ts`). A `ByteStream` is an async
 * input; the whole point of the streaming form is to AVOID draining where it can be piped straight to a
 * function-op impl (a file, a hash, a subprocess — §7.4). Where materialization IS required — hashing
 * for a memo key, a bind-time-known fan-out, or storing a value as a run RESULT — the stream is drained
 * to its `Uint8Array` and the resolved value is upgraded IN PLACE by its caller. This module owns the
 * drain; the engine/executor own the three decisions to call it.
 *
 * Three properties this must hold, because the source is a single-use resource:
 *  - CONCURRENT drains share ONE read. Two consumers of one producer must never both `getReader()` — a
 *    stream can only be read once. The in-flight promise is cached ON THE LEAF (a `WeakMap` keyed by the
 *    stream object), so a second `materialize` of the same stream awaits the first drain instead of
 *    acquiring a second reader that would see nothing.
 *  - CANCELLATION propagates. The run's `abortSignal` is tied to `reader.cancel()`, or an abandoned drain
 *    leaks the source; and the drain then FAILS (non-retriably) rather than returning the truncated bytes
 *    read so far.
 *  - The reader lock is RELEASED on every exit — success, error, abort — so the stream is never left
 *    locked behind a failed drain.
 */
import type { Bytes, ByteStream } from "@declarative-ai/exec";

/**
 * A drain that failed. Carries a `<context>: <underlying message>` reason and is treated as a
 * NON-RETRIABLE (permanent) failure by the engine: a consumed stream cannot be re-read, so a later
 * attempt would only find the same drained/canceled source. The class is the signal the engine matches
 * to classify the failure `permanent` rather than guessing from `err.name`.
 */
export class MaterializeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MaterializeError";
  }
}

/** Structural `ByteStream` test — the same `getReader`-presence check `exec`'s memo hasher uses, so the
 *  two agree on what counts as "still a live stream". */
export function isByteStream(v: unknown): v is ByteStream {
  return v !== null && typeof v === "object" && typeof (v as { getReader?: unknown }).getReader === "function";
}

/** The in-flight drain for a given stream leaf — the dedup window between "started reading" and
 *  "bytes ready". Keyed weakly so a materialized stream takes its entry with it. A settled entry (bytes
 *  OR a `MaterializeError`) is kept: re-draining a consumed stream can only fail, so a cached rejection
 *  is the correct, non-masking answer to a repeat call. */
const inFlight = new WeakMap<ByteStream, Promise<Uint8Array>>();

/**
 * Drain `value` to a `Uint8Array`. A `Uint8Array` is returned as-is (idempotent — re-materializing an
 * already-materialized leaf is a no-op). A `ByteStream` is drained once, its in-flight promise shared by
 * concurrent callers via {@link inFlight}. `context` names the op+slot for a drain-failure message.
 */
export function materialize(value: Bytes, signal: AbortSignal | undefined, context: string): Promise<Uint8Array> {
  if (value instanceof Uint8Array) return Promise.resolve(value);
  const existing = inFlight.get(value);
  if (existing) return existing;
  const drained = drain(value, signal, context);
  inFlight.set(value, drained);
  return drained;
}

async function drain(stream: ByteStream, signal: AbortSignal | undefined, context: string): Promise<Uint8Array> {
  const reader = stream.getReader();
  // Tie the run's cancellation to the reader: an aborted run must release the source, not merely stop
  // awaiting it. `cancel` rejections are swallowed — the drain is already failing and the abort reason,
  // not a cancel hiccup, is what surfaces.
  const onAbort = (): void => void reader.cancel(reasonOf(signal)).catch(() => {});
  if (signal) {
    if (signal.aborted) onAbort();
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      // Abort makes `cancel()` resolve the next `read()` with `{ done: true }`, so the loop would
      // otherwise exit CLEANLY with the truncated prefix. Checking the signal turns that into the
      // failure §7.3 requires: a canceled drain fails, it does not return half a file.
      const { done, value } = await reader.read();
      if (signal?.aborted) throw abortError(signal);
      if (done) break;
      if (value !== undefined) {
        chunks.push(value);
        total += value.length;
      }
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  } catch (e) {
    // The underlying error's OWN message, prefixed with the op+slot context — the same `<context>:
    // <message>` shape the engine formats every operation failure with.
    throw new MaterializeError(`${context}: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    if (signal) signal.removeEventListener("abort", onAbort);
    // Release on EVERY path — a stream left locked behind a failed drain can never be canceled or reread.
    reader.releaseLock();
  }
}

function reasonOf(signal: AbortSignal | undefined): unknown {
  return signal !== undefined && "reason" in signal ? (signal as { reason?: unknown }).reason : undefined;
}

function abortError(signal: AbortSignal): Error {
  const reason = reasonOf(signal);
  return reason instanceof Error ? reason : new Error("the drain was aborted");
}
