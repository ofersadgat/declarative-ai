import { Agent, setGlobalDispatcher } from "undici";
import { createLogger } from "./logger";

const log = createLogger("engine.providers.dispatcher");

/**
 * A single long generation must not be killed by undici's default socket timeouts
 * (§5.1). We install a global undici dispatcher with large header/body timeouts so
 * Node's global `fetch` (which the AI SDK providers use) inherits them — no custom
 * `fetch` wiring needed. The actual hard ceiling on a call is the §6.2 deadline-driven
 * `AbortSignal`, not the socket.
 *
 * Idempotent: only installs once per process.
 */
let installed = false;

export interface DispatcherOptions {
  /** Idle timeout waiting for response headers (ms). Default 15 min. */
  headersTimeoutMs?: number;
  /** Idle timeout waiting for the response body (ms). Default 15 min. */
  bodyTimeoutMs?: number;
}

export function installLongTimeoutDispatcher(opts: DispatcherOptions = {}): void {
  if (installed) return;
  const { headersTimeoutMs = 15 * 60_000, bodyTimeoutMs = 15 * 60_000 } = opts;
  setGlobalDispatcher(
    new Agent({ headersTimeout: headersTimeoutMs, bodyTimeout: bodyTimeoutMs }),
  );
  installed = true;
  log.debug("installed long-timeout undici dispatcher", { headersTimeoutMs, bodyTimeoutMs });
}
