/**
 * Ask claude how much of the account's allowance is spent, WITHOUT spending a turn.
 *
 * A claude process started for nothing else — `-p` with streaming input, the `initialize` handshake,
 * then the `get_usage` control request — answers with every plan window's utilization and reset time,
 * and exits when its input closes. No prompt is ever written, so no model is called.
 *
 * MEASURED 2026-09-24: the Agent SDK's bundled claude (2.1.223) answers `get_usage`; an older installed
 * claude (2.1.142) answers "Unsupported control request subtype: get_usage". Signed out, the answer
 * carries `rate_limits: null`. Every one of those comes back here as `undefined` — a refresh that
 * could not learn anything leaves the board as it was.
 */
import { limitReadingOfClaudeUsage } from "@declarative-ai/agents-api";
import type { LimitReading } from "@declarative-ai/exec";
import { defaultSpawn, type AgentProcess, type SpawnProcess } from "./process.js";

export interface ClaudeUsageProbeOptions {
  /** The claude binary to ask. Default `claude`. */
  command?: string;
  /** The route the reading is reported under. Default `claude-cli`. */
  route?: string;
  /** Process seam (tests inject a fake). */
  spawn?: SpawnProcess;
  /** How long the whole exchange may take. Default 15 s. */
  timeoutMs?: number;
  signal?: AbortSignal;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
}

/** What `get_usage` came back with, parsed — or why there was nothing to parse. */
export interface ClaudeUsageAnswer {
  reading?: LimitReading;
  /** The process refused the question or answered without windows. */
  unavailable?: string;
}

export async function probeClaudeUsage(options: ClaudeUsageProbeOptions = {}): Promise<ClaudeUsageAnswer> {
  const spawn = options.spawn ?? (await defaultSpawn());
  const route = options.route ?? "claude-cli";
  const argv = [options.command ?? "claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"];
  let child: AgentProcess | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const kill = (): void => child?.kill();
  try {
    child = spawn(argv, {
      ...(options.cwd !== undefined ? { cwd: options.cwd } : {}),
      ...(options.env !== undefined ? { env: options.env } : {}),
      keepInputOpen: true,
    });
    const c = child;
    if (c.write === undefined) return { unavailable: "the process seam offers no input channel" };
    if (options.signal?.aborted) return { unavailable: "aborted" };
    options.signal?.addEventListener("abort", kill, { once: true });
    timer = setTimeout(kill, options.timeoutMs ?? 15_000);
    c.write(`${JSON.stringify({ type: "control_request", request_id: "usage_init", request: { subtype: "initialize" } })}\n`);
    c.write(`${JSON.stringify({ type: "control_request", request_id: "usage_1", request: { subtype: "get_usage" } })}\n`);
    for await (const line of c.lines) {
      let msg: { type?: unknown; response?: { request_id?: unknown; subtype?: unknown; response?: unknown; error?: unknown } };
      try {
        msg = JSON.parse(line) as typeof msg;
      } catch {
        continue;
      }
      if (msg.type !== "control_response" || msg.response?.request_id !== "usage_1") continue;
      if (msg.response.subtype !== "success") return { unavailable: typeof msg.response.error === "string" ? msg.response.error : "claude refused the question" };
      const reading = limitReadingOfClaudeUsage(msg.response.response, route);
      return reading !== undefined ? { reading } : { unavailable: "claude answered without any plan windows (signed out, or an API key)" };
    }
    return { unavailable: "claude ended without answering" };
  } catch (e) {
    return { unavailable: (e as Error).message };
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", kill);
    child?.endInput?.();
    child?.kill();
  }
}

/** The reading alone — the shape a limits board's refresh takes. */
export async function readClaudeUsage(options: ClaudeUsageProbeOptions = {}): Promise<LimitReading | undefined> {
  return (await probeClaudeUsage(options)).reading;
}
