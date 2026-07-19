/**
 * Minimal structured logger for @declarative-ai/llm — a drop-in for findmyprompt's
 * `lib/logger` at the call sites this package uses (`createLogger(scope, opts?)` →
 * `{debug,info,warn,error}(message, fields?)`). No-ops unless `AI_EXEC_LOG` is set,
 * in which case records go to stderr as one JSON-ish line each.
 */

export interface LogFields {
  [key: string]: unknown;
}

export interface Logger {
  debug(message: string, fields?: LogFields): void;
  info(message: string, fields?: LogFields): void;
  warn(message: string, fields?: LogFields): void;
  error(message: string, fields?: LogFields): void;
}

function emit(level: string, scope: string, message: string, fields?: LogFields): void {
  if (!process.env.AI_EXEC_LOG) return;
  let suffix = "";
  if (fields !== undefined) {
    try {
      suffix = " " + JSON.stringify(fields);
    } catch {
      suffix = " [unserializable fields]";
    }
  }
  console.error(`[${level}] ${scope}: ${message}${suffix}`);
}

export function createLogger(scope: string, _opts?: { tag?: string } & LogFields): Logger {
  return {
    debug: (message, fields) => emit("debug", scope, message, fields),
    info: (message, fields) => emit("info", scope, message, fields),
    warn: (message, fields) => emit("warn", scope, message, fields),
    error: (message, fields) => emit("error", scope, message, fields),
  };
}
