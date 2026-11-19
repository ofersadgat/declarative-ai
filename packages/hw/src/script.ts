/**
 * `@declarative-ai/hw/script` — the hooks a workflow script imports (SCRIPTS.md §6, §9, §17).
 *
 * Under the engine, the compiler cuts a script at every call into this module and runs each piece as a
 * state (durable). Under `node`, the same calls run directly, through the standalone host
 * `@declarative-ai/script` installs (non-durable). The module is the one surface both see.
 */
export {
  llm,
  agent,
  input,
  parallel,
  pipeline,
  phase,
  log,
  workflow,
  call,
  recorded,
  now,
  random,
  budget,
  installStandaloneHost,
  standaloneHost,
  INPUT_FUNCTION,
  type AgentOptions,
  type InputAnswer,
  type InputMessage,
  type InputOptions,
  type LlmCall,
  type ScriptHost,
  type WithSession,
} from "./scriptHooks.js";
export type { SessionRef } from "./session.js";
