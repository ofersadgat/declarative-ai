/**
 * The PROMPT side of a call (DESIGN §4.1) — an input shape, which with the output schema
 * makes a `CallSignature`. It mirrors the AI SDK `Prompt` capability surface so no expressiveness is
 * lost, and it is deliberately separate from `LlmCallConfig`: a config is "purely how to decode", a
 * reusable, mergeable, searchable preset.
 */
import type { ModelMessage, SystemModelMessage } from "ai";
import type { JsonSchema, JsonValue } from "@declarative-ai/json";
import type { FileInput } from "./files";

// `CallPromptInput` names these AI SDK types in its PUBLIC shape, so a consumer cannot describe a
// prompt without them. `llm` owns the AI SDK boundary — re-exporting here is what lets a package above
// (promptop) build message arrays while depending only on `llm`, rather than taking its own `ai` dep.
export type { ModelMessage, SystemModelMessage };

/**
 * The prompt inputs for a call, mirroring the AI SDK `Prompt` capability surface so no expressiveness is
 * lost: a `system` prompt (a plain string OR structured system message(s)) plus EITHER a `prompt` (a plain
 * string or a message array) OR a `messages` array — the latter two carry multi-turn conversation and
 * MULTIMODAL content (image/file parts live inside `ModelMessage`). Provide exactly ONE of
 * `prompt`/`messages`; both are optional at the type level so the shape threads cleanly through the layers,
 * and the SDK enforces the "one or the other" rule at the call. NB for the definition to stay serializable,
 * any file/image data inside messages must be a base64 string or URL (not a live `Uint8Array`).
 */
export interface CallPromptInput {
  system?: string | SystemModelMessage | SystemModelMessage[];
  prompt?: string | ModelMessage[];
  messages?: ModelMessage[];
  /** Neutral file/media inputs (pdf/image/audio/video) lowered to provider file parts + merged into the
   *  user turn at the call boundary. Sources are the caller's problem: `data` is bytes, base64, or a
   *  URL — there is no reference form and no store to inject (see `FileInput`). */
  attachments?: FileInput[];
}

/** The plain-text content of a message's `content` (a string, or the text parts of a content array). */
function messageContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (part && typeof part === "object" && typeof (part as { text?: unknown }).text === "string" ? (part as { text: string }).text : ""))
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

/** Normalize a prompt input to its MESSAGE-LIST form: explicit `messages` win, an array prompt IS the
 *  messages, a non-empty string prompt becomes one user turn, nothing ⇒ `[]`. The SINGLE implementation of
 *  the three-way prompt-shape branch (session transcripts, attachment lowering, repair hints). */
export function promptAsMessages(p: CallPromptInput): ModelMessage[] {
  if (p.messages) return p.messages;
  if (Array.isArray(p.prompt)) return p.prompt;
  if (typeof p.prompt === "string" && p.prompt.length > 0) return [{ role: "user", content: p.prompt }];
  return [];
}

/** Extract all plain text from a prompt input (system + prompt/messages) — used for the json-specifier
 *  check (§5.1) and cheap token estimation. Non-text parts (images/files/tool calls) contribute nothing. */
export function promptText(p: CallPromptInput): string {
  const parts: string[] = [];
  if (typeof p.system === "string") parts.push(p.system);
  else if (Array.isArray(p.system)) parts.push(p.system.map((m) => m.content).join("\n"));
  else if (p.system) parts.push(p.system.content);
  if (typeof p.prompt === "string") parts.push(p.prompt);
  else if (Array.isArray(p.prompt)) parts.push(p.prompt.map((m) => messageContentText(m.content)).join("\n"));
  if (p.messages) parts.push(p.messages.map((m) => messageContentText(m.content)).join("\n"));
  return parts.join("\n");
}

/**
 * An input shape plus an output shape — which is what a signature IS, and what makes this the natural
 * name (§5.2). It mirrors `PromptOp` exactly: `input` params + `output` param, with `config` a sibling
 * of both.
 *
 * The output schema goes on the PROMPT side, not the config, for two reasons that are not stylistic:
 * `resolveConfig` merges configs as LAYERS and merging output schemas across a defaults/preset/inline
 * stack is meaningless; and findmyprompt's search point is `LlmParameters = LlmCallConfig & {
 * systemPrompt, userPrompt }`, where the optimizer searches decoding knobs and never the output type.
 */
export type CallSignature<T = JsonValue> = CallPromptInput & {
  /** The output JSON Schema, OR omitted for a TEXT-output call (plain text, no structured output).
   *  It belongs IN the definition because it is declarative and serializable — its absence is why the
   *  old lowering had to smuggle it through `spec.outputSchema` and cast the phantom away. */
  schema?: JsonSchema<T>;
  /** Per-call wall-clock budget (ms). */
  timeoutMs?: number;
};
