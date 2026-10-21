/**
 * Pictures in a tool's result, for the MODEL to see.
 *
 * A {@link Tool} answers a JSON value, and JSON has no picture in it: a screenshot handed back as a
 * base64 string reaches the model as a megabyte of text it cannot see. So a tool that has pictures puts
 * them under one key of its result — `{ …, "$images": [{ mimeType, data }] }`, `data` base64 — and every
 * route that hands a tool's result to a model takes them out ({@link splitToolImages}) and sends them as
 * images beside the rest: the MCP bridge as `image` content (`agents-api`'s `runInjectedTool`), the
 * in-process route as `image-data` parts (`llm`'s tool set). The value without them is what the call
 * answered, and what the model reads as text.
 *
 * A key with a `$`, because nothing a tool means as data is spelt that way, and a route that does not
 * know the key still passes the result on whole rather than dropping it.
 */
import type { JsonValue } from "./json.js";

/** The key a tool's result carries its pictures under. */
export const TOOL_IMAGES = "$images";

/** One picture: its media type and its bytes, base64. */
export interface ToolImage {
  mimeType: string;
  data: string;
}

/**
 * A tool's result, split: the value without its pictures, and the pictures. A result with none —
 * anything that is not an object, or an object without a well-formed `$images` — comes back as it was,
 * with no pictures.
 */
export function splitToolImages(output: unknown): { value: unknown; images: ToolImage[] } {
  if (output === null || typeof output !== "object" || Array.isArray(output)) return { value: output, images: [] };
  const record = output as Record<string, unknown>;
  const raw = record[TOOL_IMAGES];
  if (!Array.isArray(raw)) return { value: output, images: [] };
  const images = raw.flatMap((item): ToolImage[] => {
    if (item === null || typeof item !== "object") return [];
    const { mimeType, data } = item as { mimeType?: unknown; data?: unknown };
    return typeof mimeType === "string" && mimeType.startsWith("image/") && typeof data === "string" && data.length > 0 ? [{ mimeType, data }] : [];
  });
  const { [TOOL_IMAGES]: _dropped, ...value } = record;
  return { value: value as JsonValue, images };
}
