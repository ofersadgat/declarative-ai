/**
 * Neutral file/media inputs (DESIGN §3.7).
 *
 * `BlobStore`, `BlobRef`, and `FileInput.data`'s `{contentHash}`/`{path}` reference forms are GONE.
 * Binary data is a leaf value, so hydration is the ref family's business — a separate injected store
 * next to that was a second mechanism doing the first one's job. What remains is the honest surface:
 *
 * > **Sources are the caller's problem.** The library takes bytes or a stream. URLs, filesystem paths,
 * > and base64 are resolved by the caller BEFORE the API is called.
 *
 * That is what keeps `json`, `ops`, and `llm` free of `fetch` and `node:fs`, and it is why `BlobStore`
 * genuinely vanishes rather than being renamed.
 *
 * NB on streaming (§7.4): `DataContent` in the AI SDK is `string | Uint8Array | ArrayBuffer | Buffer` —
 * there is NO `ReadableStream`. The SDK cannot take a streaming file input, so the attachment path
 * materializes regardless. Streaming inputs pay off for FUNCTION ops (pipe to a file, a hash, a
 * subprocess, an upload) and for outputs; it is recorded here so "stream to an LLM without
 * materializing" is not written down as a motivating example it cannot currently be.
 */

/**
 * A file/media input (document/pdf/image/audio/video). The bytes travel INLINE — as raw bytes, a
 * base64 string, or a URL the provider fetches itself. Lowered to the provider's file/image message
 * part at the call boundary.
 */
export interface FileInput {
  /** IANA media type, e.g. `application/pdf`, `image/png`, `audio/mp3`. */
  mediaType: string;
  filename?: string;
  /** Exactly one source: raw bytes, inline `base64`, or a `url` the provider fetches. */
  data: Uint8Array | { base64: string } | { url: string };
}

/** A FILE the model GENERATED (image/audio/…). It lands in a `blob`-kind output slot — not in a
 *  parallel `artifacts` channel on the outcome, which is exactly what §7.1 removed. */
export interface GeneratedFile {
  mediaType: string;
  bytes: Uint8Array;
}
