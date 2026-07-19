/**
 * Injected STORE seams (declarative-ai environment). Both are optional — absent means the capability is
 * simply unavailable (a `sessionId` with no session store, or a blob ref with no blob store, is an error at
 * resolve/execute time, never a silent no-op). Kept in core as pure interfaces (no AI-SDK types) so the
 * engine, the LLM layer, and consumers all share them.
 *
 * The two are deliberately separate: a BlobStore is IMMUTABLE + content-addressed (files by hash); a
 * SessionStore is MUTABLE + keyed by a logical id (a conversation's transcript, or a provider handle).
 */

/**
 * The state a session accumulates, keyed by a LOGICAL session id. Client-managed conversations store the
 * `messages` transcript (the LLM layer treats these as AI-SDK `ModelMessage`s — typed `unknown[]` here to
 * keep core AI-SDK-free); a provider-side (stateful) executor instead stores the opaque `providerSessionId`
 * handle it resumes. A logical id NEVER carries the provider handle in the portable declaration — it lives
 * here, mapped from the logical id.
 */
export interface SessionState {
  /** Client-managed conversation transcript (prior turns), as AI-SDK `ModelMessage`s. */
  messages?: unknown[];
  /** Provider-assigned session handle to resume (for a stateful executor). */
  providerSessionId?: string;
}

/** A mutable, logical-id-keyed session store. Both methods may be sync or async. */
export interface SessionStore {
  get(logicalId: string): SessionState | undefined | Promise<SessionState | undefined>;
  put(logicalId: string, state: SessionState): void | Promise<void>;
}

/** A plain in-memory session store. */
export class MapSessionStore implements SessionStore {
  private readonly map = new Map<string, SessionState>();
  get(logicalId: string): SessionState | undefined {
    return this.map.get(logicalId);
  }
  put(logicalId: string, state: SessionState): void {
    this.map.set(logicalId, state);
  }
}

/** A reference to a blob: by content hash (immutable, memo-sound), URL, or workspace path. */
export interface BlobRef {
  contentHash?: string;
  url?: string;
  path?: string;
}

/**
 * A neutral, SERIALIZABLE file/media input (document/pdf/image/audio/video). The bytes travel as inline
 * base64, a URL, or a REFERENCE (content hash / workspace path) resolved via the injected {@link BlobStore}
 * at call time — so large media stays out of the declaration + memo key. Lowered to the provider's file/
 * image message part by `@declarative-ai/llm`.
 */
export interface FileInput {
  /** IANA media type, e.g. `application/pdf`, `image/png`, `audio/mp3`. */
  mediaType: string;
  filename?: string;
  /** Exactly one source: inline `base64`, a `url`, or a `contentHash`/`path` reference (needs a BlobStore). */
  data: { base64: string } | { url: string } | { contentHash: string } | { path: string };
}

/**
 * A content-addressed blob store for file/media I/O (Phase 5). `load` resolves a reference to bytes or a
 * URL the provider can fetch; `put` stores bytes and returns their content hash (so large media is
 * referenced by hash — small + memo-sound — rather than inlined into the declaration/memo key).
 */
export interface BlobStore {
  load(ref: BlobRef): Promise<{ bytes?: Uint8Array; url?: string }>;
  put(bytes: Uint8Array, mediaType: string): Promise<{ contentHash: string }>;
}
