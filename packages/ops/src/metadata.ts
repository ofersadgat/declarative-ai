/**
 * Op metadata (API.md, "Op metadata") — annotations keyed BY an op's identity, never PART of it.
 * findmyprompt's `ref_metadata` pattern ({key, value} rows annotating any ref) as an
 * interface; this module ships the in-memory implementation, findmyprompt's table is the
 * durable instantiation.
 *
 * Typical keys: a resolved registry entry's capabilities (a cache, so gating doesn't
 * re-look-up per call), the binding checker's inferred schemas, provenance. Because metadata
 * is keyed by identity and never part of it, caching anything here can never change what an
 * op IS (content ids and memo keys are unaffected).
 */
import type { JsonValue } from "@declarative-ai/json";
import type { Id } from "./model";

/** How an op is identified for annotation: a content id (id family) or the op OBJECT's
 *  identity (inline family). */
export type OpRef = Id | object;

/**
 * Generic in the annotation VALUE, defaulting to `JsonValue` per the §3.4 policy — generic defaults
 * are `JsonValue`, never `unknown`. The default covers what this is actually for (cached capabilities,
 * inferred schemas, provenance — all JSON-shaped and all storable alongside a durable `ref_metadata`
 * row); a store that genuinely holds live handles names its own `V` rather than making every reader
 * narrow an `unknown`.
 */
export interface OpMetadata<V = JsonValue> {
  get(op: OpRef, key: string): V | undefined;
  set(op: OpRef, key: string, value: V): void;
}

/** In-memory `OpMetadata`: a `Map` over content ids, a `WeakMap` over inline op objects (so
 *  annotations die with the op object — no leak, no identity confusion across runs). */
export class InMemoryOpMetadata<V = JsonValue> implements OpMetadata<V> {
  private readonly byId = new Map<Id, Map<string, V>>();
  private readonly byObject = new WeakMap<object, Map<string, V>>();

  get(op: OpRef, key: string): V | undefined {
    return this.entries(op, false)?.get(key);
  }

  set(op: OpRef, key: string, value: V): void {
    this.entries(op, true)!.set(key, value);
  }

  private entries(op: OpRef, create: boolean): Map<string, V> | undefined {
    if (typeof op === "string") {
      let m = this.byId.get(op);
      if (!m && create) {
        m = new Map();
        this.byId.set(op, m);
      }
      return m;
    }
    let m = this.byObject.get(op);
    if (!m && create) {
      m = new Map();
      this.byObject.set(op, m);
    }
    return m;
  }
}
