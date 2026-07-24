/**
 * THE ajv wrapper (API.md, "Schema validation"). findmyprompt's `ajv.ts` and the old `services/validator.ts`
 * were the same wrapper written twice, differing only in how a `$ref` resolves; this is that wrapper,
 * once, with the resolver INJECTED. It is the only module in the workspace that imports ajv, which is
 * what makes `npm i @declarative-ai/llm` install no ajv at all.
 *
 * Two modes:
 *  - **Inline schemas** (`validateValue`) — hierarchical-workflow states and llm-call output
 *    contracts carry their schema documents inline; validators are compiled synchronously and
 *    cached by the schema's content hash. Implements `@declarative-ai/exec`'s `OutputValidator` seam.
 *  - **Store-backed schemas** (`compile`/`validate`) — content-addressed schema artifacts whose
 *    `$ref`s are store ids (not URLs), registered in Ajv under `$id = content hash` and
 *    resolved lazily from an injected resolver — no network parser needed.
 */
import { Ajv, type ValidateFunction } from "ajv";
import { hashCanonical, type JsonValue, type SchemaDocument, type ValidationResult } from "@declarative-ai/json";
import type { OutputValidator } from "@declarative-ai/exec";

export type { ValidationResult };

/** Lazy schema-document lookup by content id (findmyprompt: the artifact store). */
export interface SchemaResolver {
  getSchema(id: string): Promise<SchemaDocument | undefined>;
}

/** A content-addressed json store — the minimal structural surface {@link SchemaValidator.forJsonStore}
 *  adapts into a {@link SchemaResolver} (findmyprompt: the artifact store's `getJson`). */
export interface JsonArtifactSource {
  getJson(id: string): Promise<{ json: unknown } | null | undefined>;
}

export interface SchemaValidatorOptions {
  /**
   * The caller's content-address MINT for schema documents: given a document, the canonical id the
   * caller's store knows it by. When set, a document arriving as a DOCUMENT (the
   * {@link asBoundaryValidator} path) registers and caches under `getId(schema)` — the SAME id a
   * store-backed `validate(id, …)` call would use — so both entry points share one registration
   * namespace and one compiled cache. Absent, document ids derive from `hashCanonical`.
   */
  getId?: (schema: SchemaDocument) => string;
}

export class SchemaValidator implements OutputValidator {
  private readonly ajv: Ajv;
  private readonly compiled = new Map<string, ValidateFunction>();
  private readonly registered = new Set<string>();
  /** See {@link SchemaValidatorOptions.getId}; read by {@link asBoundaryValidator}. */
  readonly getId?: (schema: SchemaDocument) => string;

  constructor(private readonly resolver?: SchemaResolver, opts: SchemaValidatorOptions = {}) {
    // strict:false — our schemas carry harmless extras (title) and we validate against the
    // ORIGINAL (unpatched) schema; we don't want Ajv to throw on unknown-keyword strictness.
    this.ajv = new Ajv({ allErrors: true, strict: false });
    this.getId = opts.getId;
  }

  /**
   * Wire a validator to a content-addressed json store: `$ref`s resolve as store ids, and `getId`
   * — the caller's own content-address mint — keys document registration so every document lands
   * under the SAME id the store knows it by (one namespace across document and `validate(id, …)`
   * entry points). This is the whole store-backed configuration; callers need no subclass.
   */
  static forJsonStore(store: JsonArtifactSource, getId?: (schema: SchemaDocument) => string): SchemaValidator {
    return new SchemaValidator(
      { getSchema: async (id) => (await store.getJson(id))?.json as SchemaDocument | undefined },
      { getId },
    );
  }

  /** Validate a value against an INLINE schema document (no `$ref` store resolution) — the SYNC seam
   *  (`SyncOutputValidator`), which mid-walk consumers (hw slot validation) rely on. A store-backed
   *  document goes through {@link asBoundaryValidator} instead. Compiled validators are cached by the
   *  document's content hash. */
  validateValue(schema: SchemaDocument, value: JsonValue): ValidationResult {
    const id = "inline:" + hashCanonical(schema);
    let fn = this.compiled.get(id);
    if (!fn) {
      fn = this.ajv.compile(schema);
      this.compiled.set(id, fn);
    }
    return this.outcome(fn, value);
  }

  private outcome(fn: ValidateFunction, value: JsonValue): ValidationResult {
    return fn(value) ? { ok: true } : { ok: false, errors: this.ajv.errorsText(fn.errors) };
  }

  /** Compile (and cache) a validator for a stored schema, after registering its `$ref` graph. */
  async compile(schemaId: string, schemaDoc?: unknown): Promise<ValidateFunction> {
    const cached = this.compiled.get(schemaId);
    if (cached) return cached;
    const doc = (schemaDoc ?? (await this.resolver?.getSchema(schemaId))) as Record<string, unknown> | undefined;
    if (doc == null) throw new Error(`SchemaValidator: schema ${schemaId} not found`);
    await this.register(schemaId, doc);
    const fn = this.ajv.getSchema(schemaId) as ValidateFunction | undefined;
    if (!fn) throw new Error(`SchemaValidator: failed to compile schema ${schemaId}`);
    this.compiled.set(schemaId, fn);
    return fn;
  }

  /** Validate a value against a stored schema. */
  async validate(schemaId: string, value: unknown, schemaDoc?: unknown): Promise<ValidationResult> {
    const fn = await this.compile(schemaId, schemaDoc);
    return fn(value) ? { ok: true } : { ok: false, errors: this.ajv.errorsText(fn.errors) };
  }

  /** Human-readable rendering of a failed validator's errors (for the error artifact). */
  errorsText(fn: ValidateFunction): string {
    return this.ajv.errorsText(fn.errors);
  }

  private async register(id: string, doc: Record<string, unknown>): Promise<void> {
    if (this.registered.has(id)) return;
    this.registered.add(id);
    if (!this.ajv.getSchema(id)) this.ajv.addSchema(doc, id);
    for (const refId of collectRefs(doc)) {
      if (this.registered.has(refId)) continue;
      const refDoc = (await this.resolver?.getSchema(refId)) as Record<string, unknown> | undefined;
      if (refDoc) await this.register(refId, refDoc);
    }
  }
}

/**
 * Lift a {@link SchemaValidator} to the BOUNDARY seam (`OutputValidator`, maybe-async): a document
 * whose `$ref` closure needs the injected resolver (content-addressed store ids) compiles through the
 * async store-backed path; a ref-free document (or a warm cache) answers synchronously through
 * `validateValue`. This is the validator an llm-call environment wants when schemas are id-family
 * artifacts — inject it as `env.validator` / `ctx.validator` and the call layer awaits it.
 *
 * With `getId` configured, EVERY document goes through the store-backed path under the caller's
 * minted id (ref-free ones included) — the id then matches the caller's stored artifact, which is
 * what keeps one registration namespace across document and `validate(id, …)` entry points.
 */
export function asBoundaryValidator(validator: SchemaValidator): OutputValidator {
  return {
    validateValue(schema: SchemaDocument, value: JsonValue) {
      const getId = validator.getId;
      if (getId) return validator.validate(getId(schema), value, schema);
      if (collectRefs(schema).size > 0) {
        const id = "ref:" + hashCanonical(schema);
        return validator.validate(id, value, schema);
      }
      return validator.validateValue(schema, value);
    },
  };
}

/** Collect every `$ref` string target reachable in a schema document. */
export function collectRefs(node: unknown, out = new Set<string>()): Set<string> {
  if (Array.isArray(node)) {
    for (const x of node) collectRefs(x, out);
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node)) {
      if (k === "$ref" && typeof v === "string") out.add(v);
      else collectRefs(v, out);
    }
  }
  return out;
}
