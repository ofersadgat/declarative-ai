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

export class SchemaValidator implements OutputValidator {
  private readonly ajv: Ajv;
  private readonly compiled = new Map<string, ValidateFunction>();
  private readonly registered = new Set<string>();

  constructor(private readonly resolver?: SchemaResolver) {
    // strict:false — our schemas carry harmless extras (title) and we validate against the
    // ORIGINAL (unpatched) schema; we don't want Ajv to throw on unknown-keyword strictness.
    this.ajv = new Ajv({ allErrors: true, strict: false });
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
 */
export function asBoundaryValidator(validator: SchemaValidator): OutputValidator {
  return {
    validateValue(schema: SchemaDocument, value: JsonValue) {
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
