/**
 * @declarative-ai/validate — the schema CHECKING layer (DESIGN §2, §5).
 *
 * Three things that were previously written twice, or in the wrong place, land here together:
 *
 *  - `subtype.ts` — structural JSON-Schema subtyping, re-homed out of `ops` (a checker is not part of
 *    the op model) and taught that `x-type` is CONSTRAINING (§3.3).
 *  - `checker.ts` — ONE generic binding checker, parameterized by the ref family with injectable
 *    resolution, replacing findmyprompt's `checker.ts` and hw's hand-rolled twin (§6.2).
 *  - `ajv.ts` — ONE ajv wrapper with an injectable `$ref` resolver.
 *
 * This is the ONLY package carrying a heavy dependency (ajv), and nothing below it imports it — which
 * is the point: a structured LLM call runs with `json + llm` and nothing else.
 *
 * It declares no `ExecServices` augmentation: `exec` already names the MINIMAL structural
 * `OutputValidator` seam it consumes (`validateValue`) and never learns the concrete type, which is the
 * §1.2 rule applied in its lighter form. `SchemaValidator` simply implements that seam.
 */
export * from "./subtype";
export * from "./checker";
export * from "./ajv";
