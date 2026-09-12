/**
 * CALLABLE subtyping (hw SPEC §6.2) — is a producer of one callable type acceptable where a consumer
 * expects another?
 *
 * A callable's type is its I/O contract (`@declarative-ai/ops`' `CallableSchema`), and the rule is
 * the ordinary one for functions, restated over slots:
 *
 *  - every slot the CONSUMER declares is one the producer accepts, with the consumer's schema a
 *    subschema of the producer's — an argument the consumer will pass must be one the producer
 *    takes, so inputs are CONTRAVARIANT;
 *  - every slot the PRODUCER requires is one the consumer declares — the consumer must be able to
 *    fill it;
 *  - the producer's output is a subschema of the consumer's — outputs are COVARIANT.
 *
 * A consumer with no `input` asks nothing of the producer's parameters, and one with no `output`
 * accepts any return. A producer with no `input` accepts any arguments. Each side that declared
 * nothing has said nothing to disagree with, which is the same rule the binding checker applies to
 * an unconstrained slot.
 */
import type { CallableSchema } from "@declarative-ai/ops";
import { isSubschema, type ResolveRef, type Schema, type SubtypeResult } from "./subtype.js";

const OK: SubtypeResult = { ok: true };
const fail = (reason: string): SubtypeResult => ({ ok: false, reason });

/** True when a schema constrains nothing — absent, or a document with no keys. */
function unconstrained(schema: Schema | undefined): boolean {
  return schema === undefined || Object.keys(schema).length === 0;
}

export function isSubcallable(producer: CallableSchema, consumer: CallableSchema, resolve?: ResolveRef): SubtypeResult {
  if (producer.kind !== consumer.kind) return fail(`expects a ${consumer.kind} but the producer is a ${producer.kind}`);

  if (consumer.input !== undefined && producer.input !== undefined) {
    for (const [name, want] of Object.entries(consumer.input)) {
      const takes = producer.input[name];
      if (takes === undefined) return fail(`the consumer passes '${name}', which the producer does not accept`);
      if (unconstrained(want.schema as Schema | undefined) || unconstrained(takes.schema as Schema | undefined)) continue;
      // CONTRAVARIANT: what the consumer will pass must be acceptable to the producer.
      const r = isSubschema(want.schema as Schema, takes.schema as Schema, resolve);
      if (!r.ok) return fail(`parameter '${name}': ${r.reason}`);
    }
    for (const [name, takes] of Object.entries(producer.input)) {
      if (takes.optional === true || name in consumer.input) continue;
      return fail(`the producer requires '${name}', which the consumer does not pass`);
    }
  }

  const produced = producer.output?.schema as Schema | undefined;
  const wanted = consumer.output?.schema as Schema | undefined;
  if (!unconstrained(produced) && !unconstrained(wanted)) {
    // COVARIANT: what the producer returns must satisfy what the consumer expects back.
    const r = isSubschema(produced!, wanted!, resolve);
    if (!r.ok) return fail(`output: ${r.reason}`);
  }
  return OK;
}
