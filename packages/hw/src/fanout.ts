/**
 * Static FAN-OUT analysis (SPEC §7.3, rule 2).
 *
 * A `blob` producer read by two consumers cannot stay a stream: a stream is read once, so the second
 * consumer would find it drained. The materialization is therefore forced — but WHEN it is forced is
 * decided HERE, at load time, not discovered when a second reader turns up at run time. Fan-out is a
 * property of the DOCUMENT: every consumer of a producer is visible in the wiring, so the loader can
 * simply count them. The engine then drains a fanned-out blob output exactly once, at the producer
 * child's completion, and hands both consumers the bytes.
 *
 * This is the SAME multiple-consumers signal the validator computes when it type-checks every binding
 * against its producer (`validate.ts`); the difference is only that this pass TALLIES the consumers
 * rather than checking each in isolation.
 *
 * The result keys a producer output as `"<childKey>\0<output>"`, or `"<childKey>\0*"` when a whole-child
 * edge is read enough times that every one of its outputs fans out. A SINGLE-consumer producer is
 * deliberately absent: its stream must survive un-drained so it can be piped (§7.4).
 */
import type { InlineFamily, Ref, RefTree } from "@declarative-ai/exec";
import { pathOfRef } from "./lowerExpr.js";
import { consumptionOf, RESOLVER_REFS, WHOLE_CHILD, type LoadedState } from "./format.js";

/** Marks a whole-child edge (`{ child: P }`, no output selected) — it consumes every output of P. */
const WHOLE = WHOLE_CHILD;
const SEP = "\0";

/**
 * Compute the fan-out set for one loaded state, or `undefined` when nothing fans out. A producer output
 * fans out when at least TWO distinct consumers read it, counting a whole-child read as a read of every
 * output.
 */
export function computeFanOut(state: LoadedState): readonly string[] | undefined {
  /** Per-(child,output) distinct-consumer counts, and per-child whole-edge counts (which apply to
   *  EVERY output of that child). Split so a whole read + a specific read of the same output tally to 2. */
  const specific = new Map<string, number>();
  const whole = new Map<string, number>();

  for (const binding of consumerBindings(state)) {
    // One consumer may name a producer more than once (an expression using `children.P.outputs.o`
    // twice); dedupe within the consumer so that is not mistaken for two consumers.
    const consumed = new Set<string>();
    collect(binding, consumed);
    for (const key of consumed) {
      const [child, output] = key.split(SEP) as [string, string];
      if (output === WHOLE) whole.set(child, (whole.get(child) ?? 0) + 1);
      else specific.set(key, (specific.get(key) ?? 0) + 1);
    }
  }

  const out = new Set<string>();
  for (const [key, n] of specific) {
    const child = key.slice(0, key.indexOf(SEP));
    if (n + (whole.get(child) ?? 0) >= 2) out.add(key);
  }
  for (const [child, n] of whole) {
    if (n >= 2) out.add(`${child}${SEP}${WHOLE}`);
  }
  // A SORTED ARRAY, not the `Set` this used to be. A loaded state is a resolved DEFINITION, and a
  // definition has to be plain JSON — `JSON.stringify(new Set([...]))` is `{}`, which loses every
  // entry with no error to notice. Sorted so the serialized form is stable whatever order the walk
  // above happened to add keys in, which is what makes it safe to fold into a content hash.
  return out.size > 0 ? [...out].sort() : undefined;
}

/** Whether a produced output `output` of child `childKey` is fanned out, per a precomputed list. */
export function isFannedOut(fanOut: readonly string[] | undefined, childKey: string, output: string): boolean {
  return fanOut !== undefined && (fanOut.includes(`${childKey}${SEP}${output}`) || fanOut.includes(`${childKey}${SEP}${WHOLE}`));
}

/** Every top-level binding in a state that resolves against the run — each one a CONSUMER. */
function* consumerBindings(state: LoadedState): Iterable<Ref<InlineFamily>> {
  const op = state.operation;
  if (op) for (const p of Object.values(op.input)) if (p.binding) yield p.binding;
  for (const slot of Object.values(state.outputs ?? {})) if (slot.binding) yield slot.binding;
  for (const slot of Object.values(state.inputs ?? {})) if (slot.binding) yield slot.binding;
  for (const child of Object.values(state.children ?? {})) for (const wire of Object.values(child.inputs ?? {})) yield wire;
}

/** Accumulate the `(child, output)` producer references one binding consumes. */
function collect(ref: Ref<InlineFamily>, out: Set<string>): void {
  if ("op" in ref) {
    const producer = ref.op;
    if (typeof producer === "string") {
      // A whole-child edge (`{ child: P }`) — consumes every output of P.
      out.add(`${producer}${SEP}${WHOLE}`);
      return;
    }
    if (producer.kind !== "function") return;
    if (producer.functionRef === RESOLVER_REFS.select) {
      // `{ child: P, output: o }` lowers to a select over the child edge — a read of P.o.
      const value = producer.input.value?.binding;
      const key = producer.input.key?.binding;
      const child = value !== undefined && "op" in value && typeof value.op === "string" ? value.op : undefined;
      const output = key !== undefined && "text" in key ? key.text : undefined;
      if (child !== undefined && output !== undefined) out.add(`${child}${SEP}${output}`);
      else if (value !== undefined) collect(value, out);
      return;
    }
    // A lowered EXPRESSION reads children through the context, not through a child edge — it has to,
    // or it would stop meaning what the interpreter meant (EXPRESSIONS.md §1.3). So the read is
    // recognized structurally here instead: a `context.get("children")` at the bottom of an
    // `op.member` chain IS the same consumption that wiring the child would be.
    //
    // This replaced re-parsing the expression's source text. WHAT each shape consumes is
    // `consumptionOf`'s to say (`format.ts`), next to the namespace vocabulary that defines the
    // shape — rather than a set of path indices here, which is what this was, and which made every
    // namespace segment added anywhere a silent miscount waiting to happen.
    const path = pathOfRef(ref);
    if (path !== undefined) {
      const consumed = consumptionOf(path);
      if (consumed !== undefined) out.add(`${consumed.child}${SEP}${consumed.output}`);
      return;
    }
    // scope/artifact/conversation resolvers name no child; operators and embedded ops carry nested
    // edges, and an operand may be a child read even when the whole node is not a path.
    for (const p of Object.values(producer.input)) if (p.binding) collect(p.binding, out);
    // A lowered CALL keeps its ARGUMENTS in `parameters`, not in the callee's own `input` — so a
    // child read inside one (`classify(children.c.outputs.doc)`) is invisible to a walk that only
    // descends `input`. Missing it under-counts fan-out, and an under-counted blob is never
    // materialized: two readers then race one stream, silently (§1.3).
    for (const p of Object.values(ref.parameters ?? {})) if (p.binding) collect(p.binding, out);
    return;
  }
  if ("refs" in ref) collectTree(ref.refs, out);
}

function collectTree(tree: RefTree<InlineFamily>, out: Set<string>): void {
  if (tree === null || typeof tree !== "object") return;
  if (Array.isArray(tree)) {
    for (const item of tree) collectTree(item, out);
    return;
  }
  // A tree node may be a leaf ref or a plain record; either way, walk its ref-shaped values.
  collect(tree as Ref<InlineFamily>, out);
  for (const sub of Object.values(tree)) collectTree(sub as RefTree<InlineFamily>, out);
}
