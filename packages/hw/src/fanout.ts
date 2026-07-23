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
import { parseExpression, referencesOf } from "./expr";
import { RESOLVER_REFS, type LoadedState } from "./format";

/** Marks a whole-child edge (`{ child: P }`, no output selected) — it consumes every output of P. */
const WHOLE = "*";
const SEP = "\0";

/**
 * Compute the fan-out set for one loaded state, or `undefined` when nothing fans out. A producer output
 * fans out when at least TWO distinct consumers read it, counting a whole-child read as a read of every
 * output.
 */
export function computeFanOut(state: LoadedState): ReadonlySet<string> | undefined {
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
  return out.size > 0 ? out : undefined;
}

/** Whether a produced output `output` of child `childKey` is fanned out, per a precomputed set. */
export function isFannedOut(fanOut: ReadonlySet<string> | undefined, childKey: string, output: string): boolean {
  return fanOut !== undefined && (fanOut.has(`${childKey}${SEP}${output}`) || fanOut.has(`${childKey}${SEP}${WHOLE}`));
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
    if (producer.functionRef === RESOLVER_REFS.expr) {
      // An `{ expr }` leaf reads children through the DSL — the same edge as wiring it (§7.2).
      const src = producer.input.source?.binding;
      if (src !== undefined && "text" in src) {
        try {
          for (const path of referencesOf(parseExpression(src.text))) {
            if (path[0] === "children" && path[1] !== undefined) {
              if (path[2] === "outputs" && path[3] !== undefined) {
                out.add(`${path[1]}${SEP}${path[3]}`); // a specific output — the one blob consumed
              } else if (path[2] !== "outcome") {
                // A bare/coarse child reference consumes EVERY output. `children.P.outcome` is the
                // termination status string, not an output, so it consumes NONE — counting it as a whole
                // read would inflate the fan-out tally and force a needless materialization (§7.4).
                out.add(`${path[1]}${SEP}${WHOLE}`);
              }
            }
          }
        } catch {
          // A malformed expression is the validator's report, not this pass's concern.
        }
      }
      return;
    }
    // scope/artifact/conversation resolvers name no child; an embedded op might carry nested edges.
    for (const p of Object.values(producer.input)) if (p.binding) collect(p.binding, out);
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
