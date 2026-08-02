/**
 * The workflow format as DATA (REFERENCES.md §3).
 *
 * Reference detection is type-directed: a string where an object is expected is a reference, and
 * `{"$ref": …}` where a string is expected is a reference. That rule needs to know what each
 * position expects, and TypeScript interfaces are gone by run time — so the format's shape is
 * written out once, here, and the expander walks a document guided by it.
 *
 * This is deliberately NOT a JSON Schema. It answers exactly one question — "what kind of thing
 * goes here?" — and adding validation vocabulary to it would create a second, half-complete
 * description of the format competing with `validate.ts`.
 */

/** What a position expects. */
export type Shape =
  /** A record with known fields, and optionally an open tail (`rest`) for map-shaped blocks. */
  | { t: "object"; fields?: Record<string, Shape>; rest?: Shape }
  | { t: "array"; of: Shape }
  /** A plain string or number: a reference here must be written `{"$ref": …}`. */
  | { t: "scalar" }
  /** A string that IS a path — `id`, `children[].state`. Resolved, never transcluded. */
  | { t: "ref" }
  /**
   * A JSON Schema document. Transparent to expansion (a shared type library is the point) but
   * `$ref` inside one is always JSON Schema's own, never ours — see §6.
   */
  | { t: "schema" }
  /**
   * A binding. A leading-dot string here is a RUNTIME reference and passes through to the
   * desugarer; any other string is a document reference and is transcluded (§5).
   */
  | { t: "binding" }
  /** Arbitrary JSON whose type nothing knows: only an explicit `{"$ref"}` counts (§3.1). */
  | { t: "any" };

const scalar: Shape = { t: "scalar" };
const anyJson: Shape = { t: "any" };
const schema: Shape = { t: "schema" };

/** A declared slot — `inputs.<name>`, `outputs.<name>`, `operation.input.<name>`. */
const parameter: Shape = {
  t: "object",
  fields: {
    kind: scalar,
    schema,
    binding: { t: "binding" },
    index: scalar,
    default: anyJson,
    optional: scalar,
    description: scalar,
    name: scalar,
  },
};

const slotMap: Shape = { t: "object", rest: parameter };

/**
 * Everything an operation may carry — the union flattened, since a document is walked before its
 * `kind` is necessarily known (an ancestor's `environment` may supply it).
 *
 * `args` is the only `any` position left: a function's arguments are known to the function alone.
 * Every field of a prompt operation is typed, which is what §7.2's discriminated union buys.
 */
const operationFields: Record<string, Shape> = {
  kind: scalar,
  prompt: scalar,
  system: scalar,
  function: scalar,
  args: { t: "object", rest: anyJson },
  input: slotMap,
  output: parameter,
  // Execution environment, inherited by the same rule as everything else.
  session: scalar,
  sessionId: scalar,
  tools: { t: "array", of: scalar },
  conversation: { t: "object", fields: { mode: scalar, artifacts: { t: "array", of: scalar } } },
  permissions: {
    t: "object",
    fields: { profile: scalar, default: scalar, tools: { t: "object", rest: scalar } },
  },
  // The LlmConfiguration surface, inline on a prompt operation (§7.2).
  model: scalar,
  maxOutputTokens: scalar,
  stopSequences: { t: "array", of: scalar },
  seed: scalar,
  maxSteps: scalar,
  toolChoice: anyJson,
  providerOptions: { t: "object", rest: anyJson },
};

const operation: Shape = { t: "object", fields: operationFields };

const child: Shape = {
  t: "object",
  fields: {
    state: { t: "ref" },
    inputs: { t: "object", rest: { t: "binding" } },
    async: scalar,
    // Per-mount defaults, written in the same shape as any other environment — so a `$ref` inside one
    // transcludes exactly as it does at state level.
    environment: operation,
  },
};

/** One state file. */
export const STATE_SHAPE: Shape = {
  t: "object",
  fields: {
    id: { t: "ref" },
    label: scalar,
    description: scalar,
    inputs: slotMap,
    outputs: slotMap,
    operation,
    environment: operation,
    children: { t: "object", rest: child },
    sequence: { t: "array", of: scalar },
    transitions: { t: "array", of: { t: "object", fields: { to: scalar, when: scalar } } },
    limits: { t: "object", fields: { max_iterations: scalar, timeout: scalar } },
  },
};

/** The shape of a named field, or the container's open tail. */
export function fieldShape(container: Shape, key: string): Shape | undefined {
  if (container.t !== "object") return undefined;
  // OWN fields only: a document key named `constructor` or `toString` otherwise picked a prototype
  // member up off the `fields` literal and handed expansion a FUNCTION as the shape to walk, instead
  // of falling through to the container's open tail (or to "unknown field").
  const fields = container.fields;
  const named = fields !== undefined && Object.hasOwn(fields, key) ? fields[key] : undefined;
  return named ?? container.rest;
}
