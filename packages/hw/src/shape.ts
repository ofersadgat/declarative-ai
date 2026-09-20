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
  /**
   * A record with known fields, and optionally an open tail (`rest`) for map-shaped blocks.
   *
   * `value` marks a position whose whole content is a VALUE the engine can compute at entry (SPEC
   * §5.3) — `tools`, `permissions` — as against STRUCTURE the loader reads (`inputs`, `children`,
   * `operation`). Only a value position can hold a scoped name (NAMES.md §5): a name reads as a value
   * per instance, and structure has to be there before there is an instance to read it for.
   */
  | { t: "object"; fields?: Record<string, Shape>; rest?: Shape; value?: true }
  | { t: "array"; of: Shape; value?: true }
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
  | { t: "any" }
  /**
   * A position that PROVIDES what a name is bound to (NAMES.md §4) — `session` creates the
   * conversation the first time a name is used, so the name needs no entry and no file. It expects
   * an object (the thing provided), which is what makes a string here a variable string: `$…` is a
   * reference as everywhere, and anything else is a scoped name that expansion leaves for the
   * loader to anchor. It never falls back to a file, because the position itself is the fallback.
   */
  | { t: "name"; position: string }
  /**
   * `environment.names` — a map of name → configuration block (NAMES.md §4). Untyped inside, like
   * `args`, with one reading of its own: inside an entry, the entry's own name means the ENCLOSING
   * entry, so `{ "$ref": "impl", … }` pastes rather than cycles.
   */
  | { t: "names" };

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
  session: { t: "name", position: "session" },
  workspace: { t: "name", position: "workspace" },
  sessionId: scalar,
  tools: { t: "array", of: scalar, value: true },
  conversation: { t: "object", fields: { mode: scalar, artifacts: { t: "array", of: scalar } } },
  permissions: {
    t: "object",
    fields: { profile: scalar, default: scalar, tools: { t: "object", rest: scalar } },
    value: true,
  },
  // The LlmConfiguration surface, inline on a prompt operation (§7.2).
  model: scalar,
  maxOutputTokens: scalar,
  stopSequences: { t: "array", of: scalar, value: true },
  seed: scalar,
  maxSteps: scalar,
  toolChoice: anyJson,
  providerOptions: { t: "object", rest: anyJson, value: true },
};

const operation: Shape = { t: "object", fields: operationFields };

/**
 * An `environment` block: an operation's fields, all optional, plus what only a DEFAULTS layer can
 * say. `names` configures scoped names for the subtree (NAMES.md §4) and `functions` gives a
 * function its default arguments (§7); neither is on `operation`, because an operation is one call
 * and both of these are statements about a subtree.
 */
const environment: Shape = {
  t: "object",
  fields: {
    ...operationFields,
    names: { t: "names" },
    // Default arguments, under the FUNCTION's name (NAMES.md §7). `args` is the same untyped bag it
    // is on an operation, so only the explicit `{"$ref"}` form counts inside it (§3.1).
    functions: { t: "object", rest: { t: "object", fields: { args: { t: "object", rest: anyJson } } } },
  },
};

/** A transition list, in the one shape both the state level and a child mount write it. */
const transitions: Shape = { t: "array", of: { t: "object", fields: { to: scalar, when: scalar } } };

const child: Shape = {
  t: "object",
  fields: {
    state: { t: "ref" },
    inputs: { t: "object", rest: { t: "binding" } },
    async: scalar,
    // Per-mount defaults, written in the same shape as any other environment — so a `$ref` inside one
    // transcludes exactly as it does at state level.
    environment,
    // Considered when this child finishes, before the state's own. `to` is a plain scalar here for
    // the same reason it is there: a transition target is a CHILD KEY, not a state path, so it is
    // never resolved as a reference.
    transitions,
  },
};

/** One state file. */
export const STATE_SHAPE: Shape = {
  t: "object",
  fields: {
    id: { t: "ref" },
    label: scalar,
    // Slot-shaped (SPEC §5.2): a binding under `binding`, so a path there transcludes as one.
    title: { t: "object", fields: { binding: { t: "binding" }, description: scalar } },
    description: scalar,
    inputs: slotMap,
    outputs: slotMap,
    operation,
    environment,
    children: { t: "object", rest: child },
    sequence: { t: "array", of: scalar },
    transitions,
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
