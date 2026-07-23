/**
 * Bundle loading, DESUGARING, and snapshot hashing (SPEC §2.4, §12; API.md, "Binding desugaring").
 *
 * A bundle is a map of state-ID → state-file JSON. State IDs are file paths relative to the
 * workflow root without suffix; `id` inside a file may be omitted and is derived from the path
 * (a present-but-mismatched `id` is a load error).
 *
 * Loading also DESUGARS: every authored binding form (`{ child, output }`, `{ input }`,
 * `{ expr }`, `{ artifact }`, `{ conversation }`) is lowered to a base `Ref<InlineFamily>` case
 * — a literal or a producer edge over a well-known resolver function (`RESOLVER_REFS`). After
 * this pass nothing downstream — checker, hasher, engine — knows the sugar exists.
 *
 * The snapshot hash is the workflow-version identity (SPEC §12, JaiRA DESIGN §5.3):
 * `sha256(canonicalize(sorted [(stateId, contentHash(stateFile))]))` over the transitive closure
 * of states reachable from the root. It hashes the AUTHORED file, so two spellings of the same
 * sugar hash differently but a desugaring change never invalidates a stored snapshot.
 */
import { canonicalize, hashCanonical, kindFor, sha256Hex, type InlineFamily, type JsonSchema, type JsonValue, type NamedParameter, type Operation, type Parameter, type Ref, type RefKind } from "@declarative-ai/exec";
import { computeFanOut } from "./fanout";
import {
  RESOLVER_REFS,
  type BindingDecl,
  type ChildDecl,
  type LoadedChild,
  type LoadedState,
  type NamedParameterDecl,
  type OperationDecl,
  type ParameterDecl,
  type SlotMeta,
  type StateDef,
  type WorkflowBundle,
} from "./format";

export class WorkflowLoadError extends Error {
  constructor(
    message: string,
    readonly stateId?: string,
  ) {
    super(stateId ? `${stateId}: ${message}` : message);
    this.name = "WorkflowLoadError";
  }
}

/** Strip a state-file name to its state ID: forward slashes, no extension. */
export function stateIdFromPath(relPath: string): string {
  return relPath.replace(/\\/g, "/").replace(/\.state\.json$|\.json$/i, "");
}

// --- Desugaring (§2.1) --------------------------------------------------------

/**
 * The kind a slot carries when the author didn't say — DELEGATED to `kindFor`, never re-derived here.
 *
 * This used to be a local copy of the rule that only knew `text` vs `json`, which is exactly the drift
 * `kindFor` exists to prevent: an artifact slot authored with JSON Schema's own content keywords (the
 * thing §7 replaced the bespoke `x-artifact` marker with) loaded as `text`, the engine's artifact
 * registration gates on `kind === "blob"`, and so the artifact silently never existed.
 */
function kindOf(decl: { kind?: RefKind; schema?: JsonSchema }): RefKind {
  return decl.kind ?? kindFor(decl.schema);
}

/** A producer edge on an embedded resolver `FunctionOp` — the shape EVERY sugar lowers to. */
function resolverEdge(functionRef: string, args: Record<string, Ref<InlineFamily>>, outKind: RefKind = "json"): Ref<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, binding] of Object.entries(args)) {
    input[name] = { kind: "text" in binding ? "text" : "json", binding };
  }
  return {
    op: { kind: "function", functionRef, input, output: { name: "value", kind: outKind } },
  };
}

/** True for the base `Ref<InlineFamily>` cases — everything else in `BindingDecl` is sugar. */
function isBaseRef(b: BindingDecl): b is Ref<InlineFamily> {
  return "text" in b || "json" in b || "result" in b || "refs" in b || "op" in b;
}

/**
 * Lower ONE authored binding to a base `Ref<InlineFamily>`. The mapping is §2.1's table:
 * every sugar becomes a producer edge (or a literal), so the base vocabulary stays closed.
 */
export function desugarBinding(binding: BindingDecl, where: string, stateId: string): Ref<InlineFamily> {
  if (isBaseRef(binding)) return binding;

  if ("child" in binding) {
    // A producer edge on the declared child, plus a `select` projection when a specific output is
    // named (hw states lower to single-object-output ops, so a named output IS a property select).
    const childEdge: Ref<InlineFamily> = { op: binding.child };
    if (binding.output === undefined) return childEdge;
    return resolverEdge(RESOLVER_REFS.select, { value: childEdge, key: { text: binding.output } });
  }
  if ("input" in binding) {
    return resolverEdge(RESOLVER_REFS.scope, { scope: { text: "inputs" }, name: { text: binding.input } });
  }
  if ("expr" in binding) {
    // An expression IS a pure FunctionOp producer whose output schema is the inferred type (§7.2),
    // so ordinary binding type-checking applies to it with no special case.
    return resolverEdge(RESOLVER_REFS.expr, { source: { text: binding.expr } });
  }
  if ("artifact" in binding) {
    return resolverEdge(RESOLVER_REFS.artifact, { name: { text: binding.artifact } });
  }
  if ("conversation" in binding) {
    const args: Record<string, Ref<InlineFamily>> = { session: { text: binding.conversation } };
    if (binding.message !== undefined) args.message = { json: binding.message };
    return resolverEdge(RESOLVER_REFS.conversation, args);
  }
  throw new WorkflowLoadError(`${where}: unrecognized binding form ${JSON.stringify(binding)}`, stateId);
}

/** Lower an authored slot to a `Parameter<InlineFamily>`, splitting off its authoring metadata. */
function desugarParameter(decl: ParameterDecl, where: string, stateId: string): { param: Parameter<InlineFamily>; meta?: SlotMeta } {
  const param: Parameter<InlineFamily> = { kind: kindOf(decl) };
  if (decl.schema !== undefined) param.schema = decl.schema;
  if (decl.binding !== undefined) param.binding = desugarBinding(decl.binding, where, stateId);
  if (decl.index !== undefined) param.index = decl.index;
  const meta: SlotMeta = {};
  if (decl.default !== undefined) meta.default = decl.default;
  if (decl.optional !== undefined) meta.optional = decl.optional;
  if (decl.description !== undefined) meta.description = decl.description;
  return { param, ...(Object.keys(meta).length > 0 ? { meta } : {}) };
}

function desugarNamedParameter(name: string, decl: NamedParameterDecl, where: string, stateId: string): { param: NamedParameter<InlineFamily>; meta?: SlotMeta } {
  const { param, meta } = desugarParameter(decl, where, stateId);
  return { param: { ...param, name: decl.name ?? name }, ...(meta ? { meta } : {}) };
}

function desugarSlotMap(
  section: string,
  fields: Record<string, ParameterDecl> | undefined,
  stateId: string,
  slotMeta: Record<string, SlotMeta>,
): Record<string, Parameter<InlineFamily>> | undefined {
  if (!fields) return undefined;
  const out: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, decl] of Object.entries(fields)) {
    const { param, meta } = desugarParameter(decl, `${section}.${name}`, stateId);
    out[name] = param;
    if (meta) slotMeta[`${section}.${name}`] = meta;
  }
  return out;
}

/** The default output slot of a state operation: one object carrying the state's declared outputs. */
function defaultOutput(): NamedParameter<InlineFamily> {
  return { name: "output", kind: "json" };
}

/** Lower an authored operation block to a real `Operation<InlineFamily>` (§7.1). */
export function desugarOperation(decl: OperationDecl, stateId: string, outputs?: Record<string, NamedParameterDecl>): Operation<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, p] of Object.entries(decl.input ?? {})) {
    input[name] = desugarParameter(p, `operation.input.${name}`, stateId).param;
  }

  // The op's output is the authored one, or a single object slot whose schema is built from the
  // state's declared outputs — the projection `{ child, output }` selects against.
  const output = decl.output
    ? desugarNamedParameter("output", decl.output, "operation.output", stateId).param
    : outputSlotFor(outputs);

  if (decl.kind === "prompt") {
    // The template's `{{inputs.*}}` scope IS the operation's resolved inputs (§3.1: authored render
    // variables ride bound input slots, never a field on the op shape), so there is nothing to merge
    // in here — every render variable is just one of `input`.
    const op: Operation<InlineFamily> = {
      kind: "prompt",
      // A `skill` prompt resolves through `registry.skills` at render time; the op carries the
      // reference in the same `user` slot, marked so the engine can tell the two apart.
      user: decl.prompt?.skill !== undefined ? skillRef(decl.prompt.skill) : (decl.prompt?.template ?? ""),
      config: (decl.config ?? {}) as JsonValue,
      input,
      output,
    };
    if (decl.system !== undefined) op.system = decl.system;
    return op;
  }
  // A FunctionOp — a host function, a sub-workflow, or a delegated runtime adapter alike (§3.1).
  // The authored surface rides a bound `config` input; the op shape gains nothing.
  if (decl.config !== undefined && input.config === undefined) {
    input.config = { kind: "json", binding: { json: decl.config as JsonValue } };
  }
  return { kind: "function", functionRef: decl.function, input, output };
}

/** The marker prefix distinguishing a SKILL reference from an inline template in a `user` slot. */
export const SKILL_PREFIX = "skill:";
export const skillRef = (name: string): string => `${SKILL_PREFIX}${name}`;
export const skillNameOf = (user: string): string | undefined => (user.startsWith(SKILL_PREFIX) ? user.slice(SKILL_PREFIX.length) : undefined);

/**
 * Build the operation's single object-output slot from the outputs the OPERATION produces — which is
 * the state's declared outputs MINUS the bound ones. An output with a binding is derived when the
 * state terminates (from a child, an expression, …), never returned by the operation, so requiring it
 * of the operation would be a contract the operation cannot meet. This is the same filter the engine
 * applies at run time (`producedOutputSlots`); keeping them in step means a consumer reading
 * `op.output.schema` statically sees the same contract the engine enforces.
 */
function outputSlotFor(outputs: Record<string, NamedParameterDecl> | undefined): NamedParameter<InlineFamily> {
  const produced = Object.entries(outputs ?? {}).filter(([, decl]) => decl.binding === undefined);
  if (produced.length === 0) return defaultOutput();
  const properties: Record<string, JsonValue> = {};
  const required: string[] = [];
  for (const [name, decl] of produced) {
    // An unconstrained slot (no schema) constrains nothing; the checker treats it as universal.
    properties[name] = (decl.schema ?? {}) as JsonValue;
    if (decl.optional !== true && decl.default === undefined) required.push(name);
  }
  const schema: JsonSchema = { type: "object", properties, ...(required.length > 0 ? { required } : {}) };
  return { name: "output", kind: "json", schema };
}

/** Desugar one authored state file into its loaded form. */
export function desugarState(id: string, def: StateDef): LoadedState {
  const slotMeta: Record<string, SlotMeta> = {};
  const inputs = desugarSlotMap("inputs", def.inputs, id, slotMeta);

  let outputs: Record<string, NamedParameter<InlineFamily>> | undefined;
  if (def.outputs) {
    outputs = {};
    for (const [name, decl] of Object.entries(def.outputs)) {
      const { param, meta } = desugarNamedParameter(name, decl, `outputs.${name}`, id);
      outputs[name] = param;
      if (meta) slotMeta[`outputs.${name}`] = meta;
    }
  }

  let children: Record<string, LoadedChild> | undefined;
  if (def.children) {
    children = {};
    for (const [key, child] of Object.entries(def.children)) {
      const wired: Record<string, Ref<InlineFamily>> = {};
      for (const [inputName, binding] of Object.entries(child.inputs ?? {})) {
        wired[inputName] = desugarBinding(binding, `children.${key}.inputs.${inputName}`, id);
      }
      children[key] = {
        state: child.state,
        ...(child.inputs ? { inputs: wired } : {}),
        ...(child.async !== undefined ? { async: child.async } : {}),
      };
    }
  }

  const { operation, inputs: _i, outputs: _o, children: _c, ...rest } = def;
  return {
    ...rest,
    id,
    ...(inputs ? { inputs } : {}),
    ...(outputs ? { outputs } : {}),
    ...(children ? { children } : {}),
    ...(operation ? { operation: desugarOperation(operation, id, def.outputs) } : {}),
    ...(Object.keys(slotMeta).length > 0 ? { slotMeta } : {}),
  };
}

// --- Bundle loading -----------------------------------------------------------

/**
 * Load a bundle from raw file contents (`stateId or relative path` → parsed JSON), desugaring
 * each state. Restricts the bundle to the transitive closure reachable from `rootId` so the
 * snapshot hash never varies with unrelated files lying around the workflow dir.
 */
export function loadBundle(files: Record<string, unknown>, rootId: string): WorkflowBundle {
  const authored = new Map<string, StateDef>();
  for (const [key, raw] of Object.entries(files)) {
    const id = stateIdFromPath(key);
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
      throw new WorkflowLoadError("state file is not a JSON object", id);
    }
    const def = raw as StateDef;
    if (def.id !== undefined && def.id !== id) {
      throw new WorkflowLoadError(`declared id '${def.id}' does not match path-derived id '${id}'`, id);
    }
    authored.set(id, def);
  }
  if (!authored.has(rootId)) {
    throw new WorkflowLoadError(`root state '${rootId}' not found in bundle`);
  }
  // Transitive closure from the root.
  const states: Record<string, LoadedState> = {};
  const rawById = new Map<string, StateDef>(authored);
  const queue = [rootId];
  while (queue.length > 0) {
    const id = queue.shift()!;
    if (states[id]) continue;
    const def = rawById.get(id);
    if (!def) {
      // Missing children are a VALIDATION error (with context), not a load error —
      // keep loading so the validator can report all of them at once.
      continue;
    }
    const loaded = desugarState(id, def);
    // Fan-out is a static property of the wiring (§7.3, rule 2): with every consumer of every producer
    // desugared to a base ref, the loader can tally them once here rather than the engine discovering a
    // second reader at run time.
    const fanOut = computeFanOut(loaded);
    if (fanOut !== undefined) loaded.fanOut = fanOut;
    states[id] = loaded;
    for (const child of Object.values(def.children ?? {})) {
      queue.push((child as ChildDecl).state);
    }
  }
  // Keep the authored files for hashing — the snapshot identity is what the AUTHOR wrote.
  const source: Record<string, StateDef> = {};
  for (const id of Object.keys(states)) {
    const def = rawById.get(id);
    if (def) source[id] = def;
  }
  return { rootId, states, source };
}

/**
 * The snapshot hash — the bundle's version identity (SPEC §12). This is the content-identity a
 * `hierarchical-workflow` execution memoizes under: `workflowDefinitionHash` returns it and it becomes
 * the memo-key's definition-hash component (DESIGN §3.4) via `withMemoize`'s `identify` seam.
 */
export function snapshotHash(bundle: WorkflowBundle): string {
  const entries = Object.keys(bundle.states)
    .map((id) => [id, hashCanonical(stripDerivedId(bundle.source?.[id] ?? bundle.states[id]!))] as const)
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return sha256Hex(canonicalize({ rootId: bundle.rootId, states: entries }));
}

/** Hash the file as authored: a derived (previously absent) `id` must not change the hash. */
function stripDerivedId(def: StateDef | LoadedState): JsonValue {
  const { id: _id, ...rest } = def;
  return rest as unknown as JsonValue;
}

/**
 * Node-only convenience: load every `*.json` under a directory as a bundle rooted at
 * `rootId`. Uses dynamic imports so the module stays edge-safe when unused.
 */
export async function loadBundleFromDir(dir: string, rootId: string): Promise<WorkflowBundle> {
  const { readdir, readFile } = await import("node:fs/promises");
  const { join, relative } = await import("node:path");
  const files: Record<string, unknown> = {};
  const walk = async (d: string): Promise<void> => {
    for (const entry of await readdir(d, { withFileTypes: true })) {
      const full = join(d, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) {
        const rel = relative(dir, full);
        files[rel] = JSON.parse(await readFile(full, "utf8")) as unknown;
      }
    }
  };
  await walk(dir);
  return loadBundle(files, rootId);
}
