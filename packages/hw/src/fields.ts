/**
 * COMPUTED FIELDS (SPEC §5.3) — every value in a state file may be a binding.
 *
 * Three things live here, and they are the three halves of one contract:
 *
 *  - **Extraction**, at load. The loader hands in the authored top-level fields and the MERGED
 *    operation block; this takes every binding out of a value position, leaves the position empty,
 *    and returns the bindings as {@link LoadedField}s with the path each value goes back to.
 *  - **Materialization**, at instance entry. The engine evaluates the fields and hands the values
 *    back; this writes each at its path and returns the definition the instance actually runs — a
 *    fresh object per instance, since two instances of one state may resolve differently.
 *  - **The view**, for expressions. A field is read back under its authored name (§6.1) —
 *    `.title`, `.operation.config.model`, `.environment.tools` — and the view is what puts those
 *    names in the expression context, PENDING where a field has not settled.
 *
 * What stays authored is STRUCTURE: `id`, `children`, `sequence`, `transitions`, a slot's `schema`.
 * Those are what the values are checked against, and computing them would defer every check that
 * makes the rest of this safe to run time.
 *
 * The extraction knows two spellings (§5.3). Where the format knows a field's type — `label`, a
 * limit, `prompt`, `model` — the binding is written BARE in the value's place. Inside `config` and
 * `args`, whose values are opaque JSON, it is WRAPPED as `{ "binding": … }`, because a bare binding
 * there could not be told from a literal object with an `expr` key.
 */
import type { InlineFamily, JsonSchema, JsonValue, Operation, Parameter, Ref, RefKind, ResolvedValue } from "@declarative-ai/exec";
import { PENDING } from "./expr.js";
import {
  BOUND_CALLEE,
  CONFIG_FIELD_SCHEMAS,
  FIELD_NAMESPACES,
  isBindingDecl,
  isWrappedBinding,
  literalPermissions,
  OPERATION_OWN_FIELDS,
  type BindingDecl,
  type EnvironmentDecl,
  type ExecEnvironmentDecl,
  type LoadedField,
  type LoadedState,
  type OperationFields,
  type PermissionsDecl,
  type StateDef,
} from "./format.js";
import { referencePathsOf } from "./lowerExpr.js";
import { OPERATION_METADATA_FIELDS } from "./operationNode.js";
import { isOperationValue } from "./resolve.js";

/** How the loader lowers one binding — `desugarBinding`, with the site already bound in. */
export type LowerBinding = (binding: BindingDecl, where: string) => Ref<InlineFamily>;

/** A binding taken out of a value position, before its `environment` is merged with the state's. */
interface Extracted {
  path: string;
  binding: BindingDecl;
  schema?: JsonSchema;
  kind?: RefKind;
}

const STRING: JsonSchema = { type: "string" };
const STRINGS: JsonSchema = { type: "array", items: { type: "string" } };
const INTEGER: JsonSchema = { type: "integer" };
const NUMBER: JsonSchema = { type: "number" };

/** The known TOP-LEVEL value positions and their types. */
const TOP_LEVEL: ReadonlyArray<{ path: string; schema: JsonSchema; kind: RefKind }> = [
  { path: "label", schema: STRING, kind: "text" },
  { path: "description", schema: STRING, kind: "text" },
  { path: "limits.max_iterations", schema: INTEGER, kind: "json" },
  { path: "limits.timeout", schema: NUMBER, kind: "json" },
];

function readPath(root: unknown, path: readonly string[]): unknown {
  let at: unknown = root;
  for (const segment of path) {
    if (at === null || typeof at !== "object") return undefined;
    at = (at as Record<string, unknown>)[segment];
  }
  return at;
}

/** Write at a nested path, COPYING every object on the way — the source is shared across instances. */
function writePath<T extends object>(root: T, path: readonly string[], value: unknown): T {
  if (path.length === 0) return value as T;
  const [head, ...rest] = path as [string, ...string[]];
  const copy = (Array.isArray(root) ? [...root] : { ...root }) as Record<string, unknown>;
  const inner = copy[head];
  copy[head] = rest.length === 0 ? value : writePath(inner !== null && typeof inner === "object" ? (inner as object) : {}, rest, value);
  return copy as T;
}

/** Delete at a nested path, copying on the way; an empty container left behind is kept. */
function deletePath<T extends object>(root: T, path: readonly string[]): T {
  const [head, ...rest] = path as [string, ...string[]];
  const copy = { ...root } as Record<string, unknown>;
  if (rest.length === 0) delete copy[head];
  else {
    const inner = copy[head];
    if (inner !== null && typeof inner === "object") copy[head] = deletePath(inner as object, rest);
  }
  return copy as T;
}

/**
 * Take the bindings out of a state's TOP-LEVEL value positions: `label`, `title`, `description`,
 * `limits.*`. Returns the document with those positions cleared, and what was found.
 */
export function extractTopLevelFields(def: StateDef): { def: StateDef; found: Extracted[] } {
  const found: Extracted[] = [];
  let out: StateDef = def;
  // `title` is slot-shaped (§5.2): always a binding, under `binding`, typed as a string.
  if (def.title !== undefined) {
    found.push({ path: "title", binding: def.title.binding, schema: STRING, kind: "text" });
    out = deletePath(out, ["title"]);
  }
  for (const { path, schema, kind } of TOP_LEVEL) {
    const segments = path.split(".");
    const value = readPath(def, segments);
    if (value === undefined || !isBindingDecl(value)) continue;
    // A bare string in one of these positions is the value, not a reference: `"label": "Planning"`
    // has always meant the words. Only the object forms are bindings here.
    if (typeof value === "string") continue;
    found.push({ path, binding: value as BindingDecl, schema, kind });
    out = deletePath(out, segments);
  }
  return { def: out, found };
}

/**
 * Take the bindings out of a MERGED operation block — `prompt`, `system`, `function`, `tools`, the
 * known call-configuration fields, and every wrapped value under an unknown knob. Returns the block
 * with those positions cleared (the kind pinned, since the field that said it is gone), and what was
 * found under the authored path each value goes back to.
 *
 * `args` is handled by the caller: a wrapped value there is an INPUT slot's binding, not a field.
 */
export function extractOperationFields(merged: OperationFields): { op: OperationFields; found: Extracted[] } {
  const found: Extracted[] = [];
  let op: OperationFields = { ...merged };

  if (merged.prompt !== undefined && typeof merged.prompt !== "string") {
    // A prompt-kind value, or a template string computed at entry (SPEC §7.1). Typed by kind alone:
    // the validator accepts either a string or a prompt callable here.
    found.push({ path: "operation.prompt", binding: merged.prompt, kind: "prompt" });
    delete op.prompt;
    op.kind = "prompt";
  }
  if (merged.system !== undefined && typeof merged.system !== "string") {
    found.push({ path: "operation.system", binding: merged.system, schema: STRING, kind: "text" });
    delete op.system;
  }
  if (merged.function !== undefined && typeof merged.function !== "string") {
    // Bound callee (SPEC §7.1): the loaded op carries a placeholder name the engine replaces with the
    // callable the field resolves to. Typed as "some function"; the field's inferred signature is what
    // the validator checks the state's arguments against.
    found.push({ path: "operation.function", binding: merged.function, kind: "function" });
    op.function = BOUND_CALLEE;
    op.kind = "function";
  }
  if (merged.tools !== undefined && !Array.isArray(merged.tools)) {
    found.push({ path: "environment.tools", binding: merged.tools, schema: STRINGS, kind: "json" });
    delete op.tools;
  }
  // The permission baseline: the whole block, or the three scalar fields the format knows the type of.
  const permissions = merged.permissions;
  if (permissions !== undefined) {
    if (typeof permissions !== "string" && isBindingDecl(permissions)) {
      found.push({ path: "environment.permissions", binding: permissions as BindingDecl, schema: { type: "object" }, kind: "json" });
      delete op.permissions;
    } else if (typeof permissions === "object") {
      const block = permissions as PermissionsDecl;
      for (const key of ["profile", "default", "other"] as const) {
        const value = block[key];
        if (value === undefined || typeof value === "string" || !isBindingDecl(value)) continue;
        found.push({ path: `environment.permissions.${key}`, binding: value as BindingDecl, schema: STRING, kind: "text" });
        op = deletePath(op, ["permissions", key]);
      }
    }
  }
  // Call configuration: a KNOWN field written bare, or any value written wrapped — nested to any
  // depth, since `providerOptions` is a bag of bags.
  for (const [key, value] of Object.entries(merged)) {
    if (OPERATION_OWN_FIELDS.has(key) || value === undefined) continue;
    const known = CONFIG_FIELD_SCHEMAS[key];
    if (known !== undefined && isBindingDecl(value) && typeof value !== "string") {
      found.push({ path: `operation.config.${key}`, binding: value as BindingDecl, schema: known, kind: "json" });
      op = deletePath(op, [key]);
      continue;
    }
    const walk = (at: unknown, path: string[]): void => {
      if (isWrappedBinding(at)) {
        found.push({ path: `operation.config.${path.join(".")}`, binding: at.binding, kind: "json" });
        op = deletePath(op, path);
        return;
      }
      if (at === null || typeof at !== "object" || Array.isArray(at)) return;
      for (const [k, v] of Object.entries(at)) walk(v, [...path, k]);
    };
    walk(value, [key]);
  }
  return { op, found };
}

/**
 * The `args` of an operation with its wrapped values moved into INPUT slots (SPEC §5.3).
 *
 * `{"args": {"mode": {"binding": {"expr": …}}}}` is `{"input": {"mode": {"binding": {"expr": …}}}}`
 * — the same thing `args` has always been shorthand for, with a computed value where a constant
 * was. Moving it here rather than recording a field keeps one rule for how a value reaches a slot.
 */
export function liftWrappedArgs(decl: OperationFields): OperationFields {
  if (decl.args === undefined) return decl;
  const args: Record<string, JsonValue> = {};
  let input = decl.input;
  let moved = false;
  for (const [name, value] of Object.entries(decl.args)) {
    if (!isWrappedBinding(value)) {
      args[name] = value;
      continue;
    }
    moved = true;
    // An authored `input` slot of the same name wins, exactly as it does over a literal arg.
    if (input?.[name] !== undefined) continue;
    input = { ...input, [name]: { binding: value.binding } };
  }
  if (!moved) return decl;
  const out: OperationFields = { ...decl, args };
  if (input !== undefined) out.input = input;
  return out;
}

/**
 * Lower what extraction found into {@link LoadedField}s, in declaration order — their priority.
 *
 * `environment` on a binding merges ABOVE the state's own execution environment: the layer's `tools`
 * and `permissions` replace and merge as they do down the chain, so a field's calls run under what
 * the author wrote for them and inherit the rest.
 */
export function lowerFields(found: readonly Extracted[], lower: LowerBinding, environment: ExecEnvironmentDecl | undefined): LoadedField[] {
  return found.map((f) => {
    const field: LoadedField = { path: f.path, ref: lower(f.binding, f.path) };
    if (f.schema !== undefined) field.schema = f.schema;
    if (f.kind !== undefined) field.kind = f.kind;
    if (typeof f.binding === "object" && "expr" in f.binding) {
      if (f.binding.failureValue !== undefined) field.failureValue = f.binding.failureValue;
      const own = f.binding.environment;
      if (own !== undefined) field.environment = mergeExecEnvironment(environment, own);
    }
    return field;
  });
}

/** The state's execution environment with a binding's own layer merged over it — nearest wins. */
function mergeExecEnvironment(base: ExecEnvironmentDecl | undefined, over: EnvironmentDecl): ExecEnvironmentDecl {
  const out: ExecEnvironmentDecl = { ...base };
  if ("session" in over) out.session = over.session;
  if (over.tools !== undefined && Array.isArray(over.tools)) out.tools = over.tools;
  const overPermissions = literalPermissions({ permissions: over.permissions });
  if (overPermissions !== undefined) {
    const basePermissions = literalPermissions(base);
    out.permissions = {
      ...basePermissions,
      ...overPermissions,
      ...(basePermissions?.tools || overPermissions.tools ? { tools: { ...basePermissions?.tools, ...overPermissions.tools } } : {}),
    };
  }
  return out;
}

/**
 * Which OTHER fields a field reads — the edges of the dependency graph (SPEC §5.3).
 *
 * A reference reaches a field when one is a prefix of the other: reading `.operation.config` reads
 * every `operation.config.*` field, and reading `.operation.config.model` reads exactly that one.
 * References into anything that is not a field — an input, an artifact — are not dependencies.
 */
export function fieldDependencies(field: LoadedField, all: readonly LoadedField[]): string[] {
  const deps = new Set<string>();
  for (const reference of referencePathsOf(field.ref)) {
    const read = reference.join(".");
    for (const other of all) {
      if (other.path === field.path) continue;
      if (read === other.path || read.startsWith(`${other.path}.`) || other.path.startsWith(`${read}.`)) deps.add(other.path);
    }
  }
  return [...deps];
}

/** The first cycle among a state's fields, as the paths on it — `undefined` when there is none. */
export function fieldCycle(fields: readonly LoadedField[]): string[] | undefined {
  const deps = new Map(fields.map((f) => [f.path, fieldDependencies(f, fields)] as const));
  const state = new Map<string, "visiting" | "done">();
  const stack: string[] = [];
  const visit = (path: string): string[] | undefined => {
    const seen = state.get(path);
    if (seen === "done") return undefined;
    if (seen === "visiting") return [...stack.slice(stack.indexOf(path)), path];
    state.set(path, "visiting");
    stack.push(path);
    for (const dep of deps.get(path) ?? []) {
      const cycle = visit(dep);
      if (cycle !== undefined) return cycle;
    }
    stack.pop();
    state.set(path, "done");
    return undefined;
  };
  for (const f of fields) {
    const cycle = visit(f.path);
    if (cycle !== undefined) return cycle;
  }
  return undefined;
}

/**
 * The reference roots a field may NOT read (SPEC §5.3): a field is evaluated at instance entry, when
 * no child has run, no output has been produced, and no operation has been made. Inputs and the
 * other fields are the whole of its world.
 */
export function forbiddenFieldRead(reference: readonly string[]): string | undefined {
  const [root, second] = reference;
  if (root === "children" || root === "outputs" || root === "run" || root === "each") {
    return `a field is evaluated at entry, before any child runs or any output exists, so it cannot read '.${reference.join(".")}'`;
  }
  if (root === "operation" && second !== undefined && (OPERATION_METADATA_FIELDS.has(second) || second === "output")) {
    return `a field is evaluated before the operation runs, so it cannot read its result '.${reference.join(".")}'`;
  }
  return undefined;
}

/** True for a reference root that names a field namespace — read back from the materialized definition. */
export function isFieldRoot(root: string): boolean {
  return (FIELD_NAMESPACES as readonly string[]).includes(root) || root === "operation" || root === "limits";
}

// --- Materialization -----------------------------------------------------------

/** What writing one field's value into the definition can report. */
export type Materialized = { def: LoadedState } | { error: string };

/**
 * The definition an INSTANCE runs: the loaded state with these field values written at their paths.
 *
 * A fresh object each time, sharing everything structural with the loaded state by reference — the
 * children, the transitions, the slots — and copying only the containers a value is written into.
 */
export function materializeFields(def: LoadedState, values: ReadonlyMap<string, ResolvedValue>, bindCallee: BindCallee): Materialized {
  let out: LoadedState = def;
  for (const field of def.fields ?? []) {
    if (!values.has(field.path)) continue;
    const value = values.get(field.path);
    const written = writeField(out, field.path, value, bindCallee);
    if ("error" in written) return written;
    out = written.def;
  }
  return { def: out };
}

/**
 * How a bound `function` field's callable becomes the state's operation — the loader's own
 * `bindIntoSlots`, handed in because it lives there and throws its own error type.
 */
export type BindCallee = (callee: Operation<InlineFamily>, authored: Record<string, Parameter<InlineFamily>>) => Record<string, Parameter<InlineFamily>> | { error: string };

function writeField(def: LoadedState, path: string, value: ResolvedValue | undefined, bindCallee: BindCallee): Materialized {
  const segments = path.split(".");
  const [root, ...rest] = segments as [string, ...string[]];
  if (root === "operation") {
    const op = def.operation;
    if (op === undefined) return { error: `field '${path}' has no operation to write into` };
    const [field, ...more] = rest as [string, ...string[]];
    switch (field) {
      case "prompt": {
        if (op.kind !== "prompt") return { error: `field '${path}' on a function operation` };
        if (typeof value === "string") return { def: { ...def, operation: { ...op, user: value } } };
        if (!isOperationValue(value) || value.kind !== "prompt") {
          return { error: `field '${path}' resolved to ${describe(value)}, not a template or a prompt` };
        }
        // A prompt passed BY VALUE (SPEC §7.1): its template, and what the state did not itself say.
        const merged: Operation<InlineFamily> = {
          ...op,
          user: value.user,
          ...(op.system === undefined && value.system !== undefined ? { system: value.system } : {}),
          config: { ...(value.config as Record<string, JsonValue>), ...(op.config as Record<string, JsonValue>) } as JsonValue,
          input: { ...value.input, ...op.input },
          output: op.output.schema !== undefined ? op.output : value.output,
        };
        return { def: { ...def, operation: merged } };
      }
      case "system": {
        if (op.kind !== "prompt") return { error: `field '${path}' on a function operation` };
        if (typeof value !== "string") return { error: `field '${path}' resolved to ${describe(value)}, not a string` };
        return { def: { ...def, operation: { ...op, system: value } } };
      }
      case "function": {
        if (op.kind !== "function") return { error: `field '${path}' on a prompt operation` };
        // A NAME resolves at dispatch, as a literal `function` naming an unregistered entry does.
        if (typeof value === "string") return { def: { ...def, operation: { ...op, functionRef: value } } };
        if (!isOperationValue(value) || value.kind !== "function") {
          return { error: `field '${path}' resolved to ${describe(value)}, not a function` };
        }
        // The callee's slots carry the TYPES, the state's carry the VALUES — the same rule the loader
        // applies to a named callee, applied to one nobody could name until now.
        const callee: Operation<InlineFamily> = value;
        const bound = bindCallee(callee, op.input);
        if (isBindError(bound)) return bound;
        const merged: Operation<InlineFamily> = {
          ...callee,
          input: bound,
          ...(op.spread !== undefined ? { spread: op.spread } : {}),
          output: op.output.schema !== undefined ? op.output : callee.output,
        } as Operation<InlineFamily>;
        return { def: { ...def, operation: merged } };
      }
      case "config": {
        const config = (op as { config?: JsonValue }).config;
        const base = config !== null && typeof config === "object" && !Array.isArray(config) ? (config as Record<string, unknown>) : {};
        return { def: { ...def, operation: { ...op, config: writePath(base, more, value) as JsonValue } as Operation<InlineFamily> } };
      }
      default:
        return { error: `field '${path}' names no operation field` };
    }
  }
  if (root === "environment") {
    return { def: { ...def, environment: writePath(def.environment ?? {}, rest, value) } };
  }
  if (root === "limits") {
    return { def: { ...def, limits: writePath(def.limits ?? {}, rest, value) } };
  }
  if (root === "title" || root === "label" || root === "description") {
    if (typeof value !== "string") return { error: `field '${path}' resolved to ${describe(value)}, not a string` };
    return { def: { ...def, [root]: value } };
  }
  return { error: `field '${path}' names no value position` };
}

/** A slot map has parameters for values; a bind failure has one string. */
function isBindError(v: Record<string, Parameter<InlineFamily>> | { error: string }): v is { error: string } {
  return typeof (v as { error?: unknown }).error === "string";
}

function describe(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "nothing";
  if (Array.isArray(value)) return "an array";
  return `a ${typeof value}`;
}

// --- The view ----------------------------------------------------------------

/**
 * The state's own fields as an expression reads them (SPEC §6.1) — from the definition the instance
 * runs, with PENDING at every path that has not settled yet.
 *
 * `operation` here is the AUTHORED half of that root: `prompt`, `system`, `function`, `config`. The
 * engine lays the call's RESULT node over it; the key sets do not overlap.
 */
export function fieldsView(def: LoadedState, unsettled: ReadonlySet<string>): Record<string, unknown> {
  const op = def.operation;
  const operation: Record<string, unknown> = {};
  if (op !== undefined) {
    if (op.kind === "prompt") {
      operation.prompt = op.user;
      if (op.system !== undefined) operation.system = op.system;
    } else {
      operation.function = op.functionRef === BOUND_CALLEE ? undefined : op.functionRef;
    }
    const config = (op as { config?: JsonValue }).config;
    if (config !== undefined) operation.config = config;
  }
  const env = def.environment ?? {};
  let view: Record<string, unknown> = {
    ...(def.title !== undefined ? { title: def.title } : {}),
    ...(def.label !== undefined ? { label: def.label } : {}),
    ...(def.description !== undefined ? { description: def.description } : {}),
    limits: { ...(def.limits ?? {}) },
    operation,
    environment: { ...env },
  };
  for (const path of unsettled) view = writePath(view, path.split("."), PENDING);
  return view;
}
