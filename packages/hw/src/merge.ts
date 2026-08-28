/**
 * Operation inheritance (§5): how an `environment` chain becomes one effective operation.
 *
 * A state's effective operation is
 *
 * ```text
 * merge(root.environment, …, parent.environment, own.environment, own.operation)
 * ```
 *
 * with the NEAREST layer winning. `environment` is the same shape as `operation` with every field
 * optional, so "the defaults for this subtree" and "the operation here" are written the same way and
 * there is one vocabulary to learn instead of two.
 *
 * The merge is field-aware rather than a generic deep walk, because the right answer differs per
 * field and a generic walker gets two of them badly wrong:
 *
 *  - **`prompt` is one opaque leaf.** Deep-merging `{ template }` under `{ skill }` would produce a
 *    block with both, which is exactly the "exactly one of" error the format forbids. Replacing
 *    wholesale means a layer that supplies a prompt supplies ALL of it.
 *  - **`schema` and `binding` are opaque leaves.** Merging `{ child: "a" }` with `{ input: "b" }`
 *    yields `{ child: "a", input: "b" }` — not a binding at all, and the loader would desugar the
 *    first key it happened to test. A JSON Schema deep-merged against another JSON Schema is
 *    likewise meaningless (`{ type: "string" }` under `{ type: "array", items }`).
 *
 * Everything map-shaped — `config`, `input`, `permissions.tools` — merges per key, so a root that
 * sets `config.model` and a state that sets `config.temperature` end up with both. Arrays (`tools`,
 * `conversation.artifacts`) REPLACE, which is what makes `"tools": []` the way to drop an inherited
 * tool; a unioning merge would leave no way to take one away.
 */
import type { JsonValue } from "@declarative-ai/exec";
import { inferredKind, OPERATION_OWN_FIELDS, type NamedParameterDecl, type OperationFields, type ParameterDecl } from "./format.js";

/** Fields with a merge rule of their own — everything else is nearest-wins, wholesale. */
const MERGED_FIELDS: ReadonlySet<string> = new Set(["args", "input", "output", "conversation", "permissions", "tools", "path"]);

/** The `path` entry that expands to whatever the environment chain supplied (EXPRESSIONS.md §4.2). */
export const INHERITED_PATH = "$INHERITED";

/** True for a plain `{}` object — the only thing worth recursing into. */
function isPlainObject(value: unknown): value is Record<string, JsonValue> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Keys an authored layer may never contribute.
 *
 * `JSON.parse` creates `__proto__` as a real OWN property rather than invoking the setter, so it
 * survives into `Object.entries` — but assigning it back out with `out[key] = value` DOES invoke the
 * setter, replacing the merged object's prototype with authored content. The result carries fields
 * that property access sees and `Object.keys`, spreads and `JSON.stringify` do not.
 *
 * Every consumer today happens to read the merge result through a later `{...base}` spread, which
 * drops the prototype again — so this is currently corruption without a victim. That stops being
 * true the moment anything reads a field off the merge result DIRECTLY, which is exactly what an
 * inherited resolution field (a search path) would do: `__proto__` would then be a way to inject a
 * root that decides where references resolve from, invisibly to anything that enumerates keys.
 */
const FORBIDDEN_KEYS: ReadonlySet<string> = new Set(["__proto__"]);

/** Per-key merge of two plain objects; arrays and scalars in `over` replace. Used for `config`. */
function mergeJson(base: Record<string, JsonValue>, over: Record<string, JsonValue>): Record<string, JsonValue> {
  const out: Record<string, JsonValue> = { ...base };
  for (const [key, value] of Object.entries(over)) {
    if (FORBIDDEN_KEYS.has(key)) continue;
    const prior = out[key];
    out[key] = isPlainObject(prior) && isPlainObject(value) ? (mergeJson(prior, value) as JsonValue) : value;
  }
  return out;
}

/** Merge one slot declaration. `schema` and `binding` are opaque — see the module header. */
function mergeParameter(base: ParameterDecl, over: ParameterDecl): ParameterDecl {
  const out: ParameterDecl = { ...base, ...over };
  // Spread already took `over`'s value for every key it declares; the only thing to restore is a
  // key `over` left out, which `...base` supplied. That is the intent for every field here —
  // including `schema`/`binding`, which is precisely why this is a spread and not a deep merge.
  return out;
}

function mergeSlotMap(
  base: Record<string, ParameterDecl> | undefined,
  over: Record<string, ParameterDecl> | undefined,
): Record<string, ParameterDecl> | undefined {
  if (!base) return over;
  if (!over) return base;
  const out: Record<string, ParameterDecl> = { ...base };
  for (const [name, decl] of Object.entries(over)) {
    if (FORBIDDEN_KEYS.has(name)) continue;
    const prior = out[name];
    out[name] = prior ? mergeParameter(prior, decl) : decl;
  }
  return out;
}

/**
 * The fields whose MEANING depends on `kind`, and which an inheriting layer therefore drops when it
 * changes the kind.
 *
 * `config` is the reason this exists. For a prompt op it is the model surface; for a function op it
 * is the authored argument bag the function receives. A root defaulting
 * `config: { model: "anthropic/claude-sonnet-5" }` for its prompt states would otherwise hand
 * `model` to every `choose_option` gate in the subtree as if the author had written it there — a
 * silent, plausible-looking argument no one asked for.
 */
const KIND_SPECIFIC = ["args", "prompt", "system", "function"] as const;

/**
 * Refuse `sessionId`, which used to be a synonym for `session`.
 *
 * A refusal rather than a silent drop, because `sessionId` is NOT in `OPERATION_OWN_FIELDS`: an
 * unrecognized field is passed through to the LLM call configuration, so ignoring it would ship the
 * author's session declaration to the model as a call parameter and start a fresh conversation
 * without saying so.
 */
export function refuseSynonyms<T extends OperationFields>(fields: T): T {
  if ((fields as { sessionId?: unknown }).sessionId === undefined) return fields;
  throw new Error("operation declares 'sessionId'; the field is called 'session'");
}

type SessionDeclValue = OperationFields["session"];

/**
 * Two session declarations are the same if they name the same stream.
 *
 * Refs compare by id and expressions by source text. Comparing expressions textually is exact in the
 * direction that matters: two DIFFERENT spellings of the same position compare unequal, which costs
 * a spurious variant id, while two identical spellings can only mean the same thing.
 */
function sameSession(a: SessionDeclValue, b: SessionDeclValue): boolean {
  if (a === b) return true;
  const keyOf = (v: SessionDeclValue): string | undefined =>
    v !== null && typeof v === "object" ? ("id" in v ? v.id : `expr:${v.expr}`) : undefined;
  const [left, right] = [keyOf(a), keyOf(b)];
  return left !== undefined && left === right;
}

function describeSession(value: SessionDeclValue): string {
  if (value === null) return "null";
  if (typeof value !== "object") return String(value);
  return "id" in value ? value.id : `{expr: ${value.expr}}`;
}

/** Merge `over` onto `base`, field by field. `over` is the NEARER layer and wins. */
export function mergeOperationFields(base: OperationFields, over: OperationFields): OperationFields {
  base = refuseSynonyms(base);
  over = refuseSynonyms(over);
  // Read off each layer's OWN fields rather than off `kind`, because `kind` is optional now and the
  // drop below is the reason that matters. `environment: {prompt, model}` over `operation: {function}`
  // never writes the word "kind" anywhere, and if the change went unnoticed the gate would inherit
  // the ancestor's prompt and its model as if the author had written them there — the exact silent
  // result this whole block exists to prevent, reintroduced by the field it used to key on becoming
  // optional.
  const [overKind, baseKind] = [inferredKind(over), inferredKind(base)];
  if (overKind !== undefined && baseKind !== undefined && overKind !== baseKind) {
    const narrowed: OperationFields = { ...base };
    for (const field of KIND_SPECIFIC) delete narrowed[field];
    // Call configuration belongs to a prompt operation; carrying it onto a function op would hand
    // `model` to a UI gate as if the author had written it there.
    for (const key of Object.keys(narrowed)) {
      if (!OPERATION_OWN_FIELDS.has(key)) delete (narrowed as Record<string, unknown>)[key];
    }
    base = narrowed;
  }
  const out: OperationFields = { ...base };

  // Scalars and opaque leaves: present in `over` ⇒ `over` wins, wholesale. That covers every
  // call-config field too, which is why they are not listed one by one.
  for (const [key, value] of Object.entries(over)) {
    if (FORBIDDEN_KEYS.has(key)) continue;
    if (value !== undefined && !MERGED_FIELDS.has(key)) (out as Record<string, unknown>)[key] = value;
  }
  // An array replaces rather than unions, so `"tools": []` drops what the chain inherited.
  if (over.tools !== undefined) out.tools = over.tools;

  // `path` obeys that same rule, and splices with a SENTINEL rather than being exempted from it: a
  // `"$INHERITED"` entry expands to whatever the chain supplied, so `["./ops", "$INHERITED"]`
  // prepends and `["./ops"]` deliberately shadows everything (EXPRESSIONS.md §4.2). Exempting the
  // field instead would have left no way to say "ignore what I inherited".
  if (over.path !== undefined) {
    out.path = over.path.flatMap((entry) => (entry === INHERITED_PATH ? (base.path ?? []) : [entry]));
  }

  if (over.args !== undefined) out.args = base.args ? mergeJson(base.args, over.args) : over.args;

  const input = mergeSlotMap(base.input, over.input);
  if (input !== undefined) out.input = input;
  // `output` is a slot map like `input` now, so it merges as one — per NAME, with an overriding
  // layer restating only the fields it changes, rather than replacing the whole declaration.
  const output = mergeSlotMap(base.output, over.output);
  if (output !== undefined) out.output = output;

  if (over.conversation !== undefined) {
    out.conversation = base.conversation ? { ...base.conversation, ...over.conversation } : over.conversation;
  }
  if (over.permissions !== undefined) {
    const priorPerms = base.permissions;
    out.permissions = priorPerms
      ? {
          ...priorPerms,
          ...over.permissions,
          ...(priorPerms.tools || over.permissions.tools
            ? { tools: { ...priorPerms.tools, ...over.permissions.tools } }
            : {}),
        }
      : over.permissions;
  }

  return out;
}

/**
 * The environment a state RESOLVES IN: the chain that reached it, plus its own `environment` layer.
 *
 * One definition, because there are two consumers that must not drift — a state's effective
 * operation is this merged onto its `operation`, and its children inherit this unchanged (never its
 * `operation`, which describes what the state does rather than what its subtree defaults to). They
 * were two separate expressions of the same rule, and they had already drifted: one read the
 * EXPANDED document and the other the raw one, so a transcluded `environment` reached the state that
 * declared it and never reached its children.
 *
 * It is also the value a state's REFERENCES resolve against, which is why the loader computes it
 * before it desugars anything rather than at the point the operation is assembled. A field that
 * steers resolution is inherited by exactly this rule, so resolution has to run with it in hand.
 */
export function resolutionEnvironment(inherited: OperationFields, own: OperationFields | undefined): OperationFields {
  return own !== undefined ? mergeOperationFields(inherited, own) : inherited;
}

/** Fold a chain of layers, outermost first. */
export function mergeOperationChain(layers: ReadonlyArray<OperationFields | undefined>): OperationFields {
  let out: OperationFields = {};
  for (const layer of layers) {
    if (layer !== undefined) out = mergeOperationFields(out, layer);
  }
  return out;
}

/** The fields `LoadedState.environment` keeps — the execution environment, split off after merging. */
export const EXEC_ENVIRONMENT_FIELDS = ["session", "fork", "tools", "conversation", "permissions"] as const;

/**
 * A stable identity for one merged environment, so the loader can tell whether a state mounted under
 * two parents inherits the SAME defaults from both. Key order is normalized because two chains that
 * assign the same values in a different order are the same environment.
 */
export function environmentIdentity(fields: OperationFields): string {
  const stable = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(stable);
    if (isPlainObject(value)) {
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(value).sort()) out[key] = stable(value[key]);
      return out;
    }
    return value;
  };
  return JSON.stringify(stable(fields) ?? null);
}
