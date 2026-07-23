/**
 * Wiring RESOLUTION (SPEC §4.2) — turning an operation's `Parameter` bindings into values
 * against a run.
 *
 * The engine no longer renders a bespoke operation payload: it resolves a state's operation's
 * bindings (literals, ref trees, and producer edges) and dispatches the resolved op by kind. The
 * producer semantics are findmyprompt's, in memory: `kind ∈ {text, json}` RUNS the producer and its
 * output fills the slot; a producer already run in this scope is REUSED (the memo, run-scoped);
 * `kind ∈ {prompt, function}` passes the op definition itself (higher-order). An explicitly-passed
 * value overrides a binding.
 *
 * Every authored sugar was lowered by the loader onto the well-known resolver functions
 * (`RESOLVER_REFS`), which is why this module has one uniform producer path and no wiring
 * special cases.
 */
import type { InlineFamily, JsonValue, Operation, Parameter, Ref, RefTree, ResolvedValue } from "@declarative-ai/exec";
import { isOk } from "@declarative-ai/exec";
import { evaluate, isPending, parseExpression, PENDING, type Pending } from "./expr";
import { RESOLVER_REFS } from "./format";

/** What a resolution can yield: a value, PENDING (an async producer still in flight), or an error. */
export type Resolved = { value: ResolvedValue } | Pending | { error: string };

export function isResolvedValue(r: Resolved): r is { value: JsonValue } {
  return typeof r === "object" && r !== null && "value" in r;
}
export function isResolveError(r: Resolved): r is { error: string } {
  return typeof r === "object" && r !== null && "error" in r;
}

/**
 * The run-scoped view a resolution needs. The engine implements it over the current instance; it is
 * an interface so the resolver stays free of engine internals (and testable on its own).
 */
export interface ResolutionScope {
  /** The DSL evaluation context (`inputs`, `children.*.outputs`, `run`, `limits`, …). */
  exprContext: Record<string, unknown>;
  /** A declared child's outputs: a value when it has run, PENDING while in flight, undefined when
   *  it has not started. Producer edges named by a local key resolve through this — the child
   *  already having run IS the memo hit. */
  childOutputs(key: string): JsonValue | Pending | undefined;
  /** This state's resolved input values, by name. */
  scopeValue(name: string): JsonValue | undefined;
  /** A session-owned artifact's content, by name. */
  artifact(name: string): JsonValue | undefined;
  /** A session's transcript, or one message of it. */
  conversation(session: string, message?: number): JsonValue | undefined;
}

/** Resolve one binding to a value. */
export function resolveRef(ref: Ref<InlineFamily>, scope: ResolutionScope): Resolved {
  if ("text" in ref) return { value: ref.text };
  if ("json" in ref) return { value: ref.json };
  // A `blob` leaf IS the bytes (DESIGN §3.7): hydration is the family's business, so there is
  // no store to consult and no reference form to resolve.
  if ("blob" in ref) return { value: ref.blob };
  if ("result" in ref) {
    // An ALREADY-EXISTING OperationRecord: its recorded output value fills the parameter. The record
    // stores the same `Result` envelope the live call returned, so there is no kind-tagged
    // `{text|json|blob}` to unwrap — that tag was a third copy of the producing op's `Parameter.kind`.
    const r = ref.result.result;
    return isOk(r) ? { value: r.value } : { error: r.error.reason };
  }
  if ("refs" in ref) return resolveTree(ref.refs, scope);
  return resolveProducer(ref.op, scope);
}

/** A tree position that is not a primitive — the only shape the leaf/node discrimination applies to. */
type TreeNode = Exclude<RefTree<InlineFamily>, string | number | boolean | null>;

/**
 * Is this tree node a LEAF (a `Ref`) or an object node whose keys happen to include a leaf keyword?
 *
 * `RefTree` is an untagged union — `{ text: "hi" }` is a leaf, `{ text: <subtree>, body: <subtree> }`
 * is an object with a property called `text` — so the discriminator has to be structural. Testing
 * `"text" in node` alone read the second form as a leaf and silently produced the wrong value. Two
 * conditions, both required:
 *
 *  - exactly ONE own key, and it is a leaf keyword (a multi-key node is an object, always); and
 *  - the value has the SHAPE that keyword promises (`text` holds a string, `blob` holds bytes or a
 *    stream, `result` holds an operation record — one carrying its own `result` envelope) — so
 *    `{ text: { text: "hi" } }` is an object node, which is the only reading under which it means
 *    anything.
 *
 * `{ json: x }` stays irreducibly ambiguous, because a json leaf holds any JSON value and so does an
 * object property named `json`. It resolves as a LEAF; to build a record with a literal `json` key,
 * nest it (`{ refs: { wrapper: { json: { json: … } } } }`).
 */
function isRefLeaf(tree: TreeNode): boolean {
  const keys = Object.keys(tree);
  if (keys.length !== 1) return false;
  const node = tree as Record<string, unknown>;
  const key = keys[0]!;
  if (key === "json") return true;
  if (key === "text") return typeof node["text"] === "string";
  if (key === "blob") {
    const v = node["blob"];
    return v instanceof Uint8Array || (v !== null && typeof v === "object" && typeof (v as { getReader?: unknown }).getReader === "function");
  }
  if (key === "result") {
    // A `result` leaf holds an `OperationRecord`, whose defining field is its own `result` envelope.
    // Without that check `{ result: { text: "x" } }` read as a leaf and then dereferenced a `result`
    // that was not there; as an OBJECT NODE it means the one thing it can mean, `{ result: "x" }`.
    const v = node["result"];
    return v !== null && typeof v === "object" && !Array.isArray(v) && "result" in v;
  }
  return false;
}

/**
 * Producer-edge and unlowered-sugar shapes that are legal in a `Ref` but NOT in a tree, mapped to the
 * other keys each form may carry.
 *
 * `{ op }` is a `Ref` case, deliberately not a `RefTree` leaf: a tree is an inline ARRANGEMENT of
 * already-resolvable leaves, so a producer that has to be run belongs at the parameter that binds it.
 * The sugar forms (`BindingDecl`) are lowered by the loader at a parameter's `binding` only — it does
 * not walk into `refs` trees — so one written inside a tree arrives here unlowered. Both used to fall
 * into the structural walk below and recurse until the stack died.
 */
const NON_TREE_FORMS: Readonly<Record<string, readonly string[]>> = {
  op: ["parameters"],
  child: ["output"],
  input: [],
  expr: [],
  artifact: [],
  conversation: ["message"],
};

/** The offending keyword when a tree node is really a producer edge or unlowered sugar. */
function nonTreeFormOf(node: Record<string, unknown>): string | undefined {
  const keys = Object.keys(node);
  for (const [keyword, companions] of Object.entries(NON_TREE_FORMS)) {
    if (!keys.includes(keyword)) continue;
    if (keys.every((k) => k === keyword || companions.includes(k))) return keyword;
  }
  return undefined;
}

/**
 * Resolve an inline arrangement of refs — the same shape with each leaf replaced by its value.
 *
 * TOTAL by construction: every input shape either resolves, parks, or errors. It used to structurally
 * walk anything it did not recognize, which turned a string into an infinite recursion over its own
 * characters, a number or boolean into `{}` (silently wrong data flowing on down the graph), and
 * `null` into a thrown `TypeError`.
 */
function resolveTree(tree: RefTree<InlineFamily>, scope: ResolutionScope): Resolved {
  // A PRIMITIVE is a literal JSON value: its own value, resolved as itself. `null` included — it is a
  // value, and it used to reach `Object.entries(null)` and throw.
  if (tree === null || typeof tree === "string" || typeof tree === "number" || typeof tree === "boolean") return { value: tree };
  // Anything else non-object cannot be a tree at all: unreachable by type, REPORTED rather than thrown.
  if (typeof tree !== "object") return { error: `a ref tree cannot contain ${typeof tree}` };
  if (Array.isArray(tree)) {
    const out: ResolvedValue[] = [];
    for (const item of tree) {
      const r = resolveTree(item, scope);
      if (!isResolvedValue(r)) return r;
      out.push(r.value);
    }
    return { value: out };
  }
  if (isRefLeaf(tree)) return resolveRef(tree as Ref<InlineFamily>, scope);
  const offending = nonTreeFormOf(tree as Record<string, unknown>);
  if (offending !== undefined) {
    return {
      error:
        offending === "op"
          ? "a producer edge ({ op }) cannot be nested in a ref tree — bind it at the parameter instead"
          : `'${offending}' binding sugar cannot be nested in a ref tree — bind it at the parameter instead`,
    };
  }
  const out: Record<string, ResolvedValue> = {};
  for (const [key, sub] of Object.entries(tree)) {
    const r = resolveTree(sub, scope);
    if (!isResolvedValue(r)) return r;
    out[key] = r.value;
  }
  return { value: out };
}

/**
 * Resolve a PRODUCER edge. A local key names a declared child — resolved against the run (already
 * run ⇒ reuse its outputs; in flight ⇒ PENDING). An embedded op is either one of the well-known
 * resolvers the loader synthesized, or an author-embedded operation the engine must run.
 */
function resolveProducer(producer: Operation<InlineFamily> | string, scope: ResolutionScope): Resolved {
  if (typeof producer === "string") {
    const outputs = scope.childOutputs(producer);
    if (outputs === undefined) return { error: `child '${producer}' has not run` };
    if (isPending(outputs)) return PENDING;
    return { value: outputs };
  }
  if (producer.kind !== "function") {
    // A higher-order slot: the op DEFINITION itself is the value (§2.1).
    return { value: producer as unknown as JsonValue };
  }
  return runResolver(producer, scope);
}

/** Run one of the well-known resolver functions (the desugaring targets, §2.1). */
function runResolver(op: Operation<InlineFamily> & { kind: "function" }, scope: ResolutionScope): Resolved {
  const arg = (name: string): Resolved | undefined => {
    const p = op.input[name] as Parameter<InlineFamily> | undefined;
    return p?.binding ? resolveRef(p.binding, scope) : undefined;
  };
  const text = (name: string): string | undefined => {
    const r = arg(name);
    return r && isResolvedValue(r) && typeof r.value === "string" ? r.value : undefined;
  };

  switch (op.functionRef) {
    case RESOLVER_REFS.expr: {
      const source = text("source");
      if (source === undefined) return { error: "expr producer has no source" };
      let value: unknown;
      try {
        value = evaluate(parseExpression(source), scope.exprContext);
      } catch (e) {
        return { error: `expression failed: ${(e as Error).message}` };
      }
      if (isPending(value)) return PENDING;
      return { value: value as JsonValue };
    }
    case RESOLVER_REFS.select: {
      const base = arg("value");
      const key = text("key");
      if (base === undefined || key === undefined) return { error: "select producer is missing value/key" };
      if (!isResolvedValue(base)) return base;
      const v = base.value;
      if (v === null || typeof v !== "object" || Array.isArray(v)) {
        return { error: `cannot select '${key}' from a non-object producer output` };
      }
      const picked = v[key];
      return picked === undefined ? { error: `producer output has no '${key}'` } : { value: picked };
    }
    case RESOLVER_REFS.scope: {
      const name = text("name");
      if (name === undefined) return { error: "scope producer has no name" };
      const v = scope.scopeValue(name);
      return v === undefined ? { error: `input '${name}' is not set` } : { value: v };
    }
    case RESOLVER_REFS.artifact: {
      const name = text("name");
      if (name === undefined) return { error: "artifact producer has no name" };
      const v = scope.artifact(name);
      return v === undefined ? { error: `artifact '${name}' is not available` } : { value: v };
    }
    case RESOLVER_REFS.conversation: {
      const session = text("session");
      if (session === undefined) return { error: "conversation producer has no session" };
      const messageArg = arg("message");
      const message = messageArg && isResolvedValue(messageArg) && typeof messageArg.value === "number" ? messageArg.value : undefined;
      const v = scope.conversation(session, message);
      return v === undefined ? { error: `conversation '${session}' is not available` } : { value: v };
    }
    default:
      // An author-embedded op is a genuine sub-operation; running it belongs to the engine's
      // dispatch, not to reference resolution.
      return { error: `embedded operation '${op.functionRef}' cannot be resolved as a reference` };
  }
}

/**
 * Resolve every BOUND parameter of an operation into a value map, leaving free slots to the
 * caller (which fills them by name from the consuming scope, per the model's §3.8 rule).
 * PENDING short-circuits — the state parks until the producers resolve (the dataflow join).
 */
export function resolveInputs(
  input: Record<string, Parameter<InlineFamily>>,
  scope: ResolutionScope,
): { values: Record<string, ResolvedValue> } | Pending | { error: string } {
  const values: Record<string, ResolvedValue> = {};
  for (const [name, param] of Object.entries(input)) {
    if (!param.binding) continue;
    const r = resolveRef(param.binding, scope);
    if (isPending(r)) return PENDING;
    if (isResolveError(r)) return { error: `input '${name}': ${r.error}` };
    values[name] = r.value;
  }
  return { values };
}
