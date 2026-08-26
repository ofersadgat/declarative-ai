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
import { carryCall, isOk } from "@declarative-ai/exec";
import { applyBinary, evaluate, isPending, memberOf, parseExpression, PENDING, type BinaryOp, type Pending } from "./expr.js";
import type { Failure } from "@declarative-ai/json";
import { admitsError, errorValueSchemaFor, resolutionFailure } from "./errorValue.js";
import { BUILTINS } from "./builtins.js";
import { RESOLVER_REFS, RESOLVER_REF_SET } from "./format.js";

/**
 * What a resolution can yield: a value, PENDING (an async producer still in flight), or an error.
 *
 * The error case carries the CLASSIFIED failure when there is one. Flattening it to `reason` alone
 * loses the thing that distinguishes kinds of failure — a recorded `network-retriable` provider
 * error and a wiring mistake would arrive identical, and anything routing on classification (§5)
 * would see every failure as the same one.
 */
export type Resolved = { value: ResolvedValue } | Pending | { error: string; failure?: Failure };

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
  /**
   * The result of an EMBEDDED operation — a lowered call (EXPRESSIONS.md §3).
   *
   * Resolution does not run operations: it is synchronous (`renderTemplate` resolves inside a
   * `String.replace` callback, which cannot await) and it is re-run every scheduling round. So an
   * embedded op follows exactly the protocol a child does — the engine runs it and records the
   * result, resolution READS it — and `undefined` means "not run yet", which parks the consumer
   * through the same PENDING join.
   */
  operationResult?(op: Operation<InlineFamily>): Resolved | undefined;
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
    // The recorded failure travels WHOLE, not flattened to its reason: it is the one error here that
    // already carries a real classification, and a consumer declaring it handles retriable failures
    // has to be able to tell one from a wiring mistake (§5).
    return isOk(r) ? { value: r.value } : { error: r.error.reason, failure: r.error };
  }
  if ("refs" in ref) return resolveTree(ref.refs, scope);
  return resolveProducer(ref, scope);
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
  expr: [],
};

/**
 * The id inside a session ref — `{ id }`, the one enumerable field {@link SESSION_REF} declares.
 *
 * A bare string is refused deliberately. It would be a second spelling, and the plausible one to
 * write by hand is a session NAME, which no longer addresses anything: names became positions.
 */
function sessionIdOf(value: unknown): string | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
  const id = (value as { id?: unknown }).id;
  return typeof id === "string" && id !== "" ? id : undefined;
}

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
function resolveProducer(ref: { op: Operation<InlineFamily> | string; parameters?: Record<string, Parameter<InlineFamily>> }, scope: ResolutionScope): Resolved {
  const producer = ref.op;
  if (typeof producer === "string") {
    const outputs = scope.childOutputs(producer);
    // A child that HAS NOT RUN is `undefined`, not an error (SPEC §3.4): "references such as
    // `children.<id>.outputs` ... evaluate to `undefined` if the child has not run". Refusing here
    // made the same reference mean two different things depending on which form it was written in —
    // `".children.c.outputs.x"` lowers to this edge and refused, while `{ expr: ".children.c.outputs.x" }`
    // lowers to a `member` chain and yielded `undefined` — so a guard was lenient and a wire was not.
    // Whether an absent value is ACCEPTABLE is the consuming slot's to say, through `optional`/`default`
    // (§7.2's named opt-out); whether it is REACHABLE at all is the validator's, through the
    // definite-assignment analysis. Neither question belongs to the resolver.
    //
    // IN FLIGHT is still PENDING and not `undefined` — §6 is explicit that a started-but-unfinished
    // child parks its consumer rather than reading as permanently missing.
    if (outputs === undefined) return { value: undefined as unknown as JsonValue };
    if (isPending(outputs)) return PENDING;
    return { value: outputs };
  }
  const isResolver = producer.kind === "function" && RESOLVER_REF_SET.has(producer.functionRef);
  // A HIGHER-ORDER slot passes the op definition through as the value (§2.1) — but only when the
  // consumer asked for a definition, which it signals by binding an edge with NO arguments. A lowered
  // call always carries `parameters`, so the two are distinguishable here without threading the
  // consuming parameter's `kind` down through a recursive resolver.
  //
  // This branch used to key on `producer.kind !== "function"`, which meant every PROMPT operation was
  // read as higher-order — so calling one returned its own definition as the value instead of running
  // it, and the binding got an object where the callee's result belonged.
  if (!isResolver && ref.parameters === undefined && producer.kind !== "function") {
    return { value: producer as unknown as JsonValue };
  }
  if (!isResolver) {
    // A lowered CALL (§3). Running it belongs to the engine, exactly as running a child does; here
    // it is only READ — and read under the identity of the op WITH ITS ARGUMENTS BOUND, so a call
    // whose argument changed between rounds is a different call and not a stale hit.
    const resolved = resolveEmbedded(producer, ref.parameters, scope);
    if (isPending(resolved)) return PENDING;
    if ("error" in resolved) return resolved;
    return scope.operationResult?.(resolved.op) ?? PENDING;
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
  /**
   * An OPERAND — an argument with a PENDING that arrived as a *value* normalized onto the PENDING
   * result.
   *
   * The engine seeds the expression context with the sentinel itself for a running child
   * (`{ outputs: PENDING, outcome: PENDING }`), so reading `children.running.outcome` produces a
   * resolved value that IS PENDING rather than a pending resolution. The interpreter collapses the
   * two with its `isPending(obj)` check before every operator; without the same collapse here,
   * `children.running.outcome === 'success'` compared the sentinel against a string and answered
   * `false` — a guard firing on a child that has not finished.
   */
  const operand = (name: string): Resolved | undefined => {
    const r = arg(name);
    return r !== undefined && isResolvedValue(r) && isPending(r.value) ? PENDING : r;
  };
  /** A produced value, with the same collapse applied on the way out. */
  const produced = (v: unknown): Resolved => (isPending(v) ? PENDING : { value: v as JsonValue });

  switch (op.functionRef) {
    // `expr.eval` — an interpreter invoked on a source string carried through the document — is gone
    // (EXPRESSIONS.md §1). An expression is lowered to a tree of the operator resolvers below, so it
    // is parsed exactly once, at load, and resolved by the same walk as every other binding.
    case RESOLVER_REFS.select: {
      const base = arg("value");
      const key = text("key");
      if (base === undefined || key === undefined) return { error: "select producer is missing value/key" };
      if (!isResolvedValue(base)) return base;
      const v = base.value;
      // `select` is the child-output projection and NOTHING else — the loader emits it in exactly one
      // place, for `.children.<key>.outputs.<name>` — so it reads exactly as `memberOf` does, which is
      // what the same path means when it is written as an expression. Absence at either step (the child
      // never ran, or ran without producing this output) is `undefined`, per SPEC §3.4; OWN properties
      // only, so `.outputs.constructor` is absent rather than a function off the prototype.
      //
      // A name the producer does not DECLARE is a load-time error already ("selects output 'x', which
      // the producer does not declare"), so refusing it again here only ever fired for an optional
      // output a run legitimately omitted — the one case where `undefined` is the right answer.
      return produced(memberOf(v, key));
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
      // A session REF, not a name. The engine mirrors transcripts under the session's id — a
      // POSITION, `planning@3` — so a name only ever matched a conversation that had had no calls,
      // which is the one nobody wants to read. The ref comes from `.operation.output.session`,
      // which is how a conversation is addressable at all once it is a position rather than a name.
      const sessionArg = arg("session");
      if (sessionArg === undefined || !isResolvedValue(sessionArg)) return sessionArg ?? { error: "messages() has no session" };
      const id = sessionIdOf(sessionArg.value);
      if (id === undefined) {
        return { error: `messages() needs a session ref ({ id }), got ${JSON.stringify(sessionArg.value)}` };
      }
      const v = scope.conversation(id, undefined);
      return v === undefined ? { error: `the conversation at '${id}' is not available` } : { value: v };
    }
    // --- Operators (EXPRESSIONS.md §2) ----------------------------------------
    //
    // The semantics come from `expr.ts` — `applyBinary`, `memberOf` — rather than being restated
    // here, so a lowered expression cannot mean something different from what the interpreter
    // meant.
    case RESOLVER_REFS.context: {
      const name = text("name");
      if (name === undefined) return { error: "context producer has no name" };
      // An unknown root is `undefined`, exactly as an identifier missing from the context is —
      // catching a typo is the VALIDATOR's job, and failing here would turn a lint error into a
      // run-time one.
      return produced(scope.exprContext[name]);
    }
    case RESOLVER_REFS.member: {
      const base = operand("value");
      const prop = text("prop");
      if (base === undefined || prop === undefined) return { error: "member producer is missing value/prop" };
      if (!isResolvedValue(base)) return base; // PENDING or an error propagates
      return produced(memberOf(base.value, prop));
    }
    case RESOLVER_REFS.not: {
      const v = operand("value");
      if (v === undefined) return { error: "not producer has no value" };
      if (!isResolvedValue(v)) return v;
      return { value: !v.value };
    }
    case RESOLVER_REFS.eq:
    case RESOLVER_REFS.ne:
    case RESOLVER_REFS.strictEq:
    case RESOLVER_REFS.strictNe:
    case RESOLVER_REFS.lt:
    case RESOLVER_REFS.le:
    case RESOLVER_REFS.gt:
    case RESOLVER_REFS.ge: {
      const l = operand("left");
      const r = operand("right");
      if (l === undefined || r === undefined) return { error: `'${op.functionRef}' producer is missing left/right` };
      if (!isResolvedValue(l)) return l;
      if (!isResolvedValue(r)) return r;
      return { value: applyBinary(BINARY_FOR[op.functionRef]!, l.value, r.value) };
    }
    // The lazy three. `arg` RESOLVES ON DEMAND, so not asking for a branch is not evaluating it —
    // which is what keeps `false && PENDING` determinate at `false`, and what will keep an untaken
    // branch from spending money once a branch can be an operation (EXPRESSIONS.md §6).
    case RESOLVER_REFS.and: {
      const l = operand("left");
      if (l === undefined) return { error: "and producer is missing left" };
      if (!isResolvedValue(l)) return l;
      if (!l.value) return l; // determinate-falsy short-circuit, past a pending right
      const r = operand("right");
      return r === undefined ? { error: "and producer is missing right" } : r;
    }
    case RESOLVER_REFS.or: {
      const l = operand("left");
      if (l === undefined) return { error: "or producer is missing left" };
      if (!isResolvedValue(l)) return l;
      if (l.value) return l;
      const r = operand("right");
      return r === undefined ? { error: "or producer is missing right" } : r;
    }
    case RESOLVER_REFS.record: {
      // An OBJECT LITERAL: every input slot is a key the author wrote, so this reads them all rather
      // than asking for a signature's worth of names. Strict in every value — `PENDING` or an error
      // in one entry is the whole object's answer, because an object holding a sentinel is not one
      // a consumer can read.
      const out: Record<string, JsonValue> = {};
      for (const name of Object.keys(op.input)) {
        const v = operand(name);
        if (v === undefined) continue; // a key bound to nothing is absent, not `null`
        if (!isResolvedValue(v)) return v;
        // `defineProperty`, exactly as `fromEntries` does: `{ __proto__: x }` must store a key, not
        // re-parent the object through an inherited setter.
        Object.defineProperty(out, name, { value: v.value, writable: true, enumerable: true, configurable: true });
      }
      return { value: out };
    }
    case RESOLVER_REFS.cond: {
      const t = operand("test");
      if (t === undefined) return { error: "cond producer is missing test" };
      if (!isResolvedValue(t)) return t;
      const taken = operand(t.value ? "then" : "else");
      return taken === undefined ? { error: `cond producer is missing '${t.value ? "then" : "else"}'` } : taken;
    }
    case "reduce": {
      // A FOLD: each step consumes the previous result, so the applications cannot be built up front
      // the way `map`'s can. Both sides rebuild the same CHAIN — the engine to run it, this to read
      // it — which works because every step's result is recorded under its own content hash.
      const higher = higherOrderOf({ op });
      if (higher === undefined) return { error: "'reduce' is missing its array or its operation" };
      const source = operand("value");
      const seed = operand("initial");
      if (source === undefined || !isResolvedValue(source)) return source ?? { error: "'reduce' has no array" };
      if (!Array.isArray(source.value)) return { error: "'reduce' expects an array" };
      if (seed !== undefined && !isResolvedValue(seed)) return seed;

      let acc = (seed?.value ?? null) as JsonValue;
      for (const element of source.value) {
        const applied = scope.operationResult?.(bindElement(higher.op, element as unknown as JsonValue, acc));
        if (applied === undefined || isPending(applied)) return PENDING;
        // A failed step stops the fold: there is no accumulator to carry forward.
        if (isResolveError(applied)) return applied;
        acc = applied.value as JsonValue;
      }
      return { value: acc };
    }
    case "map":
    case "filter":
    case "flatMap": {
      // Applied per element by the ENGINE (it dispatches; this does not). Here the results are only
      // READ — one per element, under the same identity the engine recorded them with.
      const higher = higherOrderOf({ op });
      if (higher === undefined) return { error: `'${op.functionRef}' is missing its array or its operation` };
      const source = operand("value");
      if (source === undefined) return { error: `'${op.functionRef}' has no array` };
      if (!isResolvedValue(source)) return source;
      if (!Array.isArray(source.value)) return { error: `'${op.functionRef}' expects an array` };

      const results: JsonValue[] = [];
      for (const element of source.value) {
        const applied = scope.operationResult?.(bindElement(higher.op, element as unknown as JsonValue));
        if (applied === undefined) return PENDING; // not run yet — park, exactly as a child read does
        if (isPending(applied)) return PENDING;
        // A failed element is error DATA (§5): it travels in the array, and the CONSUMING slot
        // decides whether that is acceptable. Failing the whole thing here would throw away both the
        // successful results and which element failed.
        const value = (isResolveError(applied) ? { error: applied.failure ?? { classification: "permanent", reason: applied.error } } : applied.value) as JsonValue;
        if (op.functionRef === "map") results.push(value);
        else if (op.functionRef === "flatMap") results.push(...(Array.isArray(value) ? (value as JsonValue[]) : [value]));
        else if (value) results.push(element as unknown as JsonValue); // filter keeps the ELEMENT, not the test
      }
      return { value: results };
    }
    default: {
      // A BUILT-IN from the operation library (§3): eager, pure and total, so it is applied right
      // here — the same treatment the comparison operators get, because it is the same kind of thing.
      const builtin = BUILTINS[op.functionRef];
      if (builtin === undefined) {
        // Unreachable: an embedded op is handled by `resolveProducer` before the resolver switch.
        return { error: `embedded operation '${op.functionRef}' cannot be resolved as a reference` };
      }
      const values: Record<string, unknown> = {};
      for (const name of builtin.params) {
        const a = operand(name);
        if (a === undefined) continue; // an omitted optional argument is `undefined`, not an error
        if (!isResolvedValue(a)) return a; // PENDING or an error propagates
        values[name] = a.value;
      }
      return produced(builtin.fn(values));
    }
  }
}

/** Which binary operator each comparison resolver applies — the names are the only difference. */
const BINARY_FOR: Readonly<Record<string, BinaryOp>> = {
  [RESOLVER_REFS.eq]: "==",
  [RESOLVER_REFS.ne]: "!=",
  [RESOLVER_REFS.strictEq]: "===",
  [RESOLVER_REFS.strictNe]: "!==",
  [RESOLVER_REFS.lt]: "<",
  [RESOLVER_REFS.le]: "<=",
  [RESOLVER_REFS.gt]: ">",
  [RESOLVER_REFS.ge]: ">=",
};

/**
 * An embedded operation with its call ARGUMENTS bound into its own input slots.
 *
 * The one definition of a lowered call's identity. The engine runs this and records the result under
 * its content hash; resolution rebuilds the same thing to read it back. Two copies of the rule would
 * hash differently and the memo would never hit — which is precisely the bug that made this a shared
 * function rather than two.
 */
export function resolveEmbedded(
  op: Operation<InlineFamily>,
  parameters: Record<string, Parameter<InlineFamily>> | undefined,
  scope: ResolutionScope,
): { op: Operation<InlineFamily> } | Pending | { error: string } {
  const bound = parameters ? resolveInputs(parameters, scope) : { values: {} as Record<string, ResolvedValue> };
  if (isPending(bound)) return PENDING;
  if ("error" in bound) return { error: `call argument: ${bound.error}` };
  const spread = resolveSpread(op, scope);
  if (isPending(spread)) return PENDING;
  if ("error" in spread) return spread;
  const values = { ...spread.values, ...bound.values };
  const input = Object.fromEntries(
    Object.entries(op.input).map(([name, p]) => {
      const value = values[name];
      return [name, value === undefined ? p : { ...p, binding: { json: value as JsonValue } }];
    }),
  );
  // `spread` is DROPPED from the result, not carried: it has become `input` entries, and leaving it
  // on would make the op hash as though the arguments were still pending — two identities for one
  // call, and the memo would miss on the second.
  const { spread: _expanded, ...rest } = op;
  return { op: { ...rest, input } as Operation<InlineFamily> };
}

/**
 * Expand an operation's {@link SpreadArguments} into the values they name (SPEC §6.3).
 *
 * The deferred half of `f(...opts)`: at load its keys were in a type nobody had computed yet, and
 * here they are simply the keys of a value. That asymmetry is the whole reason the field exists —
 * everything static about the call was checked by the validator, and what is left is a read.
 *
 * Later spreads win over earlier ones, and an explicitly bound slot wins over both. The validator
 * refuses a call whose spread names a slot already filled, so a workflow that reaches this rule was
 * not checked; it resolves the tie the way the rest of the loader does, with the more specific
 * statement — a slot the author wrote out by name — beating the bag it came in.
 */
function resolveSpread(
  op: Operation<InlineFamily>,
  scope: ResolutionScope,
): { values: Record<string, ResolvedValue> } | Pending | { error: string } {
  const values: Record<string, ResolvedValue> = {};
  for (const ref of op.spread ?? []) {
    const r = resolveRef(ref, scope);
    if (isPending(r)) return PENDING;
    if (isResolveError(r)) return { error: `spread argument: ${r.error}` };
    const value = r.value;
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return {
        error: `spread argument resolved to ${value === null ? "null" : Array.isArray(value) ? "an array" : `a ${typeof value}`}, which names no arguments`,
      };
    }
    Object.assign(values, value as Record<string, ResolvedValue>);
  }
  return { values };
}

/**
 * An operation with already-resolved values bound into its input slots as literals.
 *
 * The precondition at the dispatch boundary: an operation handed to an `Executor` is READY TO RUN,
 * which means everything it takes is a literal on the op itself. An executor reads inputs off the op
 * (`resolveLiteralInputs`) and cannot see the instance the engine resolved them against, so the
 * engine has to write them down before dispatching. `resolveEmbedded` is the same move for a call's
 * arguments; this is it for a state's own operation.
 *
 * A name the op does not DECLARE is added as a `json` slot rather than dropped, which preserves what
 * the impl was handed when the engine called it directly: `{...instance.inputs, ...resolved.values}`
 * — the state's inputs merged under the operation's own bound ones. Narrowing that to declared slots
 * only is a separate decision, and a breaking one: it would stop a function impl reading a state
 * input its operation never declared.
 */
export function bindInputs(op: Operation<InlineFamily>, values: Record<string, ResolvedValue>): Operation<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = { ...op.input };
  for (const [name, value] of Object.entries(values)) {
    const binding = { json: value as JsonValue };
    const declared = input[name];
    input[name] = declared ? { ...declared, binding } : { kind: "json", binding };
  }
  // A spread has BECOME these values by now — the caller resolved it through `resolveOperationInputs`
  // — so the field is dropped rather than carried. Leaving it on would hash the op as though its
  // arguments were still pending, giving one call two identities and costing the memo every hit.
  const { spread: _expanded, ...rest } = op;
  // `carryCall` because the spread drops a pre-resolved entry, and this runs immediately before the
  // dispatch that would have used it — the one place losing it costs the most.
  return carryCall(op, { ...rest, input } as Operation<InlineFamily>);
}

/**
 * The higher-order operations: they APPLY an operation to each element of an array (EXPRESSIONS.md
 * §3.5). Unlike every other builtin they dispatch, so the engine runs them and resolution reads.
 */
export const HIGHER_ORDER: ReadonlySet<string> = new Set(["map", "filter", "flatMap", "reduce"]);

/** A higher-order edge, taken apart: the array it walks and the operation it applies. */
export function higherOrderOf(
  ref: Ref<InlineFamily>,
): { name: string; value: Ref<InlineFamily>; op: Operation<InlineFamily> } | undefined {
  if (!("op" in ref)) return undefined;
  const producer = ref.op;
  if (typeof producer === "string" || producer.kind !== "function" || !HIGHER_ORDER.has(producer.functionRef)) return undefined;
  const value = producer.input.value?.binding;
  const opArg = producer.input.op?.binding;
  if (value === undefined || opArg === undefined || !("op" in opArg) || typeof opArg.op === "string") return undefined;
  return { name: producer.functionRef, value, op: opArg.op };
}

/** Every higher-order edge in a binding tree, innermost first. */
export function higherOrderEdgesOf(ref: Ref<InlineFamily>): Ref<InlineFamily>[] {
  const out: Ref<InlineFamily>[] = [];
  const walk = (node: Ref<InlineFamily>): void => {
    if (!("op" in node)) return;
    const producer = node.op;
    if (typeof producer === "string" || producer.kind !== "function") return;
    for (const [name, p] of Object.entries(producer.input)) {
      // Not the `op` position: that is a DEFINITION to apply, not an edge to run.
      if (name === "op" && higherOrderOf(node) !== undefined) continue;
      if (p.binding) walk(p.binding);
    }
    for (const p of Object.values(node.parameters ?? {})) if (p.binding) walk(p.binding);
    for (const s of producer.spread ?? []) walk(s);
    if (higherOrderOf(node) !== undefined) out.push(node);
  };
  walk(ref);
  return out;
}

/**
 * One element bound into an operation — the identity of ONE application.
 *
 * The same role `resolveEmbedded` plays for a call, and shared for the same reason: the engine
 * hashes this to record a result and resolution hashes it to read one back. Two copies of the rule
 * would hash differently and every lookup would miss.
 *
 * The element fills the operation's FIRST parameter, by declared `index` where it has one.
 */
export function bindElement(op: Operation<InlineFamily>, element: JsonValue, accumulator?: JsonValue): Operation<InlineFamily> {
  const entries = Object.entries(op.input);
  const indexed = entries.filter(([, p]) => p.index !== undefined);
  const ordered = indexed.length > 0 ? indexed.sort(([, a], [, b]) => (a.index ?? 0) - (b.index ?? 0)) : entries;
  const first = ordered[0]?.[0];
  if (first === undefined) return op;
  // `reduce` folds, so its operation takes TWO: the accumulator first, then the element. Everything
  // else takes the element alone.
  if (accumulator === undefined) {
    return { ...op, input: { ...op.input, [first]: { ...op.input[first]!, binding: { json: element } } } } as Operation<InlineFamily>;
  }
  const second = ordered[1]?.[0];
  const input = { ...op.input, [first]: { ...op.input[first]!, binding: { json: accumulator } } };
  if (second !== undefined) input[second] = { ...op.input[second]!, binding: { json: element } };
  return { ...op, input } as Operation<InlineFamily>;
}

/**
 * Every EMBEDDED operation a binding tree carries — the lowered calls the engine must run before
 * resolution can read their results (§3).
 *
 * Nested ones come out innermost-first, so `f(g(x))` yields `g` before `f`: an outer call's
 * parameters cannot resolve until the inner one has a result.
 */
export function embeddedOpsOf(ref: Ref<InlineFamily>): Array<{ op: Operation<InlineFamily>; parameters?: Record<string, Parameter<InlineFamily>> }> {
  const out: Array<{ op: Operation<InlineFamily>; parameters?: Record<string, Parameter<InlineFamily>> }> = [];
  const walk = (node: Ref<InlineFamily>): void => {
    if (!("op" in node)) return;
    const producer = node.op;
    if (typeof producer === "string") return; // a declared child: the engine already runs it
    if (producer.kind === "function" && RESOLVER_REF_SET.has(producer.functionRef)) {
      // A resolver is computed inline, but its ARGUMENTS may carry calls. A HIGHER-ORDER edge is the
      // exception: its `op` argument is a DEFINITION to apply per element, not a call to run once —
      // running it here would invoke it a single time with no element bound.
      const higher = higherOrderOf(node);
      for (const [name, p] of Object.entries(producer.input)) {
        if (higher !== undefined && name === "op") continue;
        if (p.binding) walk(p.binding);
      }
      for (const s of producer.spread ?? []) walk(s);
      return;
    }
    for (const p of Object.values(node.parameters ?? {})) if (p.binding) walk(p.binding);
    // A SPREAD's operand is an argument, so a call inside it must run before this one — `f(...g())`
    // yields `g` first for exactly the reason `f(g(x))` does.
    for (const s of producer.spread ?? []) walk(s);
    out.push({ op: producer, ...(node.parameters !== undefined ? { parameters: node.parameters } : {}) });
  };
  walk(ref);
  return out;
}

/**
 * An operation's resolved arguments: its bound `input` slots, over its expanded {@link SpreadArguments}.
 *
 * The one entry point for "what is this operation being passed", so a spread cannot be forgotten at
 * one of the two places a call reaches dispatch. A slot the author wrote out by name beats one the
 * spread named, on the same rule `bindIntoSlots` follows: the more specific statement wins.
 */
export function resolveOperationInputs(
  op: Operation<InlineFamily>,
  scope: ResolutionScope,
): { values: Record<string, ResolvedValue> } | Pending | { error: string } {
  const spread = resolveSpread(op, scope);
  if (isPending(spread)) return PENDING;
  if ("error" in spread) return spread;
  const bound = resolveInputs(op.input, scope);
  if (isPending(bound)) return PENDING;
  if ("error" in bound) return bound;
  return { values: { ...spread.values, ...bound.values } };
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
    if (isResolveError(r)) {
      // A failure is DATA (EXPRESSIONS.md §5). If the consuming slot declared that it accepts one of
      // this shape, it flows in as a value and the operation runs; otherwise this is the implicit
      // unwrap and the operation terminates with it — which is what always happened before.
      //
      // Acceptance is checked against THIS failure's shape, not against "any failure": a slot that
      // declares it handles `policy-denied` must accept a policy denial and refuse a timeout, which
      // is the whole point of failures being differently shaped by kind.
      const failure = r.failure !== undefined ? { error: r.failure } : resolutionFailure(r.error);
      if (admitsError(param.schema, errorValueSchemaFor(failure.error))) {
        values[name] = failure as unknown as ResolvedValue;
        continue;
      }
      return { error: `input '${name}': ${r.error}` };
    }
    values[name] = r.value;
  }
  return { values };
}
