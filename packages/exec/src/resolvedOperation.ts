/**
 * A function operation carrying the entry its name resolves to, so the lookup is paid once instead of
 * on every evaluation.
 *
 * The problem is the hot loop. A lowered expression is a tree of operations — `a === b` is
 * `strictEq(a, b)` and differs from a user's `classify(x)` only in spelling (EXPRESSIONS §2) — and a
 * guard is re-evaluated every scheduling round. Resolving each node's name through the registry per
 * round, then framing each with a handle, an abort link and a metrics record, is a lot of machinery to
 * compute `a === b`. Resolution is the half that can be hoisted: the registry does not change mid-run.
 *
 * THE SHAPE. `ResolvedFunctionOp` EXTENDS `FunctionOp` rather than wrapping it. That is the whole
 * design: a resolved operation IS an operation, so it passes anywhere one is accepted, and a tree
 * holds resolved nodes in the ordinary `input[name].binding.op` slot rather than in a parallel
 * structure that has to be kept in step with it. Resolving is therefore progressive — a partly
 * resolved tree is still a perfectly good operation tree, and every existing walk over it keeps
 * working unchanged.
 *
 * WHY `call` IS NON-ENUMERABLE. The document must stay hashable: `hashOperation` canonicalizes an
 * operation to key the memo, and a snapshot stores resolved operations (EXPRESSIONS §11).
 * Canonicalization walks OWN ENUMERABLE properties (`Object.entries`) and refuses a function outright.
 * A non-enumerable `call` is invisible to it, so a resolved operation canonicalizes — and therefore
 * hashes — identically to the plain document it was built from. That identity is not incidental: a
 * memo key must not depend on whether the caller happened to pre-resolve.
 *
 * WHY YOU NARROW RATHER THAN ASSUME. `call` is REQUIRED on `ResolvedFunctionOp` — an operation that
 * carries one is resolved, by definition. What varies is whether the operation you are holding is one,
 * so a consumer receives a `FunctionOp` and tests with {@link isResolvedCall}. That matters because
 * spreading copies own enumerable properties only, so `{...op}` — which this codebase does constantly
 * (`{...op, input}`, `{...op, user}`) — yields a plain `FunctionOp` that is simply no longer resolved.
 * Nothing has lied: the narrowing just fails and the consumer falls back to a registry lookup.
 * Resolution is therefore an optimization a holder may or may not have, never a guarantee threaded
 * through signatures, and correctness never depends on it.
 */
import type {
  FunctionInputs,
  FunctionOp,
  FunctionRegistry,
  InlineFamily,
  JsonValue,
  Operation,
  Parameter,
  RefFamily,
  RegisteredFunction,
  ResolvedValue,
} from "@declarative-ai/ops";
import { failureOf, isOk } from "@declarative-ai/ops";
import type { ExecMetrics, ExecResult, ExecServices } from "./contract";
import { resolveLiteralInputs } from "./operationExecutor";

/** The entry a resolved operation carries. */
export type CallTarget = RegisteredFunction<ExecServices, ExecMetrics>;

/**
 * A `FunctionOp` that may carry its resolved entry. Structurally a `FunctionOp`, so it is accepted
 * everywhere one is — including as the `op` of a nested producer edge inside another operation.
 */
export type ResolvedFunctionOp<F extends RefFamily = InlineFamily> = FunctionOp<F> & {
  readonly call: CallTarget;
};

/**
 * Attach a resolved entry to an operation, non-enumerably so canonicalization and spread both ignore
 * it. Returns the same object rather than a copy: the property is invisible to every reader that
 * treats an operation as data, so there is nothing to protect callers from by cloning.
 */
export function withCall<T extends FunctionOp<InlineFamily>>(op: T, call: CallTarget): T & ResolvedFunctionOp {
  Object.defineProperty(op, "call", { value: call, enumerable: false, writable: false, configurable: true });
  return op as T & ResolvedFunctionOp;
}

/**
 * Whether this operation carries its resolved entry — the test a consumer makes on the `FunctionOp` it
 * was handed, since resolution is something a holder may or may not have done.
 */
export function isResolvedCall(op: Operation<InlineFamily>): op is ResolvedFunctionOp {
  return op.kind === "function" && (op as ResolvedFunctionOp).call !== undefined;
}

/**
 * Carry a resolved entry across a REBUILD — `{...op, input}` and friends.
 *
 * Spread copies own enumerable properties and `call` is deliberately not one, so every transform that
 * means "this same operation with one field changed" de-resolves the node unless it re-attaches. That
 * costs only speed — the holder falls back to a lookup — which is precisely why it needs a named
 * helper rather than being left to each caller to remember: a silent loss of speed is the kind of
 * thing nobody notices.
 */
export function carryCall<T extends Operation<InlineFamily>>(from: Operation<InlineFamily>, to: T): T {
  if (!isResolvedCall(from) || to.kind !== "function") return to;
  // Attaches in place and returns the same object, so `to` keeps its own narrower type: the cast says
  // "this is still what it was", not "this is now something else".
  withCall(to, from.call);
  return to;
}

/**
 * Resolve every name in an operation tree against a registry, returning the same tree with its
 * function nodes carrying their entries.
 *
 * An unknown name is reported HERE rather than at run time. That is the second reason to resolve up
 * front: today a missing function surfaces mid-run, where the engine has to treat it as run-fatal
 * because a transition could otherwise re-enter the state forever.
 */
export function resolveCalls(
  op: Operation<InlineFamily>,
  functions: FunctionRegistry<ExecServices, ExecMetrics>,
): { op: Operation<InlineFamily> } | { error: string } {
  const input: Record<string, Parameter<InlineFamily>> = { ...op.input };
  for (const [name, param] of Object.entries(op.input)) {
    const binding = param.binding;
    if (binding === undefined || !("op" in binding) || typeof binding.op === "string") continue;
    // A `prompt`/`function` KIND is higher-order: the definition itself is the value, so there is
    // nothing to run and nothing to resolve (model.ts, `Parameter`).
    if (param.kind === "prompt" || param.kind === "function") continue;
    const inner = resolveCalls(binding.op, functions);
    if ("error" in inner) return { error: `input '${name}': ${inner.error}` };
    input[name] = { ...param, binding: { op: inner.op } };
  }
  const out = { ...op, input } as Operation<InlineFamily>;
  if (out.kind !== "function") return { op: out };
  const entry = functions.get(out.functionRef);
  if (!entry) return { error: `no function '${out.functionRef}' is registered` };
  return { op: withCall(out, entry) };
}

/**
 * True when this node needs nothing from the execution environment — a `pure` entry's impl takes no
 * `ctx` at all (`runFunction`), which is that property stated as data rather than inferred from a name.
 *
 * Drawing the fast path on the CAPABILITY rather than on a set of built-in names is the point: a
 * user's registered pure function gets the same treatment as `eq`, which is what EXPRESSIONS §2 claims
 * and what a hardcoded operator set cannot deliver.
 */
export function isPureNode(op: Operation<InlineFamily>): boolean {
  return isResolvedCall(op) && op.call.kind === "pure";
}

/** Whether a whole tree is resolved and pure — the precondition for evaluating it without any async. */
export function isPureTree(op: Operation<InlineFamily>): boolean {
  if (!isPureNode(op)) return false;
  return Object.values(op.input).every((p) => {
    const b = p.binding;
    if (b === undefined || !("op" in b) || typeof b.op === "string") return true;
    if (p.kind === "prompt" || p.kind === "function") return true; // higher-order: a value, not a callee
    return isPureTree(b.op);
  });
}

/**
 * Evaluate a wholly-pure resolved tree with NO async machinery: no handle, no `AbortController`, no
 * metrics frame, no microtask. This is the path a guard re-evaluated every round takes.
 *
 * Returns `undefined` when the tree is not eligible — the caller then dispatches normally. Nothing has
 * been run when that happens, so there is nothing to unwind.
 *
 * A pure entry needs no cancellation surface by construction: it takes no ctx, so it cannot observe an
 * abort signal, and the built-in set is TOTAL and bounded (`div(1,0)` and `at(xs,99)` have answers,
 * `range` is capped) precisely so a synchronous resolver cannot hang or throw. A pure impl that
 * returns a promise is not eligible either — it is doing something a synchronous path cannot wait for.
 */
export function tryRunPureSync(op: Operation<InlineFamily>): ExecResult<ResolvedValue, ExecMetrics> | undefined {
  if (!isPureTree(op)) return undefined;
  return runPureSync(op);
}

function runPureSync(op: Operation<InlineFamily>): ExecResult<ResolvedValue, ExecMetrics> | undefined {
  // Children first, then literals — the same order dispatch uses, and necessary rather than stylistic:
  // `resolveLiteralInputs` refuses a producer edge on a data kind, so a nested edge has to become a
  // literal before that walk sees it.
  const input: Record<string, Parameter<InlineFamily>> = { ...op.input };
  for (const [name, param] of Object.entries(op.input)) {
    const b = param.binding;
    if (b === undefined || !("op" in b) || typeof b.op === "string") continue;
    if (param.kind === "prompt" || param.kind === "function") continue;
    const inner = runPureSync(b.op);
    if (inner === undefined) return undefined; // a nested node turned out ineligible mid-walk
    // A nested failure travels WHOLE, keeping its classification — the same rule dispatch follows, so
    // a tree evaluated fast and a tree evaluated slow report a failure identically.
    if (!isOk(inner)) return inner;
    input[name] = { ...param, binding: { json: inner.value as JsonValue } };
  }
  const literal = resolveLiteralInputs({ ...op, input } as Operation<InlineFamily>);
  if ("error" in literal) return { error: { classification: "permanent", reason: literal.error }, metrics: { durationMs: 0 } };
  const inputs: FunctionInputs = { ...literal.values };
  if (!isResolvedCall(op)) return undefined;
  const entry = op.call;
  if (entry.kind !== "pure") return undefined;
  let produced;
  try {
    produced = entry.impl(inputs);
  } catch (e) {
    // `runFunction` never throws; neither does this. An impl that throws instead of resolving a
    // failure is classified here rather than escaping into the scheduling loop.
    return { error: failureOf(e), metrics: { durationMs: 0 } };
  }
  if (produced !== null && typeof produced === "object" && "then" in produced) return undefined; // async: not this path
  return isOk(produced)
    ? { value: produced.value, metrics: { durationMs: 0 } }
    : { error: produced.error, metrics: { durationMs: 0 } };
}
