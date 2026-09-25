/**
 * The LOAD-TIME evaluator — what a `$ref` expression runs on (SPEC §5.4).
 *
 * A `$ref` that is not a path spelling is an expression over DOCUMENTS: nothing has run yet, so there
 * is no instance, no scope and no call to make — only files, and plain values read out of them. That
 * is a much smaller world than the run-time one, and this evaluator is exactly that world:
 *
 *  - a REFERENCE spelling (`$BASE/workflows/system/events.transitions`, `lib/rules`, `$/x.y`) is the
 *    value it names, read through the same resolver a path `$ref` goes through;
 *  - a LAMBDA (`(t) => …`, `t => …`) is a value, applied by `filter`, `map`, `flatMap`, `find`,
 *    `some`, `every` and `reduce` — and a parameter is read by its bare name, a dotted read off it
 *    being a property (`t.name`), so a parameter shadows a document of the same name;
 *  - every BUILT-IN is available, with the operators, `?:`, the three lazy forms and `coalesce`
 *    meaning what they mean at run time (`builtins.ts`, `expr.ts`), and `recv.name(args)` is
 *    `name(recv, args)` as it is there (NAMES.md §8) — `['a'].includes(t.name)`;
 *  - there is no `.` root: a leading-dot read addresses an INSTANCE's data, and at load there is no
 *    instance to read. It is refused, naming the fix.
 *
 * Kept apart from `evaluate` (expr.ts) on purpose. That one is the reference semantics of the
 * LOWERED form and refuses a reference and a call through a value, because a run resolves those
 * elsewhere; here both are the whole point. What the two share is the meaning of every operator,
 * which comes from the same `BUILTINS` table, `memberOf` and `applyBinary`.
 */
import { BUILTINS } from "./builtins.js";
import { applyBinary, canonicalOperation, ExprError, isSpread, memberOf, parseExpression, pathOf, type Argument, type BinaryOp, type Expr } from "./expr.js";
import { RESOLVER_REFS } from "./format.js";

/** A lambda as a value: its parameters, its body, and the parameters bound where it was written. */
interface Closure {
  readonly lambda: Expr & { type: "lambda" };
  readonly bound: ReadonlyMap<string, unknown>;
}

const CLOSURE = Symbol("hw load-time lambda");
type ClosureValue = { readonly [CLOSURE]: Closure };

function closureOf(v: unknown): Closure | undefined {
  return v !== null && typeof v === "object" && CLOSURE in v ? (v as ClosureValue)[CLOSURE] : undefined;
}

export interface LoadExpressionOptions {
  /**
   * The value a REFERENCE spelling names — a bare dotted path or a `$`-rooted one, exactly as written.
   * Throws what resolution throws when it names nothing.
   */
  reference: (spelling: string) => unknown;
}

/** Operations whose second argument is a function applied per element. */
const HIGHER_ORDER = new Set(["filter", "map", "flatMap", "find", "some", "every", "reduce"]);

const BINARY: Readonly<Record<string, BinaryOp>> = {
  [RESOLVER_REFS.eq]: "==",
  [RESOLVER_REFS.ne]: "!=",
  [RESOLVER_REFS.strictEq]: "===",
  [RESOLVER_REFS.strictNe]: "!==",
  [RESOLVER_REFS.lt]: "<",
  [RESOLVER_REFS.le]: "<=",
  [RESOLVER_REFS.gt]: ">",
  [RESOLVER_REFS.ge]: ">=",
};

/** Parse and evaluate one load-time expression. Throws `ExprError` on anything it cannot answer. */
export function evaluateLoadExpression(source: string, options: LoadExpressionOptions): unknown {
  return evaluateAt(parseExpression(source), new Map(), options);
}

/** Evaluate a parsed expression with these lambda parameters bound. Exported for the tests. */
export function evaluateAt(expr: Expr, bound: ReadonlyMap<string, unknown>, options: LoadExpressionOptions): unknown {
  const ev = (e: Expr): unknown => evaluateAt(e, bound, options);
  switch (expr.type) {
    case "lit":
      return expr.value;
    case "self":
      throw new ExprError("a leading '.' reads an instance's data, and a load-time expression has no instance — name the document instead ($/…, $BASE/…, or a bare path)", 0);
    case "ident":
      return bound.has(expr.name) ? bound.get(expr.name) : options.reference(expr.name);
    case "member": {
      // A dotted path rooted at a PARAMETER is property reads off its value; one rooted anywhere else
      // is ONE reference, whose file half the resolver decides — `lib.rules.push` is not `lib`, then
      // `.rules`, then `.push`.
      const path = pathOf(expr);
      if (path !== undefined && !bound.has(path[0]!)) return options.reference(path.join("."));
      return memberOf(ev(expr.obj), expr.prop);
    }
    case "object": {
      const out: Record<string, unknown> = {};
      for (const entry of expr.entries) {
        Object.defineProperty(out, entry.key, { value: ev(entry.value), writable: true, enumerable: true, configurable: true });
      }
      return out;
    }
    case "array":
      return expr.items.map(ev);
    case "lambda":
      return { [CLOSURE]: { lambda: expr, bound } } satisfies ClosureValue;
    case "call": {
      // `recv.name(args)` — the receiver form of an operation (NAMES.md §8) — or a lambda applied.
      const callee = expr.callee;
      if (callee.type === "member") {
        const recv = ev(callee.obj);
        return applyNamed(callee.prop, [() => recv, ...thunks(expr.args, ev, callee.prop)], options);
      }
      const fn = closureOf(ev(callee));
      if (fn === undefined) throw new ExprError("only a lambda can be called through a value at load", 0);
      return applyClosure(fn, thunks(expr.args, ev, "a lambda").map((t) => t()), options);
    }
    case "apply": {
      // A DOTTED callee rooted at a parameter or a reference is a receiver call the parser could not
      // tell apart: `t.name.startsWith('x')` names `t.name.startsWith`. Split at the last dot.
      const dot = expr.op.lastIndexOf(".");
      if (dot > 0 && !isLoadOperation(expr.op)) {
        const head = expr.op.slice(0, dot);
        const name = expr.op.slice(dot + 1);
        const [root, ...rest] = head.split(".");
        const recv = bound.has(root!) ? rest.reduce<unknown>((v, p) => memberOf(v, p), bound.get(root!)) : options.reference(head);
        return applyNamed(name, [() => recv, ...thunks(expr.args, ev, name)], options);
      }
      return applyNamed(expr.op, thunks(expr.args, ev, expr.op), options);
    }
  }
}

/** Is this a name the load-time evaluator can run — an operator, a built-in, a higher-order op? */
function isLoadOperation(name: string): boolean {
  const op = canonicalOperation(name);
  return BUILTINS[op] !== undefined || HIGHER_ORDER.has(op) || BINARY[op] !== undefined || LAZY.has(op);
}

const LAZY: ReadonlySet<string> = new Set([RESOLVER_REFS.not, RESOLVER_REFS.and, RESOLVER_REFS.or, RESOLVER_REFS.cond]);

/**
 * An argument list as THUNKS, so the lazy forms stay lazy. A spread has no meaning without a
 * signature to bind it against, which a load-time expression does not consult: refused.
 */
function thunks(args: readonly Argument[], ev: (e: Expr) => unknown, callee: string): Array<() => unknown> {
  return args.map((a) => {
    if (isSpread(a)) throw new ExprError(`'${callee}' is called with a spread, which a load-time expression does not bind — pass arguments by position`, 0);
    return () => ev(a);
  });
}

/** Apply an operation by NAME to its arguments, each evaluated only when the operation reads it. */
function applyNamed(authored: string, args: ReadonlyArray<() => unknown>, options: LoadExpressionOptions): unknown {
  const op = canonicalOperation(authored);
  const value = (i: number): unknown => args[i]?.();
  switch (op) {
    case RESOLVER_REFS.not:
      return !value(0);
    case RESOLVER_REFS.and: {
      const l = value(0);
      return l ? value(1) : l;
    }
    case RESOLVER_REFS.or: {
      const l = value(0);
      return l ? l : value(1);
    }
    case RESOLVER_REFS.cond:
      return value(0) ? value(1) : value(2);
    case "coalesce": {
      const a = value(0);
      return a !== null && a !== undefined ? a : value(1);
    }
  }
  if (HIGHER_ORDER.has(op)) return higherOrder(op, authored, value(0), value(1), args.length > 2 ? value(2) : undefined, options);
  const binary = BINARY[op];
  if (binary !== undefined) return applyBinary(binary, value(0), value(1));
  const builtin = BUILTINS[op];
  if (builtin === undefined) {
    throw new ExprError(`'${authored}' is not something a load-time expression can run — it runs built-ins, the operators, and filter/map/flatMap/find/some/every/reduce with a lambda`, 0);
  }
  if (args.length > builtin.params.length) {
    throw new ExprError(`'${authored}' takes ${builtin.params.length} argument${builtin.params.length === 1 ? "" : "s"} (${builtin.params.join(", ")}), but ${args.length} were given`, 0);
  }
  const named: Record<string, unknown> = {};
  builtin.params.forEach((p, i) => {
    if (i < args.length) named[p] = value(i);
  });
  return builtin.fn(named);
}

/** `filter(list, λ)` and its family, over a lambda's closure. */
function higherOrder(op: string, authored: string, list: unknown, fn: unknown, initial: unknown, options: LoadExpressionOptions): unknown {
  const closure = closureOf(fn);
  if (closure === undefined) throw new ExprError(`'${authored}' takes a lambda — '(x) => …' — as its second argument`, 0);
  if (!Array.isArray(list)) throw new ExprError(`'${authored}' takes a list as its first argument, and was given ${describe(list)}`, 0);
  const call = (...values: unknown[]): unknown => applyClosure(closure, values, options);
  switch (op) {
    case "filter":
      return list.filter((x, i) => Boolean(call(x, i)));
    case "map":
      return list.map((x, i) => call(x, i));
    case "flatMap":
      return list.flatMap((x, i) => {
        const r = call(x, i);
        return Array.isArray(r) ? r : [r];
      });
    case "find":
      return list.find((x, i) => Boolean(call(x, i)));
    case "some":
      return list.some((x, i) => Boolean(call(x, i)));
    case "every":
      return list.every((x, i) => Boolean(call(x, i)));
    case "reduce":
      return list.reduce<unknown>((acc, x, i) => call(acc, x, i), initial);
  }
  throw new ExprError(`'${authored}' is not a higher-order operation`, 0);
}

/** Bind a lambda's parameters (missing ones are `undefined`, extras ignored — JavaScript's rule). */
function applyClosure(closure: Closure, values: readonly unknown[], options: LoadExpressionOptions): unknown {
  const bound = new Map(closure.bound);
  closure.lambda.params.forEach((p, i) => bound.set(p, values[i]));
  return evaluateAt(closure.lambda.body, bound, options);
}

function describe(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "a list";
  if (closureOf(v)) return "a lambda";
  return typeof v === "object" ? "an object" : `a ${typeof v}`;
}

/** True when a value is (or holds) a lambda — which cannot be spliced into a document. */
export function holdsLambda(v: unknown): boolean {
  if (closureOf(v)) return true;
  if (Array.isArray(v)) return v.some(holdsLambda);
  if (v !== null && typeof v === "object") return Object.values(v).some(holdsLambda);
  return false;
}
