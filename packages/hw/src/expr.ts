/**
 * The transition-expression language (SPEC §6). A deliberately tiny, pure language with
 * JavaScript evaluation semantics — equality, comparison, and truthiness behave exactly
 * as in JavaScript — implemented as a hand-written lexer + Pratt parser + tree-walking
 * evaluator so the "pure and limited" guarantee is enforced by the grammar itself:
 * no calls, no indexing, no mutation, no loops, no imports.
 *
 * Two deviations from JS, both from the spec:
 *  - Property access on `undefined`/missing (and, for totality, `null`) yields
 *    `undefined` instead of throwing — implicit optional chaining, so
 *    `children.x.outputs.y` is safely `undefined` when `x` has never started.
 *  - PENDING propagation (SPEC §6/§10.4): a reference to a child that has started but
 *    not finished resolves to the PENDING sentinel; any operator or property access
 *    touching PENDING yields PENDING. Short-circuit operators only short-circuit on
 *    determinate values (`false && PENDING` is `false`; `PENDING && x` is PENDING).
 *    A transition whose `when` evaluates to PENDING is skipped for the round; input
 *    wiring that evaluates to PENDING waits.
 */

import { BUILTIN_PARAMS } from "./builtins";
import { RESOLVER_REFS } from "./format";

/** The pending sentinel — placed in the evaluation context at unresolved async-child
 *  output nodes, and propagated through every operator that touches it. */
export const PENDING: unique symbol = Symbol("ai-exec/hw pending");
export type Pending = typeof PENDING;

export function isPending(v: unknown): v is Pending {
  return v === PENDING;
}

// --- AST ---------------------------------------------------------------------

export type Expr =
  | { type: "lit"; value: string | number | boolean | null }
  /**
   * THIS STATE — the root a leading `.` names, and the only route to runtime data.
   *
   * `.inputs.issue` is `member(member(self, "inputs"), "issue")`. The dot is what separates the two
   * things a name can mean, uniformly in every position (REFERENCES.md §5): a leading dot reads a
   * property of the current state, and a BARE name is resolved along the search `path` to a
   * document. There is no third rule for expressions, which is why `add(.inputs.n, 1)` and
   * `"binding": ".inputs.n"` say the same thing about `.inputs.n`.
   *
   * `self` never stands alone — `pathOf` returns `undefined` for a chain rooted here, so runtime
   * data can reach neither callee position nor a `/` path segment. EXPRESSIONS.md §10's open
   * question about the callee sandbox is answered structurally rather than by a rule to remember.
   */
  | { type: "self" }
  /**
   * A BARE name — resolved along the search `path`, never against this instance's data.
   *
   * This is the half of the grammar that used to read the expression context: `inputs.issue` meant
   * the `inputs` namespace. It now means a document called `inputs`, and the runtime read is spelled
   * `.inputs.issue`. One rule for a bare name in every position is worth the migration; two rules
   * that differ by position is what made `{ expr }` need its own explanation.
   */
  | { type: "ident"; name: string }
  | { type: "member"; obj: Expr; prop: string }
  /**
   * APPLYING a named operation to arguments — the only compute node.
   *
   * `!x`, `a === b`, `a && b`, `a ? b : c` and `classify(x)` are all this: an operation name plus
   * ordered arguments. There were five node types for it, which was a taxonomy over syntax rather
   * than over meaning — every one of them lowers to the same `FunctionOp` with its arguments bound,
   * so the only thing that ever differed was which name went in `functionRef`.
   *
   * `op` is that name. A built-in operator is one whose name resolves in `RESOLVER_REFS`; anything
   * else is a REFERENCE, resolved along the `path` to an operation document (EXPRESSIONS.md §3).
   * Which is §2's stated aim reached — "a user-defined pure function is indistinguishable from
   * `eq`" — since the two differ only in where the name resolves.
   *
   * Note what `op` is NOT: a sub-expression. An operation is *named*, so `(a ? f : g)(x)` has
   * nothing to name; and the name must stay out of the data scope, or `classify(x)` would read as a
   * reference to an undeclared namespace called `classify`.
   */
  | { type: "apply"; op: string; args: Expr[] };

export class ExprError extends Error {
  constructor(
    message: string,
    readonly position: number,
  ) {
    super(`${message} (at ${position})`);
    this.name = "ExprError";
  }
}

/** The operation each comparison SYNTAX names. Syntax is sugar; the operation is the meaning. */
const BINARY_OPS: Readonly<Record<string, string>> = {
  "==": RESOLVER_REFS.eq,
  "!=": RESOLVER_REFS.ne,
  "===": RESOLVER_REFS.strictEq,
  "!==": RESOLVER_REFS.strictNe,
  "<": RESOLVER_REFS.lt,
  "<=": RESOLVER_REFS.le,
  ">": RESOLVER_REFS.gt,
  ">=": RESOLVER_REFS.ge,
};

/**
 * The ordered parameter names each built-in operation binds its arguments to.
 *
 * This IS the positional-to-named mapping (§3.3), and it is deliberately the same mechanism a
 * user-defined callee will use through its declared `index`: an operator is not special, it just
 * has a signature that ships with the language.
 */
export const OPERATOR_PARAMS: Readonly<Record<string, readonly string[]>> = {
  ...BUILTIN_PARAMS,
  // The higher-order operations (EXPRESSIONS.md §3.5): an array, and the operation to apply to each
  // element. The `op` position takes an operation REFERENCE, not a data path — see `lowerExpression`.
  map: ["value", "op"],
  filter: ["value", "op"],
  flatMap: ["value", "op"],
  reduce: ["value", "op", "initial"],
  [RESOLVER_REFS.not]: ["value"],
  [RESOLVER_REFS.eq]: ["left", "right"],
  [RESOLVER_REFS.ne]: ["left", "right"],
  [RESOLVER_REFS.strictEq]: ["left", "right"],
  [RESOLVER_REFS.strictNe]: ["left", "right"],
  [RESOLVER_REFS.lt]: ["left", "right"],
  [RESOLVER_REFS.le]: ["left", "right"],
  [RESOLVER_REFS.gt]: ["left", "right"],
  [RESOLVER_REFS.ge]: ["left", "right"],
  [RESOLVER_REFS.and]: ["left", "right"],
  [RESOLVER_REFS.or]: ["left", "right"],
  [RESOLVER_REFS.cond]: ["test", "then", "else"],
  [RESOLVER_REFS.member]: ["value", "prop"],
  [RESOLVER_REFS.context]: ["name"],
};

// --- Lexer -------------------------------------------------------------------

type Token =
  | { kind: "num"; value: number; pos: number }
  | { kind: "str"; value: string; pos: number }
  | { kind: "ident"; value: string; pos: number }
  | { kind: "punct"; value: string; pos: number }
  | { kind: "eof"; pos: number };

const PUNCT = ["===", "!==", "==", "!=", "<=", ">=", "&&", "||", "<", ">", "!", "?", ":", "(", ")", ".", ",", "/"];
const IDENT_START = /[A-Za-z_$]/;
const IDENT_PART = /[A-Za-z0-9_$]/;

function lex(src: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  outer: while (i < src.length) {
    const c = src[i]!;
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      i++;
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c;
      const pos = i;
      i++;
      let s = "";
      while (i < src.length && src[i] !== quote) {
        if (src[i] === "\\") {
          const esc = src[i + 1];
          if (esc === undefined) throw new ExprError("unterminated escape", i);
          s += esc === "n" ? "\n" : esc === "t" ? "\t" : esc === "r" ? "\r" : esc;
          i += 2;
        } else {
          s += src[i]!;
          i++;
        }
      }
      if (i >= src.length) throw new ExprError("unterminated string", pos);
      i++; // closing quote
      out.push({ kind: "str", value: s, pos });
      continue;
    }
    // A leading `-` is part of the NUMBER, not an operator: this language has no arithmetic syntax,
    // so `-` can only ever be a sign. Without it a negative literal had no spelling at all — `at(xs,
    // -1)`, the natural way to index from the end, was a lex error.
    const negative = c === "-" && src[i + 1] !== undefined && src[i + 1]! >= "0" && src[i + 1]! <= "9";
    if (negative || (c >= "0" && c <= "9")) {
      const pos = i;
      let j = negative ? i + 1 : i;
      while (j < src.length && src[j]! >= "0" && src[j]! <= "9") j++;
      if (src[j] === "." && src[j + 1] !== undefined && src[j + 1]! >= "0" && src[j + 1]! <= "9") {
        j++;
        while (j < src.length && src[j]! >= "0" && src[j]! <= "9") j++;
      }
      if (src[j] === "e" || src[j] === "E") {
        let k = j + 1;
        if (src[k] === "+" || src[k] === "-") k++;
        if (src[k] !== undefined && src[k]! >= "0" && src[k]! <= "9") {
          k++;
          while (k < src.length && src[k]! >= "0" && src[k]! <= "9") k++;
          j = k;
        }
      }
      out.push({ kind: "num", value: Number(src.slice(i, j)), pos });
      i = j;
      continue;
    }
    if (IDENT_START.test(c)) {
      const pos = i;
      let j = i + 1;
      while (j < src.length && IDENT_PART.test(src[j]!)) j++;
      out.push({ kind: "ident", value: src.slice(i, j), pos });
      i = j;
      continue;
    }
    for (const p of PUNCT) {
      if (src.startsWith(p, i)) {
        out.push({ kind: "punct", value: p, pos: i });
        i += p.length;
        continue outer;
      }
    }
    throw new ExprError(`unexpected character '${c}'`, i);
  }
  out.push({ kind: "eof", pos: src.length });
  return out;
}

// --- Parser (Pratt / precedence climbing) ------------------------------------

class Parser {
  private i = 0;
  constructor(private readonly tokens: Token[]) {}

  private peek(): Token {
    return this.tokens[this.i]!;
  }
  private next(): Token {
    return this.tokens[this.i++]!;
  }
  private expectPunct(p: string): void {
    const t = this.next();
    if (t.kind !== "punct" || t.value !== p) throw new ExprError(`expected '${p}'`, t.pos);
  }
  private atPunct(p: string): boolean {
    const t = this.peek();
    return t.kind === "punct" && t.value === p;
  }

  parse(): Expr {
    const e = this.ternary();
    const t = this.peek();
    if (t.kind !== "eof") throw new ExprError("unexpected trailing input", t.pos);
    return e;
  }

  /** Lowest precedence: `?:` (right-associative). */
  private ternary(): Expr {
    const test = this.or();
    if (!this.atPunct("?")) return test;
    this.next();
    const cons = this.ternary();
    this.expectPunct(":");
    const alt = this.ternary();
    return { type: "apply", op: RESOLVER_REFS.cond, args: [test, cons, alt] };
  }

  private or(): Expr {
    let left = this.and();
    while (this.atPunct("||")) {
      this.next();
      left = { type: "apply", op: RESOLVER_REFS.or, args: [left, this.and()] };
    }
    return left;
  }

  private and(): Expr {
    let left = this.equality();
    while (this.atPunct("&&")) {
      this.next();
      left = { type: "apply", op: RESOLVER_REFS.and, args: [left, this.equality()] };
    }
    return left;
  }

  private equality(): Expr {
    let left = this.relational();
    for (;;) {
      const t = this.peek();
      if (t.kind === "punct" && (t.value === "==" || t.value === "!=" || t.value === "===" || t.value === "!==")) {
        this.next();
        left = { type: "apply", op: BINARY_OPS[t.value]!, args: [left, this.relational()] };
      } else return left;
    }
  }

  private relational(): Expr {
    let left = this.unary();
    for (;;) {
      const t = this.peek();
      if (t.kind === "punct" && (t.value === "<" || t.value === "<=" || t.value === ">" || t.value === ">=")) {
        this.next();
        left = { type: "apply", op: BINARY_OPS[t.value]!, args: [left, this.unary()] };
      } else return left;
    }
  }

  private unary(): Expr {
    if (this.atPunct("!")) {
      const t = this.next();
      void t;
      return { type: "apply", op: RESOLVER_REFS.not, args: [this.unary()] };
    }
    return this.member();
  }

  /**
   * Property access and application, left to right.
   *
   * A `(` turns the DOTTED PATH accumulated so far into a callee rather than applying an arbitrary
   * expression: an operation is named by a reference (§3), so `classify(x)` and `lib.review(x)` are
   * calls while `(a ? f : g)(x)` is not a thing this language has. Keeping application to a path is
   * also what keeps a callee out of the data scope — see the `call` node.
   */
  private member(): Expr {
    let e = this.primary();
    // A callee may be a full REFERENCE — `$JAIRA/prompts/review(…)`, `$/functions/classify(…)` — so
    // once a `/` appears the accumulated name is built as reference TEXT rather than as member
    // access. `/` is unambiguous here because the language has no arithmetic: it cannot be division.
    let reference: string | undefined;
    for (;;) {
      if (this.atPunct(".")) {
        this.next();
        const t = this.next();
        if (t.kind !== "ident") throw new ExprError("expected property name after '.'", t.pos);
        if (reference !== undefined) reference += `.${t.value}`;
        else e = { type: "member", obj: e, prop: t.value };
        continue;
      }
      if (this.atPunct("/")) {
        const at = this.peek().pos;
        const base = reference ?? pathOf(e)?.join(".");
        if (base === undefined) throw new ExprError("'/' is only meaningful inside an operation reference", at);
        this.next();
        const t = this.next();
        if (t.kind !== "ident") throw new ExprError("expected a path segment after '/'", t.pos);
        reference = `${base}/${t.value}`;
        continue;
      }
      if (this.atPunct("(")) {
        const at = this.peek().pos;
        const callee = reference ?? pathOf(e)?.join(".");
        if (callee === undefined) throw new ExprError("only a name may be called", at);
        this.next();
        e = { type: "apply", op: callee, args: this.args() };
        reference = undefined;
        continue;
      }
      if (reference !== undefined) {
        // A rooted path is a reference to an OPERATION, and the only thing this language does with
        // one is call it — reading data uses the leading-dot runtime form, not a file path.
        throw new ExprError(`'${reference}' is an operation reference; it is only meaningful called`, this.peek().pos);
      }
      return e;
    }
  }

  /** A call's argument list, already past the `(`. */
  private args(): Expr[] {
    const out: Expr[] = [];
    if (this.atPunct(")")) {
      this.next();
      return out;
    }
    for (;;) {
      out.push(this.ternary());
      if (this.atPunct(",")) {
        this.next();
        continue;
      }
      this.expectPunct(")");
      return out;
    }
  }

  private primary(): Expr {
    // A leading `.` roots the chain at THIS STATE. The dot is deliberately NOT consumed here: the
    // member loop reads it as the first property access, so `.inputs.n` and `x.inputs.n` take one
    // code path and cannot drift. A lone `.`, or `.5`, therefore fails in that loop with
    // "expected property name after '.'", which is the accurate complaint.
    if (this.atPunct(".")) return { type: "self" };
    const t = this.next();
    if (t.kind === "num") return { type: "lit", value: t.value };
    if (t.kind === "str") return { type: "lit", value: t.value };
    if (t.kind === "ident") {
      if (t.value === "true") return { type: "lit", value: true };
      if (t.value === "false") return { type: "lit", value: false };
      if (t.value === "null") return { type: "lit", value: null };
      return { type: "ident", name: t.value };
    }
    if (t.kind === "punct" && t.value === "(") {
      const e = this.ternary();
      this.expectPunct(")");
      return e;
    }
    throw new ExprError("unexpected token", t.pos);
  }
}

/**
 * The RUNTIME path a self-rooted chain reads — `.inputs.n` → `["inputs","n"]`, `undefined` when the
 * expression is not rooted at this state.
 *
 * The counterpart of {@link pathOf}, and the two are deliberately disjoint: exactly one of them
 * answers for any given chain, because the dot is what decides whether a path names instance data or
 * a document. Callers that used to get a runtime path out of `pathOf` want this one.
 */
export function selfPathOf(expr: Expr): string[] | undefined {
  if (expr.type === "self") return [];
  if (expr.type === "member") {
    const base = selfPathOf(expr.obj);
    return base === undefined ? undefined : [...base, expr.prop];
  }
  return undefined;
}

/** The dotted path an expression names, or `undefined` when it is a computation rather than a name. */
export function pathOf(expr: Expr): string[] | undefined {
  if (expr.type === "ident") return [expr.name];
  if (expr.type === "member") {
    const base = pathOf(expr.obj);
    return base ? [...base, expr.prop] : undefined;
  }
  return undefined;
}

/** Parse an expression source string to an AST. Throws `ExprError` on invalid input. */
export function parseExpression(src: string): Expr {
  return new Parser(lex(src)).parse();
}

// --- Operator semantics ------------------------------------------------------
//
// Extracted so there is ONE definition of what each operator means. The tree-walking evaluator below
// is one caller; the operator RESOLVERS (`resolve.ts`, EXPRESSIONS.md §2) are the other, and an
// expression lowered to a producer tree has to mean exactly what the interpreter meant. Two
// implementations of "what does `.prop` reach" is precisely how the evaluator came to disagree with
// its own type-checker about prototype properties.

export type BinaryOp = "==" | "!=" | "===" | "!==" | "<" | "<=" | ">" | ">=";

/** Apply a binary operator to two DETERMINATE values (PENDING is the caller's to short-circuit). */
export function applyBinary(op: BinaryOp, l: unknown, r: unknown): boolean {
  switch (op) {
    case "==":
      // eslint-disable-next-line eqeqeq
      return l == r;
    case "!=":
      // eslint-disable-next-line eqeqeq
      return l != r;
    case "===":
      return l === r;
    case "!==":
      return l !== r;
    case "<":
      return (l as never) < (r as never);
    case "<=":
      return (l as never) <= (r as never);
    case ">":
      return (l as never) > (r as never);
    case ">=":
      return (l as never) >= (r as never);
  }
}

/**
 * Read a property off a value, with this language's two departures from a native lookup:
 *
 *  - property access on `undefined`/`null` yields `undefined` rather than throwing (implicit
 *    optional chaining, so `children.x.outputs.y` is safe before `x` has started);
 *  - `.length` is the ONLY property a string or an array exposes, and objects expose OWN properties
 *    only — a native lookup reaches `constructor`, `__proto__` and every prototype method, which
 *    puts a FUNCTION into a dataflow that is JSON all the way down.
 */
export function memberOf(obj: unknown, prop: string): unknown {
  if (obj === undefined || obj === null) return undefined;
  if (typeof obj === "string" || Array.isArray(obj)) return prop === "length" ? obj.length : undefined;
  if (typeof obj === "object") {
    return Object.hasOwn(obj as object, prop) ? (obj as Record<string, unknown>)[prop] : undefined;
  }
  // Primitives (number/boolean): no useful properties in this language.
  return undefined;
}

// --- Evaluator ---------------------------------------------------------------

export type ExprValue = unknown; // may be PENDING

/**
 * Evaluate an AST against a read-only context object. The context is a plain object
 * graph; identifiers resolve to its top-level properties (missing → `undefined`).
 * PENDING sentinels anywhere in the graph propagate per the module header.
 */
export function evaluate(expr: Expr, context: Record<string, unknown>): ExprValue {
  switch (expr.type) {
    case "lit":
      return expr.value;
    case "self":
      // The context IS this state's runtime data, so the self root is the context object and
      // `.inputs.n` walks into it by ordinary member access.
      return context;
    case "ident":
      // A bare name is a document on the search `path`, and resolving one needs a filesystem and a
      // referring state — the loader's knowledge, not a walk over a context. Refused for the same
      // reason `applyOperator` refuses a non-built-in: this is the reference semantics the lowering
      // is checked against, not a second execution path.
      throw new ExprError(`'${expr.name}' is a reference, which only the lowered form can resolve`, 0);
    case "member": {
      const obj = evaluate(expr.obj, context);
      if (isPending(obj)) return PENDING;
      return memberOf(obj, expr.prop);
    }
    case "apply":
      return applyOperator(expr, context);
  }
}

/**
 * Interpret ONE application, for the built-in operations only.
 *
 * `evaluate` is the reference semantics the lowering is checked against (EXPRESSIONS.md §1.1), not a
 * second execution path — so an operation that is not a built-in is refused rather than
 * half-implemented. Running one needs a resolved callee and a scope, which a pure walk over a
 * context does not have.
 */
function applyOperator(expr: Expr & { type: "apply" }, context: Record<string, unknown>): ExprValue {
  const arg = (i: number): ExprValue => evaluate(expr.args[i]!, context);
  switch (expr.op) {
    case RESOLVER_REFS.not: {
      const v = arg(0);
      return isPending(v) ? PENDING : !v;
    }
    case RESOLVER_REFS.and: {
      // Determinate-falsy short-circuit even past a pending right side.
      const l = arg(0);
      if (isPending(l)) return PENDING;
      return l ? arg(1) : l;
    }
    case RESOLVER_REFS.or: {
      const l = arg(0);
      if (isPending(l)) return PENDING;
      return l ? l : arg(1);
    }
    case RESOLVER_REFS.cond: {
      const t = arg(0);
      if (isPending(t)) return PENDING;
      return t ? arg(1) : arg(2);
    }
    default: {
      const op = BINARY_FOR_NAME[expr.op];
      if (op === undefined) {
        throw new ExprError(`'${expr.op}' is an operation, which only the lowered form can run`, 0);
      }
      const l = arg(0);
      if (isPending(l)) return PENDING;
      const r = arg(1);
      if (isPending(r)) return PENDING;
      return applyBinary(op, l, r);
    }
  }
}

/** The comparison each built-in comparison NAME performs — the inverse of `BINARY_OPS`. */
const BINARY_FOR_NAME: Readonly<Record<string, BinaryOp>> = Object.fromEntries(
  Object.entries(BINARY_OPS).map(([syntax, name]) => [name, syntax as BinaryOp]),
);

/** Parse + evaluate in one step. */
export function evaluateExpression(src: string, context: Record<string, unknown>): ExprValue {
  return evaluate(parseExpression(src), context);
}

// --- Static analysis ---------------------------------------------------------

/**
 * Every root-anchored reference path in the expression, e.g.
 * `children.critique.outputs.result === 'clean' && run.iteration < 3` →
 * [["children","critique","outputs","outcome"], ["run","iteration"]].
 * Used by the workflow validator's static reference checks.
 */
export function referencesOf(expr: Expr): string[][] {
  const out: string[][] = [];
  const walk = (e: Expr): string[] | undefined => {
    switch (e.type) {
      case "self":
        // The root of every runtime path, contributing no segment of its own: `.inputs.n` is the
        // reference `["inputs","n"]`, exactly as `inputs.n` used to be.
        return [];
      case "ident":
        // A bare name reads no instance data — it is resolved along the `path` at load. Reporting it
        // would make `classify` look like a read of an undeclared namespace, which is the same
        // mistake the `apply` case below avoids for a callee.
        return undefined;
      case "member": {
        const base = walk(e.obj);
        if (base) return [...base, e.prop];
        return undefined;
      }
      case "lit":
        return undefined;
      case "apply":
        // The ARGUMENTS read data; the OPERATION NAME does not — it is resolved along the path, not
        // against this instance. Reporting it here would make `classify(x)` look like a read of an
        // undeclared namespace called `classify`.
        for (const arg of e.args) collect(arg);
        return undefined;
    }
  };
  const collect = (e: Expr): void => {
    const path = walk(e);
    if (path) out.push(path);
  };
  collect(expr);
  return out;
}
