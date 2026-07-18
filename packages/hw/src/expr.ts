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
  | { type: "ident"; name: string }
  | { type: "member"; obj: Expr; prop: string }
  | { type: "unary"; op: "!"; arg: Expr }
  | { type: "binary"; op: "==" | "!=" | "===" | "!==" | "<" | "<=" | ">" | ">="; left: Expr; right: Expr }
  | { type: "logical"; op: "&&" | "||"; left: Expr; right: Expr }
  | { type: "cond"; test: Expr; cons: Expr; alt: Expr };

export class ExprError extends Error {
  constructor(
    message: string,
    readonly position: number,
  ) {
    super(`${message} (at ${position})`);
    this.name = "ExprError";
  }
}

// --- Lexer -------------------------------------------------------------------

type Token =
  | { kind: "num"; value: number; pos: number }
  | { kind: "str"; value: string; pos: number }
  | { kind: "ident"; value: string; pos: number }
  | { kind: "punct"; value: string; pos: number }
  | { kind: "eof"; pos: number };

const PUNCT = ["===", "!==", "==", "!=", "<=", ">=", "&&", "||", "<", ">", "!", "?", ":", "(", ")", "."];
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
    if (c >= "0" && c <= "9") {
      const pos = i;
      let j = i;
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
    return { type: "cond", test, cons, alt };
  }

  private or(): Expr {
    let left = this.and();
    while (this.atPunct("||")) {
      this.next();
      left = { type: "logical", op: "||", left, right: this.and() };
    }
    return left;
  }

  private and(): Expr {
    let left = this.equality();
    while (this.atPunct("&&")) {
      this.next();
      left = { type: "logical", op: "&&", left, right: this.equality() };
    }
    return left;
  }

  private equality(): Expr {
    let left = this.relational();
    for (;;) {
      const t = this.peek();
      if (t.kind === "punct" && (t.value === "==" || t.value === "!=" || t.value === "===" || t.value === "!==")) {
        this.next();
        left = { type: "binary", op: t.value, left, right: this.relational() };
      } else return left;
    }
  }

  private relational(): Expr {
    let left = this.unary();
    for (;;) {
      const t = this.peek();
      if (t.kind === "punct" && (t.value === "<" || t.value === "<=" || t.value === ">" || t.value === ">=")) {
        this.next();
        left = { type: "binary", op: t.value, left, right: this.unary() };
      } else return left;
    }
  }

  private unary(): Expr {
    if (this.atPunct("!")) {
      const t = this.next();
      void t;
      return { type: "unary", op: "!", arg: this.unary() };
    }
    return this.member();
  }

  private member(): Expr {
    let e = this.primary();
    while (this.atPunct(".")) {
      this.next();
      const t = this.next();
      if (t.kind !== "ident") throw new ExprError("expected property name after '.'", t.pos);
      e = { type: "member", obj: e, prop: t.value };
    }
    return e;
  }

  private primary(): Expr {
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

/** Parse an expression source string to an AST. Throws `ExprError` on invalid input. */
export function parseExpression(src: string): Expr {
  return new Parser(lex(src)).parse();
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
    case "ident":
      return context[expr.name];
    case "member": {
      const obj = evaluate(expr.obj, context);
      if (isPending(obj)) return PENDING;
      if (obj === undefined || obj === null) return undefined;
      if (typeof obj === "string" || Array.isArray(obj)) {
        // `.length` is the only meaningful property on these (SPEC §6); other
        // properties fall through to native lookup, which is harmless and pure.
        return (obj as unknown as Record<string, unknown>)[expr.prop];
      }
      if (typeof obj === "object") return (obj as Record<string, unknown>)[expr.prop];
      // Primitives (number/boolean): no useful properties in this language.
      return undefined;
    }
    case "unary": {
      const v = evaluate(expr.arg, context);
      if (isPending(v)) return PENDING;
      return !v;
    }
    case "binary": {
      const l = evaluate(expr.left, context);
      if (isPending(l)) return PENDING;
      const r = evaluate(expr.right, context);
      if (isPending(r)) return PENDING;
      switch (expr.op) {
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
      break;
    }
    case "logical": {
      const l = evaluate(expr.left, context);
      if (expr.op === "&&") {
        // Determinate-falsy short-circuit even past a pending right side.
        if (isPending(l)) return PENDING;
        if (!l) return l;
        return evaluate(expr.right, context);
      }
      if (isPending(l)) return PENDING;
      if (l) return l;
      return evaluate(expr.right, context);
    }
    case "cond": {
      const t = evaluate(expr.test, context);
      if (isPending(t)) return PENDING;
      return t ? evaluate(expr.cons, context) : evaluate(expr.alt, context);
    }
  }
}

/** Parse + evaluate in one step. */
export function evaluateExpression(src: string, context: Record<string, unknown>): ExprValue {
  return evaluate(parseExpression(src), context);
}

// --- Static analysis ---------------------------------------------------------

/**
 * Every root-anchored reference path in the expression, e.g.
 * `children.critique.outputs.outcome === 'clean' && run.iteration < 3` →
 * [["children","critique","outputs","outcome"], ["run","iteration"]].
 * Used by the workflow validator's static reference checks.
 */
export function referencesOf(expr: Expr): string[][] {
  const out: string[][] = [];
  const walk = (e: Expr): string[] | undefined => {
    switch (e.type) {
      case "ident":
        return [e.name];
      case "member": {
        const base = walk(e.obj);
        if (base) return [...base, e.prop];
        return undefined;
      }
      case "lit":
        return undefined;
      case "unary":
        collect(e.arg);
        return undefined;
      case "binary":
      case "logical":
        collect(e.left);
        collect(e.right);
        return undefined;
      case "cond":
        collect(e.test);
        collect(e.cons);
        collect(e.alt);
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
