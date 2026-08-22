/**
 * An EMBEDDED body — a JSON function document whose implementation is js/ts (SPEC §7.5.1).
 *
 * The middle of the three body forms. An `{ "expr" }` document is the closed language of §6 and runs
 * in the interpreter; a module is a file on disk with its own signature and its own approval. An
 * embedded body sits between: it declares its slots the way a state does, and supplies a body that
 * needs a compiler but no separate file — so it stays INLINED into the document, which is what keeps
 * it part of the workflow's snapshot identity and out of §7.5.5's integrity machinery entirely.
 *
 * This module turns that authored body into module source. It does not run it: compiling and loading
 * are the module pipeline's job, and this deliberately produces the same kind of artifact a module
 * form produces, so exactly one execution path exists downstream.
 *
 * ## Statement list or single expression
 *
 * The distinction an arrow function already draws between `x => { return f(x); }` and `x => f(x)`.
 * All three of these are the same function:
 *
 * ```text
 * "return Math.max(0, 1 - 0.35 * severity);"
 * "Math.max(0, 1 - 0.35 * severity);"
 * "Math.max(0, 1 - 0.35 * severity)"
 * ```
 *
 * The rule is the obvious one: a body that parses as a SINGLE EXPRESSION is one, and its value is
 * returned. Anything else is a statement list and must `return` for itself. A trailing semicolon
 * decides nothing, because an expression followed by one still parses as an expression statement.
 *
 * That form is not a keystroke saving. It is what makes the three body forms a progression rather
 * than three syntaxes: an `{ "expr" }` document that outgrows §6 — it wants a loop, or a built-in the
 * language does not have — becomes an embedded body by changing which key it is written under, and
 * the text between the quotes very often does not change at all.
 */
import type * as TS from "typescript";
import { loadCompiler } from "./moduleExports.js";
import { positionalOrder, type ParameterDecl } from "./format.js";

export class FunctionBodyError extends Error {}

/** How a body returns its value. */
export type BodyForm = "expression" | "statements";

/** An authored body, compiled to module source. */
export interface SynthesizedBody {
  /** Module source, in the shape a `.ts` module form would have had. */
  source: string;
  /** The wrapper's parameters, in binding order. */
  parameters: readonly string[];
  form: BodyForm;
}

/**
 * Classify a body without generating anything — the half `validate` wants on its own.
 *
 * Syntax errors are reported HERE rather than at transpile time, because this is the point where the
 * authored text is still identifiable: by the time it is spliced into a wrapper, a reported position
 * is an offset into generated code that the author never wrote.
 */
export async function classifyBody(body: string, where: string): Promise<BodyForm> {
  return classifyBodyWith(await loadCompiler(), body, where);
}

/** {@link classifyBody} with the compiler already in hand — the synchronous core. */
export function classifyBodyWith(ts: typeof TS, body: string, where: string): BodyForm {

  // BEFORE the parser is consulted, because the parser's answer here is worse than useless. A body
  // opening with `{` is a BLOCK, exactly as JavaScript reads it — so `{ score: s }` is a labelled
  // statement and `{ score: s, reasons: r }` is a syntax error at the comma. Reporting "Expression
  // expected at offset 12" for a perfectly ordinary record sends the author looking anywhere but at
  // the brace, which is the one thing they need to change.
  if (body.trimStart().startsWith("{")) {
    throw new FunctionBodyError(
      `${where}: a body beginning with '{' is read as a statement block, as JavaScript reads it — ` +
        `parenthesise it ('({ … })') to return a record, or write 'return { … }'`,
    );
  }

  const file = ts.createSourceFile("<body>.ts", body, ts.ScriptTarget.Latest, /* setParentNodes */ true);

  // `parseDiagnostics` is not on the public `SourceFile`, and reaching for it is deliberate: the
  // alternative is transpiling to find out, which reports positions in generated code and cannot run
  // until a wrapper has already been built around text that may not parse.
  const diagnostics = (file as unknown as { parseDiagnostics?: readonly TS.Diagnostic[] }).parseDiagnostics ?? [];
  const first = diagnostics[0];
  if (first !== undefined) {
    const message = ts.flattenDiagnosticMessageText(first.messageText, " ");
    const at = first.start === undefined ? "" : ` at offset ${first.start}`;
    throw new FunctionBodyError(`${where}: body does not parse${at} — ${message}`);
  }

  const only = file.statements.length === 1 ? file.statements[0] : undefined;
  if (only !== undefined && ts.isExpressionStatement(only)) return "expression";

  if (!returnsSomewhere(ts, file.statements)) {
    throw new FunctionBodyError(
      `${where}: a statement body must 'return' — only a body that is a single expression returns implicitly`,
    );
  }
  return "statements";
}

/**
 * Does this statement list return on any path?
 *
 * Descends through the structures that are still THIS function — blocks, conditionals, loops, `try`,
 * `switch` — and stops at anything that introduces a new one, since a `return` inside a callback
 * returns from the callback. Deliberately asks "is there a return at all" rather than "does every
 * path return": the latter is definite-assignment analysis, and a body whose `if` has no `else` is
 * ordinary rather than wrong.
 */
function returnsSomewhere(ts: typeof TS, nodes: readonly TS.Node[]): boolean {
  for (const node of nodes) {
    if (ts.isReturnStatement(node)) return true;
    if (
      ts.isFunctionDeclaration(node) ||
      ts.isFunctionExpression(node) ||
      ts.isArrowFunction(node) ||
      ts.isClassDeclaration(node) ||
      ts.isClassExpression(node)
    ) {
      continue; // a return in there belongs to it, not to us
    }
    let found = false;
    node.forEachChild((child) => {
      if (!found && returnsSomewhere(ts, [child])) found = true;
    });
    if (found) return true;
  }
  return false;
}

/**
 * Wrap an authored body in the module a compiler can take.
 *
 * The parameters are the declared input slots in {@link positionalOrder} — the SAME order a call
 * binds its positional arguments in, which is why that rule lives in one place. A module form gets
 * this ordering free from its parameter list; an embedded body has no parameter list, so the
 * document's `index` (or key order) is what supplies it.
 */
export async function synthesizeBody(
  name: string,
  body: string,
  input: Readonly<Record<string, ParameterDecl>>,
  where = name,
): Promise<SynthesizedBody> {
  return synthesizeBodyWith(await loadCompiler(), name, body, input, where);
}

/** {@link synthesizeBody} with the compiler already in hand — the synchronous core. */
export function synthesizeBodyWith(
  ts: typeof TS,
  name: string,
  body: string,
  input: Readonly<Record<string, ParameterDecl>>,
  where = name,
): SynthesizedBody {
  const form = classifyBodyWith(ts, body, where);
  const parameters = positionalOrder(input);

  for (const parameter of parameters) {
    if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(parameter)) {
      // A slot name is a JSON key and may be anything; a parameter name may not. Caught here rather
      // than producing source that does not parse for a reason the author cannot see.
      throw new FunctionBodyError(
        `${where}: input '${parameter}' cannot be a parameter name — an embedded body binds its inputs as identifiers`,
      );
    }
  }

  // The expression form is spliced into a `return`, so one execution path exists downstream whichever
  // way it was written. Its own trailing semicolon, if any, is already part of the text and harmless.
  const inner = form === "expression" ? `  return (\n${indent(body, 4)}\n  );` : indent(body, 2);
  const source = `export default function (${parameters.join(", ")}) {\n${inner}\n}\n`;
  return { source, parameters, form };
}

function indent(text: string, by: number): string {
  const pad = " ".repeat(by);
  return text
    .split("\n")
    .map((line) => (line.trim() === "" ? line : pad + line))
    .join("\n");
}
