/**
 * Expression LOWERING (EXPRESSIONS.md §1) — an expression becomes a tree of producer edges over the
 * operator resolvers, rather than a source string handed to an interpreter at resolution time.
 *
 * This is what stops `{ expr }` being the one construct in the format that needs its own static
 * analysis: a reference is a LEAF of the tree, so the fan-out planner and the validator walk it with
 * the same code they already walk every other binding with, instead of re-parsing the source.
 *
 * **Everything lowers through `context.get`, including child reads.** The tempting alternative —
 * lowering `children.c.outputs.x` onto a child producer edge, the way the `{ child }` sugar does —
 * would change what the expression MEANS in two places, and both are silent:
 *
 *  - `children.c` is `{ outputs, outcome }` in the expression context, while a `{ op: "c" }` edge
 *    resolves to the outputs alone;
 *  - for a declared child that has not started, the context yields `undefined` (so
 *    `children.c.outputs.x === 'y'` is simply false), while a child edge REFUSES with "child 'c' has
 *    not run" and fails the whole binding.
 *
 * Preserving the interpreter's semantics exactly is worth more than making the fan-out planner's job
 * structural for free — so `fanout.ts` recognizes a context-rooted `children` chain instead, which
 * is still a walk over the tree rather than a second parse.
 */
import type { InlineFamily, Operation, Parameter, Ref } from "@declarative-ai/exec";
import { ExprError, OPERATOR_PARAMS, pathOf, type Expr } from "./expr.js";

/** The operations whose `op` argument is an operation REFERENCE rather than a data path (§3.5). */
const HIGHER_ORDER_NAMES: ReadonlySet<string> = new Set(["map", "filter", "flatMap", "reduce"]);
import { positionalOrder, RESOLVER_REFS } from "./format.js";

/** A producer edge on one operator resolver. Mirrors the loader's `resolverEdge`. */
function edge(functionRef: string, args: Record<string, Ref<InlineFamily>>): Ref<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, binding] of Object.entries(args)) {
    input[name] = { kind: "text" in binding ? "text" : "json", binding };
  }
  return { op: { kind: "function", functionRef, input, output: { name: "value", kind: "json" } } };
}

/**
 * Lower one parsed expression to the producer edge that computes it.
 *
 * One case per AST node. Every APPLICATION lowers the same way whatever it
 * applies — its name goes in `functionRef`, its arguments bind to that operation's parameters in
 * order — which is why an operator and a call need no separate treatment here: `a === b` and
 * `classify(x)` differ only in the name and in where that name resolves.
 */
export function lowerExpression(expr: Expr, options: LowerOptions = {}): Ref<InlineFamily> {
  const down = (e: Expr): Ref<InlineFamily> => lowerExpression(e, options);
  switch (expr.type) {
    case "lit":
      // A string literal is a `text` leaf so it reads like every other authored string; everything
      // else — number, boolean, null — is a `json` leaf, which is always a leaf whatever it holds.
      return typeof expr.value === "string" ? { text: expr.value } : { json: expr.value };
    case "self":
      // Never lowered on its own: the self root only exists to be walked into. The parser cannot
      // produce a bare one — a lone `.` fails as "expected property name" — so this is a guard on
      // hand-built ASTs rather than a reachable authoring error.
      throw new ExprError("'.' names this state, so it must be followed by a property", 0);
    case "ident":
      // A bare name is a REFERENCE to a document on the search `path`. Resolving one needs the path,
      // the referring state and a filesystem — the loader's knowledge — so it arrives injected,
      // exactly as a callee's does. What comes back may be an operation (the higher-order value of
      // §3.1) or an ordinary value; both are refs by the time they get here.
      return resolveName(expr.name, options);
    case "member": {
      // `.inputs.n` lowers to precisely the tree `inputs.n` used to lower to — `context.get` at the
      // root, `op.member` above it. That is what keeps this change confined to the surface syntax:
      // `fanout.ts`, `pathOfRef`, `referencePathsOf`, `resolve.ts` and the validator's reachability
      // walk all read the lowered form and need no knowledge of the dot.
      if (expr.obj.type === "self") return edge(RESOLVER_REFS.context, { name: { text: expr.prop } });
      // A bare dotted path is ONE name, not a member access on a resolved value: `lib.helpers.trim`
      // is a reference whose file half the resolver decides by longest match (REFERENCES.md §1.1).
      // Lowering it property-by-property would resolve `lib` alone and then project, which is a
      // different question with a different answer.
      const path = pathOf(expr);
      if (path !== undefined) return resolveName(path.join("."), options);
      return edge(RESOLVER_REFS.member, { value: down(expr.obj), prop: { text: expr.prop } });
    }
    case "object":
      // An object literal's KEYS ARE THE PARAMETER NAMES. Every other node maps ordered arguments
      // onto a signature the language or the callee wrote down; this one has no signature, because
      // the author names the slots as they fill them — which is exactly the shape a producer edge's
      // `input` already is, so the literal lowers onto it with nothing in between.
      return edge(
        RESOLVER_REFS.record,
        Object.fromEntries(expr.entries.map((entry) => [entry.key, down(entry.value)])),
      );
    case "apply": {
      const builtin = OPERATOR_PARAMS[expr.op];
      if (builtin !== undefined) {
        // A HIGHER-ORDER operation takes an operation in its `op` position, so that argument is a
        // REFERENCE rather than a data path: `map(xs, classify)` passes the operation `classify`,
        // and lowering it as `context.get("classify")` would read a namespace that does not exist.
        // The model already says how a producer edge with no arguments is used — a `prompt`/
        // `function`-kind parameter receives the DEFINITION — so this needs no new mechanism.
        if (HIGHER_ORDER_NAMES.has(expr.op)) {
          const args: Record<string, Ref<InlineFamily>> = {};
          const array = expr.args[0];
          if (array !== undefined) args.value = down(array);
          const applied = expr.args[1];
          const name = applied !== undefined ? pathOf(applied) : undefined;
          if (name === undefined) {
            throw new ExprError(`'${expr.op}' takes an operation to apply, named — not an expression`, 0);
          }
          const resolved = options.resolveOperation?.(name.join("."));
          if (resolved === undefined) throw new ExprError(`'${name.join(".")}' is not a known operation`, 0);
          args.op = { op: resolved };
          // `reduce` takes a seed as its third argument, an ordinary value.
          const seed = expr.args[2];
          if (seed !== undefined) args.initial = down(seed);
          return edge(expr.op, args);
        }
        return edge(expr.op, bindPositionally(expr.args, builtin, down));
      }

      // Not a built-in: the name is a REFERENCE to an operation document. Resolving it needs the
      // search path, the referring state and a filesystem — the loader's knowledge — so it arrives
      // injected rather than being reached for here.
      const resolved = options.resolveOperation?.(expr.op);
      if (resolved === undefined) throw new ExprError(`'${expr.op}' is not a known operation`, 0);
      // The CALLEE's own parameter order binds the arguments (§3.3) — exactly the rule a built-in
      // follows through `OPERATOR_PARAMS`. The only difference is that a built-in's signature ships
      // with the language.
      const names = positionalNames(resolved);
      // An argument past the last slot has nowhere to go, and `bindPositionally` DROPS it silently.
      // That was tolerable while a callee's slots came only from a document somebody wrote by hand
      // beside the call; it is not now that they are read off a TypeScript parameter list or a
      // registry entry, where a signature can change under a call site that still type-checks.
      if (expr.args.length > names.length) {
        throw new ExprError(
          `'${expr.op}' takes ${names.length === 0 ? "no arguments" : `${names.length} argument${names.length === 1 ? "" : "s"} (${names.join(", ")})`}, but ${expr.args.length} were given`,
          0,
        );
      }
      return {
        op: resolved,
        parameters: parametersFor(bindPositionally(expr.args, names, down)),
      };
    }
  }
}

/** What lowering needs from its caller: how to turn a NAME into the thing it names. */
export interface LowerOptions {
  /** Resolve a non-built-in operation name — a reference along the `path`, read as an operation. */
  resolveOperation?: (name: string) => Operation<InlineFamily> | undefined;
  /**
   * Resolve a bare name in VALUE position to whatever it names.
   *
   * Deliberately wider than {@link resolveOperation}: a reference can point at a string, a JSON
   * value, another binding, or an operation, and which one it is decides how it reads — the
   * shape-mismatch principle (REFERENCES.md §3) applied to the target rather than to the position.
   * The loader owns that dispatch because `desugarBinding` is already exactly it.
   */
  resolveName?: (name: string) => Ref<InlineFamily> | undefined;
  /**
   * How a js/ts body or symbol becomes an operation (SPEC §7.5).
   *
   * Carried here because `desugarState` already receives these options, and an embedded body is
   * lowered in exactly the place an expression is — a second channel for one state's compilation
   * context would be a second thing to keep in step.
   */
  userFunctions?: import("./userFunctions.js").UserFunctions;
}

/** The runtime namespaces, for the one diagnostic that has to survive the dot becoming required. */
const RUNTIME_ROOTS: ReadonlySet<string> = new Set(["inputs", "outputs", "children", "artifacts", "conversations", "run", "limits"]);

/**
 * Lower a bare name to what it names, or fail saying so.
 *
 * The error carries the migration hint deliberately. `children.a.outputs.n` was the spelling for a
 * runtime read until the dot became required, so the overwhelmingly likely cause of "no document
 * called `children.a.outputs.n`" is a missing dot rather than a missing file — and a resolver
 * message about the filesystem would send the reader looking in the wrong place entirely.
 */
function resolveName(name: string, options: LowerOptions): Ref<InlineFamily> {
  const resolved = options.resolveName?.(name) ?? asRef(options.resolveOperation?.(name));
  if (resolved !== undefined) return resolved;
  const root = name.split(".")[0]!;
  const hint = RUNTIME_ROOTS.has(root) ? ` — did you mean '.${name}', which reads this state's data?` : "";
  throw new ExprError(`'${name}' resolves to no document on the search path${hint}`, 0);
}

function asRef(op: Operation<InlineFamily> | undefined): Ref<InlineFamily> | undefined {
  return op === undefined ? undefined : { op };
}

/** Bind ordered arguments to ordered parameter names, ignoring any the callee has no slot for. */
function bindPositionally(
  args: readonly Expr[],
  names: readonly string[],
  lower: (e: Expr) => Ref<InlineFamily>,
): Record<string, Ref<InlineFamily>> {
  const out: Record<string, Ref<InlineFamily>> = {};
  args.forEach((arg, i) => {
    const name = names[i];
    if (name !== undefined) out[name] = lower(arg);
  });
  return out;
}

/** The order an operation binds POSITIONAL arguments in — {@link positionalOrder}, over its slots. */
function positionalNames(op: Operation<InlineFamily>): string[] {
  return positionalOrder(op.input);
}

/** Bound argument refs as the `parameters` of a producer edge — the callee's free slots, filled. */
function parametersFor(args: Record<string, Ref<InlineFamily>>): Record<string, Parameter<InlineFamily>> {
  const out: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, binding] of Object.entries(args)) {
    out[name] = { kind: "text" in binding ? "text" : "json", binding };
  }
  return out;
}

/** The resolver refs an expression lowers onto — the operators, plus the two that read data. */
export const EXPRESSION_REFS: ReadonlySet<string> = new Set<string>([
  RESOLVER_REFS.conversation,
  RESOLVER_REFS.context,
  RESOLVER_REFS.member,
  RESOLVER_REFS.not,
  RESOLVER_REFS.eq,
  RESOLVER_REFS.ne,
  RESOLVER_REFS.strictEq,
  RESOLVER_REFS.strictNe,
  RESOLVER_REFS.lt,
  RESOLVER_REFS.le,
  RESOLVER_REFS.gt,
  RESOLVER_REFS.ge,
  RESOLVER_REFS.and,
  RESOLVER_REFS.or,
  RESOLVER_REFS.cond,
  RESOLVER_REFS.record,
]);

/**
 * The input a call may declare to be told WHICH RULE it is part of.
 *
 * A call in a transition guard is written inside a rule that already says where the run goes if it
 * comes back true. Nothing could read that: a call's arguments are the ones the author typed, and the
 * `to` beside them was not one of them — so a function whose whole job is to offer somebody the move
 * this rule describes had to be told the destination a second time, in its own options, where it
 * could silently disagree with the rule it sat in.
 *
 * Declaring an input by this name is a call's way of asking. The loader binds `{ to }` — the
 * transition's target, exactly as authored — and only when the author has not bound it themselves, so
 * an explicit argument still wins.
 *
 * It is bound as an ARGUMENT rather than folded into the callee's definition, which is what keeps it
 * part of the call's identity: two rules offering the same event to two different places are two
 * calls, hash differently, and get one registration each.
 */
export const TRANSITION_INPUT = "transition";

/**
 * Fill {@link TRANSITION_INPUT} on every call in a lowered guard that declares it and left it unbound.
 *
 * A rebuild rather than a mutation: a callee's operation comes from a resolved document, and writing
 * into it would put one rule's destination on every other rule that names the same document.
 */
export function bindTransitionContext(ref: Ref<InlineFamily>, to: string): Ref<InlineFamily> {
  if (!("op" in ref)) return ref;
  const producer = ref.op;
  if (typeof producer === "string") return ref;

  const walked = (params: Record<string, Parameter<InlineFamily>> | undefined): Record<string, Parameter<InlineFamily>> | undefined => {
    if (params === undefined) return undefined;
    const out: Record<string, Parameter<InlineFamily>> = {};
    for (const [name, p] of Object.entries(params)) {
      out[name] = p.binding ? { ...p, binding: bindTransitionContext(p.binding, to) } : p;
    }
    return out;
  };

  const input = walked(producer.input) ?? producer.input;
  const parameters = walked(ref.parameters);
  const declared = Object.hasOwn(producer.input, TRANSITION_INPUT);
  const bound = parameters?.[TRANSITION_INPUT]?.binding !== undefined || producer.input[TRANSITION_INPUT]?.binding !== undefined;
  const filled: Record<string, Parameter<InlineFamily>> | undefined =
    declared && !bound ? { ...parameters, [TRANSITION_INPUT]: { kind: "json", binding: { json: { to } } } } : parameters;

  return {
    ...ref,
    op: { ...producer, input } as Operation<InlineFamily>,
    ...(filled !== undefined ? { parameters: filled } : {}),
  };
}

/**
 * Every root-anchored reference path in a lowered tree — the structural counterpart of
 * `referencesOf` over the AST, and what the validator's reachability obligation is checked over.
 *
 * A node that IS a path contributes it and is not descended into: the whole member chain is one
 * reference, exactly as `children.critique.outputs.outcome` is one reference and not four.
 */
export function referencePathsOf(ref: Ref<InlineFamily>): string[][] {
  const out: string[][] = [];
  const walk = (node: Ref<InlineFamily>): void => {
    const path = pathOfRef(node);
    if (path !== undefined) {
      out.push(path);
      return;
    }
    if (!("op" in node)) return;
    const producer = node.op;
    if (typeof producer === "string" || producer.kind !== "function") return;
    for (const p of Object.values(producer.input)) if (p.binding) walk(p.binding);
    // A lowered CALL's arguments live in `parameters`. Skipping them would exempt a call from the
    // reachability obligation — `classify(children.c.outputs.x)` reads `c` exactly as wiring it does.
    for (const p of Object.values(node.parameters ?? {})) if (p.binding) walk(p.binding);
  };
  walk(ref);
  return out;
}

/**
 * The root-anchored reference path a lowered sub-tree reads, or `undefined` when it is not one.
 *
 * The structural counterpart of `referencesOf` over the AST: a `context.get` at the bottom of a
 * `op.member` chain is a path, and anything else is a computation. This is what lets `fanout.ts` and
 * the validator ask "which child does this read?" of a tree instead of of a source string.
 */
export function pathOfRef(ref: Ref<InlineFamily>): string[] | undefined {
  if (!("op" in ref)) return undefined;
  const producer = ref.op;
  if (typeof producer === "string" || producer.kind !== "function") return undefined;
  if (producer.functionRef === RESOLVER_REFS.context) {
    const name = producer.input.name?.binding;
    return name !== undefined && "text" in name ? [name.text] : undefined;
  }
  if (producer.functionRef === RESOLVER_REFS.member) {
    const base = producer.input.value?.binding;
    const prop = producer.input.prop?.binding;
    if (base === undefined || prop === undefined || !("text" in prop)) return undefined;
    const head = pathOfRef(base);
    return head === undefined ? undefined : [...head, prop.text];
  }
  return undefined;
}
