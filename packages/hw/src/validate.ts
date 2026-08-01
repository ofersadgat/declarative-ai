/**
 * Structural + TYPE validation of a workflow (JaiRA DESIGN §5.2; SPEC §6.2). Runs
 * before a snapshot is accepted for execution; the same checks back a lint surface. Errors block
 * execution; warnings don't.
 *
 * Since the ops redesign this is the tier-2 check: once a workflow exists as VALUES every
 * schema is concrete, so the wiring can be fully type-checked before anything runs. Three checks
 * replace "the expression parses":
 *
 *  1. **Binding compatibility** — every producer's output schema must be an `isSubschema`
 *     of the consuming slot's schema. That walk is no longer written here: it is
 *     `@declarative-ai/validate`'s ONE generic checker (API.md, "The binding checker"), parameterized by the
 *     ref family. What stays local is the hw-SPECIFIC knowledge it takes as hooks — how a child key
 *     resolves, what the loader's synthesized resolver functions produce, and which producers are
 *     proven to have run.
 *  2. **Expression typing** — every `{ expr }` leaf and every `when` guard is inferred; a
 *     guard that doesn't infer to boolean is an error (strict, no truthiness coercion), and a
 *     declared schema on an expr leaf is an assertion checked against the inferred type.
 *  3. **Reachability** (§7.2, decided) — a reference to a producer not provably run on every path
 *     to its evaluation point is an error; a declared `default` is the explicit opt-out. So
 *     `T | undefined` never propagates silently.
 */
import type { FunctionCapabilities, InlineFamily, JsonSchema, JsonValue, Operation, Parameter, Ref, RefKind, RefTree } from "@declarative-ai/exec";
import { checkBinding as checkBindingGeneric, isSubschema, producerSchemaOf, type CheckerHooks, type CheckIssue, type Schema } from "@declarative-ai/validate";
import { parseExpression, referencesOf, type Expr } from "./expr";
import { EXPRESSION_REFS, pathOfRef, referencePathsOf } from "./lowerExpr";
import { embeddedOpsOf } from "./resolve";
import { validateSessionDecl } from "./session";
import { operationNodeSchema } from "./operationNode";
import { ANY_SCHEMA, inferExpression, inferRef, isBooleanSchema, isUniversalSchema, type ExprScope } from "./inferExpr";
import {
  GUARD_NAMESPACES,
  REF_NAMESPACES,
  RESOLVER_REFS,
  TERMINATE_TARGETS,
  type LoadedState,
  type SlotMeta,
  type WorkflowBundle,
} from "./format";

export interface ValidationIssue {
  stateId: string;
  /** Where in the state file, e.g. "transitions[2].when", "children.critique.inputs.plan_doc". */
  path: string;
  message: string;
}

export interface ValidationReport {
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
}

const NAMESPACES: ReadonlySet<string> = new Set([...REF_NAMESPACES, ...GUARD_NAMESPACES]);
const TERMINATES: ReadonlySet<string> = new Set(TERMINATE_TARGETS);
/** The termination outcomes a child's `outcome` can carry (SPEC §3.6). */
const TERMINATE_OUTCOMES = ["success", "error", "canceled", "timeout"] as const;
/** Every legal slot kind — `blob` joined the set when binary data became a leaf kind (§7). */
const SLOT_KINDS: ReadonlySet<string> = new Set<RefKind>(["text", "json", "blob", "prompt", "function"]);

/**
 * What the checker needs to know about the RUNTIME a bundle will execute against (DESIGN §7).
 * Validation is a function of *(document, registry)*, not of the document alone: a `functionRef`
 * naming nothing registered is an authoring error, and "an interactive function in a search-only
 * workflow" is only decidable by reading the entry's capabilities. Optional — a lint pass over a
 * document with no runtime in hand still checks everything else.
 */
export interface ValidationEnvironment {
  /** The registry the bundle will run against. */
  functions?: ReadonlyMap<string, FunctionCapabilities>;
  /**
   * Treat an unregistered `functionRef` as an ERROR rather than a warning. Off by default, and that
   * default is load-bearing: `validateBundle` checks the WHOLE document, but a state the run never
   * enters never needs its function — and NOT registering a function is the documented way a search
   * context refuses a human gate (see the executor's "gate isn't reached" case). A lint/CI surface,
   * which wants every reference to resolve, turns this on; the pre-run gate does not.
   */
  strict?: boolean;
  /** Assert a NON-interactive context (search/optimizer, which cannot answer a prompt): an operation
   *  bound to an interactive entry is then an error. Unset ⇒ not checked. */
  interactive?: boolean;
}

export function validateBundle(bundle: WorkflowBundle, env: ValidationEnvironment = {}): ValidationReport {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  for (const [id, def] of Object.entries(bundle.states)) {
    validateState(id, def, bundle, errors, warnings, env);
  }
  return { errors, warnings };
}

function validateState(
  id: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
  env: ValidationEnvironment,
): void {
  const err = (path: string, message: string): void => {
    errors.push({ stateId: id, path, message });
  };
  const warn = (path: string, message: string): void => {
    warnings.push({ stateId: id, path, message });
  };

  const children = def.children ?? {};
  const childKeys = new Set(Object.keys(children));
  const scope = exprScopeOf(def, bundle);
  const reachable = reachabilityOf(def);

  // --- children ---------------------------------------------------------------
  for (const [key, child] of Object.entries(children)) {
    const childDef = bundle.states[child.state];
    if (!childDef) {
      err(`children.${key}.state`, `references unknown state '${child.state}'`);
    } else if (!child.state.startsWith(id + "/")) {
      // The tree convention (SPEC §2.4/§3.1). Kept a warning so shared/library states mounted
      // cross-tree remain expressible; the engine only needs the reference to resolve.
      warn(`children.${key}.state`, `'${child.state}' is not a descendant path of '${id}'`);
    }
    for (const [inputName, binding] of Object.entries(child.inputs ?? {})) {
      const path = `children.${key}.inputs.${inputName}`;
      const consumer = childDef?.inputs?.[inputName];
      if (childDef && childDef.inputs && !consumer) {
        err(path, `child '${child.state}' declares no input '${inputName}'`);
        continue;
      }
      const consumerMeta = childDef?.slotMeta?.[`inputs.${inputName}`];
      checkBinding(binding, consumer?.schema, path, id, def, bundle, scope, reachable, errors, isOptOut(consumerMeta));
    }
    // Required child inputs must be wired (or defaulted/optional).
    if (childDef) {
      for (const inputName of Object.keys(childDef.inputs ?? {})) {
        const wired = child.inputs && inputName in child.inputs;
        const meta = childDef.slotMeta?.[`inputs.${inputName}`];
        if (!wired && meta?.optional !== true && meta?.default === undefined) {
          err(`children.${key}.inputs`, `required child input '${inputName}' is not wired`);
        }
      }
    }
  }

  // A lowered CALL names an operation the run must be able to dispatch. The registry check ran only
  // over a state's OWN `operation.functionRef`, so `shout(x)` with nothing registered passed lint and
  // then failed mid-run with "no function 'shout' is registered".
  for (const [where, binding] of bindingsOf(def)) {
    for (const { op } of embeddedOpsOf(binding)) {
      // A PROMPT callee needs no registry entry — it dispatches to the prompt executor.
      if (op.kind !== "function") continue;
      checkAgainstRegistry(op.functionRef, where, err, warn, env);
      checkAgainstSignature(op, where, err, env);
    }
  }

  // An operation the environment chain never completed (§5) — carried from the loader as data so it
  // is reported here with everything else, instead of aborting the load.
  if (def.operationError !== undefined) err("operation", def.operationError);

  // A `session` that cannot mean anything (DESIGN.md §1.6). Statically checkable because the merge
  // has already run, so what is tested is the EFFECTIVE declaration — including one an ancestor's
  // `environment` supplied. The empty string is the case worth catching at load time: `""` would
  // otherwise start an isolated conversation and report success, which is the failure mode that
  // makes `null` the only explicit "fresh" marker.
  const sessionComplaint = validateSessionDecl(def.environment?.session);
  if (sessionComplaint !== undefined) err("operation.session", sessionComplaint);

  // --- sequence ---------------------------------------------------------------
  const sequence = def.sequence ?? [];
  const seen = new Set<string>();
  sequence.forEach((entry, i) => {
    if (!childKeys.has(entry)) err(`sequence[${i}]`, `'${entry}' is not a declared child`);
    if (seen.has(entry)) err(`sequence[${i}]`, `duplicate sequence entry '${entry}'`);
    seen.add(entry);
  });

  // --- transitions ------------------------------------------------------------
  (def.transitions ?? []).forEach((t, i) => {
    if (!TERMINATES.has(t.to) && !childKeys.has(t.to)) {
      err(`transitions[${i}].to`, `'${t.to}' is neither a declared child nor a terminate.* outcome`);
    }
    let ast: Expr | undefined;
    if (t.when !== undefined) {
      const path = `transitions[${i}].when`;
      ast = checkExpression(t.when, path, def, childKeys, err);
      if (ast) {
        // A guard must INFER to boolean — strict, no truthiness coercion (§7.2): a `when` that
        // infers to `number` is a validation error, not a falsy surprise at run time.
        const { schema, unresolved } = inferExpression(ast, scope);
        // Quoted with the leading dot the author had to write: `unresolved` carries the path with
        // its self root already dropped, and a message spelling an internal path sends the reader
        // looking for a name that appears nowhere in their file.
        for (const ref of unresolved) err(path, `references '.${ref.join(".")}', which resolves to no declared value`);
        if (!isBooleanSchema(schema) && !isUniversalSchema(schema)) {
          err(path, `guard must infer to boolean, but infers to ${describeSchema(schema)} — compare explicitly`);
        }
      }
    }
    // Unguarded-cycle warning: a transition that re-enters a sequence member resets the cursor
    // (SPEC §3.3) and can loop forever without an iteration guard.
    //
    // Only an AUTHORED sequence counts. Every state with children has a sequence now (§6), so
    // testing the effective one would warn about every either/or state in existence — a transition
    // into one of two mutually exclusive children is ordinary control flow, not a declared order
    // being contradicted. Writing the sequence out is what turns "these run in this order" into a
    // claim a transition can violate, and that is the case worth flagging.
    if (childKeys.has(t.to) && def.sequenceAuthored === true && sequence.includes(t.to) && def.limits?.max_iterations === undefined) {
      const guarded = ast !== undefined && referencesOf(ast).some((p) => p[0] === "run" && p[1] === "iteration");
      if (!guarded) {
        warn(`transitions[${i}]`, `transition to sequence member '${t.to}' can cycle; add limits.max_iterations or a run.iteration guard`);
      }
    }
  });

  // --- declared slots ---------------------------------------------------------
  for (const [section, slots] of [
    ["inputs", def.inputs],
    ["outputs", def.outputs],
  ] as const) {
    for (const [name, slot] of Object.entries(slots ?? {})) {
      const path = `${section}.${name}`;
      if (!SLOT_KINDS.has(slot.kind)) {
        err(`${path}.kind`, `unknown slot kind '${String(slot.kind)}'`);
      }
      if (slot.binding !== undefined) {
        checkBinding(slot.binding, slot.schema, path, id, def, bundle, scope, reachable, errors, isOptOut(def.slotMeta?.[path]));
      }
    }
  }

  // --- operation --------------------------------------------------------------
  if (def.operation) {
    checkOperation(def.operation, "operation", id, def, bundle, scope, reachable, errors, warn, env);
  }

  const hasOperation = def.operation !== undefined || Object.keys(children).length > 0;
  if (!hasOperation) {
    warn("", "state declares no operation and no children; it will terminate immediately");
  }
}

// --- Operations ---------------------------------------------------------------

function checkOperation(
  op: Operation<InlineFamily>,
  path: string,
  stateId: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  scope: ExprScope,
  reachable: Reachability,
  errors: ValidationIssue[],
  warn: (path: string, message: string) => void,
  env: ValidationEnvironment,
): void {
  const err = (p: string, m: string): void => {
    errors.push({ stateId, path: p, message: m });
  };
  if (op.kind === "function") {
    if (typeof op.functionRef !== "string" || op.functionRef.length === 0) {
      err(`${path}.function`, "a function operation must name a function");
    } else {
      checkAgainstRegistry(op.functionRef, `${path}.function`, err, warn, env);
    }
  } else if (op.user === undefined || op.user === "") {
    warn(`${path}.prompt`, "prompt operation has an empty prompt (no template and no skill)");
  }
  for (const [name, param] of Object.entries(op.input)) {
    if (param.binding !== undefined) {
      // The state's operation runs BEFORE any child (engine loop step 2/5), so no child output exists
      // when it resolves its inputs — a child reference here can only ever fail at run time ("child 'X'
      // has not run"). An operation input is a value/expression over the state's OWN scope (inputs,
      // artifacts, …) or an embedded operation with its own scope, never a reach into children.
      const child = firstChildRefOf(param.binding);
      if (child !== undefined) {
        err(
          `${path}.input.${name}`,
          `an operation input cannot reference child '${child}': the operation runs before any child, so no child output exists when its inputs resolve — bind it to an input, a literal, or an embedded operation instead`,
        );
        continue;
      }
      checkBinding(param.binding, param.schema, `${path}.input.${name}`, stateId, def, bundle, scope, reachable, errors);
    }
  }
}

/**
 * The first CHILD a desugared binding reads, or `undefined`. Used to enforce that an OPERATION input
 * names no child (the operation runs first, §7.4). Mirrors the producer-edge traversal in `fanout.ts`:
 * a whole-child edge, a `select` over a child, an `{ expr }` reading `children.*`, or any of these
 * nested inside a `refs` tree. An embedded (non-resolver) operation carries its OWN scope, so its
 * internals are not the outer state's children — and it is rejected separately as un-runnable.
 */
function firstChildRefOf(ref: Ref<InlineFamily>): string | undefined {
  if ("op" in ref) {
    const producer = ref.op;
    if (typeof producer === "string") return producer; // `{ child: P }` — a whole-child edge
    if (producer.kind !== "function") return undefined; // an embedded op carries its own scope
    if (producer.functionRef === RESOLVER_REFS.select) {
      const value = producer.input.value?.binding; // `{ child: P, output: o }` lowers to a select over the edge
      return value !== undefined ? firstChildRefOf(value) : undefined;
    }
    // A lowered expression reads a child through the context (EXPRESSIONS.md §1.3), so the child it
    // names is read structurally rather than out of a re-parsed source string.
    const path = pathOfRef(ref);
    if (path !== undefined) return path[0] === "children" ? path[1] : undefined;
    // An operator's operand may still name one, even when the node as a whole is not a path.
    for (const p of Object.values(producer.input)) {
      if (!p.binding) continue;
      const child = firstChildRefOf(p.binding);
      if (child !== undefined) return child;
    }
    return undefined; // scope / artifact / conversation resolvers name no child
  }
  if ("refs" in ref) return firstChildRefInTree(ref.refs);
  return undefined;
}

function firstChildRefInTree(tree: RefTree<InlineFamily>): string | undefined {
  if (tree === null || typeof tree !== "object") return undefined;
  if (Array.isArray(tree)) {
    for (const item of tree) {
      const found = firstChildRefInTree(item);
      if (found !== undefined) return found;
    }
    return undefined;
  }
  const here = firstChildRefOf(tree as Ref<InlineFamily>);
  if (here !== undefined) return here;
  for (const sub of Object.values(tree)) {
    const found = firstChildRefInTree(sub as RefTree<InlineFamily>);
    if (found !== undefined) return found;
  }
  return undefined;
}

/**
 * The checks that need the RUNTIME, not just the document (§2). Skipped entirely when no registry was
 * supplied — a lint pass over a bundle with no runtime in hand is still worth running, and guessing
 * would turn every such pass into a wall of false errors.
 *
 * The loader's own synthesized resolvers (`RESOLVER_REFS`) are engine built-ins, never registry
 * entries, so they are exempt.
 */
function checkAgainstRegistry(
  functionRef: string,
  path: string,
  err: (path: string, message: string) => void,
  warn: (path: string, message: string) => void,
  env: ValidationEnvironment,
): void {
  const registry = env.functions;
  if (!registry || RESOLVER_REF_SET.has(functionRef)) return;
  const entry = registry.get(functionRef);
  if (!entry) {
    // Reached at RUN time this is fatal to the whole run (a transition could otherwise re-enter the
    // state forever) — but a state the run never enters never needs its function, and leaving one
    // unregistered is how a search context refuses a human gate. So: a warning by default, an error
    // only where the caller has said every reference must resolve.
    (env.strict === true ? err : warn)(path, `no function '${functionRef}' is registered`);
    return;
  }
  // "An interactive function in a search-only workflow" — the check §2 names as the reason validation
  // reads the registry at all. Capabilities are REQUIRED and total per variant, so this reads a
  // definite value rather than falling through an `undefined`.
  if (env.interactive === false && entry.kind !== "pure" && entry.capabilities.interactive) {
    err(path, `function '${functionRef}' is interactive, but this workflow is validated for a non-interactive context`);
  }
}

const RESOLVER_REF_SET: ReadonlySet<string> = new Set(Object.values(RESOLVER_REFS));

// --- Binding type-checking ---------------------------------------------

/**
 * The hw-specific knowledge the shared checker needs (API.md, "The binding checker"). Everything family-
 * generic — literal typing, kind agreement for higher-order slots, the `isSubschema` call, the
 * reachability REPORT — lives in `@declarative-ai/validate`; these three hooks are what only hw knows.
 */
function hooksFor(
  stateId: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  scope: ExprScope,
  reachable: Reachability,
  errors: ValidationIssue[],
  optOut: boolean,
): CheckerHooks<InlineFamily> {
  const hooks: CheckerHooks<InlineFamily> = {
    /** A producer named by a LOCAL KEY is a declared child — the inline family's analog of an op id.
     *  It is modeled as a synthetic op whose output schema is the child's declared outputs, so the
     *  generic checker needs no special case for it. */
    producer: (ref) => {
      if (typeof ref !== "string") return ref;
      const child = def.children?.[ref];
      if (!child) return undefined;
      const childState = bundle.states[child.state];
      const schema = childState ? outputsObjectSchema(childState) : undefined;
      return {
        kind: "function",
        functionRef: ref,
        input: {},
        output: { name: "output", kind: "json", ...(schema !== undefined ? { schema } : {}) },
      };
    },
    reachable: (ref) => (typeof ref === "string" ? reachable.always.has(ref) : true),
    producerSchema: (op, path, report) => resolverSchema(op, path, stateId, def, bundle, scope, reachable, errors, optOut, report),
  };
  return hooks;
}

/**
 * A callee DOCUMENT declares the parameters a call's arguments bind to; the implementation it names
 * lives in the registry. When the entry declares a signature, the two must agree — and this is the
 * only check that can catch them drifting.
 *
 * Without it the document is the sole authority: a call type-checks against parameters the impl does
 * not have, and the disagreement surfaces as a missing argument at run time. With it, renaming a
 * parameter in one place and not the other is a lint error.
 */
function checkAgainstSignature(
  op: Operation<InlineFamily> & { kind: "function" },
  path: string,
  err: (path: string, message: string) => void,
  env: ValidationEnvironment,
): void {
  const entry = env.functions?.get(op.functionRef);
  const declared = entry?.signature;
  if (declared === undefined) return; // an entry that declares nothing constrains nothing
  const expected = declared.input.schema as JsonSchema | undefined;
  const accepted = expected?.properties;
  // Only a declared PROPERTY SET constrains anything: an impl taking an open object accepts whatever
  // the document names, and there is nothing to disagree about.
  if (accepted === null || typeof accepted !== "object" || Array.isArray(accepted)) return;

  for (const [name, param] of Object.entries(op.input)) {
    // NAMES first, and this is the check that matters: a document parameter the impl has no slot for
    // arrives as an argument nothing reads. Comparing SCHEMAS alone would miss it entirely, because
    // JSON Schema objects are open — an impl declaring `body` happily validates `{text: …}`.
    if (!Object.hasOwn(accepted, name)) {
      err(path, `operation '${op.functionRef}' declares a parameter '${name}' its registered implementation does not accept`);
      continue;
    }
    // Where BOTH sides declare a type, they must agree.
    const want = (accepted as Record<string, JsonValue>)[name] as JsonSchema | undefined;
    if (param.schema === undefined || want === undefined || isUniversalSchema(want)) continue;
    const check = isSubschema(param.schema as Schema, want as Schema);
    if (!check.ok) {
      err(path, `operation '${op.functionRef}' declares parameter '${name}' as a type its implementation does not accept: ${check.reason}`);
    }
  }
}

/** Every binding a state carries, with the field that named it — guards included. */
function* bindingsOf(def: LoadedState): Iterable<[string, Ref<InlineFamily>]> {
  const op = def.operation;
  if (op) for (const [name, p] of Object.entries(op.input)) if (p.binding) yield [`operation.input.${name}`, p.binding];
  for (const [name, slot] of Object.entries(def.outputs ?? {})) if (slot.binding) yield [`outputs.${name}`, slot.binding];
  for (const [name, slot] of Object.entries(def.inputs ?? {})) if (slot.binding) yield [`inputs.${name}`, slot.binding];
  for (const [key, child] of Object.entries(def.children ?? {})) {
    for (const [name, wire] of Object.entries(child.inputs ?? {})) yield [`children.${key}.inputs.${name}`, wire];
  }
  for (const [i, t] of (def.transitions ?? []).entries()) {
    if (t.whenRef !== undefined) yield [`transitions[${i}].when`, t.whenRef];
  }
}

/**
 * Check ONE desugared binding against the schema of the slot it fills, through the shared checker.
 */
function checkBinding(
  binding: Ref<InlineFamily>,
  consumerSchema: JsonSchema | undefined,
  path: string,
  stateId: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  scope: ExprScope,
  reachable: Reachability,
  errors: ValidationIssue[],
  optOut = false,
): void {
  // Reference and reachability checks run over the WHOLE binding, once, before the type check.
  //
  // They used to live inside the expression branch of `resolverSchema`, which made them depend on
  // which node happened to be the ROOT: `shout(children.c.outputs.x) === 'y'` was checked (the root
  // is an operator) and `shout(children.c.outputs.x)` was not (the root is a call, and a call is not
  // an expression ref). Checking the binding itself has no such blind spot, and it is where a call's
  // ARGUMENTS live — they are on the ref, not on the callee's own `input`.
  for (const reference of referencePathsOf(binding)) {
    const root = reference[0]!;
    if (!NAMESPACES.has(root)) {
      errors.push({ stateId, path, message: `expression uses unknown reference root '${root}' (expected one of: ${[...NAMESPACES].join(", ")})` });
      continue;
    }
    if (optOut) continue;
    if (root === "children" && reference[1] !== undefined && !reachable.always.has(reference[1])) {
      errors.push({
        stateId,
        path,
        message:
          def.children?.[reference[1]] === undefined
            ? `expression references undeclared child '${reference[1]}'`
            : `expression reads child '${reference[1]}', which is not proven to have run on every path to this point`,
      });
    }
  }
  const issues = checkBindingGeneric(binding, consumerSchema, hooksFor(stateId, def, bundle, scope, reachable, errors, optOut), path, {
    optOut,
  });
  for (const issue of issues) errors.push({ stateId, path: issue.path, message: issue.message });
}

/**
 * The output type of a producer the LOADER synthesized (`RESOLVER_REFS`) — the hw-specific half of
 * producer typing. Returns `undefined` for anything else, so the generic checker falls through to its
 * "the declared output schema is the type" rule.
 */
function resolverSchema(
  op: Operation<InlineFamily>,
  path: string,
  stateId: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  scope: ExprScope,
  reachable: Reachability,
  errors: ValidationIssue[],
  optOut: boolean,
  err: (message: string) => void,
): JsonSchema | undefined {
  if (op.kind !== "function") return undefined;

  // A lowered EXPRESSION (EXPRESSIONS.md §1): infer its result type over the TREE — that IS its
  // producer schema (§7.2), which is what makes ordinary `isSubschema` binding checking apply to an
  // expression with no special case. It used to re-parse a source string; the parse now happens once,
  // in the loader, and a malformed expression fails there rather than here.
  if (EXPRESSION_REFS.has(op.functionRef)) {
    const asRef: Ref<InlineFamily> = { op };
    const { schema, unresolved } = inferRef(asRef, scope);
    for (const unres of unresolved) err(`expression references '.${unres.join(".")}', which resolves to no declared value`);
    // Reachability applies to expressions too: reading a child's outputs from an expression is the
    // same edge as wiring it, so it carries the same proof obligation.
    for (const reference of referencePathsOf(asRef)) {
      const root = reference[0]!;
      if (!NAMESPACES.has(root)) {
        err(`expression uses unknown reference root '${root}' (expected one of: ${[...NAMESPACES].join(", ")})`);
        continue;
      }
      if (optOut) continue;
      if (root === "children" && reference[1] !== undefined && !reachable.always.has(reference[1])) {
        if (def.children?.[reference[1]] === undefined) err(`expression references undeclared child '${reference[1]}'`);
        else err(`expression reads child '${reference[1]}', which is not proven to have run on every path to this point`);
      }
    }
    // A declared schema on the leaf is an ASSERTION, checked against the inferred type.
    const declared = op.output.schema;
    if (declared !== undefined && !isUniversalSchema(declared) && !isUniversalSchema(schema)) {
      const check = isSubschema(schema as Schema, declared as Schema);
      if (!check.ok) err(`expression infers to ${describeSchema(schema)}, which does not satisfy the declared schema: ${check.reason}`);
    }
    return schema;
  }

  switch (op.functionRef) {
    case RESOLVER_REFS.select: {
      // `{ child, output }`: project one property off the child's outputs object.
      const value = op.input.value;
      const key = literalTextOf(op.input.key);
      // Recurse into the producer feeding the `value` slot — its issues (an undeclared child, an
      // unproven reachability edge) are THIS binding's issues, so they are forwarded, never swallowed.
      const inner: CheckIssue[] = [];
      const base = value?.binding
        ? producerSchemaOf(value.binding, hooksFor(stateId, def, bundle, scope, reachable, errors, optOut), path, inner, optOut, value.kind)
        : undefined;
      for (const issue of inner) err(issue.message);
      if (base === undefined || key === undefined) return undefined;
      const props = base.properties;
      if (props !== null && typeof props === "object" && !Array.isArray(props)) {
        // OWN properties only: an inherited hit is not a declared output, and treating one as
        // declared both skipped this error and returned a FUNCTION where a schema was expected.
        const p = Object.hasOwn(props, key) ? (props as Record<string, JsonValue>)[key] : undefined;
        if (p === undefined) {
          err(`selects output '${key}', which the producer does not declare`);
          return undefined;
        }
        return p as JsonSchema;
      }
      return undefined;
    }
    case RESOLVER_REFS.scope: {
      // `{ input }`: the declared slot's own schema.
      const name = literalTextOf(op.input.name);
      if (name === undefined) return undefined;
      const slot = def.inputs?.[name];
      if (!slot) {
        err(`references undeclared input '${name}'`);
        return undefined;
      }
      return slot.schema;
    }
    default:
      // An artifact/conversation resolver is session-owned (content known only at run time), and a
      // DECLARED CHILD arrives here as the stand-in op `hooksFor.producer` synthesizes (its
      // `functionRef` is the child key). For both, the declared output schema is the producer type —
      // the generic rule.
      //
      // Anything else is an operation EMBEDDED in a binding — a lowered CALL (EXPRESSIONS.md §3).
      //
      // This used to be an error, and rejecting it was right at the time: `runResolver`'s default
      // branch refused to run one, so a binding to it validated clean and then never resolved. Now
      // the engine runs it (memoized, before resolution reads the result), so the pair is honest the
      // other way round — and leaving the rejection would fail every workflow that calls anything.
      //
      // Its type is its declared output schema, which is the generic rule, hence `undefined`.
      return undefined;
  }
}

/** The literal text a parameter is bound to, when it is a plain `{ text }` literal. */
function literalTextOf(param: Parameter<InlineFamily> | undefined): string | undefined {
  const b = param?.binding;
  return b !== undefined && "text" in b ? b.text : undefined;
}

/** The schema an inline JSON literal satisfies — precise enough for the subschema check. */
function schemaOfValue(v: JsonValue): JsonSchema {
  if (v === null) return { type: "null" };
  if (Array.isArray(v)) return { type: "array" };
  switch (typeof v) {
    case "string":
      return { type: "string", const: v };
    case "number":
      return { type: Number.isInteger(v) ? "integer" : "number", const: v };
    case "boolean":
      return { type: "boolean", const: v };
    default: {
      // An object literal: describe it exactly (every key present and required), so it satisfies
      // any consumer that requires a subset of these properties.
      const properties: Record<string, JsonValue> = {};
      for (const [k, val] of Object.entries(v)) properties[k] = schemaOfValue(val) as JsonValue;
      return { type: "object", properties, required: Object.keys(v) };
    }
  }
}

/** A state's declared outputs as one object schema — what a producer edge on it emits. */
function outputsObjectSchema(state: LoadedState): JsonSchema | undefined {
  const outputs = state.outputs;
  if (!outputs || Object.keys(outputs).length === 0) return undefined;
  const properties: Record<string, JsonValue> = {};
  const required: string[] = [];
  for (const [name, slot] of Object.entries(outputs)) {
    properties[name] = (slot.schema ?? {}) as JsonValue;
    const meta = state.slotMeta?.[`outputs.${name}`];
    if (meta?.optional !== true && meta?.default === undefined) required.push(name);
  }
  return { type: "object", properties, ...(required.length > 0 ? { required } : {}) };
}

// --- Reachability analysis (§7.2) ---------------------------------------------

/**
 * Whether a consuming slot has explicitly opted OUT of the strict reachability rule (§7.2). A declared
 * `default` is the doc's named opt-out; an `optional` slot is the same declaration for outputs — both
 * say "absent is acceptable here", which is exactly what the rule otherwise forbids from propagating
 * silently.
 */
function isOptOut(meta: SlotMeta | undefined): boolean {
  return meta?.default !== undefined || meta?.optional === true;
}

interface Reachability {
  /** Children proven to have run on EVERY path reaching the state's evaluation point. */
  always: Set<string>;
}

/**
 * Definite-assignment analysis over `sequence`/`transitions`. A `sequence` runs its members in
 * order, so a member is proven to run on the path to the state's wiring and termination — UNLESS a
 * transition can fire first and terminate or divert the state. A child reachable ONLY through a
 * conditional transition is likewise not proven; that is the hole §7.2 closes, and
 * `optional`/`default` on the consuming slot is the opt-out.
 *
 * The pre-emption test is what makes this sound now that EVERY state with children has a sequence
 * (§6): "the sequence runs unconditionally" stopped being true the moment a derived sequence sat
 * under a state whose transitions terminate early. Critique is the case — it can answer `clean` and
 * terminate before either child runs, so neither is proven, and a required slot reading one is
 * still the error it was before the sequence was derived.
 *
 * A guard can only fire once everything it READS has resolved: a guard over
 * `children.<key>.outputs` evaluates to PENDING until that child completes, and PENDING is skipped
 * rather than taken (SPEC §6/§10.4). So a transition guarded on the LAST sequence member cannot
 * pre-empt any of them, while a guard reading only this state's own `outputs` can fire the moment
 * the operation completes and therefore pre-empts everything.
 *
 * An `async` member counts as proven. Async means "started but not awaited", i.e. its outputs may
 * be PENDING at read time — and PENDING is a RUNTIME park (the dataflow join, SPEC §10.4), not a
 * value that can be permanently missing. The engine parks the consumer until the producer resolves,
 * so the read is sound; a fan-out feeding a synthesize step is exactly this pattern. What the rule
 * must forbid is a producer that might never run at all, which is the conditional case.
 *
 * Deliberately simple and CONSERVATIVE beyond that: it proves ordered sequences and refuses
 * everything else.
 */
function reachabilityOf(def: LoadedState): Reachability {
  const sequence = def.sequence ?? [];
  const indexOf = new Map(sequence.map((key, i) => [key, i] as const));

  // The earliest point each transition could be taken, as a sequence index: -1 = "as soon as the
  // operation completes", n = "not before member n has". A guard naming no child is unblocked.
  let earliest = Number.POSITIVE_INFINITY;
  for (const t of def.transitions ?? []) {
    let blockedUntil = -1;
    if (t.when !== undefined) {
      let ast: Expr | undefined;
      try {
        ast = parseExpression(t.when);
      } catch {
        ast = undefined; // a guard that does not parse is reported elsewhere; assume the worst
      }
      if (ast) {
        for (const path of referencesOf(ast)) {
          if (path[0] !== "children" || path[1] === undefined) continue;
          const at = indexOf.get(path[1]);
          if (at !== undefined) blockedUntil = Math.max(blockedUntil, at);
        }
      }
    }
    earliest = Math.min(earliest, blockedUntil);
  }

  const always = new Set<string>();
  sequence.forEach((key, i) => {
    // A transition that can fire at `earliest` pre-empts every member after it.
    if (def.children?.[key] && earliest >= i) always.add(key);
  });
  return { always };
}

// --- Expression scope ---------------------------------------------------------

/** Build the typed namespace map an expression is inferred against (§7.2/§7.5). */
function exprScopeOf(def: LoadedState, bundle: WorkflowBundle): ExprScope {
  const objectOf = (slots: Record<string, { schema?: JsonSchema }> | undefined): JsonSchema => {
    const properties: Record<string, JsonValue> = {};
    for (const [name, slot] of Object.entries(slots ?? {})) properties[name] = (slot.schema ?? {}) as JsonValue;
    return { type: "object", properties };
  };

  // Each child exposes its `outputs` and its termination `outcome` — the two things the engine puts
  // in the expression context for it (SPEC §3.6).
  const childrenProps: Record<string, JsonValue> = {};
  const outcomeSchema: JsonValue = { type: "string", enum: [...TERMINATE_OUTCOMES] };
  for (const [key, child] of Object.entries(def.children ?? {})) {
    const childState = bundle.states[child.state];
    const outputs = childState ? (outputsObjectSchema(childState) ?? ANY_SCHEMA) : ANY_SCHEMA;
    // A child's own operation node, typed by ITS operation's kind — so
    // `children.plan.operation.outputs.session` is checked against what `plan` actually runs, and
    // pointing it at a `ui` gate is a load-time error rather than a runtime undefined.
    const childOperation = childState ? operationNodeSchema(childState.operation?.kind) : undefined;
    childrenProps[key] = {
      type: "object",
      properties: {
        outputs: outputs as JsonValue,
        outcome: outcomeSchema,
        ...(childOperation !== undefined ? { operation: childOperation as JsonValue } : {}),
      },
    } as JsonValue;
  }

  // The state's OWN operation node (SPEC.md §6.1). Absent for a pure composite, so
  // `operation.cost` there is an unresolved reference rather than an object of unknowns.
  const operation = operationNodeSchema(def.operation?.kind);

  const sequenceKeys = def.sequence ?? [];

  return {
    inputs: objectOf(def.inputs),
    outputs: objectOf(def.outputs),
    ...(operation !== undefined ? { operation } : {}),
    children: { type: "object", properties: childrenProps },
    // Session-owned resources: addressable, contents known only at run time.
    artifacts: { type: "object" },
    // Guard-only control-flow scalars (§7.5) — never a reference binding.
    run: {
      type: "object",
      properties: {
        iteration: { type: "integer" } as JsonValue,
        // Where the sequence cursor is, by child key and by index — so a guard can say "if we are
        // at x and y holds, go to z". Typed as the declared child keys, so a typo is a lint error
        // rather than a comparison that is silently always false.
        cursor: (sequenceKeys.length > 0 ? { type: "string", enum: ["", ...sequenceKeys] } : { type: "string" }) as JsonValue,
        position: { type: "integer" } as JsonValue,
      },
    },
    limits: { type: "object", properties: { max_iterations: { type: "integer" } as JsonValue, timeout: { type: "integer" } as JsonValue } },
  };
}

function describeSchema(s: JsonSchema): string {
  if (isUniversalSchema(s)) return "any";
  return typeof s.type === "string" ? s.type : JSON.stringify(s);
}

/**
 * Parse an expression and statically check its reference ROOTS: the root must be a known
 * namespace and `children.<key>` must be declared. Type-level checking is `inferExpression`'s job;
 * this is the syntactic gate that runs first.
 */
function checkExpression(
  src: string,
  path: string,
  def: LoadedState,
  childKeys: ReadonlySet<string>,
  err: (path: string, message: string) => void,
): Expr | undefined {
  let ast: Expr;
  try {
    ast = parseExpression(src);
  } catch (e) {
    err(path, `expression does not parse: ${(e as Error).message}`);
    return undefined;
  }
  for (const ref of referencesOf(ast)) {
    const root = ref[0]!;
    if (!NAMESPACES.has(root)) {
      err(path, `unknown reference root '${root}' (expected one of: ${[...NAMESPACES].join(", ")})`);
      continue;
    }
    if (root === "children" && ref[1] !== undefined && !childKeys.has(ref[1])) {
      err(path, `references undeclared child '${ref[1]}'`);
    }
    if (root === "inputs" && ref[1] !== undefined && !(def.inputs && ref[1] in def.inputs)) {
      err(path, `references undeclared input '${ref[1]}'`);
    }
    if (root === "outputs" && ref[1] !== undefined && !(def.outputs && ref[1] in def.outputs)) {
      err(path, `references undeclared output '${ref[1]}'`);
    }
  }
  return ast;
}
