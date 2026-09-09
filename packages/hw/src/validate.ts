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
import type { FunctionCapabilities, InlineFamily, JsonSchema, JsonValue, NamedParameter, Operation, Parameter, Ref, RefKind, RefTree } from "@declarative-ai/exec";
import { isRequiredSlot } from "@declarative-ai/exec";
import { checkBinding as checkBindingGeneric, isSubschema, producerSchemaOf, type CheckerHooks, type CheckIssue, type Schema } from "@declarative-ai/validate";
import { parseExpression, referencesOf, type Expr } from "./expr.js";
import { EXPRESSION_REFS, pathOfRef, referencePathsOf } from "./lowerExpr.js";
import { embeddedOpsOf } from "./resolve.js";
import { validateSessionDecl } from "./session.js";
import { operationNodeSchema } from "./operationNode.js";
import { ANY_SCHEMA, inferExpression, inferRef, isBooleanSchema, isUniversalSchema, type ExprScope } from "./inferExpr.js";
import {
  EACH_NAMESPACE,
  GUARD_NAMESPACES,
  REF_NAMESPACES,
  RESOLVER_REFS,
  TERMINATE_TARGETS,
  type LoadedState,
  type LoadedTransition,
  type SlotMeta,
  type WorkflowBundle,
} from "./format.js";

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
  /**
   * Whether a model's WEIGHTS are on this machine — for locally-served models, which cannot run until
   * something has fetched multiple gigabytes of them.
   *
   * Three properties of the signature are deliberate:
   *
   *  - **It is a predicate, not a lookup.** `hw` does not depend on `@declarative-ai/llm` and must not
   *    learn what a route is; the caller, who does, decides which ids it manages.
   *  - **`undefined` means NOT MINE.** A remote model has no weights to be missing, so a caller returns
   *    `undefined` for it and nothing is reported. Only an explicit `false` is a finding.
   *  - **It is SYNCHRONOUS.** This validator backs a lint surface and never performs I/O beyond a stat;
   *    "is it here" is answerable, "go and fetch it" is not. Provisioning is a separate step the host
   *    runs between validating and executing.
   *
   * Unset ⇒ not checked, like `interactive` above. The finding is a WARNING rather than an error
   * because a state the run never enters never needs its weights, which is the same reasoning that
   * keeps an unregistered `functionRef` a warning by default.
   */
  weightsPresent?: (modelId: string) => boolean | undefined;
  /**
   * Where a locally-served model's working set would LAND on this machine — the three placement tiers.
   *
   * `"vram"` runs at full speed, `"ram"` spills off the GPU and runs slowly, `"swap"` spills past system
   * memory into the pagefile and effectively does not run at all. `undefined` means "not a model I
   * manage", exactly as {@link weightsPresent} does.
   *
   * Synchronous for the same reason: the validator does no async work. The caller predicts placements
   * ahead of time (the prediction is cheap — it reads the GGUF header, not the weights) and hands in a
   * lookup. Doing it at validation rather than at admission is the point: the loader resolves the
   * `environment` chain first, so a workflow's whole model working set is knowable before anything runs
   * instead of discovered one failed call at a time.
   */
  placement?: (modelId: string) => "vram" | "ram" | "swap" | undefined;
  /**
   * How severely a degraded placement is reported. Defaults: spilling to RAM is a WARNING (it works,
   * slowly, and the author should know), spilling to swap is an ERROR (it does not meaningfully run).
   *
   * This is the acknowledgment surface. A host whose user has said "yes, run the 70B degraded anyway"
   * sets `swap: "warn"`; one that will not tolerate any spill sets `ram: "error"`. The policy lives in
   * the host's configuration, never in a workflow document — a workflow that could downgrade its own
   * safety check would not be a check.
   */
  placementPolicy?: { ram?: "ignore" | "warn" | "error"; swap?: "ignore" | "warn" | "error" };
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
  /** How this state types one binding — see {@link typeOf}. */
  const typed = typeOf(def.id, def, bundle, scope, reachable, errors, false);

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
    // A per-mount `environment` is an ordinary defaults layer, so its `session` is checkable exactly
    // where a state's is — and reporting it HERE names the line the author wrote, rather than
    // surfacing it against the child state after the merge has moved it.
    const childSessionComplaint = validateSessionDecl(child.environment?.session);
    if (childSessionComplaint !== undefined) err(`children.${key}.environment.session`, childSessionComplaint);
    // A mount that FANS OUT (§6.2): its `each` wires must carry an array of what the child declares,
    // and every wire on it may read `.each` — the axes are what that namespace is typed from.
    const axes = child.each !== undefined && child.each.length > 0 ? child.each : undefined;
    for (const [inputName, binding] of Object.entries(child.inputs ?? {})) {
      const path = `children.${key}.inputs.${inputName}`;
      const consumer = childDef?.inputs?.[inputName];
      if (childDef && childDef.inputs && !consumer) {
        err(path, `child '${child.state}' declares no input '${inputName}'`);
        continue;
      }
      const consumerMeta = childDef?.slotMeta?.[`inputs.${inputName}`];
      // An `each` wire feeds the child ONE element per entry, so the wire itself must produce a list
      // of them: the consumer, for the check, is array-of-what-the-child-declares.
      const wanted: JsonSchema | undefined = axes?.includes(inputName) ? { type: "array", items: (consumer?.schema ?? {}) as JsonValue } : consumer?.schema;
      // Asked AT THE MOUNT: a wire into `key` resolves when `key` is entered, so a sibling that runs
      // after it is not proven for this binding even though it is for the state's own outputs.
      checkBinding(binding, wanted, path, id, def, bundle, scope, reachable.enteredAt(key), errors, isOptOut(consumerMeta), axes);
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
    for (const { op, parameters } of embeddedOpsOf(binding)) {
      // A PROMPT callee needs no registry entry — it dispatches to the prompt executor.
      if (op.kind !== "function") continue;
      checkAgainstRegistry(op.functionRef, where, err, warn, env);
      checkAgainstSignature(op, suppliedByCall(op, parameters, typed), where, err, env);
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

  // A `session` whose SCOPE could not be resolved — an `in` or a `join` naming no ancestor, or a
  // `join` whose target declares nothing (DESIGN.md §1.6). Answered by the loader rather than here
  // because it is a question about the TREE, and only the walk knows which state wrote what; carried
  // to this pass so it is reported beside every other authoring error instead of aborting the load.
  if (def.sessionError !== undefined) err(def.sessionError.path, def.sessionError.message);

  // --- sequence ---------------------------------------------------------------
  const sequence = def.sequence ?? [];
  const seen = new Set<string>();
  sequence.forEach((entry, i) => {
    if (!childKeys.has(entry)) err(`sequence[${i}]`, `'${entry}' is not a declared child`);
    if (seen.has(entry)) err(`sequence[${i}]`, `duplicate sequence entry '${entry}'`);
    seen.add(entry);
  });

  // --- transitions ------------------------------------------------------------
  //
  // One check for both lists: a child's transitions are the state's, narrowed to the round that child
  // finishes in (SPEC §3.3), so every rule about a target, a guard's type and a cycle holds identically
  // — only the path in the message says which list the author wrote it in.
  /**
   * `mountKey` is the child whose list this is, absent for the state's own.
   *
   * It decides what a rule's WIRING may read. A rule on a child's mount fires in the round that
   * child finished, so that child has demonstrably run — and asking the question at the TARGET's
   * mount instead said the opposite, refusing `.children.b.output.x` on `b`'s own rule. The way
   * out of that would have been to declare the slot optional, which is exactly the pressure that
   * produced the silent-empty input this whole change exists to end.
   */
  const checkTransitions = (list: readonly LoadedTransition[] | undefined, where: string, mountKey?: string): void =>
    (list ?? []).forEach((t, i) => {
      if (!TERMINATES.has(t.to) && !childKeys.has(t.to)) {
        err(`${where}[${i}].to`, `'${t.to}' is neither a declared child nor a terminate.* outcome`);
      }
      // A transition's own wiring for the child it enters, checked exactly as the mount's is: same
      // scope, same reachability question, same "does the target declare this name" rule. Silence
      // here would be the failure this whole feature exists to end — a wire that reads as correct
      // and fills nothing.
      // The LOWERED wiring — `checkBinding` reads producer edges, and the mount's loop is handed
      // the same thing for the same reason.
      for (const [name, binding] of Object.entries(t.inputRefs ?? {})) {
        const path = `${where}[${i}].inputs.${name}`;
        if (TERMINATES.has(t.to)) {
          err(path, `'${t.to}' terminates the state; there is no child to pass inputs to`);
          continue;
        }
        const target = children[t.to];
        const targetDef = target === undefined ? undefined : bundle.states[target.state];
        const consumer = targetDef?.inputs?.[name];
        if (targetDef && targetDef.inputs && !consumer) {
          err(path, `'${t.to}' declares no input '${name}'`);
          continue;
        }
        const meta = targetDef?.slotMeta?.[`inputs.${name}`];
        // Reachability is asked AT THE RULE, which is where the value actually resolves — before
        // the pass opens and before the reset. On a child's mount that proves the child itself as
        // well as everything before it; on the state's own list it proves what always runs.
        const at = mountKey === undefined ? reachable : reachable.enteredAt(mountKey);
        const proven = mountKey === undefined ? at : { ...at, always: new Set([...at.always, mountKey]) };
        checkBinding(binding, consumer?.schema, path, id, def, bundle, scope, proven, errors, isOptOut(meta));
      }
      let ast: Expr | undefined;
      if (t.when !== undefined) {
        const path = `${where}[${i}].when`;
        ast = checkExpression(t.when, path, def, childKeys, err);
        // The guard PARSED but could not be LOWERED — a call naming something that resolves nowhere:
        // not a document on the search path, not a symbol a module contributes, not an entry the
        // registry puts there. The loader carries that as data rather than throwing
        // (`LoadedTransition.whenError`) so one bad guard cannot hide every other mistake in the
        // workflow, and reporting it is this function's half of that bargain.
        //
        // It went unreported, and the failure mode was the worst available: the engine SKIPS a
        // transition carrying `whenError`, so the rule simply stopped existing. A mistyped callee
        // read as a workflow with one fewer rule — lint clean, run silent, and where the rule was the
        // one offering a person a decision, an offer that was never made with nothing to explain why.
        //
        // Gated on `ast` because `checkExpression` owns the PARSE diagnostic. A guard that does not
        // parse fails to lower too, and reporting both would say the same thing twice in two
        // vocabularies. What is left here is precisely the failure re-parsing cannot see: resolution
        // needs the search path, the referring state and a filesystem, so it happens at load and only
        // its verdict reaches this far.
        if (ast && t.whenError !== undefined) {
          err(path, `guard could not be resolved: ${t.whenError}`);
        } else if (ast) {
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
          warn(`${where}[${i}]`, `transition to sequence member '${t.to}' can cycle; add limits.max_iterations or a run.iteration guard`);
        }
      }
    });

  checkTransitions(def.transitions, "transitions");
  for (const [key, child] of Object.entries(children)) checkTransitions(child.transitions, `children.${key}.transitions`, key);

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

  // A state with nothing to run and nothing to compute. Both halves matter: an output with a BINDING
  // is resolved when the state terminates (§3.7), so a state whose outputs all bind is a pure
  // computation — no model, no host code, no children — and terminating immediately is the whole
  // point of it rather than a symptom. Warning on those made the message unreadable where the shape
  // is deliberate: a scoring state over signals already in the run, or an arithmetic verdict.
  //
  // A PRODUCED output is the opposite case and the one worth keeping: nothing fills it, because
  // filling it is the operation's job and there is no operation, so the state fails at run time with
  // "required output was not produced". That is left to the unbound-output check, which reports it
  // against the slot; here we only decline to claim the state does nothing.
  const runsSomething = def.operation !== undefined || Object.keys(children).length > 0;
  const computesSomething = Object.values(def.outputs ?? {}).some((slot) => slot.binding !== undefined);
  if (!runsSomething && !computesSomething) {
    warn("", "state declares no operation, no children and no bound output; it will terminate immediately having done nothing");
  }
}

// --- Operations ---------------------------------------------------------------

/**
 * Report a locally-served model whose weights are not on this machine (§5).
 *
 * The check is possible at all because the loader has already merged the `environment` chain, so a
 * state's operation carries its RESOLVED model — including one supplied by an ancestor's defaults or by
 * a per-mount layer. That is what makes a workflow's whole model working-set knowable before anything
 * runs, rather than discovered one failed call at a time.
 *
 * The model has to be a literal to be checked: an `{expr}`-bound config is only knowable at run time,
 * and guessing would produce findings about models the run never asks for.
 */
function checkWeights(
  op: Extract<Operation<InlineFamily>, { kind: "prompt" }>,
  path: string,
  stateId: string,
  errors: ValidationIssue[],
  warn: (path: string, message: string) => void,
  env: ValidationEnvironment,
): void {
  const model = literalModelOf(op);
  if (model === undefined) return;

  // `undefined` is "not a model I manage" — a remote model has no weights to be missing.
  if (env.weightsPresent?.(model) === false) {
    warn(
      `${path}.config.model`,
      `model '${model}' is served locally but its weights are not present — provision them before running (validation cannot fetch them)`,
    );
  }

  const tier = env.placement?.(model);
  if (tier === undefined || tier === "vram") return;
  const policy = env.placementPolicy ?? {};
  // The defaults ARE the policy this exists to express: a RAM spill works and should be said out loud;
  // a swap spill does not meaningfully run and blocks until someone acknowledges it.
  const severity = tier === "swap" ? (policy.swap ?? "error") : (policy.ram ?? "warn");
  if (severity === "ignore") return;
  const message =
    tier === "swap"
      ? `model '${model}' would spill past system memory into swap on this machine — it will not run usefully. Reduce its context, choose a smaller quantization, or acknowledge the cost (placementPolicy.swap)`
      : `model '${model}' would spill out of VRAM into system memory on this machine — it will run, considerably slower`;
  if (severity === "error") errors.push({ stateId, path: `${path}.config.model`, message });
  else warn(`${path}.config.model`, message);
}

/** A prompt op's model id, when it is a LITERAL. An `{expr}`-bound config is only knowable at run time,
 *  and guessing would produce findings about models the run never asks for. */
function literalModelOf(op: Extract<Operation<InlineFamily>, { kind: "prompt" }>): string | undefined {
  const config: unknown = op.config;
  if (config === null || typeof config !== "object" || Array.isArray(config)) return undefined;
  const model = (config as Record<string, JsonValue>)["model"];
  return typeof model === "string" && model.length > 0 ? model : undefined;
}

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
  /** How this state types one binding — see {@link typeOf}. */
  const typed = typeOf(stateId, def, bundle, scope, reachable, errors, false);
  if (op.kind === "function") {
    if (typeof op.functionRef !== "string" || op.functionRef.length === 0) {
      err(`${path}.function`, "a function operation must name a function");
    } else {
      checkAgainstRegistry(op.functionRef, `${path}.function`, err, warn, env);
      // The state's OWN call, against the signature its implementation declares.
      //
      // This ran only over embedded calls inside bindings — a `{ op }` producer — so the operation a
      // state exists to run was the one call nobody compared against its impl. A state could name
      // every parameter wrongly, or pass nothing at all to a function that requires three arguments,
      // and lint clean right up until dispatch.
      checkAgainstSignature(op, suppliedByState(op, def, typed), path, err, env);
    }
  } else if (op.user === undefined || op.user === "") {
    warn(`${path}.prompt`, "prompt operation has an empty prompt (no template and no skill)");
  }
  if (op.kind === "prompt") checkWeights(op, path, stateId, errors, warn, env);
  // A state's operation reaches dispatch without ever being a binding, so its spreads are checked
  // here rather than in `checkSpreads`. A slot is already filled when the operation BOUND it — which
  // for a state op is the only spelling there is, `args` and `input` alike having become bindings by
  // the time they get here.
  checkSpreadArguments(op, (n) => op.input[n]?.binding !== undefined, path, stateId, typed, errors);
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
  /** States whose outputs are already being inferred — see {@link outputsObjectSchema}. */
  seen?: ReadonlySet<string>,
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
      const own = childState ? outputsObjectSchema(childState, bundle, seen ?? EMPTY_STATE_SET) : undefined;
      // A mount that FANS OUT (§6.2) is read back elementwise: every output an array, in element order.
      const schema = own !== undefined && child.each !== undefined && child.each.length > 0 ? elementwise(own) : own;
      return {
        kind: "function",
        functionRef: ref,
        input: {},
        output: { name: "output", kind: "json", ...(schema !== undefined ? { schema } : {}) },
      };
    },
    reachable: (ref) => (typeof ref === "string" ? reachable.always.has(ref) : true),
    producerSchema: (op, path, report) =>
      resolverSchema(op, path, stateId, def, bundle, scope, reachable, errors, optOut, report, seen),
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
  /** Whether a required slot of this name will actually have a value at dispatch — see the callers. */
  supplied: (name: string) => boolean,
  path: string,
  err: (path: string, message: string) => void,
  env: ValidationEnvironment,
): void {
  const entry = env.functions?.get(op.functionRef);
  const declared = entry?.signature;
  if (declared === undefined) return; // an entry that declares nothing constrains nothing
  const accepted = declared.input;
  // An entry declaring no slots at all constrains nothing — it is the "I take whatever the document
  // names" case, which is what every entry written before signatures existed means.
  if (Object.keys(accepted).length === 0) return;

  for (const [name, param] of Object.entries(op.input)) {
    // NAMES first, and this is the check that matters: a document parameter the impl has no slot for
    // arrives as an argument nothing reads. Comparing SCHEMAS alone would miss it entirely, because
    // JSON Schema objects are open — an impl declaring `body` happily validates `{text: …}`.
    const slot = accepted[name];
    if (slot === undefined) {
      err(path, `operation '${op.functionRef}' declares a parameter '${name}' its registered implementation does not accept`);
      continue;
    }
    // Where BOTH sides declare a type, they must agree.
    const want = slot.schema;
    if (param.schema === undefined || want === undefined || isUniversalSchema(want)) continue;
    const check = isSubschema(param.schema as Schema, want as Schema);
    if (!check.ok) {
      err(path, `operation '${op.functionRef}' declares parameter '${name}' as a type its implementation does not accept: ${check.reason}`);
    }
  }

  // The OTHER direction, and the one that actually stops a run: a parameter the impl REQUIRES and the
  // document never passes. The loop above catches a document naming something the impl has no slot
  // for — an argument nothing reads, which is untidy — while this catches the impl being handed
  // nothing for a slot it cannot work without, which fails the call.
  //
  // It is the same check `children.<key>.inputs` already gets ("required child input 'x' is not
  // wired"), asked of an operation. A state that mounts a child badly was a lint error; a state that
  // calls a function badly was not, and there is no reason for the two to differ.
  for (const [name, slot] of Object.entries(accepted)) {
    if (!isRequiredSlot(slot) || supplied(name)) continue;
    err(path, `operation '${op.functionRef}' requires an input '${name}', which this state does not pass`);
  }
}

/**
 * Whether a required slot of a STATE's own operation will have a value at dispatch.
 *
 * Two ways it can, and the check needs both. A BOUND slot on the operation is the obvious one. The
 * other is the engine's own rule: free slots are filled by name from the state's resolved inputs
 * (`opInputs = { ...instance.inputs, ...resolved.values }`), so a state whose declared input happens
 * to be called `text` fills the callee's `text` with nothing wired.
 *
 * "The operation declares a slot of this name" was the old test, and it stopped meaning anything once
 * `operation.function` resolved along the path: the callee's slots are copied ONTO the operation, so
 * every declared parameter is present whether or not anybody supplied it, and the check could never
 * have fired again.
 */
function suppliedByState(
  op: Operation<InlineFamily> & { kind: "function" },
  def: LoadedState,
  typed: (ref: Ref<InlineFamily>) => JsonSchema,
): (name: string) => boolean {
  const spread = spreadKeysOf(op, typed);
  return (name) =>
    op.input[name]?.binding !== undefined ||
    spread.has(name) ||
    (def.inputs !== undefined && Object.hasOwn(def.inputs, name));
}

/**
 * The same question for an EMBEDDED call, where the answer is a different one.
 *
 * A call's arguments ride the producer edge's `parameters`, and `resolveEmbedded` binds those into
 * the callee's slots and nothing else — there is no enclosing instance to spread in, which is exactly
 * what makes an expression's call a closed thing. So the state's inputs do not count here, and a slot
 * the callee itself binds (a document's declared default) does.
 */
function suppliedByCall(
  op: Operation<InlineFamily> & { kind: "function" },
  parameters: Record<string, Parameter<InlineFamily>> | undefined,
  typed: (ref: Ref<InlineFamily>) => JsonSchema,
): (name: string) => boolean {
  const spread = spreadKeysOf(op, typed);
  return (name) => parameters?.[name] !== undefined || op.input[name]?.binding !== undefined || spread.has(name);
}

/**
 * The slot names an operation's deferred spreads will fill.
 *
 * A required parameter passed ONLY by `f(...opts)` is passed, and the "requires an input it does not
 * pass" check has to know it — otherwise the one form that cannot be read at load would be the one
 * form that always reports a missing argument. Computable here for the same reason the spread is
 * checked here at all: the keys are in the operand's type, and this is where types exist.
 */
function spreadKeysOf(op: Operation<InlineFamily>, typed: (ref: Ref<InlineFamily>) => JsonSchema): ReadonlySet<string> {
  if (op.spread === undefined || op.spread.length === 0) return EMPTY_SLOT_SET;
  const out = new Set<string>();
  for (const ref of op.spread) {
    for (const key of Object.keys(propertiesOf(typed(ref)) ?? {})) out.add(key);
  }
  return out;
}

const EMPTY_SLOT_SET: ReadonlySet<string> = new Set<string>();

/** Every binding a state carries, with the field that named it — guards included. */
function* bindingsOf(def: LoadedState): Iterable<[string, Ref<InlineFamily>]> {
  const op = def.operation;
  if (op) for (const [name, p] of Object.entries(op.input)) if (p.binding) yield [`operation.input.${name}`, p.binding];
  for (const [name, slot] of Object.entries(def.outputs ?? {})) if (slot.binding) yield [`outputs.${name}`, slot.binding];
  for (const [name, slot] of Object.entries(def.inputs ?? {})) if (slot.binding) yield [`inputs.${name}`, slot.binding];
  for (const [key, child] of Object.entries(def.children ?? {})) {
    for (const [name, wire] of Object.entries(child.inputs ?? {})) yield [`children.${key}.inputs.${name}`, wire];
    for (const [i, t] of (child.transitions ?? []).entries()) {
      if (t.whenRef !== undefined) yield [`children.${key}.transitions[${i}].when`, t.whenRef];
    }
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
  /**
   * The `each` axes of the mount this wire is on, when it fans out (§6.2) — the one place `.each` may
   * be read, and what it is typed from. Absent everywhere else, where a `.each` read is refused.
   */
  eachAxes?: readonly string[],
): void {
  const bindingScope: ExprScope = eachAxes === undefined ? scope : { ...scope, [EACH_NAMESPACE]: eachSchema(eachAxes) };
  // Reference and reachability checks run over the WHOLE binding, once, before the type check.
  //
  // They used to live inside the expression branch of `resolverSchema`, which made them depend on
  // which node happened to be the ROOT: `shout(children.c.outputs.x) === 'y'` was checked (the root
  // is an operator) and `shout(children.c.outputs.x)` was not (the root is a call, and a call is not
  // an expression ref). Checking the binding itself has no such blind spot, and it is where a call's
  // ARGUMENTS live — they are on the ref, not on the callee's own `input`.
  for (const reference of referencePathsOf(binding)) {
    const root = reference[0]!;
    if (root === EACH_NAMESPACE) {
      // Readable only where there IS an element: the wiring of a mount that fans out. Anywhere else
      // — an output, a guard, an ordinary mount's wire — it would resolve to nothing at run time,
      // and "nothing" is the answer that gets bound silently.
      if (eachAxes === undefined) {
        errors.push({ stateId, path, message: `'.each' is only readable in the wiring of a child mount that fans out (an input marked each: true)` });
      }
      continue;
    }
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
  checkSpreads(binding, path, stateId, typeOf(stateId, def, bundle, bindingScope, reachable, errors, optOut), errors);
  const issues = checkBindingGeneric(binding, consumerSchema, hooksFor(stateId, def, bundle, bindingScope, reachable, errors, optOut), path, {
    optOut,
  });
  for (const issue of issues) errors.push({ stateId, path: issue.path, message: issue.message });
}

/**
 * What `.each` is typed as on a mount with these axes (§6.2): the element's number, and its position
 * along each `each` wire by that wire's input name — so `.each.axis.flows` on a mount whose only axis
 * is `component` is a reference to nothing, reported as one.
 */
function eachSchema(axes: readonly string[]): JsonSchema {
  const axis: Record<string, JsonValue> = {};
  for (const name of axes) axis[name] = { type: "integer" } as JsonValue;
  return { type: "object", properties: { index: { type: "integer" } as JsonValue, axis: { type: "object", properties: axis } as JsonValue } };
}

/**
 * Every DEFERRED spread argument in a binding, checked (SPEC §6.3).
 *
 * This is the half of `f(...opts)` the loader could not do. Binding a spread needs the operand's
 * KEYS, and those come from its type — computable only against a scope built over the whole loaded
 * workflow, which is to say here and not there. So the loader bound what it could read in the source
 * and left the rest on the op; this is where the rest is held to the same rules.
 *
 * Walked from the BINDING rather than reached through the producer-schema hook, for the reason the
 * reference walk above gives: a call's arguments live on the ref, and the hook sees only the callee.
 */
function checkSpreads(
  binding: Ref<InlineFamily>,
  path: string,
  stateId: string,
  typed: (ref: Ref<InlineFamily>) => JsonSchema,
  errors: ValidationIssue[],
): void {
  const walk = (node: Ref<InlineFamily>): void => {
    if (!("op" in node)) return;
    const producer = node.op;
    if (typeof producer === "string") return;
    for (const p of Object.values(producer.input)) if (p.binding) walk(p.binding);
    for (const p of Object.values(node.parameters ?? {})) if (p.binding) walk(p.binding);
    for (const ref of producer.spread ?? []) walk(ref);
    checkSpreadArguments(
      producer,
      // A lowered call's arguments are the edge's `parameters`; a slot bound on the callee itself is
      // the other spelling, and both mean the same thing here — this name already has a value.
      (name) => node.parameters?.[name] !== undefined || producer.input[name]?.binding !== undefined,
      path,
      stateId,
      typed,
      errors,
    );
  };
  walk(binding);
}

/**
 * One operation's deferred spreads, against the slots they claim to fill.
 *
 * Three rules, and the first is the one that makes the other two possible:
 *
 *  - the operand must compute to an OBJECT WITH KNOWN KEYS. A spread whose keys nobody can name is
 *    not an argument list, it is a hope — nothing could say which slot it fills, so nothing could
 *    catch it filling none of them. Refused, rather than passed through to fail at dispatch.
 *  - a key the callee has no slot for is an argument nothing reads — the same failure a written-out
 *    key gets at load, asked one phase later because that is when the key became legible.
 *  - a key already filled is a slot filled twice, which is silent at run time whichever form did it.
 *
 * Only the KEYS are strictly required, because only the keys are what binding cannot proceed without.
 * "Known keys" is a question about the type and not about how much the type constrains: `{properties:
 * {a: {}}}` names one slot whose type happens to be the unconstrained one, which is a fully computed
 * answer and a perfectly good operand. A type declaring no properties is the one that names nothing.
 *
 * Each property's type is then checked against the slot it fills by the rule `resolverSchema` already
 * applies to an inferred type meeting a declared one: an unconstrained type is unknown rather than
 * wrong, and passes.
 */
function checkSpreadArguments(
  op: Operation<InlineFamily>,
  bound: (name: string) => boolean,
  path: string,
  stateId: string,
  typed: (ref: Ref<InlineFamily>) => JsonSchema,
  errors: ValidationIssue[],
): void {
  const name = op.kind === "function" ? `'${op.functionRef}'` : "the operation";
  const err = (message: string): void => {
    errors.push({ stateId, path, message });
  };
  const slots = Object.keys(op.input);
  for (const ref of op.spread ?? []) {
    const schema = typed(ref);
    const properties = propertiesOf(schema);
    if (properties === undefined || Object.keys(properties).length === 0) {
      err(
        `a spread argument to ${name} infers to ${describeSchema(schema)}, whose keys are not known — a spread must compute to an object whose properties are declared, or nothing can say which parameters it fills`,
      );
      continue;
    }
    for (const [key, declared] of Object.entries(properties)) {
      if (slots.length > 0 && !slots.includes(key)) {
        err(`a spread argument passes '${key}', which ${name} does not accept`);
        continue;
      }
      if (bound(key)) {
        err(`${name} is passed '${key}' twice: once by name and once in a spread`);
        continue;
      }
      const want = op.input[key]?.schema;
      if (want === undefined || isUniversalSchema(want) || isUniversalSchema(declared)) continue;
      const check = isSubschema(declared as Schema, want as Schema);
      if (!check.ok) err(`a spread argument passes '${key}' as a type ${name} does not accept: ${check.reason}`);
    }
  }
}

/**
 * How this state types ONE binding — the whole producer vocabulary, not the expression subset.
 *
 * `inferRef` knows the operators an expression lowers onto; the hw-specific resolvers a BINDING
 * lowers onto — `scope.get`, `artifact.get`, a child `select` — are typed by `resolverSchema`
 * behind the shared checker's hook. A spread's operand can be either, since a call form's arguments
 * are lowered exactly as `input` bindings are, so reading its type through `inferRef` alone made
 * `f(....inputs.bag)` infer to the universal schema and report keys it could perfectly well see.
 */
function typeOf(
  stateId: string,
  def: LoadedState,
  bundle: WorkflowBundle,
  scope: ExprScope,
  reachable: Reachability,
  errors: ValidationIssue[],
  optOut: boolean,
): (ref: Ref<InlineFamily>) => JsonSchema {
  const hooks = hooksFor(stateId, def, bundle, scope, reachable, errors, optOut);
  // Issues found while typing are DISCARDED here: this walk exists to read a shape, and the binding
  // it is reading was already checked — or is about to be — by the pass that owns it. Reporting them
  // twice would double every message a spread's operand happens to contain.
  return (ref) => producerSchemaOf(ref, hooks, "", [], optOut) ?? ANY_SCHEMA;
}

/** A schema's declared `properties`, as a map — the one thing a spread's keys can be read from. */
function propertiesOf(schema: JsonSchema): Record<string, JsonSchema> | undefined {
  const raw = schema.properties;
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  return raw as Record<string, JsonSchema>;
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
  /** States whose outputs are already being inferred — see {@link outputsObjectSchema}. */
  seen?: ReadonlySet<string>,
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
      // `.each` is in the scope exactly when this expression is a fanned-out mount's wire (§6.2) —
      // the scope is what says where it may be read, so a read anywhere else is refused here.
      if (root === EACH_NAMESPACE) {
        if (!(EACH_NAMESPACE in scope)) err(`'.each' is only readable in the wiring of a child mount that fans out (an input marked each: true)`);
        continue;
      }
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
        ? producerSchemaOf(
            value.binding,
            // `seen` rides along: this recursion is how a `{ child, output }` selection reaches the
            // child's outputs, and dropping it here restarted the inference stack from empty — which
            // is an unbounded loop the moment a child mounts one of its own ancestors.
            hooksFor(stateId, def, bundle, scope, reachable, errors, optOut, seen),
            path,
            inner,
            optOut,
            value.kind,
          )
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

/**
 * A state's declared outputs as one object schema — what a producer edge on it emits.
 *
 * ## An output's type is OPTIONAL, and an untyped one takes the type of what fills it
 *
 * A declared `schema` is a CONSTRAINT: the binding filling the slot is checked against it, and a
 * consumer of the slot is checked against it in turn. Declaring none is a legitimate thing to do
 * — most outputs are a value passed straight out of a child, and restating its type is a second
 * place to keep in step with the first.
 *
 * What an undeclared one must NOT mean is "the top type". `{}` is a schema every value satisfies
 * and no typed consumer accepts, so a state that left one output undeclared could not be wired
 * into anything typed: "consumer requires type string but producer declares none", reported
 * against the parent, about a slot the parent did not write. That is the opposite of optional —
 * it made the declaration mandatory wherever the value was actually used.
 *
 * So an untyped output is INFERRED from its binding, in the state's own scope: `.inputs.seed` on a
 * state whose `seed` is a string makes the output a string, and the parent's typed slot accepts it.
 * A slot with no binding at all is filled by the operation, whose result is not statically typed,
 * and stays `{}` — genuinely unknown rather than merely undeclared.
 *
 * `seen` guards the recursion. Inferring needs the state's scope, the scope names its children's
 * outputs, and a child may mount an ancestor; a state already on the stack contributes `{}` rather
 * than looping.
 */
function outputsObjectSchema(
  state: LoadedState,
  bundle: WorkflowBundle,
  seen: ReadonlySet<string> = EMPTY_STATE_SET,
): JsonSchema | undefined {
  const outputs = state.outputs;
  if (!outputs || Object.keys(outputs).length === 0) return undefined;
  const properties: Record<string, JsonValue> = {};
  const required: string[] = [];
  // Built once, and only when something actually needs it: typing a binding costs a whole scope and
  // a hook set, and most states declare every output they have.
  let typing: { hooks: CheckerHooks<InlineFamily> } | undefined;
  const inferred = (slot: NamedParameter<InlineFamily>): JsonValue => {
    // No binding ⇒ the OPERATION fills this slot, and its result is not statically typed. `{}` there
    // is honest: unknown, not merely undeclared.
    if (slot.binding === undefined || seen.has(state.id)) return {} as JsonValue;
    if (typing === undefined) {
      const inner = new Set([...seen, state.id]);
      // A throwaway error sink and `optOut`: this is a TYPE query, not a validation pass. Whatever is
      // wrong with the binding is reported where the binding itself is checked, against the state
      // that wrote it — surfacing it a second time here would name it against every consumer.
      const discarded: ValidationIssue[] = [];
      const scope = exprScopeOf(state, bundle, inner);
      typing = { hooks: hooksFor(state.id, state, bundle, scope, reachabilityOf(state), discarded, true, inner) };
    }
    // The same function the binding CHECK computes its producer type with, so an inferred output and
    // a declared one are decided by one rule rather than two that can drift.
    return (producerSchemaOf(slot.binding, typing.hooks, "", [], true, slot.kind) ?? {}) as JsonValue;
  };
  for (const [name, slot] of Object.entries(outputs)) {
    properties[name] = slot.schema === undefined ? inferred(slot) : (slot.schema as JsonValue);
    const meta = state.slotMeta?.[`outputs.${name}`];
    if (meta?.optional !== true && meta?.default === undefined) required.push(name);
  }
  return { type: "object", properties, ...(required.length > 0 ? { required } : {}) };
}

/** The empty stack {@link outputsObjectSchema} starts from — a constant, so it is not reallocated. */
const EMPTY_STATE_SET: ReadonlySet<string> = new Set<string>();

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
  /**
   * The same analysis asked at the moment `key` is ENTERED, rather than at the state's own
   * evaluation point.
   *
   * A child mount's input wires resolve when that child is entered — so the members that run AFTER
   * it are not proven for them, however proven they are for the state's outputs and guards. Asking
   * the state-level question for a mount is what let `{ "b": { "inputs": { "x": ".children.c…" } } }`
   * lint clean with `c` declared after `b`, and then fail at run time every single time.
   */
  enteredAt(key: string): Reachability;
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
  //
  // A CHILD's transition is blocked until that child finishes whatever its guard reads — that is what
  // makes it a child transition — so its mount is a floor on when it can pre-empt anything. An
  // unguarded one written on the first sequence member still pre-empts every member after it, which is
  // exactly the diversion this analysis exists to notice.
  const lists: Array<[readonly LoadedTransition[] | undefined, number]> = [
    [def.transitions, -1],
    ...Object.entries(def.children ?? {}).map(
      ([key, child]) => [child.transitions, indexOf.get(key) ?? -1] as [readonly LoadedTransition[] | undefined, number],
    ),
  ];
  let earliest = Number.POSITIVE_INFINITY;
  for (const [list, floor] of lists) {
    for (const t of list ?? []) {
      let blockedUntil = floor;
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
  }

  const always = new Set<string>();
  sequence.forEach((key, i) => {
    // A transition that can fire at `earliest` pre-empts every member after it.
    if (def.children?.[key] && earliest >= i) always.add(key);
  });

  /**
   * Proven at the point `key` is entered: the members strictly before it, and only those that were
   * proven at all. A child OUTSIDE the sequence is entered only by a transition and could fire from
   * anywhere, so nothing precedes it provably — the same conservatism the rest of this applies.
   */
  const enteredAt = (key: string): Reachability => {
    const limit = indexOf.get(key);
    const proven =
      limit === undefined ? new Set<string>() : new Set([...always].filter((k) => (indexOf.get(k) ?? Number.POSITIVE_INFINITY) < limit));
    return { always: proven, enteredAt };
  };

  return { always, enteredAt };
}

// --- Expression scope ---------------------------------------------------------

/**
 * The operation's declared output, in the shape {@link operationNodeSchema} types `outputs` from.
 *
 * Read off the DESUGARED state, so it is the slot the loader actually built — including the one it
 * synthesizes when the author declared none. Reading the authored file instead would type the
 * namespace against a declaration that is often absent.
 */
function outputDeclOf(def: LoadedState): { name: string; kind?: string; schema?: JsonValue } | undefined {
  const output = (def as { operation?: { output?: { name?: unknown; kind?: unknown; schema?: unknown } } }).operation?.output;
  if (output === undefined || typeof output.name !== "string") return undefined;
  return {
    name: output.name,
    ...(typeof output.kind === "string" ? { kind: output.kind } : {}),
    ...(output.schema !== undefined ? { schema: output.schema as JsonValue } : {}),
  };
}

/**
 * Build the typed namespace map an expression is inferred against (§7.2/§7.5).
 *
 * `seen` is the chain of states whose outputs are already being inferred — see
 * {@link outputsObjectSchema}. It is threaded rather than reset because the recursion runs through
 * here: a child's untyped output is inferred in the CHILD's scope, which names its own children.
 */
function exprScopeOf(def: LoadedState, bundle: WorkflowBundle, seen: ReadonlySet<string> = EMPTY_STATE_SET): ExprScope {
  const objectOf = (slots: Record<string, { schema?: JsonSchema }> | undefined): JsonSchema => {
    const properties: Record<string, JsonValue> = {};
    for (const [name, slot] of Object.entries(slots ?? {})) properties[name] = (slot.schema ?? {}) as JsonValue;
    return { type: "object", properties };
  };

  // Each child exposes its `output` and its termination `outcome` — the two things the engine puts
  // in the expression context for it (SPEC §3.6) — as of every PASS it took.
  //
  // The schema is an ARRAY of that view which also declares the view's own properties, because the
  // value is: `.children.c[-2].output` indexes a pass, `.children.c.output` reads the current one,
  // and both have to check. An array schema carrying `properties` is unusual and exact, the same way
  // `operationNodeSchema` types a prompt op that returns a list with `session` hung on it.
  const childrenProps: Record<string, JsonValue> = {};
  const outcomeSchema: JsonValue = { type: "string", enum: [...TERMINATE_OUTCOMES] };
  for (const [key, child] of Object.entries(def.children ?? {})) {
    const childState = bundle.states[child.state];
    const own = childState ? (outputsObjectSchema(childState, bundle, seen) ?? ANY_SCHEMA) : ANY_SCHEMA;
    // A mount that FANS OUT (§6.2) reads back every output as an ARRAY in element order — what the
    // engine's one record per key holds — so a consumer typed against the child's own declaration
    // would be typed against a single element it will never be handed.
    const outputs = child.each !== undefined && child.each.length > 0 ? elementwise(own) : own;
    // A child's own operation node, typed by ITS operation's kind — so
    // `children.plan.operation.output.session` is checked against what `plan` actually runs, and
    // pointing it at a `ui` gate is a load-time error rather than a runtime undefined.
    const childOperation = childState
      ? operationNodeSchema(childState.operation?.kind, outputDeclOf(childState))
      : undefined;
    const passProperties: Record<string, JsonValue> = {
      output: outputs as JsonValue,
      outcome: outcomeSchema,
      ...(childOperation !== undefined ? { operation: childOperation as JsonValue } : {}),
    };
    childrenProps[key] = {
      type: "array",
      items: { type: "object", properties: passProperties } as JsonValue,
      properties: passProperties,
    } as JsonValue;
  }

  // The state's OWN operation node (SPEC.md §6.1). Absent for a pure composite, so
  // `operation.cost` there is an unresolved reference rather than an object of unknowns.
  const operation = operationNodeSchema(def.operation?.kind, outputDeclOf(def));

  const sequenceKeys = def.sequence ?? [];

  return {
    inputs: objectOf(def.inputs),
    // The state's OWN outputs, through the same reader a CHILD's go through — so an output that
    // adopts its binding's type adopts it for `.outputs.<name>` here as well. A second output
    // deriving from a first is the documented pattern, and it is a consumer like any other.
    outputs: outputsObjectSchema(def, bundle, seen) ?? { type: "object", properties: {} },
    ...(operation !== undefined ? { operation } : {}),
    children: { type: "object", properties: childrenProps },
    // Session-owned resources: addressable, contents known only at run time.
    artifacts: { type: "object" },
    // Guard-only control-flow scalars (§7.5) — never a reference binding.
    run: {
      type: "object",
      properties: {
        // Every transition taken. `iteration` counts only the backward ones — see `Instance.index`.
        index: { type: "integer" } as JsonValue,
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

/** An outputs object with every property lifted to an array of itself — a fanned-out child's view. */
function elementwise(outputs: JsonSchema): JsonSchema {
  const properties = (outputs as { properties?: Record<string, JsonValue> }).properties;
  if (properties === undefined) return outputs;
  const lifted: Record<string, JsonValue> = {};
  for (const [name, schema] of Object.entries(properties)) lifted[name] = { type: "array", items: schema } as JsonValue;
  return { ...outputs, properties: lifted } as JsonSchema;
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
