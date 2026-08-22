/**
 * Bundle loading, DESUGARING, and snapshot hashing (SPEC §2.4, §12; API.md, "Binding desugaring").
 *
 * A bundle is a map of state-ID → state-file JSON. State IDs are file paths relative to the
 * workflow root without suffix; `id` inside a file may be omitted and is derived from the path
 * (a present-but-mismatched `id` is a load error).
 *
 * Loading also DESUGARS: every authored binding form (`{ child, output }`, `{ input }`,
 * `{ expr }`, `{ artifact }`, `{ conversation }`) is lowered to a base `Ref<InlineFamily>` case
 * — a literal or a producer edge over a well-known resolver function (`RESOLVER_REFS`). After
 * this pass nothing downstream — checker, hasher, engine — knows the sugar exists.
 *
 * The snapshot hash is the workflow-version identity (SPEC §12, JaiRA DESIGN §5.3):
 * `sha256(canonicalize(sorted [(stateId, contentHash(stateFile))]))` over the transitive closure
 * of states reachable from the root. It hashes the AUTHORED file, so two spellings of the same
 * sugar hash differently but a desugaring change never invalidates a stored snapshot.
 */
import { canonicalize, hashCanonical, kindFor, sha256Hex, type InlineFamily, type JsonSchema, type JsonValue, type NamedParameter, type Operation, type Parameter, type Ref, type RefKind } from "@declarative-ai/exec";
import { computeFanOut } from "./fanout.js";
import {
  bindingForDocument,
  RESOLVER_REFS,
  type BindingDecl,
  type ChildDecl,
  type ExecEnvironmentDecl,
  type LoadedChild,
  type LoadedState,
  type LoadedTransition,
  type NamedParameterDecl,
  OPERATION_OWN_FIELDS,
  type OperationFields,
  type OutputSpread,
  type ParameterDecl,
  type SlotMeta,
  type StateDef,
  type TransitionDecl,
  type WorkflowBundle,
} from "./format.js";
import { parseExpression } from "./expr.js";
import { bindTransitionContext, lowerExpression, type LowerOptions } from "./lowerExpr.js";
import { environmentIdentity, mergeOperationFields, resolutionEnvironment } from "./merge.js";
import { resolveStateRef, StateRefError, type StateRefOptions } from "./ref.js";
import { expandReferences } from "./expand.js";
import { isDataFile, isRuntimeReference, MODULE_EXTENSIONS, parseReferencedFile, resolveReference, selectProperty, type Vfs } from "./reference.js";
import type { SymbolIndex } from "./reference.js";
import type { UserFunctions } from "./userFunctions.js";

export class WorkflowLoadError extends Error {
  constructor(
    message: string,
    readonly stateId?: string,
  ) {
    super(stateId ? `${stateId}: ${message}` : message);
    this.name = "WorkflowLoadError";
  }
}

/**
 * Strip a state-file name to its state ID: forward slashes, no data suffix.
 *
 * A state is authored in JSON or YAML, and the two are interchangeable — both parse to the same
 * value tree, and `snapshotHash` hashes that value rather than the bytes, so one workflow has one
 * identity whichever it was written in.
 */
/** True for a js/ts module — a file whose value is its export rather than its text. */
function isModuleFile(file: string): boolean {
  const ext = file.split("/").pop()?.split(".").pop()?.toLowerCase();
  return ext !== undefined && (MODULE_EXTENSIONS as readonly string[]).includes(ext);
}

export function stateIdFromPath(relPath: string): string {
  return relPath.replace(/\\/g, "/").replace(/\.state\.json$|\.(json|yaml|yml)$/i, "");
}

// --- Desugaring (§2.1) --------------------------------------------------------

/**
 * The kind a slot carries when the author didn't say — DELEGATED to `kindFor`, never re-derived here.
 *
 * This used to be a local copy of the rule that only knew `text` vs `json`, which is exactly the drift
 * `kindFor` exists to prevent: an artifact slot authored with JSON Schema's own content keywords (the
 * thing §7 replaced the bespoke `x-artifact` marker with) loaded as `text`, the engine's artifact
 * registration gates on `kind === "blob"`, and so the artifact silently never existed.
 */
function kindOf(decl: { kind?: RefKind; schema?: JsonSchema }): RefKind {
  return decl.kind ?? kindFor(decl.schema);
}

/** A producer edge on an embedded resolver `FunctionOp` — the shape EVERY sugar lowers to. */
function resolverEdge(functionRef: string, args: Record<string, Ref<InlineFamily>>, outKind: RefKind = "json"): Ref<InlineFamily> {
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, binding] of Object.entries(args)) {
    input[name] = { kind: "text" in binding ? "text" : "json", binding };
  }
  return {
    op: { kind: "function", functionRef, input, output: { name: "value", kind: outKind } },
  };
}

/** True for the base `Ref<InlineFamily>` cases — everything else in `BindingDecl` is sugar. */
function isBaseRef(b: Exclude<BindingDecl, string>): b is Ref<InlineFamily> {
  return "text" in b || "json" in b || "result" in b || "refs" in b || "op" in b;
}

/**
 * Lower ONE authored binding to a base `Ref<InlineFamily>`. The mapping is §2.1's table:
 * every sugar becomes a producer edge (or a literal), so the base vocabulary stays closed.
 */
export function desugarBinding(
  binding: BindingDecl,
  where: string,
  stateId: string,
  /** The slot this binding fills — the default for a `{ child }` binding's `output`. */
  slotName?: string,
  /** How an expression resolves an operation NAME to the operation (§3). */
  lower: LowerOptions = {},
): Ref<InlineFamily> {
  if (typeof binding === "string") {
    // A leading-dot path is this instance's data and has direct lowerings worth keeping (`.inputs.x`
    // is `scope.get`, not `context.get` — a missing input REFUSES rather than reading `undefined`).
    // Everything else is an expression: a bare path cannot reach here, because expansion resolved it.
    if (isRuntimeReference(binding)) return desugarRuntimeReference(binding, where, stateId, slotName);
    return desugarExpression(binding, where, stateId, lower);
  }
  if (isBaseRef(binding)) return binding;
  // An operation document — what a reference in binding position resolves to when it names an
  // operation. §3.1's first tier: the operation itself, as a value, with nothing applied to it.
  if (isOperationDecl(binding)) return { op: desugarOperation(binding as OperationFields, stateId) };

  if ("expr" in binding) return desugarExpression(binding.expr, where, stateId, lower);
  throw new WorkflowLoadError(`${where}: unrecognized binding form ${JSON.stringify(binding)}`, stateId);
}

/**
 * An operation document, as distinct from a binding: `kind` says which call it is.
 *
 * Safe to test by that key alone — no binding form carries a bare `kind`, since a `Parameter`'s
 * `kind` lives on the slot rather than on the binding inside it.
 */
function isOperationDecl(binding: object): boolean {
  const kind = (binding as { kind?: unknown }).kind;
  return kind === "prompt" || kind === "function";
}

/**
 * Parse and lower one expression — the single place a source string becomes a producer tree.
 *
 * An expression IS a producer, structurally, as a TREE of operator edges rather than a source string
 * handed to an interpreter at resolution time (EXPRESSIONS.md §1). Its references are leaves, so the
 * fan-out planner and the validator walk it with the code they already walk every other binding
 * with, and its type is inferred by `inferRef` rather than by re-parsing.
 */
function desugarExpression(source: string, where: string, stateId: string, lower: LowerOptions): Ref<InlineFamily> {
  try {
    return lowerExpression(parseExpression(source), lower);
  } catch (e) {
    // A malformed expression is a binding error like any other here — the same treatment
    // `unrecognized binding form` already gets.
    throw new WorkflowLoadError(`${where}: expression does not parse: ${(e as Error).message}`, stateId);
  }
}

/** The child a spread republishes — `.children.<key>.outputs`, the whole object it fans out. */
function spreadChildOf(binding: BindingDecl | undefined): string | undefined {
  if (typeof binding !== "string") return undefined;
  const parts = binding.split(".");
  return parts.length === 4 && parts[0] === "" && parts[1] === "children" && parts[3] === "outputs" ? parts[2] : undefined;
}

/**
 * Lower a RUNTIME reference — `.children.critique.outputs.outcome` — to a producer edge
 * (REFERENCES.md §5).
 *
 * This is the ONLY spelling. `{ child }`, `{ input }`, `{ artifact }` and `{ conversation }` each
 * said one of these separately and were kept alongside it while workflows migrated; they are gone,
 * so there is one way to name a runtime value and no table mapping five spellings onto it.
 */
function desugarRuntimeReference(reference: string, where: string, stateId: string, slotName?: string): Ref<InlineFamily> {
  const path = reference.slice(1).split(".");
  const [namespace, ...rest] = path;
  const bad = (why: string): never => {
    throw new WorkflowLoadError(`${where}: '${reference}' ${why}`, stateId);
  };

  switch (namespace) {
    case "inputs": {
      if (rest.length !== 1) bad("must name exactly one input, as '.inputs.<name>'");
      return resolverEdge(RESOLVER_REFS.scope, { scope: { text: "inputs" }, name: { text: rest[0]! } });
    }
    case "outputs": {
      // This state's own outputs are only reachable by evaluation, which is what `expr` is.
      if (rest.length === 0) bad("must name an output");
      return desugarBinding({ expr: `.outputs.${rest.join(".")}` }, where, stateId);
    }
    case "children": {
      const [child, section, ...tail] = rest;
      if (child === undefined) bad("must name a child, as '.children.<key>.outputs.<name>'");
      if (section === undefined || section === "outputs") {
        // `.children.c.outputs` is the whole object; `.children.c.outputs.x` projects one output.
        // hw states lower to single-object-output ops, so a named output IS a property select.
        const childEdge: Ref<InlineFamily> = { op: child! };
        if (tail.length === 0) return childEdge;
        return resolverEdge(RESOLVER_REFS.select, { value: childEdge, key: { text: tail.join(".") } });
      }
      // `outcome` and anything else about a child is control-flow state, which guards read.
      return desugarBinding({ expr: `.children.${child}.${[section, ...tail].join(".")}` }, where, stateId);
    }
    case "artifacts": {
      if (rest.length !== 1) bad("must name exactly one artifact, as '.artifacts.<name>'");
      return resolverEdge(RESOLVER_REFS.artifact, { name: { text: rest[0]! } });
    }
    case "operation": {
      // The state's OWN call, as data: what it returned (`.operation.output.<name>`), and what the
      // engine measured about it (`outcome`, `cost`, `model`, `usage`).
      //
      // Evaluated rather than lowered to a producer edge, exactly as `.children.<key>.outcome` is.
      // The operation is not a node in the producer graph — it runs AT this state rather than being
      // resolved for it — so there is no edge to point at; the value is read off the instance once
      // the call has completed. `operationNodeSchema` types it, so a name the call never returns is
      // still a load-time error and not a silent undefined.
      if (rest.length === 0) bad("must name something on the operation, as '.operation.output.<name>'");
      return desugarBinding({ expr: `.operation.${rest.join(".")}` }, where, stateId);
    }
    default:
      return bad(
        `starts with '${String(namespace)}', which is not a runtime namespace — ` +
          `expected inputs, outputs, operation, children or artifacts — a conversation is read with ` +
          `messages(<session ref>), since a session is a position and not a name`,
      );
  }
}

/**
 * The `output` value meaning "the child's whole output object", and the slot-key suffix meaning
 * "spread the child's outputs into this state's, under this prefix" (§3.4).
 *
 * The spread marker lives in the slot KEY rather than in `output` because a spread declares N slots,
 * not one, and that has to be visible where the slots are declared. Overloading `output` for it
 * would also be ambiguous: `{ "output": "plan_" }` could not be told apart from selecting an output
 * genuinely named `plan_`.
 */
export const WHOLE_OUTPUT = "*";
export const SPREAD_SUFFIX = "*";

/** Lower an authored slot to a `Parameter<InlineFamily>`, splitting off its authoring metadata. */
function desugarParameter(
  decl: ParameterDecl,
  where: string,
  stateId: string,
  slotName?: string,
  lower: LowerOptions = {},
): { param: Parameter<InlineFamily>; meta?: SlotMeta } {
  const param: Parameter<InlineFamily> = { kind: kindOf(decl) };
  if (decl.schema !== undefined) param.schema = decl.schema;
  if (decl.binding !== undefined) param.binding = desugarBinding(decl.binding, where, stateId, slotName, lower);
  if (decl.index !== undefined) param.index = decl.index;
  const meta: SlotMeta = {};
  if (decl.default !== undefined) meta.default = decl.default;
  if (decl.optional !== undefined) meta.optional = decl.optional;
  if (decl.description !== undefined) meta.description = decl.description;
  return { param, ...(Object.keys(meta).length > 0 ? { meta } : {}) };
}

function desugarNamedParameter(name: string, decl: NamedParameterDecl, where: string, stateId: string, slotName?: string, lower: LowerOptions = {}): { param: NamedParameter<InlineFamily>; meta?: SlotMeta } {
  const { param, meta } = desugarParameter(decl, where, stateId, slotName, lower);
  return { param: { ...param, name: decl.name ?? name }, ...(meta ? { meta } : {}) };
}

function desugarSlotMap(
  section: string,
  fields: Record<string, ParameterDecl> | undefined,
  stateId: string,
  slotMeta: Record<string, SlotMeta>,
  lower: LowerOptions = {},
): Record<string, Parameter<InlineFamily>> | undefined {
  if (!fields) return undefined;
  const out: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, decl] of Object.entries(fields)) {
    const { param, meta } = desugarParameter(decl, `${section}.${name}`, stateId, undefined, lower);
    out[name] = param;
    if (meta) slotMeta[`${section}.${name}`] = meta;
  }
  return out;
}

/** The default output slot of a state operation: one object carrying the state's declared outputs. */
function defaultOutput(): NamedParameter<InlineFamily> {
  return { name: "output", kind: "json" };
}

/**
 * Split a merged operation into the part the engine EXECUTES and the part it executes it IN (§5).
 *
 * Both halves are authored in one block now, but they are consumed by different machinery — the op
 * goes to the prompt runner or the function registry, the environment decides the session, the tool
 * set and the permission baseline — so the loader hands each consumer only what it needs.
 */
export function splitExecEnvironment(fields: OperationFields): { op: OperationFields; env: ExecEnvironmentDecl } {
  const { session, fork, tools, conversation, permissions, ...op } = fields;
  const env: ExecEnvironmentDecl = {};
  // `session` is tested against `undefined` rather than for truthiness because `null` is a REAL
  // declaration — "start fresh, whatever the chain said" — and dropping it here would silently
  // restore the inherited session the author was opting out of.
  if (session !== undefined) env.session = session;
  if (fork !== undefined) env.fork = fork;
  if (tools !== undefined) env.tools = tools;
  if (conversation !== undefined) env.conversation = conversation;
  if (permissions !== undefined) env.permissions = permissions;
  return { op, env };
}

/**
 * Lower a MERGED operation block to a real `Operation<InlineFamily>` (§7.1).
 *
 * `decl` is post-inheritance (§5), so `kind` and `function` may have come from an ancestor's
 * `environment` rather than from the state file — which is why the "did the author say enough to
 * build an operation?" checks live here rather than in the type.
 */
export function desugarOperation(
  decl: OperationFields,
  stateId: string,
  outputs?: Record<string, NamedParameterDecl>,
  userFunctions?: UserFunctions,
): Operation<InlineFamily> {
  // An EMBEDDED BODY (SPEC §7.5.1, form 2). The document declares its slots the way a state does and
  // supplies js/ts; the wrapper's parameters are those slots in `positionalOrder`, so a call binds
  // against exactly what the author declared. From here down it is a module like any other.
  if (decl.body !== undefined) {
    if (userFunctions === undefined) {
      throw new WorkflowLoadError(
        "operation declares a 'body', but this loader was given no way to compile js/ts",
        stateId,
      );
    }
    if (decl.function !== undefined) {
      throw new WorkflowLoadError(
        "operation declares both a 'body' and a 'function' — exactly one says what it runs",
        stateId,
      );
    }
    return userFunctions.operationForBody(stateId, decl.body, decl.input ?? {}).operation;
  }
  if (decl.kind === undefined) {
    throw new WorkflowLoadError(
      "operation declares no 'kind', and no ancestor's environment supplies one — expected 'prompt' or 'function'",
      stateId,
    );
  }
  if (decl.kind !== "prompt" && decl.kind !== "function") {
    throw new WorkflowLoadError(`operation.kind '${String(decl.kind)}' is not 'prompt' or 'function'`, stateId);
  }
  if (decl.kind === "function" && decl.function === undefined) {
    throw new WorkflowLoadError(
      "function operation names no 'function', and no ancestor's environment supplies one",
      stateId,
    );
  }
  const input: Record<string, Parameter<InlineFamily>> = {};
  for (const [name, p] of Object.entries(decl.input ?? {})) {
    input[name] = desugarParameter(p, `operation.input.${name}`, stateId).param;
  }

  // The op's output, in precedence order: the single slot the author declared outright (the only way
  // to say "the whole return value is a blob"), then the operation's OWN `outputs` map, then — for a
  // state that declares neither — the state's unbound outputs.
  //
  // The middle case is the one that matters. An operation that names what it returns owns its own
  // signature, so the model's structured-output contract comes from the call rather than from
  // whatever the state around it happens to publish, and `.operation.output.<name>` has names to
  // expose. The last case is the older rule, kept for states that still lean on it.
  const output = decl.output
    ? desugarNamedParameter("output", decl.output, "operation.output", stateId).param
    : outputSlotFor(decl.outputs);

  if (decl.kind === "prompt") {
    // The template's `{{.inputs.*}}` scope IS the operation's resolved inputs (§3.1: authored render
    // variables ride bound input slots, never a field on the op shape), so there is nothing to merge
    // in here — every render variable is just one of `input`.
    const op: Operation<InlineFamily> = {
      kind: "prompt",
      user: decl.prompt ?? "",
      config: callConfigOf(decl) as JsonValue,
      input,
      output,
    };
    if (decl.system !== undefined) op.system = decl.system;
    return op;
  }
  // A FunctionOp — a host function, a sub-workflow, or a delegated runtime adapter alike (§3.1).
  return { kind: "function", functionRef: decl.function!, input: withArgs(input, decl.args), output };
}

/**
 * Bind an operation's authored `args` to its input slots, BY NAME.
 *
 * They used to be shoved whole into one slot literally called `config`, so an impl read
 * `inputs.config.mode` rather than `inputs.mode` and the op's shape said nothing about what the call
 * actually passes. That was the only thing it could do while a registered function had no way to
 * declare named parameters: with no slots to bind to, the blob was the whole of what the op could
 * carry. Now that a signature declares slots (`EntrySignature`), an argument has a name to arrive
 * under, and the checker has two sides to compare — a parameter the impl does not accept, and one it
 * requires that nobody passes, both of which the `config` blob made unaskable.
 *
 * A slot the author DECLARED wins over an `args` entry of the same name. The overlap is ordinary
 * rather than a mistake: `args` merges per key down the environment chain (§7.1a), so an ancestor's
 * default and a state's own typed slot routinely name the same thing, and the typed one is the more
 * specific statement of the two.
 *
 * A string binds as `text` and everything else as `json`, which is the same call `parametersFor`
 * makes for an expression's arguments — one rule for how a literal argument reaches a slot, rather
 * than two that could disagree about the kind of `"plan"`.
 */
function withArgs(
  input: Record<string, Parameter<InlineFamily>>,
  args: Record<string, JsonValue> | undefined,
): Record<string, Parameter<InlineFamily>> {
  if (args === undefined) return input;
  for (const [name, value] of Object.entries(args)) {
    if (input[name] !== undefined) continue;
    input[name] = typeof value === "string" ? { kind: "text", binding: { text: value } } : { kind: "json", binding: { json: value } };
  }
  return input;
}

/**
 * The call configuration a prompt operation assembles: every authored field hw does not own itself.
 *
 * "The operation IS the call" (REFERENCES.md §7.2) — `model`, `temperature` and the rest sit
 * directly on the operation rather than nested under a `config` bag, so an author writes one flat
 * block and a knob hw has never heard of still reaches the model.
 */
function callConfigOf(decl: OperationFields): Record<string, JsonValue> {
  const config: Record<string, JsonValue> = {};
  for (const [key, value] of Object.entries(decl)) {
    if (!OPERATION_OWN_FIELDS.has(key) && value !== undefined) config[key] = value as JsonValue;
  }
  return config;
}

/**
 * Build the operation's single object-output slot from the names it returns.
 *
 * The executor seam takes ONE output — a prompt op asks the model for one object — so a map of named
 * returns is lowered into an object schema here. This is what a prompt operation's structured-output
 * contract IS.
 *
 * Built from the OPERATION's own `outputs` and nothing else. It used to fall back to the state's
 * unbound outputs, so a call borrowed its signature from whatever the state around it happened to
 * publish — which is why its result had no address and could only be received, never renamed. A
 * state now says what it publishes and the operation says what it returns, with a binding between
 * them.
 */
function outputSlotFor(outputs: Record<string, NamedParameterDecl> | undefined): NamedParameter<InlineFamily> {
  const produced = Object.entries(outputs ?? {});
  if (produced.length === 0) return defaultOutput();
  const properties: Record<string, JsonValue> = {};
  const required: string[] = [];
  for (const [name, decl] of produced) {
    // An unconstrained slot (no schema) constrains nothing; the checker treats it as universal.
    properties[name] = (decl.schema ?? {}) as JsonValue;
    if (decl.optional !== true && decl.default === undefined) required.push(name);
  }
  const schema: JsonSchema = { type: "object", properties, ...(required.length > 0 ? { required } : {}) };
  return { name: "output", kind: "json", schema };
}

/**
 * Desugar one authored state file into its loaded form.
 *
 * `inherited` is the merged `environment` of every ancestor on the path that reached this state
 * (§5) — empty for a root, and for any caller that loads a state in isolation.
 */
export function desugarState(
  id: string,
  def: StateDef,
  inherited: OperationFields = {},
  refs: StateRefOptions = {},
  inferredChildren?: Record<string, ChildDecl>,
  /** How an expression in THIS state resolves an operation name (§3) — the loader supplies it. */
  lower: LowerOptions = {},
): LoadedState {
  if (def.children === undefined && inferredChildren !== undefined && Object.keys(inferredChildren).length > 0) {
    def = { ...def, children: inferredChildren };
  }
  // The environment this state resolves in, computed BEFORE anything below is desugared. Every
  // reference in the slots, the wiring and the children resolves against it, so it cannot be
  // assembled at the point the operation is (which is where it used to be) — see
  // `resolutionEnvironment`.
  const environment = resolutionEnvironment(inherited, def.environment);
  const slotMeta: Record<string, SlotMeta> = {};
  const inputs = desugarSlotMap("inputs", def.inputs, id, slotMeta, lower);

  let outputs: Record<string, NamedParameter<InlineFamily>> | undefined;
  // A `prefix*` output declares however many slots the child has, so it cannot be lowered until that
  // child is loaded. Collected here, expanded by the bundle once the closure is known (§3.4).
  const spreads: OutputSpread[] = [];
  if (def.outputs) {
    outputs = {};
    for (const [name, decl] of Object.entries(def.outputs)) {
      if (name.endsWith(SPREAD_SUFFIX)) {
        const child = spreadChildOf(decl.binding);
        if (child === undefined) {
          throw new WorkflowLoadError(
            `outputs.${name}: a '${SPREAD_SUFFIX}' output must bind a child's outputs — '.children.<key>.outputs'`,
            id,
          );
        }
        spreads.push({ prefix: name.slice(0, -SPREAD_SUFFIX.length), child, ...(decl.optional !== undefined ? { optional: decl.optional } : {}) });
        continue;
      }
      const { param, meta } = desugarNamedParameter(name, decl, `outputs.${name}`, id, name, lower);
      outputs[name] = param;
      if (meta) slotMeta[`outputs.${name}`] = meta;
    }
  }

  // Guards, lowered HERE rather than by the engine: a guard may CALL an operation, and resolving a
  // callee needs the search path, the referring state and a filesystem — load-time knowledge. Engine-
  // side lowering worked while expressions were operators only, and threw the moment one held a call.
  //
  // One function for both lists: a child's transitions are the state's, narrowed to when that child
  // finishes (§3.3), and lowering them differently would make the same guard mean two things.
  const lowerTransitions = (list: TransitionDecl[] | undefined): LoadedTransition[] | undefined =>
    list?.map((t) => {
      if (t.when === undefined) return { ...t };
      try {
        return { ...t, whenRef: bindTransitionContext(lowerExpression(parseExpression(t.when), lower), t.to) };
      } catch (e) {
        // Carried, not thrown — see `LoadedTransition.whenError`.
        return { ...t, whenError: (e as Error).message };
      }
    });

  let children: Record<string, LoadedChild> | undefined;
  if (def.children) {
    children = {};
    for (const [key, child] of Object.entries(def.children)) {
      const wired: Record<string, Ref<InlineFamily>> = {};
      for (const [inputName, binding] of Object.entries(child.inputs ?? {})) {
        wired[inputName] = desugarBinding(binding, `children.${key}.inputs.${inputName}`, id, undefined, lower);
      }
      children[key] = {
        // A child that declares no `state` is the one its KEY names (REFERENCES.md §7.3), so the
        // common case says the path nowhere — and a declared child differs from an inferred one
        // (§6) only by the wiring it adds.
        state: resolveChildRef(child.state ?? `./${key}`, id, key, refs),
        ...(child.inputs ? { inputs: wired } : {}),
        ...(child.async !== undefined ? { async: child.async } : {}),
        // Carried, never merged here: this layer belongs to the CHILD's chain, and folding it into
        // the parent's own environment would apply it to the parent's operation too (§5).
        ...(child.environment !== undefined ? { environment: child.environment } : {}),
        ...(child.transitions !== undefined ? { transitions: lowerTransitions(child.transitions) } : {}),
      };
    }
  }

  const transitions = lowerTransitions(def.transitions);

  // The effective operation (§5): the resolution environment above, then the op itself. Only a
  // state that DECLARES an operation gets one — otherwise every pure composite under an
  // `environment`-declaring root would inherit its ancestor's op and start running it.
  const merged = def.operation !== undefined ? mergeOperationFields(environment, def.operation) : undefined;
  const split = merged !== undefined ? splitExecEnvironment(merged) : undefined;

  // The cursor's order, defaulted to the order the children were declared in (§6). Resolved here
  // rather than in the engine so the validator's reachability pass and the lint surface see the same
  // spine the engine will walk.
  const sequence = def.sequence ?? (children ? Object.keys(children) : undefined);

  // An operation the merged chain never completed is an AUTHORING error, so it is carried as data
  // and reported by the validator alongside every other one. Throwing here instead aborted the load
  // at the first such state, and the author saw a downstream consequence — "this leaf has no kind" —
  // in place of the mistake they actually made two files away.
  const operationOrError = describeOperation(split, id, def.outputs, lower.userFunctions);

  const { operation, environment: _e, inputs: _i, outputs: _o, children: _c, sequence: _s, transitions: _t, ...rest } = def;
  return {
    ...rest,
    id,
    ...(inputs ? { inputs } : {}),
    ...(outputs ? { outputs } : {}),
    ...(children ? { children } : {}),
    ...(transitions ? { transitions } : {}),
    ...(sequence ? { sequence, sequenceAuthored: def.sequence !== undefined } : {}),
    ...(operationOrError.operation !== undefined ? { operation: operationOrError.operation } : {}),
    ...(operationOrError.error !== undefined ? { operationError: operationOrError.error } : {}),
    ...(split && Object.keys(split.env).length > 0 ? { environment: split.env } : {}),
    // The session this state's SUBTREE resolves in, recorded separately from `environment` because
    // `environment` exists only on a state that declares an OPERATION. A pure composite that
    // declares `environment.session` — the ordinary way to give a whole subtree one session — would
    // otherwise carry no trace of it, and the engine would key its children's resource bundle on the
    // run instead of on the name the author wrote (DESIGN.md §1.6).
    ...("session" in environment ? { scopeSession: environment.session } : {}),
    ...(spreads.length > 0 ? { outputSpreads: spreads } : {}),
    ...(Object.keys(slotMeta).length > 0 ? { slotMeta } : {}),
  };
}

/**
 * Expand a state's `prefix*` outputs, now that its children are loaded (§3.4).
 *
 * Each of the named child's declared outputs becomes an output of THIS state, prefixed and bound to
 * it — so a parent can republish a child's whole result without restating every slot, and still get
 * one typed slot per value rather than one opaque object.
 */
function expandOutputSpreads(state: LoadedState, states: Record<string, LoadedState>): void {
  if (state.outputSpreads === undefined) return;
  const outputs = state.outputs ?? {};
  const slotMeta = state.slotMeta ?? {};
  for (const spread of state.outputSpreads) {
    const childRef = state.children?.[spread.child];
    const childState = childRef ? states[childRef.state] : undefined;
    // An unknown child is a VALIDATION error with the field attached; expanding nothing keeps the
    // load going so the validator can say so properly.
    if (!childState) continue;
    for (const [name, slot] of Object.entries(childState.outputs ?? {})) {
      const target = `${spread.prefix}${name}`;
      // An explicitly declared slot wins over a spread — the author named that one on purpose.
      if (outputs[target] !== undefined) continue;
      outputs[target] = {
        name: target,
        kind: slot.kind,
        ...(slot.schema !== undefined ? { schema: slot.schema } : {}),
        binding: desugarBinding(`.children.${spread.child}.outputs.${name}`, `outputs.${spread.prefix}${SPREAD_SUFFIX}`, state.id),
      };
      const childMeta = childState.slotMeta?.[`outputs.${name}`];
      const optional = spread.optional ?? childMeta?.optional;
      if (optional !== undefined) slotMeta[`outputs.${target}`] = { optional };
    }
  }
  state.outputs = outputs;
  if (Object.keys(slotMeta).length > 0) state.slotMeta = slotMeta;
}

/** Lower the merged operation, keeping an incomplete one as a reportable error rather than a throw. */
function describeOperation(
  split: { op: OperationFields } | undefined,
  id: string,
  outputs: Record<string, NamedParameterDecl> | undefined,
  userFunctions?: UserFunctions,
): { operation?: Operation<InlineFamily>; error?: string } {
  if (split === undefined) return {};
  try {
    return { operation: desugarOperation(split.op, id, outputs, userFunctions) };
  } catch (e) {
    if (e instanceof WorkflowLoadError) return { error: e.message.replace(`${id}: `, "") };
    throw e;
  }
}

/** Resolve a `children[].state` reference, reporting it against the field that named it. */
function resolveChildRef(ref: string, parentId: string, key: string, refs: StateRefOptions): string {
  try {
    return resolveStateRef(ref, { ...refs, from: parentId });
  } catch (e) {
    if (e instanceof StateRefError) throw new WorkflowLoadError(`children.${key}.state: ${e.message}`, parentId);
    throw e;
  }
}

// --- Bundle loading -----------------------------------------------------------

export interface LoadBundleOptions extends Omit<StateRefOptions, "defaultRoot"> {
  /**
   * Where a bare reference hangs off — one root, or an ordered SEARCH PATH (EXPRESSIONS.md §4).
   *
   * A list is searched in order for a *document* reference, where a filesystem is in hand to test a
   * candidate against. A `children[].state` does not search — state-id resolution is pure path
   * arithmetic with nothing to check existence with — but it does fold a rooted or absolute spelling
   * back against EVERY entry, so `$BASE/lib/review` and a project's own `lib/review` name one state
   * rather than two.
   */
  defaultRoot?: string | readonly string[];
  /**
   * Whether a bare reference matching at more than one path entry is worth reporting. `"override"`
   * suits a caller that layers roots deliberately — see `ReferenceOptions.shadowing`.
   */
  shadowing?: "warn" | "override";
  /** The ordered layer roots a bare `$` searches — see `ReferenceOptions.rootPath`. */
  rootPath?: readonly string[];
  /**
   * Fetch a state the `files` map does not hold, by canonical id — how an out-of-tree reference
   * (`/opt/workflows/lib/review`, `$JAIRA/shared/x`) is read. Sync, because loading is; returning
   * `undefined` leaves the reference to the validator to report as unknown.
   */
  loadState?: (id: string) => unknown | undefined;
  /**
   * The filesystem document references resolve against (REFERENCES.md §1.1). Absent ⇒ expansion is
   * skipped, which is what an in-memory bundle with no references wants.
   */
  vfs?: Vfs;
  /**
   * Operation documents the HOST ships, by bare name — searched only where the path finds nothing.
   *
   * A call in an expression names a document (§3), and its `input` declarations ARE the signature its
   * positional arguments bind against (§3.3). That is exactly right for an operation somebody wrote
   * in a file, and it leaves a host-provided function with nowhere to declare itself: registering the
   * IMPLEMENTATION says what it does and nothing about how it is called.
   *
   * So a host may ship the document too. Consulted LAST, so a project file of the same name shadows
   * it — the same precedence a base-layer document has, and for the same reason: what the host ships
   * is a default, not a reservation.
   */
  documents?: Readonly<Record<string, unknown>>;
  /**
   * How a dotted symbol resolves to the MODULE contributing it (SPEC §7.5.2).
   *
   * Built by `createSymbolIndex`, which is asynchronous once and synchronous thereafter — which is
   * what lets a sync loader consult it. Absent ⇒ no module contributes anything, which is the
   * behavior that predates the feature.
   */
  symbols?: SymbolIndex;
  /**
   * Parsed documents, shared across every reference this load resolves.
   *
   * Named apart from {@link LoadBundleOptions.documents}, which is the host's SHIPPED operation
   * documents — two different things that would otherwise collide on one word.
   */
  documentCache?: Map<string, unknown>;
  /**
   * The bridge from a resolved js/ts symbol to an operation (SPEC §7.5).
   *
   * Absent ⇒ a module callee is an authoring error rather than silently unresolvable, because a
   * loader that can FIND a symbol and cannot type it would report "not an operation document" about
   * a file that is perfectly good.
   */
  userFunctions?: UserFunctions;
  /** Non-fatal ambiguities from reference resolution (§9). */
  onWarn?: (message: string) => void;
  /** Every file expansion read, for the snapshot closure (§8.1). */
  onReferencedFile?: (file: string) => void;
}

/**
 * Load a bundle from raw file contents (`stateId or relative path` → parsed JSON), desugaring
 * each state. Restricts the bundle to the transitive closure reachable from `rootId` so the
 * snapshot hash never varies with unrelated files lying around the workflow dir.
 *
 * The walk carries the ENVIRONMENT CHAIN (§5): each state's effective operation is merged from every
 * ancestor's `environment` on the path that reached it. That is a property of the path, not of the
 * file — the tree convention is only a convention, so a shared library state can be mounted under
 * two different parents. When those two parents would give it two DIFFERENT environments the load
 * fails rather than picking one, because either choice would be silently wrong for the other mount.
 */
export function loadBundle(files: Record<string, unknown>, rootRef: string, options: LoadBundleOptions = {}): WorkflowBundle {
  // `resolveStateRef` does path arithmetic with no filesystem in hand, so it cannot SEARCH a path —
  // it has nothing to test a candidate's existence against. It is given the whole path anyway,
  // because folding a rooted spelling back to a bare id needs no oracle: `$BASE/lib/review` is
  // `lib/review` whether or not the file is there. Searching for a `children[].state` would need an
  // existence oracle and remains a separate decision.
  const refs: StateRefOptions = {
    ...(options.defaultRoot !== undefined ? { defaultRoot: options.defaultRoot } : {}),
    ...(options.roots !== undefined ? { roots: options.roots } : {}),
    ...(options.rootPath !== undefined ? { rootPath: options.rootPath } : {}),
  };
  const authored = new Map<string, StateDef>();
  for (const [key, raw] of Object.entries(files)) {
    const id = resolveStateRef(stateIdFromPath(key), refs);
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
      throw new WorkflowLoadError("state file is not a JSON object", id);
    }
    const def = raw as StateDef;
    if (def.id !== undefined && resolveStateRef(def.id, refs) !== id) {
      throw new WorkflowLoadError(`declared id '${def.id}' does not match path-derived id '${id}'`, id);
    }
    authored.set(id, def);
  }
  const rootId = resolveStateRef(rootRef, refs);

  const rawById = new Map<string, StateDef>(authored);

  /**
   * Expand one state's document references (REFERENCES.md §4), or pass it through untouched when
   * the caller supplied no filesystem — an in-memory bundle has nothing to resolve against.
   */
  const expandFor = (id: string, def: StateDef, path?: readonly string[]): StateDef => {
    if (options.vfs === undefined) return def;
    // The inherited search path, when the chain declared one, replaces the single default root for
    // this state's bare references (EXPRESSIONS.md §4). The primary root stays first, so an id under
    // it keeps its bare spelling and the snapshot identity is unaffected.
    const defaultRoot = path !== undefined && path.length > 0 ? path : options.defaultRoot;
    try {
      return expandReferences(def, {
        vfs: options.vfs,
        from: id,
        ...(defaultRoot !== undefined ? { defaultRoot } : {}),
        ...(options.roots !== undefined ? { roots: options.roots } : {}),
        ...(options.rootPath !== undefined ? { rootPath: options.rootPath } : {}),
        ...(options.onWarn !== undefined ? { onWarn: options.onWarn } : {}),
        ...(options.shadowing !== undefined ? { shadowing: options.shadowing } : {}),
        ...(options.onReferencedFile !== undefined ? { onRead: options.onReferencedFile } : {}),
      }) as StateDef;
    } catch (e) {
      throw new WorkflowLoadError((e as Error).message, id);
    }
  };

  /**
   * Children by DIRECTORY, for a state that declares none (§6).
   *
   * A state owns the namespace under its own id, so the states one segment below it are its
   * children — `feature/plan/goals` and `feature/plan/context` are `feature/plan`'s, keyed by
   * basename, alphabetically. The bundle's own file map IS the directory listing, so this needs no
   * filesystem and gives the same answer whether it is reading live `workflows/` or a snapshot.
   *
   * Only an ABSENT `children` infers. `"children": {}` is an author saying "none", and a leaf that
   * happens to have a directory beside it must be able to say so.
   */
  const childrenByParent = new Map<string, Record<string, ChildDecl>>();
  for (const childId of [...authored.keys()].sort()) {
    const cut = childId.lastIndexOf("/");
    if (cut <= 0) continue;
    const parent = childId.slice(0, cut);
    if (!authored.has(parent)) continue;
    const bucket = childrenByParent.get(parent) ?? {};
    bucket[childId.slice(cut + 1)] = { state: childId };
    childrenByParent.set(parent, bucket);
  }

  /** Read a state the files map doesn't hold — the out-of-tree escape hatch. */
  const fetch = (id: string): StateDef | undefined => {
    const known = rawById.get(id);
    if (known !== undefined) return known;
    const raw = options.loadState?.(id);
    if (raw === null || raw === undefined) return undefined;
    if (typeof raw !== "object" || Array.isArray(raw)) throw new WorkflowLoadError("state file is not a JSON object", id);
    const def = raw as StateDef;
    rawById.set(id, def);
    return def;
  };

  if (fetch(rootId) === undefined) {
    throw new WorkflowLoadError(`root state '${rootId}' not found in bundle`);
  }
  /**
   * The id one MOUNT of a state loads under.
   *
   * A state mounted under two parents that hand it different environments is running as two
   * different things, so it gets two entries — the first mount keeps the plain id, and any later
   * one that inherits something different gets a `#`-suffixed VARIANT. Everything downstream
   * (validation, snapshots, the board, events) then goes on treating a state as one id with one
   * operation, which is what it needs to be; only the loader knows a variant exists.
   *
   * The suffix hashes the inherited environment rather than the parent's name, so the id depends on
   * what the state RUNS AS and not on where it was reached from: two parents that happen to pass the
   * same environment collapse back onto one entry, which is both correct and what keeps the ordinary
   * shared-library case from sprouting duplicates.
   */
  const variantsById = new Map<string, Map<string, string>>();
  const variantFor = (id: string, identity: string): string => {
    let byIdentity = variantsById.get(id);
    if (byIdentity === undefined) {
      byIdentity = new Map();
      variantsById.set(id, byIdentity);
    }
    const existing = byIdentity.get(identity);
    if (existing !== undefined) return existing;
    const variant = byIdentity.size === 0 ? id : `${id}${VARIANT_SEPARATOR}${sha256Hex(identity).slice(0, 8)}`;
    byIdentity.set(identity, variant);
    return variant;
  };

  // Transitive closure from the root, carrying each path's accumulated environment.
  const states: Record<string, LoadedState> = {};
  const sourceOf = new Map<string, string>();
  const queue: Array<{ id: string; variant: string; inherited: OperationFields }> = [
    { id: rootId, variant: rootId, inherited: {} },
  ];
  while (queue.length > 0) {
    const { id, variant, inherited } = queue.shift()!;
    if (states[variant]) continue;
    const def = rawById.get(id);
    if (!def) {
      // Missing children are a VALIDATION error (with context), not a load error —
      // keep loading so the validator can report all of them at once.
      continue;
    }
    sourceOf.set(variant, id);
    // Document references are expanded FIRST, so children inference, environment inheritance and
    // desugaring all see a state as though the author had typed it out in full (§8).
    // The search path this state resolves under has to be known BEFORE its own document is expanded,
    // which is why it is read off the raw `environment` rather than the expanded one. That is the
    // cycle §9 flagged, and this is where it breaks: a state whose `environment` is itself a
    // transclusion cannot use a path declared inside that same transclusion — you cannot resolve a
    // reference with a path you have not loaded yet. Everything else inherits normally.
    const ownEnvironment = def.environment !== undefined && !Array.isArray(def.environment) && typeof def.environment === "object" ? def.environment : undefined;
    const searchPath = resolutionEnvironment(inherited, ownEnvironment).path;
    const expanded = expandFor(id, def, searchPath);
    // Desugared against the SOURCE id: `./goals` means "under the state's own path", and a variant
    // suffix is an identity for this mount, not a different place on disk.
  /**
   * Resolve an operation NAME used in an expression to the operation it denotes (EXPRESSIONS.md §3).
   *
   * A path-resolved callee is an operation DOCUMENT — the same `OperationFields` an `operation` block
   * is written in — so `$/functions/classify.json` looks exactly like the operation it would be if
   * typed inline. That is what makes a built-in and a project's own indistinguishable (§2), and it is
   * why the document carries its own `input` declarations: they are the signature a call's positional
   * arguments bind against (§3.3).
   *
   * A callee is ALWAYS a reference — children and callees are separate namespaces that do not
   * interact, so a child named `classify` and a callee `classify` simply coexist. An earlier draft
   * had a declared child win, on a "lexical scope beats a module path" analogy; but a child is a
   * STATE (with its own children, sequence, transitions and limits), and "calling" one would
   * duplicate what `children[].inputs` and `.children.k.outputs.x` already do, with no clear answer
   * for re-entry or the cursor.
   */
  /** Locate a name along the search `path` and read what is there, or `undefined` if nothing is. */
  const resolveDocument = (
    name: string,
    stateId: string,
    path: readonly string[] | undefined,
  ): { document: unknown; file: string; symbol?: readonly string[] } | undefined => {
    // A HOST-SHIPPED document has no file, so it is reached without one — which is also why it is
    // tried after the path rather than before: a name that resolves to something on disk is a
    // document somebody can open, and that always wins.
    const shipped = (): { document: unknown; file: string } | undefined => {
      const document = options.documents?.[name];
      return document === undefined ? undefined : { document, file: `<built-in>/${name}` };
    };
    if (options.vfs === undefined) return shipped();
    const defaultRoot = path !== undefined && path.length > 0 ? path : options.defaultRoot;
    let located;
    try {
      located = resolveReference(name, {
        vfs: options.vfs,
        from: stateId,
        ...(defaultRoot !== undefined ? { defaultRoot } : {}),
        ...(options.roots !== undefined ? { roots: options.roots } : {}),
        ...(options.symbols !== undefined ? { symbols: options.symbols } : {}),
        ...(options.documentCache !== undefined ? { documents: options.documentCache } : {}),
        ...(options.onWarn !== undefined ? { onWarn: options.onWarn } : {}),
      });
    } catch (e) {
      // A name that matches nothing on the path THROWS rather than returning empty-handed, so the
      // fallback has to be reached from here as well. Only when there is something to fall back to:
      // with no shipped document of that name the original error is the right one, and it names every
      // directory that was looked in.
      const document = shipped();
      if (document === undefined) throw e;
      return document;
    }
    if (located.file === undefined) return shipped();
    // A MODULE is not read as a document. `parseReferencedFile` returns a `.ts` file's TEXT (which is
    // right for a `.md` prompt and wrong here), and `selectProperty` on text reports "text has no
    // properties" — an error about the wrong thing entirely. A module's value is its EXPORT, reached
    // by loading it, so it is handed on as a symbol for `resolveOperationName` to type.
    if (isModuleFile(located.file)) {
      return { document: undefined, file: located.file, symbol: located.property };
    }
    const text = options.vfs.read(located.file);
    if (text === undefined) return shipped();
    return { document: selectProperty(parseReferencedFile(located.file, text), located.property, name), file: located.file };
  };

  const resolveOperationName = (
    name: string,
    stateId: string,
    path: readonly string[] | undefined,
  ): Operation<InlineFamily> | undefined => {
    const found = resolveDocument(name, stateId, path);
    if (found === undefined) return undefined;
    // A js/ts SYMBOL: its signature IS its declaration (SPEC §7.5.2), so the operation is read out of
    // the TypeScript rather than out of a JSON block somebody also had to write.
    if (found.symbol !== undefined) {
      if (options.userFunctions === undefined) {
        throw new WorkflowLoadError(
          `'${name}' resolves to '${found.file}', but this loader was given no way to read a js/ts signature`,
          stateId,
        );
      }
      return options.userFunctions.operationFor(found.file, found.symbol).operation;
    }
    const { document } = found;
    if (document === null || typeof document !== "object" || Array.isArray(document)) {
      throw new WorkflowLoadError(`operation '${name}' resolved to ${typeof document}, not an operation document`, stateId);
    }
    return desugarOperation(document as OperationFields, stateId, undefined, options.userFunctions);
  };

  /**
   * Resolve a bare name in VALUE position — inside an expression — to whatever it names.
   *
   * The counterpart of `resolveOperationName`, and deliberately wider: a name in callee position
   * must be an operation, while a name in an argument may be an operation, another binding, or plain
   * data. `bindingForDocument` makes that call once for both this and expansion, so
   * `"binding": "lib/x"` and `add(lib/x, 1)` cannot disagree about what `lib/x` is.
   */
  const resolveValueName = (
    name: string,
    stateId: string,
    path: readonly string[] | undefined,
    lower: LowerOptions,
  ): Ref<InlineFamily> | undefined => {
    const found = resolveDocument(name, stateId, path);
    if (found === undefined) return undefined;
    return desugarBinding(bindingForDocument(found.document, isDataFile(found.file)), `reference '${name}'`, stateId, undefined, lower);
  };

    const lower: LowerOptions = {
      resolveOperation: (name) => resolveOperationName(name, id, searchPath),
      resolveName: (name) => resolveValueName(name, id, searchPath, lower),
      ...(options.userFunctions !== undefined ? { userFunctions: options.userFunctions } : {}),
    };
    const loaded = desugarState(id, expanded, inherited, refs, childrenByParent.get(id), lower);
    loaded.id = variant;
    // Fan-out is a static property of the wiring (§7.3, rule 2): with every consumer of every producer
    // desugared to a base ref, the loader can tally them once here rather than the engine discovering a
    // second reader at run time.
    const fanOut = computeFanOut(loaded);
    if (fanOut !== undefined) loaded.fanOut = fanOut;
    states[variant] = loaded;
    // Children inherit the chain plus THIS state's `environment` — never its `operation`, which
    // describes what this state does, not what its subtree defaults to. Read off the EXPANDED
    // document, like every other pass: expansion runs first precisely so nothing downstream has to
    // know references exist, and reading the raw one here meant a transcluded `environment` reached
    // this state and not its children.
    const forChildren = resolutionEnvironment(inherited, expanded.environment);
    // From the LOADED children, so inferred ones (§6) are walked exactly like declared ones and
    // their references are already resolved to canonical ids. Each child's `state` is then rewritten
    // to the variant THIS mount reaches, so a parent always points at the child it actually runs.
    for (const child of Object.values(loaded.children ?? {})) {
      const childSource = child.state;
      fetch(childSource);
      // PER CHILD, not per state: a child's own `environment` is the nearest layer above the state it
      // mounts, so two children of one parent can hand the SAME state two different environments —
      // which is the whole point (`ChildDecl.environment`). With none declared this is the parent's
      // chain unchanged, and the identity is the one every sibling shares, so the ordinary
      // shared-library case still collapses onto a single entry.
      const childEnvironment = child.environment !== undefined ? resolutionEnvironment(forChildren, child.environment) : forChildren;
      const childVariant = variantFor(childSource, environmentIdentity(childEnvironment));
      child.state = childVariant;
      queue.push({ id: childSource, variant: childVariant, inherited: childEnvironment });
    }
  }
  // Spreads resolve against the loaded closure, so they run once the walk is done. Fan-out is
  // recomputed after, because republishing a child's outputs adds consumers of that child.
  for (const state of Object.values(states)) {
    if (state.outputSpreads === undefined) continue;
    expandOutputSpreads(state, states);
    const fanOut = computeFanOut(state);
    if (fanOut !== undefined) state.fanOut = fanOut;
  }

  // Keep the authored files for hashing — the snapshot identity is what the AUTHOR wrote. A variant
  // maps back to the file it came from, so two mounts of one state store its one authored form
  // twice, under the two ids they run as.
  const source: Record<string, StateDef> = {};
  for (const variant of Object.keys(states)) {
    const def = rawById.get(sourceOf.get(variant) ?? variant);
    if (def) source[variant] = def;
  }
  return { rootId, states, source };
}

/**
 * Separates a state id from the mount VARIANT suffix the loader may append (see `variantFor`).
 *
 * `#` because a state id is a path (§2.1) and this is not another path segment — it names which
 * *reading* of that path is meant, exactly as a URI fragment does.
 */
export const VARIANT_SEPARATOR = "#";

/** The authored state a (possibly variant) id came from. */
export function sourceStateId(id: string): string {
  const cut = id.lastIndexOf(VARIANT_SEPARATOR);
  return cut > 0 ? id.slice(0, cut) : id;
}

/**
 * The snapshot hash — the bundle's version identity (SPEC §12). This is the content-identity a
 * `hierarchical-workflow` execution memoizes under: `workflowDefinitionHash` returns it and it becomes
 * the memo-key's definition-hash component (DESIGN §3.4) via `withMemoize`'s `identify` seam.
 *
 * Over the RESOLVED states, not over `bundle.source`.
 *
 * Definition evaluation — path lookup, reference resolution, transclusion, expression lowering — is a
 * pre-pass, and each stage hashes the OUTPUT of its own stage: `hashOperation` keys a memo on a
 * resolved op, and this keys a snapshot on a resolved definition. Hashing the authored form instead
 * made a pin fix BYTES rather than MEANING, so a task pinned before a lowering change replayed its
 * files through the new loader with no hash change to signal it; it also left every file a reference
 * pulled in to be tracked into the identity separately, since resolution reads the project
 * filesystem rather than the snapshot. Both problems are absent here rather than mitigated: what was
 * referenced is inlined, and a change to what anything lowers to is a different hash by construction.
 *
 * This is why `LoadedState` has to be plain JSON — see `fanOut`, which was a `Set` and hashed as `{}`.
 */
export function snapshotHash(bundle: WorkflowBundle): string {
  const entries = Object.keys(bundle.states)
    .map((id) => [id, hashCanonical(stripDerivedId(bundle.states[id]!))] as const)
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  // The module digest is SPLICED IN only when there is one (SPEC §7.5.5). A js/ts module reached by
  // name cannot be inlined the way every other reference is, so its content has to reach the identity
  // some other way or a pinned task would run edited code under an unchanged version. Omitting the
  // key entirely when absent is what keeps every snapshot taken before modules existed unchanged —
  // `{ rootId, states }` must hash to exactly what it always did.
  const document =
    bundle.moduleDigest === undefined
      ? { rootId: bundle.rootId, states: entries }
      : { rootId: bundle.rootId, states: entries, modules: bundle.moduleDigest };
  return sha256Hex(canonicalize(document));
}

/** The `id` is the map KEY already, and a variant suffix names a mount rather than content. */
function stripDerivedId(def: StateDef | LoadedState): JsonValue {
  const { id: _id, ...rest } = def;
  return rest as unknown as JsonValue;
}

/**
 * Node-only convenience: load every `*.json` under a directory as a bundle rooted at
 * `rootId`. Uses dynamic imports so the module stays edge-safe when unused.
 */
export async function loadBundleFromDir(dir: string, rootId: string, options: LoadBundleOptions = {}): Promise<WorkflowBundle> {
  const { readdir, readFile } = await import("node:fs/promises");
  const { join, relative, resolve } = await import("node:path");
  const files: Record<string, unknown> = {};
  const walk = async (d: string): Promise<void> => {
    for (const entry of await readdir(d, { withFileTypes: true })) {
      const full = join(d, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) {
        const rel = relative(dir, full);
        files[rel] = JSON.parse(await readFile(full, "utf8")) as unknown;
      }
    }
  };
  await walk(dir);
  // The directory IS the default root, so a bare reference in any of these files means "under here".
  return loadBundle(files, rootId, { defaultRoot: resolve(dir), ...options });
}
