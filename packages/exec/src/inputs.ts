/**
 * Reading a RESOLVED operation's inputs — the binding walk, shared by everything that dispatches one.
 *
 * Split out of `operationExecutor.ts` when the function half became {@link FunctionExecutor}: both
 * executors need this walk, `resolvedOperation.ts` needs it too, and leaving it in either executor
 * would have made the three modules import each other in a circle. It depends on nothing but the op
 * model, which is the honest shape for it — the walk is a fact about `Parameter`, not about who runs
 * the op.
 */
import type { FunctionInputs, InlineFamily, Operation, Parameter, Ref, ResolvedValue } from "@declarative-ai/ops";
import { isOk } from "@declarative-ai/ops";

/**
 * Read a resolved op's inputs. Free slots are absent (the caller fills them by name); a bound slot must
 * carry a value that is ALREADY here by the time it reaches an executor. Four binding forms are:
 *
 *  - `{text}` / `{json}` / `{blob}` — a literal leaf; the value is the binding's payload.
 *  - `{op}` with `param.kind` in `{prompt, function}` — the op DEFINITION itself is the value
 *    (higher-order: the consumer receives an op to apply, it does not run it). `model.ts` states this as
 *    the meaning of the two non-data kinds, so refusing it here made the documented higher-order form
 *    undispatchable. An embedded `Operation` only: a local child NAME is a family-scope lookup, not a
 *    definition we hold.
 *  - `{result}` — an already-RESOLVED `OperationRecord`, whose value is right there. Nothing needs to
 *    run; a failed record is an error, because there is no value to pass.
 *
 * `{op}` with a DATA kind (run the producer, use its output) and `{refs}` (resolve a tree) both stay
 * errors: walking a producer edge is the family's business — hw's engine does it against its own scope,
 * PENDING joins and all — so one arriving here is a wiring bug and is reported as one.
 */
export function resolveLiteralInputs(op: Operation<InlineFamily>): { values: FunctionInputs } | { error: string } {
  const values: FunctionInputs = {};
  for (const [name, param] of Object.entries(op.input)) {
    const binding = param.binding;
    if (binding === undefined) continue; // free slot
    const resolved = resolveBinding(name, param, binding);
    if ("error" in resolved) return resolved;
    values[name] = resolved.value;
  }
  return { values };
}

function resolveBinding(name: string, param: Parameter<InlineFamily>, binding: Ref<InlineFamily>): { value: ResolvedValue } | { error: string } {
  if ("text" in binding) return { value: binding.text };
  if ("json" in binding) return { value: binding.json };
  if ("blob" in binding) return { value: binding.blob };
  if ("result" in binding) {
    const record = binding.result;
    if (!isOk(record.result)) {
      return { error: `input '${name}' is bound to a result record that FAILED (${record.result.error.reason}) — there is no value to pass` };
    }
    return { value: record.result.value };
  }
  if ("op" in binding) {
    // The kind decides what a producer edge MEANS (model.ts, `Parameter`): a data kind runs it, the two
    // op kinds pass the definition through untouched.
    if (param.kind !== "prompt" && param.kind !== "function") {
      return {
        // Unreachable for an EMBEDDED operation — `OperationExecutor.start` runs those and substitutes
        // their outputs before this walk. What is left is the case it cannot run: a producer edge
        // naming a declared CHILD, which is a lookup in the enclosing family's scope.
        error: `input '${name}' still carries an unresolved binding — a producer edge on a '${param.kind}' parameter must be RUN and its output substituted before dispatching the operation`,
      };
    }
    if (typeof binding.op === "string") {
      return {
        error: `input '${name}' names a declared child '${binding.op}' rather than embedding its definition — resolve the local name against the enclosing scope before dispatching the operation`,
      };
    }
    // Higher-order: the value IS the op document. Structurally JSON-shaped, but `Operation` is not
    // declared as one (its `blob`/`schema` members are not `JsonValue`), so the cast is where "an op
    // definition is data" is asserted once.
    return { value: binding.op as unknown as ResolvedValue };
  }
  return {
    error: `input '${name}' is bound to a ref TREE ({refs}) — resolving a tree is the ref family's job (hw's engine walks its own scope); dispatch the operation with the tree already resolved to a value`,
  };
}

/**
 * True when any input binds a producer edge carrying a whole operation on a DATA kind — the edges
 * `OperationExecutor.start` resolves by running them. An edge on a `prompt`/`function` kind is
 * higher-order (the definition itself is the value) and a string names a declared child, so neither
 * counts.
 *
 * The check keeps the common case free: an operation whose inputs are already literals takes the
 * same path it always did, straight to dispatch.
 */
export function hasEmbeddedOperation(op: Operation<InlineFamily>): boolean {
  return Object.values(op.input).some((p) => {
    const b = p.binding;
    return b !== undefined && "op" in b && typeof b.op !== "string" && p.kind !== "prompt" && p.kind !== "function";
  });
}
