/**
 * Structural workflow validation (JaiRA DESIGN §5.2). Runs before a snapshot is
 * accepted for execution; the same checks back a lint surface. Errors block
 * execution; warnings don't.
 */
import { parseExpression, referencesOf, type Expr } from "./expr";
import { CONTEXT_NAMESPACES, TERMINATE_TARGETS, type StateDef, type WorkflowBundle } from "./format";

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

const FIELD_TYPES = new Set(["string", "number", "integer", "boolean", "array", "object", "artifact", "passthrough"]);
const NAMESPACES: ReadonlySet<string> = new Set(CONTEXT_NAMESPACES);
const TERMINATES: ReadonlySet<string> = new Set(TERMINATE_TARGETS);

export function validateBundle(bundle: WorkflowBundle): ValidationReport {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  for (const [id, def] of Object.entries(bundle.states)) {
    validateState(id, def, bundle, errors, warnings);
  }
  return { errors, warnings };
}

function validateState(
  id: string,
  def: StateDef,
  bundle: WorkflowBundle,
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
): void {
  const err = (path: string, message: string): void => {
    errors.push({ stateId: id, path, message });
  };
  const warn = (path: string, message: string): void => {
    warnings.push({ stateId: id, path, message });
  };

  const children = def.children ?? {};
  const childKeys = new Set(Object.keys(children));

  // --- children ---------------------------------------------------------------
  for (const [key, child] of Object.entries(children)) {
    if (!bundle.states[child.state]) {
      err(`children.${key}.state`, `references unknown state '${child.state}'`);
    } else if (!child.state.startsWith(id + "/")) {
      // The tree convention (SPEC §2.4/§3.1). Kept a warning so shared/library
      // states mounted cross-tree remain expressible; the engine only needs the
      // reference to resolve.
      warn(`children.${key}.state`, `'${child.state}' is not a descendant path of '${id}'`);
    }
    for (const [inputName, wiring] of Object.entries(child.inputs ?? {})) {
      if (typeof wiring === "string") {
        checkExpression(wiring, `children.${key}.inputs.${inputName}`, def, childKeys, err);
      } else if (wiring === null || typeof wiring !== "object" || !("value" in wiring)) {
        err(`children.${key}.inputs.${inputName}`, "wiring must be an expression string or { value: ... }");
      }
      const childDef = bundle.states[child.state];
      if (childDef && childDef.inputs && !(inputName in childDef.inputs)) {
        err(`children.${key}.inputs.${inputName}`, `child '${child.state}' declares no input '${inputName}'`);
      }
    }
    // Required child inputs must be wired (or defaulted/optional).
    const childDef = bundle.states[child.state];
    if (childDef) {
      for (const [inputName, schema] of Object.entries(childDef.inputs ?? {})) {
        const wired = child.inputs && inputName in child.inputs;
        if (!wired && !schema.optional && schema.default === undefined) {
          err(`children.${key}.inputs`, `required child input '${inputName}' is not wired`);
        }
      }
    }
  }

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
      ast = checkExpression(t.when, `transitions[${i}].when`, def, childKeys, err);
    }
    // Unguarded-cycle warning: a transition that re-enters a sequence member resets
    // the cursor (SPEC §3.3) and can loop forever without an iteration guard.
    if (childKeys.has(t.to) && sequence.includes(t.to) && def.limits?.max_iterations === undefined) {
      const guarded = ast !== undefined && referencesOf(ast).some((p) => p[0] === "run" && p[1] === "iteration");
      if (!guarded) {
        warnings.push({
          stateId: id,
          path: `transitions[${i}]`,
          message: `transition to sequence member '${t.to}' can cycle; add limits.max_iterations or a run.iteration guard`,
        });
      }
    }
  });

  // --- field schemas ----------------------------------------------------------
  for (const [section, fields] of [
    ["params", def.params],
    ["inputs", def.inputs],
    ["outputs", def.outputs],
  ] as const) {
    for (const [name, schema] of Object.entries(fields ?? {})) {
      if (typeof schema.type !== "string" || !FIELD_TYPES.has(schema.type)) {
        err(`${section}.${name}.type`, `unknown field type '${String(schema.type)}'`);
      }
      if (schema.type === "passthrough" && section !== "outputs") {
        err(`${section}.${name}.type`, "'passthrough' is only valid for outputs");
      }
      if (schema.enum !== undefined && !Array.isArray(schema.enum)) {
        err(`${section}.${name}.enum`, "enum must be an array");
      }
      const from = (schema as { from?: unknown }).from;
      if (from !== undefined) {
        if (section !== "outputs") err(`${section}.${name}.from`, "'from' is only valid for outputs");
        else if (typeof from !== "string") err(`${section}.${name}.from`, "'from' must be an expression string");
        else checkExpression(from, `outputs.${name}.from`, def, childKeys, err);
      }
      if (schema.type === "passthrough" && from === undefined) {
        err(`${section}.${name}`, "passthrough outputs require a 'from' expression");
      }
    }
  }

  // --- operations -------------------------------------------------------------
  if (def.agent) {
    if (typeof def.agent.provider !== "string" || def.agent.provider.length === 0) {
      err("agent.provider", "agent.provider is required");
    }
    const mode = def.agent.conversation?.mode;
    if (mode !== undefined && !["full_history", "summary", "fresh", "selected_artifacts"].includes(mode)) {
      err("agent.conversation.mode", `unknown conversation mode '${String(mode)}'`);
    }
  }
  if (def.skill && (typeof def.skill.name !== "string" || def.skill.name.length === 0)) {
    err("skill.name", "skill.name is required");
  }
  if (def.ui && (typeof def.ui.component !== "string" || def.ui.component.length === 0)) {
    err("ui.component", "ui.component is required");
  }
  const hasOperation =
    def.ui !== undefined || def.agent !== undefined || def.skill !== undefined || Object.keys(children).length > 0;
  if (!hasOperation) {
    warn("", "state declares no operations (no ui/agent/skill/children); it will terminate immediately");
  }
}

/**
 * Parse an expression and statically check its references: the root must be a known
 * namespace; `children.<key>` must be declared; `inputs/outputs/params.<name>` must be
 * declared fields. Returns the AST when the expression parses.
 */
function checkExpression(
  src: string,
  path: string,
  def: StateDef,
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
    if (root === "children") {
      const key = ref[1];
      if (key !== undefined && !childKeys.has(key)) {
        err(path, `references undeclared child '${key}'`);
      }
    }
    if (root === "inputs" && ref[1] !== undefined && !(def.inputs && ref[1] in def.inputs)) {
      err(path, `references undeclared input '${ref[1]}'`);
    }
    if (root === "params" && ref[1] !== undefined && !(def.params && ref[1] in def.params)) {
      err(path, `references undeclared param '${ref[1]}'`);
    }
    if (root === "outputs" && ref[1] !== undefined && !(def.outputs && ref[1] in def.outputs)) {
      err(path, `references undeclared output '${ref[1]}'`);
    }
  }
  return ast;
}
