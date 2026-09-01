export * from "./expr.js";
// Expression TYPE inference (§7.2) and binding RESOLUTION (§7.4) — the two halves of the ops-redesign
// wiring model. Exported because both are reusable against a custom engine or a lint surface, not
// just from the validator and engine here.
export * from "./inferExpr.js";
// Expression LOWERING (§1) — an expression IS a producer tree, so a host walking bindings needs the
// same reader the fan-out planner and validator use.
export * from "./lowerExpr.js";
// Failures in the data plane (§5): whether a slot admits one, and what one looks like as a value.
export * from "./errorValue.js";
export * from "./format.js";
// Operation inheritance (§5) and the state PATH REFERENCE grammar (§2.1) — both consumed by the
// loader, both exported because a host resolving workflow locations needs the same rules.
export * from "./merge.js";
// What an operation's `session` declaration means (DESIGN.md §1.6), including the split between
// the conversation and the resource bundle the one string used to conflate. Exported because a host
// wiring its own session store resolves the same declaration.
export * from "./session.js";
export * from "./load.js";
// Durable id minting — the UUIDv7 every instance is named by. Exported because a host that mints
// ids in the same space (derived instance ids, later assigned session/task ids) must share the shape.
export * from "./ids.js";
// The `operation.*` expression namespace (SPEC.md §6.1) — a state's own call as an addressable
// node, which is where engine metadata about it finally has somewhere to live.
export * from "./operationNode.js";
export * from "./ref.js";
export * from "./reference.js";
// What a js/ts module CONTRIBUTES to the search path, and the per-directory index that answers for
// it (SPEC.md §7.5.2). Exported because a host that owns its own approval store builds the index
// itself — which is the point of `ModuleIndexOptions.approved`.
export * from "./moduleExports.js";
export * from "./moduleIndex.js";
// An embedded js/ts body compiled to module source (SPEC.md §7.5.1).
export * from "./functionBody.js";
// Transpiling and running a user module — the require path, the closure, and the CJS scope
// (SPEC.md §7.5.4, §7.5.6).
export * from "./moduleLoader.js";
// The wire boundary (SPEC.md §7.5.3): a TypeScript type's JSON Schema, and the value adapter
// between the wire and TypeScript.
export * from "./wireType.js";
export * from "./marshal.js";
// A function's signature, read from its TypeScript (SPEC.md §7.5.2).
export * from "./signature.js";
// Hashing, approval and the freeze (SPEC.md §7.5.5) — what makes running a user module
// acceptable, and what makes a frozen run actually frozen.
export * from "./integrity.js";
// Where every piece meets the rest of the system: a resolved js/ts symbol or an embedded body
// becomes an ordinary registry-dispatched operation (SPEC.md §7.5).
export * from "./userFunctions.js";
export * from "./shape.js";
export * from "./expand.js";
export * from "./loader.js";
export * from "./resolve.js";
export * from "./materialize.js";
export * from "./fanout.js";
export * from "./validate.js";
export * from "./ports.js";
export * from "./engine.js";
export * from "./executor.js";
