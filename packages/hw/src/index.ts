export * from "./expr";
// Expression TYPE inference (§7.2) and binding RESOLUTION (§7.4) — the two halves of the ops-redesign
// wiring model. Exported because both are reusable against a custom engine or a lint surface, not
// just from the validator and engine here.
export * from "./inferExpr";
// Expression LOWERING (§1) — an expression IS a producer tree, so a host walking bindings needs the
// same reader the fan-out planner and validator use.
export * from "./lowerExpr";
// Failures in the data plane (§5): whether a slot admits one, and what one looks like as a value.
export * from "./errorValue";
export * from "./format";
// Operation inheritance (§5) and the state PATH REFERENCE grammar (§2.1) — both consumed by the
// loader, both exported because a host resolving workflow locations needs the same rules.
export * from "./merge";
// What an operation's `session` declaration means (DESIGN.md §1.6), including the split between
// the conversation and the resource bundle the one string used to conflate. Exported because a host
// wiring its own session store resolves the same declaration.
export * from "./session";
// The `operation.*` expression namespace (SPEC.md §6.1) — a state's own call as an addressable
// node, which is where engine metadata about it finally has somewhere to live.
export * from "./operationNode";
export * from "./ref";
export * from "./reference";
export * from "./shape";
export * from "./expand";
export * from "./loader";
export * from "./resolve";
export * from "./materialize";
export * from "./fanout";
export * from "./validate";
export * from "./ports";
export * from "./engine";
export * from "./executor";
