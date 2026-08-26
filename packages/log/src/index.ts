/**
 * The logging spine, in a package of its own.
 *
 * It began inside `@declarative-ai/llm` and was private to it — which was fine while the model layer
 * was the only thing with anything to say, and wrong the moment anything else did. A library cannot
 * reach a logger it has no dependency on, so every other package either invented its own or, more
 * often, said nothing at all: failures were thrown and the reason went to whoever caught them.
 *
 * It lives here rather than in `@declarative-ai/json` because a package named for a data format is
 * not where anyone looks for a logger, and because this file has no dependencies at all — the one
 * thing that lets every other package depend on it without a cycle.
 */
export * from "./logger.js";
