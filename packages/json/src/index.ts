/**
 * @declarative-ai/json — the bottom of the package graph (DESIGN §2).
 *
 * Everything here is about JSON: the value/document vocabulary, the wire projection of a decoded type
 * (`Jsonify`) and its codec/type-name registry (`x-type`), pure schema transforms (templates,
 * inference, JSONPath select-typing), canonical serialization + hashing, and the classified error +
 * telemetry vocabulary all three result types share.
 *
 * Nothing in this package can be declined: its only dependencies are `canonicalize` and
 * `@noble/hashes`, both tiny and runtime-agnostic. It knows nothing about operations, execution,
 * providers, or validation.
 */
export * from "./json.js";
export * from "./codec.js";
export * from "./template.js";
export * from "./infer.js";
export * from "./selectType.js";
export * from "./hashing.js";
export * from "./classification.js";
export * from "./failure.js";
export * from "./result.js";
export * from "./encodedError.js";
