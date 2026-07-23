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
export * from "./json";
export * from "./codec";
export * from "./template";
export * from "./infer";
export * from "./selectType";
export * from "./hashing";
export * from "./classification";
export * from "./failure";
export * from "./result";
export * from "./encodedError";
