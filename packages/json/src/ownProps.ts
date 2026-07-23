/**
 * Own-property access, used by every walk in this package that REBUILDS an object from another one's
 * keys (./codec's schema walk, ./template's substitution and hole collapse, ./selectType's descent).
 *
 * Internal — deliberately not re-exported from ./index: it is a hazard fix, not vocabulary.
 */

/**
 * Assign an OWN data property. Plain `out[k] = v` invokes an INHERITED SETTER when `k` is
 * `"__proto__"` — `Object.prototype`'s accessor — so the key is silently dropped from the result (or,
 * worse, re-parents it) instead of being stored. `JSON.parse` really does produce `__proto__` own
 * properties, so every wire value reaches this. `defineProperty` never consults the prototype chain.
 */
export function setOwn(target: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(target, key, { value, writable: true, enumerable: true, configurable: true });
}

/** Read a key only when the object OWNS it — `bindings["constructor"]` / `schema["__proto__"]` would
 *  otherwise resolve off `Object.prototype` and hand back a function or the prototype itself. */
export function getOwn<T>(source: { readonly [key: string]: T }, key: string): T | undefined {
  return Object.prototype.hasOwnProperty.call(source, key) ? source[key] : undefined;
}
