/**
 * The built-in operation library (JaiRA EXPRESSIONS.md §3).
 *
 * These are ordinary operations that happen to ship with the language: a name, an ordered parameter
 * list, and a pure implementation. They are computed inline by the resolver, exactly as the
 * comparison and logical operators are — which is the point of §2's "the operator set is a
 * registry". `add(a, b)` and `a === b` differ only in spelling.
 *
 * Three rules every entry obeys:
 *
 *  - **Pure.** No clock, no randomness, no I/O. Resolution runs on every scheduling round, so an
 *    impure builtin would make a guard's answer depend on when it was asked.
 *  - **Total.** It returns a value for every input and never throws. `div(1, 0)`, `at(xs, 99)` and
 *    `parse_json("{")` all have answers; a thrown exception would escape a synchronous resolver into
 *    the engine's scheduling loop, which is the one place there is no good way to report it.
 *  - **Non-mutating.** `append` returns a new array; there is no `push`. Every binding in the system
 *    is a value, and an operator that edited one in place would let two consumers of the same
 *    producer see different data.
 *
 * A built-in NAME wins over a path lookup, so a project cannot accidentally shadow `add` with a file
 * called `add.json`. Naming a project operation the same as a built-in is a mistake worth making
 * loud rather than silently resolving one way.
 */

/** One built-in: its ordered parameter names, and a pure total implementation over their values. */
export interface Builtin {
  readonly params: readonly string[];
  readonly fn: (args: Record<string, unknown>) => unknown;
}

const num = (v: unknown): number => (typeof v === "number" ? v : Number.NaN);
const str = (v: unknown): string => (typeof v === "string" ? v : v === undefined || v === null ? "" : String(v));
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const obj = (v: unknown): Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};

/** A stable comparison for `sort`, so a workflow's output does not depend on the JS engine's. */
function compare(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return str(a) < str(b) ? -1 : str(a) > str(b) ? 1 : 0;
}

/**
 * A TOTAL stand-in for `JSON.stringify`, used wherever a value has to be compared or printed.
 *
 * `JSON.stringify` is not total, and three of its failure modes are reachable here. A user `.ts`
 * function's return value enters the dataflow unconverted whenever its schema needs no marshalling
 * (`marshalOut`), so anything JavaScript can hold can arrive: a CIRCULAR object ("Converting circular
 * structure to JSON"), a structure deep enough to exhaust the stack, or a BIGINT ("Do not know how to
 * serialize a BigInt"). Each of those threw straight out of a synchronous resolver — the one place
 * this module's header says there is no good way to report anything.
 *
 * Output is byte-identical to `JSON.stringify` for every value that JSON can represent, which is what
 * lets it replace it without moving the meaning of `same`. Key ORDER is preserved rather than sorted,
 * deliberately: sorting would be a better equality, and a different one.
 */
function safeKey(v: unknown, seen: Set<object> = new Set(), depth = 0): string {
  // Deep enough that no honest workflow value reaches it, shallow enough to leave stack to spare.
  if (depth > 200) return '"[too deep]"';
  if (v === null || v === undefined) return "null";
  switch (typeof v) {
    case "number":
      // `JSON.stringify(NaN)` is `null`, and matching that here is what keeps the two agreeing.
      return Number.isFinite(v) ? String(v) : "null";
    case "boolean":
      return String(v);
    case "string":
      return JSON.stringify(v);
    case "bigint":
      // JSON has no bigint. A marker beats a throw, and beats `Number(v)` silently losing precision.
      return JSON.stringify(`[bigint ${v.toString()}]`);
    case "function":
    case "symbol":
      return "null";
  }
  const o = v as Record<string, unknown> & { toJSON?: () => unknown };
  // `Date` and anything else defining it, exactly as `JSON.stringify` would.
  if (typeof o.toJSON === "function") {
    try {
      return safeKey(o.toJSON(), seen, depth + 1);
    } catch {
      return '"[unserializable]"'; // a throwing `toJSON` is the author's bug, not a reason to fail here
    }
  }
  if (seen.has(o)) return '"[circular]"';
  seen.add(o);
  try {
    if (Array.isArray(v)) return `[${v.map((x) => safeKey(x, seen, depth + 1)).join(",")}]`;
    const parts: string[] = [];
    for (const [k, val] of Object.entries(o)) {
      // Omitted from an object and nulled in an array — JSON's own asymmetry, reproduced.
      if (val === undefined || typeof val === "function" || typeof val === "symbol") continue;
      parts.push(`${JSON.stringify(k)}:${safeKey(val, seen, depth + 1)}`);
    }
    return `{${parts.join(",")}}`;
  } finally {
    // Removed on the way out so a value REPEATED in two branches is not mistaken for a cycle.
    seen.delete(o);
  }
}

const same = (a: unknown, b: unknown): boolean => safeKey(a) === safeKey(b);

function define(params: readonly string[], fn: (...values: unknown[]) => unknown): Builtin {
  return { params, fn: (args) => fn(...params.map((p) => args[p])) };
}

export const BUILTINS: Readonly<Record<string, Builtin>> = {
  // --- arithmetic -------------------------------------------------------------
  // Missing entirely until now: the language has no `+ - * /` and never did, so nothing could
  // compute a number. `reduce` in particular has nothing to reduce with without these.
  add: define(["a", "b"], (a, b) => num(a) + num(b)),
  sub: define(["a", "b"], (a, b) => num(a) - num(b)),
  mul: define(["a", "b"], (a, b) => num(a) * num(b)),
  div: define(["a", "b"], (a, b) => num(a) / num(b)),
  mod: define(["a", "b"], (a, b) => num(a) % num(b)),
  min: define(["a", "b"], (a, b) => Math.min(num(a), num(b))),
  max: define(["a", "b"], (a, b) => Math.max(num(a), num(b))),
  abs: define(["a"], (a) => Math.abs(num(a))),
  round: define(["a"], (a) => Math.round(num(a))),
  floor: define(["a"], (a) => Math.floor(num(a))),
  ceil: define(["a"], (a) => Math.ceil(num(a))),

  // --- dynamic access ---------------------------------------------------------
  // Member access takes a LITERAL name (`inputs.issue`), so a computed key had no spelling at all.
  get: define(["value", "key"], (v, k) => {
    const o = obj(v);
    return Object.hasOwn(o, str(k)) ? o[str(k)] : undefined;
  }),
  at: define(["value", "index"], (v, i) => {
    const xs = arr(v);
    const n = num(i);
    return Number.isInteger(n) ? xs[n < 0 ? xs.length + n : n] : undefined;
  }),

  // --- arrays -----------------------------------------------------------------
  len: define(["value"], (v) => (typeof v === "string" ? v.length : Array.isArray(v) ? v.length : Object.keys(obj(v)).length)),
  isEmpty: define(["value"], (v) => (typeof v === "string" ? v.length === 0 : Array.isArray(v) ? v.length === 0 : Object.keys(obj(v)).length === 0)),
  first: define(["value"], (v) => arr(v)[0]),
  last: define(["value"], (v) => arr(v)[arr(v).length - 1]),
  slice: define(["value", "start", "end"], (v, s, e) =>
    typeof v === "string" ? v.slice(num(s), e === undefined ? undefined : num(e)) : arr(v).slice(num(s), e === undefined ? undefined : num(e)),
  ),
  reverse: define(["value"], (v) => [...arr(v)].reverse()),
  sort: define(["value"], (v) => [...arr(v)].sort(compare)),
  /**
   * First occurrence of each distinct value, by the same deep comparison `same` defines.
   *
   * Keyed through a Set rather than scanning what has been kept. The scan was `out.some(same)`, which
   * is O(n²) comparisons and serializes both sides on every one of them: 8,000 distinct elements took
   * 6.5 SECONDS, 4,000 took one, and the curve keeps going. Being synchronous, that is not a slow
   * expression — it is the whole process stopped, engine and UI together, by an array that is merely
   * large. Same result, same order, one pass.
   */
  unique: define(["value"], (v) => {
    const out: unknown[] = [];
    const keys = new Set<string>();
    for (const item of arr(v)) {
      const k = safeKey(item);
      if (keys.has(k)) continue;
      keys.add(k);
      out.push(item);
    }
    return out;
  }),
  /** NOT `push`: every value here is immutable, so this returns a new array. */
  append: define(["value", "item"], (v, item) => [...arr(v), item]),
  /**
   * ONE level, as `Array.prototype.flat()` does, and an element that is not an array stays where it
   * is. The case that earns it a place: a fan-out fed by a sibling that itself fanned out reads an
   * array of arrays (WORKFLOWS.md §6.2), and the union is `flatten` of that — one level, because
   * one level is what one fan-out added.
   */
  flatten: define(["value"], (v) => arr(v).flatMap((x) => (Array.isArray(x) ? x : [x]))),
  range: define(["start", "end"], (s, e) => {
    const from = Math.trunc(num(s));
    const to = Math.trunc(num(e));
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return [];
    // Bounded, because an expression must not be able to exhaust memory from a typo'd argument.
    return Array.from({ length: Math.min(to - from, 10_000) }, (_, i) => from + i);
  }),
  join: define(["value", "separator"], (v, sep) => arr(v).map(str).join(sep === undefined ? "" : str(sep))),

  // --- aggregates -------------------------------------------------------------
  //
  // The arithmetic above computes over two numbers; these compute over a LIST, which is where the
  // arithmetic a real workflow needs actually lives — a weighted score, a winning candidate, a margin
  // between two of them.
  //
  // Every one that needs to look inside an element takes a KEY NAME rather than an operation, which
  // is what keeps them in here — pure and synchronous — instead of in the engine's embedded-op
  // machinery beside `map` and `reduce` (§3.5, which is built).
  //
  // So these are not a workaround for a missing feature. What differs is what has to exist on disk:
  // the op position of `reduce` takes a REFERENCE, so folding a column means authoring a document for
  // the fold, while `sum(pluck(xs, 'total'))` needs no file at all. For the cases these serve — a
  // weighted score, a winning row, a margin between two — that is the entire cost, which is why both
  // spellings earn their place.
  sum: define(["value"], (v) => arr(v).reduce<number>((t, x) => t + num(x), 0)),
  /**
   * Empty averages to 0 rather than to NaN.
   *
   * Both are defensible for a mean of nothing; only one is safe here. Every comparison against NaN is
   * false, so a NaN score would make `score < ask_below` false and a gate would silently resolve on a
   * measurement that does not exist — the exact failure a confidence score is there to prevent.
   */
  avg: define(["value"], (v) => (arr(v).length === 0 ? 0 : arr(v).reduce<number>((t, x) => t + num(x), 0) / arr(v).length)),
  /** Weighted sum — `dot(weights, signals)`. Pairs past the shorter list are dropped, not zero-filled:
   *  a weight with no signal is a mistake, and inventing a 0 for it would hide the mistake in a number. */
  dot: define(["a", "b"], (a, b) => {
    const xs = arr(a);
    const ys = arr(b);
    let total = 0;
    for (let i = 0; i < Math.min(xs.length, ys.length); i++) total += num(xs[i]) * num(ys[i]);
    return total;
  }),
  /** One property across every element — `map(xs, x => x[key])` without needing a function value. */
  pluck: define(["value", "key"], (v, k) => arr(v).map((x) => obj(x)[str(k)])),
  any: define(["value"], (v) => arr(v).some(Boolean)),
  all: define(["value"], (v) => arr(v).every(Boolean)),
  /** `undefined` for an empty list, exactly as `first` does — there is no maximum of nothing. */
  maxBy: define(["value", "key"], (v, k) =>
    arr(v).reduce<unknown>((best, x) => (best === undefined || num(obj(x)[str(k)]) > num(obj(best)[str(k)]) ? x : best), undefined),
  ),
  minBy: define(["value", "key"], (v, k) =>
    arr(v).reduce<unknown>((best, x) => (best === undefined || num(obj(x)[str(k)]) < num(obj(best)[str(k)]) ? x : best), undefined),
  ),
  /** Ascending, by the same stable `compare` `sort` uses — so `at(sortBy(xs, 'total'), -1)` is `maxBy`. */
  sortBy: define(["value", "key"], (v, k) => [...arr(v)].sort((a, b) => compare(obj(a)[str(k)], obj(b)[str(k)]))),
  /** The first element whose `key` equals `value`, by the same deep comparison `contains` uses. */
  find: define(["value", "key", "match"], (v, k, m) => arr(v).find((x) => same(obj(x)[str(k)], m))),

  // --- strings ----------------------------------------------------------------
  concat: define(["a", "b"], (a, b) => (Array.isArray(a) || Array.isArray(b) ? [...arr(a), ...arr(b)] : str(a) + str(b))),
  split: define(["value", "separator"], (v, sep) => str(v).split(sep === undefined ? "" : str(sep))),
  trim: define(["value"], (v) => str(v).trim()),
  lower: define(["value"], (v) => str(v).toLowerCase()),
  upper: define(["value"], (v) => str(v).toUpperCase()),
  replace: define(["value", "find", "with"], (v, f, w) => str(v).split(str(f)).join(str(w))),
  startsWith: define(["value", "prefix"], (v, p) => str(v).startsWith(str(p))),
  endsWith: define(["value", "suffix"], (v, sfx) => str(v).endsWith(str(sfx))),
  /** Substring for a string, membership for an array — what an author means by "contains". */
  contains: define(["value", "item"], (v, item) => (Array.isArray(v) ? v.some((x) => same(x, item)) : str(v).includes(str(item)))),

  // --- objects ----------------------------------------------------------------
  keys: define(["value"], (v) => Object.keys(obj(v))),
  values: define(["value"], (v) => Object.values(obj(v))),
  entries: define(["value"], (v) => Object.entries(obj(v)).map(([k, val]) => [k, val])),
  fromEntries: define(["value"], (v) => {
    const out: Record<string, unknown> = {};
    for (const pair of arr(v)) {
      const [k, val] = arr(pair);
      // `defineProperty`, not assignment: `out["__proto__"] = x` invokes the inherited setter and
      // re-parents the object instead of storing a key.
      if (k !== undefined) Object.defineProperty(out, str(k), { value: val, writable: true, enumerable: true, configurable: true });
    }
    return out;
  }),
  merge: define(["a", "b"], (a, b) => ({ ...obj(a), ...obj(b) })),
  pick: define(["value", "keys"], (v, ks) => {
    const o = obj(v);
    const out: Record<string, unknown> = {};
    for (const k of arr(ks)) if (Object.hasOwn(o, str(k))) out[str(k)] = o[str(k)];
    return out;
  }),
  omit: define(["value", "keys"], (v, ks) => {
    const drop = new Set(arr(ks).map(str));
    return Object.fromEntries(Object.entries(obj(v)).filter(([k]) => !drop.has(k)));
  }),

  // --- json, types, null ------------------------------------------------------
  parse_json: define(["value"], (v) => {
    try {
      return JSON.parse(str(v)) as unknown;
    } catch {
      return undefined; // total: a malformed document is `undefined`, not a thrown error
    }
  }),
  /** Total, like everything else here: a circular, too-deep or bigint value is described, not thrown. */
  to_json: define(["value"], (v) => safeKey(v)),
  typeof: define(["value"], (v) =>
    v === null ? "null" : Array.isArray(v) ? "array" : v === undefined ? "undefined" : typeof v,
  ),
  isArray: define(["value"], (v) => Array.isArray(v)),
  isNull: define(["value"], (v) => v === null || v === undefined),
  /** The first argument that is neither `null` nor `undefined` — `a ?? b`, which the grammar lacks. */
  coalesce: define(["a", "b"], (a, b) => (a === null || a === undefined ? b : a)),
};

/** Ordered parameter names per built-in — the positional-argument mapping (§3.3). */
export const BUILTIN_PARAMS: Readonly<Record<string, readonly string[]>> = Object.fromEntries(
  Object.entries(BUILTINS).map(([name, b]) => [name, b.params]),
);
