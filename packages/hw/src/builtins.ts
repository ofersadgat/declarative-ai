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

const same = (a: unknown, b: unknown): boolean => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

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
  unique: define(["value"], (v) => {
    const out: unknown[] = [];
    for (const item of arr(v)) if (!out.some((x) => same(x, item))) out.push(item);
    return out;
  }),
  /** NOT `push`: every value here is immutable, so this returns a new array. */
  append: define(["value", "item"], (v, item) => [...arr(v), item]),
  range: define(["start", "end"], (s, e) => {
    const from = Math.trunc(num(s));
    const to = Math.trunc(num(e));
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return [];
    // Bounded, because an expression must not be able to exhaust memory from a typo'd argument.
    return Array.from({ length: Math.min(to - from, 10_000) }, (_, i) => from + i);
  }),
  join: define(["value", "separator"], (v, sep) => arr(v).map(str).join(sep === undefined ? "" : str(sep))),

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
  to_json: define(["value"], (v) => JSON.stringify(v ?? null)),
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
