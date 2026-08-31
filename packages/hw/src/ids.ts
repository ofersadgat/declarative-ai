/**
 * UUIDv7 (RFC 9562 §5.7): a 48-bit millisecond timestamp, then version and variant
 * bits over 74 random bits.
 *
 * Chosen for instance ids because the timestamp prefix makes lexicographic order
 * agree with creation order — a reader can sort minted ids without a counter, which
 * is what lets the id be durable instead of a per-walk sequence number.
 *
 * The timestamp is a PARAMETER rather than `Date.now()` so the engine can feed its
 * own clock: under a virtual clock in tests, an id's time half and the journal's
 * timestamps must not disagree about when the same entry happened.
 *
 * `globalThis.crypto` rather than `node:crypto` keeps this module free of node
 * imports (the same rule the rest of the repo follows for CSPRNG access).
 */

const HEX = "0123456789abcdef";

export function uuidv7(timestampMs: number): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  // 48-bit big-endian millisecond timestamp. `Math.trunc` + clamp because a virtual
  // clock may hand out 0 or a float; a negative time has no 48-bit encoding.
  let ms = Math.max(0, Math.trunc(timestampMs));
  for (let i = 5; i >= 0; i--) {
    bytes[i] = ms % 256;
    ms = Math.floor(ms / 256);
  }
  bytes[6] = 0x70 | (bytes[6]! & 0x0f); // version 7
  bytes[8] = 0x80 | (bytes[8]! & 0x3f); // variant 10xx
  let out = "";
  for (let i = 0; i < 16; i++) {
    const b = bytes[i]!;
    out += HEX[b >> 4]! + HEX[b & 0x0f]!;
    if (i === 3 || i === 5 || i === 7 || i === 9) out += "-";
  }
  return out;
}
