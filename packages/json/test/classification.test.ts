import { describe, expect, it } from "vitest";
import {
  classifyError,
  describeError,
  extractRateLimitInfo,
  isNetworkError,
  isRateLimit,
  isTimeoutOrAbort,
  retryAfterMs,
} from "../src/classification";

describe("error classification (§10.4)", () => {
  it("honors an explicit retryable flag first", () => {
    expect(classifyError({ retryable: true, status: 400 })).toBe("network-retriable");
    expect(classifyError({ retryable: false, status: 503 })).toBe("permanent");
  });

  it("honors the AI SDK `isRetryable` flag (APICallError field name)", () => {
    // A network APICallError: isRetryable true, NO http status. Must be transient, not
    // permanent (the bug: only `retryable` was read, so this fell through to permanent).
    expect(classifyError({ name: "AI_APICallError", isRetryable: true })).toBe("network-retriable");
    expect(classifyError({ name: "AI_APICallError", isRetryable: false, statusCode: 400 })).toBe(
      "permanent",
    );
  });

  it("classifies low-level network failures as transient", () => {
    expect(classifyError(Object.assign(new Error("read ECONNRESET"), { code: "ECONNRESET" }))).toBe(
      "network-retriable",
    );
    expect(classifyError(new TypeError("fetch failed"))).toBe("network-retriable");
    expect(classifyError({ code: "EAI_AGAIN", message: "getaddrinfo EAI_AGAIN" })).toBe("network-retriable");
    expect(classifyError({ message: "other side closed", code: "UND_ERR_SOCKET" })).toBe("network-retriable");
    expect(isNetworkError({ code: "ECONNREFUSED" })).toBe(true);
    expect(isNetworkError(new Error("schema mismatch"))).toBe(false);
  });

  it("finds a retryable/network signal through the AI SDK cause chain", () => {
    const wrapped = { name: "AI_RetryError", message: "failed after retries", cause: { name: "APICallError", isRetryable: true } };
    expect(classifyError(wrapped)).toBe("network-retriable");
    const wrappedNet = new Error("request failed");
    (wrappedNet as { cause?: unknown }).cause = Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
    expect(classifyError(wrappedNet)).toBe("network-retriable");
  });

  it("classifies 429 and 5xx as transient", () => {
    expect(classifyError({ status: 429 })).toBe("network-retriable");
    expect(classifyError({ statusCode: 500 })).toBe("network-retriable");
    expect(classifyError({ status: 503 })).toBe("network-retriable");
    expect(isRateLimit({ status: 429 })).toBe(true);
  });

  it("classifies 4xx (non-429) and unknown as permanent", () => {
    expect(classifyError({ status: 400 })).toBe("permanent");
    expect(classifyError({ status: 422 })).toBe("permanent");
    expect(classifyError(new Error("schema mismatch"))).toBe("permanent");
    expect(classifyError(null)).toBe("permanent");
  });

  it("classifies timeouts / aborts as transient", () => {
    expect(classifyError({ name: "AbortError" })).toBe("network-retriable");
    expect(classifyError({ name: "TimeoutError" })).toBe("network-retriable");
    expect(classifyError(new Error("request timeout exceeded"))).toBe("network-retriable");
    expect(isTimeoutOrAbort({ name: "AbortError" })).toBe(true);
  });

  it("captures which rate-limit dimension tripped + retry-after", () => {
    const err = {
      status: 429,
      responseHeaders: {
        "retry-after": "12",
        "anthropic-ratelimit-input-tokens-remaining": "0",
        "anthropic-ratelimit-input-tokens-reset": "2026-06-02T13:00:00Z",
        "x-unrelated": "ignore-me",
      },
    };
    const info = extractRateLimitInfo(err)!;
    expect(info["retry-after"]).toBe("12");
    expect(info["anthropic-ratelimit-input-tokens-remaining"]).toBe("0");
    expect(info["x-unrelated"]).toBeUndefined();
    expect(retryAfterMs(err)).toBe(12_000);
  });

  it("treats an EMPTY retry-after as no signal, not as 'retry immediately'", () => {
    // `Number("")` is a finite 0, so an empty header returned a 0 ms backoff — the caller would retry
    // with no delay at all instead of falling back to its own.
    expect(retryAfterMs({ status: 429, responseHeaders: { "retry-after": "" } })).toBeUndefined();
    expect(retryAfterMs({ status: 429, responseHeaders: { "retry-after": "   " } })).toBeUndefined();
    expect(retryAfterMs({ status: 429, responseHeaders: { "retry-after": "0" } })).toBe(0); // an explicit 0 IS a signal
  });

  it("describeError terminates on a CYCLIC cause chain — failure formatting must not crash", () => {
    // Each level is a bare `new Error("")`: name "Error" is skipped and the message is empty, so the
    // level contributes no parts and the walk recurses into `cause` — forever, without a depth cap.
    const a = new Error("");
    const b = new Error("");
    (a as { cause?: unknown }).cause = b;
    (b as { cause?: unknown }).cause = a;
    expect(describeError(a)).toBe("unknown error");
  });

  it("describeError surfaces the real cause, not a generic message", () => {
    const desc = describeError({
      status: 429,
      message: "rate limited",
      responseHeaders: { "retry-after": "5" },
    });
    expect(desc).toContain("HTTP 429");
    expect(desc).toContain("rate limited");
    expect(desc).toContain("retry-after=5");
  });
});
