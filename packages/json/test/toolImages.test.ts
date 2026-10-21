/**
 * A tool's pictures, split from its value — what every route hands a model as images.
 */
import { describe, expect, it } from "vitest";
import { splitToolImages, TOOL_IMAGES } from "../src/toolImages.js";

describe("splitToolImages", () => {
  it("takes the pictures out of a result and leaves the rest as it was", () => {
    const out = splitToolImages({ width: 2, [TOOL_IMAGES]: [{ mimeType: "image/png", data: "AAA=" }] });
    expect(out).toEqual({ value: { width: 2 }, images: [{ mimeType: "image/png", data: "AAA=" }] });
  });

  it("leaves a result with no pictures — or not an object at all — alone", () => {
    expect(splitToolImages("text")).toEqual({ value: "text", images: [] });
    expect(splitToolImages([1, 2])).toEqual({ value: [1, 2], images: [] });
    expect(splitToolImages({ a: 1 })).toEqual({ value: { a: 1 }, images: [] });
    expect(splitToolImages(null)).toEqual({ value: null, images: [] });
  });

  it("drops what is not a picture, and the key with it", () => {
    const out = splitToolImages({ a: 1, [TOOL_IMAGES]: [{ mimeType: "text/plain", data: "x" }, { mimeType: "image/jpeg", data: "" }, 7] });
    expect(out).toEqual({ value: { a: 1 }, images: [] });
  });
});
