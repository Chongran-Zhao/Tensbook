// Headless tests for the DOM-free render helpers.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  formatPlotTick,
  plotTicks,
  plotPath,
  stripDisplayMath,
  escapeHtml,
  escapeAttr,
} from "../render.js";

test("plot tick formatting: 3 significant figures, exponent extremes", () => {
  assert.equal(formatPlotTick(Math.PI), "3.14");
  assert.equal(formatPlotTick(-1.5707963), "-1.57");
  assert.equal(formatPlotTick(0), "0");
  assert.equal(formatPlotTick(1e-14), "0"); // sub-epsilon collapses to zero
  assert.equal(formatPlotTick(12345), "1.2e4");
  assert.equal(formatPlotTick(0.004), "4.0e-3");
  assert.equal(formatPlotTick(Number.NaN), "");
});

test("plotTicks spans the range inclusively and rejects degenerate input", () => {
  assert.deepEqual(plotTicks(0, 4), [0, 1, 2, 3, 4]);
  assert.deepEqual(plotTicks(2, 2), []);
  assert.deepEqual(plotTicks(Number.NaN, 1), []);
});

test("plotPath emits one M then Ls with 2-decimal pixels", () => {
  const id = (v) => v;
  assert.equal(
    plotPath(
      [
        [0, 0],
        [1, 2.345],
      ],
      id,
      id,
    ),
    "M0.00,0.00 L1.00,2.35",
  );
});

test("stripDisplayMath removes only the $$ fence", () => {
  assert.equal(stripDisplayMath("$$\nx+1\n$$"), "x+1");
  assert.equal(stripDisplayMath("x+1"), "x+1");
});

test("html and attribute escaping", () => {
  assert.equal(escapeHtml("a<b & c>d"), "a&lt;b &amp; c&gt;d");
  assert.equal(escapeAttr("\"x\" & 'y'"), "&quot;x&quot; &amp; &#39;y&#39;");
});
