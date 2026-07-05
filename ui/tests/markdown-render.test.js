// Headless tests for the Markdown renderer. Math falls back to escaped code
// when KaTeX is not present, which is exactly what Node exercises here.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  inlineMarkdown,
  renderMarkdown,
  safeMarkdownHref,
  tableAlignments,
  tableCells,
} from "../markdown-render.js";

test("safeMarkdownHref only allows http, https, and mailto", () => {
  assert.equal(safeMarkdownHref("https://example.com/a"), "https://example.com/a");
  assert.equal(safeMarkdownHref("mailto:a@example.com"), "mailto:a@example.com");
  assert.equal(safeMarkdownHref("javascript:alert(1)"), null);
  assert.equal(safeMarkdownHref("file:///tmp/a"), null);
  assert.equal(safeMarkdownHref("not a url"), null);
});

test("inlineMarkdown handles links, styles, code, and unsafe links", () => {
  assert.equal(
    inlineMarkdown("**bold** *em* `x<y` [ok](https://example.com)"),
    '<strong>bold</strong> <em>em</em> <code>x&lt;y</code> <a href="https://example.com/">ok</a>',
  );
  assert.equal(
    inlineMarkdown("[x](javascript:alert(1))"),
    "[x](javascript:alert(1))",
  );
});

test("table helpers preserve cell text and alignment", () => {
  assert.deepEqual(tableCells("| a | b |"), [" a ", " b "]);
  assert.deepEqual(tableAlignments("| :--- | ---: | :---: |"), ["left", "right", "center"]);
});

test("renderMarkdown keeps source line attributes for headings and lists", () => {
  const html = renderMarkdown("# Title\n\n- one\n- two", 10);
  assert.match(html, /<h1 data-source-line="10">Title<\/h1>/);
  assert.match(html, /<ul data-source-line="12">/);
  assert.match(html, /<li data-source-line="13">two<\/li>/);
});

test("renderMarkdown renders fences, display math fallback, and tables", () => {
  const html = renderMarkdown("```rust\nx < y\n```\n\n$$\nx+1\n$$\n\n| A | B |\n| --- | ---: |\n| 1 | 2 |");
  assert.match(html, /<pre class="md-code"><code>x &lt; y<\/code><\/pre>/);
  assert.match(html, /<div class="markdown-math"><code>x\+1<\/code><\/div>/);
  assert.match(html, /<table class="md-table">/);
  assert.match(html, /<td style="text-align:right">2<\/td>/);
});
