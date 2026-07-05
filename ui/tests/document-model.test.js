// Headless characterization tests for the .tens document model.
// Run with `npm run test:ui` (node --test); no browser or editor required.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  TENS_OPEN,
  TENS_CLOSE,
  tensSentinelKind,
  canonicalDamagedTensSentinel,
  looksLikeTensbookSource,
  parseSourceDocument,
  resetBlockIds,
  sourceWithPreviewDirectivesRemoved,
  visibleMarkdownLineNumber,
  isSentinelDocLine,
  viewSourceRegionsForBlock,
} from "../document-model.js";

// Minimal stand-in for a CodeMirror doc: 1-based lines with .text.
function fakeDoc(source) {
  const lines = source.split("\n");
  return {
    lines: lines.length,
    line(n) {
      return { text: lines[n - 1], from: 0 };
    },
  };
}

test("sentinel kind: exact, spaced, case-insensitive, damaged", () => {
  assert.equal(tensSentinelKind(TENS_OPEN), "open");
  assert.equal(tensSentinelKind(TENS_CLOSE), "close");
  assert.equal(tensSentinelKind("   <!-- TENSORFORGE:TENS -->  "), "open");
  // A damaged `--` ending still canonicalizes instead of leaking raw HTML.
  assert.equal(tensSentinelKind("<!-- /tensorforge:tens --"), "close");
  assert.equal(canonicalDamagedTensSentinel("<!--tensorforge:tens--"), TENS_OPEN);
  assert.equal(tensSentinelKind("C = F.T * F"), null);
  assert.equal(tensSentinelKind("<!-- tensorforge:note -->"), null);
});

test("parseSourceDocument: markdown / tens / markdown", () => {
  resetBlockIds();
  const src = ["# Title", "", TENS_OPEN, 'a = Scalar("a")', "a.show()", TENS_CLOSE, "", "tail"].join(
    "\n",
  );
  const blocks = parseSourceDocument(src);
  assert.deepEqual(
    blocks.map((b) => [b.kind, b.sourceLine, b.sourceEndLine]),
    [
      ["markdown", 1, 1],
      ["tens", 4, 5],
      ["markdown", 8, 8],
    ],
  );
  assert.equal(blocks[1].text, 'a = Scalar("a")\na.show()');
});

test("freeform sources classify as tens or markdown", () => {
  resetBlockIds();
  assert.equal(looksLikeTensbookSource("C = F.T * F"), true);
  assert.equal(looksLikeTensbookSource("# notes\njust prose"), false);
  assert.equal(parseSourceDocument("Tr(A).show()")[0].kind, "tens");
  assert.equal(parseSourceDocument("plain prose")[0].kind, "markdown");
});

test("legacy note fences become markdown blocks", () => {
  resetBlockIds();
  const src = ["<!-- tensorforge:note -->", "note text", "<!-- /tensorforge:note -->"].join("\n");
  const blocks = parseSourceDocument(src);
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0].kind, "markdown");
  assert.equal(blocks[0].text, "note text");
});

test("ViewSource directives are stripped only inside tens code", () => {
  const src = [TENS_OPEN, "ViewSource on", "x = Var(\"x\")", "ViewSource off", TENS_CLOSE, "ViewSource on"].join(
    "\n",
  );
  const cleaned = sourceWithPreviewDirectivesRemoved(src).split("\n");
  assert.equal(cleaned[1], "");
  assert.equal(cleaned[3], "");
  // Outside a tens block the words stay (they are ordinary Markdown there).
  assert.equal(cleaned[5], "ViewSource on");
});

test("visible markdown line numbers skip tens blocks and sentinels", () => {
  const doc = fakeDoc(["# one", TENS_OPEN, "code", TENS_CLOSE, "after"].join("\n"));
  assert.equal(visibleMarkdownLineNumber(doc, 1), 1);
  assert.equal(visibleMarkdownLineNumber(doc, 2), null); // open sentinel
  assert.equal(visibleMarkdownLineNumber(doc, 3), null); // tens code
  assert.equal(visibleMarkdownLineNumber(doc, 4), null); // close sentinel
  assert.equal(visibleMarkdownLineNumber(doc, 5), 2);
  assert.equal(isSentinelDocLine(doc, 2), true);
  assert.equal(isSentinelDocLine(doc, 3), false);
});

test("ViewSource regions: closed, unclosed, and empty", () => {
  const block = {
    id: 7,
    kind: "tens",
    sourceLine: 10,
    text: ["ViewSource on", "a = 1", "b = 2", "ViewSource off", "c = 3", "ViewSource on", "d = 4"].join(
      "\n",
    ),
  };
  const regions = viewSourceRegionsForBlock(block);
  assert.equal(regions.length, 2);
  assert.deepEqual(
    regions.map((r) => [r.sourceLine, r.sourceEndLine, r.sourceText]),
    [
      [11, 12, "a = 1\nb = 2"],
      [16, 16, "d = 4"], // unclosed region runs to the end of the block
    ],
  );
  // An `on` immediately followed by `off` yields no region.
  const empty = { id: 1, kind: "tens", sourceLine: 1, text: "ViewSource on\nViewSource off" };
  assert.equal(viewSourceRegionsForBlock(empty).length, 0);
  // Non-tens blocks never produce regions.
  assert.equal(viewSourceRegionsForBlock({ ...block, kind: "markdown" }).length, 0);
});

test("block ids reset deterministically", () => {
  resetBlockIds();
  const first = parseSourceDocument("a = 1")[0].id;
  resetBlockIds();
  const again = parseSourceDocument("a = 1")[0].id;
  assert.equal(first, again);
});
