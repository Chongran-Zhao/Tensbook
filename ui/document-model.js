// Document model for .tens sources: sentinel detection, block parsing,
// ViewSource regions, and Markdown line numbering. Pure functions only —
// no editor or DOM access — so it runs headless under `node --test`
// (ui/tests/) as well as in the app.

export const TENS_OPEN = "<!-- tensorforge:tens -->";
export const TENS_CLOSE = "<!-- /tensorforge:tens -->";
export const NOTE_OPEN = "<!-- tensorforge:note -->";
export const NOTE_CLOSE = "<!-- /tensorforge:note -->";

export const VIEW_SOURCE_DIRECTIVE_RE = /^\s*ViewSource\s+(on|off)\s*$/i;

let nextBlockId = 1;

export function resetBlockIds() {
  nextBlockId = 1;
}

export function lineCount(text) {
  return text.length === 0 ? 1 : text.split(/\r?\n/).length;
}

export function isTensOpen(line) {
  return line.trim().toLowerCase() === TENS_OPEN.toLowerCase();
}

export function isTensClose(line) {
  return line.trim().toLowerCase() === TENS_CLOSE.toLowerCase();
}

export function canonicalDamagedTensSentinel(line) {
  const match = line.match(/^(\s*)<!--\s*(\/?)tensorforge:tens\s*--?\s*$/i);
  if (!match) return null;
  return `${match[1]}<!-- ${match[2] ? "/" : ""}tensorforge:tens -->`;
}

export function tensSentinelKind(line) {
  if (isTensOpen(line)) return "open";
  if (isTensClose(line)) return "close";
  const canonical = canonicalDamagedTensSentinel(line);
  if (!canonical) return null;
  return isTensOpen(canonical) ? "open" : "close";
}

export function isNoteOpen(line) {
  return line.trim().toLowerCase() === NOTE_OPEN.toLowerCase();
}

export function isNoteClose(line) {
  return line.trim().toLowerCase() === NOTE_CLOSE.toLowerCase();
}

export function looksLikeTensbookSource(source) {
  for (const line of source.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith("#")) continue;
    return (
      /\.(?:show|solve)\s*\(/.test(t) ||
      /^(?:Scalar|Var|Function|Tensor|ScalarSet|VectorSet|Diff|Derivative|Simplify|Sum|Det|Tr|Inv|Equation|ODE|BoundaryCondition|Integrate|Integral|log|sqrt|exp|sin|cos|tan|sinh|cosh|tanh)\s*\(/.test(t) ||
      /^\[[A-Za-z_]\w*\s*,\s*[A-Za-z_]\w*\]\s*=\s*(?:Spec_Decomp|Spectral)\s*\(/.test(t) ||
      /^[A-Za-z_]\w*(?:\[[^\]]+\])+\s*=/.test(t) ||
      /^[A-Za-z_]\w*\s*=/.test(t)
    );
  }
  return false;
}

export function blockKindForFreeform(source) {
  return looksLikeTensbookSource(source) ? "tens" : "markdown";
}

function trimBlockLines(lines, startLine) {
  let start = 0;
  let end = lines.length;
  while (start < end && lines[start].trim() === "") start++;
  while (end > start && lines[end - 1].trim() === "") end--;
  return {
    text: lines.slice(start, end).join("\n"),
    sourceLine: startLine + start,
    sourceEndLine: Math.max(startLine + start, startLine + end - 1),
  };
}

function makeParsedBlock(kind, text, sourceLine, sourceEndLine = sourceLine + lineCount(text) - 1) {
  return { id: nextBlockId++, kind, text, sourceLine, sourceEndLine };
}

export function makeFreeformBlock(text, sourceLine = 1) {
  const lines = text.split(/\r?\n/);
  const trimmed = trimBlockLines(lines, sourceLine);
  const cleaned = trimmed.text;
  if (!cleaned.trim()) return null;
  return makeParsedBlock(
    blockKindForFreeform(cleaned),
    cleaned,
    trimmed.sourceLine,
    trimmed.sourceEndLine,
  );
}

function parseLegacyNoteDocument(source) {
  const lines = source.split(/\r?\n/);
  const blocks = [];
  let start = 0;
  for (let i = 0; i < lines.length; i++) {
    if (!isNoteOpen(lines[i])) continue;
    const before = makeFreeformBlock(lines.slice(start, i).join("\n"), start + 1);
    if (before) blocks.push(before);
    i++;
    const noteStart = i;
    while (i < lines.length && !isNoteClose(lines[i])) i++;
    const note = trimBlockLines(lines.slice(noteStart, i), noteStart + 1);
    if (note.text.trim()) {
      blocks.push(makeParsedBlock("markdown", note.text, note.sourceLine, note.sourceEndLine));
    }
    start = i + 1;
  }
  const tail = makeFreeformBlock(lines.slice(start).join("\n"), start + 1);
  if (tail) blocks.push(tail);
  return blocks.length ? blocks : [makeParsedBlock("markdown", "", 1, 1)];
}

export function parseSourceDocument(source) {
  if (source.split(/\r?\n/).some(isNoteOpen)) return parseLegacyNoteDocument(source);

  const lines = source.split(/\r?\n/);
  const blocks = [];
  let start = 0;
  let sawTens = false;

  for (let i = 0; i < lines.length; i++) {
    if (!isTensOpen(lines[i])) continue;
    sawTens = true;
    const md = trimBlockLines(lines.slice(start, i), start + 1);
    if (md.text.trim()) {
      blocks.push(makeParsedBlock("markdown", md.text, md.sourceLine, md.sourceEndLine));
    }
    i++;
    const codeStart = i;
    while (i < lines.length && !isTensClose(lines[i])) i++;
    const body = lines.slice(codeStart, i).join("\n");
    blocks.push(
      makeParsedBlock(
        "tens",
        body,
        codeStart + 1,
        Math.max(codeStart + 1, i),
      ),
    );
    start = i + 1;
  }

  if (sawTens) {
    const tail = trimBlockLines(lines.slice(start), start + 1);
    if (tail.text.trim()) {
      blocks.push(makeParsedBlock("markdown", tail.text, tail.sourceLine, tail.sourceEndLine));
    }
    return blocks.length ? blocks : [makeParsedBlock("markdown", "", 1, 1)];
  }

  return [makeParsedBlock(blockKindForFreeform(source), source, 1, lineCount(source))];
}

export function isViewSourceDirective(line) {
  return VIEW_SOURCE_DIRECTIVE_RE.test(line);
}

export function sourceWithPreviewDirectivesRemoved(source) {
  const hasTensSentinels = source.split(/\r?\n/).some((line) => isTensOpen(line));
  const lines = source.split(/\r?\n/);
  let inTens = !hasTensSentinels && blockKindForFreeform(source) === "tens";
  for (let i = 0; i < lines.length; i++) {
    if (isTensOpen(lines[i])) {
      inTens = true;
      continue;
    }
    if (isTensClose(lines[i])) {
      inTens = false;
      continue;
    }
    if (inTens && isViewSourceDirective(lines[i])) lines[i] = "";
  }
  return lines.join("\n");
}

export function isSentinelDocLine(doc, lineNumber) {
  if (!doc || lineNumber < 1 || lineNumber > doc.lines) return false;
  const line = doc.line(lineNumber);
  return tensSentinelKind(line.text) != null;
}

export function visibleMarkdownLineNumber(doc, lineNumber) {
  if (!doc || lineNumber < 1 || lineNumber > doc.lines) return null;
  let inTens = false;
  let visible = 0;
  for (let n = 1; n <= lineNumber; n++) {
    const text = doc.line(n).text;
    const sentinel = tensSentinelKind(text);
    if (sentinel === "open") {
      inTens = true;
      if (n === lineNumber) return null;
      continue;
    }
    if (sentinel === "close") {
      if (n === lineNumber) return null;
      inTens = false;
      continue;
    }
    if (inTens) {
      if (n === lineNumber) return null;
      continue;
    }
    visible++;
  }
  return visible;
}

export function viewSourceRegionsForBlock(block) {
  if (!block || block.kind !== "tens") return [];
  const lines = block.text.split(/\r?\n/);
  const regions = [];
  let startIndex = null;
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(VIEW_SOURCE_DIRECTIVE_RE);
    if (!match) continue;
    if (match[1].toLowerCase() === "on") {
      startIndex = i + 1;
    } else if (startIndex != null) {
      regions.push(makeViewSourceRegion(block, startIndex, i - 1));
      startIndex = null;
    }
  }
  if (startIndex != null) {
    regions.push(makeViewSourceRegion(block, startIndex, lines.length - 1));
  }
  return regions.filter((region) => region.sourceText.trim());
}

function makeViewSourceRegion(block, startIndex, endIndex) {
  const lines = block.text.split(/\r?\n/);
  let start = Math.max(0, startIndex);
  let end = Math.min(lines.length - 1, endIndex);
  while (start <= end && lines[start].trim() === "") start++;
  while (end >= start && lines[end].trim() === "") end--;
  const sourceLines =
    start <= end ? lines.slice(start, end + 1).filter((line) => !isViewSourceDirective(line)) : [];
  return {
    blockId: block.id,
    sourceLine: block.sourceLine + start,
    sourceEndLine: block.sourceLine + end,
    sourceText: sourceLines.join("\n"),
  };
}

export function viewSourceRegionKey(region) {
  return region ? `${region.blockId}:${region.sourceLine}:${region.sourceEndLine}` : "";
}
