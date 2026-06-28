// TensorForge frontend. The visible editor is a single CodeMirror source
// buffer: Markdown is the default, TensorForge code lives inside the saved
// tensorforge:tens sentinels. The on-disk .tens format stays unchanged.

import {
  Annotation,
  Decoration,
  EditorState,
  EditorView,
  HighlightStyle,
  RangeSetBuilder,
  ViewPlugin,
  acceptCompletion,
  autocompletion,
  completionKeymap,
  defaultKeymap,
  drawSelection,
  highlightActiveLine,
  highlightActiveLineGutter,
  history,
  historyKeymap,
  indentWithTab,
  keymap,
  lineNumbers,
  markdown,
  searchKeymap,
  syntaxHighlighting,
  tags,
} from "./vendor/codemirror/cm.bundle.js";
import {
  BUILTINS,
  markdownSlashCompletionSource,
  signatureHint,
  tensorForgeCompletionSource,
} from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const TENS_OPEN = "<!-- tensorforge:tens -->";
const TENS_CLOSE = "<!-- /tensorforge:tens -->";
const NOTE_OPEN = "<!-- tensorforge:note -->";
const NOTE_CLOSE = "<!-- /tensorforge:note -->";

const DEFAULT_SOURCE_URL = "start.tens";
const FALLBACK_DEFAULT_SOURCE = "# TensorForge\n\nClick **Open** or start writing.";

const KATEX_MACROS = { "\\bm": "\\boldsymbol{#1}" };
const SCROLL_SYNC_ANCHOR = 0.22;
const TENS_BUILTINS = new Set([...BUILTINS.map((b) => b.name), "show"]);
const TENS_KEYWORDS = new Set([
  "true",
  "false",
  "order",
  "dim",
  "rules",
  "identity",
  "symmetric",
  "antisymmetric",
  "orthogonal",
  "isotropic",
  "symbol",
  "components",
  "matrix",
  "block_components",
  "algebra",
  "tensor",
  "continuum",
]);

const editorWrap = document.getElementById("editor-wrap");
const mainEl = document.querySelector("main");
const output = document.getElementById("output");
const splitResizer = document.getElementById("split-resizer");
const runBtn = document.getElementById("run");
const openBtn = document.getElementById("open");
const newBtn = document.getElementById("new-file");
const saveBtn = document.getElementById("save");
const fileRail = document.getElementById("file-rail");
const railToggle = document.getElementById("rail-toggle");
const railFolder = document.getElementById("rail-folder");
const railFiles = document.getElementById("rail-files");
const insertTableBtn = document.getElementById("insert-table");
const tableMenu = document.getElementById("table-menu");
const printRoot = document.getElementById("print-root");
const addTensBtn = document.getElementById("add-note");
const exportBtn = document.getElementById("export");
const exportMenu = document.getElementById("export-menu");
const exportMdBtn = document.getElementById("export-md");
const exportPdfBtn = document.getElementById("export-pdf");
const themeBtn = document.getElementById("theme");
const filenameEl = document.getElementById("filename");

let currentPath = null;
let currentSavePath = null;
let lastRenderedOutputs = [];
let liveTimer = null;
let lastGoodShown = false;
let nextBlockId = 1;
let docBlocks = [];
let editorView = null;
let replacingDocument = false;
let isDirty = false;
let scrollSyncSource = null;
let scrollSyncFrame = null;
let scrollSyncSuppressedUntil = 0;

const VIEW_SOURCE_DIRECTIVE_RE = /^\s*ViewSource\s+(on|off)\s*$/i;

// ---- theme ----------------------------------------------------------------

function applyTheme(theme, { persist = true } = {}) {
  document.documentElement.dataset.theme = theme;
  if (persist) localStorage.setItem("tensorforge.theme", theme);
}

const lightQuery = window.matchMedia?.("(prefers-color-scheme: light)");
const systemTheme = () => (lightQuery?.matches ? "light" : "dark");

// Follow the system theme until the user explicitly picks one (then it sticks).
applyTheme(localStorage.getItem("tensorforge.theme") ?? systemTheme(), {
  persist: false,
});
lightQuery?.addEventListener?.("change", () => {
  if (!localStorage.getItem("tensorforge.theme")) {
    applyTheme(systemTheme(), { persist: false });
  }
});

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
}

// ---- about dialog ----------------------------------------------------------

const aboutOverlay = document.getElementById("about-overlay");
document.getElementById("about-btn").addEventListener("click", () => {
  aboutOverlay.classList.add("show");
});
document.getElementById("about-close").addEventListener("click", () => {
  aboutOverlay.classList.remove("show");
});
aboutOverlay.addEventListener("click", (e) => {
  if (e.target === aboutOverlay) aboutOverlay.classList.remove("show");
});

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href^='http'], a[href^='mailto:']");
  if (!a) return;
  e.preventDefault();
  const openUrl = window.__TAURI__?.opener?.openUrl;
  if (openUrl) openUrl(a.href);
  else window.open(a.href, "_blank");
});

function setCurrentPath(path, options = {}) {
  currentPath = path;
  const hasSavePath = Object.prototype.hasOwnProperty.call(options, "savePath");
  currentSavePath = hasSavePath ? (options.savePath ?? null) : path;
  updateFilename();
  if (currentSavePath) localStorage.setItem("tensorforge.path.v1", currentSavePath);
  else localStorage.removeItem("tensorforge.path.v1");
}

function updateFilename() {
  const name = currentPath ? currentPath.split("/").pop() : "";
  filenameEl.textContent = `${name}${isDirty ? " *" : ""}`;
  filenameEl.title = currentPath ? `${currentPath}${isDirty ? " (unsaved)" : ""}` : "";
}

function markDirty() {
  if (isDirty) return;
  isDirty = true;
  updateFilename();
}

function markClean() {
  if (!isDirty) {
    updateFilename();
    return;
  }
  isDirty = false;
  updateFilename();
}

function confirmDiscardUnsaved() {
  if (!isDirty) return true;
  return window.confirm("Discard unsaved changes?");
}

// ---- document model --------------------------------------------------------

const internalEdit = Annotation.define();

function lineCount(text) {
  return text.length === 0 ? 1 : text.split(/\r?\n/).length;
}

function isTensOpen(line) {
  return line.trim().toLowerCase() === TENS_OPEN.toLowerCase();
}

function isTensClose(line) {
  return line.trim().toLowerCase() === TENS_CLOSE.toLowerCase();
}

function canonicalDamagedTensSentinel(line) {
  const match = line.match(/^(\s*)<!--\s*(\/?)tensorforge:tens\s*--?\s*$/i);
  if (!match) return null;
  return `${match[1]}<!-- ${match[2] ? "/" : ""}tensorforge:tens -->`;
}

function tensSentinelKind(line) {
  if (isTensOpen(line)) return "open";
  if (isTensClose(line)) return "close";
  const canonical = canonicalDamagedTensSentinel(line);
  if (!canonical) return null;
  return isTensOpen(canonical) ? "open" : "close";
}

function isNoteOpen(line) {
  return line.trim().toLowerCase() === NOTE_OPEN.toLowerCase();
}

function isNoteClose(line) {
  return line.trim().toLowerCase() === NOTE_CLOSE.toLowerCase();
}

function looksLikeTensorForgeSource(source) {
  for (const line of source.split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith("#")) continue;
    return (
      /\.show\s*\(/.test(t) ||
      /^(?:Scalar|Var|Function|Tensor|ScalarSet|VectorSet|Diff|Derivative|Simplify|Sum|Det|Tr|Inv|Equation|ODE|BoundaryCondition|Integrate|Integral|log|sqrt|exp|sin|cos|tan|sinh|cosh|tanh)\s*\(/.test(t) ||
      /^\[[A-Za-z_]\w*\s*,\s*[A-Za-z_]\w*\]\s*=\s*(?:Spec_Decomp|Spectral)\s*\(/.test(t) ||
      /^[A-Za-z_]\w*(?:\[[^\]]+\])+\s*=/.test(t) ||
      /^[A-Za-z_]\w*\s*=/.test(t)
    );
  }
  return false;
}

function blockKindForFreeform(source) {
  return looksLikeTensorForgeSource(source) ? "tens" : "markdown";
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

function makeFreeformBlock(text, sourceLine = 1) {
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

function parseSourceDocument(source) {
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

function documentSource() {
  return editorView?.state.doc.toString() ?? "";
}

function isViewSourceDirective(line) {
  return VIEW_SOURCE_DIRECTIVE_RE.test(line);
}

function sourceWithPreviewDirectivesRemoved(source) {
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

function executableSource() {
  return docBlocks.some((block) => block.kind === "tens")
    ? sourceWithPreviewDirectivesRemoved(documentSource())
    : null;
}

function refreshDocumentModel() {
  nextBlockId = 1;
  docBlocks = parseSourceDocument(documentSource());
}

function setDocumentSource(source, options = {}) {
  replacingDocument = true;
  try {
    replaceEditorSource(source);
  } finally {
    replacingDocument = false;
  }
  refreshDocumentModel();
  if (options.dirty === false) markClean();
  if (options.run !== false) scheduleLiveRun();
}

function blockForSourceLine(line) {
  return (
    docBlocks.find((b) => line >= b.sourceLine && line <= b.sourceEndLine) ??
    docBlocks[docBlocks.length - 1]
  );
}

function blockForId(id) {
  return docBlocks.find((b) => String(b.id) === String(id));
}

// ---- CodeMirror source editor ---------------------------------------------

function tensRegions(source) {
  const lines = source.split(/\r?\n/);
  const regions = [];
  let line = 1;
  let inTens = false;
  for (const text of lines) {
    const sentinel = tensSentinelKind(text);
    if (sentinel === "open") inTens = true;
    regions.push({ line, kind: sentinel ? "sentinel" : inTens ? "tens" : "markdown" });
    if (sentinel === "close") inTens = false;
    line++;
  }
  return regions;
}

function regionKindAtLine(lineNumber) {
  const source = documentSource();
  return tensRegions(source).find((region) => region.line === lineNumber)?.kind ?? "markdown";
}

function regionKindAtPos(pos) {
  if (!editorView) return "markdown";
  return regionKindAtLine(editorView.state.doc.lineAt(pos).number);
}

function isMarkdownAtPos(pos) {
  return regionKindAtPos(pos) === "markdown";
}

function isSentinelDocLine(doc, lineNumber) {
  if (!doc || lineNumber < 1 || lineNumber > doc.lines) return false;
  const line = doc.line(lineNumber);
  return tensSentinelKind(line.text) != null;
}

function visibleMarkdownLineNumber(doc, lineNumber) {
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

function firstEditableDocPos(doc) {
  if (!doc) return 0;
  for (let n = 1; n <= doc.lines; n++) {
    if (!isSentinelDocLine(doc, n)) return doc.line(n).from;
  }
  return 0;
}

function addMark(builder, line, from, to, className) {
  if (to > from) builder.add(line.from + from, line.from + to, Decoration.mark({ class: className }));
}

function decorateTensTokens(builder, line) {
  const text = line.text;
  const directive = text.match(/^(\s*)(ViewSource)(\s+)(on|off)(\s*)$/i);
  if (directive) {
    const keywordFrom = directive[1].length;
    const keywordTo = keywordFrom + directive[2].length;
    const valueFrom = keywordTo + directive[3].length;
    addMark(builder, line, keywordFrom, keywordTo, "tf-token-directive");
    addMark(builder, line, valueFrom, valueFrom + directive[4].length, "tf-token-constant");
    return;
  }
  for (let i = 0; i < text.length; ) {
    const ch = text[i];
    if (ch === "#") {
      addMark(builder, line, i, text.length, "tf-token-comment");
      break;
    }
    if (ch === '"') {
      let j = i + 1;
      while (j < text.length) {
        if (text[j] === "\\" && text[j + 1] === '"') {
          j += 2;
          continue;
        }
        if (text[j] === '"') {
          j++;
          break;
        }
        j++;
      }
      addMark(builder, line, i, j, "tf-token-string");
      i = j;
      continue;
    }
    if (/[0-9]/.test(ch)) {
      const m = text.slice(i).match(/^\d+(?:\.\d+)?/);
      addMark(builder, line, i, i + m[0].length, "tf-token-number");
      i += m[0].length;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const m = text.slice(i).match(/^[A-Za-z_]\w*/);
      const word = m[0];
      const cls = TENS_BUILTINS.has(word)
        ? "tf-token-builtin"
        : TENS_KEYWORDS.has(word)
          ? "tf-token-keyword"
          : "tf-token-ident";
      addMark(builder, line, i, i + word.length, cls);
      i += word.length;
      continue;
    }
    if ("+-*/^=.:,&[]()".includes(ch)) addMark(builder, line, i, i + 1, "tf-token-op");
    i++;
  }
}

const sourceDecorations = ViewPlugin.fromClass(
  class {
    constructor(view) {
      this.decorations = this.build(view);
    }
    update(update) {
      if (update.docChanged || update.viewportChanged) this.decorations = this.build(update.view);
    }
    build(view) {
      const decorations = new RangeSetBuilder();
      const doc = view.state.doc;
      for (let n = 1; n <= doc.lines; n++) {
        const line = doc.line(n);
        if (isTensOpen(line.text)) {
          const contentLines = [];
          for (let m = n + 1; m <= doc.lines; m++) {
            const innerLine = doc.line(m);
            if (isTensClose(innerLine.text)) {
              const isEmptyBlock = contentLines.every((contentLine) => contentLine.text.trim() === "");
              for (let i = 0; i < contentLines.length; i++) {
                const contentLine = contentLines[i];
                const isLastContentLine = i === contentLines.length - 1;
                const classes = ["tf-tens-line"];
                if (i === 0) classes.push("tf-tens-first");
                if (isLastContentLine) classes.push("tf-tens-last");
                if (isEmptyBlock && i === 0) classes.push("tf-tens-empty");
                if (!isEmptyBlock && isLastContentLine && contentLine.text.trim() === "") {
                  classes.push("tf-tens-exit-hint");
                }
                decorations.add(
                  contentLine.from,
                  contentLine.from,
                  Decoration.line({ class: classes.join(" ") }),
                );
                if (!isEmptyBlock && contentLine.text.trim() !== "") {
                  decorateTensTokens(decorations, contentLine);
                }
              }
              n = m;
              break;
            }
            contentLines.push(innerLine);
          }
        }
      }
      return decorations.finish();
    }
  },
  { decorations: (plugin) => plugin.decorations },
);

function buildSentinelHiding(doc) {
  const decorations = new RangeSetBuilder();
  for (let n = 1; n <= doc.lines; n++) {
    const line = doc.line(n);
    if (tensSentinelKind(line.text) == null) continue;
    decorations.add(
      line.from,
      line.from,
      Decoration.line({ class: "tf-sentinel-hidden-line" }),
    );
    decorations.add(line.from, line.to, Decoration.replace({}));
  }
  return decorations.finish();
}

function repairDamagedTensSentinels(view) {
  const changes = [];
  const doc = view.state.doc;
  for (let n = 1; n <= doc.lines; n++) {
    const line = doc.line(n);
    const canonical = canonicalDamagedTensSentinel(line.text);
    if (canonical && canonical !== line.text) {
      changes.push({ from: line.from, to: line.to, insert: canonical });
    }
  }
  if (changes.length === 0) return false;
  view.dispatch({
    changes,
    annotations: internalEdit.of(true),
  });
  return true;
}

function stripOuterBlankLines(lines, side) {
  const next = [...lines];
  if (side === "left" || side === "both") {
    while (next.length > 0 && next[next.length - 1].trim() === "") next.pop();
  }
  if (side === "right" || side === "both") {
    while (next.length > 0 && next[0].trim() === "") next.shift();
  }
  return next;
}

function mergedTensContentLines(leftLines, rightLines) {
  const left = stripOuterBlankLines(leftLines, "left");
  const right = stripOuterBlankLines(rightLines, "right");
  const leftHasCode = left.some((text) => text.trim() !== "");
  const rightHasCode = right.some((text) => text.trim() !== "");
  if (leftHasCode && rightHasCode) return [...left, "", ...right];
  if (leftHasCode) return left;
  if (rightHasCode) return right;
  return [""];
}

function findTensCloseLineAfter(doc, openLineNumber) {
  for (let n = openLineNumber + 1; n <= doc.lines; n++) {
    const kind = tensSentinelKind(doc.line(n).text);
    if (kind === "open") return null;
    if (kind === "close") return n;
  }
  return null;
}

function normalizeAdjacentTensBlocks(view) {
  const doc = view.state.doc;
  for (let closeLineNo = 1; closeLineNo < doc.lines; closeLineNo++) {
    if (tensSentinelKind(doc.line(closeLineNo).text) !== "close") continue;

    // Only merge blocks that are *directly* adjacent (no lines between the
    // close and open sentinels). A blank line is an intentional Markdown gap
    // — e.g. from splitting a block with a double Enter — and must survive.
    const nextOpenLineNo = closeLineNo + 1;
    if (nextOpenLineNo > doc.lines) continue;
    if (tensSentinelKind(doc.line(nextOpenLineNo).text) !== "open") continue;

    const leftBlock = tensBlockAroundLine(doc, closeLineNo);
    const rightCloseLineNo = findTensCloseLineAfter(doc, nextOpenLineNo);
    if (!leftBlock || rightCloseLineNo == null) continue;

    const leftLines = [];
    const rightLines = [];
    for (let n = leftBlock.openLine + 1; n < closeLineNo; n++) leftLines.push(doc.line(n).text);
    for (let n = nextOpenLineNo + 1; n < rightCloseLineNo; n++) rightLines.push(doc.line(n).text);

    const mergedLines = mergedTensContentLines(leftLines, rightLines);
    const insert = `${TENS_OPEN}\n${mergedLines.join("\n")}\n${TENS_CLOSE}`;
    const from = doc.line(leftBlock.openLine).from;
    const to = doc.line(rightCloseLineNo).to;
    const selection = view.state.selection.main;
    const transaction = {
      changes: { from, to, insert },
      annotations: internalEdit.of(true),
    };
    if (selection.empty && selection.head >= from && selection.head <= to) {
      transaction.selection = { anchor: from + TENS_OPEN.length + 1 + mergedLines[0].length };
    }
    view.dispatch(transaction);
    return true;
  }
  return false;
}

function completionSource(context) {
  const kind = regionKindAtPos(context.pos);
  if (kind === "tens") return tensorForgeCompletionSource(context);
  if (kind === "markdown") return markdownSlashCompletionSource(context);
  return null;
}

const editorTheme = EditorView.theme({
  "&": {
    height: "100%",
    width: "100%",
    background: "var(--panel)",
    color: "var(--text)",
    font: '13px/1.65 "SF Mono", Menlo, Consolas, monospace',
  },
  ".cm-scroller": {
    overflow: "auto",
    fontFamily: '"SF Mono", Menlo, Consolas, monospace',
  },
  ".cm-content": {
    minHeight: "100%",
    padding: "18px 20px 42px 0",
    caretColor: "var(--text)",
  },
  ".cm-line": {
    padding: "0 0 0 13px",
  },
  ".cm-gutters": {
    background: "var(--panel)",
    color: "color-mix(in srgb, var(--muted) 68%, transparent)",
    borderRight: "1px solid color-mix(in srgb, var(--border) 62%, transparent)",
    userSelect: "none",
  },
  ".cm-lineNumbers .cm-gutterElement": {
    minWidth: "34px",
    padding: "0 9px 0 4px",
    fontSize: "12px",
    fontWeight: "400",
    fontVariantNumeric: "tabular-nums",
    textAlign: "right",
  },
  ".cm-activeLine": {
    background: "color-mix(in srgb, var(--accent) 5%, transparent)",
  },
  ".cm-activeLineGutter": {
    background: "transparent",
    color: "color-mix(in srgb, var(--accent) 50%, var(--muted))",
  },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": {
    background: "color-mix(in srgb, var(--accent) 34%, transparent)",
  },
  ".cm-tooltip": {
    background: "var(--panel-2)",
    color: "var(--text)",
    border: "1px solid var(--border)",
    borderRadius: "8px",
    boxShadow: "var(--shadow)",
  },
  ".cm-tooltip.cm-tooltip-autocomplete": {
    background: "var(--panel)",
    border: "1px solid var(--border)",
    borderRadius: "8px",
    overflow: "hidden",
  },
  ".cm-tooltip-autocomplete ul": {
    fontFamily: '"SF Mono", Menlo, Consolas, monospace',
    padding: "3px 0",
  },
  ".cm-tooltip-autocomplete ul li": {
    color: "var(--text)",
    padding: "3px 10px",
  },
  ".cm-tooltip-autocomplete ul li[aria-selected]": {
    background: "color-mix(in srgb, var(--accent) 15%, transparent)",
    color: "var(--text)",
  },
  ".cm-completionSection": {
    color: "var(--muted)",
    fontSize: "10px",
    fontWeight: "700",
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  ".cm-completionIcon": {
    alignItems: "center",
    borderRadius: "4px",
    display: "inline-flex",
    fontSize: "9px",
    fontWeight: "700",
    height: "15px",
    justifyContent: "center",
    marginRight: "8px",
    width: "15px",
  },
  ".cm-completionIcon-function": {
    background: "color-mix(in srgb, var(--syntax-function) 14%, transparent)",
    color: "var(--syntax-function)",
  },
  ".cm-completionIcon-variable": {
    background: "color-mix(in srgb, var(--syntax-number) 15%, transparent)",
    color: "var(--syntax-number)",
  },
  ".cm-completionIcon-keyword": {
    background: "color-mix(in srgb, var(--syntax-keyword) 13%, transparent)",
    color: "var(--syntax-keyword)",
  },
  ".cm-completionIcon-property": {
    background: "color-mix(in srgb, var(--accent) 12%, transparent)",
    color: "var(--accent)",
  },
  ".cm-completionLabel": {
    color: "var(--text)",
  },
  ".cm-completionDetail": {
    color: "var(--muted)",
    fontSize: "11px",
    marginLeft: "8px",
  },
  ".tf-sentinel-hidden-line": {
    height: "0",
    lineHeight: "0",
    minHeight: "0",
    overflow: "hidden",
    paddingTop: "0",
    paddingBottom: "0",
  },
  ".tf-tens-line": {
    background: "color-mix(in srgb, var(--tens-frame) 4%, transparent)",
    borderLeft: "3px solid color-mix(in srgb, var(--tens-frame) 55%, var(--panel))",
    boxSizing: "border-box",
    minHeight: "21.45px",
    paddingLeft: "11px",
    paddingRight: "11px",
    position: "relative",
  },
  ".tf-tens-line.cm-activeLine": {
    background: "color-mix(in srgb, var(--tens-frame) 12%, transparent)",
    borderLeftColor: "color-mix(in srgb, var(--tens-frame) 82%, var(--panel))",
  },
  ".tf-tens-empty::before": {
    content: '"TensorForge code"',
    position: "absolute",
    left: "11px",
    color: "var(--muted)",
    opacity: "0.6",
    pointerEvents: "none",
    whiteSpace: "nowrap",
  },
  ".tf-tens-exit-hint::before": {
    content: '"Enter again for Markdown"',
    position: "absolute",
    left: "11px",
    color: "var(--muted)",
    opacity: "0.52",
    pointerEvents: "none",
    whiteSpace: "nowrap",
  },
  ".tf-token-comment": { color: "var(--syntax-comment)", fontStyle: "italic" },
  ".tf-token-string": { color: "var(--syntax-string)" },
  ".tf-token-number": { color: "var(--syntax-number)" },
  ".tf-token-builtin": { color: "var(--syntax-function)", fontWeight: "600" },
  ".tf-token-keyword": { color: "var(--syntax-keyword)" },
  ".tf-token-directive": { color: "var(--syntax-keyword)", fontWeight: "650" },
  ".tf-token-constant": { color: "var(--syntax-number)", fontWeight: "550" },
  ".tf-token-op": { color: "var(--muted)" },
});

const markdownHighlight = HighlightStyle.define([
  { tag: tags.heading, color: "var(--accent)", fontWeight: "650" },
  { tag: tags.strong, color: "var(--syntax-strong)", fontWeight: "650" },
  { tag: tags.emphasis, color: "var(--syntax-keyword)", fontStyle: "italic" },
  { tag: tags.monospace, color: "var(--syntax-string)" },
  { tag: tags.link, color: "var(--accent)" },
  { tag: tags.meta, color: "var(--muted)" },
  { tag: tags.processingInstruction, color: "var(--muted)" },
]);

function handleEditorUpdate(update) {
  if (update.docChanged && repairDamagedTensSentinels(update.view)) return;
  if (update.docChanged && normalizeAdjacentTensBlocks(update.view)) return;
  if (update.docChanged) {
    refreshDocumentModel();
    if (!replacingDocument) {
      markDirty();
      scheduleLiveRun();
    }
  }
  if (update.docChanged || update.selectionSet) {
    updateInsertTableState();
  }
  if (update.focusChanged && !update.view.hasFocus) {
    hideSignatureHint();
  } else if (update.docChanged || update.selectionSet || update.focusChanged) {
    updateSignatureHint();
  }
}

function insertTextAtRange(from, to, text, { block = false } = {}) {
  if (!editorView) return;
  let insert = text;
  if (block) {
    const before = editorView.state.doc.sliceString(Math.max(0, from - 1), from);
    const after = editorView.state.doc.sliceString(to, Math.min(editorView.state.doc.length, to + 1));
    if (from > 0 && before !== "\n") insert = `\n${insert}`;
    if (!insert.endsWith("\n")) insert += "\n";
    if (to < editorView.state.doc.length && after !== "\n") insert += "\n";
  }
  const marker = insert.indexOf(MARKDOWN_CURSOR);
  if (marker !== -1) insert = insert.replace(MARKDOWN_CURSOR, "");
  const head = from + (marker === -1 ? insert.length : marker);
  editorView.dispatch({
    changes: { from, to, insert },
    selection: { anchor: head },
    annotations: internalEdit.of(true),
  });
  editorView.focus();
}

function insertAtSelection(text, { block = false } = {}) {
  if (!editorView) return;
  const sel = editorView.state.selection.main;
  insertTextAtRange(sel.from, sel.to, text, { block });
}

function replaceEditorSource(source) {
  if (!editorView) return;
  const nextDoc = EditorState.create({ doc: source }).doc;
  editorView.dispatch({
    changes: { from: 0, to: editorView.state.doc.length, insert: source },
    selection: { anchor: firstEditableDocPos(nextDoc) },
    annotations: internalEdit.of(true),
  });
}

function initEditor() {
  editorView = new EditorView({
    parent: editorWrap,
    state: EditorState.create({
      doc: "",
      extensions: [
        lineNumbers({
          formatNumber: (lineNo, state) => {
            const visibleLine = visibleMarkdownLineNumber(state.doc, lineNo);
            return visibleLine == null ? "" : String(visibleLine);
          },
        }),
        history(),
        drawSelection(),
        highlightActiveLine(),
        highlightActiveLineGutter(),
        markdown(),
        syntaxHighlighting(markdownHighlight),
        EditorView.lineWrapping,
        EditorView.decorations.compute(["doc"], (state) => buildSentinelHiding(state.doc)),
        editorTheme,
        sourceDecorations,
        autocompletion({
          override: [completionSource],
          activateOnTyping: true,
        }),
        EditorView.updateListener.of(handleEditorUpdate),
        EditorView.domEventHandlers({
          beforeinput(event, view) {
            if (event.inputType === "insertParagraph") {
              if (exitTensBlockOnDoubleBlankEnter(view)) {
                event.preventDefault();
                return true;
              }
              return false;
            }
            if (event.inputType !== "deleteContentBackward" && event.inputType !== "deleteContentForward") {
              return false;
            }
            const direction = event.inputType === "deleteContentBackward" ? -1 : 1;
            if (deleteEmptyTensBlock(view) || protectTensSentinelBoundary(view, direction)) {
              event.preventDefault();
              return true;
            }
            return false;
          },
          paste(event, view) {
            if (!isMarkdownAtPos(view.state.selection.main.head)) return false;
            const item = [...(event.clipboardData?.items ?? [])].find((it) =>
              it.type.startsWith("image/"),
            );
            if (!item) return false;
            event.preventDefault();
            const file = item.getAsFile();
            if (!file) return true;
            const reader = new FileReader();
            reader.onload = () => insertAtSelection(`![pasted image](${reader.result})`);
            reader.readAsDataURL(file);
            return true;
          },
        }),
        keymap.of([
          { key: "Backspace", run: deleteEmptyTensBlock },
          { key: "Backspace", run: (view) => protectTensSentinelBoundary(view, -1) },
          { key: "Delete", run: deleteEmptyTensBlock },
          { key: "Delete", run: (view) => protectTensSentinelBoundary(view, 1) },
          { key: "Enter", run: exitTensBlockOnDoubleBlankEnter },
          { key: "Tab", run: acceptCompletion },
          indentWithTab,
          ...completionKeymap,
          ...searchKeymap,
          ...historyKeymap,
          ...defaultKeymap,
        ]),
      ],
    }),
  });
  editorView.scrollDOM.addEventListener("scroll", syncOutputToEditor);
}

function syncSourceFromBlocks() {
  refreshDocumentModel();
}

function renderSourceEditor() {}

function resizeAllTextareas() {
  editorView?.requestMeasure();
}

function activeSourceLine() {
  if (!editorView) return null;
  return editorView.state.doc.lineAt(editorView.state.selection.main.head).number;
}

function tensBlockAroundLine(doc, lineNumber) {
  if (!doc || lineNumber < 1 || lineNumber > doc.lines) return null;
  let openLine = null;
  for (let n = lineNumber; n >= 1; n--) {
    const line = doc.line(n);
    const sentinel = tensSentinelKind(line.text);
    if (sentinel === "close" && n !== lineNumber) return null;
    if (sentinel === "open") {
      openLine = n;
      break;
    }
  }
  if (openLine == null) return null;
  for (let n = openLine + 1; n <= doc.lines; n++) {
    const line = doc.line(n);
    const sentinel = tensSentinelKind(line.text);
    if (sentinel === "open") return null;
    if (sentinel === "close") {
      if (lineNumber >= openLine && lineNumber <= n) {
        return { openLine, closeLine: n };
      }
      return null;
    }
  }
  return null;
}

function isEmptyTensBlock(doc, block) {
  if (!block || block.closeLine <= block.openLine + 1) return true;
  for (let n = block.openLine + 1; n < block.closeLine; n++) {
    if (doc.line(n).text.trim() !== "") return false;
  }
  return true;
}

function nearestEditableDocPos(doc, lineNumber, direction) {
  const forwardStart = direction >= 0 ? lineNumber + 1 : lineNumber - 1;
  for (
    let n = forwardStart;
    direction >= 0 ? n <= doc.lines : n >= 1;
    n += direction >= 0 ? 1 : -1
  ) {
    if (isSentinelDocLine(doc, n)) continue;
    const line = doc.line(n);
    return direction >= 0 ? line.from : line.to;
  }
  for (
    let n = direction >= 0 ? lineNumber - 1 : lineNumber + 1;
    direction >= 0 ? n >= 1 : n <= doc.lines;
    n += direction >= 0 ? -1 : 1
  ) {
    if (isSentinelDocLine(doc, n)) continue;
    const line = doc.line(n);
    return direction >= 0 ? line.to : line.from;
  }
  return 0;
}

function deleteBlankTensContentLine(view, line, block, direction) {
  const doc = view.state.doc;
  if (!block || line.number <= block.openLine || line.number >= block.closeLine) return false;
  if (line.text.trim() !== "" || isEmptyTensBlock(doc, block)) return false;

  let from = line.from;
  let to = line.to;
  if (direction < 0) {
    if (line.number > block.openLine + 1) {
      from = doc.line(line.number - 1).to;
    } else if (line.number + 1 < block.closeLine) {
      to = doc.line(line.number + 1).from;
    }
  } else if (line.number + 1 < block.closeLine) {
    to = doc.line(line.number + 1).from;
  } else if (line.number > block.openLine + 1) {
    from = doc.line(line.number - 1).to;
  }

  if (from >= to) return false;
  view.dispatch({
    changes: { from, to, insert: "" },
    selection: { anchor: from },
    annotations: internalEdit.of(true),
  });
  view.focus();
  return true;
}

function lastTensContentLineBeforeClose(doc, closeLineNumber) {
  const block = tensBlockAroundLine(doc, closeLineNumber);
  if (!block) return null;
  for (let n = block.closeLine - 1; n > block.openLine; n--) {
    const line = doc.line(n);
    if (line.text.trim() !== "") return line;
  }
  return block.closeLine > block.openLine + 1 ? doc.line(block.openLine + 1) : null;
}

function deleteBlankMarkdownLineAfterTens(view, line, direction) {
  const selection = view.state.selection.main;
  const doc = view.state.doc;
  if (direction >= 0 || selection.head !== line.from || line.text.trim() !== "") return false;
  if (line.number <= 1) return false;

  const previous = doc.line(line.number - 1);
  if (tensSentinelKind(previous.text) !== "close") return false;

  const target = lastTensContentLineBeforeClose(doc, previous.number);
  if (!target) return false;

  const nextFrom = line.number < doc.lines ? doc.line(line.number + 1).from : line.to;
  const from = line.number < doc.lines ? line.from : previous.to;
  const to = line.number < doc.lines ? nextFrom : line.to;
  if (from >= to) return false;

  view.dispatch({
    changes: { from, to, insert: "" },
    selection: { anchor: target.to },
    annotations: internalEdit.of(true),
  });
  view.focus();
  return true;
}

function protectTensSentinelBoundary(view, direction) {
  const selection = view.state.selection.main;
  if (!selection.empty) return false;
  const doc = view.state.doc;
  const line = doc.lineAt(selection.head);

  if (isSentinelDocLine(doc, line.number)) {
    view.dispatch({
      selection: { anchor: nearestEditableDocPos(doc, line.number, direction) },
      annotations: internalEdit.of(true),
    });
    view.focus();
    return true;
  }

  const block = tensBlockAroundLine(doc, line.number);
  if (deleteBlankTensContentLine(view, line, block, direction)) return true;
  if (deleteBlankMarkdownLineAfterTens(view, line, direction)) return true;

  if (direction < 0 && selection.head === line.from && line.number > 1) {
    const previous = doc.line(line.number - 1);
    if (tensSentinelKind(previous.text) != null) return true;
  }
  if (direction > 0 && selection.head === line.to && line.number < doc.lines) {
    const next = doc.line(line.number + 1);
    if (tensSentinelKind(next.text) != null) return true;
  }
  return false;
}

function deleteEmptyTensBlock(view) {
  const selection = view.state.selection.main;
  if (!selection.empty) return false;
  const doc = view.state.doc;
  const line = doc.lineAt(selection.head);
  const block = tensBlockAroundLine(doc, line.number);
  if (!block || line.number <= block.openLine || line.number >= block.closeLine) return false;
  if (!isEmptyTensBlock(doc, block)) return false;

  const fromLine = doc.line(block.openLine);
  const closeLine = doc.line(block.closeLine);
  const afterClose = block.closeLine < doc.lines ? doc.line(block.closeLine + 1).from : closeLine.to;
  const from = fromLine.from;
  const to = afterClose;
  view.dispatch({
    changes: { from, to, insert: "" },
    selection: { anchor: from },
    annotations: internalEdit.of(true),
  });
  view.focus();
  return true;
}

function exitTensBlockOnDoubleBlankEnter(view) {
  const selection = view.state.selection.main;
  if (!selection.empty) return false;
  const doc = view.state.doc;
  const line = doc.lineAt(selection.head);
  const block = tensBlockAroundLine(doc, line.number);
  if (!block || line.number <= block.openLine || line.number >= block.closeLine) return false;
  if (line.text.trim() !== "") return false;

  const beforeLines = [];
  const afterLines = [];
  for (let n = block.openLine + 1; n < line.number; n++) beforeLines.push(doc.line(n).text);
  for (let n = line.number + 1; n < block.closeLine; n++) afterLines.push(doc.line(n).text);
  const beforeHasCode = beforeLines.some((text) => text.trim() !== "");
  const afterHasCode = afterLines.some((text) => text.trim() !== "");

  const openLine = doc.line(block.openLine);
  const closeLine = doc.line(block.closeLine);
  const to = block.closeLine < doc.lines ? doc.line(block.closeLine + 1).from : closeLine.to;
  const from = beforeHasCode ? line.from : openLine.from;
  let insert = beforeHasCode ? `${TENS_CLOSE}\n\n` : "\n";
  const anchor = beforeHasCode ? from + TENS_CLOSE.length + 1 : from;

  if (afterHasCode) {
    const afterText = afterLines.join("\n");
    insert += `${TENS_OPEN}\n${afterText}\n${TENS_CLOSE}`;
    if (block.closeLine < doc.lines) insert += "\n";
  }

  view.dispatch({
    changes: { from, to, insert },
    selection: { anchor },
    annotations: internalEdit.of(true),
  });
  view.focus();
  return true;
}

function endOfTensBlockAfter(lineNumber) {
  if (!editorView) return null;
  const doc = editorView.state.doc;
  for (let n = lineNumber; n <= doc.lines; n++) {
    const line = doc.line(n);
    if (isTensClose(line.text)) return line.to;
  }
  return null;
}

function focusEmptyTensBlock(block) {
  if (!editorView || !block || !isEmptyTensBlock(editorView.state.doc, block)) return false;
  const contentLine = editorView.state.doc.line(block.openLine + 1);
  editorView.dispatch({
    selection: { anchor: contentLine.from },
    annotations: internalEdit.of(true),
  });
  editorView.focus();
  return true;
}

function emptyTensBlockNearLine(doc, lineNumber) {
  const current = tensBlockAroundLine(doc, lineNumber);
  if (current && isEmptyTensBlock(doc, current)) return current;

  if (lineNumber > 1 && tensSentinelKind(doc.line(lineNumber - 1).text) === "close") {
    const previous = tensBlockAroundLine(doc, lineNumber - 1);
    if (previous && isEmptyTensBlock(doc, previous)) return previous;
  }

  if (lineNumber < doc.lines && tensSentinelKind(doc.line(lineNumber + 1).text) === "open") {
    const next = tensBlockAroundLine(doc, lineNumber + 1);
    if (next && isEmptyTensBlock(doc, next)) return next;
  }

  return null;
}

function insertTensBlock() {
  if (!editorView) return;
  const template = `${TENS_OPEN}\n${MARKDOWN_CURSOR}\n${TENS_CLOSE}`;
  const sel = editorView.state.selection.main;
  const line = editorView.state.doc.lineAt(sel.head);
  if (focusEmptyTensBlock(emptyTensBlockNearLine(editorView.state.doc, line.number))) return;
  const kind = regionKindAtLine(line.number);
  if (kind === "markdown") {
    insertAtSelection(template, { block: true });
    return;
  }
  const pos = endOfTensBlockAfter(line.number) ?? line.to;
  insertTextAtRange(pos, pos, template, { block: true });
}

function sourceLineLocation(line) {
  if (!editorView || !Number.isFinite(line)) return null;
  const n = Math.max(1, Math.min(editorView.state.doc.lines, line));
  const docLine = editorView.state.doc.line(n);
  return { line: docLine, offset: docLine.from };
}

function focusSourceLine(line) {
  const loc = sourceLineLocation(line);
  if (!loc) return;
  editorView.dispatch({
    selection: { anchor: loc.offset },
    effects: EditorView.scrollIntoView(loc.offset, { y: "center" }),
    annotations: internalEdit.of(true),
  });
  editorView.focus();
}

function scrollEditorToLine(line, options = {}) {
  const loc = sourceLineLocation(line);
  if (!loc) return;
  editorView.dispatch({
    effects: EditorView.scrollIntoView(loc.offset, { y: options.align ?? "center" }),
    annotations: internalEdit.of(true),
  });
}

function jumpToSourceLine(line, options = {}) {
  if (options.sync === false) suppressScrollSync();
  if (options.focus) focusSourceLine(line);
  else scrollEditorToLine(line, { align: "start" });
}

// ---- files -----------------------------------------------------------------

async function openFile() {
  if (!invoke) return;
  if (!confirmDiscardUnsaved()) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return;
  loadOpenedFile(opened);
}

function newFile() {
  if (!confirmDiscardUnsaved()) return;
  setCurrentPath(null);
  setDocumentSource("", { run: false, dirty: false });
  lastRenderedOutputs = [];
  lastGoodShown = false;
  output.innerHTML = '<div class="placeholder">No output yet. Add expr.show(...).</div>';
  railFiles.querySelectorAll(".file.active").forEach((el) => el.classList.remove("active"));
  editorView?.scrollDOM.scrollTo({ top: 0 });
  output.scrollTop = 0;
  editorView?.focus();
}

function loadOpenedFile(opened) {
  const savePath = Object.prototype.hasOwnProperty.call(opened, "save_path")
    ? opened.save_path
    : Object.prototype.hasOwnProperty.call(opened, "savePath")
      ? opened.savePath
      : opened.path;
  setCurrentPath(opened.path, { savePath });
  setDocumentSource(opened.source, { dirty: false });
  renderFileRail(opened.folder, opened.files);
}

function renderFileRail(folder, files) {
  if (!folder || !files || files.length === 0) {
    railFolder.textContent = "";
    const empty = document.createElement("div");
    empty.className = "rail-empty";
    empty.textContent = "Open a .tens or Markdown file to browse its folder here.";
    railFiles.replaceChildren(empty);
    return;
  }
  localStorage.setItem("tensorforge.folder", folder);
  railFolder.textContent = folder;
  railFolder.title = folder;
  railFiles.replaceChildren(
    ...files.map((f) => {
      const div = document.createElement("div");
      div.className = "file" + (f.path === currentPath ? " active" : "");
      div.textContent = f.name;
      div.title = f.path;
      div.addEventListener("click", () => switchToFile(f.path));
      return div;
    }),
  );
}

async function restoreFileRail() {
  const folder = localStorage.getItem("tensorforge.folder");
  if (!invoke || !folder) return;
  const files = await invoke("list_folder", { path: folder }).catch(() => null);
  if (files) renderFileRail(folder, files);
}

async function switchToFile(path) {
  if (!invoke || path === currentPath) return;
  if (!confirmDiscardUnsaved()) return;
  const opened = await invoke("read_tens", { path }).catch(showError);
  if (opened) loadOpenedFile(opened);
}

async function saveFile() {
  if (!invoke) return;
  syncSourceFromBlocks();
  const path = await invoke("save_tens", {
    source: documentSource(),
    path: currentSavePath,
    defaultFilename: defaultSaveName(),
  }).catch(showError);
  if (path) {
    setCurrentPath(path, { savePath: path });
    markClean();
    await refreshFolderListing(path);
  }
}

async function refreshFolderListing(path) {
  if (!invoke || !path) return;
  const opened = await invoke("read_tens", { path }).catch(() => null);
  if (opened) renderFileRail(opened.folder, opened.files);
}

// ---- layout ----------------------------------------------------------------

const railResizer = document.getElementById("rail-resizer");
const storedRailWidth = Number(localStorage.getItem("tensorforge.railWidth"));
if (storedRailWidth >= 90) fileRail.style.flexBasis = `${storedRailWidth}px`;
applyRailCollapsed(localStorage.getItem("tensorforge.railCollapsed") === "true", { persist: false });
railToggle.addEventListener("click", () => {
  applyRailCollapsed(!mainEl.classList.contains("rail-collapsed"));
});

function applyRailCollapsed(collapsed, options = {}) {
  if (collapsed && fileRail.offsetWidth > 60) {
    localStorage.setItem("tensorforge.railWidth", String(fileRail.offsetWidth));
  } else if (!collapsed) {
    const width = Number(localStorage.getItem("tensorforge.railWidth"));
    fileRail.style.flexBasis = `${width >= 90 ? width : 160}px`;
  }
  mainEl.classList.toggle("rail-collapsed", collapsed);
  railToggle.textContent = collapsed ? "›" : "‹";
  railToggle.title = collapsed ? "Expand file sidebar" : "Collapse file sidebar";
  railToggle.setAttribute("aria-label", railToggle.title);
  if (options.persist !== false) {
    localStorage.setItem("tensorforge.railCollapsed", String(collapsed));
  }
}

railResizer.addEventListener("pointerdown", (e) => {
  if (mainEl.classList.contains("rail-collapsed")) return;
  e.preventDefault();
  railResizer.setPointerCapture(e.pointerId);
  railResizer.classList.add("dragging");
  const move = (ev) => {
    const width = Math.min(420, Math.max(90, ev.clientX - fileRail.getBoundingClientRect().left));
    fileRail.style.flexBasis = `${width}px`;
  };
  railResizer.addEventListener("pointermove", move);
  railResizer.addEventListener(
    "pointerup",
    () => {
      railResizer.removeEventListener("pointermove", move);
      railResizer.classList.remove("dragging");
      localStorage.setItem("tensorforge.railWidth", String(fileRail.offsetWidth));
    },
    { once: true },
  );
});

const storedEditorWidth = Number(localStorage.getItem("tensorforge.editorWidth"));
if (storedEditorWidth >= 280) editorWrap.style.flexBasis = `${storedEditorWidth}px`;

function editorWidthBounds() {
  const mainRect = document.querySelector("main").getBoundingClientRect();
  const left = editorWrap.getBoundingClientRect().left;
  const minEditor = 300;
  const minOutput = 360;
  const maxEditor = Math.max(minEditor, mainRect.right - left - minOutput);
  return { minEditor, maxEditor };
}

function setEditorWidthFromClientX(clientX) {
  const { minEditor, maxEditor } = editorWidthBounds();
  const left = editorWrap.getBoundingClientRect().left;
  const width = Math.min(maxEditor, Math.max(minEditor, clientX - left));
  editorWrap.style.flexBasis = `${width}px`;
  requestAnimationFrame(resizeAllTextareas);
}

let splitDragActive = false;
splitResizer.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  splitDragActive = true;
  splitResizer.setPointerCapture(e.pointerId);
  splitResizer.classList.add("dragging");
  const move = (ev) => setEditorWidthFromClientX(ev.clientX);
  splitResizer.addEventListener("pointermove", move);
  splitResizer.addEventListener(
    "pointerup",
    () => {
      splitDragActive = false;
      splitResizer.removeEventListener("pointermove", move);
      splitResizer.classList.remove("dragging");
      localStorage.setItem("tensorforge.editorWidth", String(editorWrap.offsetWidth));
    },
    { once: true },
  );
});

splitResizer.addEventListener("mousedown", (e) => {
  if (splitDragActive) return;
  e.preventDefault();
  splitDragActive = true;
  splitResizer.classList.add("dragging");
  const move = (ev) => setEditorWidthFromClientX(ev.clientX);
  const up = () => {
    splitDragActive = false;
    document.removeEventListener("mousemove", move);
    splitResizer.classList.remove("dragging");
    localStorage.setItem("tensorforge.editorWidth", String(editorWrap.offsetWidth));
  };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up, { once: true });
});

// ---- markdown rendering ----------------------------------------------------

function isInnerFenceOpen(line) {
  const trimmed = line.trim();
  return trimmed.startsWith("```") && trimmed !== "```";
}

function isBareFenceClose(line) {
  return /^```\s*$/.test(line.trim());
}

function sourceLineAttr(line) {
  return Number.isFinite(line) ? ` data-source-line="${line}"` : "";
}

function renderMarkdown(markdown, baseLine = null) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const sourceLine = baseLine == null ? null : baseLine + i;
    if (isInnerFenceOpen(line)) {
      const startLine = sourceLine;
      const code = [];
      i++;
      while (i < lines.length && !isBareFenceClose(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      out.push(`<pre class="md-code"${sourceLineAttr(startLine)}><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const oneLineMath = line.match(/^\s*\$\$\s*(.*?)\s*\$\$\s*$/);
    if (oneLineMath) {
      out.push(`<div class="markdown-math"${sourceLineAttr(sourceLine)}>${renderMath(oneLineMath[1], true)}</div>`);
      continue;
    }
    if (line.trim() === "$$") {
      const startLine = sourceLine;
      const tex = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") {
        tex.push(lines[i]);
        i++;
      }
      out.push(`<div class="markdown-math"${sourceLineAttr(startLine)}>${renderMath(tex.join("\n"), true)}</div>`);
      continue;
    }
    if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const startLine = sourceLine;
      const aligns = tableAlignments(lines[i + 1]);
      const header = tableCells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(tableCells(lines[i]));
        i++;
      }
      i--;
      out.push(renderTable(header, rows, aligns, startLine));
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}${sourceLineAttr(sourceLine)}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      out.push(`<hr${sourceLineAttr(sourceLine)}>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const startLine = sourceLine;
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const itemLine = baseLine == null ? null : baseLine + i;
        items.push(`<li${sourceLineAttr(itemLine)}>${inlineMarkdown(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      i--;
      out.push(`<ul${sourceLineAttr(startLine)}>${items.join("")}</ul>`);
      continue;
    }
    if (line.trim() === "") continue;
    out.push(`<p${sourceLineAttr(sourceLine)}>${inlineMarkdown(line)}</p>`);
  }
  return out.join("") || "<p></p>";
}

function isTableRow(line) {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.length > 1;
}

function isTableSeparator(line) {
  const t = line.trim();
  if (!isTableRow(t)) return false;
  return tableCells(t).every((c) => /^:?-{3,}:?$/.test(c.trim()) || c.trim() === "");
}

function tableCells(line) {
  const t = line.trim();
  return t.slice(1, t.length - 1).split("|");
}

function tableAlignments(sepLine) {
  return tableCells(sepLine).map((c) => {
    const t = c.trim();
    if (t.startsWith(":") && t.endsWith(":")) return "center";
    if (t.endsWith(":")) return "right";
    return "left";
  });
}

function renderTable(header, rows, aligns, sourceLine = null) {
  const cell = (tag, content, k) =>
    `<${tag} style="text-align:${aligns[k] ?? "left"}">${inlineMarkdown(content.trim())}</${tag}>`;
  return `<table class="md-table"${sourceLineAttr(sourceLine)}><thead><tr>${header
    .map((c, k) => cell("th", c, k))
    .join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c, k) => cell("td", c, k)).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function inlineMarkdown(text) {
  const parts = [];
  let plain = "";
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        if (plain) parts.push(renderInlinePlain(plain));
        plain = "";
        parts.push(`<code>${escapeHtml(text.slice(i + 1, end))}</code>`);
        i = end;
        continue;
      }
    }
    if (text[i] === "$" && text[i + 1] !== "$") {
      const end = findInlineMathEnd(text, i + 1);
      if (end !== -1) {
        if (plain) parts.push(renderInlinePlain(plain));
        plain = "";
        parts.push(renderMath(text.slice(i + 1, end), false));
        i = end;
        continue;
      }
    }
    plain += text[i];
  }
  if (plain) parts.push(renderInlinePlain(plain));
  return parts.join("");
}

function findInlineMathEnd(text, start) {
  for (let i = start; i < text.length; i++) {
    if (text[i] === "$" && text[i - 1] !== "\\") return i;
  }
  return -1;
}

function renderInlinePlain(text) {
  const parts = [];
  let rest = text;
  const img = /!\[([^\]]*)\]\((data:image\/[a-zA-Z+.-]+;base64,[A-Za-z0-9+/=]+)\)/;
  for (let m = rest.match(img); m; m = rest.match(img)) {
    parts.push(inlineStyles(rest.slice(0, m.index)));
    parts.push(`<img class="md-img" alt="${escapeAttr(m[1])}" src="${m[2]}">`);
    rest = rest.slice(m.index + m[0].length);
  }
  const link = /\[([^\]]+)\]\(([^)\s]+)\)/;
  for (let m = rest.match(link); m; m = rest.match(link)) {
    parts.push(inlineStyles(rest.slice(0, m.index)));
    const href = safeMarkdownHref(m[2]);
    if (href) {
      parts.push(`<a href="${escapeAttr(href)}">${inlineStyles(m[1])}</a>`);
    } else {
      parts.push(inlineStyles(m[0]));
    }
    rest = rest.slice(m.index + m[0].length);
  }
  parts.push(inlineStyles(rest));
  return parts.join("");
}

function safeMarkdownHref(raw) {
  try {
    const url = new URL(raw);
    return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function inlineStyles(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderMath(tex, displayMode) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      macros: KATEX_MACROS,
      throwOnError: true,
    });
  } catch {
    return `<code>${escapeHtml(tex)}</code>`;
  }
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// ---- output/export ---------------------------------------------------------

function copyButton(label, text) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.textContent = label;
  btn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(text);
    btn.textContent = "copied";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = label;
      btn.classList.remove("copied");
    }, 1200);
  });
  return btn;
}

function renderMarkdownBlock(block) {
  const el = document.createElement("div");
  el.className = "block markdown-doc";
  el.dataset.sourceBlockId = String(block.id);
  el.dataset.sourceLine = String(block.sourceLine);
  el.innerHTML = renderMarkdown(block.text, block.sourceLine);
  makeBlockJump(el, block.sourceLine);
  return el;
}

function sourceTextAtLine(line) {
  if (!editorView || !Number.isFinite(line)) return "";
  const doc = editorView.state.doc;
  if (line < 1 || line > doc.lines) return "";
  return doc.line(line).text;
}

function looksLikeAccidentalNotesInTens(lineText, error) {
  const text = lineText.trim();
  if (!text || !/^undefined variable\b/i.test(String(error))) return false;
  if (/[\u3400-\u9fff]/.test(text)) return true;
  if (/[=^*/+&|<>\\[\]{}]/.test(text)) return false;
  return /^[\p{L}\p{N}_\s.,;:!?'"“”‘’()（）\-—]+$/u.test(text);
}

function tensBlockNoteHint(line, error) {
  if (!Number.isFinite(line) || !editorView) return null;
  const sourceBlock = blockForSourceLine(line);
  if (!sourceBlock || sourceBlock.kind !== "tens") return null;
  if (isSentinelDocLine(editorView.state.doc, line)) return null;
  const lineText = sourceTextAtLine(line);
  if (!looksLikeAccidentalNotesInTens(lineText, error)) return null;
  return "This line is inside a TensorForge code block. Press Enter on an empty tens line to return to Markdown.";
}

function viewSourceRegionsForBlock(block) {
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

function viewSourceRegionForLine(line) {
  if (!Number.isFinite(line)) return null;
  const block = blockForSourceLine(line);
  if (!block || block.kind !== "tens") return null;
  return (
    viewSourceRegionsForBlock(block).find(
      (region) => line >= region.sourceLine && line <= region.sourceEndLine,
    ) ?? null
  );
}

function outputLineForPreviewBlock(block) {
  if (block.kind === "output") return block.line;
  if (block.kind === "output-row") {
    return block.items.find((item) => Number.isFinite(item.line))?.line;
  }
  return null;
}

function viewSourceRegionKey(region) {
  return region ? `${region.blockId}:${region.sourceLine}:${region.sourceEndLine}` : "";
}

function groupViewSourcePreviewBlocks(blocks) {
  const grouped = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    const region = viewSourceRegionForLine(outputLineForPreviewBlock(block));
    if (!region) {
      grouped.push(block);
      continue;
    }

    const key = viewSourceRegionKey(region);
    const items = [block];
    while (i + 1 < blocks.length) {
      const nextRegion = viewSourceRegionForLine(outputLineForPreviewBlock(blocks[i + 1]));
      if (viewSourceRegionKey(nextRegion) !== key) break;
      items.push(blocks[++i]);
    }
    grouped.push({ kind: "view-source", region, items });
  }
  return grouped;
}

function renderOutputBlock(item) {
  const { header, latex, line, error } = item;
  const block = document.createElement("div");
  block.className = "block";
  const sourceBlock = Number.isFinite(line) ? blockForSourceLine(line) : null;
  if (sourceBlock) block.dataset.sourceBlockId = String(sourceBlock.id);
  if (Number.isFinite(line)) block.dataset.sourceLine = String(line);
  makeBlockJump(block, line);
  const head = document.createElement("div");
  head.className = "head";

  if (error) {
    head.textContent = `[line ${line}]`;
    const err = document.createElement("div");
    err.className = "error";
    err.textContent = error;
    const noteHint = tensBlockNoteHint(line, error);
    if (noteHint) {
      const hint = document.createElement("div");
      hint.textContent = noteHint;
      hint.style.marginTop = "8px";
      hint.style.color = "var(--muted)";
      hint.style.font = '12px/1.4 -apple-system, "Segoe UI", sans-serif';
      err.appendChild(hint);
    }
    block.append(head, err);
    return block;
  }

  const label = document.createElement("span");
  label.className = "head-label";
  label.textContent = `[${header}]`;
  head.appendChild(label);
  const tex = stripDisplayMath(latex);
  const bar = document.createElement("span");
  bar.className = "copy-bar";
  bar.append(copyButton("copy latex", tex));
  head.appendChild(bar);

  const math = document.createElement("div");
  math.className = "math";
  try {
    katex.render(tex, math, {
      displayMode: true,
      macros: KATEX_MACROS,
      throwOnError: true,
    });
  } catch (e) {
    math.innerHTML = `<div class="error">KaTeX: ${escapeHtml(e.message)}</div><pre>${escapeHtml(tex)}</pre>`;
  }
  block.append(head, math);
  return block;
}

function renderViewSourceBlock(group) {
  const wrapper = document.createElement("div");
  wrapper.className = "block tens-source-preview";
  wrapper.dataset.sourceBlockId = String(group.region.blockId);
  wrapper.dataset.sourceLine = String(group.region.sourceLine);
  makeBlockJump(wrapper, group.region.sourceLine);

  const sourcePane = document.createElement("div");
  sourcePane.className = "tens-source-pane";
  const sourceHead = document.createElement("div");
  sourceHead.className = "tens-source-head";
  sourceHead.textContent = "source";
  const pre = document.createElement("pre");
  pre.className = "tens-source-code";
  const code = document.createElement("code");
  code.textContent = group.region.sourceText;
  pre.appendChild(code);
  sourcePane.append(sourceHead, pre);

  const renderedPane = document.createElement("div");
  renderedPane.className = "tens-render-pane";
  const renderHead = document.createElement("div");
  renderHead.className = "tens-source-head";
  renderHead.textContent = "preview";
  renderedPane.appendChild(renderHead);
  for (const item of group.items) {
    if (item.kind === "output-row") renderedPane.appendChild(renderOutputRowBlock(item.items));
    else renderedPane.appendChild(renderOutputBlock(item));
  }

  wrapper.append(sourcePane, renderedPane);
  return wrapper;
}

function renderOutputRowBlock(items) {
  const row = document.createElement("div");
  row.className = "output-row";
  const firstLine = items.find((item) => Number.isFinite(item.line))?.line;
  const sourceBlock = Number.isFinite(firstLine) ? blockForSourceLine(firstLine) : null;
  if (sourceBlock) row.dataset.sourceBlockId = String(sourceBlock.id);
  if (Number.isFinite(firstLine)) row.dataset.sourceLine = String(firstLine);
  for (const item of items) row.appendChild(renderOutputBlock(item));
  return row;
}

function countOutputCalls(source) {
  return source
    .split(/\r?\n/)
    .reduce(
      (count, line) =>
        count +
        (line.replace(/#.*/, "").match(/\.(?:show|classify|solve)\s*\(/g)?.length ?? 0),
      0,
    );
}

function orderedPreviewBlocks(outputs) {
  const items = outputs.map((item, index) => ({ kind: "output", index, ...item }));
  const buckets = new Map(docBlocks.map((block) => [block.id, []]));
  const expected = new Map(
    docBlocks
      .filter((block) => block.kind === "tens")
      .map((block) => [block.id, countOutputCalls(block.text)]),
  );
  const assigned = new Set();
  let sequentialBlockIndex = 0;

  const tensBlocks = docBlocks.filter((block) => block.kind === "tens");
  const outputInBlock = (item, block) =>
    Number.isFinite(item.line) &&
    item.line >= block.sourceLine &&
    item.line <= block.sourceEndLine;

  const put = (item, block) => {
    buckets.get(block.id)?.push(item);
    assigned.add(item.index);
  };

  for (const item of items) {
    const block = tensBlocks.find((candidate) => outputInBlock(item, candidate));
    if (block) put(item, block);
  }

  for (const item of items) {
    if (assigned.has(item.index)) continue;

    while (
      sequentialBlockIndex < tensBlocks.length &&
      (buckets.get(tensBlocks[sequentialBlockIndex].id)?.filter((b) => !b.error).length ?? 0) >=
        (expected.get(tensBlocks[sequentialBlockIndex].id) ?? 0)
    ) {
      sequentialBlockIndex++;
    }

    const block = tensBlocks[sequentialBlockIndex];
    if (block) put(item, block);
  }

  const blocks = [];

  for (const docBlock of docBlocks) {
    if (docBlock.kind === "markdown" && docBlock.text.trim()) {
      blocks.push({ kind: "markdown", ...docBlock });
      continue;
    }

    if (docBlock.kind !== "tens") continue;
    blocks.push(...(buckets.get(docBlock.id) ?? []));
  }

  for (const item of items) {
    if (!assigned.has(item.index)) blocks.push(item);
  }

  return blocks;
}

function groupOutputRows(blocks) {
  const grouped = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    if (block.kind !== "output" || !block.row || block.error) {
      grouped.push(block);
      continue;
    }
    const items = [block];
    while (
      i + 1 < blocks.length &&
      blocks[i + 1].kind === "output" &&
      blocks[i + 1].row === block.row &&
      !blocks[i + 1].error
    ) {
      items.push(blocks[++i]);
    }
    grouped.push({ kind: "output-row", row: block.row, items });
  }
  return grouped;
}

function renderOutputs(outputs, options = {}) {
  const preserveScroll = options.preserveScroll === true;
  const editorTop = editorView?.scrollDOM.scrollTop ?? 0;
  const outputTop = output.scrollTop;
  if (preserveScroll) suppressScrollSync();
  output.innerHTML = "";
  lastRenderedOutputs = outputs;
  const blocks = groupViewSourcePreviewBlocks(groupOutputRows(orderedPreviewBlocks(outputs)));

  if (blocks.length === 0) {
    output.innerHTML = '<div class="placeholder">No output yet. Add expr.show(...).</div>';
    if (preserveScroll) {
      output.scrollTop = outputTop;
      editorView?.scrollDOM.scrollTo({ top: editorTop });
    }
    return;
  }
  for (const block of blocks) {
    if (block.kind === "markdown") output.appendChild(renderMarkdownBlock(block));
    else if (block.kind === "view-source") output.appendChild(renderViewSourceBlock(block));
    else if (block.kind === "output-row") output.appendChild(renderOutputRowBlock(block.items));
    else output.appendChild(renderOutputBlock(block));
  }
  if (preserveScroll) {
    output.scrollTop = clampScrollTop(output, outputTop);
    editorView?.scrollDOM.scrollTo({ top: editorTop });
  } else {
    syncOutputToEditor();
  }
}

function showError(message) {
  output.innerHTML = "";
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = String(message);
  output.appendChild(div);
  lastGoodShown = false;
}

async function run() {
  syncSourceFromBlocks();
  if (!invoke) {
    renderOutputs([]);
    return;
  }
  const source = executableSource();
  if (!source) {
    renderOutputs([]);
    lastGoodShown = true;
    return;
  }
  const result = await invoke("run_tens", { source });
  if (!result.ok) {
    showError(result.error);
    return;
  }
  renderOutputs(result.outputs);
}

async function liveRun() {
  syncSourceFromBlocks();
  if (!invoke) {
    renderOutputs([], { preserveScroll: true });
    return;
  }
  const source = executableSource();
  if (!source) {
    renderOutputs([], { preserveScroll: true });
    lastGoodShown = true;
    return;
  }
  const result = await invoke("run_tens", { source });
  if (!result.ok) {
    if (!lastGoodShown) showError(result.error);
    else {
      let banner = output.querySelector(".parse-banner");
      if (!banner) {
        banner = document.createElement("div");
        banner.className = "error parse-banner";
        output.prepend(banner);
      }
      banner.textContent = result.error;
    }
    return;
  }
  renderOutputs(result.outputs, { preserveScroll: true });
  lastGoodShown = true;
}

function scheduleLiveRun() {
  clearTimeout(liveTimer);
  liveTimer = setTimeout(liveRun, 350);
}

// ---- signature hint -------------------------------------------------------
// A small floating tooltip showing the parameter list of the builtin call the
// caret sits inside (and the valid modes for X.show(...)). Driven purely from
// the editor selection — no backend round-trip, no other completion clutter.

let signatureHintEl = null;

function ensureSignatureHintEl() {
  if (signatureHintEl) return signatureHintEl;
  const el = document.createElement("div");
  el.className = "tf-sig-hint";
  el.style.cssText = [
    "position:fixed",
    "z-index:40",
    "display:none",
    "max-width:min(560px, 80vw)",
    "padding:4px 9px",
    "border:1px solid var(--border)",
    "border-radius:7px",
    "background:var(--panel-2)",
    "color:var(--muted)",
    "font-family:ui-monospace, SFMono-Regular, Menlo, monospace",
    "font-size:12px",
    "line-height:1.5",
    "white-space:nowrap",
    "overflow:hidden",
    "text-overflow:ellipsis",
    "box-shadow:0 6px 18px rgba(0,0,0,0.18)",
    "pointer-events:none",
  ].join(";");
  document.body.appendChild(el);
  signatureHintEl = el;
  return el;
}

function hideSignatureHint() {
  if (signatureHintEl) signatureHintEl.style.display = "none";
}

function renderSignatureHint(el, hint) {
  el.replaceChildren();
  const signature = document.createElement("div");
  const open = document.createElement("span");
  open.textContent = `${hint.fn}(`;
  open.style.color = "var(--text)";
  signature.appendChild(open);
  const sep = hint.choice ? " | " : ", ";
  hint.params.forEach((param, i) => {
    if (i > 0) signature.appendChild(document.createTextNode(sep));
    const span = document.createElement("span");
    span.textContent = param;
    if ((!hint.choice && i === hint.argIndex) || (hint.choice && i === hint.argIndex)) {
      span.style.color = "var(--accent)";
      span.style.fontWeight = "600";
    } else {
      span.style.color = "var(--muted)";
    }
    signature.appendChild(span);
  });
  const close = document.createElement("span");
  close.textContent = ")";
  close.style.color = "var(--text)";
  signature.appendChild(close);
  el.appendChild(signature);

  if (hint.detail) {
    const detail = document.createElement("div");
    detail.textContent = hint.detail;
    detail.style.color = "var(--muted)";
    detail.style.fontSize = "11px";
    detail.style.marginTop = "1px";
    el.appendChild(detail);
  }
}

function updateSignatureHint() {
  if (!editorView) return hideSignatureHint();
  const sel = editorView.state.selection.main;
  if (!sel.empty) return hideSignatureHint();
  const pos = sel.head;
  if (regionKindAtPos(pos) !== "tens") return hideSignatureHint();
  const line = editorView.state.doc.lineAt(pos);
  const prefix = editorView.state.doc.sliceString(line.from, pos);
  const hint = signatureHint(prefix, editorView.state.doc.toString(), pos);
  if (!hint) return hideSignatureHint();
  const coords = editorView.coordsAtPos(pos);
  if (!coords) return hideSignatureHint();

  const el = ensureSignatureHintEl();
  renderSignatureHint(el, hint);
  el.style.display = "block";
  const rect = el.getBoundingClientRect();
  let top = coords.top - rect.height - 6;
  if (top < 6) top = coords.bottom + 6;
  let left = coords.left;
  const maxLeft = window.innerWidth - rect.width - 8;
  if (left > maxLeft) left = Math.max(8, maxLeft);
  el.style.top = `${Math.round(top)}px`;
  el.style.left = `${Math.round(left)}px`;
}

function stripDisplayMath(latex) {
  return latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
}

async function buildExportDocument() {
  syncSourceFromBlocks();
  let outputs = [];
  const source = executableSource();

  if (source) {
    if (invoke) {
      const result = await invoke("run_tens", { source });
      if (!result.ok) {
        showError(`Export stopped: fix the TensorForge error first.\n\n${result.error}`);
        return null;
      }
      outputs = result.outputs ?? [];
      const firstError = outputs.find((item) => item.error);
      if (firstError) {
        const where = Number.isFinite(firstError.line) ? ` on line ${firstError.line}` : "";
        showError(`Export stopped: fix the TensorForge error${where} first.\n\n${firstError.error}`);
        return null;
      }
      renderOutputs(outputs, { preserveScroll: true });
      lastGoodShown = true;
    } else {
      outputs = lastRenderedOutputs;
    }
  }

  const blocks = groupOutputRows(orderedPreviewBlocks(outputs.filter((item) => !item.error)));
  return { blocks };
}

function buildExportMarkdown(doc) {
  return doc.blocks
    .map((block) => {
      if (block.kind === "markdown") return block.text.trim();
      if (block.kind === "output-row") {
        return block.items
          .map((item) => `### ${item.header}\n\n$$\n${stripDisplayMath(item.latex)}\n$$`)
          .join("\n\n");
      }
      return `### ${block.header}\n\n$$\n${stripDisplayMath(block.latex)}\n$$`;
    })
    .filter(Boolean)
    .join("\n\n");
}

function printableMathStyle(tex) {
  const compact = tex.replace(/\s+/g, "");
  let scale = 1;
  if (compact.length > 380) scale = 0.62;
  else if (compact.length > 280) scale = 0.72;
  else if (compact.length > 190) scale = 0.82;
  else if (/\\begin\{(?:array|aligned|matrix|pmatrix|bmatrix)\}/.test(tex) && compact.length > 130) {
    scale = 0.88;
  }
  return scale < 1 ? ` style="--tf-print-math-scale:${scale}"` : "";
}

function printableMathBlock(latex) {
  const tex = stripDisplayMath(latex);
  return `<div class="math-block"${printableMathStyle(tex)}>${renderMath(tex, true)}</div>`;
}

function buildPrintableBody(doc) {
  const blocks = doc.blocks;
  if (blocks.length === 0) return "";
  return blocks
    .map((block) => {
      if (block.kind === "markdown") {
        return `<section class="doc-block markdown-doc">${renderMarkdown(block.text)}</section>`;
      }
      if (block.kind === "output-row") {
        return `<section class="doc-block output-row">${block.items
          .map(
            (item) =>
              `<div class="print-output-item"><div class="head">${escapeHtml(item.header)}</div>${printableMathBlock(item.latex)}</div>`,
          )
          .join("")}</section>`;
      }
      return `<section class="doc-block"><div class="head">${escapeHtml(block.header)}</div>${printableMathBlock(block.latex)}</section>`;
    })
    .join("\n");
}

async function exportMarkdown() {
  closeExportMenu();
  const doc = await buildExportDocument().catch((e) => {
    showError(`Export failed: ${e}`);
    return null;
  });
  if (!doc) return;
  const markdown = buildExportMarkdown(doc);
  if (!markdown.trim()) {
    showError("Nothing to export yet. Add Markdown or .show(...) output first.");
    return;
  }
  if (invoke) {
    await invoke("export_text", {
      content: markdown,
      defaultFilename: defaultExportName("md"),
      filterName: "Markdown",
      extensions: ["md", "markdown"],
    }).catch(showError);
    return;
  }
  downloadText(markdown, defaultExportName("md"), "text/markdown");
}

async function exportPdf() {
  closeExportMenu();
  const doc = await buildExportDocument().catch((e) => {
    showError(`Export failed: ${e}`);
    return null;
  });
  if (!doc) return;
  const body = buildPrintableBody(doc);
  if (!body) {
    showError("Nothing to export yet. Add Markdown or .show(...) output first.");
    return;
  }
  printDocument(body);
}

function defaultExportName(ext) {
  const base = currentPath ? currentPath.split("/").pop().replace(/\.[^.]+$/, "") : "tensorforge-export";
  return `${base}.${ext}`;
}

function defaultSaveName() {
  const base = currentPath ? currentPath.split("/").pop().replace(/\.[^.]+$/, "") : "untitled";
  return `${base || "untitled"}.tens`;
}

function downloadText(text, filename, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function printDocument(bodyHtml) {
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const previousTitle = document.title;
  printRoot.innerHTML = bodyHtml;
  document.title = defaultExportName("pdf").replace(/\.pdf$/i, "");
  document.body.classList.add("printing");
  let cleaned = false;
  const cleanup = () => {
    if (cleaned) return;
    cleaned = true;
    document.body.classList.remove("printing");
    printRoot.innerHTML = "";
    document.title = previousTitle;
    window.scrollTo(scrollX, scrollY);
    document.removeEventListener("pointerdown", cleanup);
    document.removeEventListener("keydown", cleanup);
  };
  try {
    if (invoke) invoke("print_window").catch(showError);
    else window.print();
  } catch (e) {
    showError(`Print failed: ${e}`);
  }
  window.addEventListener("afterprint", cleanup, { once: true });
  document.addEventListener("pointerdown", cleanup, { once: true });
  document.addEventListener("keydown", cleanup, { once: true });
}

// ---- markdown templates and paste ------------------------------------------

function currentIsMarkdown() {
  return editorView ? isMarkdownAtPos(editorView.state.selection.main.head) : false;
}

function updateInsertTableState() {
  insertTableBtn.disabled = !currentIsMarkdown();
}

function openTableMenu() {
  if (!currentIsMarkdown()) return;
  tableMenu.classList.add("show");
}

function closeTableMenu() {
  tableMenu.classList.remove("show");
}

function markdownTable(rows, cols, align) {
  const sep = { left: "---", center: ":---:", right: "---:" }[align] ?? "---";
  const cells = (fill) => `| ${Array.from({ length: cols }, () => fill).join(" | ")} |`;
  const lines = [cells("   "), cells(sep)];
  for (let r = 0; r < rows; r++) lines.push(cells("   "));
  return lines.join("\n");
}

const MARKDOWN_CURSOR = "__TF_CURSOR__";

function insertMarkdownText(text, { block = true } = {}) {
  if (!currentIsMarkdown()) return;
  insertAtSelection(text, { block });
  closeTableMenu();
}

function markdownTemplate(kind, selected) {
  const text = selected || MARKDOWN_CURSOR;
  switch (kind) {
    case "h2":
      return { text: `## ${text}`, block: true };
    case "math":
      return { text: `$$\n${text}\n$$`, block: true };
    case "code":
      return { text: `\`\`\`\n${text}\n\`\`\``, block: true };
    case "quote":
      return { text: `> ${text}`, block: true };
    case "list":
      return { text: `- ${text}`, block: true };
    case "todo":
      return { text: `- [ ] ${text}`, block: true };
    case "bold":
      return { text: selected ? `**${selected}**` : `**${MARKDOWN_CURSOR}**`, block: false };
    case "link":
      return { text: `[${selected || "text"}](${MARKDOWN_CURSOR})`, block: false };
    case "hr":
      return { text: "---", block: true };
    case "image":
      return { text: `![${selected || "alt"}](${MARKDOWN_CURSOR})`, block: false };
    case "logic-tree":
      return { text: logicTreeTemplate(selected || `${MARKDOWN_CURSOR}Main idea`), block: true };
    case "three-cases":
      return { text: threeCasesTemplate(selected || `${MARKDOWN_CURSOR}3 cases`), block: true };
    default:
      return null;
  }
}

function insertMarkdownTemplate(kind) {
  if (!currentIsMarkdown() || !editorView) return;
  const sel = editorView.state.selection.main;
  const selected = editorView.state.doc.sliceString(sel.from, sel.to);
  const template = markdownTemplate(kind, selected);
  if (!template) return;
  insertMarkdownText(template.text, { block: template.block });
}

function logicTreeTemplate(title) {
  return String.raw`$$
\begin{array}{ccccc}
&&{\color{green}\text{${title}}}&&\\[6pt]
&\swarrow&&\searrow&\\[6pt]
{\color{green}\text{Left branch}}&&&&{\color{green}\text{Right branch}}\\[6pt]
\downarrow&&&&\downarrow\\[6pt]
{\color{green}\text{Left result}}&&&&{\color{green}\text{Right result}}
\end{array}
$$`;
}

function threeCasesTemplate(title) {
  return String.raw`$$
\begin{array}{ccc}
&{\color{blue}\text{${title}}}&\\[6pt]
{\color{blue}\begin{array}{c}\text{case 1}\\\text{assumptions}\end{array}}
&
{\color{blue}\begin{array}{c}\text{case 2}\\\text{assumptions}\end{array}}
&
{\color{blue}\begin{array}{c}\text{case 3}\\\text{assumptions}\end{array}}
\\[24pt]
{\color{green}\begin{array}{c}\text{method 1}\\\text{result}\end{array}}
&
{\color{green}\begin{array}{c}\text{method 2}\\\text{result}\end{array}}
&
{\color{green}\begin{array}{c}\text{method 3}\\\text{result}\end{array}}
\end{array}
$$`;
}

function insertTableIntoMarkdown() {
  const rows = Math.max(1, Number(document.getElementById("table-rows").value) || 2);
  const cols = Math.max(1, Number(document.getElementById("table-cols").value) || 3);
  const align = document.getElementById("table-align").value;
  insertMarkdownText(markdownTable(rows, cols, align));
}

// ---- scroll linking --------------------------------------------------------

function withScrollSyncSource(source, fn) {
  scrollSyncSource = source;
  fn();
  requestAnimationFrame(() => {
    if (scrollSyncSource === source) scrollSyncSource = null;
  });
}

function suppressScrollSync(ms = 350) {
  scrollSyncSuppressedUntil = Math.max(scrollSyncSuppressedUntil, performance.now() + ms);
  cancelAnimationFrame(scrollSyncFrame);
}

function isScrollSyncSuppressed() {
  return performance.now() < scrollSyncSuppressedUntil;
}

function clampScrollTop(el, top) {
  const max = Math.max(0, el.scrollHeight - el.clientHeight);
  return Math.max(0, Math.min(max, top));
}

function clampUnit(x) {
  return Math.max(0, Math.min(1, Number.isFinite(x) ? x : 0));
}

function anchorY(el) {
  return el.scrollTop + el.clientHeight * SCROLL_SYNC_ANCHOR;
}

function sourceAnchor() {
  if (!editorView || docBlocks.length === 0) return null;
  const scroller = editorView.scrollDOM;
  const blockAtHeight = editorView.lineBlockAtHeight(anchorY(scroller));
  const lineNo = editorView.state.doc.lineAt(blockAtHeight.from).number;
  const block = blockForSourceLine(lineNo);
  if (!block) return null;
  const lineSpan = Math.max(1, block.sourceEndLine - block.sourceLine + 1);
  return {
    blockId: block.id,
    progress: clampUnit((lineNo - block.sourceLine) / lineSpan),
  };
}

function previewChildren() {
  return [...output.children].filter((el) => el.dataset?.sourceBlockId);
}

function previewGroupForBlock(blockId) {
  const children = previewChildren().filter((el) => el.dataset.sourceBlockId === String(blockId));
  if (children.length === 0) return null;
  const top = Math.min(...children.map((el) => el.offsetTop));
  const bottom = Math.max(...children.map((el) => el.offsetTop + el.offsetHeight));
  return { blockId: String(blockId), top, bottom, height: Math.max(1, bottom - top) };
}

function nearestPreviewGroup(blockId) {
  const block = blockForId(blockId);
  if (!block) return null;
  const groups = new Map();
  for (const child of previewChildren()) {
    const id = child.dataset.sourceBlockId;
    if (!groups.has(id)) groups.set(id, previewGroupForBlock(id));
  }
  let best = null;
  let bestDistance = Infinity;
  for (const group of groups.values()) {
    const candidate = blockForId(group.blockId);
    if (!candidate) continue;
    const distance = Math.abs(candidate.sourceLine - block.sourceLine);
    if (distance < bestDistance) {
      best = group;
      bestDistance = distance;
    }
  }
  return best;
}

function previewAnchor() {
  const children = previewChildren();
  if (children.length === 0) return null;
  const y = anchorY(output);
  let child = children.find((el) => el.offsetTop + el.offsetHeight >= y);
  child ??= children[children.length - 1];
  const group = previewGroupForBlock(child.dataset.sourceBlockId);
  if (!group) return null;
  return {
    blockId: group.blockId,
    progress: clampUnit((y - group.top) / group.height),
  };
}

function syncOutputToEditor() {
  if (isScrollSyncSuppressed()) return;
  if (scrollSyncSource === "output") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const anchor = sourceAnchor();
    if (!anchor) return;
    const group = previewGroupForBlock(anchor.blockId) ?? nearestPreviewGroup(anchor.blockId);
    if (!group) return;
    const target = group.top + anchor.progress * group.height - output.clientHeight * SCROLL_SYNC_ANCHOR;
    withScrollSyncSource("editor", () => {
      output.scrollTo({
        top: clampScrollTop(output, target),
        behavior: "auto",
      });
    });
  });
}

function syncEditorToOutput() {
  if (isScrollSyncSuppressed()) return;
  if (scrollSyncSource === "editor") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const anchor = previewAnchor();
    if (!anchor) return;
    const block = blockForId(anchor.blockId);
    if (!block || !editorView) return;
    const lineSpan = Math.max(1, block.sourceEndLine - block.sourceLine + 1);
    const lineNo = Math.max(
      1,
      Math.min(
        editorView.state.doc.lines,
        Math.round(block.sourceLine + anchor.progress * lineSpan),
      ),
    );
    const pos = editorView.state.doc.line(lineNo).from;
    const lineBlock = editorView.lineBlockAt(pos);
    const scroller = editorView.scrollDOM;
    const target = lineBlock.top - scroller.clientHeight * SCROLL_SYNC_ANCHOR;
    withScrollSyncSource("output", () => {
      scroller.scrollTo({
        top: clampScrollTop(scroller, target),
        behavior: "auto",
      });
    });
  });
}

function makeBlockJump(el, line) {
  if (line) el.dataset.line = String(line);
  el.title = `Click to jump to source`;
  el.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    const targetLine = Number(e.target.closest("[data-source-line]")?.dataset.sourceLine ?? line);
    jumpToSourceLine(targetLine, { focus: true, behavior: "auto", sync: false });
  });
}

// ---- menus/events ----------------------------------------------------------

function toggleExportMenu() {
  exportMenu.classList.toggle("show");
}

function closeExportMenu() {
  exportMenu.classList.remove("show");
}

runBtn.addEventListener("click", run);
openBtn.addEventListener("click", openFile);
newBtn.addEventListener("click", newFile);
saveBtn.addEventListener("click", saveFile);
addTensBtn.addEventListener("click", insertTensBlock);
themeBtn.addEventListener("click", toggleTheme);

exportBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleExportMenu();
});
exportMenu.addEventListener("click", (e) => e.stopPropagation());
exportMdBtn.addEventListener("click", exportMarkdown);
exportPdfBtn.addEventListener("click", exportPdf);

insertTableBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (insertTableBtn.disabled) return;
  if (tableMenu.classList.contains("show")) closeTableMenu();
  else openTableMenu();
});
insertTableBtn.addEventListener("click", (e) => e.stopPropagation());
tableMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  const button = e.target.closest("button[data-md-template]");
  if (!button) return;
  insertMarkdownTemplate(button.dataset.mdTemplate);
});
document.getElementById("table-insert").addEventListener("click", insertTableIntoMarkdown);

output.addEventListener("scroll", syncEditorToOutput);
window.addEventListener("resize", resizeAllTextareas);
window.addEventListener("beforeunload", (e) => {
  if (!isDirty) return;
  e.preventDefault();
  e.returnValue = "";
});

document.addEventListener("click", () => {
  closeExportMenu();
  closeTableMenu();
  updateInsertTableState();
});
document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key === "Enter") {
    e.preventDefault();
    run();
  } else if (mod && e.key === "s") {
    e.preventDefault();
    saveFile();
  } else if (mod && e.key === "o") {
    e.preventDefault();
    openFile();
  } else if (mod && e.key === "n") {
    e.preventDefault();
    newFile();
  }
});

// ---- boot ------------------------------------------------------------------

async function loadBundledDefaultSource() {
  try {
    const response = await fetch(DEFAULT_SOURCE_URL, { cache: "no-store" });
    if (response.ok) return await response.text();
  } catch {
    // Fall through to the tiny fallback so the editor can still boot.
  }
  return FALLBACK_DEFAULT_SOURCE;
}

async function boot() {
  initEditor();
  localStorage.removeItem("tensorforge.source.v3");
  const bundledDefault = await loadBundledDefaultSource();
  const storedPath = localStorage.getItem("tensorforge.path.v1");
  const opened =
    invoke && storedPath ? await invoke("read_tens", { path: storedPath }).catch(() => null) : null;
  if (opened) {
    loadOpenedFile(opened);
  } else {
    if (storedPath) localStorage.removeItem("tensorforge.path.v1");
    setCurrentPath(null);
    setDocumentSource(bundledDefault, { run: false, dirty: false });
  }
  restoreFileRail();
  scheduleLiveRun();
}

boot();
