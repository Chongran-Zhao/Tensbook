// TensorForge frontend. The visible editor is a small document/block editor:
// Markdown is the default, TensorForge code lives in blue .tens blocks. The
// saved/runner source is still the plain .tens text format with hidden
// tensorforge:tens sentinels, so files stay backward compatible.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const TENS_OPEN = "<!-- tensorforge:tens -->";
const TENS_CLOSE = "<!-- /tensorforge:tens -->";
const NOTE_OPEN = "<!-- tensorforge:note -->";
const NOTE_CLOSE = "<!-- /tensorforge:note -->";

const DEFAULT_SOURCE = `# Incompressible uniaxial tension

This document derives the axial first Piola-Kirchhoff stress component for a
Mooney-Rivlin material. Ordinary text is Markdown by default; blue .tens blocks
are executed.

$$
\\bm F = \\operatorname{diag}(\\lambda, \\lambda^{-1/2}, \\lambda^{-1/2})
$$

${TENS_OPEN}
lam = Var("\\lambda")
C1 = Scalar("C_1")
C2 = Scalar("C_2")

F = Tensor("\\bm F", order=2, dim=3)
F[1][1] = lam
F[2][2] = lam^(-1/2)
F[3][3] = lam^(-1/2)

C = F.T * F
${TENS_CLOSE}

For this deformation, the invariants reduce to

$$
I_1 = \\lambda^2 + 2\\lambda^{-1},
\\qquad
I_2 = 2\\lambda + \\lambda^{-2}.
$$

${TENS_OPEN}
I1 = Scalar("I_1")
I2 = Scalar("I_2")
I1 = lam^2 + 2 * lam^(-1)
I2 = 2 * lam + lam^(-2)

W = C1 * (I1 - 3) + C2 * (I2 - 3)

P11 = Scalar("P_{11}")
P11 = simplify(diff(W, lam))
${TENS_CLOSE}

The first component of the first Piola-Kirchhoff stress is \\(P_{11}=dW/d\\lambda\\).

${TENS_OPEN}
display(F, mode=matrix)
display(C, mode=matrix)
display(I1, mode=symbol)
display(I2, mode=symbol)
display(W, mode=symbol)
display(P11, mode=symbol)
export(P11, format=latex)
${TENS_CLOSE}
`;

const KATEX_MACROS = { "\\bm": "\\boldsymbol{#1}" };

const editor = document.getElementById("editor"); // hidden serialized source
const sourceEditor = document.getElementById("source-editor");
const sourceGutter = document.getElementById("source-gutter");
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
let lastRenderedOutputs = [];
let liveTimer = null;
let lastGoodShown = false;
let nextBlockId = 1;
let docBlocks = [];
let activeBlockId = null;
let activeTextarea = null;
let scrollSyncSource = null;
let scrollSyncFrame = null;

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

function setCurrentPath(path) {
  currentPath = path;
  filenameEl.textContent = path ? path.split("/").pop() : "";
  filenameEl.title = path ?? "";
}

// ---- document model --------------------------------------------------------

function makeBlock(kind, text, sourceLine = 1) {
  if (kind === "markdown") text = cleanMarkdownText(text);
  return { id: nextBlockId++, kind, text, sourceLine, sourceEndLine: sourceLine };
}

function trimOuterBlankLines(text) {
  return text.replace(/^\s*\n/, "").replace(/\n\s*$/, "");
}

function cleanMarkdownText(text) {
  let lines = text
    .split(/\r?\n/)
    .filter((line) => {
      const trimmed = line.trim().toLowerCase();
      return ![
        TENS_OPEN.toLowerCase(),
        TENS_CLOSE.toLowerCase(),
        NOTE_OPEN.toLowerCase(),
        NOTE_CLOSE.toLowerCase(),
      ].includes(trimmed);
    });

  // Older note UI sometimes stored an empty ```notes fenced wrapper. It was
  // an editor implementation detail, not user-authored Markdown.
  if (lines[0]?.trim().toLowerCase() === "```notes") {
    const close = lines.findIndex((line, i) => i > 0 && /^```\s*$/.test(line.trim()));
    if (close !== -1) lines = [...lines.slice(1, close), ...lines.slice(close + 1)];
  }
  return trimOuterBlankLines(lines.join("\n"));
}

function lineCount(text) {
  return text.length === 0 ? 1 : text.split(/\r?\n/).length;
}

function isTensOpen(line) {
  return line.trim().toLowerCase() === TENS_OPEN.toLowerCase();
}

function isTensClose(line) {
  return line.trim().toLowerCase() === TENS_CLOSE.toLowerCase();
}

function isNoteOpen(line) {
  return line.trim().toLowerCase() === NOTE_OPEN.toLowerCase();
}

function isNoteClose(line) {
  return line.trim().toLowerCase() === NOTE_CLOSE.toLowerCase();
}

function looksLikeTensorForgeSource(source) {
  return /^\s*(display|export|[A-Za-z_]\w*\s*=|[A-Za-z_]\w*\[[^\]]+\]\s*=)/m.test(source);
}

function makeFreeformBlock(text) {
  const cleaned = trimOuterBlankLines(text);
  if (!cleaned.trim()) return null;
  return makeBlock(looksLikeTensorForgeSource(cleaned) ? "tens" : "markdown", cleaned);
}

function parseLegacyNoteDocument(source) {
  const lines = source.split(/\r?\n/);
  const blocks = [];
  let start = 0;
  for (let i = 0; i < lines.length; i++) {
    if (!isNoteOpen(lines[i])) continue;
    const before = makeFreeformBlock(lines.slice(start, i).join("\n"));
    if (before) blocks.push(before);
    i++;
    const noteStart = i;
    while (i < lines.length && !isNoteClose(lines[i])) i++;
    const note = trimOuterBlankLines(lines.slice(noteStart, i).join("\n"));
    if (note.trim()) blocks.push(makeBlock("markdown", note));
    start = i + 1;
  }
  const tail = makeFreeformBlock(lines.slice(start).join("\n"));
  if (tail) blocks.push(tail);
  return blocks.length ? blocks : [makeBlock("markdown", "")];
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
    const md = trimOuterBlankLines(lines.slice(start, i).join("\n"));
    if (md.trim()) blocks.push(makeBlock("markdown", md));
    i++;
    const codeStart = i;
    while (i < lines.length && !isTensClose(lines[i])) i++;
    blocks.push(makeBlock("tens", lines.slice(codeStart, i).join("\n")));
    start = i + 1;
  }

  if (sawTens) {
    const tail = trimOuterBlankLines(lines.slice(start).join("\n"));
    if (tail.trim()) blocks.push(makeBlock("markdown", tail));
    return blocks.length ? blocks : [makeBlock("markdown", "")];
  }

  return [makeBlock(looksLikeTensorForgeSource(source) ? "tens" : "markdown", source)];
}

function serializeBlock(block) {
  if (block.kind === "tens") return `${TENS_OPEN}\n${block.text}\n${TENS_CLOSE}`;
  return block.text;
}

function serializeDocument(blocks) {
  return blocks.map(serializeBlock).join("\n\n");
}

function recomputeSourceLines() {
  let line = 1;
  for (const block of docBlocks) {
    block.sourceLine = line;
    const count = block.kind === "tens" ? lineCount(block.text) + 2 : lineCount(block.text);
    block.sourceEndLine = line + count - 1;
    line += count + 2; // blank separator inserted by serializeDocument
  }
}

function syncSourceFromBlocks() {
  recomputeSourceLines();
  editor.value = serializeDocument(docBlocks);
  localStorage.setItem("tensorforge.source.v3", editor.value);
}

function setDocumentSource(source, options = {}) {
  docBlocks = parseSourceDocument(source);
  syncSourceFromBlocks();
  renderSourceEditor();
  if (options.run !== false) scheduleLiveRun();
}

function blockForSourceLine(line) {
  return (
    docBlocks.find((b) => line >= b.sourceLine && line <= b.sourceEndLine) ??
    docBlocks[docBlocks.length - 1]
  );
}

// ---- source editor ---------------------------------------------------------

function renderSourceEditor() {
  sourceEditor.replaceChildren();
  for (const block of docBlocks) sourceEditor.appendChild(renderSourceBlock(block));
  renderSourceGutter();
  resizeAllTextareas();
  updateInsertTableState();
}

function renderSourceGutter() {
  sourceGutter.replaceChildren(
    ...docBlocks.map((block) => {
      const section = document.createElement("section");
      section.className = `gutter-block ${block.kind === "tens" ? "gutter-tens" : "gutter-markdown"}`;
      const firstLine = block.kind === "tens" ? block.sourceLine + 1 : block.sourceLine;
      const lines = lineCount(block.text);
      for (let i = 0; i < lines; i++) {
        const row = document.createElement("div");
        row.className = "gutter-line";
        row.textContent = String(firstLine + i);
        section.appendChild(row);
      }
      return section;
    }),
  );
}

const TENS_BUILTINS = new Set([
  "Scalar",
  "Tensor",
  "Var",
  "ScalarSet",
  "VectorSet",
  "Spec_Decomp",
  "eigvals",
  "eigvecs",
  "display",
  "export",
  "diff",
  "det",
  "tr",
  "log",
  "sqrt",
  "exp",
  "sinh",
  "cosh",
  "sum",
  "inv",
  "dot",
  "ddot",
  "outer",
  "otimes",
  "simplify",
]);

const TENS_KEYWORDS = new Set([
  "true",
  "false",
  "order",
  "dim",
  "mode",
  "format",
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
  "latex",
  "markdown",
  "algebra",
  "tensor",
  "continuum",
]);

function span(className, text) {
  return `<span class="${className}">${escapeHtml(text)}</span>`;
}

function highlightSource(kind, text) {
  const html = kind === "tens" ? highlightTens(text) : highlightMarkdownSource(text);
  return html || " ";
}

function updateHighlight(textarea) {
  const blockEl = textarea.closest(".source-block");
  const highlight = blockEl?.querySelector(".source-highlight");
  if (!highlight) return;
  highlight.innerHTML = highlightSource(
    blockEl.classList.contains("source-tens") ? "tens" : "markdown",
    textarea.value,
  );
}

function highlightMarkdownSource(text) {
  const lines = text.split("\n");
  let inFence = false;
  return lines
    .map((line) => {
      const fence = line.match(/^(\s*```)(.*)$/);
      if (fence) {
        inFence = !inFence;
        return span("hl-md-fence", fence[1]) + span("hl-md-code", fence[2]);
      }
      if (inFence) return span("hl-md-code", line);

      const heading = line.match(/^(#{1,6})(\s+.*)$/);
      if (heading) {
        return span("hl-md-marker", heading[1]) + span("hl-md-heading", heading[2]);
      }

      const list = line.match(/^(\s*[-*+]\s+)(.*)$/);
      if (list) {
        return span("hl-md-marker", list[1]) + highlightMarkdownInline(list[2]);
      }

      const quote = line.match(/^(\s*>\s+)(.*)$/);
      if (quote) {
        return span("hl-md-marker", quote[1]) + highlightMarkdownInline(quote[2]);
      }

      if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
        return highlightMarkdownTableLine(line);
      }

      return highlightMarkdownInline(line);
    })
    .join("\n");
}

function highlightMarkdownTableLine(line) {
  let out = "";
  for (let i = 0; i < line.length; ) {
    if (line[i] === "|") {
      out += span("hl-md-marker", "|");
      i++;
      continue;
    }
    const end = line.indexOf("|", i);
    const j = end === -1 ? line.length : end;
    out += highlightMarkdownInline(line.slice(i, j));
    i = j;
  }
  return out;
}

function highlightMarkdownInline(text) {
  let out = "";
  for (let i = 0; i < text.length; ) {
    if (text.startsWith("**", i)) {
      const end = text.indexOf("**", i + 2);
      if (end !== -1) {
        out += span("hl-md-marker", "**");
        out += span("hl-md-strong", text.slice(i + 2, end));
        out += span("hl-md-marker", "**");
        i = end + 2;
        continue;
      }
    }
    if (text[i] === "`") {
      const end = text.indexOf("`", i + 1);
      if (end !== -1) {
        out += span("hl-md-marker", "`");
        out += span("hl-md-code", text.slice(i + 1, end));
        out += span("hl-md-marker", "`");
        i = end + 1;
        continue;
      }
    }
    if (text[i] === "$") {
      const end = findInlineMathEnd(text, i + 1);
      if (end !== -1) {
        out += span("hl-md-marker", "$");
        out += span("hl-md-math", text.slice(i + 1, end));
        out += span("hl-md-marker", "$");
        i = end + 1;
        continue;
      }
    }
    if (text[i] === "*" && text[i + 1] !== "*") {
      const end = text.indexOf("*", i + 1);
      if (end !== -1) {
        out += span("hl-md-marker", "*");
        out += span("hl-md-em", text.slice(i + 1, end));
        out += span("hl-md-marker", "*");
        i = end + 1;
        continue;
      }
    }
    out += escapeHtml(text[i]);
    i++;
  }
  return out;
}

function highlightTens(text) {
  let out = "";
  for (let i = 0; i < text.length; ) {
    const ch = text[i];
    if (ch === "#") {
      const end = text.indexOf("\n", i);
      const j = end === -1 ? text.length : end;
      out += span("hl-tens-comment", text.slice(i, j));
      i = j;
      continue;
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
      out += span("hl-tens-string", text.slice(i, j));
      i = j;
      continue;
    }
    if (/[0-9]/.test(ch)) {
      const m = text.slice(i).match(/^\d+(?:\.\d+)?/);
      out += span("hl-tens-number", m[0]);
      i += m[0].length;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const m = text.slice(i).match(/^[A-Za-z_]\w*/);
      const word = m[0];
      const cls = TENS_BUILTINS.has(word)
        ? "hl-tens-builtin"
        : TENS_KEYWORDS.has(word)
          ? "hl-tens-keyword"
          : "hl-tens-ident";
      out += span(cls, word);
      i += word.length;
      continue;
    }
    if ("+-*/^=.:,&[]()".includes(ch)) {
      out += span("hl-tens-op", ch);
      i++;
      continue;
    }
    out += escapeHtml(ch);
    i++;
  }
  return out;
}

function renderSourceBlock(block) {
  const section = document.createElement("section");
  section.className = `source-block ${block.kind === "tens" ? "source-tens" : "source-markdown"}`;
  section.dataset.blockId = String(block.id);
  section.dataset.sourceLine = String(block.sourceLine);

  const layer = document.createElement("div");
  layer.className = "source-editor-layer";

  const highlight = document.createElement("pre");
  highlight.className = `source-highlight ${block.kind === "tens" ? "highlight-tens" : "highlight-markdown"}`;
  highlight.setAttribute("aria-hidden", "true");
  highlight.innerHTML = highlightSource(block.kind, block.text);

  const textarea = document.createElement("textarea");
  textarea.className = "source-textarea";
  textarea.spellcheck = block.kind === "markdown";
  textarea.wrap = "soft";
  textarea.value = block.text;
  textarea.placeholder =
    block.kind === "tens" ? "TensorForge code" : "Write Markdown here";

  textarea.addEventListener("focus", () => {
    activeBlockId = block.id;
    activeTextarea = textarea;
    updateInsertTableState();
  });
  textarea.addEventListener("input", (e) => {
    if (block.kind === "tens" && splitTensBlockOnBlankGap(block, textarea)) return;
    block.text = textarea.value;
    if (
      block.kind === "markdown" &&
      e.inputType?.startsWith("delete") &&
      collapseEmptyMarkdownBetweenTens(block)
    ) {
      return;
    }
    autoResize(textarea);
    syncSourceFromBlocks();
    refreshBlockLineAttributes();
    scheduleLiveRun();
  });
  textarea.addEventListener("keydown", (e) => {
    if (moveAcrossBlocksWithArrow(e, block, textarea)) return;
    if (
      block.kind === "tens" &&
      textarea.value.trim() === "" &&
      textarea.selectionStart === textarea.selectionEnd &&
      (e.key === "Backspace" || e.key === "Delete")
    ) {
      if (removeEmptyTensBlock(block, e.key === "Backspace" ? -1 : 1)) e.preventDefault();
      return;
    }
    if (
      block.kind === "markdown" &&
      textarea.value.trim() === "" &&
      textarea.selectionStart === textarea.selectionEnd &&
      (e.key === "Backspace" || e.key === "Delete")
    ) {
      if (collapseEmptyMarkdownBetweenTens(block)) e.preventDefault();
    }
  });
  textarea.addEventListener("paste", (e) => {
    if (block.kind === "markdown") handleMarkdownPaste(e, textarea, block);
  });
  textarea.addEventListener("click", updateInsertTableState);
  textarea.addEventListener("keyup", updateInsertTableState);

  layer.append(highlight, textarea);
  section.appendChild(layer);
  if (block.kind === "tens") setupCompletion(textarea);
  return section;
}

function moveAcrossBlocksWithArrow(e, block, textarea) {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return false;
  if (textarea.selectionStart !== textarea.selectionEnd) return false;

  const line = lineForOffset(textarea.value, textarea.selectionStart);
  const lines = lineCount(textarea.value);
  const direction = e.key === "ArrowDown" ? 1 : -1;
  if ((direction > 0 && line < lines) || (direction < 0 && line > 1)) return false;

  const idx = docBlocks.findIndex((b) => b.id === block.id);
  const target = docBlocks[idx + direction];
  if (!target) return false;

  e.preventDefault();
  focusBlockAtColumn(target, caretColumn(textarea.value, textarea.selectionStart), direction);
  return true;
}

function focusBlockAtColumn(block, column, direction) {
  const textarea = sourceEditor.querySelector(`[data-block-id="${block.id}"] textarea`);
  if (!textarea) return;

  const targetLine = direction > 0 ? 1 : lineCount(textarea.value);
  const lineStart = offsetForLine(textarea.value, targetLine);
  const lineEnd = offsetForLine(textarea.value, targetLine + 1);
  const end = lineEnd > lineStart && textarea.value[lineEnd - 1] === "\n" ? lineEnd - 1 : lineEnd;
  const offset = Math.min(lineStart + column, end);
  textarea.focus();
  textarea.setSelectionRange(offset, offset);
}

function lineForOffset(source, offset) {
  return lineCount(source.slice(0, offset));
}

function caretColumn(source, offset) {
  const lineStart = source.lastIndexOf("\n", Math.max(0, offset - 1)) + 1;
  return offset - lineStart;
}

function refreshBlockLineAttributes() {
  for (const block of docBlocks) {
    const section = sourceEditor.querySelector(`[data-block-id="${block.id}"]`);
    if (section) section.dataset.sourceLine = String(block.sourceLine);
  }
  renderSourceGutter();
}

function autoResize(textarea) {
  updateHighlight(textarea);
  textarea.style.height = "auto";
  textarea.style.height = `${Math.max(textarea.scrollHeight, 22)}px`;
}

function resizeAllTextareas() {
  sourceEditor.querySelectorAll("textarea").forEach(autoResize);
}

function activeBlock() {
  return docBlocks.find((b) => b.id === activeBlockId) ?? docBlocks[docBlocks.length - 1];
}

function insertTensBlock() {
  const current = activeBlock();
  const idx = Math.max(0, docBlocks.findIndex((b) => b.id === current?.id));
  const block = makeBlock("tens", "");
  docBlocks.splice(idx + 1, 0, block);
  activeBlockId = block.id;
  syncSourceFromBlocks();
  renderSourceEditor();
  const textarea = sourceEditor.querySelector(`[data-block-id="${block.id}"] textarea`);
  textarea?.focus();
}

function splitTensBlockOnBlankGap(block, textarea) {
  const match = textarea.value.match(/\n[ \t]*\n[ \t]*\n/);
  if (!match) return false;

  const idx = docBlocks.findIndex((b) => b.id === block.id);
  if (idx === -1) return false;

  const before = textarea.value.slice(0, match.index).replace(/\s+$/, "");
  const after = textarea.value.slice(match.index + match[0].length).replace(/^\s+/, "");
  if (!before && !after) return false;

  const markdown = makeBlock("markdown", "");
  const replacement = [];
  if (before) replacement.push(makeBlock("tens", before));
  replacement.push(markdown);
  if (after) replacement.push(makeBlock("tens", after));

  docBlocks.splice(idx, 1, ...replacement);
  activeBlockId = markdown.id;
  syncSourceFromBlocks();
  renderSourceEditor();
  sourceEditor.querySelector(`[data-block-id="${markdown.id}"] textarea`)?.focus();
  scheduleLiveRun();
  return true;
}

function removeEmptyTensBlock(block, direction = -1) {
  if (block.kind !== "tens" || block.text.trim() !== "") return false;
  const idx = docBlocks.findIndex((b) => b.id === block.id);
  if (idx === -1) return false;

  const focusTarget = docBlocks[idx + direction] ?? docBlocks[idx - direction];
  docBlocks.splice(idx, 1);
  if (docBlocks.length === 0) {
    docBlocks.push(makeBlock("markdown", ""));
  }

  const target = focusTarget && docBlocks.includes(focusTarget) ? focusTarget : docBlocks[Math.min(idx, docBlocks.length - 1)];
  activeBlockId = target.id;
  syncSourceFromBlocks();
  renderSourceEditor();

  const textarea = sourceEditor.querySelector(`[data-block-id="${target.id}"] textarea`);
  if (textarea) {
    const offset = direction < 0 ? textarea.value.length : 0;
    textarea.focus();
    textarea.setSelectionRange(offset, offset);
  }
  scheduleLiveRun();
  return true;
}

function collapseEmptyMarkdownBetweenTens(block) {
  if (block.kind !== "markdown" || block.text.trim() !== "") return false;
  const idx = docBlocks.findIndex((b) => b.id === block.id);
  if (idx <= 0 || idx >= docBlocks.length - 1) return false;

  const prev = docBlocks[idx - 1];
  const next = docBlocks[idx + 1];
  if (prev.kind !== "tens" || next.kind !== "tens") return false;

  const before = prev.text.replace(/\s+$/, "");
  const after = next.text.replace(/^\s+/, "");
  const cursor = before.length + 2;
  prev.text = [before, after].filter(Boolean).join("\n\n");
  docBlocks.splice(idx, 2);
  activeBlockId = prev.id;

  syncSourceFromBlocks();
  renderSourceEditor();
  const textarea = sourceEditor.querySelector(`[data-block-id="${prev.id}"] textarea`);
  if (textarea) {
    textarea.focus();
    textarea.setSelectionRange(Math.min(cursor, textarea.value.length), Math.min(cursor, textarea.value.length));
  }
  scheduleLiveRun();
  return true;
}

function focusSourceLine(line) {
  const block = blockForSourceLine(line);
  const section = sourceEditor.querySelector(`[data-block-id="${block.id}"]`);
  const textarea = section?.querySelector("textarea");
  if (!textarea) return;
  const innerLine =
    block.kind === "tens"
      ? Math.max(1, line - block.sourceLine)
      : Math.max(1, line - block.sourceLine + 1);
  const offset = offsetForLine(textarea.value, innerLine);
  textarea.focus();
  textarea.setSelectionRange(offset, offset);
}

function offsetForLine(source, line) {
  let offset = 0;
  for (let l = 1; l < line; l++) {
    const nl = source.indexOf("\n", offset);
    if (nl === -1) return source.length;
    offset = nl + 1;
  }
  return offset;
}

function scrollEditorToLine(line, options = {}) {
  const block = blockForSourceLine(line);
  const section = sourceEditor.querySelector(`[data-block-id="${block.id}"]`);
  if (!section) return;
  const align = options.align ?? 0.18;
  const behavior = options.behavior ?? "smooth";
  const target = Math.max(0, section.offsetTop - sourceEditor.clientHeight * align);
  sourceEditor.scrollTo({ top: target, behavior });
}

function jumpToSourceLine(line, options = {}) {
  scrollEditorToLine(line, { behavior: options.behavior ?? "smooth" });
  if (options.focus) focusSourceLine(line);
}

// ---- files -----------------------------------------------------------------

async function openFile() {
  if (!invoke) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return;
  loadOpenedFile(opened);
}

function newFile() {
  setCurrentPath(null);
  setDocumentSource("", { run: false });
  lastRenderedOutputs = [];
  lastGoodShown = false;
  output.innerHTML = '<div class="placeholder">No output yet. Add display(...) or export(...).</div>';
  railFiles.querySelectorAll(".file.active").forEach((el) => el.classList.remove("active"));
  sourceEditor.scrollTop = 0;
  sourceGutter.scrollTop = 0;
  output.scrollTop = 0;
  const first = sourceEditor.querySelector("textarea");
  first?.focus();
}

function loadOpenedFile(opened) {
  setCurrentPath(opened.path);
  setDocumentSource(opened.source);
  renderFileRail(opened.folder, opened.files);
}

function renderFileRail(folder, files) {
  if (!folder || !files || files.length === 0) {
    railFolder.textContent = "";
    const empty = document.createElement("div");
    empty.className = "rail-empty";
    empty.textContent = "Open a .tens file to browse its folder here.";
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
  syncSourceFromBlocks();
  if (currentPath) {
    await invoke("save_tens", { source: editor.value, path: currentPath }).catch(showError);
  }
  const opened = await invoke("read_tens", { path }).catch(showError);
  if (opened) loadOpenedFile(opened);
}

async function saveFile() {
  if (!invoke) return;
  syncSourceFromBlocks();
  const path = await invoke("save_tens", {
    source: editor.value,
    path: currentPath,
  }).catch(showError);
  if (path) {
    setCurrentPath(path);
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

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (isInnerFenceOpen(line)) {
      const code = [];
      i++;
      while (i < lines.length && !isBareFenceClose(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      out.push(`<pre class="md-code"><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const oneLineMath = line.match(/^\s*\$\$\s*(.*?)\s*\$\$\s*$/);
    if (oneLineMath) {
      out.push(`<div class="markdown-math">${renderMath(oneLineMath[1], true)}</div>`);
      continue;
    }
    if (line.trim() === "$$") {
      const tex = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") {
        tex.push(lines[i]);
        i++;
      }
      out.push(`<div class="markdown-math">${renderMath(tex.join("\n"), true)}</div>`);
      continue;
    }
    if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const aligns = tableAlignments(lines[i + 1]);
      const header = tableCells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(tableCells(lines[i]));
        i++;
      }
      i--;
      out.push(renderTable(header, rows, aligns));
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(`<li>${inlineMarkdown(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      i--;
      out.push(`<ul>${items.join("")}</ul>`);
      continue;
    }
    if (line.trim() === "") continue;
    out.push(`<p>${inlineMarkdown(line)}</p>`);
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

function renderTable(header, rows, aligns) {
  const cell = (tag, content, k) =>
    `<${tag} style="text-align:${aligns[k] ?? "left"}">${inlineMarkdown(content.trim())}</${tag}>`;
  return `<table class="md-table"><thead><tr>${header
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
    parts.push(`<img class="md-img" alt="${escapeHtml(m[1])}" src="${m[2]}">`);
    rest = rest.slice(m.index + m[0].length);
  }
  parts.push(inlineStyles(rest));
  return parts.join("");
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
  el.innerHTML = renderMarkdown(block.text);
  makeBlockJump(el, block.sourceLine);
  return el;
}

function renderOutputBlock(item) {
  const { header, latex, line, error } = item;
  const block = document.createElement("div");
  block.className = "block";
  makeBlockJump(block, line);
  const head = document.createElement("div");
  head.className = "head";

  if (error) {
    head.textContent = `[line ${line}]`;
    const err = document.createElement("div");
    err.className = "error";
    err.textContent = error;
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

function renderOutputRowBlock(items) {
  const row = document.createElement("div");
  row.className = "output-row";
  for (const item of items) row.appendChild(renderOutputBlock(item));
  return row;
}

function countOutputCalls(source) {
  return source
    .split(/\r?\n/)
    .reduce(
      (count, line) =>
        count + (line.replace(/#.*/, "").match(/\b(?:display|export)\s*\(/g)?.length ?? 0),
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
    item.line >= block.sourceLine && item.line <= block.sourceEndLine;

  const put = (item, block) => {
    buckets.get(block.id)?.push(item);
    assigned.add(item.index);
  };

  for (const item of items) {
    if (!item.error) continue;
    const block = tensBlocks.find((candidate) => outputInBlock(item, candidate));
    if (block) put(item, block);
  }

  for (const item of items) {
    if (assigned.has(item.index)) continue;
    if (item.error) {
      const block = tensBlocks.find((candidate) => outputInBlock(item, candidate));
      if (block) {
        put(item, block);
        continue;
      }
    }

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

function renderOutputs(outputs) {
  output.innerHTML = "";
  lastRenderedOutputs = outputs;
  const blocks = groupOutputRows(orderedPreviewBlocks(outputs));

  if (blocks.length === 0) {
    output.innerHTML = '<div class="placeholder">No output yet. Add display(...) or export(...).</div>';
    return;
  }
  for (const block of blocks) {
    if (block.kind === "markdown") output.appendChild(renderMarkdownBlock(block));
    else if (block.kind === "output-row") output.appendChild(renderOutputRowBlock(block.items));
    else output.appendChild(renderOutputBlock(block));
  }
  syncOutputToEditor();
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
  const result = await invoke("run_tens", { source: editor.value });
  if (!result.ok) {
    showError(result.error);
    return;
  }
  renderOutputs(result.outputs);
}

async function liveRun() {
  syncSourceFromBlocks();
  if (!invoke) {
    renderOutputs([]);
    return;
  }
  const result = await invoke("run_tens", { source: editor.value });
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
  renderOutputs(result.outputs);
  lastGoodShown = true;
}

function scheduleLiveRun() {
  clearTimeout(liveTimer);
  liveTimer = setTimeout(liveRun, 350);
}

function stripDisplayMath(latex) {
  return latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
}

function exportBlocks() {
  return groupOutputRows(orderedPreviewBlocks(lastRenderedOutputs.filter((item) => !item.error)));
}

function buildExportMarkdown() {
  return exportBlocks()
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

function buildPrintableBody() {
  const blocks = exportBlocks();
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
              `<div><div class="head">${escapeHtml(item.header)}</div><div class="math-block">${renderMath(stripDisplayMath(item.latex), true)}</div></div>`,
          )
          .join("")}</section>`;
      }
      return `<section class="doc-block"><div class="head">${escapeHtml(block.header)}</div><div class="math-block">${renderMath(stripDisplayMath(block.latex), true)}</div></section>`;
    })
    .join("\n");
}

async function exportMarkdown() {
  closeExportMenu();
  const markdown = buildExportMarkdown();
  if (!markdown.trim()) {
    showError("Nothing to export yet. Add Markdown or display(...) output first.");
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

function exportPdf() {
  closeExportMenu();
  const body = buildPrintableBody();
  if (!body) {
    showError("Nothing to export yet. Add Markdown or display(...) output first.");
    return;
  }
  printDocument(body);
}

function defaultExportName(ext) {
  const base = currentPath ? currentPath.split("/").pop().replace(/\.[^.]+$/, "") : "tensorforge-export";
  return `${base}.${ext}`;
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
  printRoot.innerHTML = bodyHtml;
  document.body.classList.add("printing");
  const cleanup = () => {
    document.body.classList.remove("printing");
    printRoot.innerHTML = "";
    window.scrollTo(0, 0);
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

// ---- table and paste -------------------------------------------------------

function currentIsMarkdown() {
  return activeBlock()?.kind === "markdown";
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

function insertTableIntoMarkdown() {
  const block = activeBlock();
  const textarea = activeTextarea;
  if (!block || block.kind !== "markdown" || !textarea) return;
  const rows = Math.max(1, Number(document.getElementById("table-rows").value) || 2);
  const cols = Math.max(1, Number(document.getElementById("table-cols").value) || 3);
  const align = document.getElementById("table-align").value;
  const s = Math.min(textarea.selectionStart, textarea.value.length);
  const e = Math.min(textarea.selectionEnd, textarea.value.length);
  const atLineStart = s === 0 || textarea.value[s - 1] === "\n";
  textarea.setRangeText(`${atLineStart ? "" : "\n"}${markdownTable(rows, cols, align)}\n`, s, e, "end");
  block.text = textarea.value;
  autoResize(textarea);
  syncSourceFromBlocks();
  closeTableMenu();
  scheduleLiveRun();
}

function handleMarkdownPaste(e, textarea, block) {
  const item = [...(e.clipboardData?.items ?? [])].find((it) => it.type.startsWith("image/"));
  if (!item) return;
  e.preventDefault();
  const file = item.getAsFile();
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    textarea.setRangeText(`![pasted image](${reader.result})`, textarea.selectionStart, textarea.selectionEnd, "end");
    block.text = textarea.value;
    autoResize(textarea);
    syncSourceFromBlocks();
    scheduleLiveRun();
  };
  reader.readAsDataURL(file);
}

// ---- scroll linking --------------------------------------------------------

function withScrollSyncSource(source, fn) {
  scrollSyncSource = source;
  fn();
  requestAnimationFrame(() => {
    if (scrollSyncSource === source) scrollSyncSource = null;
  });
}

function scrollRatio(el) {
  const max = el.scrollHeight - el.clientHeight;
  return max <= 0 ? 0 : el.scrollTop / max;
}

function syncOutputToEditor() {
  if (scrollSyncSource === "output") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const targetMax = output.scrollHeight - output.clientHeight;
    if (targetMax <= 0) return;
    withScrollSyncSource("editor", () => {
      output.scrollTo({
        top: Math.max(0, Math.min(targetMax, scrollRatio(sourceEditor) * targetMax)),
        behavior: "auto",
      });
    });
  });
}

function syncEditorToOutput() {
  if (scrollSyncSource === "editor") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const targetMax = sourceEditor.scrollHeight - sourceEditor.clientHeight;
    if (targetMax <= 0) return;
    withScrollSyncSource("output", () => {
      sourceEditor.scrollTo({
        top: Math.max(0, Math.min(targetMax, scrollRatio(output) * targetMax)),
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
    jumpToSourceLine(line, { focus: true, behavior: "auto" });
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
tableMenu.addEventListener("click", (e) => e.stopPropagation());
document.getElementById("table-insert").addEventListener("click", insertTableIntoMarkdown);

sourceEditor.addEventListener("scroll", () => {
  sourceGutter.scrollTop = sourceEditor.scrollTop;
  syncOutputToEditor();
});
output.addEventListener("scroll", syncEditorToOutput);
window.addEventListener("resize", resizeAllTextareas);

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

setDocumentSource(localStorage.getItem("tensorforge.source.v3") ?? DEFAULT_SOURCE, { run: false });
restoreFileRail();
scheduleLiveRun();
