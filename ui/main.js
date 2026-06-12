// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const NOTE_OPEN = "<!-- tensorforge:note -->";
const NOTE_CLOSE = "<!-- /tensorforge:note -->";

const DEFAULT_SOURCE = `${NOTE_OPEN}
# Hill's class hyperelasticity

Hand-written spectral strain with a symbolic Hill-CR scale function.

$$
\\bm C = \\sum_{a=1}^{3} \\lambda_a^2 \\, \\bm N_a \\otimes \\bm N_a,
\\qquad
\\bm E = \\sum_{a=1}^{3} E(\\lambda_a) \\, \\bm N_a \\otimes \\bm N_a.
$$
${NOTE_CLOSE}

mu = Scalar("\\mu")
kappa = Scalar("\\kappa")
m = Scalar("m")
n = Scalar("n")

lambda = ScalarSet("\\lambda", dim=3)
N = VectorSet("\\bm N", dim=3)

lam = Var("\\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)

${NOTE_OPEN}
## Spectral strain

The source expression mirrors the math directly:

- \`&\` is the outer product.
- \`Q = 2 * diff(E, C)\` builds the fourth-order tangent from the two spectral sums.
${NOTE_CLOSE}

C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)

${NOTE_OPEN}
## Energy and stress

Hill's quadratic energy gives the thermodynamic force $\\bm T = \\partial W / \\partial \\bm E$.
${NOTE_CLOSE}

W = mu * ddot(E, E) + kappa/2 * tr(E)^2

T = diff(W, E)
S = T : Q

display(C, mode=symbol)
display(E, mode=symbol)
display(Q, mode=symbol)
display(W, mode=symbol)
display(T, mode=symbol)
display(S, mode=symbol)
`;

const KATEX_MACROS = { "\\bm": "\\boldsymbol{#1}" };
const NOTE_PLACEHOLDER = "Write Markdown here.";
let pendingNoteFocus = null;

const editor = document.getElementById("editor");
const gutter = document.getElementById("gutter");
const noteLayer = document.getElementById("note-layer");
const output = document.getElementById("output");
const runBtn = document.getElementById("run");
const openBtn = document.getElementById("open");
const saveBtn = document.getElementById("save");
const fileRail = document.getElementById("file-rail");
const railFolder = document.getElementById("rail-folder");
const railFiles = document.getElementById("rail-files");
const insertTableBtn = document.getElementById("insert-table");
const tableMenu = document.getElementById("table-menu");
const printRoot = document.getElementById("print-root");
const addNoteBtn = document.getElementById("add-note");
const exportBtn = document.getElementById("export");
const exportMenu = document.getElementById("export-menu");
const exportMdBtn = document.getElementById("export-md");
const exportPdfBtn = document.getElementById("export-pdf");
const themeBtn = document.getElementById("theme");
const filenameEl = document.getElementById("filename");

// ---- theme ----------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("tensorforge.theme", theme);
}

applyTheme(
  localStorage.getItem("tensorforge.theme") ??
    (window.matchMedia?.("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark"),
);

function toggleTheme() {
  const cur = document.documentElement.dataset.theme;
  applyTheme(cur === "dark" ? "light" : "dark");
}

// ---- about dialog -----------------------------------------------------------

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

// External links: the Tauri webview blocks target=_blank popups, so route
// them through the opener plugin (falls back to window.open in a browser).
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href^='http'], a[href^='mailto:']");
  if (!a) return;
  e.preventDefault();
  const openUrl = window.__TAURI__?.opener?.openUrl;
  if (openUrl) openUrl(a.href);
  else window.open(a.href, "_blank");
});

// Path of the currently open file; null = unsaved buffer.
let currentPath = null;
let lastRenderedOutputs = [];

function setCurrentPath(path) {
  currentPath = path;
  filenameEl.textContent = path ? path.split("/").pop() : "";
  filenameEl.title = path ?? "";
}

// v3 key: the bundled template changed to the hand-written spectral Hill
// derivation; older cached sources are intentionally not migrated.
editor.value = localStorage.getItem("tensorforge.source.v3") ?? DEFAULT_SOURCE;
setupCompletion(editor);

// ---- editor gutter -----------------------------------------------------------

let gutterLineCount = 0;

function syncGutter(errorLines = new Set(), options = {}) {
  const syncNotes = options.syncNotes ?? true;
  const count = editor.value.split("\n").length;
  const needsRebuild = count !== gutterLineCount;
  if (needsRebuild) {
    gutter.replaceChildren(
      ...Array.from({ length: count }, (_, i) => {
        const line = i + 1;
        const div = document.createElement("div");
        div.className = "ln";
        div.dataset.line = String(line);
        div.textContent = String(line);
        return div;
      }),
    );
    gutterLineCount = count;
  }
  for (const line of gutter.children) {
    line.classList.toggle("err", errorLines.has(Number(line.dataset.line)));
    line.classList.remove("note-fence");
  }
  for (const block of noteBlocksFromSource(editor.value)) {
    gutter.children[block.line - 1]?.classList.add("note-fence");
    if (block.closed) gutter.children[block.endLine - 1]?.classList.add("note-fence");
  }
  gutter.scrollTop = editor.scrollTop;
  if (syncNotes && !isEditingNote()) syncNoteBoxes();
}

function clearGutterErrors() {
  syncGutter();
}

function markGutterErrors(outputs, options = {}) {
  const lines = new Set(
    outputs
      .filter((item) => item.error && item.line)
      .map((item) => Number(item.line)),
  );
  syncGutter(lines, options);
}

syncGutter();

async function openFile() {
  if (!invoke) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return; // cancelled
  loadOpenedFile(opened);
}

function loadOpenedFile(opened) {
  editor.value = opened.source;
  setCurrentPath(opened.path);
  localStorage.setItem("tensorforge.source.v3", editor.value);
  renderFileRail(opened.folder, opened.files);
  clearGutterErrors();
  scheduleLiveRun();
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

// Restore the last-used folder listing on startup.
async function restoreFileRail() {
  const folder = localStorage.getItem("tensorforge.folder");
  if (!invoke || !folder) return;
  const files = await invoke("list_folder", { path: folder }).catch(() => null);
  if (files) renderFileRail(folder, files);
}
restoreFileRail();

// ---- rail resizer ----
const railResizer = document.getElementById("rail-resizer");
const storedRailWidth = Number(localStorage.getItem("tensorforge.railWidth"));
if (storedRailWidth >= 90) fileRail.style.flexBasis = `${storedRailWidth}px`;
railResizer.addEventListener("pointerdown", (e) => {
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

async function switchToFile(path) {
  if (!invoke || path === currentPath) return;
  // Notebook-style: persist the current file before switching away.
  if (currentPath) {
    await invoke("save_tens", {
      source: editor.value,
      path: currentPath,
    }).catch(showError);
  }
  const opened = await invoke("read_tens", { path }).catch(showError);
  if (opened) loadOpenedFile(opened);
}

async function saveFile() {
  if (!invoke) return;
  localStorage.setItem("tensorforge.source.v3", editor.value);
  // No path yet (scratch buffer): Save behaves as Save As.
  const path = await invoke("save_tens", {
    source: editor.value,
    path: currentPath,
  }).catch(showError);
  if (path) {
    setCurrentPath(path);
    await refreshFolderListing(path);
  }
}

// Re-list the folder of `path` so the rail picks up new files.
async function refreshFolderListing(path) {
  if (!invoke || !path) return;
  const opened = await invoke("read_tens", { path }).catch(() => null);
  if (opened) renderFileRail(opened.folder, opened.files);
}

async function exportMarkdown() {
  closeExportMenu();
  const markdown = buildExportMarkdown();
  if (!markdown.trim()) {
    showError("Nothing to export yet. Add notes or display(...) output first.");
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
    showError("Nothing to export yet. Add notes or display(...) output first.");
    return;
  }
  printDocument(body);
}

function defaultExportName(ext) {
  const base = currentPath
    ? currentPath.split("/").pop().replace(/\.[^.]+$/, "")
    : "tensorforge-export";
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

function toggleExportMenu() {
  exportMenu.classList.toggle("show");
}

function closeExportMenu() {
  exportMenu.classList.remove("show");
}

function showError(message) {
  output.innerHTML = "";
  clearGutterErrors();
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = String(message);
  output.appendChild(div);
  // The output pane now holds only this error; a later live run must not
  // treat it as good results and prepend a second parse banner on top.
  lastGoodShown = false;
}

function copyButton(label, text) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.textContent = label;
  btn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(text);
    btn.textContent = "✓ copied";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = label;
      btn.classList.remove("copied");
    }, 1200);
  });
  return btn;
}

function markdownBlocksFromSource(source) {
  return noteBlocksFromSource(source)
    .map((block) => ({
      kind: "markdown",
      line: block.line,
      markdown: block.lines.join("\n").trim(),
    }))
    .filter((block) => block.markdown);
}

function noteBlocksFromSource(source) {
  const blocks = [];
  const lines = source.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const kind = noteOpenKind(lines[i]);
    if (!kind) continue;
    const block = { line: i + 1, endLine: i + 1, closed: false, kind, lines: [] };
    // Legacy ```notes blocks can contain fenced code only when the inner
    // fence carries an info string (```rust … ```). New note blocks use
    // HTML-comment sentinels, so ordinary Markdown fences never conflict.
    // Mirrors strip_note_blocks in src/parser/mod.rs — keep the two in sync.
    let inInnerFence = false;
    i++;
    while (i < lines.length) {
      if (kind === "backtick") {
        if (inInnerFence) {
          if (isLegacyNoteClose(lines[i])) inInnerFence = false;
        } else if (isInnerFenceOpen(lines[i])) {
          inInnerFence = true;
        } else if (isNoteClose(lines[i], kind)) {
          break;
        }
      } else if (isNoteClose(lines[i], kind)) {
        break;
      }
      block.lines.push(lines[i]);
      i++;
    }
    block.closed = i < lines.length;
    if (!block.closed) continue;
    block.endLine = i + 1;
    blocks.push(block);
  }
  return blocks;
}

function noteOpenKind(line) {
  const trimmed = line.trim();
  if (trimmed.toLowerCase() === NOTE_OPEN.toLowerCase()) return "html";
  if (/^```notes\s*$/i.test(trimmed)) return "backtick";
  return null;
}

function isNoteClose(line, kind) {
  const trimmed = line.trim();
  return kind === "html"
    ? trimmed.toLowerCase() === NOTE_CLOSE.toLowerCase()
    : isLegacyNoteClose(trimmed);
}

function isLegacyNoteClose(line) {
  return /^```\s*$/.test(line.trim());
}

function isInnerFenceOpen(line) {
  const trimmed = line.trim();
  return trimmed.startsWith("```") && trimmed !== "```";
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (isInnerFenceOpen(line)) {
      const code = [];
      i++;
      while (i < lines.length && !isLegacyNoteClose(lines[i])) {
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
  const head = `<thead><tr>${header.map((c, k) => cell("th", c, k)).join("")}</tr></thead>`;
  const body = rows
    .map((r) => `<tr>${r.map((c, k) => cell("td", c, k)).join("")}</tr>`)
    .join("");
  return `<table class="md-table">${head}<tbody>${body}</tbody></table>`;
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
  // Pasted images: only data:image/ URLs are honored (no remote loads).
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
  } catch (e) {
    return `<code>${escapeHtml(tex)}</code>`;
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdownBlock(block) {
  const el = document.createElement("div");
  el.className = "block markdown-doc";
  el.innerHTML = renderMarkdown(block.markdown);
  makeBlockJump(el, block.line);
  return el;
}

function exportBlocks() {
  // Error outputs are working-state, not document content: exports carry
  // only notes and rendered results.
  return [
    ...markdownBlocksFromSource(editor.value),
    ...lastRenderedOutputs
      .filter((item) => !item.error)
      .map((item) => ({ kind: "output", ...item })),
  ].sort((a, b) => a.line - b.line || (a.kind === "markdown" ? -1 : 1));
}

function buildExportMarkdown() {
  return exportBlocks()
    .map((block) => {
      if (block.kind === "markdown") return block.markdown.trim();
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
        return `<section class="doc-block markdown-doc">${renderMarkdown(block.markdown)}</section>`;
      }
      return `<section class="doc-block"><div class="head">${escapeHtml(block.header)}</div><div class="math-block">${renderMath(stripDisplayMath(block.latex), true)}</div></section>`;
    })
    .join("\n");
}

function stripDisplayMath(latex) {
  return latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
}

// Print through the main window: WKWebView ignores print() on iframe
// content windows, but wry hooks window.print() on the top window. The
// printable DOM lives in #print-root and @media print hides the app shell.
async function printDocument(bodyHtml) {
  printRoot.innerHTML = bodyHtml;
  document.body.classList.add("printing");
  const cleanup = () => {
    document.body.classList.remove("printing");
    printRoot.innerHTML = "";
    window.scrollTo(0, 0);
  };
  try {
    if (invoke) {
      // window.print() is not wired up in WKWebView; the Rust side calls
      // the native print operation on the webview.
      await invoke("print_window");
    } else {
      window.print();
    }
  } catch (e) {
    showError(`Print failed: ${e}`);
  }
  // The invoke may resolve while the native dialog is still open, and an
  // early cleanup would blank the printed pages. #print-root is invisible
  // on screen anyway, so defer cleanup until the user is demonstrably back
  // in the app (or a real afterprint fires).
  window.addEventListener("afterprint", cleanup, { once: true });
  document.addEventListener("pointerdown", cleanup, { once: true });
  document.addEventListener("keydown", cleanup, { once: true });
}

function insertNoteBlock() {
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  const needsLeadingBreak = start > 0 && editor.value[start - 1] !== "\n";
  const needsTrailingBreak = end < editor.value.length && editor.value[end] !== "\n";
  const block = `${needsLeadingBreak ? "\n" : ""}${NOTE_OPEN}\n\n${NOTE_CLOSE}\n${needsTrailingBreak ? "\n" : ""}`;
  editor.setRangeText(block, start, end, "end");
  const openFenceOffset = start + (needsLeadingBreak ? 1 : 0);
  pendingNoteFocus = {
    startLine: lineForOffset(editor.value, openFenceOffset),
    selectionStart: 0,
    selectionEnd: 0,
  };
  const selectStart = openFenceOffset + `${NOTE_OPEN}\n`.length;
  editor.focus();
  editor.setSelectionRange(selectStart, selectStart);
  editor.dispatchEvent(new Event("input", { bubbles: true }));
}

// Pixel y of the top of `line`, taken from the matching gutter row so the
// overlays can never drift from the text by sub-pixel line-height rounding.
function rowTop(line) {
  const row = gutter.children[line - 1];
  if (row) return row.offsetTop - editor.scrollTop;
  const style = getComputedStyle(editor);
  return (
    Number.parseFloat(style.paddingTop) +
    (line - 1) * Number.parseFloat(style.lineHeight) -
    editor.scrollTop
  );
}

function rowSpanHeight(startLine, endLine, minLines = 1) {
  const lines = Math.max(minLines, endLine - startLine + 1);
  const lineHeight = Number.parseFloat(getComputedStyle(editor).lineHeight);
  return lines * lineHeight;
}

function syncNoteBoxes() {
  noteLayer.style.left = `${gutter.offsetWidth}px`;
  const boxes = noteBlocksFromSource(editor.value).map((block) => {
    const div = document.createElement("div");
    div.className = "source-note-box";
    div._block = block; // shared with the closures; repositionNoteBoxes refreshes it
    div.dataset.startLine = String(block.line);
    div.style.top = `${rowTop(block.line) - 2}px`;
    div.style.height = `${rowSpanHeight(block.line, block.endLine, 3) + 4}px`;
    const tag = document.createElement("span");
    tag.className = "region-tag";
    tag.textContent = "markdown";
    div.appendChild(tag);
    const noteEditor = document.createElement("textarea");
    noteEditor.className = "source-note-editor";
    noteEditor.spellcheck = false;
    noteEditor.wrap = "off";
    noteEditor.placeholder = NOTE_PLACEHOLDER;
    noteEditor.value = block.lines.join("\n");
    noteEditor.addEventListener("input", () => updateNoteBlock(block, noteEditor));
    noteEditor.addEventListener("keydown", (e) => handleNoteKeydown(e, block, noteEditor));
    noteEditor.addEventListener("paste", (e) => handleNotePaste(e, noteEditor));
    div.appendChild(noteEditor);
    return div;
  });
  noteLayer.replaceChildren(...boxes);
  syncCodeBoxes();
  restorePendingNoteFocus();
}

/// Contiguous non-note, non-blank line spans: the `.tens` code regions.
function codeRegionsFromSource(source) {
  const lines = source.split(/\r?\n/);
  const inNote = new Array(lines.length).fill(false);
  for (const block of noteBlocksFromSource(source)) {
    for (let l = block.line; l <= block.endLine; l++) inNote[l - 1] = true;
  }
  // One frame per contiguous code paragraph: blank lines split regions so
  // the frames hug the statements instead of wrapping the whole file.
  const regions = [];
  let start = null;
  for (let i = 0; i <= lines.length; i++) {
    const isCode = i < lines.length && !inNote[i] && lines[i].trim() !== "";
    if (isCode) {
      if (start === null) start = i + 1;
      continue;
    }
    if (start !== null) {
      regions.push({ line: start, endLine: i });
      start = null;
    }
  }
  return regions;
}

function syncCodeBoxes() {
  const codeLayer = document.getElementById("code-layer");
  codeLayer.style.left = `${gutter.offsetWidth}px`;
  codeLayer.replaceChildren(
    ...codeRegionsFromSource(editor.value).map((region) => {
      const div = document.createElement("div");
      div.className = "source-code-box";
      div.style.top = `${rowTop(region.line) - 2}px`;
      div.style.height = `${rowSpanHeight(region.line, region.endLine) + 4}px`;
      const tag = document.createElement("span");
      tag.className = "region-tag";
      tag.textContent = "tens";
      div.appendChild(tag);
      return div;
    }),
  );
}

function updateNoteBlock(block, noteEditor) {
  const current = currentNoteBlock(block) ?? block;
  replaceNoteBlock(current, noteEditor.value);
  localStorage.setItem("tensorforge.source.v3", editor.value);
  syncGutter(new Set(), { syncNotes: false });
  // Reposition every box (without rebuilding, to keep focus): edits in this
  // note shift the line spans of all notes below it.
  repositionNoteBoxes();
  scheduleLiveRun();
}

function handleNoteKeydown(e, block, noteEditor) {
  const empty = noteEditor.value.length === 0;
  const collapsed = noteEditor.selectionStart === noteEditor.selectionEnd;
  if (empty && collapsed && (e.key === "Backspace" || e.key === "Delete")) {
    e.preventDefault();
    removeNoteBlock(block);
  }
}

function replaceNoteBlock(block, markdown) {
  // setRangeText (not a value assignment) keeps the main editor's undo
  // history intact and leaves the rest of the file byte-identical.
  const [start, end] = lineRangeOffsets(editor.value, block.line, block.endLine);
  editor.setRangeText(`${NOTE_OPEN}\n${markdown}\n${NOTE_CLOSE}`, start, end, "preserve");
}

function removeNoteBlock(block) {
  // Re-resolve the block: the closure's line span goes stale while the
  // note is being edited, and a stale span would delete following code.
  const current = currentNoteBlock(block) ?? block;
  const start = offsetForLine(editor.value, current.line);
  const end = offsetForLine(editor.value, current.endLine + 1); // incl. newline
  editor.setRangeText("", start, end, "start");
  editor.focus();
  editor.setSelectionRange(start, start);
  localStorage.setItem("tensorforge.source.v3", editor.value);
  clearGutterErrors();
  scheduleLiveRun();
}

// Character offset of the start of `line` (1-based).
function offsetForLine(source, line) {
  let offset = 0;
  for (let l = 1; l < line; l++) {
    const nl = source.indexOf("\n", offset);
    if (nl === -1) return source.length;
    offset = nl + 1;
  }
  return offset;
}

// [start, end) covering lines startLine..=endLine, excluding the trailing
// newline of endLine.
function lineRangeOffsets(source, startLine, endLine) {
  const start = offsetForLine(source, startLine);
  let end = offsetForLine(source, endLine + 1);
  if (end > start && source[end - 1] === "\n") end -= 1;
  return [start, end];
}

function lineForOffset(source, offset) {
  let line = 1;
  for (let i = 0; i < Math.min(offset, source.length); i++) {
    if (source[i] === "\n") line++;
  }
  return line;
}

function currentNoteBlock(block) {
  return noteBlocksFromSource(editor.value).find((candidate) => candidate.line === block.line);
}

/// Refresh position, size and the closure-shared block objects of all note
/// boxes from the current source, without rebuilding the DOM (rebuilding
/// would destroy the focused note editor). Falls back to doing nothing when
/// the block count changed (a fence was typed); the next full sync rebuilds.
function repositionNoteBoxes() {
  const fresh = noteBlocksFromSource(editor.value);
  const boxes = [...noteLayer.children];
  if (fresh.length !== boxes.length) return;
  boxes.forEach((box, i) => {
    const block = fresh[i];
    if (box._block) Object.assign(box._block, block);
    box.dataset.startLine = String(block.line);
    box.style.top = `${rowTop(block.line) - 2}px`;
    box.style.height = `${rowSpanHeight(block.line, block.endLine, 3) + 4}px`;
  });
  syncCodeBoxes();
}

function isEditingNote() {
  return document.activeElement?.classList?.contains("source-note-editor");
}

function restorePendingNoteFocus() {
  if (!pendingNoteFocus) return;
  const focus = pendingNoteFocus;
  pendingNoteFocus = null;
  const box = noteLayer.querySelector(
    `.source-note-box[data-start-line="${focus.startLine}"] .source-note-editor`,
  );
  if (!box) return;
  const start = Math.min(focus.selectionStart, box.value.length);
  const end = Math.min(focus.selectionEnd, box.value.length);
  box.focus();
  box.setSelectionRange(start, end);
}

// Pasted images are embedded as data-URL markdown so the .tens file stays
// self-contained.
function handleNotePaste(e, noteEditor) {
  const item = [...(e.clipboardData?.items ?? [])].find((it) =>
    it.type.startsWith("image/"),
  );
  if (!item) return;
  e.preventDefault();
  const file = item.getAsFile();
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const md = `![pasted image](${reader.result})`;
    noteEditor.setRangeText(md, noteEditor.selectionStart, noteEditor.selectionEnd, "end");
    noteEditor.dispatchEvent(new Event("input", { bubbles: true }));
  };
  reader.readAsDataURL(file);
}

// ---- insert-table popover --------------------------------------------------
// Enabled only while a note editor has focus. Opening the popover moves
// focus into its inputs, so the target note and caret are captured (by
// start line, which is stable while no edits happen) at open time.
let tableTarget = null;

function updateInsertTableState() {
  insertTableBtn.disabled = !isEditingNote();
}

function openTableMenu() {
  const noteEditor = document.activeElement;
  const box = noteEditor.closest(".source-note-box");
  if (!box) return;
  tableTarget = {
    startLine: Number(box.dataset.startLine),
    selectionStart: noteEditor.selectionStart,
    selectionEnd: noteEditor.selectionEnd,
  };
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

function insertTableIntoNote() {
  if (!tableTarget) return;
  const rows = Math.max(1, Number(document.getElementById("table-rows").value) || 2);
  const cols = Math.max(1, Number(document.getElementById("table-cols").value) || 3);
  const align = document.getElementById("table-align").value;
  const block = noteBlocksFromSource(editor.value).find(
    (b) => b.line === tableTarget.startLine,
  );
  if (!block) {
    closeTableMenu();
    return;
  }
  const body = block.lines.join("\n");
  const s = Math.min(tableTarget.selectionStart, body.length);
  const e = Math.min(tableTarget.selectionEnd, body.length);
  const atLineStart = s === 0 || body[s - 1] === "\n";
  const table = `${atLineStart ? "" : "\n"}${markdownTable(rows, cols, align)}\n`;
  replaceNoteBlock(block, body.slice(0, s) + table + body.slice(e));
  localStorage.setItem("tensorforge.source.v3", editor.value);
  pendingNoteFocus = {
    startLine: tableTarget.startLine,
    selectionStart: s + table.length,
    selectionEnd: s + table.length,
  };
  tableTarget = null;
  closeTableMenu();
  clearGutterErrors(); // full resync rebuilds boxes and restores focus
  scheduleLiveRun();
}

function jumpToSourceLine(line) {
  if (!line) return;
  const target = Math.max(0, rowTop(line) + editor.scrollTop - editor.clientHeight / 3);
  editor.scrollTo({ top: target, behavior: "smooth" });
  const row = gutter.children[line - 1];
  if (row) {
    row.classList.add("jump");
    setTimeout(() => row.classList.remove("jump"), 1200);
  }
}

function makeBlockJump(el, line) {
  el.title = `Click to jump to line ${line}`;
  el.addEventListener("click", (e) => {
    if (e.target.closest("button")) return; // copy buttons keep their job
    jumpToSourceLine(line);
  });
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

  head.textContent = `[${header}]`;
  const bar = document.createElement("span");
  bar.className = "copy-bar";
  // export(..., format=markdown) wraps in $$...$$; strip for rendering.
  const tex = latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
  bar.append(
    copyButton("copy latex", tex),
    copyButton("copy markdown", `$$\n${tex}\n$$`),
  );
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
    math.innerHTML = `<div class="error">KaTeX: ${e.message}</div><pre>${tex}</pre>`;
  }
  block.append(head, math);
  return block;
}

function renderOutputs(outputs) {
  output.innerHTML = "";
  lastRenderedOutputs = outputs;
  markGutterErrors(outputs, { syncNotes: !isEditingNote() });
  const blocks = [
    ...markdownBlocksFromSource(editor.value),
    ...outputs.map((item) => ({ kind: "output", ...item })),
  ].sort((a, b) => a.line - b.line || (a.kind === "markdown" ? -1 : 1));

  if (blocks.length === 0) {
    output.innerHTML =
      '<div class="placeholder">No output yet — add display(...) or export(...) statements.</div>';
    return;
  }
  for (const block of blocks) {
    output.appendChild(
      block.kind === "markdown" ? renderMarkdownBlock(block) : renderOutputBlock(block),
    );
  }
}

async function run() {
  localStorage.setItem("tensorforge.source.v3", editor.value);
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

// ---- live rendering ---------------------------------------------------------
// Re-run automatically as the user types (debounced). The engine recovers
// per statement, so a broken line shows one error block while everything
// else keeps rendering. Parse errors (whole-file) keep the previous output
// to avoid flickering away good results mid-edit.

let liveTimer = null;
let lastGoodShown = false;

async function liveRun() {
  localStorage.setItem("tensorforge.source.v3", editor.value);
  if (!invoke) {
    renderOutputs([]);
    return;
  }
  const result = await invoke("run_tens", { source: editor.value });
  if (!result.ok) {
    // Whole-file parse error: show it only if we have nothing better.
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

editor.addEventListener("input", () => {
  clearGutterErrors();
  scheduleLiveRun();
});
editor.addEventListener("scroll", () => {
  gutter.scrollTop = editor.scrollTop;
  // A full rebuild would destroy the focused note editor mid-edit.
  if (isEditingNote()) repositionNoteBoxes();
  else syncNoteBoxes();
});
// Leaving the note layer (blur to the main editor, a button, …) is the
// moment to rebuild boxes so stale spans never survive an editing session.
noteLayer.addEventListener("focusin", updateInsertTableState);
noteLayer.addEventListener("focusout", (e) => {
  // Focus moving into the table popover must not disable the button or
  // rebuild the boxes mid-interaction.
  if (tableMenu.contains(e.relatedTarget)) return;
  updateInsertTableState();
  if (!noteLayer.contains(e.relatedTarget)) syncNoteBoxes();
});
// initial render on startup
scheduleLiveRun();

runBtn.addEventListener("click", run);
openBtn.addEventListener("click", openFile);
saveBtn.addEventListener("click", () => saveFile());

addNoteBtn.addEventListener("click", insertNoteBlock);
exportBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleExportMenu();
});
exportMenu.addEventListener("click", (e) => e.stopPropagation());
// pointerdown + preventDefault keeps the note editor focused while the
// popover opens; without it the click would blur the note first.
insertTableBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (insertTableBtn.disabled) return;
  if (tableMenu.classList.contains("show")) closeTableMenu();
  else openTableMenu();
});
insertTableBtn.addEventListener("click", (e) => e.stopPropagation());
tableMenu.addEventListener("click", (e) => e.stopPropagation());
document.getElementById("table-insert").addEventListener("click", insertTableIntoNote);
exportMdBtn.addEventListener("click", exportMarkdown);
exportPdfBtn.addEventListener("click", exportPdf);
themeBtn.addEventListener("click", toggleTheme);
document.addEventListener("click", () => {
  closeExportMenu();
  closeTableMenu();
});
document.addEventListener("keydown", (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key === "Enter") {
    e.preventDefault();
    run();
  } else if (mod && e.key === "s") {
    e.preventDefault();
    saveFile(e.shiftKey);
  } else if (mod && e.key === "o") {
    e.preventDefault();
    openFile();
  }
});
