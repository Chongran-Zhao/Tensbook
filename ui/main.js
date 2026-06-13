// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const TENS_OPEN = "<!-- tensorforge:tens -->";
const TENS_CLOSE = "<!-- /tensorforge:tens -->";

const DEFAULT_SOURCE = `# Incompressible uniaxial tension

This document derives the axial first Piola-Kirchhoff stress component for a
Mooney-Rivlin material. Ordinary text is Markdown by default; only blue
.tens blocks are executed.

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

The first component of the first Piola-Kirchhoff stress is \(P_{11}=dW/d\\lambda\).

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
const TENS_PLACEHOLDER = "Write TensorForge code here.";
let pendingTensFocus = null;

const editor = document.getElementById("editor");
const editorWrap = document.getElementById("editor-wrap");
const gutter = document.getElementById("gutter");
const output = document.getElementById("output");
const splitResizer = document.getElementById("split-resizer");
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
  const syncBlocks = options.syncBlocks ?? true;
  const sourceLines = editor.value.split("\n");
  const count = sourceLines.length;
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
  // Measure wrapping so each gutter row matches its (possibly multi-row)
  // logical line; everything downstream reads sourceLineTop, so frames and
  // scroll sync align automatically.
  const tops = recomputeLineMetrics();
  const cs = getComputedStyle(editor);
  [...gutter.children].forEach((line, i) => {
    line.classList.toggle("err", errorLines.has(Number(line.dataset.line)));
    line.classList.remove("block-fence");
    line.style.height = `${tops[i + 1] - tops[i]}px`;
    line.style.lineHeight = cs.lineHeight; // keep the number on the first row
  });
  for (const block of tensBlocksFromSource(editor.value)) {
    gutter.children[block.line - 1]?.classList.add("block-fence");
    if (block.closed) gutter.children[block.endLine - 1]?.classList.add("block-fence");
  }
  gutter.scrollTop = editor.scrollTop;
  syncSourceLayerScroll();
  if (syncBlocks && !isEditingTens()) syncTensBoxes();
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

// ---- editor/output resizer ----
const storedEditorWidth = Number(localStorage.getItem("tensorforge.editorWidth"));
if (storedEditorWidth >= 280) editorWrap.style.flexBasis = `${storedEditorWidth}px`;

function refreshEditorGeometry() {
  syncGutter(new Set(), { syncBlocks: !isEditingTens() });
  if (isEditingTens()) repositionTensBoxes();
}

function editorWidthBounds() {
  const mainRect = document.querySelector("main").getBoundingClientRect();
  const left = editorWrap.getBoundingClientRect().left;
  const minEditor = 280;
  const minOutput = 360;
  const maxEditor = Math.max(minEditor, mainRect.right - left - minOutput);
  return { minEditor, maxEditor };
}

function setEditorWidthFromClientX(clientX) {
  const { minEditor, maxEditor } = editorWidthBounds();
  const left = editorWrap.getBoundingClientRect().left;
  const width = Math.min(maxEditor, Math.max(minEditor, clientX - left));
  editorWrap.style.flexBasis = `${width}px`;
  requestAnimationFrame(refreshEditorGeometry);
}

let splitDragActive = false;

splitResizer.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  splitDragActive = true;
  splitResizer.setPointerCapture(e.pointerId);
  splitResizer.classList.add("dragging");
  const move = (ev) => {
    setEditorWidthFromClientX(ev.clientX);
  };
  splitResizer.addEventListener("pointermove", move);
  splitResizer.addEventListener(
    "pointerup",
    () => {
      splitDragActive = false;
      splitResizer.removeEventListener("pointermove", move);
      splitResizer.classList.remove("dragging");
      localStorage.setItem("tensorforge.editorWidth", String(editorWrap.offsetWidth));
      refreshEditorGeometry();
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
    refreshEditorGeometry();
  };
  document.addEventListener("mousemove", move);
  document.addEventListener("mouseup", up, { once: true });
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
  const tens = tensBlocksFromSource(source);
  if (tens.length === 0) {
    const markdown = source.trim();
    return markdown ? [{ kind: "markdown", line: 1, markdown }] : [];
  }
  const lines = source.split(/\r?\n/);
  const blocks = [];
  let start = 1;
  for (const block of tens) {
    pushMarkdownBlock(blocks, lines, start, block.line - 1);
    start = block.endLine + 1;
  }
  pushMarkdownBlock(blocks, lines, start, lines.length);
  return blocks;
}

function pushMarkdownBlock(blocks, lines, startLine, endLine) {
  if (endLine < startLine) return;
  const markdown = lines.slice(startLine - 1, endLine).join("\n").trim();
  if (markdown) blocks.push({ kind: "markdown", line: startLine, markdown });
}

function tensBlocksFromSource(source) {
  const blocks = [];
  const lines = source.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    if (!isTensOpen(lines[i])) continue;
    const block = { line: i + 1, endLine: i + 1, closed: false, lines: [] };
    i++;
    while (i < lines.length) {
      if (isTensClose(lines[i])) break;
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

function isTensOpen(line) {
  return line.trim().toLowerCase() === TENS_OPEN.toLowerCase();
}

function isTensClose(line) {
  return line.trim().toLowerCase() === TENS_CLOSE.toLowerCase();
}

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
  // Error outputs are working-state, not document content: exports carry
  // only Markdown and rendered results.
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

function insertTensBlock() {
  const start = editor.selectionStart;
  const end = editor.selectionEnd;
  const needsLeadingBreak = start > 0 && editor.value[start - 1] !== "\n";
  const needsTrailingBreak = end < editor.value.length && editor.value[end] !== "\n";
  const block = `${needsLeadingBreak ? "\n" : ""}${TENS_OPEN}\n\n${TENS_CLOSE}\n${needsTrailingBreak ? "\n" : ""}`;
  editor.setRangeText(block, start, end, "end");
  const openFenceOffset = start + (needsLeadingBreak ? 1 : 0);
  pendingTensFocus = {
    startLine: lineForOffset(editor.value, openFenceOffset),
    selectionStart: 0,
    selectionEnd: 0,
  };
  const selectStart = openFenceOffset + `${TENS_OPEN}\n`.length;
  editor.focus();
  editor.setSelectionRange(selectStart, selectStart);
  editor.dispatchEvent(new Event("input", { bubbles: true }));
}

// Top offsets (relative to the editor content box, including padding-top)
// of every logical line, plus a sentinel = content bottom at index n.
// Recomputed by recomputeLineMetrics whenever content or width changes.
let lineTopsCache = null;

function recomputeLineMetrics() {
  const cs = getComputedStyle(editor);
  let mirror = document.getElementById("wrap-mirror");
  if (!mirror) {
    mirror = document.createElement("div");
    mirror.id = "wrap-mirror";
    editorWrap.appendChild(mirror);
  }
  // Match the editor's text box so the mirror wraps identically.
  mirror.style.font = cs.font;
  mirror.style.lineHeight = cs.lineHeight;
  mirror.style.letterSpacing = cs.letterSpacing;
  mirror.style.tabSize = cs.tabSize;
  const padTop = Number.parseFloat(cs.paddingTop);
  const padLeft = Number.parseFloat(cs.paddingLeft);
  const padRight = Number.parseFloat(cs.paddingRight);
  mirror.style.width = `${editor.clientWidth - padLeft - padRight}px`;

  const lines = editor.value.split("\n");
  mirror.replaceChildren(
    ...lines.map((text) => {
      const d = document.createElement("div");
      // A zero-width char keeps empty lines one visual row tall.
      d.textContent = text.length ? text : "​";
      return d;
    }),
  );
  const tops = new Array(lines.length + 1);
  for (let i = 0; i < lines.length; i++) {
    tops[i] = padTop + mirror.children[i].offsetTop;
  }
  tops[lines.length] = padTop + mirror.scrollHeight;
  lineTopsCache = tops;
  return tops;
}

function sourceLineTop(line) {
  const tops = lineTopsCache;
  if (tops && line >= 1 && line <= tops.length) return tops[line - 1];
  const style = getComputedStyle(editor);
  return (
    Number.parseFloat(style.paddingTop) +
    (line - 1) * Number.parseFloat(style.lineHeight)
  );
}

function rowTop(line) {
  return sourceLineTop(line) - editor.scrollTop;
}

// Pixel height covering startLine..endLine inclusive, accounting for wrapped
// lines via the measured offsets; never below `minLines` single rows.
function rowSpanHeight(startLine, endLine, minLines = 1) {
  const lineHeight = Number.parseFloat(getComputedStyle(editor).lineHeight);
  const span = sourceLineTop(endLine + 1) - sourceLineTop(startLine);
  return Math.max(span, minLines * lineHeight);
}

function syncSourceLayerScroll() {
  const y = `translateY(${-editor.scrollTop}px)`;
  document.getElementById("code-layer").style.transform = y;
}

function syncTensBoxes() {
  const codeLayer = document.getElementById("code-layer");
  codeLayer.style.left = `${gutter.offsetWidth}px`;
  const boxes = tensBlocksFromSource(editor.value).map((block) => {
    const div = document.createElement("div");
    div.className = "source-code-box";
    div._block = block; // shared with the closures; repositionTensBoxes refreshes it
    div.dataset.startLine = String(block.line);
    div.style.top = `${sourceLineTop(block.line) - 2}px`;
    div.style.height = `${rowSpanHeight(block.line, block.endLine, 3) + 4}px`;
    const tensEditor = document.createElement("textarea");
    tensEditor.className = "source-tens-editor";
    tensEditor.spellcheck = false;
    tensEditor.wrap = "soft";
    tensEditor.placeholder = TENS_PLACEHOLDER;
    tensEditor.value = block.lines.join("\n");
    tensEditor.addEventListener("focus", () => queueMicrotask(updateInsertTableState));
    tensEditor.addEventListener("input", () => updateTensBlock(block, tensEditor));
    tensEditor.addEventListener("keydown", (e) => handleTensKeydown(e, block, tensEditor));
    div.appendChild(tensEditor);
    return div;
  });
  codeLayer.replaceChildren(...boxes);
  syncSourceLayerScroll();
  restorePendingTensFocus();
}

function syncCodeBoxes() {
  syncTensBoxes();
}

function updateTensBlock(block, tensEditor) {
  const current = currentTensBlock(block) ?? block;
  replaceTensBlock(current, tensEditor.value);
  localStorage.setItem("tensorforge.source.v3", editor.value);
  syncGutter(new Set(), { syncBlocks: false });
  // Reposition every box (without rebuilding, to keep focus): edits in this
  // block shift the line spans of all blocks below it.
  repositionTensBoxes();
  scheduleLiveRun();
}

function handleTensKeydown(e, block, tensEditor) {
  const empty = tensEditor.value.length === 0;
  const collapsed = tensEditor.selectionStart === tensEditor.selectionEnd;
  if (empty && collapsed && (e.key === "Backspace" || e.key === "Delete")) {
    e.preventDefault();
    removeTensBlock(block);
  }
}

function replaceTensBlock(block, code) {
  // setRangeText (not a value assignment) keeps the main editor's undo
  // history intact and leaves the rest of the file byte-identical.
  const [start, end] = lineRangeOffsets(editor.value, block.line, block.endLine);
  editor.setRangeText(`${TENS_OPEN}\n${code}\n${TENS_CLOSE}`, start, end, "preserve");
}

function removeTensBlock(block) {
  // Re-resolve the block: the closure's line span goes stale while the
  // block is being edited, and a stale span would delete following text.
  const current = currentTensBlock(block) ?? block;
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

function currentTensBlock(block) {
  return tensBlocksFromSource(editor.value).find((candidate) => candidate.line === block.line);
}

/// Refresh position, size and the closure-shared block objects of all .tens
/// boxes from the current source, without rebuilding the DOM (rebuilding
/// would destroy the focused .tens editor). Falls back to doing nothing when
/// the block count changed (a fence was typed); the next full sync rebuilds.
function repositionTensBoxes() {
  const fresh = tensBlocksFromSource(editor.value);
  const boxes = [...document.getElementById("code-layer").children];
  if (fresh.length !== boxes.length) return;
  boxes.forEach((box, i) => {
    const block = fresh[i];
    if (box._block) Object.assign(box._block, block);
    box.dataset.startLine = String(block.line);
    box.style.top = `${sourceLineTop(block.line) - 2}px`;
    box.style.height = `${rowSpanHeight(block.line, block.endLine, 3) + 4}px`;
  });
  syncSourceLayerScroll();
}

function isEditingTens() {
  return document.activeElement?.classList?.contains("source-tens-editor");
}

function restorePendingTensFocus() {
  if (!pendingTensFocus) return;
  const focus = pendingTensFocus;
  pendingTensFocus = null;
  const box = document.getElementById("code-layer").querySelector(
    `.source-code-box[data-start-line="${focus.startLine}"] .source-tens-editor`,
  );
  if (!box) return;
  const start = Math.min(focus.selectionStart, box.value.length);
  const end = Math.min(focus.selectionEnd, box.value.length);
  box.focus();
  box.setSelectionRange(start, end);
}

// Pasted images are embedded as data-URL markdown so the .tens file stays
// self-contained.
function handleMarkdownPaste(e) {
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
    editor.setRangeText(md, editor.selectionStart, editor.selectionEnd, "end");
    editor.dispatchEvent(new Event("input", { bubbles: true }));
  };
  reader.readAsDataURL(file);
}

// ---- insert-table popover --------------------------------------------------
// Markdown is the default source mode, so table insertion targets the main
// editor and is disabled only while the caret is inside a .tens block editor.
let tableTarget = null;

function updateInsertTableState() {
  insertTableBtn.disabled = isEditingTens();
}

function openTableMenu() {
  tableTarget = {
    selectionStart: editor.selectionStart,
    selectionEnd: editor.selectionEnd,
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

function insertTableIntoMarkdown() {
  if (!tableTarget) return;
  const rows = Math.max(1, Number(document.getElementById("table-rows").value) || 2);
  const cols = Math.max(1, Number(document.getElementById("table-cols").value) || 3);
  const align = document.getElementById("table-align").value;
  const s = Math.min(tableTarget.selectionStart, editor.value.length);
  const e = Math.min(tableTarget.selectionEnd, editor.value.length);
  const atLineStart = s === 0 || editor.value[s - 1] === "\n";
  const table = `${atLineStart ? "" : "\n"}${markdownTable(rows, cols, align)}\n`;
  editor.setRangeText(table, s, e, "end");
  editor.focus();
  localStorage.setItem("tensorforge.source.v3", editor.value);
  tableTarget = null;
  closeTableMenu();
  clearGutterErrors();
  scheduleLiveRun();
}

let scrollSyncSource = null;
let scrollSyncFrame = null;

function withScrollSyncSource(source, fn) {
  scrollSyncSource = source;
  fn();
  requestAnimationFrame(() => {
    if (scrollSyncSource === source) scrollSyncSource = null;
  });
}

function blockAnchors() {
  return [...output.querySelectorAll(".block[data-line]")]
    .map((el) => ({ el, line: Number(el.dataset.line) }))
    .filter((anchor) => Number.isFinite(anchor.line));
}

function lineAtEditorTop() {
  const y = editor.scrollTop + 8;
  let current = 1;
  for (const row of gutter.children) {
    if (row.offsetTop > y) break;
    current = Number(row.dataset.line) || current;
  }
  return current;
}

function anchorForLine(line) {
  const anchors = blockAnchors();
  if (anchors.length === 0) return null;
  let best = anchors[0];
  for (const anchor of anchors) {
    if (anchor.line > line) break;
    best = anchor;
  }
  return best;
}

function anchorAtOutputTop() {
  const anchors = blockAnchors();
  if (anchors.length === 0) return null;
  const top = output.scrollTop + 8;
  let best = anchors[0];
  for (const anchor of anchors) {
    if (anchor.el.offsetTop > top) break;
    best = anchor;
  }
  return best;
}

function syncOutputToEditor() {
  if (scrollSyncSource === "output") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const anchor = anchorForLine(lineAtEditorTop());
    if (!anchor) return;
    withScrollSyncSource("editor", () => {
      output.scrollTo({ top: Math.max(0, anchor.el.offsetTop - 12), behavior: "auto" });
    });
  });
}

function syncEditorToOutput() {
  if (scrollSyncSource === "editor") return;
  cancelAnimationFrame(scrollSyncFrame);
  scrollSyncFrame = requestAnimationFrame(() => {
    const anchor = anchorAtOutputTop();
    if (!anchor) return;
    withScrollSyncSource("output", () => {
      scrollEditorToLine(anchor.line, { behavior: "auto", align: 0.12 });
    });
  });
}

function scrollEditorToLine(line, options = {}) {
  if (!line) return;
  const align = options.align ?? 1 / 3;
  const behavior = options.behavior ?? "smooth";
  const target = Math.max(0, rowTop(line) + editor.scrollTop - editor.clientHeight * align);
  editor.scrollTo({ top: target, behavior });
}

function focusSourceLine(line) {
  const tens = tensBlocksFromSource(editor.value).find(
    (block) => line >= block.line && line <= block.endLine,
  );
  if (tens) {
    if (!isEditingTens()) syncTensBoxes();
    const box = document.getElementById("code-layer").querySelector(
      `.source-code-box[data-start-line="${tens.line}"] .source-tens-editor`,
    );
    if (box) {
      const bodyLine = Math.max(1, Math.min(tens.lines.length + 1, line - tens.line));
      const offset = offsetForLine(box.value, bodyLine);
      box.focus();
      box.setSelectionRange(offset, offset);
      return;
    }
  }
  const offset = offsetForLine(editor.value, line);
  editor.focus();
  editor.setSelectionRange(offset, offset);
}

function jumpToSourceLine(line, options = {}) {
  scrollEditorToLine(line, { behavior: options.behavior ?? "smooth" });
  if (options.focus) focusSourceLine(line);
  const row = gutter.children[line - 1];
  if (row) {
    row.classList.add("jump");
    setTimeout(() => row.classList.remove("jump"), 1200);
  }
}

function makeBlockJump(el, line) {
  if (line) el.dataset.line = String(line);
  el.title = `Click to jump to line ${line}`;
  el.addEventListener("click", (e) => {
    if (e.target.closest("button")) return; // copy buttons keep their job
    jumpToSourceLine(line, { focus: true, behavior: "auto" });
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
  markGutterErrors(outputs, { syncBlocks: !isEditingTens() });
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
  syncOutputToEditor();
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
editor.addEventListener("focus", updateInsertTableState);
editor.addEventListener("paste", handleMarkdownPaste);
editor.addEventListener("scroll", () => {
  gutter.scrollTop = editor.scrollTop;
  syncSourceLayerScroll();
  // A full rebuild would destroy the focused .tens editor mid-edit.
  if (isEditingTens()) repositionTensBoxes();
  else syncTensBoxes();
  syncOutputToEditor();
});
output.addEventListener("scroll", syncEditorToOutput);
window.addEventListener("resize", refreshEditorGeometry);
// Leaving a .tens editor is the moment to rebuild boxes so stale spans never
// survive an editing session.
document.getElementById("code-layer").addEventListener("focusin", () => queueMicrotask(updateInsertTableState));
document.getElementById("code-layer").addEventListener("focusout", (e) => {
  // Focus moving into the table popover must not rebuild boxes mid-interaction.
  if (tableMenu.contains(e.relatedTarget)) return;
  updateInsertTableState();
  if (!document.getElementById("code-layer").contains(e.relatedTarget)) syncTensBoxes();
});
// initial render on startup
scheduleLiveRun();

runBtn.addEventListener("click", run);
openBtn.addEventListener("click", openFile);
saveBtn.addEventListener("click", () => saveFile());

addNoteBtn.addEventListener("click", insertTensBlock);
exportBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleExportMenu();
});
exportMenu.addEventListener("click", (e) => e.stopPropagation());
// pointerdown + preventDefault keeps focus stable while the popover opens.
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
exportMdBtn.addEventListener("click", exportMarkdown);
exportPdfBtn.addEventListener("click", exportPdf);
themeBtn.addEventListener("click", toggleTheme);
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
    saveFile(e.shiftKey);
  } else if (mod && e.key === "o") {
    e.preventDefault();
    openFile();
  }
});
