// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const DEFAULT_SOURCE = `## Hill's class hyperelasticity
# Hand-written spectral strain with a symbolic Hill-CR scale function.
#
mu = Scalar("\\mu")
kappa = Scalar("\\kappa")
m = Scalar("m")
n = Scalar("n")

lambda = ScalarSet("\\lambda", dim=3)
N = VectorSet("\\bm N", dim=3)

lam = Var("\\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)

### Spectral strain
# C = sum_a lambda_a^2 N_a ⊗ N_a
# E = sum_a Ecr(lambda_a) N_a ⊗ N_a
C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)

### Energy and stress
# Hill's quadratic energy.
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

# Thermodynamic force and second Piola-Kirchhoff stress S = T : Q.
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

const editor = document.getElementById("editor");
const gutter = document.getElementById("gutter");
const output = document.getElementById("output");
const runBtn = document.getElementById("run");
const openBtn = document.getElementById("open");
const saveBtn = document.getElementById("save");
const saveAsBtn = document.getElementById("saveas");
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

function syncGutter(errorLines = new Set()) {
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
  }
  gutter.scrollTop = editor.scrollTop;
}

function clearGutterErrors() {
  syncGutter();
}

function markGutterErrors(outputs) {
  const lines = new Set(
    outputs
      .filter((item) => item.error && item.line)
      .map((item) => Number(item.line)),
  );
  syncGutter(lines);
}

syncGutter();

async function openFile() {
  if (!invoke) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return; // cancelled
  editor.value = opened.source;
  setCurrentPath(opened.path);
  localStorage.setItem("tensorforge.source.v3", editor.value);
  clearGutterErrors();
  scheduleLiveRun();
}

async function saveFile(forceDialog) {
  if (!invoke) return;
  localStorage.setItem("tensorforge.source.v3", editor.value);
  if (!forceDialog && !currentPath) return;
  const path = await invoke("save_tens", {
    source: editor.value,
    path: forceDialog ? null : currentPath,
  }).catch(showError);
  if (path) setCurrentPath(path);
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
  const blocks = [];
  let current = null;
  const lines = source.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const trimmed = raw.trimStart();
    if (trimmed.startsWith("#")) {
      const text = trimmed.startsWith("# ") ? trimmed.slice(2) : trimmed.slice(1);
      if (!current) current = { line: i + 1, lines: [] };
      current.lines.push(text);
      continue;
    }
    if (current) {
      pushMarkdownBlock(blocks, current);
      current = null;
    }
  }
  if (current) pushMarkdownBlock(blocks, current);
  return blocks;
}

function pushMarkdownBlock(blocks, block) {
  const markdown = block.lines.join("\n").trim();
  if (!markdown) return;
  blocks.push({
    kind: "markdown",
    line: block.line,
    markdown,
  });
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
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

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdownBlock(block) {
  const el = document.createElement("div");
  el.className = "block";
  const head = document.createElement("div");
  head.className = "head";
  head.textContent = `[markdown, line ${block.line}]`;
  const doc = document.createElement("div");
  doc.className = "markdown-doc";
  doc.innerHTML = renderMarkdown(block.markdown);
  el.append(head, doc);
  return el;
}

function renderOutputBlock(item) {
  const { header, latex, line, error } = item;
  const block = document.createElement("div");
  block.className = "block";
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
  markGutterErrors(outputs);
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
});
// initial render on startup
scheduleLiveRun();

runBtn.addEventListener("click", run);
openBtn.addEventListener("click", openFile);
saveBtn.addEventListener("click", () => saveFile(false));
saveAsBtn.addEventListener("click", () => saveFile(true));
themeBtn.addEventListener("click", toggleTheme);
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
