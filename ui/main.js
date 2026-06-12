// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const DEFAULT_SOURCE = `# Hill's class hyperelasticity with a hand-written spectral strain
mu = Scalar("\\mu")
kappa = Scalar("\\kappa")
m = Scalar("m")
n = Scalar("n")

lambda = ScalarSet("\\lambda", dim=3)
N = VectorSet("\\bm N", dim=3)

lam = Var("\\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)

# C = sum_a lambda_a^2 N_a ⊗ N_a, and E = sum_a Ecr(lambda_a) N_a ⊗ N_a
C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)

# Hill's quadratic energy
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

# thermodynamic force and second Piola-Kirchhoff stress S = T : Q
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
const addNoteBtn = document.getElementById("add-note");
const themeBtn = document.getElementById("theme");
const filenameEl = document.getElementById("filename");
const notesLayer = document.getElementById("notes-layer");

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
  const previousScope = notesScope();
  currentPath = path;
  filenameEl.textContent = path ? path.split("/").pop() : "";
  filenameEl.title = path ?? "";
  migrateScratchNotes(previousScope);
  loadNotesForCurrentPath();
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

// ---- markdown notes ---------------------------------------------------------

const NOTES_KEY = "tensorforge.notes.v1";
let notes = [];
let notesReady = false;

function notesScope() {
  return currentPath ?? "__scratch__";
}

function readNotesStore() {
  try {
    const raw = localStorage.getItem(NOTES_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeNotesStore(store) {
  localStorage.setItem(NOTES_KEY, JSON.stringify(store));
}

function normalizeNote(note, fallbackIndex) {
  return {
    id: String(note.id ?? `note-${Date.now()}-${fallbackIndex}`),
    x: Number.isFinite(note.x) ? note.x : 48 + fallbackIndex * 18,
    y: Number.isFinite(note.y) ? note.y : 48 + fallbackIndex * 18,
    md: typeof note.md === "string" ? note.md : "",
  };
}

function persistNotes() {
  const store = readNotesStore();
  if (notes.length === 0) delete store[notesScope()];
  else store[notesScope()] = notes;
  writeNotesStore(store);
}

function migrateScratchNotes(previousScope) {
  if (!notesReady || previousScope !== "__scratch__" || !currentPath || notes.length === 0) {
    return;
  }
  const store = readNotesStore();
  if (!store[currentPath]) {
    store[currentPath] = notes;
    delete store.__scratch__;
    writeNotesStore(store);
  }
}

function loadNotesForCurrentPath() {
  if (!notesLayer) return;
  const store = readNotesStore();
  notes = Array.isArray(store[notesScope()])
    ? store[notesScope()].map(normalizeNote)
    : [];
  notesReady = true;
  renderNotes();
}

function addNote() {
  const rect = notesLayer.getBoundingClientRect();
  const note = {
    id: `note-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    x: Math.max(24, rect.width / 2 - 120 + Math.random() * 32),
    y: Math.max(24, rect.height / 2 - 80 + Math.random() * 32),
    md: "",
  };
  notes.push(note);
  persistNotes();
  renderNotes(note.id);
}

function renderNotes(editingId = null) {
  notesLayer.replaceChildren(...notes.map((note) => noteElement(note, note.id === editingId)));
}

function noteElement(note, editing) {
  const el = document.createElement("section");
  el.className = "note";
  el.dataset.noteId = note.id;

  const bar = document.createElement("div");
  bar.className = "note-bar";
  const del = document.createElement("button");
  del.className = "note-delete";
  del.type = "button";
  del.title = "Delete note";
  del.textContent = "×";
  del.addEventListener("click", () => deleteNote(note.id));
  bar.appendChild(del);

  const body = document.createElement("div");
  body.className = "note-body";
  if (editing) renderNoteEditor(body, note);
  else renderNoteMarkdown(body, note);

  el.append(bar, body);
  placeNote(el, note);
  attachDrag(bar, el, note);
  return el;
}

function renderNoteEditor(body, note) {
  const textarea = document.createElement("textarea");
  textarea.value = note.md;
  textarea.placeholder = "Markdown";
  body.appendChild(textarea);

  const fit = () => {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
  };
  textarea.addEventListener("input", fit);
  textarea.addEventListener("blur", () => commitNoteEdit(note, textarea.value));
  textarea.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      textarea.blur();
    }
  });
  requestAnimationFrame(() => {
    textarea.focus();
    fit();
  });
}

function renderNoteMarkdown(body, note) {
  body.innerHTML = renderMarkdown(note.md || "");
  body.addEventListener("dblclick", () => renderNotes(note.id));
}

function commitNoteEdit(note, md) {
  note.md = md;
  if (note.md.trim() === "") deleteNote(note.id);
  else {
    persistNotes();
    renderNotes();
  }
}

function deleteNote(id) {
  notes = notes.filter((note) => note.id !== id);
  persistNotes();
  renderNotes();
}

function attachDrag(bar, el, note) {
  bar.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".note-delete")) return;
    e.preventDefault();
    notesLayer.appendChild(el);
    bar.setPointerCapture(e.pointerId);
    const layer = notesLayer.getBoundingClientRect();
    const dx = e.clientX - layer.left - note.x;
    const dy = e.clientY - layer.top - note.y;

    const move = (ev) => {
      note.x = ev.clientX - layer.left - dx;
      note.y = ev.clientY - layer.top - dy;
      placeNote(el, note);
      persistNotes();
    };
    const done = () => {
      bar.removeEventListener("pointermove", move);
      persistNotes();
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", done, { once: true });
    bar.addEventListener("pointercancel", done, { once: true });
  });
}

function placeNote(el, note) {
  const layer = notesLayer.getBoundingClientRect();
  const maxX = Math.max(0, layer.width - (el.offsetWidth || 180));
  const maxY = Math.max(0, layer.height - (el.offsetHeight || 80));
  note.x = clamp(note.x, 0, maxX);
  note.y = clamp(note.y, 0, maxY);
  el.style.left = `${note.x}px`;
  el.style.top = `${note.y}px`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
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

loadNotesForCurrentPath();
window.addEventListener("resize", () => {
  for (const note of notes) {
    const el = notesLayer.querySelector(`[data-note-id="${CSS.escape(note.id)}"]`);
    if (el) placeNote(el, note);
  }
  persistNotes();
});

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

function renderOutputs(outputs) {
  output.innerHTML = "";
  markGutterErrors(outputs);
  if (outputs.length === 0) {
    output.innerHTML =
      '<div class="placeholder">No output yet — add display(...) or export(...) statements.</div>';
    return;
  }
  for (const { header, latex, line, error } of outputs) {
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
      output.appendChild(block);
      continue;
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
    output.appendChild(block);
  }
}

async function run() {
  localStorage.setItem("tensorforge.source.v3", editor.value);
  if (!invoke) {
    output.innerHTML =
      '<div class="error">Tauri bridge unavailable — open this UI through the desktop app.</div>';
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
  if (!invoke) return;
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
addNoteBtn.addEventListener("click", addNote);
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
