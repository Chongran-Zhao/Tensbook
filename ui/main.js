// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

const invoke = window.__TAURI__?.core?.invoke;

const DEFAULT_SOURCE = `# Neo-Hookean strain energy
mu = Scalar("\\mu")
lambda = Scalar("\\lambda")

F = Tensor("\\bm F", order=2, dim=3)

C = F.T * F
J = det(F)
I1 = tr(C)

W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2

P = diff(W, F)
A = diff(P, F)

display(C, mode=symbol)
display(C, mode=components)
display(P, mode=symbol)
display(A, mode=components)
`;

const KATEX_MACROS = { "\\bm": "\\boldsymbol{#1}" };

const editor = document.getElementById("editor");
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

// Path of the currently open file; null = unsaved buffer.
let currentPath = null;

function setCurrentPath(path) {
  currentPath = path;
  filenameEl.textContent = path ? path.split("/").pop() : "";
  filenameEl.title = path ?? "";
}

editor.value = localStorage.getItem("tensorforge.source") ?? DEFAULT_SOURCE;

async function openFile() {
  if (!invoke) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return; // cancelled
  editor.value = opened.source;
  setCurrentPath(opened.path);
  localStorage.setItem("tensorforge.source", editor.value);
}

async function saveFile(forceDialog) {
  if (!invoke) return;
  const path = await invoke("save_tens", {
    source: editor.value,
    path: forceDialog ? null : currentPath,
  }).catch(showError);
  if (path) setCurrentPath(path);
}

function showError(message) {
  output.innerHTML = "";
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = String(message);
  output.appendChild(div);
}

async function run() {
  localStorage.setItem("tensorforge.source", editor.value);
  if (!invoke) {
    output.innerHTML =
      '<div class="error">Tauri bridge unavailable — open this UI through the desktop app.</div>';
    return;
  }
  const result = await invoke("run_tens", { source: editor.value });
  output.innerHTML = "";
  if (!result.ok) {
    showError(result.error);
    return;
  }
  if (result.outputs.length === 0) {
    output.innerHTML =
      '<div class="placeholder">Program ran, but produced no output — add display(...) or export(...) statements.</div>';
    return;
  }
  for (const { header, latex } of result.outputs) {
    const block = document.createElement("div");
    block.className = "block";
    const head = document.createElement("div");
    head.className = "head";
    head.textContent = `[${header}]`;
    const math = document.createElement("div");
    math.className = "math";
    // export(..., format=markdown) wraps in $$...$$; strip for rendering.
    const tex = latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
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
