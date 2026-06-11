// TensorForge frontend: send DSL source to the Rust engine, render the
// returned LaTeX with KaTeX. `\bm` is mapped to `\boldsymbol` via a macro
// because KaTeX does not ship the bm package.

import { setupCompletion } from "./completion.js";

const invoke = window.__TAURI__?.core?.invoke;

const DEFAULT_SOURCE = `# Hill's class hyperelasticity with the Curnier-Rakotomanana strain
mu = Scalar("\\mu")
kappa = Scalar("\\kappa")
m = Scalar("m")
n = Scalar("n")

F = Tensor("\\bm F", order=2, dim=3)
C = F.T * F

# generalized strain E(C) = sum_a E(lambda_a) M_a,  E(l) = (l^m - l^-n)/(m+n)
E = gstrain(C, scale=CR, m=m, n=n)

# Hill's quadratic energy
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

# thermodynamic force and second Piola-Kirchhoff stress S = T : Q
T = diff(W, E)
S = 2 * diff(W, C)

display(E, mode=spectral)
display(W, mode=symbol)
display(T, mode=symbol)
display(S, mode=symbol)
display(S, mode=spectral)
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

editor.value = localStorage.getItem("tensorforge.source") ?? DEFAULT_SOURCE;
setupCompletion(editor);

async function openFile() {
  if (!invoke) return;
  const opened = await invoke("open_tens").catch(showError);
  if (!opened) return; // cancelled
  editor.value = opened.source;
  setCurrentPath(opened.path);
  localStorage.setItem("tensorforge.source", editor.value);
  scheduleLiveRun();
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
  localStorage.setItem("tensorforge.source", editor.value);
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
  localStorage.setItem("tensorforge.source", editor.value);
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

editor.addEventListener("input", scheduleLiveRun);
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
