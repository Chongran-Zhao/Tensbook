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

editor.value = localStorage.getItem("tensorforge.source") ?? DEFAULT_SOURCE;

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
    const div = document.createElement("div");
    div.className = "error";
    div.textContent = result.error;
    output.appendChild(div);
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
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    run();
  }
});
