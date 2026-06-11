// Editor completion: ghost-text suggestions + function signature hints.
//
// A hidden "mirror" div replicates the textarea's text up to the caret to
// measure the caret's pixel position; the ghost suggestion and the signature
// card are absolutely positioned at those coordinates. No dependencies.
//
// Keys: Tab accepts the ghost suggestion (or indents), Esc dismisses.

const BUILTINS = [
  { name: "Scalar", sig: 'Scalar("\\mu")', doc: "declare a symbolic scalar" },
  {
    name: "Tensor",
    sig: 'Tensor("\\bm F", order=2, dim=3, …)',
    doc: "declare a tensor; kwargs: order, dim, identity, symmetric, antisymmetric, orthogonal, isotropic",
  },
  { name: "diff", sig: "diff(expr, X)", doc: "symbolic derivative ∂expr/∂X — X: tensor variable, compound tensor (e.g. C = F.T*F), or scalar symbol" },
  { name: "det", sig: "det(A)", doc: "determinant, kept symbolic" },
  { name: "tr", sig: "tr(A)", doc: "trace, kept symbolic" },
  { name: "log", sig: "log(x)  |  log(C)", doc: "scalar log; tensor log via spectral form (requires symmetric C)" },
  { name: "sqrt", sig: "sqrt(C)", doc: "tensor square root via spectral form (requires symmetric C)" },
  { name: "exp", sig: "exp(C)", doc: "tensor exponential via spectral form (requires symmetric C)" },
  { name: "inv", sig: "inv(A)", doc: "symbolic tensor inverse" },
  { name: "outer", sig: "outer(A, B)", doc: "tensor product A ⊗ B (orders add)" },
  { name: "otimes", sig: "otimes(A, B)", doc: "tensor product A ⊗ B — alias of outer" },
  { name: "dot", sig: "dot(A, B)", doc: "single contraction AB" },
  { name: "ddot", sig: "ddot(A, B)", doc: "double contraction A : B — a scalar; infix `A : B` also works" },
  { name: "spectral", sig: "spectral(C)", doc: "spectral decomposition Σ c_a N_a ⊗ N_a (requires provably symmetric C)" },
  { name: "simplify", sig: "simplify(expr, rules=…)", doc: "exact rewriting; rules = algebra | tensor | continuum (default)" },
  { name: "display", sig: "display(expr, mode=…)", doc: "mode = symbol | components | matrix | block_components" },
  { name: "export", sig: "export(expr, format=…)", doc: "format = latex | markdown" },
];

const BUILTIN_MAP = new Map(BUILTINS.map((b) => [b.name, b]));

const ENUM_VALUES = {
  mode: ["symbol", "components", "matrix", "block_components"],
  format: ["latex", "markdown"],
  rules: ["algebra", "tensor", "continuum"],
};

const TENSOR_KWARGS = [
  "order", "dim", "identity", "symmetric", "antisymmetric", "orthogonal", "isotropic",
];

const KEYWORDS = ["true", "false"];

export function setupCompletion(editor) {
  const wrap = editor.parentElement; // #editor-wrap, position: relative

  const mirror = document.createElement("div");
  mirror.className = "cm-mirror";
  const ghost = document.createElement("span");
  ghost.className = "cm-ghost";
  const hint = document.createElement("div");
  hint.className = "cm-hint";
  wrap.append(mirror, ghost, hint);

  let suggestion = null; // remainder text currently shown as ghost

  function syncMirrorStyles() {
    const cs = getComputedStyle(editor);
    for (const prop of [
      "fontFamily", "fontSize", "lineHeight", "letterSpacing",
      "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
      "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
      "tabSize",
    ]) {
      mirror.style[prop] = cs[prop];
    }
    mirror.style.width = editor.clientWidth + "px";
  }

  /** Pixel position of the caret, relative to the wrapper. */
  function caretPosition() {
    syncMirrorStyles();
    mirror.textContent = "";
    mirror.appendChild(
      document.createTextNode(editor.value.slice(0, editor.selectionStart)),
    );
    const marker = document.createElement("span");
    marker.textContent = "​";
    mirror.appendChild(marker);
    return {
      left: marker.offsetLeft - editor.scrollLeft,
      top: marker.offsetTop - editor.scrollTop,
      height: marker.offsetHeight,
    };
  }

  /** Identifiers assigned in the buffer, e.g. C, J, W. */
  function userVariables() {
    const vars = new Set();
    for (const m of editor.value.matchAll(/^[ \t]*([A-Za-z_]\w*)[ \t]*=/gm)) {
      vars.add(m[1]);
    }
    return [...vars];
  }

  /** The identifier being typed immediately before the caret. */
  function currentToken() {
    const upto = editor.value.slice(0, editor.selectionStart);
    const m = upto.match(/([A-Za-z_]\w*)$/);
    return m ? m[1] : "";
  }

  /** Innermost unclosed call before the caret, e.g. "diff" or "Tensor". */
  function enclosingCall() {
    const upto = editor.value.slice(0, editor.selectionStart);
    let depth = 0;
    for (let i = upto.length - 1; i >= 0; i--) {
      const ch = upto[i];
      if (ch === ")") depth++;
      else if (ch === "(") {
        if (depth === 0) {
          const head = upto.slice(0, i).match(/([A-Za-z_]\w*)\s*$/);
          return head ? head[1] : null;
        }
        depth--;
      } else if (ch === "\n" && depth === 0) {
        return null; // calls don't span lines in .tens
      }
    }
    return null;
  }

  /** Ordered completion candidates for the current context. */
  function candidates(token) {
    const upto = editor.value.slice(0, editor.selectionStart - token.length);
    // value position: `mode=`, `format=`, `rules=` directly before the token
    const kw = upto.match(/([A-Za-z_]\w*)\s*=\s*$/);
    if (kw && ENUM_VALUES[kw[1]]) return ENUM_VALUES[kw[1]];
    if (kw) return KEYWORDS; // identity=tru… etc.

    const call = enclosingCall();
    const out = [];
    if (call === "Tensor") out.push(...TENSOR_KWARGS);
    out.push(...BUILTINS.map((b) => b.name));
    out.push(...KEYWORDS);
    out.push(...userVariables());
    return out;
  }

  function updateGhost() {
    suggestion = null;
    ghost.style.display = "none";
    if (editor.selectionStart !== editor.selectionEnd) return;
    const token = currentToken();
    if (!token) return;
    // only complete at the end of the token (nothing word-like after caret)
    if (/^\w/.test(editor.value.slice(editor.selectionStart))) return;
    const match = candidates(token).find(
      (c) => c.startsWith(token) && c !== token,
    );
    if (!match) return;
    suggestion = match.slice(token.length);
    const pos = caretPosition();
    ghost.textContent = suggestion;
    ghost.style.left = pos.left + "px";
    ghost.style.top = pos.top + "px";
    ghost.style.display = "block";
  }

  function updateHint() {
    const call = enclosingCall();
    const info = call && BUILTIN_MAP.get(call);
    if (!info) {
      hint.style.display = "none";
      return;
    }
    hint.innerHTML = "";
    const sig = document.createElement("div");
    sig.className = "cm-hint-sig";
    sig.textContent = info.sig;
    const doc = document.createElement("div");
    doc.className = "cm-hint-doc";
    doc.textContent = info.doc;
    hint.append(sig, doc);
    const pos = caretPosition();
    hint.style.left = Math.min(pos.left, wrap.clientWidth - 280) + "px";
    hint.style.top = pos.top + pos.height + 4 + "px";
    hint.style.display = "block";
  }

  function refresh() {
    updateGhost();
    updateHint();
  }

  function dismiss() {
    suggestion = null;
    ghost.style.display = "none";
    hint.style.display = "none";
  }

  editor.addEventListener("input", refresh);
  editor.addEventListener("click", refresh);
  editor.addEventListener("scroll", dismiss);
  editor.addEventListener("blur", dismiss);
  editor.addEventListener("keyup", (e) => {
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
      refresh();
    }
  });
  editor.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      dismiss();
      return;
    }
    if (e.key !== "Tab" || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    const start = editor.selectionStart;
    const insert = suggestion ?? "    ";
    // execCommand keeps the undo stack intact (still the sane option for textarea)
    if (!document.execCommand("insertText", false, insert)) {
      editor.setRangeText(insert, start, editor.selectionEnd, "end");
    }
    refresh();
  });
}
