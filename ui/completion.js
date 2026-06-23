// Completion data and CodeMirror completion sources.

export const BUILTINS = [
  { name: "Scalar", sig: 'Scalar("\\mu")', doc: "declare a symbolic scalar" },
  {
    name: "Tensor",
    sig: 'Tensor("\\bm F", order=2, dim=3, ...)',
    doc: "declare a tensor; kwargs: order, dim, identity, symmetric, antisymmetric, orthogonal, isotropic",
  },
  { name: "Diff", sig: "Diff(expr, X, order=1)", doc: "evaluated derivative for explicit scalar/tensor expressions" },
  { name: "Derivative", sig: "Derivative(f, x, order=1)", doc: "formal derivative or partial derivative of an unknown Function(...)" },
  { name: "Det", sig: "Det(A)", doc: "determinant, kept symbolic" },
  { name: "Tr", sig: "Tr(A)", doc: "trace, kept symbolic" },
  { name: "Inv", sig: "Inv(A)", doc: "symbolic tensor inverse" },
  { name: "Simplify", sig: "Simplify(expr, rules=...)", doc: "exact rewriting; rules = algebra | tensor | continuum (default)" },
  { name: "Sum", sig: "Sum(expr, a)", doc: "indexed symbolic sum over the set index a" },
  { name: "log", sig: "log(x)", doc: "scalar logarithm; tensor spectral forms are written with Sum(log(c[a]) * N[a] & N[a], a)" },
  { name: "sqrt", sig: "sqrt(x)", doc: "scalar square root; tensor spectral forms are written with indexed sums" },
  { name: "exp", sig: "exp(x)", doc: "scalar exponential; tensor spectral forms are written with indexed sums" },
  { name: "sin", sig: "sin(x)", doc: "scalar sine with symbolic derivative and integration rules" },
  { name: "cos", sig: "cos(x)", doc: "scalar cosine with symbolic derivative and integration rules" },
  { name: "tan", sig: "tan(x)", doc: "scalar tangent with symbolic derivative and integration rules" },
  { name: "sinh", sig: "sinh(x)", doc: "hyperbolic sine (symbolic, with derivative rule)" },
  { name: "cosh", sig: "cosh(x)", doc: "hyperbolic cosine (symbolic, with derivative rule)" },
  { name: "tanh", sig: "tanh(x)", doc: "hyperbolic tangent (symbolic, with derivative rule)" },
  { name: "Var", sig: 'Var("\\lambda")', doc: "declare a scalar function argument" },
  { name: "Function", sig: 'Function("y", x)', doc: "declare an unknown scalar function; accepts one or more Var arguments" },
  { name: "Equation", sig: "Equation(lhs, rhs)", doc: "declare a scalar equation for ODE/PDE classification and solving" },
  { name: "Integrate", sig: "Integrate(expr, x)", doc: "rule-based scalar integration; returns Integral(...) when unsupported" },
  { name: "Integral", sig: "Integral(expr, x)", doc: "formal unevaluated scalar integral" },
  { name: "ClassifyODE", sig: "ClassifyODE(eq, y, x)", doc: "classify an ODE/PDE by order, linearity, and first-order subtype" },
  { name: "SolveODE", sig: "SolveODE(eq, y, x, ic=IC(y(x0), y0))", doc: "solve supported first-order linear, separable, or exact ODEs" },
  { name: "IC", sig: "IC(y(x0), y0)", doc: "initial condition for supported ODE solvers" },
  { name: "ScalarSet", sig: 'ScalarSet("\\lambda", dim=3)', doc: "declare an indexed scalar family lambda[a]" },
  { name: "VectorSet", sig: 'VectorSet("\\bm N", dim=3)', doc: "declare an indexed vector family N[a]" },
  { name: "Spec_Decomp", sig: "[c, N] = Spec_Decomp(C)", doc: "symbolic eigendecomposition of a diagonal component-filled tensor into declared sets" },
  { name: "Spectral", sig: '[lambda, N] = Spectral(C, "\\lambda", "\\bm N")', doc: "declare symbolic eigenvalue/eigenvector sets for a symmetric tensor" },
];

const ENUM_VALUES = {
  rules: ["algebra", "tensor", "continuum"],
};

const SHOW_MODES = ["symbol", "components", "matrix", "block_components", "details", "solution", "steps"];
const TENSOR_KWARGS = [
  "order",
  "dim",
  "identity",
  "symmetric",
  "antisymmetric",
  "orthogonal",
  "isotropic",
];
const KEYWORDS = ["true", "false"];

export const MARKDOWN_COMMANDS = [
  { name: "h1", sig: "/h1", doc: "insert a level-1 heading", text: "# ", cursor: 2 },
  { name: "h2", sig: "/h2", doc: "insert a level-2 heading", text: "## ", cursor: 3 },
  { name: "h3", sig: "/h3", doc: "insert a level-3 heading", text: "### ", cursor: 4 },
  { name: "bold", sig: "/bold", doc: "insert bold text markers", text: "****", cursor: 2 },
  { name: "italic", sig: "/italic", doc: "insert italic text markers", text: "**", cursor: 1 },
  { name: "quote", sig: "/quote", doc: "insert a block quote marker", text: "> ", cursor: 2 },
  { name: "list", sig: "/list", doc: "insert a bullet list item", text: "- ", cursor: 2 },
  { name: "todo", sig: "/todo", doc: "insert a task list item", text: "- [ ] ", cursor: 6 },
  { name: "link", sig: "/link", doc: "insert a Markdown link", text: "[](url)", cursor: 1 },
  { name: "image", sig: "/image", doc: "insert a Markdown image placeholder", text: "![alt](url)", cursor: 2 },
  { name: "code", sig: "/code", doc: "insert a fenced code block", text: "```\n\n```", cursor: 4 },
  { name: "math", sig: "/math", doc: "insert a display-math block", text: "$$\n\n$$", cursor: 3 },
  { name: "table", sig: "/table", doc: "insert a small Markdown table", text: "|   |   |\n| --- | --- |\n|   |   |", cursor: 2 },
  { name: "hr", sig: "/hr", doc: "insert a horizontal rule", text: "---", cursor: 3 },
];

let completionSymbols = new Map();

export function setCompletionSymbols(symbols = []) {
  completionSymbols = new Map(symbols.map((s) => [s.name, s]));
  window.dispatchEvent(new CustomEvent("tensorforge:completion-symbols"));
}

export function completionSymbolsSnapshot() {
  return completionSymbols;
}

function cmOption(label, detail = "", info = "", apply = null) {
  const option = { label, type: "keyword" };
  if (detail) option.detail = detail;
  if (info) option.info = info;
  if (apply) option.apply = apply;
  return option;
}

function wordBefore(context) {
  return context.matchBefore(/[A-Za-z_]\w*/);
}

function linePrefix(context) {
  const line = context.state.doc.lineAt(context.pos);
  return context.state.doc.sliceString(line.from, context.pos);
}

function enclosingCall(prefix) {
  let depth = 0;
  for (let i = prefix.length - 1; i >= 0; i--) {
    const ch = prefix[i];
    if (ch === ")") depth++;
    else if (ch === "(") {
      if (depth === 0) {
        const head = prefix.slice(0, i).match(/([A-Za-z_]\w*)\s*$/);
        return head ? head[1] : null;
      }
      depth--;
    }
  }
  return null;
}

function showExpressionBefore(prefix) {
  const m = prefix.match(/\.show\s*\([^()\n]*$/);
  if (!m) return "";
  let expr = prefix.slice(0, m.index).trim();
  const rowStart = Math.max(expr.lastIndexOf("["), expr.lastIndexOf(","));
  if (rowStart >= 0) expr = expr.slice(rowStart + 1).trim();
  return expr;
}

function showModeCandidates(expr = "") {
  const component = expr.match(/^([A-Za-z_]\w*)(\[[^\]]+\])+$/);
  if (component && completionSymbols.has(component[1])) return ["symbol"];
  const info = completionSymbols.get(expr);
  const modes = info?.display_modes
    ?.filter((mode) => mode.state === "available")
    .map((mode) => mode.mode);
  return modes?.length ? modes : SHOW_MODES;
}

function assignedNames(source) {
  const names = [];
  for (const m of source.matchAll(/^[ \t]*([A-Za-z_]\w*)[ \t]*=/gm)) names.push(m[1]);
  return names;
}

export function tensorForgeCompletionSource(context) {
  const word = wordBefore(context);
  const prefix = linePrefix(context);
  const dotShow = prefix.endsWith(".");
  if (!word && !dotShow && !context.explicit) return null;

  const from = dotShow ? context.pos : (word?.from ?? context.pos);
  const kw = prefix.slice(0, from - context.state.doc.lineAt(context.pos).from).match(/([A-Za-z_]\w*)\s*=\s*$/);
  const call = enclosingCall(prefix);
  const options = [];

  if (dotShow) {
    options.push(cmOption("show", "method", "render output", "show()"));
  } else if (call === "show") {
    const expr = showExpressionBefore(prefix);
    options.push(...showModeCandidates(expr).map((mode) => cmOption(mode, "show mode")));
  } else if (kw && ENUM_VALUES[kw[1]]) {
    options.push(...ENUM_VALUES[kw[1]].map((name) => cmOption(name, "value")));
  } else if (kw) {
    options.push(...KEYWORDS.map((name) => cmOption(name, "value")));
  } else {
    if (call === "Tensor") options.push(...TENSOR_KWARGS.map((name) => cmOption(name, "kwarg")));
    options.push(
      ...BUILTINS.map((b) => cmOption(b.name, b.sig, b.doc)),
      ...KEYWORDS.map((name) => cmOption(name, "keyword")),
      ...[...completionSymbols.keys()].map((name) => cmOption(name, "symbol")),
      ...assignedNames(context.state.doc.toString()).map((name) => cmOption(name, "variable")),
    );
  }

  const seen = new Set();
  return {
    from,
    options: options.filter((option) => {
      if (seen.has(option.label)) return false;
      seen.add(option.label);
      return true;
    }),
  };
}

export function markdownSlashCompletionSource(context) {
  const line = context.state.doc.lineAt(context.pos);
  const prefix = context.state.doc.sliceString(line.from, context.pos);
  const match = prefix.match(/^(\s*)\/([A-Za-z-]*)$/);
  if (!match && !context.explicit) return null;
  if (!match) return null;
  const from = line.from + match[1].length;
  const token = match[2].toLowerCase();
  return {
    from,
    options: MARKDOWN_COMMANDS.filter((command) => command.name.startsWith(token)).map((command) => ({
      label: `/${command.name}`,
      type: "keyword",
      detail: command.sig,
      info: command.doc,
      apply(view, _completion, fromPos, toPos) {
        view.dispatch({
          changes: { from: fromPos, to: toPos, insert: command.text },
          selection: { anchor: fromPos + command.cursor },
        });
      },
    })),
  };
}
