// Completion + signature data for tens code.
// Autocomplete is frontend-only: static builtin schemas drive both the popup and
// the small signature hint rendered by main.js.

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
  { name: "log", sig: "log(x)", doc: "scalar logarithm" },
  { name: "sqrt", sig: "sqrt(x)", doc: "scalar square root" },
  { name: "exp", sig: "exp(x)", doc: "scalar exponential" },
  { name: "sin", sig: "sin(x)", doc: "scalar sine" },
  { name: "cos", sig: "cos(x)", doc: "scalar cosine" },
  { name: "tan", sig: "tan(x)", doc: "scalar tangent" },
  { name: "sinh", sig: "sinh(x)", doc: "hyperbolic sine" },
  { name: "cosh", sig: "cosh(x)", doc: "hyperbolic cosine" },
  { name: "tanh", sig: "tanh(x)", doc: "hyperbolic tangent" },
  { name: "Var", sig: 'Var("\\lambda")', doc: "declare a scalar function argument" },
  { name: "Function", sig: 'Function("y", x)', doc: "declare an unknown scalar function; accepts one or more Var arguments" },
  { name: "Equation", sig: "Equation(lhs, rhs)", doc: "declare a scalar equation for ODE/PDE classification and solving" },
  { name: "ODE", sig: "ODE(eq, y, x, BoundaryCondition(...))", doc: "declare an ODE/PDE problem object" },
  { name: "BoundaryCondition", sig: "BoundaryCondition(y(x0), y0)", doc: "boundary condition for supported ODE solvers" },
  { name: "Integrate", sig: "Integrate(expr, x)", doc: "rule-based scalar integration" },
  { name: "Integral", sig: "Integral(expr, x)", doc: "formal unevaluated scalar integral" },
  { name: "ScalarSet", sig: 'ScalarSet("\\lambda", dim=3)', doc: "declare an indexed scalar family lambda[a]" },
  { name: "VectorSet", sig: 'VectorSet("\\bm N", dim=3)', doc: "declare an indexed vector family N[a]" },
  { name: "Spec_Decomp", sig: "Spec_Decomp(C)", doc: "symbolic eigendecomposition of a diagonal component-filled tensor" },
  { name: "Spectral", sig: 'Spectral(C, "\\lambda", "\\bm N")', doc: "declare symbolic eigenvalue/eigenvector sets for a symmetric tensor" },
];

const BUILTIN_BY_NAME = new Map(BUILTINS.map((builtin) => [builtin.name, builtin]));
const ALL_SHOW_MODES = ["symbol", "components", "matrix", "block_components", "details", "solution", "steps"];
const SAFE_SHOW_MODES = ["symbol"];
const SHOW_MODE_DETAILS = {
  symbol: "symbolic display",
  components: "component display",
  matrix: "matrix display",
  block_components: "block component display",
  details: "classification details",
  solution: "solution only",
  steps: "solution steps",
};
const BOOLEAN_VALUES = ["true", "false"];
const ORDER_VALUES = ["1", "2", "4"];
const DIM_VALUES = ["2", "3"];
const SIMPLIFY_RULES = ["continuum", "tensor", "algebra"];
const VIEW_SOURCE_VALUES = ["on", "off"];

const EXPRESSION_BUILTINS = [
  "Diff",
  "Derivative",
  "Det",
  "Tr",
  "Inv",
  "Simplify",
  "Sum",
  "log",
  "sqrt",
  "exp",
  "sin",
  "cos",
  "tan",
  "sinh",
  "cosh",
  "tanh",
  "Equation",
  "ODE",
  "BoundaryCondition",
  "Integrate",
  "Integral",
];

const tensorKwargs = [
  { name: "order", detail: "integer tensor order", values: ORDER_VALUES },
  { name: "dim", detail: "tensor dimension", values: DIM_VALUES },
  { name: "identity", detail: "boolean identity tensor flag", values: BOOLEAN_VALUES },
  { name: "symmetric", detail: "boolean symmetry flag", values: BOOLEAN_VALUES },
  { name: "antisymmetric", detail: "boolean antisymmetry flag", values: BOOLEAN_VALUES },
  { name: "orthogonal", detail: "boolean orthogonal tensor flag", values: BOOLEAN_VALUES },
  { name: "isotropic", detail: "boolean isotropic tensor flag", values: BOOLEAN_VALUES },
];

const setKwargs = [{ name: "dim", detail: "set dimension", values: DIM_VALUES }];
const orderKwarg = { name: "order", detail: "integer derivative order", values: ORDER_VALUES };

const BUILTIN_SCHEMAS = {
  Scalar: {
    hintParams: ["label"],
    positional: [{ name: "label", detail: "LaTeX scalar symbol string", suggestions: [{ label: '"\\mu"', detail: "label example" }] }],
  },
  Tensor: {
    hintParams: ["label", "order=2", "dim=3", "symmetric=false", "..."],
    positional: [{ name: "label", detail: "LaTeX tensor symbol string", suggestions: [{ label: '"\\bm F"', detail: "label example" }] }],
    kwargs: tensorKwargs,
  },
  ScalarSet: {
    hintParams: ["label", "dim=3"],
    positional: [{ name: "label", detail: "LaTeX scalar-set symbol string", suggestions: [{ label: '"\\lambda"', detail: "label example" }] }],
    kwargs: setKwargs,
  },
  VectorSet: {
    hintParams: ["label", "dim=3"],
    positional: [{ name: "label", detail: "LaTeX vector-set symbol string", suggestions: [{ label: '"\\bm N"', detail: "label example" }] }],
    kwargs: setKwargs,
  },
  Var: {
    hintParams: ["label"],
    positional: [{ name: "label", detail: "LaTeX variable symbol string", suggestions: [{ label: '"x"', detail: "label example" }] }],
  },
  Function: {
    hintParams: ["name", "arg", "..."],
    positional: [
      { name: "name", detail: "unknown function name string", suggestions: [{ label: '"y"', detail: "name example" }] },
      { name: "arg", detail: "independent variable", kind: "expression", variadic: true },
    ],
  },
  Diff: {
    hintParams: ["expr", "X", "order=1"],
    positional: [
      { name: "expr", detail: "explicit scalar/tensor expression", kind: "expression" },
      { name: "X", detail: "differentiation variable or tensor", kind: "expression" },
    ],
    kwargs: [orderKwarg],
  },
  Derivative: {
    hintParams: ["f", "x", "order=1"],
    positional: [
      { name: "f", detail: "unknown Function(...) value", kind: "expression" },
      { name: "x", detail: "formal derivative variable", kind: "expression" },
    ],
    kwargs: [orderKwarg],
  },
  Simplify: {
    hintParams: ["expr", "rules=continuum"],
    positional: [{ name: "expr", detail: "expression to simplify", kind: "expression" }],
    kwargs: [{ name: "rules", detail: "simplification rule set", values: SIMPLIFY_RULES }],
  },
  Equation: {
    hintParams: ["lhs", "rhs"],
    positional: [
      { name: "lhs", detail: "left-hand scalar expression", kind: "expression" },
      { name: "rhs", detail: "right-hand scalar expression", kind: "expression" },
    ],
  },
  ODE: {
    hintParams: ["eq", "y", "x", "BoundaryCondition(...)"],
    positional: [
      { name: "eq", detail: "Equation(...) object", kind: "expression" },
      { name: "y", detail: "unknown Function(...) value", kind: "expression" },
      { name: "x", detail: "independent variable", kind: "expression" },
      { name: "BoundaryCondition(...)", detail: "optional boundary condition", kind: "expression" },
    ],
  },
  BoundaryCondition: {
    hintParams: ["y(x0)", "y0"],
    positional: [
      { name: "y(x0)", detail: "function value at boundary point", kind: "expression" },
      { name: "y0", detail: "boundary value", kind: "expression" },
    ],
  },
  Integrate: {
    hintParams: ["expr", "x"],
    positional: [
      { name: "expr", detail: "integrand expression", kind: "expression" },
      { name: "x", detail: "integration variable", kind: "expression" },
    ],
  },
  Integral: {
    hintParams: ["expr", "x"],
    positional: [
      { name: "expr", detail: "formal integrand expression", kind: "expression" },
      { name: "x", detail: "integration variable", kind: "expression" },
    ],
  },
  classify: {
    hintParams: [],
    positional: [],
  },
  solve: {
    hintParams: ["details=false"],
    positional: [],
    kwargs: [{ name: "details", detail: "show solving steps and final solution", values: BOOLEAN_VALUES }],
  },
  Sum: {
    hintParams: ["expr", "a"],
    positional: [
      { name: "expr", detail: "summand expression", kind: "expression" },
      { name: "a", detail: "set index", kind: "expression" },
    ],
  },
  Det: { hintParams: ["A"], positional: [{ name: "A", detail: "tensor expression", kind: "expression" }] },
  Tr: { hintParams: ["A"], positional: [{ name: "A", detail: "tensor expression", kind: "expression" }] },
  Inv: { hintParams: ["A"], positional: [{ name: "A", detail: "tensor expression", kind: "expression" }] },
  log: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  sqrt: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  exp: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  sin: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  cos: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  tan: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  sinh: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  cosh: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  tanh: { hintParams: ["x"], positional: [{ name: "x", detail: "scalar expression", kind: "expression" }] },
  Spec_Decomp: {
    hintParams: ["C"],
    positional: [{ name: "C", detail: "component-filled tensor", kind: "expression" }],
  },
  Spectral: {
    hintParams: ["C", "lambda", "N"],
    positional: [
      { name: "C", detail: "symmetric tensor", kind: "expression" },
      { name: "lambda", detail: "eigenvalue LaTeX label", suggestions: [{ label: '"\\lambda"', detail: "label example" }] },
      { name: "N", detail: "eigenvector LaTeX label", suggestions: [{ label: '"\\bm N"', detail: "label example" }] },
    ],
  },
};

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

function cmOption(label, { type = "keyword", detail = "", apply = null, boost = 0, section = null } = {}) {
  const option = { label, type };
  if (detail) option.detail = detail;
  if (apply) option.apply = apply;
  if (boost) option.boost = boost;
  if (section) option.section = section;
  return option;
}

function wordBefore(context) {
  return context.matchBefore(/[A-Za-z_]\w*/);
}

function linePrefix(context) {
  const line = context.state.doc.lineAt(context.pos);
  return context.state.doc.sliceString(line.from, context.pos);
}

function scanStructure(text) {
  const stack = [];
  let quote = null;
  let escape = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === "(" || ch === "[") {
      stack.push({ ch, pos: i });
    } else if (ch === ")" || ch === "]") {
      const open = stack[stack.length - 1];
      if (!open) return { stack, inString: false, invalid: true };
      if ((open.ch === "(" && ch === ")") || (open.ch === "[" && ch === "]")) {
        stack.pop();
      } else {
        return { stack, inString: false, invalid: true };
      }
    }
  }
  return { stack, inString: quote != null, invalid: false };
}

function splitTopLevelArgs(text) {
  const out = [];
  let depth = 0;
  let quote = null;
  let escape = false;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") depth = Math.max(0, depth - 1);
    else if (ch === "," && depth === 0) {
      out.push({ from: start, to: i, text: text.slice(start, i) });
      start = i + 1;
    }
  }
  out.push({ from: start, to: text.length, text: text.slice(start) });
  return out;
}

function topLevelEqualIndex(text) {
  let depth = 0;
  let quote = null;
  let escape = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (escape) escape = false;
      else if (ch === "\\") escape = true;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === "(" || ch === "[") depth++;
    else if (ch === ")" || ch === "]") depth = Math.max(0, depth - 1);
    else if (ch === "=" && depth === 0) return i;
  }
  return -1;
}

function trailingWord(text) {
  return text.match(/[A-Za-z_]\w*$/);
}

function trailingValueToken(text) {
  return text.match(/[A-Za-z0-9_]*$/);
}

function completionContext(prefix) {
  const scan = scanStructure(prefix);
  if (scan.inString) return { inString: true };
  if (scan.invalid) return null;

  const open = scan.stack[scan.stack.length - 1];
  if (!open || open.ch !== "(") return null;

  const beforeOpen = prefix.slice(0, open.pos);
  const head = beforeOpen.match(/([A-Za-z_]\w*)\s*$/);
  if (!head) return null;
  const beforeHead = beforeOpen.slice(0, head.index);
  const receiver = beforeHead.endsWith(".") ? beforeHead.slice(0, -1).trim() : null;

  const argsText = prefix.slice(open.pos + 1);
  const args = splitTopLevelArgs(argsText);
  const current = args[args.length - 1] ?? { from: 0, text: "" };
  const currentStartRel = open.pos + 1 + current.from;
  const equal = topLevelEqualIndex(current.text);
  if (equal >= 0) {
    const rawKey = current.text.slice(0, equal).trim();
    const afterEqualRel = currentStartRel + equal + 1;
    const afterEqual = prefix.slice(afterEqualRel);
    const leadingSpaces = afterEqual.match(/^\s*/)?.[0].length ?? 0;
    const valueStartRel = afterEqualRel + leadingSpaces;
    const valueText = prefix.slice(valueStartRel);
    const valueToken = trailingValueToken(valueText);
    return {
      callName: head[1],
      receiver,
      argIndex: args.length - 1,
      args,
      currentArg: current.text,
      currentArgStartRel: currentStartRel,
      keyword: /^[A-Za-z_]\w*$/.test(rawKey) ? rawKey : null,
      valueMode: true,
      typed: valueToken?.[0] ?? "",
      replaceFromRel: valueStartRel + (valueToken?.index ?? valueText.length),
    };
  }

  const token = trailingWord(current.text);
  const leadingSpaces = current.text.match(/^\s*/)?.[0].length ?? 0;
  return {
    callName: head[1],
    receiver,
    argIndex: args.length - 1,
    args,
    currentArg: current.text,
    currentArgStartRel: currentStartRel,
    keyword: null,
    valueMode: false,
    typed: token?.[0] ?? "",
    replaceFromRel: token ? currentStartRel + token.index : currentStartRel + leadingSpaces,
  };
}

function schemaFor(name) {
  if (name === "show") {
    return {
      hintParams: ALL_SHOW_MODES,
      positional: [{ name: "mode", detail: "display mode", values: ALL_SHOW_MODES }],
    };
  }
  return BUILTIN_SCHEMAS[name] ?? null;
}

function kwargFor(schema, name) {
  return schema?.kwargs?.find((kwarg) => kwarg.name === name) ?? null;
}

function positionalFor(schema, argIndex) {
  const positional = schema?.positional ?? [];
  return positional[argIndex] ?? positional.find((param) => param.variadic) ?? null;
}

function topLevelBuiltinOptions() {
  return BUILTINS.map((builtin) =>
    cmOption(builtin.name, {
      type: "function",
      detail: builtin.sig,
      section: "builtins",
    }),
  );
}

function expressionBuiltinOptions() {
  return EXPRESSION_BUILTINS.map((name) => {
    const builtin = BUILTIN_BY_NAME.get(name);
    return cmOption(name, {
      type: "function",
      detail: builtin?.sig ?? "",
      section: "builtins",
    });
  });
}

// Walk up from the cursor to the start of the enclosing tens block, tracking
// the last `ViewSource on|off` directive. Returns "off" when a source region is
// currently open above the cursor (so the natural next directive is off), else
// "on". Lets completion suggest the one that actually makes sense here.
function suggestedViewSourceValue(context) {
  const doc = context.state.doc;
  const cursorLineNo = doc.lineAt(context.pos).number;
  let open = false;
  for (let n = cursorLineNo - 1; n >= 1; n--) {
    const text = doc.line(n).text;
    if (/<!--\s*\/?tensorforge:tens\s*-->/i.test(text)) break; // top of this block
    const directive = text.match(/^\s*ViewSource\s+(on|off)\s*$/i);
    if (directive) {
      open = directive[1].toLowerCase() === "on";
      break;
    }
  }
  return open ? "off" : "on";
}

function viewSourceDirectiveCompletion(context, prefix) {
  const suggested = suggestedViewSourceValue(context);
  const orderedValues = suggested === "on" ? ["on", "off"] : ["off", "on"];

  const valueMatch = prefix.match(/^(\s*ViewSource\s+)([A-Za-z]*)$/i);
  if (valueMatch) {
    const typed = valueMatch[2].toLowerCase();
    const options = orderedValues
      .filter((value) => value.startsWith(typed))
      .map((value) =>
        cmOption(value, {
          type: "constant",
          detail: value === suggested ? "ViewSource mode · suggested" : "ViewSource mode",
          section: "notebook",
          boost: value === suggested ? 6 : 2,
        }),
      );
    if (options.length === 0) return null;
    return {
      from: context.pos - typed.length,
      options,
      validFor: /^[A-Za-z]*$/,
    };
  }

  const nameMatch = prefix.match(/^(\s*)([A-Za-z_]*)$/);
  if (!nameMatch) return null;
  const typed = nameMatch[2];
  if (!typed || !"viewsource".startsWith(typed.toLowerCase())) return null;
  const label = `ViewSource ${suggested}`;
  return {
    from: context.pos - typed.length,
    options: [
      cmOption(label, {
        type: "keyword",
        detail:
          suggested === "on"
            ? "show this tens region beside its output"
            : "stop showing source beside output",
        apply: label,
        section: "notebook",
        boost: 4,
      }),
    ],
    validFor: /^[A-Za-z_]*$/,
  };
}

function keywordOptions(schema, typed) {
  const used = new Set();
  for (const arg of schema._args ?? []) {
    const equal = topLevelEqualIndex(arg.text);
    if (equal <= 0) continue;
    const key = arg.text.slice(0, equal).trim();
    if (/^[A-Za-z_]\w*$/.test(key)) used.add(key);
  }
  return (schema.kwargs ?? [])
    .filter((kwarg) => !used.has(kwarg.name))
    .filter((kwarg) => kwarg.name.startsWith(typed))
    .map((kwarg) => cmOption(kwarg.name, { type: "property", detail: kwarg.detail, apply: `${kwarg.name}=`, boost: 1 }));
}

function valueOptions(kwarg, typed = "") {
  if (!kwarg?.values) return [];
  return kwarg.values
    .filter((value) => value.startsWith(typed))
    .map((value) => cmOption(value, { type: "constant", detail: kwarg.detail }));
}

function positionalOptions(param, typed) {
  if (!param?.suggestions) return [];
  if (typed && !typed.startsWith('"') && !typed.startsWith("'")) return [];
  return param.suggestions
    .filter((suggestion) => suggestion.label.startsWith(typed))
    .map((suggestion) =>
      cmOption(suggestion.label, {
        type: "text",
        detail: suggestion.detail ?? param.detail,
        apply: suggestion.apply ?? suggestion.label,
        boost: 2,
      }),
    );
}

function completionResult(context, prefix, replaceFromRel, options, validFor = /^[A-Za-z_]\w*$/) {
  if (options.length === 0) return null;
  return {
    from: context.pos - (prefix.length - replaceFromRel),
    options,
    validFor,
  };
}

function dotCompletion(prefix) {
  const match = prefix.match(/\.([A-Za-z_]*)$/);
  if (!match) return null;
  const beforeDot = prefix.slice(0, match.index).trim();
  if (!beforeDot) return null;
  return {
    typed: match[1],
    receiver: beforeDot,
    replaceFromRel: prefix.length - match[1].length,
  };
}

function stripOuterParens(expr) {
  let text = expr.trim();
  let changed = true;
  while (changed && text.startsWith("(") && text.endsWith(")")) {
    changed = false;
    const scan = scanStructure(text.slice(0, -1));
    const open = scan.stack[0];
    if (!scan.invalid && open?.ch === "(" && open.pos === 0 && scan.stack.length === 1) {
      text = text.slice(1, -1).trim();
      changed = true;
    }
  }
  return text;
}

function parseNumberKwarg(callText, key) {
  const match = callText.match(new RegExp(`(?:^|,)\\s*${key}\\s*=\\s*(\\d+)`));
  return match ? Number.parseInt(match[1], 10) : null;
}

function directCallName(expr) {
  const text = stripOuterParens(expr);
  const match = text.match(/^([A-Za-z_]\w*)\s*\(/);
  if (!match) return null;
  const scan = scanStructure(text);
  return scan.invalid || scan.inString ? null : match[1];
}

function directMethodName(expr) {
  const text = stripOuterParens(expr);
  const match = text.match(/\.([A-Za-z_]\w*)\s*\(/);
  if (!match) return null;
  const scan = scanStructure(text);
  return scan.invalid || scan.inString ? null : match[1];
}

function parseBooleanKwarg(callText, key) {
  const match = callText.match(new RegExp(`(?:^|,)\\s*${key}\\s*=\\s*(true|false)`));
  return match ? match[1] === "true" : null;
}

function tensorKind(order = null, dim = null, flags = {}) {
  return {
    kind: "tensor",
    order: Number.isFinite(order) ? order : null,
    dim: Number.isFinite(dim) ? dim : null,
    ...flags,
  };
}

function inferDiffKind(expr, symbols) {
  const open = expr.indexOf("(");
  if (open < 0 || !expr.endsWith(")")) return { kind: "scalar" };
  const args = splitTopLevelArgs(expr.slice(open + 1, -1));
  const num = inferExprKind(args[0]?.text ?? "", symbols);
  const den = inferExprKind(args[1]?.text ?? "", symbols);
  if (den?.kind === "tensor") {
    const numOrder = num?.kind === "tensor" && Number.isFinite(num.order) ? num.order : 0;
    const denOrder = Number.isFinite(den.order) ? den.order : 0;
    return tensorKind(numOrder + denOrder);
  }
  return { kind: "scalar" };
}

function inferExprKind(expr, symbols) {
  const text = stripOuterParens(expr);
  if (!text) return null;
  if (/^[A-Za-z_]\w*$/.test(text)) return symbols.get(text) ?? null;
  if (/^[A-Za-z_]\w*\s*\[[^\]]+\]$/.test(text)) return { kind: "scalar" };

  const method = directMethodName(text);
  if (method === "solve") return { kind: "ode_solution" };
  if (method === "classify") return { kind: "ode_classification" };

  const call = directCallName(text);
  if (call) {
    if (call === "Tensor") {
      return tensorKind(parseNumberKwarg(text, "order") ?? 2, parseNumberKwarg(text, "dim") ?? null, {
        identity: parseBooleanKwarg(text, "identity"),
        symmetric: parseBooleanKwarg(text, "symmetric"),
        antisymmetric: parseBooleanKwarg(text, "antisymmetric"),
        orthogonal: parseBooleanKwarg(text, "orthogonal"),
        isotropic: parseBooleanKwarg(text, "isotropic"),
      });
    }
    if (call === "Scalar") return { kind: "scalar" };
    if (call === "Var") return { kind: "var" };
    if (call === "Function") return { kind: "function" };
    if (call === "ScalarSet") return { kind: "scalar_set" };
    if (call === "VectorSet") return { kind: "vector_set" };
    if (call === "Equation") return { kind: "equation" };
    if (call === "ODE") return { kind: "ode_problem" };
    if (call === "BoundaryCondition") return { kind: "boundary_condition" };
    if (call === "Derivative" || call === "Det" || call === "Tr" || call === "Integrate" || call === "Integral") {
      return { kind: "scalar" };
    }
    if (call === "Diff") return inferDiffKind(text, symbols);
    if (["log", "sqrt", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh"].includes(call)) {
      return { kind: "scalar" };
    }
  }

  return null;
}

function symbolKindsFromSource(sourceText, pos) {
  const symbols = new Map();
  const text = sourceText.slice(0, Math.max(0, pos));
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("<!--")) continue;

    const destructured = trimmed.match(/^\[\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\]\s*=\s*(Spec_Decomp|Spectral)\s*\(/);
    if (destructured) {
      symbols.set(destructured[1], { kind: "scalar_set" });
      symbols.set(destructured[2], { kind: "vector_set" });
      continue;
    }

    const assigned = trimmed.match(/^([A-Za-z_]\w*)\s*=\s*(.+)$/);
    if (!assigned) continue;
    const inferred = inferExprKind(assigned[2], symbols);
    if (inferred) symbols.set(assigned[1], inferred);
  }
  return symbols;
}

function symbolTypeLabel(info) {
  switch (info?.kind) {
    case "var":
      return "var";
    case "function":
      return "function";
    case "equation":
      return "equation";
    case "ode_problem":
      return "ODE problem";
    case "boundary_condition":
      return "boundary condition";
    case "ode_solution":
      return "ODE solution";
    case "ode_classification":
      return "ODE classification";
    case "scalar_set":
      return "scalar set";
    case "vector_set":
      return "vector set";
    case "tensor":
      return "tensor";
    case "scalar":
      return "scalar";
    default:
      return "symbol";
  }
}

function symbolDetail(info) {
  if (!info) return "symbol";
  if (info.kind !== "tensor") return symbolTypeLabel(info);
  const parts = ["tensor"];
  if (Number.isFinite(info.dim) && Number.isFinite(info.order)) {
    if (info.order === 1) parts.push(`dim ${info.dim}`);
    else if (info.order === 2) parts.push(`${info.dim}x${info.dim}`);
    else parts.push(`order ${info.order} · dim ${info.dim}`);
  } else if (Number.isFinite(info.order)) {
    parts.push(`order ${info.order}`);
  }
  for (const flag of ["identity", "symmetric", "antisymmetric", "orthogonal", "isotropic"]) {
    if (info[flag] === true) parts.push(flag);
  }
  return parts.join(" · ");
}

function symbolOptions(sourceText, pos, typed = "") {
  const symbols = symbolKindsFromSource(sourceText, pos);
  return [...symbols.entries()]
    .filter(([name]) => name.startsWith(typed))
    .map(([name, info]) =>
      cmOption(name, {
        type: "variable",
        detail: symbolDetail(info),
        section: "your symbols",
        boost: 8,
      }),
    );
}

function modesForKind(kind) {
  if (!kind) return SAFE_SHOW_MODES;
  switch (kind.kind) {
    case "tensor":
      if (kind.order === 1 || kind.order === 2) return ["symbol", "components", "matrix"];
      if (kind.order === 4) return ["symbol", "block_components"];
      return SAFE_SHOW_MODES;
    case "ode_classification":
      return ["symbol", "details"];
    case "ode_solution":
      return ["symbol", "solution", "steps"];
    case "ode_problem":
      return ["symbol"];
    case "equation":
    case "boundary_condition":
    case "var":
    case "scalar":
    case "function":
    case "scalar_set":
    case "vector_set":
    default:
      return SAFE_SHOW_MODES;
  }
}

function showModeCandidates(receiver, sourceText, pos) {
  const symbols = symbolKindsFromSource(sourceText, pos);
  const kind = inferExprKind(receiver ?? "", symbols);
  return modesForKind(kind);
}

function showModeOptions(modes, typed) {
  return modes
    .filter((mode) => mode.startsWith(typed))
    .map((mode) => cmOption(mode, { type: "property", detail: SHOW_MODE_DETAILS[mode] ?? "display mode" }));
}

function contextualCompletion(context, prefix, callContext) {
  const schema = schemaFor(callContext.callName);
  if (!schema) return null;

  if (callContext.callName === "show") {
    const options = showModeOptions(
      showModeCandidates(callContext.receiver, context.state.doc.toString(), context.pos),
      callContext.typed,
    );
    return {
      from: context.pos - (prefix.length - callContext.replaceFromRel),
      options,
      validFor: /^[A-Za-z_]\w*$/,
    };
  }

  if (callContext.valueMode) {
    const kwarg = kwargFor(schema, callContext.keyword);
    if (!kwarg) return null;
    const options = kwarg.values ? valueOptions(kwarg, callContext.typed) : expressionBuiltinOptions();
    return completionResult(context, prefix, callContext.replaceFromRel, options, /^[A-Za-z0-9_]*$/);
  }

  schema._args = callContext.args;
  const param = positionalFor(schema, callContext.argIndex);
  const options = [];
  if (param?.kind === "expression") {
    options.push(...symbolOptions(context.state.doc.toString(), context.pos, callContext.typed));
    options.push(...expressionBuiltinOptions());
  } else {
    options.push(...positionalOptions(param, callContext.typed));
  }
  options.push(...keywordOptions(schema, callContext.typed));
  delete schema._args;
  return completionResult(context, prefix, callContext.replaceFromRel, options);
}

function paramDetail(param, kwarg) {
  const item = kwarg ?? param;
  if (!item) return "";
  const values = item.values?.join(" | ");
  return values ? `${item.name} · ${item.detail} · ${values}` : `${item.name} · ${item.detail}`;
}

// Given the line text up to the caret, return signature data for the enclosing
// known call, or null when the caret is not inside a known call.
export function signatureHint(prefix, sourceText = prefix, pos = sourceText.length) {
  const ctx = completionContext(prefix);
  if (!ctx || ctx.inString) return null;
  const schema = schemaFor(ctx.callName);
  if (!schema) return null;

  const params = schema.hintParams ?? schema.positional?.map((param) => param.name) ?? [];
  if (params.length === 0) return null;

  if (ctx.callName === "show") {
    const modes = showModeCandidates(ctx.receiver, sourceText, pos);
    return {
      fn: "show",
      params: modes,
      argIndex: 0,
      choice: true,
      detail: "mode · display mode",
    };
  }

  let activeParam = positionalFor(schema, ctx.argIndex);
  let activeIndex = Math.min(ctx.argIndex, params.length - 1);
  let activeKwarg = null;
  if (ctx.valueMode && ctx.keyword) {
    activeKwarg = kwargFor(schema, ctx.keyword);
    activeParam = null;
    const keywordIndex = params.findIndex((param) => param === ctx.keyword || param.startsWith(`${ctx.keyword}=`));
    if (keywordIndex >= 0) activeIndex = keywordIndex;
  } else if (ctx.typed && schema.kwargs?.length) {
    const matches = schema.kwargs.filter((kwarg) => kwarg.name.startsWith(ctx.typed));
    if (matches.length === 1) {
      activeKwarg = matches[0];
      activeParam = null;
      const keywordIndex = params.findIndex((param) => param === activeKwarg.name || param.startsWith(`${activeKwarg.name}=`));
      if (keywordIndex >= 0) activeIndex = keywordIndex;
    } else if (matches.length > 1) {
      activeParam = { name: "keyword", detail: matches.map((kwarg) => kwarg.name).join(" | ") };
    }
  }

  return {
    fn: ctx.callName,
    params,
    argIndex: activeIndex,
    choice: false,
    detail: paramDetail(activeParam, activeKwarg),
  };
}

// ---- completion sources ----------------------------------------------------

export function tensorForgeCompletionSource(context) {
  const prefix = linePrefix(context);
  const structure = scanStructure(prefix);
  if (structure.inString) return null;

  const directive = viewSourceDirectiveCompletion(context, prefix);
  if (directive) return directive;

  const dot = dotCompletion(prefix);
  if (dot) {
    const kind = inferExprKind(dot.receiver, symbolKindsFromSource(context.state.doc.toString(), context.pos));
    const options = [];
    options.push(cmOption("show", { type: "method", detail: ".show(mode)" }));
    if (kind?.kind === "ode_problem") {
      options.push(
        cmOption("classify", { type: "method", detail: ".classify()" }),
        cmOption("solve", { type: "method", detail: ".solve(details=true)" }),
      );
    }
    if (kind?.kind === "tensor") {
      options.push(cmOption("T", { type: "property", detail: "transpose" }));
    }
    const filtered = options.filter((option) => option.label.startsWith(dot.typed));
    return completionResult(context, prefix, dot.replaceFromRel, filtered);
  }

  const callContext = completionContext(prefix);
  if (callContext && !callContext.inString) {
    const result = contextualCompletion(context, prefix, callContext);
    if (result) return result;
  }

  const word = wordBefore(context);
  if (!word && !context.explicit) return null;
  const from = word?.from ?? context.pos;
  const typed = word ? context.state.doc.sliceString(word.from, context.pos) : "";
  return {
    from,
    options: [
      ...symbolOptions(context.state.doc.toString(), context.pos, typed),
      ...topLevelBuiltinOptions(),
    ],
    validFor: /^[A-Za-z_]\w*$/,
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
