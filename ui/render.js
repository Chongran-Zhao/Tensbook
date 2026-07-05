// Presentational output widgets: plot SVGs, ODE badge/step chrome, and
// small text/DOM helpers shared by the live preview and the exporter.
// No app state — everything renders from the data it is handed.

export const KATEX_MACROS = { "\\bm": "\\boldsymbol{#1}" };

export function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function escapeAttr(text) {
  return escapeHtml(text).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function copyButton(label, text) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.textContent = label;
  btn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(text);
    btn.textContent = "copied";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = label;
      btn.classList.remove("copied");
    }, 1200);
  });
  return btn;
}

// ---- Plot preview ----------------------------------------------------------

let nextPlotClipId = 1;

function svgNode(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value != null) node.setAttribute(key, String(value));
  }
  return node;
}

export function formatPlotTick(value) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) < 1e-10) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.01) return value.toExponential(1).replace("+", "");
  return Number(value.toPrecision(3)).toString();
}

export function plotTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [];
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

export function plotPath(segment, scaleX, scaleY) {
  return segment
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`)
    .join(" ");
}

function renderPlotLegend(detail) {
  if (!detail.series || detail.series.length < 2) return null;
  const legend = document.createElement("div");
  legend.className = "tf-plot-legend";
  detail.series.forEach((series, index) => {
    const item = document.createElement("span");
    item.className = "tf-plot-legend-item";
    const swatch = document.createElement("span");
    swatch.className = `tf-plot-legend-swatch tf-plot-line-${index % 6}`;
    const label = document.createElement("span");
    label.className = "tf-plot-legend-label";
    try {
      katex.render(series.label_latex, label, {
        displayMode: false,
        macros: KATEX_MACROS,
        throwOnError: true,
      });
    } catch (e) {
      label.textContent = series.label_latex;
    }
    item.append(swatch, label);
    legend.appendChild(item);
  });
  return legend;
}

export function renderPlot(detail) {
  const width = 720;
  const height = 260;
  const margin = { top: 18, right: 18, bottom: 34, left: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const [xMin, xMax] = detail.x_range;
  const [yMin, yMax] = detail.y_range;
  const scaleX = (x) => margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const scaleY = (y) => margin.top + (1 - (y - yMin) / (yMax - yMin)) * plotHeight;
  const clipId = `tf-plot-clip-${nextPlotClipId++}`;

  const wrap = document.createElement("div");
  wrap.className = "plot";
  const svg = svgNode("svg", {
    class: "tf-plot-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `Plot over ${detail.x_label}`,
  });

  const defs = svgNode("defs");
  const clip = svgNode("clipPath", { id: clipId });
  clip.appendChild(svgNode("rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
  }));
  defs.appendChild(clip);
  svg.appendChild(defs);

  svg.appendChild(svgNode("rect", {
    class: "tf-plot-frame",
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
  }));

  for (const x of plotTicks(xMin, xMax)) {
    const px = scaleX(x);
    svg.appendChild(svgNode("line", {
      class: "tf-plot-grid",
      x1: px,
      x2: px,
      y1: margin.top,
      y2: margin.top + plotHeight,
    }));
    const label = svgNode("text", {
      class: "tf-plot-tick",
      x: px,
      y: height - 11,
      "text-anchor": "middle",
    });
    label.textContent = formatPlotTick(x);
    svg.appendChild(label);
  }

  for (const y of plotTicks(yMin, yMax)) {
    const py = scaleY(y);
    svg.appendChild(svgNode("line", {
      class: "tf-plot-grid",
      x1: margin.left,
      x2: margin.left + plotWidth,
      y1: py,
      y2: py,
    }));
    const label = svgNode("text", {
      class: "tf-plot-tick",
      x: margin.left - 8,
      y: py + 4,
      "text-anchor": "end",
    });
    label.textContent = formatPlotTick(y);
    svg.appendChild(label);
  }

  if (xMin < 0 && xMax > 0) {
    const px = scaleX(0);
    svg.appendChild(svgNode("line", {
      class: "tf-plot-axis",
      x1: px,
      x2: px,
      y1: margin.top,
      y2: margin.top + plotHeight,
    }));
  }
  if (yMin < 0 && yMax > 0) {
    const py = scaleY(0);
    svg.appendChild(svgNode("line", {
      class: "tf-plot-axis",
      x1: margin.left,
      x2: margin.left + plotWidth,
      y1: py,
      y2: py,
    }));
  }

  const curves = svgNode("g", { "clip-path": `url(#${clipId})` });
  detail.series.forEach((series, index) => {
    for (const segment of series.segments ?? []) {
      if (segment.length < 2) continue;
      curves.appendChild(svgNode("path", {
        class: `tf-plot-line tf-plot-line-${index % 6}`,
        d: plotPath(segment, scaleX, scaleY),
      }));
    }
  });
  svg.appendChild(curves);

  // Inside the frame's bottom-right corner so it cannot collide with the
  // rightmost tick label (both used to share the same baseline).
  const xLabel = svgNode("text", {
    class: "tf-plot-axis-label",
    x: margin.left + plotWidth - 8,
    y: margin.top + plotHeight - 9,
    "text-anchor": "end",
  });
  xLabel.textContent = detail.x_label;
  svg.appendChild(xLabel);

  wrap.appendChild(svg);
  const legend = renderPlotLegend(detail);
  if (legend) wrap.appendChild(legend);
  return wrap;
}

// ---- ODE result chrome (badges + numbered steps) --------------------------
// Driven by the structured `detail` payload on an output; the math inside each
// step still goes through KaTeX, everything around it is plain DOM.

function odeChip(text, { muted = false, strong = false } = {}) {
  const chip = document.createElement("span");
  chip.className = "ode-chip" + (muted ? " muted" : "") + (strong ? " is-default" : "");
  chip.textContent = text;
  return chip;
}

function odeChipMath(prefix, tex) {
  const chip = document.createElement("span");
  chip.className = "ode-chip";
  chip.appendChild(document.createTextNode(prefix));
  const math = document.createElement("span");
  try {
    katex.render(tex, math, { displayMode: false, macros: KATEX_MACROS, throwOnError: true });
  } catch (e) {
    math.textContent = tex;
  }
  chip.appendChild(math);
  return chip;
}

export function renderOdeDetail(detail) {
  if (detail.type === "ode_classification") {
    const wrap = document.createElement("div");
    wrap.className = "ode-badges";
    wrap.append(
      odeChip(detail.kind),
      odeChip(`order ${detail.order}`),
      odeChip(detail.linear ? "linear" : "nonlinear"),
      odeChip(detail.homogeneous ? "homogeneous" : "non-homogeneous"),
    );
    return wrap;
  }

  if (detail.type === "ode_boundary") {
    const wrap = document.createElement("div");
    wrap.className = "ode-badges";
    wrap.append(
      detail.boundary
        ? odeChipMath("", detail.boundary)
        : odeChip("no boundary condition", { muted: true }),
    );
    return wrap;
  }

  if (detail.type === "ode_methods") {
    const wrap = document.createElement("div");
    wrap.className = "ode-badges";
    if (!detail.available || detail.available.length === 0) {
      wrap.append(odeChip(detail.default || "no symbolic method", { muted: true }));
    } else {
      for (const method of detail.available) {
        wrap.append(odeChip(method, { strong: method === detail.default }));
      }
    }
    return wrap;
  }

  if (detail.type === "ode_steps") {
    const list = document.createElement("ol");
    list.className = "ode-steps";
    detail.steps.forEach((step, index) => {
      const item = document.createElement("li");
      item.className = "ode-step";
      const num = document.createElement("span");
      num.className = "ode-step-num";
      num.textContent = String(index + 1);
      const body = document.createElement("div");
      body.className = "ode-step-body";
      const label = document.createElement("div");
      label.className = "ode-step-label";
      label.textContent = step.label;
      const math = document.createElement("div");
      math.className = "ode-step-math";
      try {
        katex.render(step.latex, math, { displayMode: true, macros: KATEX_MACROS, throwOnError: true });
      } catch (e) {
        math.textContent = step.latex;
      }
      body.append(label, math);
      item.append(num, body);
      list.append(item);
    });
    return list;
  }

  const fallback = document.createElement("div");
  fallback.className = "math";
  return fallback;
}

export function stripDisplayMath(latex) {
  return latex.replace(/^\$\$\n?/, "").replace(/\n?\$\$$/, "");
}
