// Markdown-to-HTML rendering for notebook prose. This module is intentionally
// small and dependency-light: it only reaches for KaTeX when math is present.

import { KATEX_MACROS, escapeHtml, escapeAttr } from "./render.js";

export function isInnerFenceOpen(line) {
  const trimmed = line.trim();
  return trimmed.startsWith("```") && trimmed !== "```";
}

export function isBareFenceClose(line) {
  return /^```\s*$/.test(line.trim());
}

export function sourceLineAttr(line) {
  return Number.isFinite(line) ? ` data-source-line="${line}"` : "";
}

export function renderMarkdown(markdown, baseLine = null) {
  const lines = markdown.split(/\r?\n/);
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const sourceLine = baseLine == null ? null : baseLine + i;
    if (isInnerFenceOpen(line)) {
      const startLine = sourceLine;
      const code = [];
      i++;
      while (i < lines.length && !isBareFenceClose(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      out.push(`<pre class="md-code"${sourceLineAttr(startLine)}><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const oneLineMath = line.match(/^\s*\$\$\s*(.*?)\s*\$\$\s*$/);
    if (oneLineMath) {
      out.push(`<div class="markdown-math"${sourceLineAttr(sourceLine)}>${renderMath(oneLineMath[1], true)}</div>`);
      continue;
    }
    if (line.trim() === "$$") {
      const startLine = sourceLine;
      const tex = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") {
        tex.push(lines[i]);
        i++;
      }
      out.push(`<div class="markdown-math"${sourceLineAttr(startLine)}>${renderMath(tex.join("\n"), true)}</div>`);
      continue;
    }
    if (isTableRow(line) && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const startLine = sourceLine;
      const aligns = tableAlignments(lines[i + 1]);
      const header = tableCells(line);
      const rows = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i])) {
        rows.push(tableCells(lines[i]));
        i++;
      }
      i--;
      out.push(renderTable(header, rows, aligns, startLine));
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      out.push(`<h${level}${sourceLineAttr(sourceLine)}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
      out.push(`<hr${sourceLineAttr(sourceLine)}>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const startLine = sourceLine;
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const itemLine = baseLine == null ? null : baseLine + i;
        items.push(`<li${sourceLineAttr(itemLine)}>${inlineMarkdown(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`);
        i++;
      }
      i--;
      out.push(`<ul${sourceLineAttr(startLine)}>${items.join("")}</ul>`);
      continue;
    }
    if (line.trim() === "") continue;
    out.push(`<p${sourceLineAttr(sourceLine)}>${inlineMarkdown(line)}</p>`);
  }
  return out.join("") || "<p></p>";
}

export function isTableRow(line) {
  const t = line.trim();
  return t.startsWith("|") && t.endsWith("|") && t.length > 1;
}

export function isTableSeparator(line) {
  const t = line.trim();
  if (!isTableRow(t)) return false;
  return tableCells(t).every((c) => /^:?-{3,}:?$/.test(c.trim()) || c.trim() === "");
}

export function tableCells(line) {
  const t = line.trim();
  return t.slice(1, t.length - 1).split("|");
}

export function tableAlignments(sepLine) {
  return tableCells(sepLine).map((c) => {
    const t = c.trim();
    if (t.startsWith(":") && t.endsWith(":")) return "center";
    if (t.endsWith(":")) return "right";
    return "left";
  });
}

export function renderTable(header, rows, aligns, sourceLine = null) {
  const cell = (tag, content, k) =>
    `<${tag} style="text-align:${aligns[k] ?? "left"}">${inlineMarkdown(content.trim())}</${tag}>`;
  return `<table class="md-table"${sourceLineAttr(sourceLine)}><thead><tr>${header
    .map((c, k) => cell("th", c, k))
    .join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c, k) => cell("td", c, k)).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

export function inlineMarkdown(text) {
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

export function findInlineMathEnd(text, start) {
  for (let i = start; i < text.length; i++) {
    if (text[i] === "$" && text[i - 1] !== "\\") return i;
  }
  return -1;
}

export function renderInlinePlain(text) {
  const parts = [];
  let rest = text;
  // Match both asset types in one pass so their source order is preserved.
  const asset =
    /!\[([^\]]*)\]\((data:image\/[a-zA-Z+.-]+;base64,[A-Za-z0-9+/=]+)\)|\[([^\]]+)\]\(([^)\s]+)\)/;
  for (let m = rest.match(asset); m; m = rest.match(asset)) {
    parts.push(inlineStyles(rest.slice(0, m.index)));
    if (m[2] !== undefined) {
      parts.push(`<img class="md-img" alt="${escapeAttr(m[1])}" src="${m[2]}">`);
    } else {
      const href = safeMarkdownHref(m[4]);
      if (href) {
        parts.push(`<a href="${escapeAttr(href)}">${inlineStyles(m[3])}</a>`);
      } else {
        parts.push(inlineStyles(m[0]));
      }
    }
    rest = rest.slice(m.index + m[0].length);
  }
  parts.push(inlineStyles(rest));
  return parts.join("");
}

export function safeMarkdownHref(raw) {
  try {
    const url = new URL(raw);
    return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

export function inlineStyles(text) {
  return escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

export function renderMath(tex, displayMode) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      macros: KATEX_MACROS,
      throwOnError: true,
    });
  } catch {
    return `<code>${escapeHtml(tex)}</code>`;
  }
}
