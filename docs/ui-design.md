# TensorForge UI — design & code guide

A reference for the desktop app's front-end: how it's structured, themed, and
wired. The UI is a single WebView (macOS WKWebView) hosted by Tauri; all of it
lives under `ui/`. The Rust side only exposes a few IPC commands and renders the
window — it owns no UI.

---

## 1. Architecture at a glance

```
┌──────────── Tauri (src-tauri) ─────────────┐
│  window + IPC commands:                     │
│  run_tens · open_tens · read_tens ·         │
│  list_folder · save_tens · export_text ·    │
│  print_window                               │
└───────────────────┬─────────────────────────┘
                    │ window.__TAURI__.core.invoke
┌───────────────────▼──────────  ui/  ────────┐
│  index.html  → shell, CSS tokens, DOM        │
│  main.js     → all behaviour (ES module)     │
│  completion.js → DSL builtin list            │
│  codemirror-entry.js → CM re-exports         │
│  vendor/katex/  vendor/codemirror/cm.bundle  │
└──────────────────────────────────────────────┘
```

Two panes side by side: a **CodeMirror 6 source editor** (left) and a **rendered
preview** (right). The document is a single `.tens` text buffer — Markdown by
default, with executable code fenced by hidden sentinels. Editing the buffer
debounces a "live run" that re-renders the preview in document order.

### Data flow

```
type → CodeMirror doc change → handleEditorUpdate → scheduleLiveRun (debounced)
     → invoke("run_tens", source) → outputs[]  → renderOutputs() → preview DOM
```

The editor is the single source of truth; `parseSourceDocument()` derives a block
model only for **preview mapping and scroll-sync**, not for editing.

---

## 2. File map

| File | Lines | Responsibility |
|------|------:|----------------|
| `ui/index.html` | ~666 | Shell: theme tokens (CSS variables), static chrome CSS (header, rail, footer, menus, about, print), DOM, loads KaTeX + `main.js`. |
| `ui/main.js` | ~2100 | Everything dynamic: theme, document model, CodeMirror setup, decorations, Markdown preview, run pipeline, output layout, scroll-sync, files, layout resizers, export/print, boot. |
| `ui/completion.js` | ~400 | Static builtin schemas for context-aware Tens autocomplete, Markdown slash commands, and signature hints. |
| `ui/codemirror-entry.js` | ~40 | Re-exports the exact CodeMirror 6 symbols used; the bundler's entry point. |
| `scripts/bundle-codemirror.mjs` | — | `esbuild-wasm` bundles the entry into `ui/vendor/codemirror/cm.bundle.js` (ESM, content-hashed to avoid churn). |
| `ui/vendor/katex/`, `ui/vendor/codemirror/cm.bundle.js` | — | Vendored libraries, loaded locally (CSP-safe). The CM bundle is gitignored and regenerated. |

`main.js` is organised by `// ---- section ----` banners: theme · about dialog ·
document model · CodeMirror source editor · files · layout · markdown rendering ·
output/export · markdown templates and paste · scroll linking · menus/events ·
boot.

---

## 3. Build & bundling

- The frontend has a **build step only for CodeMirror**. `package.json` declares
  the CM6 packages + `esbuild-wasm`; `npm run bundle:cm` produces
  `ui/vendor/codemirror/cm.bundle.js`.
- Tauri runs it automatically: `tauri.conf.json` →
  `build.beforeDevCommand` / `beforeBuildCommand` = `npm run bundle:cm`.
- `main.js` imports the bundle as a local ES module; **nothing loads from a CDN**
  (see §12 CSP). KaTeX is vendored the same way.

---

## 4. Design system — theme tokens

All colour/typography decisions resolve to CSS custom properties declared in
`index.html`. There are two themes; **dark (warm mocha) is the `:root` default**,
**light (warm paper)** overrides under `html[data-theme="light"]`.

| Token | Role | Light (paper) | Dark (mocha) |
|-------|------|---------------|--------------|
| `--bg` | page | `#f3f1ea` | `#2a2521` |
| `--panel` | pane surface | `#fffdf8` | `#332e28` |
| `--panel-2` | inset surface | `#ece8dd` | `#2c2722` |
| `--border` | hairline | `#e0dccf` | `#464037` |
| `--text` | body | `#2b2924` | `#ece6da` |
| `--muted` | secondary/hints | `#8c887d` | `#a59c8c` |
| `--accent` | links, Run, focus | `#355e8a` | `#93b6df` |
| `--accent-text` | text on accent | `#ffffff` | `#2a2521` |
| `--tens-frame` | tens-block accent | `#355e8a` | `#93b6df` |
| `--note-accent` | markdown accent bar | `#9a7b3a` | `#d9b96e` |
| `--error` | errors | `#b0432e` | `#e8867a` |
| `--serif` | wordmark/headings | `ui-serif, Georgia, …` | same |
| `--syntax-{comment,string,number,keyword,function,math,strong}` | code colours | warm/ink set | warm/gold set |

Conventions:
- **Never hardcode hex in JS or CSS** — read a token. The CodeMirror theme and
  decorations use these vars so the editor flips with the app.
- Fonts: serif (`--serif`) for the wordmark, footer, Markdown headings, About;
  monospace (`"SF Mono", Menlo, …`) for the whole editor and inline code; the rest
  is the system sans stack.
- One accent (`--accent`, slate-blue) carries links, the Run button, focus rings,
  and the tens accent.

The slate-blue + warm-paper palette is the app's identity; the matching app icon
(`F = [⋱]`) uses the same `--tens-frame` slate.

---

## 5. DOM layout

```
#app  (flex column, full height)
├── header        wordmark · Open · New · Save · Add Tens · Markdown▾ · Export▾ ·
│                 filename · ⌘⏎ hint · ◐ theme · ⓘ about · Run
├── main  (flex row, flex:1)
│   ├── #file-rail        collapsible folder listing (+ #rail-toggle)
│   ├── #rail-resizer     drag to resize the rail
│   ├── #editor-wrap      ← CodeMirror EditorView mounts here
│   ├── #split-resizer    drag to resize editor vs preview
│   └── #output           rendered preview
├── footer        version · tagline · © author
#about-overlay     modal (author / affiliation / links)
#print-root        offscreen container filled for Print/PDF
```

The two resizers (`#rail-resizer`, `#split-resizer`) adjust flex bases; widths
persist in `localStorage`. Chrome (header/rail/footer/menus/about/print) is styled
by the static `<style>` in `index.html`; the editor is themed in JS (§6).

---

## 6. The editor — CodeMirror 6

Mounted by `initEditor()` into `#editor-wrap`. The CM document **is** the `.tens`
source string; `documentSource()` / `executableSource()` read it back for
save/run.

### Extensions (in `initEditor`)
- `lineNumbers({ formatNumber })` — numbers are remapped by
  `visibleMarkdownLineNumber()` so hidden sentinel lines get no number.
- `history()`, `drawSelection()`, `highlightActiveLine()`/`Gutter()`.
- `markdown()` + `syntaxHighlighting(markdownHighlight)` — Markdown highlighting
  (the `markdownHighlight` `HighlightStyle` maps headings/strong/emphasis/links to
  `--accent` / `--syntax-*`).
- `EditorView.lineWrapping` — soft wrap (wrap-aware gutter is native to CM).
- `EditorView.decorations.compute([...], buildSentinelHiding)` — hides the
  `<!-- tensorforge:tens -->` markers (see below).
- `editorTheme` (`EditorView.theme`) — all editor surfaces/colours from the tokens.
- `sourceDecorations` (a `ViewPlugin`) — tens-region styling + DSL token colours.
- `autocompletion({ override:[completionSource] })`, `keymap.of([...])`
  (indent / completion / search / history / default).
- `EditorView.updateListener.of(handleEditorUpdate)` — on doc change schedules the
  live run, refreshes toolbar state, and updates the signature hint.
- `domEventHandlers.paste` — pasting an image into a Markdown region inserts a
  `![pasted image](data:…)` (read as a data URL).

### `.tens` document model
- Markdown is the default; code lives between `<!-- tensorforge:tens -->` and
  `<!-- /tensorforge:tens -->` sentinels (`TENS_OPEN` / `TENS_CLOSE`). On disk the
  format is unchanged and human-readable.
- Region helpers: `tensRegions()`, `regionKindAtPos()` / `regionKindAtLine()`,
  `isMarkdownAtPos()`, `isSentinelDocLine()`.
- `parseSourceDocument()` builds a block list (markdown vs tens, with source-line
  ranges) used **only** for preview ordering and scroll-sync; a legacy `​```notes`
  form still parses (`parseLegacyNoteDocument`).

### Sentinel hiding & tens styling
- `buildSentinelHiding(doc)` returns decorations that hide the sentinel lines so
  the user sees clean code regions, not raw comments.
- `sourceDecorations` walks each tens region and, per content line, applies the
  tens visual treatment and `decorateTensTokens()` (comment/string/number/builtin/
  keyword/operator marks driven by `TENS_BUILTINS` / `TENS_KEYWORDS` →
  `--syntax-*`).
- The invariant: every tens content line, even an empty one, must remain a real,
  editable `.cm-line`; do not hide editor lines with `display:none`.

### Add Tens
`insertTensBlock()` inserts `TENS_OPEN \n __TF_CURSOR__ \n TENS_CLOSE` at/after the
caret (`__TF_CURSOR__` marks where the caret lands). If the caret is already in a
tens region it appends a new block after it (`endOfTensBlockAfter`).

### Completion
`completionSource(context)` dispatches by region. In Tens code,
`completion.js` uses static builtin schemas to provide context-aware completion:
top-level builtins, function-specific keyword/value suggestions inside calls,
dot completion for `.show(...)` and `.T`, and a passive signature hint. In
Markdown prose, slash commands insert common Markdown snippets.

---

## 7. The preview / output pane (`#output`)

### Live run
`scheduleLiveRun()` debounces, then `invoke("run_tens", { source })`. Results are
an array of `{ header, latex, row, error, … }`. Without the Tauri backend (e.g. a
plain browser) the editor still works; only execution is unavailable.

### Document-order interleaving
`renderOutputs()`:
1. `orderedPreviewBlocks()` interleaves Markdown blocks and run outputs in source
   order (each carries `data-source-line` / `data-source-block-id`).
2. `groupOutputRows()` turns bracketed `[a.show(), b.show()]` calls into
   **side-by-side rows**.
3. Markdown blocks → `renderMarkdownBlock()` → `renderMarkdown()`; tens outputs →
   `renderOutputBlock()` (KaTeX display math + `copy latex` buttons).

### Markdown renderer (`renderMarkdown`)
Handles headings, paragraphs, lists, blockquotes, tables (`renderTable`), code
fences, horizontal rules, images (`data:` and scheme-checked `http(s)`), links
(`safeMarkdownHref` allows only `http/https/mailto`), inline code, and math:
`$$…$$` blocks and `$…$` inline via `renderMath()` (KaTeX, with the `\bm` macro).
`escapeHtml` / `escapeAttr` guard all interpolation.

### Export
- `buildExportMarkdown()` → download `.md` (`downloadText`).
- `buildPrintableBody()` fills `#print-root`; `printDocument()` triggers a native
  print (Print / Save PDF) — `window.print()` is a no-op in WKWebView, so a Tauri
  `print_window` command does the actual printing.

---

## 8. Scroll-sync & click-to-jump

Bidirectional, with a re-entrancy guard:
- `syncEditorToOutput()` / `syncOutputToEditor()` map an anchor line between the
  editor (`editorView.scrollDOM`) and the preview using `data-source-line` /
  block ids (`sourceAnchor` / `previewAnchor`).
- `suppressScrollSync()` / `isScrollSyncSuppressed()` prevent feedback loops
  during programmatic scrolls and re-renders.
- Clicking a preview block (`makeBlockJump`) scrolls the editor to that source
  line (`jumpToSourceLine` / `scrollEditorToLine`).

---

## 9. Toolbar actions

| Control | Handler |
|---------|---------|
| Open / New / Save | `openFile` (invoke `open_tens`) · `newFile` · `saveFile` (invoke `save_tens`) |
| Add Tens | `insertTensBlock` (§6) |
| Markdown ▾ | `#table-menu` popover → `insertMarkdownTemplate` / `insertTableIntoMarkdown` (headings, math, lists, quote, table, logic-tree, three-cases templates) |
| Export ▾ | `#export-menu` → `export-md` (Markdown) / `export-pdf` (Print / Save PDF) |
| ◐ theme | `toggleTheme` (§11) |
| ⓘ about | `#about-overlay` modal |
| Run / ⌘⏎ | force a run |

Menus close on outside click; `updateInsertTableState()` enables the Markdown menu
only in a Markdown region.

---

## 10. File rail & layout

- `renderFileRail(folder, files)` lists the open file's folder; clicking switches
  files (`switchToFile` → `read_tens`). Restored on boot from `localStorage`.
- `applyRailCollapsed()` collapses the rail (chevron `#rail-toggle`); state +
  width persist.
- `#rail-resizer` and `#split-resizer` set flex bases via pointer drag
  (`setEditorWidthFromClientX`, `editorWidthBounds`); widths persist in
  `localStorage`.

---

## 11. Theming logic

`applyTheme(theme)` sets `html[data-theme]` and (when the user chooses explicitly)
persists to `localStorage`. On boot the theme **follows the system**
(`prefers-color-scheme`) until the user toggles, and a `matchMedia` listener keeps
following until then. The CSS tokens (§4) and the CodeMirror `editorTheme` both
read the same variables, so one toggle reflows the whole app and editor.

---

## 12. Security (CSP)

`tauri.conf.json` sets a strict Content-Security-Policy:

```
default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';
img-src 'self' data:; font-src 'self'; connect-src 'self' ipc: http://ipc.localhost;
object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'
```

Implications for UI code:
- **No inline `<script>`, no CDN** — CodeMirror and KaTeX are bundled locally.
- `style-src 'unsafe-inline'` allows the CM/KaTeX inline styles.
- `img-src … data:` allows pasted/data-URL images; remote (`https:`) images need a
  CSP change. `blob:` is only used by the download helper, not `<img>`.
- `connect-src … ipc:` allows the Tauri IPC bridge. `freezePrototype: true` hardens
  the runtime. `read_tens` validates `.tens` paths on the Rust side.

---

## 13. "Where do I change…?"

| Want to change | Edit |
|----------------|------|
| Colours / theme | the CSS variables in `index.html` (§4) |
| Editor look (tens block, gutter, selection, fonts) | `editorTheme` + `sourceDecorations` / `buildSentinelHiding` in `main.js` |
| DSL syntax colours | `--syntax-*` tokens + `decorateTensTokens` |
| Autocomplete entries | `BUILTINS` in `completion.js` |
| Toolbar buttons / menus | header DOM in `index.html` + the handlers in `main.js` §9 |
| Markdown rendering | `renderMarkdown` and helpers (`main.js` §7) |
| Preview ordering / side-by-side rows | `orderedPreviewBlocks` / `groupOutputRows` |
| Scroll-sync behaviour | the `// ---- scroll linking ----` section |
| Add a CM extension | the `extensions: [...]` array in `initEditor` |
| CSP / allowed sources | `tauri.conf.json` `app.security.csp` |
| CodeMirror version / packages | `package.json` + `codemirror-entry.js`, then `npm run bundle:cm` |
```
