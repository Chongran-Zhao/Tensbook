# Editor rewrite guide: migrate to CodeMirror 6 (+ two independent fixes)

Self-contained spec for the implementer (e.g. Codex). The user has decided to
**adopt CodeMirror 6** for the source editor. That migration natively fixes the
editor-rendering bugs (caret, gutter, selection) and dissolves the blank-line bug;
two remaining issues (markdown links, preview-rerender scroll) live in the preview
pipeline and are fixed separately.

**Scope:** front-end only (`ui/`), plus a frontend build step for bundling and one
optional CSP line. **Do not touch the Rust engine** — all 143 Rust tests stay
green, and the on-disk `.tens` format must not change.

---

## How the six reported bugs map to this plan

| # | Bug | Resolution |
|---|-----|-----------|
| 1 | Markdown `[text](url)` links don't render | **Part B** — preview renderer (independent of the editor) |
| 2 | ⌘Z / live-run scrolls editor to the first line | **Part C** — preserve scroll across preview re-render |
| 3 | Selecting code also selects the line numbers (WKWebView) | **Part A** — CM gutter is not selectable text |
| 4 | Code block with ≥3 blank lines is collapsed on input | **Part A** — single buffer has no per-block auto-split; dissolves |
| 5 | Caret drawn in wrong place on wrapped highlighted lines | **Part A** — CM unifies caret + highlighting |
| 6 | Line numbers drift / run out under soft-wrap | **Part A** — CM gutter is wrap-aware |

So the CodeMirror migration (Part A) handles 3, 4, 5, 6 and the weak multi-textarea
undo at once. Parts B and C are small and could be done before or after A.

---

## Current architecture (what is being replaced)

Left pane `#editor-wrap` (`ui/index.html`) today:
- `#source-gutter` — separate column, one fixed-height `<div class="gutter-line">`
  per **source line** (`renderSourceGutter` in `ui/main.js`). Not wrap-aware (Bug 6),
  selectable in WKWebView (Bug 3).
- `#source-editor` — renders one `<section class="source-block">` per document
  block; each has a `.source-highlight` `<pre>` overlay (visible colored text) and a
  transparent `<textarea>` (caret only). The overlay/textarea wrap differently →
  caret drift (Bug 5).
- A `docBlocks` array is the editor model; `splitTensBlockOnBlankGap` auto-splits a
  tens block on 3 newlines and trims (Bug 4). `syncSourceFromBlocks` /
  `serializeDocument` rebuild the `.tens` text with hidden `<!-- tensorforge:tens -->`
  sentinels.
- Hidden `#editor` textarea holds the serialized source for the runner.

Functions to retire (editor side, `ui/main.js`): `renderSourceEditor`,
`renderSourceBlock`, `renderSourceGutter`, `makeBlock`, `cleanMarkdownText`,
`trimOuterBlankLines`, `parseSourceDocument` (as the *editor* model),
`serializeDocument`/`serializeBlock`, `syncSourceFromBlocks`,
`recomputeSourceLines`, `commitSourceTextareaChange`, `autoResize`,
`resizeAllTextareas`, `updateHighlight`, `highlightSource`/`highlightTens`/
`highlightMarkdownSource`, `splitTensBlockOnBlankGap`,
`collapseEmptyMarkdownBetweenTens`, `removeEmptyTensBlock`,
`moveAcrossBlocksWithArrow`, `focusBlockAtColumn`, and the textarea completion
(`setupCompletion`/`setupMarkdownCompletion` in `ui/completion.js`).

Keep/adapt (preview + app side, `ui/main.js`): `run`, `scheduleLiveRun`,
`renderOutputs`, `renderMarkdown` + the inline pipeline, `syncOutputToEditor` /
`syncEditorToOutput`, `suppressScrollSync`, export (`downloadText`,
`printDocument`), open/save (Tauri `open_tens`/`read_tens`/`save`), insert
table/link/image commands, `KATEX_MACROS`, theme handling, CSP.

The on-disk `.tens` format is unchanged: Markdown by default, code inside
`<!-- tensorforge:tens -->` … `<!-- /tensorforge:tens -->` sentinels, blocks joined
by a blank line. The CM document **is** this text.

---

## Part A — CodeMirror 6 migration

### A.0 Bundling under CSP (do this first — it gates everything)

The app runs in **macOS WKWebView** with a strict CSP (`script-src 'self'`,
`tauri.conf.json`). **No CDN, no inline scripts.** CodeMirror must be bundled
locally as ESM and loaded from `ui/`.

- The frontend currently has **no build step** (plain ESM `ui/main.js` + vendored
  `ui/vendor/katex/`). Add a minimal bundler:
  - Add `package.json` with CodeMirror deps and an `esbuild` build that outputs a
    single ESM bundle to `ui/vendor/codemirror/cm.bundle.js`.
  - Import it from `ui/main.js` (`import { EditorView, ... } from "./vendor/codemirror/cm.bundle.js"`).
  - Wire the build into Tauri: set `build.beforeDevCommand` / `build.beforeBuildCommand`
    in `tauri.conf.json` to run the bundle step (and run it in CI before
    `cargo tauri build`). Alternatively commit the prebuilt bundle and document the
    regen command. **The bundle must be committed or reproducibly built in CI** so
    the release workflow keeps working.
- CM6 packages: `@codemirror/state`, `@codemirror/view`, `@codemirror/commands`,
  `@codemirror/language`, `@codemirror/autocomplete`, `@codemirror/search`,
  `@codemirror/lang-markdown`, `@lezer/highlight`.
- Verify in `cargo tauri dev` that the bundle loads with **zero CSP violations**
  in the Web Inspector console. Keep KaTeX as-is.

### A.1 Single editor over the source buffer

- Mount **one** `EditorView` in `#editor-wrap`, replacing `#source-gutter` +
  `#source-editor` + the hidden `#editor`.
- The CM document is the full `.tens` source (the same text `serializeDocument`
  used to produce, sentinels included). Open/Save use `view.state.doc.toString()`
  directly; the runner uses the same string. Drop the `docBlocks` editor model
  (you may still parse the source into blocks for **preview** mapping only).
- Enable `lineNumbers()` (wrap-aware, non-selectable gutter) → **fixes Bug 3 & 6**.
- Enable `EditorView.lineWrapping` → wrapping with a correct caret → **fixes Bug 5**.
- Use CM's history (`@codemirror/commands` `history()` + default keymap) → proper
  document-wide undo (replaces the weak per-textarea undo).
- Because there is a single buffer, blank lines are just text — **no auto-split,
  no trimming → Bug 4 is gone** with no special code.

### A.2 Tens blocks & syntax highlighting via decorations

Preserve the current UX (blue `.tens` regions, gold "notes" accent, hidden
sentinels, distinct highlighting) using CM decorations + a language:

- **Region styling:** a `ViewPlugin`/`StateField` that scans for
  `<!-- tensorforge:tens -->` … `<!-- /tensorforge:tens -->` and applies line
  decorations: blue background + left bar + a "tens" tag on the code lines (reuse
  the existing `--tens-frame` styling), and the gold left bar on markdown regions.
- **Hide the sentinels:** apply a replace/line decoration that visually hides (or
  dims) the sentinel lines so the user still sees clean blue boxes, matching today.
- **Protect the sentinels:** add a transaction filter (`EditorState.transactionFilter`
  / `changeFilter`) or atomic ranges so users can't accidentally delete/break the
  `<!-- … -->` markers while editing around them.
- **Highlighting:** use `@codemirror/lang-markdown` for markdown regions, and a
  custom language for the tens DSL inside tens regions (a `StreamLanguage` is the
  simplest; port the token rules from the old `highlightTens` — keywords/builtins
  are already enumerated as `TENS_BUILTINS` / `TENS_KEYWORDS` in `ui/main.js`).
  Map tokens to the existing `--syntax-*` CSS variables via a `HighlightStyle`.
- **Theme:** an `EditorView.theme` (light + dark) driven by the existing CSS
  variables / `data-theme`, matching the paper/mocha palette. No hard-coded colors.

### A.3 Re-wire the surrounding behaviors to CM

- **Live run:** on document change (debounced, reuse `scheduleLiveRun`), run with
  `view.state.doc.toString()` and render the preview (`run`/`renderOutputs` reused).
- **Editor↔preview scroll sync + click-to-jump:** today this maps block/source
  lines to preview elements carrying `data-source-line` / `data-source-block-id`
  (`sourceLineAttr`, `syncOutputToEditor`/`syncEditorToOutput`). Re-implement using
  CM line numbers: editor scroll = `view.scrollDOM.scrollTop`; map a viewport line
  via `view.lineBlockAtHeight` / `view.state.doc.lineAt`; jump with
  `view.dispatch({ effects: EditorView.scrollIntoView(pos, { y: "start" }) })`.
  Clicking a preview block scrolls CM to that source line and vice-versa.
- **Completion:** reimplement with `@codemirror/autocomplete` using the data in
  `ui/completion.js` (`TENS_BUILTINS`, keywords, signatures/docs). Scope tens
  completions to tens regions.
- **Commands:** Add Tens (insert a sentinel-delimited empty tens region at the
  cursor), insert table / insert link / paste image (data: URL) → CM commands that
  dispatch a transaction inserting text at the selection. Keep ⌘↵ run, ⌘S save,
  ⌘O open, ⌘N new (move the existing `document` keydown handlers or use CM keymaps).
- **Resizable split & theme toggle:** unchanged (operate on `#editor-wrap`).

### A.4 Acceptance for Part A (verify in `cargo tauri dev`)

- Bug 3: select code (drag + select-all), ⌘C, paste → **no line numbers**.
- Bug 5: caret on a wrapped dense-LaTeX line sits exactly between the right glyphs.
- Bug 6: wrapped paragraphs/long lines — every line's number aligns with its first
  visual row; gutter height matches content; the caret line always has a number.
- Bug 4: a tens region with several blank lines — typing never deletes blank lines.
- Round-trip: open a `.tens` file → edit → save → the file is equivalent (sentinels
  intact, runner output unchanged). Existing example files still run.
- Zero CSP violations; light/dark theme correct; completion, export, image paste,
  click-to-jump all still work.

---

## Part B — Issue 1: Markdown links (preview renderer, independent of CM)

**Symptom.** `[text](url)` renders as literal text; links never clickable.

**Root cause.** The inline renderer has no link path: `renderInlinePlain`
(`ui/main.js`) only matches `![alt](data:image/...;base64,...)`; `inlineStyles`
only does bold/italic. (A click handler for `a[href^='http']`/`mailto:` already
exists, and an insert-link snippet exists — only the renderer is missing.)

**Fix.**
- In the inline pipeline add `[text](url)` → `<a href="SAFEURL">…inline-styled
  text…</a>`. Parse images first, then links. HTML-escape text and href.
- **Sanitize the URL:** allow only `http:` / `https:` / `mailto:`; anything else
  (e.g. `javascript:`, `data:`, `file:`) → render the literal `[text](url)`, no `<a>`.
- (Optional) allow `https:` images too; if so, update CSP `img-src 'self' data:` →
  `img-src 'self' data: https:` in `tauri.conf.json`.

**Acceptance.** `[Google](https://google.com)` clickable (opens via system opener);
`[x](javascript:alert(1))` stays plain text; `[a](mailto:…)` clickable. Verify in
Chrome static preview (`python3 -m http.server --directory ui`).

---

## Part C — Issue 2: ⌘Z / live-run scrolls editor to the top

**Symptom.** Undo (and edit-driven live runs) scroll the editor to the first line.

**Root cause.** `renderOutputs` does `output.innerHTML = ""` (resets
`output.scrollTop` to 0) then calls `syncOutputToEditor()`, which pulls the editor
to the top. With CM, "editor scroll" is `view.scrollDOM.scrollTop`.

**Fix.** Preserve scroll across the live-run re-render: snapshot `output.scrollTop`
(and `view.scrollDOM.scrollTop`) before the rebuild; restore `output.scrollTop`
after; and **do not** reposition the editor on an edit-driven re-render (wrap the
rebuild in `suppressScrollSync()` so the trailing `syncOutputToEditor()` no-ops).
A genuine user scroll of the preview must still sync. (CM's native undo already
removes the old textarea-undo weakness; this fix stops the remaining scroll pull.)

**Acceptance.** Scroll editor to the middle; type and ⌘Z repeatedly → editor stays
put, preview doesn't jump. Verify in `cargo tauri dev`.

---

## Suggested order
1. **A.0** bundle CM under CSP (gate) — confirm it loads with no CSP errors.
2. **A.1** single CM editor: doc = source, `lineNumbers()`, `lineWrapping`, history,
   open/save/run wired → verify Bugs 3, 5, 6, 4 and round-trip.
3. **A.2** decorations: blue tens regions, hidden+protected sentinels, highlighting,
   theme.
4. **A.3** behaviors: scroll-sync/click-to-jump, completion, commands, shortcuts.
5. **B** markdown links (small, independent — can be done anytime).
6. **C** preview-rerender scroll preservation.

## Notes / decisions already made
- **Issues 5 & 6:** Option **C (CodeMirror)** — chosen. (Supersedes the earlier
  A/B wrap-vs-caret options.)
- **Issue 4:** dissolves under the single-buffer model; the only way to create a
  tens region becomes the "Add Tens" command (inserts sentinels). No per-block
  blank-line gesture is needed.

## Hard constraints (recap)
- Rust engine untouched; 143 tests stay green.
- `.tens` on-disk format unchanged (round-trip safe; sentinels preserved).
- CSP-clean: CM bundled locally (no CDN), no inline scripts; update `img-src` only
  if enabling remote images.
- Bundle build reproducible in CI so `.github/workflows/release.yml`
  (`cargo tauri build`) still produces a working DMG.
- Verify WebKit-sensitive items (Bugs 2, 3, 5, 6) in `cargo tauri dev`, not just
  Chrome.
