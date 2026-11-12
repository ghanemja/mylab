// Everyday editing: auto-indent, "Format document", code folding, comment toggling,
// moving and copying lines, and several cursors at once.

import { toast } from "./ui.js";
import { analyze, formatCode, indentAfter, dedentFor, indentOf } from "./blocks.js";
import { splitSections } from "./sections.js";

/* global CodeMirror */

export function initEditing(app) {
  const { state } = app;
  const cm = app.editor;
  const unit = () => cm.getOption("indentUnit");
  const isCode = () => !!state.active && state.active.endsWith(".m");
  const editable = () => !!state.active && !cm.getOption("readOnly");

  /* ---------- structure cache (per document version) ---------- */

  const cache = new WeakMap();
  function structure(doc = cm.getDoc()) {
    const gen = doc.changeGeneration();
    let c = cache.get(doc);
    if (!c || c.gen !== gen) {
      const text = doc.getValue();
      const a = analyze(text, { unit: unit() });
      const folds = new Map(); // start line -> last hidden line
      for (const r of a.ranges) {
        const last = r.noEnd ? r.end - 1 : r.end - 1;
        if (last > r.start && !folds.has(r.start)) folds.set(r.start, last);
      }
      const split = splitSections(text);
      if (split.hasSections) {
        for (const s of split.sections) {
          if (!s.header) continue;
          let last = s.end - 1;
          while (last > s.start && !a.lines[last].trim()) last--;
          if (last > s.start && !folds.has(s.start)) folds.set(s.start, last);
        }
      }
      c = { gen, analysis: a, folds, split };
      cache.set(doc, c);
    }
    return c;
  }

  /* ---------- folding ---------- */

  function foldRange(editor, start) {
    if (!isCode()) return undefined;
    const last = structure(editor.getDoc()).folds.get(start.line);
    if (last === undefined) return undefined;
    return { from: CodeMirror.Pos(start.line, editor.getLine(start.line).length), to: CodeMirror.Pos(last, editor.getLine(last).length) };
  }
  cm.setOption("foldOptions", { rangeFinder: foldRange, widget: "⋯", minFoldSize: 1 });
  cm.setOption("foldGutter", true);

  function foldAt(line, how) {
    // Fold the block that starts on this line, or else the smallest block around it.
    const { folds } = structure();
    let start = line;
    if (how !== "unfold" && !folds.has(line)) {
      let best = -1;
      for (const [s, e] of folds) if (s <= line && e >= line && s > best) best = s;
      if (best < 0) return;
      start = best;
    }
    cm.foldCode(CodeMirror.Pos(start, 0), { rangeFinder: foldRange, widget: "⋯" }, how);
    if (how === "fold") cm.setCursor(start, cm.getLine(start).length);
  }

  function foldAll() {
    const { folds } = structure();
    cm.operation(() => {
      for (const s of [...folds.keys()].sort((a, b) => b - a)) cm.foldCode(CodeMirror.Pos(s, 0), { rangeFinder: foldRange, widget: "⋯" }, "fold");
    });
  }

  function unfoldAll() {
    cm.operation(() => {
      for (const m of cm.getAllMarks()) if (m.__isFold) m.clear();
    });
  }

  /* ---------- auto-indent ---------- */

  function newline(editor) {
    if (editor.getOption("readOnly")) return CodeMirror.Pass;
    const sels = editor.listSelections();
    if (!isCode() || sels.length > 1 || editor.somethingSelected()) {
      editor.execCommand("newlineAndIndent");
      return undefined;
    }
    const cur = editor.getCursor();
    const text = editor.getLine(cur.line);
    const before = text.slice(0, cur.ch);
    const after = text.slice(cur.ch);
    const lines = editor.getValue().split("\n");
    lines[cur.line] = before;
    let indent;
    try {
      indent = indentAfter(lines, cur.line, { unit: unit() });
    } catch {
      indent = indentOf(text, unit());
    }
    const skip = after.length - after.trimStart().length;
    editor.replaceRange("\n" + " ".repeat(indent), cur, CodeMirror.Pos(cur.line, cur.ch + skip), "+input");
    editor.scrollIntoView(null, 30);
    return undefined;
  }
  cm.addKeyMap({ Enter: newline });

  // Typing "end", "else", "case"… at the start of a line moves it back under its block.
  cm.on("inputRead", (editor, change) => {
    if (!isCode() || change.origin !== "+input" || change.text.length !== 1 || editor.listSelections().length > 1) return;
    const cur = editor.getCursor();
    const text = editor.getLine(cur.line);
    if (cur.ch !== text.length || !/^\s*(end|else|elseif|case|otherwise|catch)$/.test(text)) return;
    const lines = editor.getValue().split("\n");
    let want = null;
    try {
      want = dedentFor(lines, cur.line, { unit: unit() });
    } catch {
      want = null;
    }
    const have = indentOf(text, unit());
    if (want === null || want === have) return;
    const ws = text.length - text.trimStart().length;
    editor.replaceRange(" ".repeat(want), CodeMirror.Pos(cur.line, 0), CodeMirror.Pos(cur.line, ws), "+input");
  });

  /* ---------- format ---------- */

  async function formatDocument() {
    if (!editable()) return;
    if (!isCode()) {
      toast("Only .m files can be formatted.");
      return;
    }
    const before = cm.getValue();
    const after = formatCode(before, { unit: unit() });
    if (after === before) {
      toast("The indentation is already tidy.");
      return;
    }
    if (after.replace(/[ \t]+/g, "") !== before.replace(/[ \t]+/g, "")) {
      // Safety net: formatting may only ever change spaces.
      toast("Formatting was skipped because it would have changed the code.", "error");
      return;
    }
    await app.snapshot(state.active, "Before formatting");
    app.replaceText(state.active, after);
    toast("Indentation fixed. Undo with Ctrl+Z.");
  }

  /* ---------- lines ---------- */

  const exec = (name) => () => editable() && cm.execCommand(name);

  function indentLines(how) {
    if (!editable()) return;
    cm.operation(() => {
      for (const r of cm.listSelections()) {
        const from = Math.min(r.anchor.line, r.head.line);
        const to = Math.max(r.anchor.line, r.head.line);
        for (let l = from; l <= to; l++) cm.indentLine(l, how);
      }
    });
  }

  app.commands.register([
    { id: "edit.comment", group: "Edit", title: "Comment or uncomment lines", keys: "Mod-/", matlab: ["Mod-/", "Mod-R"], scope: "editor", run: () => editable() && cm.toggleComment({ indent: true }) },
    { id: "edit.duplicateLine", group: "Edit", title: "Copy line down", keys: "Shift-Alt-ArrowDown", scope: "editor", run: exec("duplicateLine") },
    { id: "edit.moveUp", group: "Edit", title: "Move line up", keys: "Alt-ArrowUp", scope: "editor", run: exec("swapLineUp") },
    { id: "edit.moveDown", group: "Edit", title: "Move line down", keys: "Alt-ArrowDown", scope: "editor", run: exec("swapLineDown") },
    { id: "edit.deleteLine", group: "Edit", title: "Delete line", keys: "Mod-Shift-K", scope: "editor", run: exec("deleteLine") },
    { id: "edit.nextOccurrence", group: "Edit", title: "Add the next match to the selection", keys: "Mod-D", scope: "editor", run: exec("selectNextOccurrence"), hint: "Several cursors: select the word, then the next place it appears" },
    {
      id: "edit.allOccurrences", group: "Edit", title: "Select every match of the word", keys: "Mod-Shift-L", scope: "editor",
      run: () => {
        if (!editable()) return;
        if (!cm.somethingSelected()) cm.execCommand("selectNextOccurrence");
        cm.execCommand("findAllUnder");
      },
    },
    { id: "edit.cursorAbove", group: "Edit", title: "Add a cursor above", keys: "Mod-Alt-ArrowUp", scope: "editor", run: exec("addCursorToPrevLine") },
    { id: "edit.cursorBelow", group: "Edit", title: "Add a cursor below", keys: "Mod-Alt-ArrowDown", scope: "editor", run: exec("addCursorToNextLine") },
    { id: "edit.indent", group: "Edit", title: "Indent lines", keys: "Mod-]", scope: "editor", run: () => indentLines("add") },
    { id: "edit.outdent", group: "Edit", title: "Outdent lines", keys: "Mod-[", scope: "editor", run: () => indentLines("subtract") },
    { id: "edit.format", group: "Code", title: "Format document (fix indentation)", keys: "Shift-Alt-F", matlab: ["Shift-Alt-F", "Mod-I"], run: formatDocument, when: () => !!state.active, hint: "Format document: fix the indentation of the whole file" },
    { id: "edit.fold", group: "Code", title: "Fold the block at the cursor", keys: "Mod-Shift-[", scope: "editor", run: () => foldAt(cm.getCursor().line, "fold") },
    { id: "edit.unfold", group: "Code", title: "Unfold the block at the cursor", keys: "Mod-Shift-]", scope: "editor", run: () => foldAt(cm.getCursor().line, "unfold") },
    { id: "edit.foldAll", group: "Code", title: "Fold all blocks", run: foldAll, when: () => isCode() },
    { id: "edit.unfoldAll", group: "Code", title: "Unfold all blocks", run: unfoldAll, when: () => isCode() },
  ]);

  Object.assign(app, { structure, formatDocument, foldAt, foldAll, unfoldAll, newline });
}
