// Step-through panel: records a run with mylab.debugger.trace (in a fresh session, so the
// user's workspace is not touched) and lets the user walk through it forwards and backwards.
//
// A trace is { steps, output, error, error_line, truncated, figures }. Each step is recorded
// just before a statement runs: { line, file, function, depth, frame, set, del, out }.
// `set`/`del` are the changes to that frame's variables since its previous step, and `out` is
// how many characters of the output had been printed. The workspace at a step is rebuilt by
// replaying set/del of all earlier steps of the same frame.

import { $, el, toast } from "./ui.js";

export const MAX_STEPS = 5000;

export function initStepper(app) {
  const { state } = app;
  let trace = null; // the active recording, plus { file, index, lastReal }
  let mark = null; // { doc, handle, cls }
  const breakpoints = new Map(); // CodeMirror.Doc -> Set(line handle)
  const pathCache = new Map();

  /* ---------- breakpoints ---------- */

  function toggleBreakpoint(line) {
    const editor = app.editor;
    const doc = editor.getDoc();
    if (!state.active || !state.active.endsWith(".m")) return;
    const handle = doc.getLineHandle(line);
    if (!handle) return;
    let set = breakpoints.get(doc);
    if (!set) breakpoints.set(doc, (set = new Set()));
    if (set.has(handle)) {
      set.delete(handle);
      editor.setGutterMarker(handle, "bp-gutter", null);
    } else {
      set.add(handle);
      editor.setGutterMarker(handle, "bp-gutter", el("div", { class: "bp-dot", title: "Breakpoint (click to remove)" }));
    }
  }

  /** 1-based breakpoint lines of a project file. */
  function breakLines(path) {
    const doc = state.docs.get(path);
    const set = doc && breakpoints.get(doc);
    const lines = new Set();
    if (!set) return lines;
    for (const h of [...set]) {
      const n = doc.getLineNumber(h);
      if (n === null) set.delete(h);
      else lines.add(n + 1);
    }
    return lines;
  }

  /* ---------- replay ---------- */

  function stepPath(step) {
    if (step.end) return null;
    if (!step.file) return trace.file;
    if (pathCache.has(step.file)) return pathCache.get(step.file);
    const dir = app.dirName(trace.file);
    const candidates = [...state.files.keys()].filter((p) => app.baseName(p) === step.file);
    const path = candidates.find((p) => app.dirName(p) === dir) || candidates.sort((a, b) => a.length - b.length)[0] || null;
    pathCache.set(step.file, path);
    return path;
  }

  function workspaceAt(index) {
    const frame = trace.steps[index].frame;
    const ws = new Map();
    for (let i = 0; i <= index; i++) {
      const s = trace.steps[i];
      if (s.frame !== frame) continue;
      for (const [name, info] of Object.entries(s.set || {})) ws.set(name, info);
      for (const name of s.del || []) ws.delete(name);
    }
    return ws;
  }

  function clearMark() {
    if (!mark) return;
    try {
      mark.doc.removeLineClass(mark.handle, "wrap", mark.cls);
      mark.doc.removeLineClass(mark.handle, "gutter", "cm-step-gutter");
    } catch {
      /* line is gone */
    }
    mark = null;
  }

  function renderWorkspace(index) {
    const step = trace.steps[index];
    const ws = workspaceAt(index);
    const changed = new Set(Object.keys(step.set || {}));
    const tbody = $("var-table").querySelector("tbody");
    tbody.innerHTML = "";
    const names = [...ws.keys()].sort((a, b) => a.localeCompare(b));
    for (const name of names) {
      const v = ws.get(name);
      tbody.appendChild(
        el(
          "tr",
          { class: changed.has(name) ? "step-changed" : "", "data-name": name, title: changed.has(name) ? `${name} was just changed` : name },
          el("td", {}, name),
          el("td", {}, v.size),
          el("td", {}, v.class),
          el("td", { title: v.preview }, v.preview)
        )
      );
    }
    $("var-empty").hidden = names.length > 0;
    $("var-empty").textContent = "No variables yet at this step.";
    $("var-table").hidden = names.length === 0;
    $("var-count").textContent = names.length ? `${names.length}` : "";
    $("ws-title").textContent = step.function && step.function !== "base" ? `Workspace of ${step.function}()` : "Workspace at this step";
  }

  function go(index) {
    if (!trace) return;
    const n = trace.steps.length;
    index = Math.max(0, Math.min(n - 1, index));
    trace.index = index;
    const step = trace.steps[index];
    const editor = app.editor;
    const failing = !!trace.error && index === trace.lastReal;
    const path = stepPath(step);

    clearMark();
    if (path && step.line > 0 && app.isText(path)) {
      if (state.active !== path) app.openFile(path, { focus: false });
      const doc = app.getDoc(path);
      if (step.line <= doc.lineCount()) {
        const cls = failing ? "cm-step-error" : "cm-step-line";
        const handle = doc.addLineClass(step.line - 1, "wrap", cls);
        doc.addLineClass(step.line - 1, "gutter", "cm-step-gutter");
        mark = { doc, handle, cls };
        editor.setCursor({ line: step.line - 1, ch: 0 });
        editor.scrollIntoView({ line: step.line - 1, ch: 0 }, 90);
      }
    }

    const bar = $("step-bar");
    bar.dataset.index = String(index);
    bar.dataset.line = String(step.line || 0);
    bar.dataset.file = path || "";
    bar.dataset.function = step.function || "";
    $("step-slider").value = String(index);
    $("step-first").disabled = $("step-back").disabled = index === 0;
    $("step-last").disabled = $("step-fwd").disabled = index === n - 1;

    let where;
    if (step.end) where = trace.error ? "the run stopped with an error" : "finished: this is the workspace after the last line";
    else {
      const file = path ? app.baseName(path) : step.file || "";
      where = `about to run line ${step.line}${file ? " of " + file : ""}`;
      if (step.function && step.function !== "base") where += ` · inside ${step.function}() · call depth ${step.depth}`;
    }
    $("step-count").textContent = `Step ${index + 1} of ${n}`;
    $("step-where").textContent = where;

    const msg = $("step-msg");
    msg.innerHTML = "";
    if (trace.error && (failing || step.end)) {
      const realPath = stepPath(trace.steps[trace.lastReal] || {});
      msg.append(
        el(
          "div",
          { class: "step-error", role: "alert" },
          el("strong", {}, failing ? "This line fails: " : `Error${trace.error_line ? ` (line ${trace.error_line}${realPath ? " of " + app.baseName(realPath) : ""})` : ""}: `),
          trace.error
        )
      );
    }
    if (trace.truncated && index === n - 1) {
      msg.append(el("div", { class: "step-note" }, `This run is long: only the first ${MAX_STEPS.toLocaleString()} steps were recorded, so the walk-through stops here.`));
    }
    msg.hidden = !msg.childElementCount;

    renderWorkspace(index);
    const out = $("step-out");
    out.textContent = trace.output.slice(0, step.out);
    $("step-out-empty").hidden = step.out > 0;
    out.parentElement.scrollTop = out.parentElement.scrollHeight;
  }

  function jumpBreakpoint(dir) {
    if (!trace) return;
    const cache = new Map();
    const isBreak = (s) => {
      const path = stepPath(s);
      if (!path || !s.line) return false;
      if (!cache.has(path)) cache.set(path, breakLines(path));
      return cache.get(path).has(s.line);
    };
    let any = false;
    for (const set of breakpoints.values()) any = any || set.size > 0;
    if (!any) {
      toast("No breakpoints yet. Click next to a line number in the editor to set one.");
      return;
    }
    for (let i = trace.index + dir; i >= 0 && i < trace.steps.length; i += dir) {
      if (isBreak(trace.steps[i])) return go(i);
    }
    toast(dir > 0 ? "No breakpoint is reached after this step — went to the end." : "No breakpoint is reached before this step — went to the start.");
    go(dir > 0 ? trace.steps.length - 1 : 0);
  }

  /* ---------- entering and leaving ---------- */

  async function startStep() {
    const f = app.runnableFile();
    if (!f || !app.canRun()) return null;
    exitStep();
    app.clearErrorMark();
    app.printEcho("", `step through ${f.name}`);
    app.setRunning(true, "Recording steps");
    let res;
    try {
      res = await app.engine.trace(f.code, { fs: app.fsSpec(), cwd: app.dirName(f.name), maxSteps: MAX_STEPS });
    } catch (e) {
      app.setRunning(false);
      if (String(e.message) !== "Stopped.") app.printError({ message: e.message });
      return null;
    }
    app.setRunning(false);
    const steps = (res.steps || []).filter((s) => !s.end || true);
    if (!steps.some((s) => !s.end)) {
      if (res.error) {
        const at = app.locateError(res, { file: f.name });
        if (at.file) app.markErrorLine(at.file, at.line);
        app.printError({ message: res.error, line: at.line, file: at.file, where: at.file || "" });
      } else {
        toast("There is nothing to step through: this file has no statements.");
      }
      return null;
    }
    pathCache.clear();
    let lastReal = steps.length - 1;
    while (lastReal > 0 && steps[lastReal].end) lastReal--;
    trace = { ...res, steps, file: f.name, index: 0, lastReal };
    document.body.dataset.stepping = "true";
    app.editor.setOption("readOnly", true);
    $("step-bar").hidden = false;
    $("step-slider").max = String(steps.length - 1);
    $("console-out").hidden = true;
    $("step-out-wrap").hidden = false;
    $("btn-step").setAttribute("aria-pressed", "true");
    if (res.figures && res.figures.length) app.addFigures(res.figures, { kind: "file", file: f.name });
    if (res.animations && res.animations.length) app.addFigures(res.animations, { kind: "file", file: f.name }, "gif");
    app.printInfo(
      `Recorded ${steps.length} step${steps.length === 1 ? "" : "s"}` +
        (res.truncated ? " (the run was cut short)" : "") +
        ". Use the arrows above the editor to walk through them. Your own workspace was not changed."
    );
    app.editor.refresh();
    go(0);
    $("step-fwd").focus({ preventScroll: true });
    return trace;
  }

  function exitStep() {
    if (!trace) return;
    clearMark();
    trace = null;
    delete document.body.dataset.stepping;
    app.editor.setOption("readOnly", false);
    $("step-bar").hidden = true;
    $("console-out").hidden = false;
    $("step-out-wrap").hidden = true;
    $("btn-step").setAttribute("aria-pressed", "false");
    $("ws-title").textContent = "Workspace";
    $("var-empty").textContent = "No variables yet. Run some code to fill the workspace.";
    app.renderVariables(state.variables);
    app.editor.refresh();
  }

  $("btn-step").addEventListener("click", () => (trace ? exitStep() : startStep()));
  $("step-exit").addEventListener("click", exitStep);
  $("step-first").addEventListener("click", () => go(0));
  $("step-back").addEventListener("click", () => go(trace.index - 1));
  $("step-fwd").addEventListener("click", () => go(trace.index + 1));
  $("step-last").addEventListener("click", () => go(trace.steps.length - 1));
  $("step-prev-bp").addEventListener("click", () => jumpBreakpoint(-1));
  $("step-next-bp").addEventListener("click", () => jumpBreakpoint(1));
  $("step-slider").addEventListener("input", (e) => go(Number(e.target.value)));

  app.editor.on("gutterClick", (cm, line) => toggleBreakpoint(line));

  // Arrow keys step while the walk-through is open (the editor is read-only then).
  document.addEventListener(
    "keydown",
    (e) => {
      if (!trace || e.ctrlKey || e.metaKey || e.altKey) return;
      if (document.querySelector("dialog[open]")) return;
      const t = e.target;
      const typing = (t.tagName === "INPUT" && t.type !== "range") || (t.tagName === "TEXTAREA" && !t.closest(".CodeMirror")) || t.tagName === "SELECT";
      if (typing) return;
      const action = {
        ArrowRight: () => go(trace.index + 1),
        ArrowLeft: () => go(trace.index - 1),
        Home: () => go(0),
        End: () => go(trace.steps.length - 1),
        Escape: exitStep,
      }[e.key];
      if (!action) return;
      e.preventDefault();
      e.stopPropagation();
      action();
    },
    true
  );

  Object.assign(app, {
    startStep,
    exitStep,
    stepGo: go,
    stepActive: () => !!trace,
    stepTrace: () => trace,
    toggleBreakpoint,
  });
}
