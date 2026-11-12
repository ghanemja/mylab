// "Publish report": run the file section by section in a fresh session and show the result as
// a standalone HTML document that can be downloaded or printed to PDF.

import { $, toast, downloadBlob } from "./ui.js";
import { splitSections, sectionParts, sectionCode, tailCode } from "./sections.js";
import { buildReport } from "./report.js";

export function initPublish(app) {
  const { state, prefs } = app;
  let last = null; // { report, html, filename }

  function render() {
    last.report.author = $("report-author").value.trim();
    last.html = buildReport(last.report);
    $("report-frame").srcdoc = last.html;
  }

  async function publish() {
    const f = app.runnableFile();
    if (!f || !app.canRun()) return null;
    app.exitStep && app.exitStep();
    const split = splitSections(f.code);
    const tail = tailCode(split);
    const codes = split.sections.map((_, i) => sectionCode(split, i));
    app.printEcho("", `publish ${f.name}`);
    app.setRunning(true, "Publishing");
    let results;
    try {
      results = await app.engine.publish(codes, { tail, fs: app.fsSpec(), cwd: app.dirName(f.name) });
    } catch (e) {
      app.setRunning(false);
      if (String(e.message) !== "Stopped.") app.printError({ message: e.message });
      return null;
    }
    app.setRunning(false);

    const sections = split.sections.map((s, i) => {
      const parts = sectionParts(split, i);
      const result = { ...(results[i] || { skipped: true }) };
      if (result.error) {
        const at = app.locateError(result, { file: f.name, lineOffset: s.start, codeLines: tail ? codes[i].split("\n").length : 0, tailStart: split.tailStart });
        result.error_line = at.line;
        result.error_file = at.file && at.file !== f.name ? at.file : null;
      }
      return { title: parts.title, prose: parts.prose, code: parts.code, result };
    });
    const base = app.baseName(f.name).replace(/\.m$/, "");
    let title = base;
    if (split.sections[0].header && sections[0].title) {
      title = sections[0].title;
      sections[0].title = "";
    }
    const failed = sections.find((s) => s.result.error);
    const report = {
      title,
      file: app.baseName(f.name),
      author: prefs.get("author", ""),
      date: new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" }),
      sections,
    };
    last = { report, html: "", filename: `${base}-report.html` };
    $("report-author").value = report.author;
    render();
    const figures = sections.reduce((n, s) => n + ((s.result.figures || []).length + (s.result.animations || []).length), 0);
    $("report-summary").textContent =
      `${sections.length} section${sections.length === 1 ? "" : "s"}, ${figures} figure${figures === 1 ? "" : "s"}` + (failed ? " — stopped at an error" : "");
    $("report-summary").classList.toggle("error-text", !!failed);
    $("dlg-report").showModal();
    app.printInfo(failed ? `Report built, but a section ended with an error: ${failed.result.error}` : "Report built.", failed ? "note" : "info");
    return last;
  }

  $("btn-publish").addEventListener("click", publish);
  $("report-author").addEventListener("input", () => {
    prefs.set("author", $("report-author").value.trim());
    if (last) render();
  });
  $("report-download").addEventListener("click", () => {
    if (!last) return;
    downloadBlob(new Blob([last.html], { type: "text/html;charset=utf-8" }), last.filename);
    toast(`Downloaded ${last.filename}`);
  });
  $("report-print").addEventListener("click", () => {
    const w = $("report-frame").contentWindow;
    if (!w) return;
    w.focus();
    w.print();
  });

  Object.assign(app, { publish, lastReport: () => last });
  void state;
}
