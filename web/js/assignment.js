// Assignments: load a bundle made with `mylab-grade export`, show the problems beside the
// editor and check the student's answers by running the bundle's tests in the browser.
//
// Bundle format ("mylab-assignment-1"):
//   { format, title, description (markdown),
//     problems: [{ id, title, points, function, prompt (markdown),
//                  starter: { "<file.m>": "<code>" }, data: { "<name>": "<text>" },
//                  tests: [{ name, points, code }] }] }
//
// A test passes when its code runs without an error. Every test starts from a clean session
// in a folder that holds the student's files for that problem plus the problem's data files.

import { $, el, icon, toast, downloadBlob, copyText } from "./ui.js";
import { renderMarkdown } from "./markdown.js";
import { encodePayload, decodePayload } from "./codec.js";
import { LONG_LINK } from "./share.js";

export const FORMAT = "mylab-assignment-1";
export const EXAMPLE_URL = "samples/intro-assignment.json";

const isObject = (v) => v && typeof v === "object" && !Array.isArray(v);

/** Check a bundle and fill in defaults. Throws an Error with a message for the user. */
export function validateBundle(raw) {
  if (!isObject(raw)) throw new Error("This file is not a mylab assignment (it is not a JSON object).");
  if (raw.format !== FORMAT) {
    throw new Error(
      typeof raw.format === "string" && raw.format.startsWith("mylab-assignment-")
        ? `This assignment uses the format “${raw.format}”, which this version of mylab can't read. Reload the page to get the newest version.`
        : "This file is not a mylab assignment (its “format” should be “mylab-assignment-1”)."
    );
  }
  if (!Array.isArray(raw.problems) || !raw.problems.length) throw new Error("This assignment has no problems in it.");
  const seen = new Set();
  const folders = new Set();
  const problems = raw.problems.map((p, i) => {
    if (!isObject(p)) throw new Error(`Problem ${i + 1} of the assignment is damaged.`);
    const id = String(p.id ?? `problem${i + 1}`);
    if (seen.has(id)) throw new Error(`Two problems share the id “${id}”.`);
    seen.add(id);
    let folder = id.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[_.]+|[_.]+$/g, "") || `problem${i + 1}`;
    while (folders.has(folder.toLowerCase())) folder += "_";
    folders.add(folder.toLowerCase());
    const textMap = (obj) => {
      const out = {};
      for (const [name, text] of Object.entries(isObject(obj) ? obj : {})) {
        const clean = String(name).replace(/\\/g, "/").split("/").filter((s) => s && s !== "." && s !== ".." && !s.startsWith(".")).join("/");
        if (clean) out[clean] = String(text ?? "");
      }
      return out;
    };
    const tests = (Array.isArray(p.tests) ? p.tests : []).filter(isObject).map((t, k) => ({
      name: String(t.name ?? `Test ${k + 1}`),
      points: Number.isFinite(Number(t.points)) ? Math.max(0, Number(t.points)) : 0,
      code: String(t.code ?? ""),
    }));
    const testSum = tests.reduce((s, t) => s + t.points, 0);
    return {
      id,
      folder,
      title: String(p.title ?? id),
      points: Number.isFinite(Number(p.points)) ? Math.max(0, Number(p.points)) : testSum,
      function: String(p.function ?? ""),
      prompt: String(p.prompt ?? ""),
      starter: textMap(p.starter),
      data: textMap(p.data),
      tests,
    };
  });
  return { format: FORMAT, title: String(raw.title ?? "Assignment"), description: String(raw.description ?? ""), problems };
}

function bundleKey(bundle) {
  let h = 5381;
  const text = bundle.title + "\n" + bundle.problems.map((p) => p.id).join("\n");
  for (let i = 0; i < text.length; i++) h = ((h << 5) + h + text.charCodeAt(i)) >>> 0;
  return "asg:" + h.toString(36);
}

const fmtPoints = (n) => String(Math.round(n * 100) / 100);

/**
 * Points earned on the tests that can be run here. A problem's `points` is its full total,
 * which may include hidden tests, so the visible tests can add up to less than that.
 * Returns { earned, visible, testPoints, counted } (counted: tests carry points; otherwise
 * earned/visible count tests).
 */
function score(problem, results) {
  const tests = problem.tests;
  const sum = tests.reduce((s, t) => s + t.points, 0);
  const counted = sum > 0;
  const w = counted ? tests.map((t) => t.points) : tests.map(() => 1);
  let earned = 0;
  tests.forEach((t, i) => {
    if (results && results[i] && results[i].status === "pass") earned += w[i];
  });
  return { earned, visible: counted ? sum : tests.length, testPoints: w, counted };
}

/** The file a student completes: <function>.m when the bundle names one, else the first starter file. */
function mainFile(p) {
  const names = Object.keys(p.starter);
  const main = names.find((n) => n === p.function + ".m") || names.find((n) => n.endsWith(".m")) || names[0];
  return main ? `${p.folder}/${main}` : null;
}

export function initAssignments(app) {
  const { state } = app;
  const config = { testTimeout: 15000 };
  let checking = null; // { cancelled }

  const bundle = () => (state.project && state.project.kind === "assignment" && state.project.meta ? state.project.meta.bundle : null);
  const results = () => (state.project.meta.results ||= { problems: {} });

  /* ---------- opening ---------- */

  async function openAssignmentBundle(raw, { source = "" } = {}) {
    const b = validateBundle(raw);
    const key = bundleKey(b);
    let project = state.projects.find((p) => p.kind === "assignment" && p.meta && p.meta.key === key && !p.temp);
    const wanted = [];
    for (const p of b.problems) {
      for (const [name, text] of Object.entries(p.starter)) wanted.push({ path: `${p.folder}/${name}`, content: text });
      for (const [name, text] of Object.entries(p.data)) wanted.push({ path: `${p.folder}/${name}`, content: text });
    }
    const firstFile = b.problems.map(mainFile).find(Boolean) || null;
    if (project) {
      // The student already started: keep every file they have and only add what is missing.
      if (state.project.id !== project.id && !(await app.switchProject(project.id))) return null;
      const missing = wanted.filter((f) => !state.files.has(f.path));
      await app.writeFiles(missing);
      state.project.meta.bundle = b;
      if (source) state.project.meta.source = source;
      await app.saveProjectRecord();
      renderPanel();
      toast(`Opened “${b.title}”. Your earlier work is kept.`);
    } else {
      project = await app.createProject({ name: b.title, kind: "assignment", meta: { key, bundle: b, source, results: { problems: {} } }, files: wanted, open: firstFile });
      toast(`Opened the assignment “${b.title}”.`);
    }
    return project;
  }

  function openAssignmentText(text, opts) {
    let raw;
    try {
      raw = JSON.parse(text);
    } catch {
      throw new Error("This file is not a mylab assignment (it is not valid JSON).");
    }
    return openAssignmentBundle(raw, opts);
  }

  async function fetchBundle(url) {
    let target;
    try {
      target = new URL(url, location.href);
    } catch {
      throw new Error("The assignment link is not a valid web address.");
    }
    const local = target.origin === location.origin || /^(localhost|127\.0\.0\.1)$/.test(target.hostname);
    if (target.protocol !== "https:" && !local) throw new Error("Assignment links must start with https://");
    let resp;
    try {
      resp = await fetch(target.href, { credentials: "omit" });
    } catch {
      throw new Error(
        `Couldn't download the assignment from ${target.host}. Check your connection. The site must also allow other pages to read the file (CORS); GitHub “raw” links and GitHub Pages do.`
      );
    }
    if (!resp.ok) throw new Error(`Couldn't download the assignment (${target.host} answered HTTP ${resp.status}). Ask your teacher to check the link.`);
    return resp.text();
  }

  /** #assignment=<url>  or  #assignmentdata=<compressed bundle> */
  async function openAssignmentFromLink(params) {
    if ($("dlg-assignment").open) $("dlg-assignment").close();
    if (params.assignmentdata) {
      const raw = await decodePayload(params.assignmentdata);
      return !!(await openAssignmentBundle(raw, { source: "link" }));
    }
    const text = await fetchBundle(params.assignment);
    return !!(await openAssignmentText(text, { source: params.assignment }));
  }

  async function openExampleAssignment() {
    try {
      await openAssignmentText(await fetchBundle(EXAMPLE_URL), { source: EXAMPLE_URL });
    } catch (e) {
      toast(e.message, "error");
    }
  }

  function openAssignmentDialog() {
    $("asg-url").value = "";
    $("asg-open-error").hidden = true;
    $("dlg-assignment").showModal();
  }

  async function guarded(fn) {
    try {
      const r = await fn();
      if ($("dlg-assignment").open) $("dlg-assignment").close();
      return r;
    } catch (e) {
      if ($("dlg-assignment").open) {
        $("asg-open-error").textContent = e.message;
        $("asg-open-error").hidden = false;
      } else {
        toast(e.message, "error");
      }
      return null;
    }
  }

  /* ---------- the panel ---------- */

  function testRow(problem, i, r, pts) {
    const t = problem.tests[i];
    const status = r ? r.status : "pending";
    const label = { pass: "Passed", fail: "Failed", timeout: "Timed out", pending: "Not checked yet", running: "Running…" }[status];
    const row = el(
      "li",
      { class: "asg-test " + status, "data-test": t.name, "data-status": status },
      el("span", { class: "asg-mark", title: label, "aria-label": label }, icon({ pass: "check", fail: "close", timeout: "clock", pending: "dot", running: "dot" }[status])),
      el("span", { class: "asg-test-name" }, t.name),
      pts === null ? null : el("span", { class: "asg-test-points muted small" }, `${status === "pass" ? fmtPoints(pts) : "0"} / ${fmtPoints(pts)}`)
    );
    if (r && r.status !== "pass" && r.message) {
      const file = r.file ? `${problem.folder}/${r.file}` : null;
      const where = r.line ? (r.file ? `line ${r.line} of ${r.file}` : `line ${r.line} of the test`) : "";
      const msg = el("div", { class: "asg-test-msg" }, r.message, where ? el("span", { class: "muted" }, ` (${where})`) : null);
      if (file && r.line && state.files.has(file)) {
        msg.append(
          " ",
          el(
            "button",
            {
              class: "linklike",
              type: "button",
              onclick: () => {
                app.openFile(file);
                app.markErrorLine(file, r.line);
                app.editor.setCursor({ line: r.line - 1, ch: 0 });
                app.editor.scrollIntoView({ line: r.line - 1, ch: 0 }, 80);
              },
            },
            "Show the line"
          )
        );
      }
      row.append(msg);
    }
    return row;
  }

  function renderPanel() {
    const b = bundle();
    const panel = $("asg-panel");
    const gutter = $("asg-gutter");
    panel.hidden = gutter.hidden = !b;
    $("editor-row").classList.toggle("with-asg", !!b);
    if (!b) return;
    const res = results();
    const body = $("asg-body");
    const scroll = body.scrollTop;
    const open = new Set([...body.querySelectorAll("details.asg-problem[open]")].map((d) => d.dataset.problem));
    const firstRender = !body.querySelector("details.asg-problem") || body.dataset.key !== state.project.id;
    body.dataset.key = state.project.id;
    body.innerHTML = "";

    let earned = 0;
    let visible = 0;
    let total = 0;
    let checked = 0;
    let allCounted = true;
    const sections = b.problems.map((p, n) => {
      const r = res.problems[p.id];
      const s = score(p, r && r.tests);
      total += p.points;
      if (p.tests.length) {
        visible += s.visible;
        allCounted = allCounted && s.counted;
        if (r) {
          earned += s.earned;
          checked++;
        }
      }
      const unit = s.counted ? "points" : "tests";
      const allPass = !!r && p.tests.length > 0 && r.tests.every((t) => t && t.status === "pass");
      const files = Object.keys(p.starter).map((name) => `${p.folder}/${name}`);
      const main = mainFile(p);
      const activeHere = state.active && state.active.startsWith(p.folder + "/");
      const pill = !p.tests.length
        ? "graded after submission"
        : r
          ? `visible tests: ${fmtPoints(s.earned)} of ${fmtPoints(s.visible)} ${unit}`
          : `${fmtPoints(s.visible)} ${unit} in visible tests`;
      return el(
        "details",
        { class: "asg-problem" + (allPass ? " done" : ""), "data-problem": p.id, open: firstRender ? activeHere || n === 0 : open.has(p.id) },
        el(
          "summary",
          {},
          el("span", { class: "asg-num" }, String(n + 1)),
          el("span", { class: "asg-problem-title" }, p.title, el("span", { class: "asg-worth muted small" }, ` · ${fmtPoints(p.points)} pts`)),
          el("span", { class: "pill small asg-points" + (allPass ? " ok" : ""), "data-points": "" }, pill)
        ),
        el("div", { class: "asg-prompt md", html: renderMarkdown(p.prompt) }),
        el(
          "div",
          { class: "asg-problem-actions" },
          files.map((path) =>
            el("button", { class: "chip" + (path === main ? " main" : ""), type: "button", title: path === main ? `Open ${path} (the file to complete)` : `Open ${path}`, onclick: () => app.openFile(path) }, app.baseName(path))
          ),
          p.tests.length ? el("button", { class: "btn sm", type: "button", "data-check": p.id, disabled: !!checking, onclick: () => check([p.id]) }, "Check this problem") : null
        ),
        p.tests.length
          ? el("ul", { class: "asg-tests" }, p.tests.map((_, i) => testRow(p, i, r && r.tests[i], s.counted ? s.testPoints[i] : null)))
          : el("p", { class: "asg-hidden-note small" }, "Graded after submission: this problem has no tests that can be run here. Check your work against the examples above.")
      );
    });

    const unit = allCounted ? "points" : "tests";
    const when = res.when ? new Date(res.when).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "";
    const testable = b.problems.filter((p) => p.tests.length).length;
    const hiddenPart = allCounted && total > visible;
    body.append(
      el("h3", { class: "asg-title" }, b.title),
      el(
        "div",
        { class: "asg-score", id: "asg-score", "data-earned": fmtPoints(earned), "data-visible": fmtPoints(visible), "data-total": fmtPoints(total) },
        el("strong", {}, checked ? `Visible tests: ${fmtPoints(earned)} of ${fmtPoints(visible)} ${unit}` : `Visible tests: ${fmtPoints(visible)} ${unit}`),
        el(
          "span",
          { class: "muted small" },
          ` · the assignment is worth ${fmtPoints(total)} points` +
            (checked ? (checked < testable ? ` · ${checked} of ${testable} problems checked` : "") + (when ? ` · last checked ${when}` : "") : " · not checked yet")
        )
      ),
      el(
        "div",
        { class: "asg-actions" },
        el("button", { class: "btn primary", id: "asg-check", type: "button", disabled: !!checking, onclick: () => check() }, icon("check"), checking ? "Checking…" : "Check my answers"),
        el("button", { class: "btn", id: "asg-download", type: "button", onclick: downloadSubmission }, icon("zip"), "Download my submission (.zip)")
      ),
      el(
        "p",
        { class: "muted small asg-hint" },
        "These checks run in your browser and you can repeat them as often as you like. Passing them is not your final grade: " +
          (hiddenPart ? `the remaining ${fmtPoints(total - visible)} points come from hidden tests that run after you hand in.` : "your teacher may use extra hidden tests after you hand in.")
      ),
      b.description ? el("div", { class: "asg-desc md", html: renderMarkdown(b.description) }) : null,
      ...sections
    );
    body.scrollTop = scroll;
  }

  /* ---------- checking ---------- */

  function problemFiles(p) {
    const files = [];
    const prefix = p.folder + "/";
    for (const path of state.files.keys()) {
      if (path.startsWith(prefix)) files.push({ path: path.slice(prefix.length), content: app.liveContent(path) });
    }
    // The data files always come from the bundle, so a test sees the data the teacher wrote.
    for (const [name, text] of Object.entries(p.data)) {
      const i = files.findIndex((f) => f.path === name);
      if (i >= 0) files[i].content = text;
      else files.push({ path: name, content: text });
    }
    return files;
  }

  function withTimeout(promise, ms) {
    let timer;
    const timeout = new Promise((resolve) => (timer = setTimeout(() => resolve({ timedOut: true }), ms)));
    return Promise.race([promise.then((value) => ({ value })), timeout]).finally(() => clearTimeout(timer));
  }

  async function check(onlyIds = null) {
    const b = bundle();
    if (!b || checking || !app.canRun()) return;
    app.exitStep && app.exitStep();
    const run = (checking = { cancelled: false });
    await app.flushAll();
    const projectId = state.project.id;
    const res = results();
    const todo = b.problems.filter((p) => p.tests.length && (!onlyIds || onlyIds.includes(p.id)));
    if (!todo.length) {
      toast("There are no tests to run here: this work is graded after you hand it in.");
      checking = null;
      return;
    }
    for (const p of todo) res.problems[p.id] = { tests: p.tests.map(() => ({ status: "pending" })) };
    app.printEcho("", onlyIds ? `check problem ${todo.map((p) => p.title).join(", ")}` : "check my answers");
    app.setRunning(true, "Checking");
    renderPanel();
    try {
      for (const p of todo) {
        const files = problemFiles(p);
        for (let i = 0; i < p.tests.length; i++) {
          if (run.cancelled || state.project.id !== projectId) return;
          res.problems[p.id].tests[i] = { status: "running" };
          renderPanel();
          let outcome;
          try {
            outcome = await withTimeout(app.engine.test(files, p.tests[i].code), config.testTimeout);
          } catch (e) {
            if (String(e.message) === "Stopped.") {
              // The user pressed Stop.
              run.cancelled = true;
              res.problems[p.id].tests[i] = { status: "fail", message: "Stopped before this test finished." };
              for (let k = i + 1; k < p.tests.length; k++) res.problems[p.id].tests[k] = { status: "pending" };
              return;
            }
            outcome = { value: { error: e.message } };
          }
          if (outcome.timedOut) {
            res.problems[p.id].tests[i] = {
              status: "timeout",
              message: `Timed out: the test was still running after ${Math.round(config.testTimeout / 1000)} seconds. Look for a loop that never ends.`,
            };
            renderPanel();
            // Python can't be interrupted from outside, so start a fresh engine and carry on.
            app.engine.restart();
            app.setRunning(true, "Checking");
            app.renderVariables([]);
            app.printInfo(`“${p.tests[i].name}” timed out, so the engine was restarted (your workspace was cleared). Continuing with the next test.`, "note");
            try {
              await app.engine.whenReady();
            } catch {
              run.cancelled = true;
              return;
            }
            app.setRunning(true, "Checking");
          } else {
            const out = outcome.value || {};
            res.problems[p.id].tests[i] = out.error
              ? { status: "fail", message: out.error, line: out.error_line || null, file: out.error_file || null }
              : { status: "pass" };
          }
        }
      }
    } finally {
      checking = null;
      if (state.project.id === projectId) {
        res.when = Date.now();
        for (const p of todo) {
          // Never keep a "running" marker around.
          res.problems[p.id].tests = res.problems[p.id].tests.map((t) => (t.status === "running" ? { status: "pending" } : t));
        }
        await app.saveProjectRecord();
        renderPanel();
        const passed = todo.reduce((s, p) => s + res.problems[p.id].tests.filter((t) => t.status === "pass").length, 0);
        const count = todo.reduce((s, p) => s + p.tests.length, 0);
        if (!run.cancelled) app.printInfo(`Checked: ${passed} of ${count} visible test${count === 1 ? "" : "s"} passed. See the Assignment panel for details. Hidden tests may still apply.`);
      }
      if (app.engine.ready) app.setRunning(false);
      else {
        $("btn-run").hidden = false;
        $("btn-stop").hidden = true;
      }
    }
  }

  /* ---------- handing in ---------- */

  async function downloadSubmission() {
    const b = bundle();
    if (!b) return;
    if (typeof JSZip === "undefined") {
      toast("The zip library hasn't loaded yet. Check your connection and try again.", "error");
      return;
    }
    await app.flushAll();
    const zip = new JSZip();
    let count = 0;
    for (const p of b.problems) {
      const prefix = p.folder + "/";
      for (const path of app.sortedPaths()) {
        if (!path.startsWith(prefix)) continue;
        const rel = path.slice(prefix.length);
        if (rel in p.data && app.liveContent(path) === p.data[rel]) continue; // unchanged data file from the bundle
        zip.file(path, app.liveContent(path));
        count++;
      }
    }
    const slug = b.title.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-|-$/g, "") || "assignment";
    const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
    downloadBlob(blob, `${slug}-submission.zip`);
    toast(`Downloaded ${slug}-submission.zip with ${count} file${count === 1 ? "" : "s"}. Upload it where your teacher asked.`);
  }

  async function copyAssignmentLink() {
    const b = bundle();
    if (!b) return;
    const src = state.project.meta.source;
    let link;
    if (src && /^https:\/\//.test(src)) link = `${location.origin}${location.pathname}#assignment=${encodeURIComponent(src)}`;
    else if (src === EXAMPLE_URL) link = `${location.origin}${location.pathname}#assignment=${EXAMPLE_URL}`;
    else {
      const clean = { ...b, problems: b.problems.map(({ folder, ...p }) => p) };
      link = `${location.origin}${location.pathname}#assignmentdata=${await encodePayload(clean)}`;
    }
    const ok = await copyText(link);
    if (!ok) toast("Couldn't copy the link.", "error");
    else if (link.length > LONG_LINK)
      toast(`Link copied, but it is long (${link.length.toLocaleString()} characters) and some chat apps cut long links. Better: put the bundle file on GitHub and share …#assignment=<its raw https address>.`, "error");
    else toast("Link to this assignment copied. Students who open it get the starter files and can check their answers.");
  }

  $("asg-share").addEventListener("click", copyAssignmentLink);
  $("asg-file-btn").addEventListener("click", () => $("asg-file-input").click());
  $("asg-file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (file) guarded(async () => openAssignmentText(await file.text(), { source: file.name }));
  });
  const openUrl = () => {
    const url = $("asg-url").value.trim();
    if (url) guarded(async () => openAssignmentText(await fetchBundle(url), { source: url }));
  };
  $("asg-url-open").addEventListener("click", openUrl);
  $("asg-url").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      openUrl();
    }
  });
  $("asg-example").addEventListener("click", (e) => {
    e.preventDefault();
    guarded(async () => openAssignmentText(await fetchBundle(EXAMPLE_URL), { source: EXAMPLE_URL }));
  });

  app.onProjectChanged = () => {
    if (checking) checking.cancelled = true;
    renderPanel();
    app.editor.refresh();
  };

  Object.assign(app, {
    openAssignmentBundle,
    openAssignmentText: (text, opts) => guarded(() => openAssignmentText(text, opts)),
    openAssignmentFromLink,
    openAssignmentDialog,
    openExampleAssignment,
    checkAssignment: check,
    assignmentConfig: config,
  });
}
