// The App panel: shows app windows built with uifigure / uicontrol and sends what the user
// does in them to the engine. The drawing itself is done by uirender.js (shared with the
// `mylab app.m` local server); this module connects it to the web app.
//
// - After a run, a result with `ui` (the component tree) opens the panel.
// - Each user action is one "uiEvent" request; the answer says which components changed.
//   The session stays alive between events, so callbacks keep their state.
// - "App mode" hides the editor and shows only the app (share links with &app=1).

import "./uirender.js";
import { $, el, icon, toast } from "./ui.js";

const MAX_LOG = 4000;

function addStylesheet(href) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  document.head.append(el("link", { rel: "stylesheet", href }));
}

/** "name;" when the file is a function without inputs or a class: Run can call it. */
export function callExpression(path, code) {
  const base = path.slice(path.lastIndexOf("/") + 1).replace(/\.m$/, "");
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(base)) return null;
  for (const raw of code.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("%")) continue;
    if (/^classdef\b/.test(line)) return `${base};`;
    const m = /^function\s+(?:(?:\[[^\]]*\]|[A-Za-z_]\w*)\s*=\s*)?([A-Za-z_]\w*)\s*(\(\s*\))?\s*(?:%.*)?$/.exec(line);
    return m ? `${base};` : null;
  }
  return null;
}

export function initUiApp(app) {
  addStylesheet("css/uirender.css");
  addStylesheet("css/apps.css");

  let view = null; // the MylabUI view, while an app is shown
  let hasApp = false;
  let appMode = false;
  let lastRun = null; // { file } of the run that started the app
  let slowTimer = null;

  /* ---------- the panel ---------- */

  const title = el("h2", { class: "app-view-title", id: "app-view-title" }, "App");
  const status = el("span", { class: "app-view-status muted small", role: "status", "aria-live": "polite" });
  const btnStop = el("button", { type: "button", class: "btn sm danger", id: "app-stop", hidden: true, title: "Stop the running callback (restarts the engine and closes the app)", onclick: () => app.stopRun() }, icon("stop"), "Stop");
  const btnRestart = el("button", { type: "button", class: "btn sm ghost", id: "app-restart", title: "Run the file again: the app starts fresh", onclick: () => restart() }, icon("restart"), el("span", { class: "hide-sm" }, "Restart"));
  const btnOnly = el("button", { type: "button", class: "btn sm ghost", id: "app-only", title: "Show only the app and hide the editor", onclick: () => setAppMode(true) }, icon("expand"), el("span", { class: "hide-sm" }, "App only"));
  const btnCode = el("button", { type: "button", class: "btn sm ghost", id: "app-code", hidden: true, title: "Show the editor and the code of this app", onclick: () => setAppMode(false) }, icon("pencil"), "Edit the code");
  const btnClose = el("button", { type: "button", class: "btn icon sm ghost", id: "app-hide", title: "Hide the app (it keeps running; the App button shows it again)", "aria-label": "Hide the app", onclick: () => hide() }, icon("close"));
  const stage = el("div", { class: "app-view-stage", id: "app-stage" });
  const log = el("pre", { class: "app-view-log", id: "app-log", role: "log", "aria-label": "Output of the app", tabindex: "0", hidden: true });
  const panel = el(
    "section",
    { id: "app-view", class: "app-view", hidden: true, "aria-labelledby": "app-view-title" },
    el("header", { class: "app-view-bar" }, title, status, el("span", { class: "spacer" }), btnStop, btnRestart, btnOnly, btnCode, btnClose),
    el("div", { class: "app-view-scroll" }, stage, log)
  );
  document.body.append(panel);

  const btnShow = el("button", { id: "btn-app", class: "btn", hidden: true, type: "button", "aria-pressed": "false", title: "Show or hide the running app", onclick: () => (panel.hidden ? show() : hide()) }, icon("expand"), el("span", { class: "hide-sm" }, "App"));
  $("btn-publish").after(btnShow);

  panel.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !appMode && !e.defaultPrevented && !e.target.closest(".mui-overlay, .mui-menu")) hide();
  });

  function show() {
    if (!hasApp) return;
    panel.hidden = false;
    btnShow.setAttribute("aria-pressed", "true");
    document.body.dataset.app = "open";
  }

  function hide() {
    if (appMode) return;
    panel.hidden = true;
    btnShow.setAttribute("aria-pressed", "false");
    delete document.body.dataset.app;
    if (app.editor && app.state.active) app.editor.focus();
  }

  function setStatus(text) {
    status.textContent = text || "";
  }

  function writeLog(text, isError = false) {
    if (!text) return;
    const span = el("span", { class: isError ? "err" : "" }, text.endsWith("\n") ? text : text + "\n");
    log.append(span);
    while (log.textContent.length > MAX_LOG && log.firstChild) log.firstChild.remove();
    log.hidden = false;
    log.scrollTop = log.scrollHeight;
  }

  function setAppMode(on) {
    appMode = on;
    if (on) document.body.dataset.appmode = "1";
    else delete document.body.dataset.appmode;
    btnOnly.hidden = on;
    btnCode.hidden = !on;
    btnClose.hidden = on;
    if (on) panel.hidden = false;
    else if (!hasApp) hide();
    if (!on && app.editor) setTimeout(() => app.editor.refresh(), 0);
  }

  /* ---------- the app ---------- */

  function clear() {
    if (view) view.destroy();
    view = null;
    hasApp = false;
    btnShow.hidden = true;
    stage.textContent = "";
    log.textContent = "";
    log.hidden = true;
    setStatus("");
    btnStop.hidden = true;
    clearTimeout(slowTimer);
  }

  async function send(event) {
    clearTimeout(slowTimer);
    slowTimer = setTimeout(() => {
      setStatus("Working…");
      btnStop.hidden = false;
    }, 1500);
    try {
      return await app.engine.uiEvent(event);
    } finally {
      clearTimeout(slowTimer);
      if (!btnStop.hidden || status.textContent === "Working…") setStatus("");
      btnStop.hidden = true;
    }
  }

  function onResult(res) {
    if (res.figures && res.figures.length) app.addFigures(res.figures, { kind: "app", label: "app" });
    if (res.animations && res.animations.length) app.addFigures(res.animations, { kind: "app", label: "app" }, "gif");
    if (res.variables) app.renderVariables(res.variables);
    for (const d of res.dialogs || []) toast((d.title ? d.title + ": " : "") + d.message, d.icon === "error" ? "error" : "");
    if (res.files && app.applyWritten) app.applyWritten(res.files);
    if (res.internal) console.warn(res.internal);
  }

  function setTree(tree) {
    if (!view) {
      stage.textContent = "";
      view = globalThis.MylabUI.mount(stage, {
        send,
        onResult,
        onOutput: (text) => {
          app.printOut(text);
          writeLog(text);
        },
        onError: (text) => {
          if (text === "Stopped.") return;
          app.printError({ message: text, where: "app callback" });
          writeLog("Error: " + text, true);
          if (panel.hidden) toast("The app reported an error: " + globalThis.MylabUI.firstLine(text), "error");
        },
        onEmpty: () => {
          clear();
          if (appMode) setAppMode(false);
          hide();
        },
      });
    }
    hasApp = true;
    btnShow.hidden = false;
    view.setTree(tree);
    if (!view) return; // the tree was empty
    const names = (tree.figures || []).map((f) => f.props.name).filter(Boolean);
    title.textContent = names[0] || "App";
  }

  async function restart() {
    if (!lastRun || !app.state.files.has(lastRun.file)) return toast("Open the file of the app and press Run.");
    if (app.state.active !== lastRun.file) app.openFile(lastRun.file);
    await app.runActiveFile();
  }

  async function closeApp() {
    try {
      await app.engine.uiClose();
    } catch {
      /* the engine is restarting */
    }
    clear();
    if (appMode) setAppMode(false);
    hide();
  }

  app.on("runStarting", (ctx) => {
    // Running a file again starts its app afresh: close the windows of the previous run first.
    if (ctx && ctx.kind === "file" && hasApp) app.engine.uiClose().catch(() => {});
  });

  app.on("runFinished", (res, error, ctx) => {
    if (!res) return;
    if (res.ui) {
      if (ctx && ctx.kind === "file" && ctx.file) lastRun = { file: ctx.file };
      if (ctx && ctx.kind === "file") {
        log.textContent = "";
        log.hidden = true;
      }
      setTree(res.ui);
      if (hasApp && (ctx ? ctx.kind !== "command" : true)) show();
      setStatus("");
    } else if (hasApp) {
      clear();
      if (appMode) setAppMode(false);
      hide();
    } else if (appMode) {
      setAppMode(false);
      toast(error ? "The app could not start. The error is in the command window." : "This file did not open an app window.", "error");
    }
  });

  // Clearing the workspace (Reset) also closes the app windows in the engine.
  app.on("variables", (list) => {
    if (hasApp && view && !list.length && app.engine.ready) {
      app.engine.uiEvent({}).then((res) => view && res.update && view.apply(res.update), () => {});
    }
  });

  app.on("engineReady", () => {
    app.engine.addEventListener("restarting", () => {
      clear();
      hide();
    });
  });
  // The engine object exists once main() has created it; "restarting" is also needed before the first "ready".
  queueMicrotask(() => {
    if (!app.engine) return;
    app.engine.addEventListener("progress", (e) => appMode && !hasApp && setStatus(e.detail.label));
    app.engine.addEventListener("fatal", () => appMode && setAppMode(false));
  });

  /** Share links with &app=1: show only the app, and run the file as soon as the engine is ready. */
  function openAsApp(projectId) {
    setAppMode(true);
    title.textContent = "App";
    setStatus("Starting the app…");
    stage.textContent = "";
    stage.append(el("p", { class: "app-view-wait muted" }, "The app is loading. The first visit takes a little longer."));
    app.engine.whenReady().then(
      async () => {
        if (app.state.project.id !== projectId) return setAppMode(false);
        const res = await app.runActiveFile();
        if (!res) setAppMode(false);
      },
      () => setAppMode(false)
    );
  }

  /* ---------- "Open as app" in the share window ---------- */

  const shareRun = $("share-run");
  if (shareRun) {
    const box = el("input", { type: "checkbox", id: "share-app" });
    const row = el("label", { class: "check" }, box, " Open as app: run it and show only the app window (for files that use uifigure or uicontrol)");
    shareRun.closest("label").after(row);
    box.addEventListener("change", () => {
      shareRun.disabled = box.checked;
      shareRun.dispatchEvent(new Event("change"));
    });
    $("btn-share").addEventListener("click", () => {
      box.checked = !!app.shareAsAppNext;
      app.shareAsAppNext = false;
      shareRun.disabled = box.checked;
      if (box.checked) shareRun.dispatchEvent(new Event("change"));
    });
  }

  app.commands.register([
    { id: "app.show", group: "App", title: "Show or hide the running app", run: () => (panel.hidden ? show() : hide()), when: () => hasApp && !appMode },
    { id: "app.restart", group: "App", title: "Restart the app (run its file again)", run: restart, when: () => hasApp && !!lastRun },
    { id: "app.close", group: "App", title: "Close the app windows", run: closeApp, when: () => hasApp },
    { id: "app.only", group: "App", title: "Show only the app (hide the editor)", run: () => setAppMode(true), when: () => hasApp && !appMode },
    { id: "app.code", group: "App", title: "Show the editor again (leave app-only view)", run: () => setAppMode(false), when: () => appMode },
    {
      id: "app.share", group: "App", title: "Share a link that opens this file as an app",
      run: () => {
        app.shareAsAppNext = true;
        $("btn-share").click();
      },
      when: () => !!app.state.active,
    },
  ]);

  Object.assign(app, {
    callableFile: callExpression,
    openAsApp,
    showApp: show,
    hideApp: hide,
    closeApp,
    setAppMode,
    get appRunning() {
      return hasApp;
    },
  });
}
