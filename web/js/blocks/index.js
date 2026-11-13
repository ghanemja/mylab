// mylab Blocks in the web app: opens .mlb.json files in the diagram editor, runs them in the
// engine and shows the scopes. Everything the rest of the app needs is in initBlocks(app).

import { $, el, icon, toast, openMenu, closeMenu } from "../ui.js";
import { CATALOG } from "./catalog.js";
import { BLOCK_EXAMPLES } from "./examples.js";
import { isModelPath, newModel, serializeModel, findType, portCounts, systemAt, EXT } from "./model.js";
import { DiagramEditor } from "./editor.js";
import { ScopeDock, formatNumber } from "./scope.js";
import { paletteIcon, CATEGORY_CLASS } from "./draw.js";
import { paramsDialog, settingsDialog, pickDialog, keysDialog, textDialog, noticeDialog, portsWith } from "./dialogs.js";

export function initBlocks(app) {
  const { state } = app;
  const css = el("link", { rel: "stylesheet", href: new URL("./blocks.css", import.meta.url).href });
  document.head.appendChild(css);

  /* ---------- the view ---------- */

  const btn = (id, title, ic, label, onclick, extra = {}) =>
    el("button", { id, class: "btn sm" + (label ? "" : " icon") + (extra.cls ? " " + extra.cls : " ghost"), type: "button", title, "aria-label": label ? null : title, onclick, ...(extra.attrs || {}) }, ic ? icon(ic) : null, label ? el("span", { class: extra.keep ? "" : "bd-lbl" }, label) : null);

  const glyph = (id, title, text, onclick) => el("button", { id, class: "btn sm icon ghost", type: "button", title, "aria-label": title, onclick }, el("span", { class: "bd-glyph", "aria-hidden": "true" }, text));

  const stopInput = el("input", { id: "bd-stoptime", class: "bd-stop mono", type: "text", spellcheck: "false", autocomplete: "off", "aria-label": "Stop time in seconds", title: "Stop time (s). Press Enter to run." });
  const runBtn = btn("bd-run", "Run the model (F5 or Ctrl+Enter)", "play", "Run", () => runModel(), { cls: "primary", keep: true });
  const stopBtn = btn("bd-stop-run", "Stop the simulation (restarts the engine, which clears the workspace)", "stop", "Stop", () => app.stopRun(), { cls: "danger", keep: true, attrs: { hidden: true } });
  const progress = el("span", { id: "bd-progress", class: "bd-progress small muted", role: "status", "aria-live": "polite" });
  const problemsBtn = btn("bd-problems", "Problems in this model", "warn", "0", (e) => showProblems(e.currentTarget), { keep: true, attrs: { hidden: true } });
  const zoomLabel = el("button", { id: "bd-zoom", class: "btn sm ghost bd-zoomlbl", type: "button", title: "Zoom: click for 100 %", onclick: () => ed.zoomBy(1 / ed.view.k) }, "100%");
  const toolbar = el(
    "div",
    { class: "bd-toolbar", role: "toolbar", "aria-label": "Diagram tools" },
    runBtn, stopBtn,
    el("label", { class: "bd-stoplabel small", for: "bd-stoptime" }, "Stop time"), stopInput,
    btn("bd-settings", "Simulation settings: solver, step size, tolerances", "gear", "", () => editSettings()),
    progress,
    el("span", { class: "bd-sep" }),
    glyph("bd-undo", "Undo (Ctrl+Z)", "↶", () => ed.undo()),
    glyph("bd-redo", "Redo (Ctrl+Y)", "↷", () => ed.redo()),
    el("span", { class: "bd-sep" }),
    btn("bd-add", "Add a block by name (A)", "plus", "Block", () => quickInsert(null, null)),
    btn("bd-arrange", "Arrange the blocks automatically (Ctrl+Shift+L)", "format", "Arrange", () => ed.layout()),
    btn("bd-more", "Align, rotate, subsystem and more", "menu", "", (e) => editMenu(e.currentTarget)),
    el("span", { class: "bd-sep" }),
    glyph("bd-zoom-out", "Zoom out (−)", "−", () => ed.zoomBy(0.8)),
    zoomLabel,
    glyph("bd-zoom-in", "Zoom in (+)", "+", () => ed.zoomBy(1.25)),
    btn("bd-fit", "Show the whole diagram (0)", "expand", "", () => ed.fit()),
    el("span", { class: "spacer" }),
    problemsBtn,
    btn("bd-scopes", "Scopes of this model", "eye", "Scopes", (e) => scopesMenu(e.currentTarget)),
    btn("bd-palette-toggle", "Show or hide the block library", "outline", "Library", () => togglePalette(), { attrs: { "aria-pressed": "true" } }),
    btn("bd-text", "Show the model file as text (JSON)", "text", "", () => showText(true)),
    btn("bd-keys", "Keys and mouse help (?)", "keyboard", "", () => keysDialog())
  );
  const crumbs = el("nav", { class: "bd-crumbs", "aria-label": "Subsystem path" });
  const status = el("div", { class: "bd-live", role: "status", "aria-live": "polite" });
  const hint = el("div", { class: "bd-hint muted" });
  const canvas = el("div", { class: "bd-canvas" }, hint, status);
  const palette = el("aside", { class: "bd-palette", "aria-label": "Block library" });
  const dockEl = el("div", { class: "bd-dock", "aria-label": "Scopes" });
  const errorBox = el("div", { class: "bd-error", hidden: true });
  const view = el(
    "div",
    { id: "blocks-view", class: "bd-view", hidden: true },
    toolbar, crumbs, el("div", { class: "bd-main" }, palette, canvas), dockEl, errorBox
  );
  $("editor").appendChild(view);
  const textBar = el(
    "div",
    { id: "bd-textbar", class: "bd-textbar small", hidden: true },
    el("span", {}, "This is the model file as text. Changes you make here show in the diagram."),
    el("button", { id: "bd-show-diagram", class: "btn sm primary", type: "button", onclick: () => showText(false) }, "Show the diagram")
  );
  $("editor").before(textBar);

  let current = null; // path of the model shown
  let textMode = false;
  let selfChange = false;
  let checkTimer = null;
  let running = false;
  const viewStates = new Map(); // path -> { path, view } so each model reopens where it was

  const ed = new DiagramEditor(canvas, {
    onChange: (text) => {
      if (!current) return;
      const doc = app.getDoc(current);
      if (doc.getValue() === text) return;
      selfChange = true;
      try {
        doc.setValue(text);
      } finally {
        selfChange = false;
      }
      scheduleCheck();
    },
    onRender: () => {
      zoomLabel.textContent = Math.round(ed.view.k * 100) + "%";
      const sys = ed.sys();
      hint.hidden = sys.blocks.length > 0;
      if (!sys.blocks.length) hint.textContent = "This diagram is empty. Drag blocks here from the library, double-click, or press A to add a block by name.";
      updateProblems();
    },
    onNavigate: () => renderCrumbs(),
    announce: (text) => (status.textContent = text),
    toast: (msg, kind) => toast(msg, kind),
    editParams: (block) => editParams(block),
    rename: (block) => renameBlock(block),
    editLabel: (line) => editLabel(line),
    quickInsert: (point, from) => quickInsert(point, from),
    connectDialog: (block) => connectDialog(block),
    contextMenu: (e, hit, world) => contextMenu(e, hit, world),
    openScope: (path) => dock.open(path, scopeMeta(path)),
    showKeys: () => keysDialog(),
  });
  // The scopes take room from the canvas: show the whole diagram again in what is left.
  const dock = new ScopeDock(dockEl, { onChange: () => requestAnimationFrame(() => ed.fit()) });

  /* ---------- palette ---------- */

  const search = el("input", { id: "bd-search", type: "search", class: "bd-search", placeholder: "Search blocks", spellcheck: "false", autocomplete: "off", "aria-label": "Search blocks" });
  const palList = el("div", { class: "bd-pal-list" });
  palette.append(el("div", { class: "bd-pal-head" }, search), palList);
  const closed = new Set(app.prefs.get("blocksClosedCats", ["Discrete", "Logic", "Nonlinear", "User-defined"]));

  function renderPalette() {
    const q = search.value.trim().toLowerCase();
    palList.replaceChildren();
    let shown = 0;
    for (const cat of CATALOG.categories) {
      const blocks = CATALOG.blocks.filter((b) => b.category === cat && (!q || (b.title + " " + b.type + " " + b.doc).toLowerCase().includes(q)));
      if (!blocks.length) continue;
      shown += blocks.length;
      const open = q ? true : !closed.has(cat);
      const head = el(
        "button",
        {
          class: "bd-pal-cat cat-" + CATEGORY_CLASS[cat], type: "button", "aria-expanded": String(open),
          onclick: () => {
            if (closed.has(cat)) closed.delete(cat);
            else closed.add(cat);
            app.prefs.set("blocksClosedCats", [...closed]);
            renderPalette();
          },
        },
        icon(open ? "chevron" : "caret"), el("span", {}, cat), el("span", { class: "muted small" }, String(blocks.length))
      );
      palList.append(head);
      if (!open) continue;
      const grid = el("div", { class: "bd-pal-grid", role: "list" });
      for (const b of blocks) {
        const item = el(
          "button",
          { class: "bd-pal-item", type: "button", role: "listitem", title: `${b.title}: ${b.doc} Drag it to the diagram, or click to add it.`, "data-type": b.type },
          paletteIcon(b.type), el("span", { class: "bd-pal-name" }, b.title)
        );
        item.addEventListener("click", () => {
          if (item._dragged) return;
          const added = ed.addBlock(b.type, null);
          if (added) ed.svg.focus();
        });
        bindPaletteDrag(item, b);
        grid.append(item);
      }
      palList.append(grid);
    }
    if (!shown) palList.append(el("p", { class: "muted small bd-pal-empty" }, "No block matches. Try a shorter word."));
  }
  search.addEventListener("input", renderPalette);
  search.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const first = palList.querySelector(".bd-pal-item");
      if (first) first.click();
    } else if (e.key === "ArrowDown") {
      const first = palList.querySelector(".bd-pal-item");
      first && first.focus();
      e.preventDefault();
    }
  });
  palList.addEventListener("keydown", (e) => {
    if (!["ArrowDown", "ArrowUp", "ArrowLeft", "ArrowRight"].includes(e.key)) return;
    const items = [...palList.querySelectorAll(".bd-pal-item, .bd-pal-cat")];
    const i = items.indexOf(document.activeElement);
    if (i < 0) return;
    const next = items[i + (e.key === "ArrowDown" || e.key === "ArrowRight" ? 1 : -1)];
    if (next) next.focus();
    else if (e.key === "ArrowUp") search.focus();
    e.preventDefault();
  });

  /** Dragging a palette item: a ghost follows the pointer; dropping over the canvas adds the block there. */
  function bindPaletteDrag(item, b) {
    item.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const start = { x: e.clientX, y: e.clientY };
      let ghost = null;
      item._dragged = false;
      const move = (ev) => {
        if (!ghost && Math.hypot(ev.clientX - start.x, ev.clientY - start.y) > 6) {
          ghost = el("div", { class: "bd-ghost" }, paletteIcon(b.type), el("span", {}, b.title));
          document.body.appendChild(ghost);
          item._dragged = true;
        }
        if (ghost) {
          ghost.style.left = ev.clientX + 8 + "px";
          ghost.style.top = ev.clientY + 8 + "px";
          ev.preventDefault();
        }
      };
      const up = (ev) => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        window.removeEventListener("pointercancel", up);
        if (!ghost) return;
        ghost.remove();
        const r = ed.svg.getBoundingClientRect();
        if (ev.type === "pointerup" && ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom) {
          ed.addBlock(b.type, ed.toWorld(ev));
          ed.svg.focus();
        }
        setTimeout(() => (item._dragged = false), 0);
      };
      window.addEventListener("pointermove", move, { passive: false });
      window.addEventListener("pointerup", up);
      window.addEventListener("pointercancel", up);
    });
  }

  function togglePalette(show) {
    const open = show === undefined ? palette.hidden : show;
    palette.hidden = !open;
    $("bd-palette-toggle").setAttribute("aria-pressed", String(open));
    app.prefs.set("blocksPalette", open);
    requestAnimationFrame(() => ed.applyView());
  }

  /* ---------- showing a model ---------- */

  function renderCrumbs() {
    const parts = [ed.model ? ed.model.name : "", ...ed.path];
    crumbs.replaceChildren(
      ...parts.flatMap((name, i) => {
        const last = i === parts.length - 1;
        const node = last
          ? el("span", { class: "bd-crumb here", "aria-current": "location" }, name)
          : el("button", { class: "bd-crumb", type: "button", title: `Go up to ${name}`, onclick: () => ed.goTo(ed.path.slice(0, i)) }, name);
        return i ? [el("span", { class: "bd-crumb-sep", "aria-hidden": "true" }, "›"), node] : [node];
      })
    );
    crumbs.hidden = ed.path.length === 0; // at the top level the tab already names the model
    if (ed.path.length) crumbs.prepend(el("button", { id: "bd-up", class: "btn icon sm ghost", type: "button", title: "Up to the parent system (Esc)", "aria-label": "Up to the parent system", onclick: () => ed.leave() }, icon("up")));
  }

  function show(path) {
    if (current && current !== path && ed.model) viewStates.set(current, { path: ed.path.slice(), view: { ...ed.view } });
    const fresh = current !== path;
    current = path;
    textMode = false;
    textBar.hidden = true;
    view.hidden = false;
    document.body.dataset.blocks = "on";
    if (fresh) {
      status.textContent = "";
      dock.closeAll();
      ed.setDiagnostics([]);
      ed.displays = new Map();
      progress.textContent = "";
    }
    load(fresh);
    if (window.innerWidth < 900 && app.prefs.get("blocksPalette", null) === null) palette.hidden = true;
    else palette.hidden = app.prefs.get("blocksPalette", true) === false;
    $("bd-palette-toggle").setAttribute("aria-pressed", String(!palette.hidden));
    scheduleCheck(300);
  }

  function load(fresh) {
    const text = app.getDoc(current).getValue();
    if (!text.trim()) {
      // an empty file becomes an empty model
      const name = app.baseName(current).slice(0, -EXT.length) || "model";
      ed.setText(serializeModel(newModel(name)));
      ed.emitChange();
    } else {
      if (!fresh && text === ed.lastText) return;
      const saved = fresh ? viewStates.get(current) : { path: ed.path.slice(), view: { ...ed.view } };
      const err = ed.setText(text, { keepView: !!saved });
      errorBox.hidden = !err;
      if (err) {
        errorBox.replaceChildren(
          el("p", {}, el("strong", {}, "This file cannot be shown as a diagram. "), err),
          el("button", { class: "btn sm", type: "button", onclick: () => showText(true) }, "Edit it as text")
        );
        return;
      }
      if (saved) {
        if (systemAt(ed.model, saved.path)) ed.path = saved.path;
        ed.view = saved.view;
        ed.render();
      }
    }
    stopInput.value = ed.model.config.StopTime;
    renderCrumbs();
    scheduleCheck();
  }

  function hide() {
    if (current && ed.model) viewStates.set(current, { path: ed.path.slice(), view: { ...ed.view } });
    current = null;
    view.hidden = true;
    textBar.hidden = true;
    delete document.body.dataset.blocks;
  }

  function showText(on) {
    if (!current) return;
    textMode = on;
    view.hidden = on;
    textBar.hidden = !on;
    if (on) {
      app.editor.refresh();
      app.editor.focus();
    } else {
      load(false);
      ed.svg.focus();
    }
  }

  app.on("fileOpened", (path) => (isModelPath(path) ? show(path) : hide()));
  app.on("tabsChanged", () => {
    if (current && state.active !== current) hide();
  });
  app.on("projectChanged", () => {
    viewStates.clear();
    if (!isModelPath(state.active)) hide();
  });
  app.on("removed", () => {
    if (current && !state.files.has(current)) hide();
  });
  app.on("moved", (from, to) => {
    if (current === from) current = to;
  });
  // The file changed from outside the diagram (text view, version history, the assistant): reload it.
  app.editor.on("change", () => {
    if (selfChange || !current || state.active !== current || textMode) return;
    clearTimeout(load.timer);
    load.timer = setTimeout(() => current && !textMode && load(false), 50);
  });
  stopInput.addEventListener("change", () => {
    const v = stopInput.value.trim();
    if (!v || !ed.model) return (stopInput.value = ed.model ? ed.model.config.StopTime : "");
    ed.begin();
    ed.model.config.StopTime = v;
    ed.commit();
  });
  stopInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      stopInput.dispatchEvent(new Event("change"));
      runModel();
    }
  });

  /* ---------- dialogs and menus ---------- */

  async function editParams(block) {
    const sys = ed.sys();
    const taken = new Set(sys.blocks.map((b) => b.name));
    const validate = app.engine.ready && !app.engine.busy
      ? async (values) => (await app.engine.background("blocks", { op: "validate", args: { values } })).errors
      : null;
    const out = await paramsDialog(block, { taken, validate });
    ed.svg.focus();
    if (!out) return;
    const before = portCounts(block);
    const after = portsWith(block, out.params);
    ed.change(() => {
      block.params = out.params;
      if (out.mask) {
        if (Object.keys(out.mask).length) block.mask = out.mask;
        else delete block.mask;
      }
      if (after && (after.in !== before.in || after.out !== before.out)) {
        // room for the new number of ports
        const n = Math.max(after.in, after.out, 1);
        if (block.position[3] < n * 18) block.position[3] = Math.ceil((n * 20) / 10) * 10;
        ed.clearRoutes(sys, block.name);
      }
    });
    if (out.name !== block.name) {
      const err = ed.renameBlock(block.name, out.name);
      if (err) toast(err, "error");
    }
    if (after && (after.in < before.in || after.out < before.out)) toast("The block has fewer ports now; lines to the removed ports were deleted. Ctrl+Z brings them back.");
  }

  async function renameBlock(block) {
    const name = await textDialog({
      title: "Rename block", label: "Name", value: block.name, okLabel: "Rename",
      validate: (v) => {
        const n = v.trim();
        if (!n) return "A block needs a name.";
        if (n !== block.name && ed.block(n)) return `There is already a block named "${n}" here.`;
        return null;
      },
    });
    ed.svg.focus();
    if (name !== null) ed.renameBlock(block.name, name);
  }

  async function editLabel(line) {
    const name = await textDialog({
      title: "Signal name", label: `Name of the signal from ${line.from[0]}`, value: line.name || "", okLabel: "OK",
      hint: "The name is shown on the line and in scope legends. Leave it empty to remove the name.",
    });
    ed.svg.focus();
    if (name === null) return;
    ed.change((sys) => {
      // every branch of one output carries the same signal, so it gets the same name
      for (const l of sys.lines) if (l.from[0] === line.from[0] && l.from[1] === line.from[1]) {
        if (name.trim()) l.name = name.trim();
        else delete l.name;
      }
    });
  }

  async function editSettings() {
    const cfg = await settingsDialog(ed.model.config);
    if (!cfg) return;
    ed.begin();
    ed.model.config = cfg;
    ed.commit();
    stopInput.value = cfg.StopTime;
  }

  /** Add a block by typing its name. from: a port to connect the new block to. */
  async function quickInsert(point, from) {
    const type = await pickDialog({
      title: from ? "Add a block and connect it" : "Add a block", placeholder: "Type a block name, for example gain", id: "bd-dlg-add",
      items: CATALOG.blocks.map((b) => ({ label: b.title, group: b.category, hint: b.doc, value: b.type, keywords: b.type })),
    });
    ed.svg.focus();
    if (!type) return;
    const added = ed.addBlock(type, point);
    if (!added || !from) return;
    const c = portCounts(added);
    if (from.io === "out" && c.in) ed.connect({ block: from.block, port: from.port }, { block: added.name, port: 1 });
    else if (from.io === "in" && c.out) ed.connect({ block: added.name, port: 1 }, { block: from.block, port: from.port });
  }

  /** Keyboard connecting: pick where the selected block's output goes (or what feeds its input). */
  async function connectDialog(block) {
    if (!block) return toast("Select a block first (Tab), then press C to connect it.");
    const sys = ed.sys();
    const items = [];
    const outs = portCounts(block).out;
    for (let o = 1; o <= outs; o++) {
      for (const b of sys.blocks) {
        if (b === block) continue;
        for (const p of ed.freeInputs(b)) items.push({ label: `${block.name}${outs > 1 ? " output " + o : ""} → ${b.name} input ${p}`, group: "to", value: { from: { block: block.name, port: o }, to: { block: b.name, port: p } } });
      }
    }
    for (const p of ed.freeInputs(block)) {
      for (const b of sys.blocks) {
        if (b === block) continue;
        const n = portCounts(b).out;
        for (let o = 1; o <= n; o++) items.push({ label: `${b.name}${n > 1 ? " output " + o : ""} → ${block.name} input ${p}`, group: "from", value: { from: { block: b.name, port: o }, to: { block: block.name, port: p } } });
      }
    }
    if (!items.length) return toast("There is nothing to connect this block to: no other block has a free input, and its own inputs are taken.");
    const pick = await pickDialog({ title: `Connect ${block.name}`, placeholder: "Type the name of the other block", items, id: "bd-dlg-connect" });
    ed.svg.focus();
    if (!pick) return;
    const err = ed.connect(pick.from, pick.to);
    if (err) toast(err, "error");
  }

  function menuAt(e, items) {
    const anchor = el("span", { style: `position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:1px;height:1px` });
    document.body.appendChild(anchor);
    openMenu(anchor, items);
    setTimeout(() => anchor.remove(), 0);
  }

  function editItems() {
    const n = ed.sel.size;
    return [
      { label: "Rotate right (R)", onclick: () => ed.rotate(1) },
      { label: "Rotate left (Shift+R)", onclick: () => ed.rotate(-1) },
      { label: "Flip (F)", onclick: () => ed.flip() },
      "sep",
      { heading: "Align" },
      { label: "Left edges", onclick: () => ed.align("left") },
      { label: "Right edges", onclick: () => ed.align("right") },
      { label: "Top edges", onclick: () => ed.align("top") },
      { label: "Bottom edges", onclick: () => ed.align("bottom") },
      { label: "Centres in a column", onclick: () => ed.align("center") },
      { label: "Centres in a row", onclick: () => ed.align("middle") },
      { label: "Space evenly across", onclick: () => ed.align("hspace") },
      { label: "Space evenly down", onclick: () => ed.align("vspace") },
      "sep",
      { label: `Make a subsystem from ${n > 1 ? "these blocks" : "this block"} (Ctrl+G)`, onclick: () => ed.group() },
    ];
  }

  function editMenu(anchor) {
    openMenu(anchor, [
      { label: "Select all (Ctrl+A)", onclick: () => ed.selectAll() },
      { label: "Copy (Ctrl+C)", onclick: () => ed.copy() },
      { label: "Paste (Ctrl+V)", onclick: () => ed.paste() },
      { label: "Duplicate (Ctrl+D)", onclick: () => ed.duplicate() },
      { label: "Delete (Del)", danger: true, onclick: () => ed.deleteSelection() },
      "sep",
      ...editItems(),
      "sep",
      { label: "Connect the selected blocks (C)", onclick: () => ed.onKey(new KeyboardEvent("keydown", { key: "c" })) },
      { heading: "Models" },
      { label: "New block diagram…", onclick: () => newDiagram() },
      { label: "Import a .slx file…", onclick: () => slxInput.click() },
      ...BLOCK_EXAMPLES.map((x) => ({ label: "Example: " + x.title, onclick: () => openExample(x.name) })),
    ]);
  }

  function contextMenu(e, hit, world) {
    if (ed.readOnly) return;
    if (hit.kind === "block") {
      const b = ed.block(hit.block);
      const t = findType(b.type);
      menuAt(e, [
        Array.isArray(b.blocks) ? { label: "Look inside (Enter)", onclick: () => ed.enter(b.name) } : null,
        t && (t.type === "Scope" || t.type === "XYGraph") ? { label: "Open the scope", onclick: () => dock.open(ed.prefix() + b.name, scopeMeta(ed.prefix() + b.name)) } : null,
        { label: "Parameters…", onclick: () => editParams(b) },
        { label: "Rename… (F2)", onclick: () => renameBlock(b) },
        { label: "Connect… (C)", onclick: () => connectDialog(b) },
        "sep",
        { label: "Copy", onclick: () => ed.copy() },
        { label: "Duplicate", onclick: () => ed.duplicate() },
        { label: "Delete", danger: true, onclick: () => ed.deleteSelection() },
        "sep",
        ...editItems(),
      ]);
    } else if (hit.kind === "line") {
      menuAt(e, [
        { label: "Name the signal… (L)", onclick: () => editLabel(hit.line) },
        { label: hit.line.log ? "Stop logging this signal" : "Log this signal (needs a name)", onclick: () => ed.change(() => (hit.line.log = !hit.line.log)) },
        { label: "Route the line afresh", onclick: () => ed.change(() => delete hit.line.points) },
        { label: "Delete the line", danger: true, onclick: () => ed.deleteSelection() },
      ]);
    } else {
      menuAt(e, [
        { label: "Add a block here…", onclick: () => quickInsert(world, null) },
        { label: "Paste here", onclick: () => ed.paste(world) },
        { label: "Select all", onclick: () => ed.selectAll() },
        { label: "Arrange the blocks", onclick: () => ed.layout() },
        { label: "Show the whole diagram", onclick: () => ed.fit() },
        ed.path.length ? { label: "Up to the parent system", onclick: () => ed.leave() } : null,
      ]);
    }
  }

  /** Every Scope and XY Graph of the model: [{ path, kind }]. */
  function allScopes() {
    const out = [];
    const walk = (sys, prefix) => {
      for (const b of sys.blocks) {
        const t = findType(b.type);
        if (t && (t.type === "Scope" || t.type === "XYGraph")) out.push({ path: prefix + b.name, kind: t.type === "XYGraph" ? "xy" : "scope", block: b, sys });
        if (Array.isArray(b.blocks)) walk(b, prefix + b.name + "/");
      }
    };
    if (ed.model) walk(ed.model, "");
    return out;
  }

  function scopeMeta(path) {
    const s = allScopes().find((x) => x.path === path);
    if (!s) return null;
    const n = portCounts(s.block).in;
    const names = [];
    for (let k = 1; k <= n; k++) {
      const l = s.sys.lines.find((x) => x.to[0] === s.block.name && x.to[1] === k);
      names.push((l && l.name) || "");
    }
    return { path, kind: s.kind, title: (s.block.params || {}).Title || "", names };
  }

  function scopesMenu(anchor) {
    const list = allScopes();
    if (!list.length) return toast("This model has no Scope blocks. Add one from the Sinks group of the library and connect a signal to it.");
    openMenu(anchor, [
      ...list.map((s) => ({ label: s.path, checked: dock.has(s.path), onclick: () => (dock.has(s.path) ? dock.close(s.path) : dock.open(s.path, scopeMeta(s.path))) })),
      "sep",
      { label: "Open all", onclick: () => list.forEach((s) => dock.open(s.path, scopeMeta(s.path))) },
      { label: "Close all", onclick: () => dock.closeAll() },
    ]);
  }

  /* ---------- problems ---------- */

  function updateProblems() {
    const list = problemList();
    const errors = list.filter((p) => p.level === "error").length;
    problemsBtn.hidden = !list.length;
    problemsBtn.classList.toggle("bd-has-errors", errors > 0);
    problemsBtn.querySelector("span").textContent = String(list.length);
    problemsBtn.title = `${list.length} problem${list.length === 1 ? "" : "s"} in this model` + (errors ? ` (${errors} that stop it from running)` : "");
    view.dataset.problems = String(list.length);
  }

  /** Engine diagnostics plus the unconnected inputs seen on the canvas, most serious first. */
  function problemList() {
    const out = [];
    const seen = new Set();
    for (const d of ed.diagnostics) {
      if (d.level === "info") continue;
      const k = d.message;
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ level: d.level, message: d.message, path: d.path });
    }
    if (ed.lastMarkers) {
      for (const [name, m] of ed.lastMarkers) {
        const path = ed.prefix() + name;
        for (const msg of [m.message, ...(m.more || [])]) {
          const text = `'${path}': ${msg}`;
          if (out.some((o) => o.path === path && (o.message === msg || o.message.includes("not connected") === msg.includes("not connected")))) continue;
          if (seen.has(text)) continue;
          seen.add(text);
          out.push({ level: m.level, message: text, path });
        }
      }
    }
    out.sort((a, b) => (a.level === "error" ? 0 : 1) - (b.level === "error" ? 0 : 1));
    return out;
  }

  function showProblems(anchor) {
    const list = problemList();
    openMenu(anchor, list.slice(0, 30).map((p) => ({ label: p.message, icon: p.level === "error" ? "error" : "warn", hint: p.message, danger: p.level === "error", onclick: () => p.path && ed.showBlock(p.path) })), { id: "bd-problems-menu" });
  }

  function scheduleCheck(delay = 700) {
    clearTimeout(checkTimer);
    checkTimer = setTimeout(async () => {
      if (!current || !ed.model || !app.engine.ready || app.engine.busy || running) return;
      const path = current, text = ed.lastText;
      try {
        const res = await app.engine.background("blocks", { op: "check", args: { model: ed.model, name: ed.model.name }, cwd: app.dirName(path) });
        if (current === path && ed.lastText === text) ed.setDiagnostics(res.diagnostics || []);
      } catch {
        /* the engine restarted; the next change checks again */
      }
    }, delay);
  }
  app.on("engineReady", () => scheduleCheck(200));
  app.on("runFinished", () => scheduleCheck(200)); // new workspace variables may fix or break parameters

  /* ---------- running ---------- */

  // The same Engine object lives on when the worker is restarted, so one listener is enough.
  app.engine.addEventListener("blocks", (e) => onEngineEvent(e.detail));

  function onEngineEvent(msg) {
    if (!msg || !running) return;
    if (msg.kind === "start") {
      dock.start(msg.scopes.map((m) => ({ ...m })), [msg.t0, msg.tf]);
    } else if (msg.kind === "progress") {
      dock.append(msg.scopes);
      progress.textContent = `t = ${formatNumber(msg.t)} of ${formatNumber(msg.tf)}`;
    }
  }

  async function runModel() {
    if (!current || !ed.model) return null;
    if (errorBox.hidden === false) return toast("Fix the model file first (it cannot be read).", "error");
    if (!app.canRun()) return null;
    const path = current;
    const model = ed.model;
    app.flushSave && app.flushSave(path);
    app.showBottom("console");
    app.printEcho(`sim('${model.name}')`, `run model ${path} to t = ${model.config.StopTime}`);
    running = true;
    runBtn.hidden = true;
    stopBtn.hidden = false;
    progress.textContent = "Starting…";
    app.setRunning(true, "Simulating");
    let res = null;
    try {
      res = await app.engine.request("blocks", { op: "run", args: { model, name: model.name }, fs: app.fsSpec(), cwd: app.dirName(path) });
    } catch (e) {
      if (String(e.message) !== "Stopped.") app.printError({ message: e.message });
      else progress.textContent = "Stopped.";
    }
    running = false;
    runBtn.hidden = false;
    stopBtn.hidden = true;
    app.setRunning(false);
    if (!res) {
      dock.finish();
      return null;
    }
    if (res.output) app.printOut(res.output);
    ed.setDiagnostics(res.diagnostics || []);
    for (const w of res.warnings || []) app.printInfo("Warning: " + w, "note");
    if (res.error) {
      dock.finish();
      progress.textContent = "The model did not run.";
      app.printError({ message: res.error.message, where: path });
      const at = (res.error.paths && res.error.paths[0]) || res.error.path;
      if (at && current === path) ed.showBlock(at);
      toast(res.error.message.length > 200 ? res.error.message.slice(0, 200) + "…" : res.error.message, "error");
    } else {
      dock.append(res.scopes);
      dock.finish();
      const shown = {};
      for (const [p, v] of Object.entries(res.displays || {})) shown[p] = v.length ? v.map(formatNumber).join(" ") : "";
      ed.setDisplays(shown);
      progress.textContent = `Done: ${res.steps} steps to t = ${formatNumber(res.t_end)}` + (res.stopped_by ? ` (stopped by ${res.stopped_by})` : "");
      app.printInfo(`Simulation finished: ${res.steps} steps to t = ${formatNumber(res.t_end)} with ${res.solver}. Results are in the workspace: ${res.assigned.join(", ")}.`);
    }
    if (res.variables) app.renderVariables(res.variables);
    view.dataset.runs = String(Number(view.dataset.runs || 0) + 1);
    app.emit("blocksRun", res);
    return res;
  }

  // The Run button and F5 of the app run the model while a diagram is the active file.
  app.runSpecial = (path) => {
    if (!isModelPath(path)) return false;
    runModel();
    return true;
  };

  /* ---------- files ---------- */

  async function newDiagram() {
    const folder = state.active ? app.dirName(state.active) : "";
    const prefix = folder ? folder + "/" : "";
    let n = 1;
    while (state.files.has(`${prefix}model${n > 1 ? n : ""}${EXT}`)) n++;
    const name = await textDialog({
      title: "New block diagram", label: "Name", value: `${prefix}model${n > 1 ? n : ""}`, okLabel: "Create",
      hint: `The model is saved as a ${EXT} file in this project. Use letters, digits and _ so that sim('name') can find it.`,
      validate: (v) => {
        const base = v.trim().replace(/\.mlb\.json$/i, "");
        if (!/^([\w.-]+\/)*[A-Za-z]\w*$/.test(base)) return "Use a name that starts with a letter and has only letters, digits and _ (folders with / are fine).";
        const r = app.validatePath(base + EXT, { addExt: false });
        return r.error || null;
      },
    });
    if (name === null) return;
    const base = name.trim().replace(/\.mlb\.json$/i, "");
    const path = app.normalizePath(base + EXT);
    await app.createFile(path, serializeModel(newModel(app.baseName(base))));
  }

  async function openExample(name) {
    const ex = BLOCK_EXAMPLES.find((x) => x.name === name);
    if (!ex) return;
    const path = "models/" + ex.file;
    if (!state.files.has(path)) await app.createFile(path, ex.text, { open: false });
    app.openFile(path);
  }

  const slxInput = el("input", { type: "file", accept: ".slx", hidden: true });
  document.body.appendChild(slxInput);
  slxInput.addEventListener("change", async () => {
    const file = slxInput.files[0];
    slxInput.value = "";
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) return toast("That file is larger than 20 MB; it is not imported.", "error");
    const path = app.normalizePath(file.name.replace(/[^\w.\-]+/g, "_"));
    await app.writeFiles([{ path, content: new Uint8Array(await file.arrayBuffer()) }]);
    app.renderFileList();
    importSlx(path);
  });

  /** Convert a .slx file of the project into a model file next to it, and open that. */
  async function importSlx(path) {
    if (!app.engine.ready) return toast("The engine is still loading. Open the .slx file again in a moment.");
    if (app.engine.busy) return toast("Wait until the running code has finished, then open the .slx file again.");
    let res;
    app.setRunning(true, "Importing");
    try {
      res = await app.engine.request("blocks", { op: "import_slx", args: { path, root: "/home/pyodide/work" }, fs: app.fsSpec() });
    } catch (e) {
      app.setRunning(false);
      return toast("The file could not be imported: " + e.message, "error");
    }
    app.setRunning(false);
    if (res.error) return toast(res.error, "error");
    const target = app.normalizePath(path.replace(/\.slx$/i, "") + EXT);
    await app.writeFiles([{ path: target, content: res.model }]);
    app.renderFileList();
    app.openFile(target);
    const missing = Object.keys(res.unsupported || {}).length;
    noticeDialog({
      title: missing ? "Imported, with blocks that mylab does not have" : "Imported",
      text: missing
        ? `${app.baseName(path)} was converted to ${app.baseName(target)}. Some blocks have no equivalent and are shown as grey placeholders with their original type; the model can be looked at, but it will not run until they are replaced.`
        : `${app.baseName(path)} was converted to ${app.baseName(target)} (${res.blocks} blocks). Check the parameters before you rely on the results: the import covers the common blocks, not every option.`,
      lines: res.notes || [],
    });
    return target;
  }

  app.openSpecial = (path) => {
    if (!/\.slx$/i.test(path)) return false;
    const target = path.replace(/\.slx$/i, "") + EXT;
    if (state.files.has(target)) app.openFile(target);
    else importSlx(path);
    return true;
  };

  /* ---------- commands ---------- */

  const inDiagram = () => !!current && !view.hidden;
  app.commands.register([
    { id: "blocks.new", group: "Blocks", title: "New block diagram…", run: newDiagram, hint: "Create a model you build from blocks and lines" },
    { id: "blocks.import", group: "Blocks", title: "Import a .slx model…", run: () => slxInput.click(), hint: "Convert a simple .slx file into a mylab Blocks model" },
    ...BLOCK_EXAMPLES.map((x) => ({ id: "blocks.example." + x.name, group: "Blocks", title: `Example model: ${x.title}`, run: () => openExample(x.name), hint: x.description })),
    { id: "blocks.run", group: "Blocks", title: "Run the model", run: runModel, when: inDiagram },
    { id: "blocks.add", group: "Blocks", title: "Add a block by name…", run: () => quickInsert(null, null), when: inDiagram },
    { id: "blocks.layout", group: "Blocks", title: "Arrange the blocks automatically", run: () => ed.layout(), when: inDiagram },
    { id: "blocks.fit", group: "Blocks", title: "Show the whole diagram", run: () => ed.fit(), when: inDiagram },
    { id: "blocks.settings", group: "Blocks", title: "Simulation settings…", run: editSettings, when: inDiagram },
    { id: "blocks.group", group: "Blocks", title: "Make a subsystem from the selection", run: () => ed.group(), when: inDiagram },
    { id: "blocks.text", group: "Blocks", title: "Show the model file as text", run: () => showText(!textMode), when: () => !!current },
    { id: "blocks.keys", group: "Blocks", title: "Keys in the diagram editor", run: keysDialog },
  ]);

  renderPalette();
  // If a model file was already open before this module started, show it.
  if (isModelPath(state.active)) show(state.active);

  app.blocks = { editor: ed, dock, runModel, openExample, newDiagram, importSlx, showText, show, hide, get current() { return current; }, closeMenu };
}
