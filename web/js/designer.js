// The app builder: lay out an app on a grid, set properties, and generate a readable .m file
// (uifigure + uigridlayout + components, with one callback stub per component).
//
// The design is kept as plain data:
//   { format, name, title, width, height, rows: ["fit" | "1x" | pixels], cols: [...],
//     items: [{ id, type, name, row, col, rowSpan, colSpan, props: {...}, callback }] }
// and is written as a JSON comment block at the end of the generated file, so reopening the
// file restores the design. Callback bodies the user wrote are kept when code is generated again.
//
// The canvas is drawn by the same renderer the running app uses (uirender.js, read-only),
// so what you see is what the app looks like.

import "./uirender.js";
import { $, el, icon, toast, promptDialog } from "./ui.js";

export const DESIGN_MARK = "mylab-app-design";
const FORMAT = "mylab-app-1";

/* ------------------------------------------------------------------ component kinds */

const AXES_PREVIEW =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 160" preserveAspectRatio="none"><rect width="240" height="160" fill="#fff"/>' +
      '<path d="M30 12v126h198" fill="none" stroke="#333" stroke-width="1.5"/><path d="M30 110c30-80 50-80 70-30s40 50 60 0 40-50 62-20" fill="none" stroke="#1f77b4" stroke-width="2.5"/></svg>'
  );

// fields: property editors shown for the kind. code: (item) => [constructor args, extra lines].
export const KINDS = {
  label: { title: "Label", base: "label", props: { text: "Label" }, fields: ["text"], event: null },
  button: { title: "Button", base: "button", props: { text: "Button" }, fields: ["text"], event: ["ButtonPushedFcn", "Pushed", "the button is pressed"] },
  editfield: { title: "Text field", base: "field", props: { value: "", placeholder: "" }, fields: ["value", "placeholder"], event: ["ValueChangedFcn", "Changed", "the text is changed"] },
  numericfield: { title: "Number field", base: "number", props: { value: 0, min: "", max: "" }, fields: ["number", "min", "max"], event: ["ValueChangedFcn", "Changed", "the number is changed"] },
  textarea: { title: "Text area", base: "text", props: { value: "" }, fields: ["lines"], event: ["ValueChangedFcn", "Changed", "the text is changed"] },
  dropdown: { title: "Drop-down", base: "choice", props: { items: ["Option 1", "Option 2"] }, fields: ["items"], event: ["ValueChangedFcn", "Changed", "another item is chosen"] },
  listbox: { title: "List box", base: "list", props: { items: ["Item 1", "Item 2", "Item 3"] }, fields: ["items"], event: ["ValueChangedFcn", "Changed", "another item is chosen"] },
  checkbox: { title: "Check box", base: "check", props: { text: "Check box", value: false }, fields: ["text", "checked"], event: ["ValueChangedFcn", "Changed", "the box is ticked or cleared"] },
  switch: { title: "Switch", base: "switch", props: { value: false }, fields: ["checked"], event: ["ValueChangedFcn", "Changed", "the switch is flipped"] },
  slider: { title: "Slider", base: "slider", props: { value: 0, min: 0, max: 100 }, fields: ["number", "min", "max"], event: ["ValueChangedFcn", "Changed", "the slider is released"] },
  spinner: { title: "Spinner", base: "spinner", props: { value: 0, min: "", max: "", step: 1 }, fields: ["number", "min", "max", "step"], event: ["ValueChangedFcn", "Changed", "the number is changed"] },
  table: { title: "Table", base: "table", props: { columns: ["A", "B"], editable: true }, fields: ["columns", "editable"], event: ["CellEditCallback", "Edited", "a cell is edited"] },
  axes: { title: "Axes (plot)", base: "ax", props: { title: "" }, fields: ["plotTitle"], event: null },
  gauge: { title: "Gauge", base: "gauge", props: { value: 0, min: 0, max: 100 }, fields: ["number", "min", "max"], event: null },
  lamp: { title: "Lamp", base: "lamp", props: { color: "green" }, fields: ["color"], event: null },
  datepicker: { title: "Date picker", base: "date", props: {}, fields: [], event: ["ValueChangedFcn", "Changed", "a date is chosen"] },
};

const FIELD_LABELS = {
  text: "Text", value: "Text in the field", placeholder: "Hint when empty", number: "Value", min: "Smallest value", max: "Largest value", step: "Step",
  lines: "Text (one line per line)", items: "Items (one per line)", checked: "On at the start", columns: "Column names (one per line)", editable: "Cells can be edited",
  plotTitle: "Title of the plot", color: "Color (name or #rrggbb)",
};

const isName = (s) => /^[A-Za-z][A-Za-z0-9_]{0,40}$/.test(s || "");
const RESERVED = new Set(["fig", "grid", "src", "event", "end", "function", "if", "for", "while", "switch", "case", "else", "try", "catch", "return", "break", "startup"]);
const num = (v, fallback) => (v === "" || v == null || Number.isNaN(Number(v)) ? fallback : Number(v));
const clone = (x) => JSON.parse(JSON.stringify(x));

export function newDesign(name = "MyApp") {
  return { format: FORMAT, name, title: "My app", width: 480, height: 360, rows: ["fit", "fit", "fit", "1x"], cols: ["1x", "1x"], items: [], nextId: 1 };
}

/* ------------------------------------------------------------------ design -> component tree (for the canvas) */

export function designTree(d) {
  const kids = d.items.map((it) => {
    const p = it.props;
    const grid = [it.row, it.row + it.rowSpan - 1, it.col, it.col + it.colSpan - 1];
    const lim = { min: num(p.min, 0), max: num(p.max, 100) };
    const node = (type, props) => ({ id: it.id, type, props: { grid, ...props }, children: [] });
    switch (it.type) {
      case "label": return node("label", { text: p.text });
      case "button": return node("button", { text: p.text });
      case "editfield": return node("editfield", { value: p.value, placeholder: p.placeholder });
      case "numericfield": return node("editfield", { value: String(num(p.value, 0)), numeric: true });
      case "textarea": return node("textarea", { value: p.value });
      case "dropdown": return node("dropdown", { items: p.items, selected: 0 });
      case "listbox": return node("listbox", { items: p.items, selected: [0], multi: false });
      case "checkbox": return node("checkbox", { text: p.text, value: !!p.value });
      case "switch": return node("switch", { items: ["Off", "On"], selected: p.value ? 1 : 0 });
      case "slider": return node("slider", { value: Math.min(Math.max(num(p.value, 0), lim.min), lim.max), ...lim });
      case "spinner": return node("spinner", { value: num(p.value, 0), min: num(p.min, -1e308), max: num(p.max, 1e308), step: num(p.step, 1) });
      case "table": return node("table", { columns: p.columns.map((name) => ({ name, type: "text", editable: false })), rows: [p.columns.map(() => ""), p.columns.map(() => "")] });
      case "axes": return node("axes", { image: AXES_PREVIEW, width: 240, height: 160, alt: p.title || "Plot" });
      case "gauge": return node("gauge", { value: Math.min(Math.max(num(p.value, 0), lim.min), lim.max), ...lim });
      case "lamp": return node("lamp", { color: /^#[0-9a-f]{6}$/i.test(p.color) ? p.color : { red: "#ff0000", green: "#00c000", blue: "#0000ff", yellow: "#ffd400", white: "#ffffff", black: "#000000" }[p.color] || "#00c000" });
      case "datepicker": return node("datepicker", { value: "" });
      default: return node("label", { text: it.type });
    }
  });
  const grid = { id: "grid", type: "gridlayout", props: { rows: d.rows, cols: d.cols, padding: [10, 10, 10, 10], rowGap: 10, colGap: 10 }, children: kids };
  return { figures: [{ id: "fig", type: "figure", props: { name: d.title || d.name, width: d.width, height: d.height }, children: [grid] }], dialogs: [] };
}

/* ------------------------------------------------------------------ design -> code */

const q = (s) => "'" + String(s ?? "").replace(/'/g, "''") + "'";
const cellOf = (list) => "{" + list.map(q).join(", ") + "}";
const sizeOf = (s) => (typeof s === "number" ? String(s) : q(s));

function constructorOf(it) {
  const p = it.props;
  const limits = () => {
    const lo = num(p.min, null);
    const hi = num(p.max, null);
    if (lo == null && hi == null) return "";
    return `, 'Limits', [${lo == null ? "-Inf" : lo} ${hi == null ? "Inf" : hi}]`;
  };
  const range = () => `, 'Limits', [${num(p.min, 0)} ${Math.max(num(p.max, 100), num(p.min, 0) + 1)}]`;
  switch (it.type) {
    case "label": return `uilabel(grid, 'Text', ${q(p.text)})`;
    case "button": return `uibutton(grid, 'Text', ${q(p.text)})`;
    case "editfield": return `uieditfield(grid, 'Value', ${q(p.value)}${p.placeholder ? `, 'Placeholder', ${q(p.placeholder)}` : ""})`;
    case "numericfield": return `uieditfield(grid, 'numeric'${limits()}, 'Value', ${num(p.value, 0)})`;
    case "textarea": return `uitextarea(grid, 'Value', ${cellOf(String(p.value || "").split("\n"))})`;
    case "dropdown": return `uidropdown(grid, 'Items', ${cellOf(p.items)})`;
    case "listbox": return `uilistbox(grid, 'Items', ${cellOf(p.items)})`;
    case "checkbox": return `uicheckbox(grid, 'Text', ${q(p.text)}, 'Value', ${p.value ? "true" : "false"})`;
    case "switch": return `uiswitch(grid, 'Value', ${p.value ? "'On'" : "'Off'"})`;
    case "slider": return `uislider(grid${range()}, 'Value', ${num(p.value, 0)})`;
    case "spinner": return `uispinner(grid${limits()}, 'Step', ${num(p.step, 1) || 1}, 'Value', ${num(p.value, 0)})`;
    case "table": return `uitable(grid, 'ColumnName', ${cellOf(p.columns)}, 'ColumnEditable', ${p.editable ? "true" : "false"}, ...\n    'Data', cell(3, ${p.columns.length}))`;
    case "axes": return "uiaxes(grid)";
    case "gauge": return `uigauge(grid${range()}, 'Value', ${num(p.value, 0)})`;
    case "lamp": return `uilamp(grid, 'Color', ${q(p.color || "green")})`;
    case "datepicker": return "uidatepicker(grid)";
    default: return "uilabel(grid)";
  }
}

/** Bodies of the nested functions of earlier generated code: { name: "lines of the body" }. */
export function callbackBodies(code) {
  const out = {};
  const re = /^ {4}function ([A-Za-z_]\w*)\([^)\n]*\)[^\n]*\n([\s\S]*?)^ {4}end[ \t]*$/gm;
  let m;
  while ((m = re.exec(code || ""))) out[m[1]] = m[2];
  return out;
}

export function generateCode(d, previous = "") {
  const kept = callbackBodies(previous);
  const L = [];
  L.push(`function ${d.name}`);
  L.push(`% ${d.name}  ${d.title || "An app"}`);
  L.push("%");
  L.push("% Made with the mylab app builder. Write your code in the callback functions at the");
  L.push("% end; they can use every component by its name. To change the layout, open this");
  L.push("% file in the app builder again (the design is stored in the last lines of the file).");
  L.push("");
  L.push(`fig = uifigure('Name', ${q(d.title || d.name)}, 'Position', [100 100 ${d.width} ${d.height}]);`);
  L.push(`grid = uigridlayout(fig, [${d.rows.length} ${d.cols.length}]);`);
  L.push(`grid.RowHeight = {${d.rows.map(sizeOf).join(", ")}};`);
  L.push(`grid.ColumnWidth = {${d.cols.map(sizeOf).join(", ")}};`);
  L.push("");
  const ordered = [...d.items].sort((a, b) => a.row - b.row || a.col - b.col);
  for (const it of ordered) {
    const span = (a, n) => (n > 1 ? `[${a} ${a + n - 1}]` : String(a));
    L.push(`${it.name} = ${constructorOf(it)};`);
    L.push(`${it.name}.Layout.Row = ${span(it.row, it.rowSpan)};`);
    L.push(`${it.name}.Layout.Column = ${span(it.col, it.colSpan)};`);
    if (it.type === "axes" && it.props.title) L.push(`title(${it.name}, ${q(it.props.title)});`);
  }
  const withCallbacks = ordered.filter((it) => KINDS[it.type].event && it.callback);
  if (withCallbacks.length) {
    L.push("");
    L.push("% Which function runs when the user does something");
    for (const it of withCallbacks) L.push(`${it.name}.${KINDS[it.type].event[0]} = @${it.callback};`);
  }
  L.push("");
  L.push("startup();");
  L.push("");
  const fn = (name, args, fallback) => {
    L.push(`    function ${name}(${args})`);
    const body = kept[name] != null ? kept[name] : fallback.map((s) => "        " + s + "\n").join("");
    L.push(body.replace(/\n$/, ""));
    L.push("    end");
    L.push("");
  };
  fn("startup", "", ["% Runs once, when the app opens.", "% Example: set the first values, or draw into an axes."]);
  const done = new Set(["startup"]);
  for (const it of withCallbacks) {
    if (done.has(it.callback)) continue;
    done.add(it.callback);
    const what = KINDS[it.type].event[2];
    const example =
      it.type === "table" ? `% event.Indices is [row column]; ${it.name}.Data holds the table.` : it.type === "button" ? "% Example: someLabel.Text = 'Pressed';" : `% The new value is ${it.name}.Value (also event.Value).`;
    fn(it.callback, "src, event", [`% Runs when ${what} (${it.name}).`, example]);
  }
  L.push("end");
  L.push("");
  L.push("%{");
  L.push(`${DESIGN_MARK}: the layout of this app for the mylab app builder. Do not edit these lines.`);
  L.push(JSON.stringify({ ...d, format: FORMAT }));
  L.push("%}");
  return L.join("\n") + "\n";
}

/** The design stored in generated code, or null. */
export function readDesign(code) {
  const i = (code || "").lastIndexOf(DESIGN_MARK + ":");
  if (i < 0) return null;
  const rest = code.slice(i).split("\n");
  try {
    const d = JSON.parse(rest[1]);
    if (!d || d.format !== FORMAT || !Array.isArray(d.items) || !Array.isArray(d.rows) || !Array.isArray(d.cols)) return null;
    d.items = d.items.filter((it) => KINDS[it.type]);
    d.nextId = Math.max(1, d.nextId || 1, ...d.items.map((it) => (Number(String(it.id).replace(/\D/g, "")) || 0) + 1));
    return d;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ the builder */

export function initDesigner(app) {
  let design = null;
  let file = null; // path of the generated file
  let selected = null; // item id
  let preview = false;
  let view = null;
  let undo = [];
  let redo = [];
  let drag = null;

  /* ---------- DOM ---------- */
  const heading = el("h2", { class: "app-view-title", id: "designer-title" }, "App builder");
  const live = el("span", { class: "visually-hidden", role: "status", "aria-live": "polite" });
  const btn = (id, label, title, onclick, ic, cls = "ghost") => el("button", { type: "button", class: `btn sm ${cls}`, id, title, onclick }, ic ? icon(ic) : null, el("span", { class: ic ? "hide-sm" : "" }, label));
  const btnUndo = btn("dsg-undo", "Undo", "Undo (Ctrl+Z)", () => stepHistory(-1), "back");
  const btnRedo = btn("dsg-redo", "Redo", "Redo (Ctrl+Y)", () => stepHistory(1), "fwd");
  const btnPreview = btn("dsg-preview", "Preview", "Show the app without the builder's grid lines and handles", () => setPreview(!preview), "eye");
  const btnCode = btn("dsg-code", "Show code", "Show the code that will be written", () => toggleCode(), "text");
  const btnSave = btn("dsg-save", "Generate code", "Write the .m file and open it in the editor", () => save({ close: true }), "check", "");
  const btnRun = btn("dsg-run", "Run", "Write the .m file and run the app", () => save({ run: true }), "play", "primary");
  const btnClose = el("button", { type: "button", class: "btn icon sm ghost", id: "dsg-close", title: "Close the app builder", "aria-label": "Close the app builder", onclick: () => close() }, icon("close"));
  const palette = el("div", { class: "dsg-palette", role: "group", "aria-label": "Components. Press a button to add one, or drag it onto the grid." });
  const canvas = el("div", { class: "dsg-canvas", id: "dsg-canvas" });
  const codeView = el("pre", { class: "dsg-code", id: "dsg-code-view", hidden: true, tabindex: "0", "aria-label": "Generated code" });
  const props = el("div", { class: "dsg-props", id: "dsg-props" });
  const root = el(
    "section",
    { id: "designer", class: "app-view designer", hidden: true, "aria-labelledby": "designer-title" },
    el("header", { class: "app-view-bar" }, heading, live, el("span", { class: "spacer" }), btnUndo, btnRedo, btnPreview, btnCode, btnSave, btnRun, btnClose),
    el(
      "div",
      { class: "dsg-body" },
      el("aside", { class: "dsg-side", "aria-label": "Components" }, el("h3", {}, "Components"), palette, el("p", { class: "muted small" }, "Click to add, or drag onto the grid.")),
      el("div", { class: "dsg-center" }, el("p", { class: "muted small dsg-hint" }, "Select a component, then drag it or use the arrow keys to move it. Shift+arrows change its size, Delete removes it."), canvas, codeView),
      el("aside", { class: "dsg-side dsg-right", "aria-label": "Properties" }, props)
    )
  );
  document.body.append(root);

  for (const [type, kind] of Object.entries(KINDS)) {
    const b = el("button", { type: "button", class: "dsg-tool", "data-type": type, title: `Add a ${kind.title.toLowerCase()}`, onclick: () => !b.dataset.dragged && addItem(type) }, kind.title);
    b.addEventListener("pointerdown", (e) => startDrag(e, { kind: "new", type, button: b }));
    palette.append(b);
  }

  /* ---------- history ---------- */
  function commit(message) {
    undo.push(JSON.stringify(design));
    if (undo.length > 200) undo.shift();
    redo = [];
    if (message) live.textContent = message;
  }
  /** Call before changing the design. */
  const change = (message, fn) => {
    commit(message);
    fn();
    render();
  };
  function stepHistory(dir) {
    const from = dir < 0 ? undo : redo;
    const to = dir < 0 ? redo : undo;
    if (!from.length) return;
    to.push(JSON.stringify(design));
    design = JSON.parse(from.pop());
    if (!design.items.some((it) => it.id === selected)) selected = null;
    live.textContent = dir < 0 ? "Undone." : "Redone.";
    render();
  }

  /* ---------- model helpers ---------- */
  const item = (id) => design.items.find((it) => it.id === id) || null;
  const occupied = (r, c, except) => design.items.some((it) => it !== except && r >= it.row && r < it.row + it.rowSpan && c >= it.col && c < it.col + it.colSpan);

  function uniqueName(base) {
    let n = 1;
    const used = new Set(design.items.map((it) => it.name));
    while (used.has(base + n) || RESERVED.has(base + n)) n++;
    return base + n;
  }

  function freeCell() {
    for (let r = 1; r <= design.rows.length; r++) for (let c = 1; c <= design.cols.length; c++) if (!occupied(r, c)) return [r, c];
    design.rows.push("fit");
    return [design.rows.length, 1];
  }

  function addItem(type, at = null) {
    const kind = KINDS[type];
    change(`${kind.title} added.`, () => {
      const [row, col] = at && !occupied(at[0], at[1]) ? at : freeCell();
      const name = uniqueName(kind.base);
      const it = { id: "c" + design.nextId++, type, name, row, col, rowSpan: 1, colSpan: 1, props: clone(kind.props), callback: kind.event ? name + kind.event[1] : "" };
      if (type === "axes" || type === "table" || type === "textarea" || type === "listbox") {
        // big components want a row that stretches
        if (design.rows[row - 1] === "fit" && !design.items.some((o) => o.row === row)) design.rows[row - 1] = "1x";
      }
      design.items.push(it);
      selected = it.id;
    });
    focusSelected();
  }

  function moveTo(it, row, col, rowSpan = it.rowSpan, colSpan = it.colSpan) {
    row = Math.max(1, row);
    col = Math.max(1, col);
    rowSpan = Math.max(1, rowSpan);
    colSpan = Math.max(1, colSpan);
    while (design.rows.length < row + rowSpan - 1) design.rows.push("fit");
    while (design.cols.length < col + colSpan - 1) design.cols.push("1x");
    Object.assign(it, { row, col, rowSpan, colSpan });
  }

  function removeItem(id) {
    const it = item(id);
    if (!it) return;
    change(`${KINDS[it.type].title} ${it.name} removed.`, () => {
      design.items = design.items.filter((x) => x.id !== id);
      selected = null;
    });
    canvas.focus();
  }

  function removeTrack(which, index) {
    const list = which === "row" ? design.rows : design.cols;
    if (list.length <= 1) return;
    const key = which === "row" ? "row" : "col";
    const span = which === "row" ? "rowSpan" : "colSpan";
    if (design.items.some((it) => index >= it[key] && index < it[key] + it[span])) return toast(`Move or delete the components in that ${which === "row" ? "row" : "column"} first.`);
    change(`${which === "row" ? "Row" : "Column"} ${index} removed.`, () => {
      list.splice(index - 1, 1);
      for (const it of design.items) if (it[key] > index) it[key]--;
    });
  }

  /* ---------- drawing ---------- */
  function render() {
    btnUndo.disabled = !undo.length;
    btnRedo.disabled = !redo.length;
    heading.textContent = `App builder: ${design.name}.m`;
    if (view) view.destroy();
    canvas.textContent = "";
    view = globalThis.MylabUI.mount(canvas, { readOnly: true });
    view.setTree(designTree(design));
    const gridEl = canvas.querySelector(".mui-gridlayout");
    canvas.classList.toggle("dsg-previewing", preview);
    if (!preview && gridEl) {
      for (let r = 1; r <= design.rows.length; r++) {
        for (let c = 1; c <= design.cols.length; c++) {
          gridEl.prepend(el("div", { class: "dsg-cell", "data-row": r, "data-col": c, style: `grid-row:${r};grid-column:${c}` }));
        }
      }
      for (const it of design.items) {
        const node = canvas.querySelector(`[data-mui-id="${it.id}"]`);
        if (!node) continue;
        const kind = KINDS[it.type];
        const cover = el("div", {
          class: "dsg-item" + (it.id === selected ? " selected" : ""), tabindex: "0", role: "button", "data-id": it.id,
          style: `grid-row:${it.row} / ${it.row + it.rowSpan};grid-column:${it.col} / ${it.col + it.colSpan}`,
          "aria-label": `${kind.title} ${it.name}, row ${it.row}, column ${it.col}${it.rowSpan > 1 || it.colSpan > 1 ? `, ${it.rowSpan} rows by ${it.colSpan} columns` : ""}`,
          "aria-pressed": String(it.id === selected),
        });
        cover.addEventListener("pointerdown", (e) => {
          select(it.id, false);
          startDrag(e, { kind: e.target.classList.contains("dsg-handle") ? "resize" : "move", id: it.id });
        });
        cover.addEventListener("focus", () => selected !== it.id && select(it.id, false));
        cover.addEventListener("keydown", (e) => itemKey(e, it));
        if (it.id === selected) cover.append(el("span", { class: "dsg-handle", title: "Drag to change the size", "aria-hidden": "true" }), el("span", { class: "dsg-tag" }, it.name));
        gridEl.append(cover);
      }
    }
    renderProps();
    if (!codeView.hidden) codeView.textContent = generateCode(design, previousCode());
  }

  function select(id, focus = true) {
    if (selected === id) return;
    selected = id;
    for (const c of canvas.querySelectorAll(".dsg-item")) {
      const on = c.dataset.id === id;
      c.classList.toggle("selected", on);
      c.setAttribute("aria-pressed", String(on));
      c.querySelectorAll(".dsg-handle, .dsg-tag").forEach((x) => x.remove());
      if (on) c.append(el("span", { class: "dsg-handle", title: "Drag to change the size", "aria-hidden": "true" }), el("span", { class: "dsg-tag" }, item(id).name));
    }
    renderProps();
    if (focus) focusSelected();
  }

  function focusSelected() {
    const c = selected && canvas.querySelector(`.dsg-item[data-id="${selected}"]`);
    if (c) c.focus({ preventScroll: true });
  }

  function itemKey(e, it) {
    const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (d) {
      e.preventDefault();
      const [dr, dc] = d;
      if (e.shiftKey) {
        if (it.rowSpan + dr < 1 || it.colSpan + dc < 1) return;
        change(`${it.name} is now ${it.rowSpan + dr} rows by ${it.colSpan + dc} columns.`, () => moveTo(it, it.row, it.col, it.rowSpan + dr, it.colSpan + dc));
      } else {
        if (it.row + dr < 1 || it.col + dc < 1) return;
        change(`${it.name} moved to row ${it.row + dr}, column ${it.col + dc}.`, () => moveTo(it, it.row + dr, it.col + dc));
      }
      focusSelected();
    } else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      removeItem(it.id);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const first = props.querySelector("input, textarea, select");
      if (first) first.focus();
    }
  }

  /* ---------- dragging (mouse and touch) ---------- */
  function cellAt(x, y) {
    for (const node of document.elementsFromPoint(x, y)) if (node.classList && node.classList.contains("dsg-cell")) return [Number(node.dataset.row), Number(node.dataset.col)];
    return null;
  }

  function startDrag(e, info) {
    if (e.button > 0 || preview) return;
    drag = { ...info, x: e.clientX, y: e.clientY, moved: false };
    if (info.button) delete info.button.dataset.dragged;
    window.addEventListener("pointermove", onDragMove);
    window.addEventListener("pointerup", onDragEnd, { once: true });
    window.addEventListener("pointercancel", onDragEnd, { once: true });
    if (info.kind !== "new") e.preventDefault();
  }

  function onDragMove(e) {
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 6) return;
    drag.moved = true;
    document.body.classList.add("dsg-dragging");
    canvas.querySelectorAll(".dsg-cell.target").forEach((c) => c.classList.remove("target"));
    const at = cellAt(e.clientX, e.clientY);
    if (!at) return;
    const it = drag.id ? item(drag.id) : null;
    let r0 = at[0], c0 = at[1], r1 = at[0], c1 = at[1];
    if (drag.kind === "move" && it) {
      r1 = r0 + it.rowSpan - 1;
      c1 = c0 + it.colSpan - 1;
    } else if (drag.kind === "resize" && it) {
      r0 = it.row;
      c0 = it.col;
      r1 = Math.max(at[0], it.row);
      c1 = Math.max(at[1], it.col);
    }
    for (const c of canvas.querySelectorAll(".dsg-cell")) {
      const r = Number(c.dataset.row), k = Number(c.dataset.col);
      if (r >= r0 && r <= r1 && k >= c0 && k <= c1) c.classList.add("target");
    }
  }

  function onDragEnd(e) {
    window.removeEventListener("pointermove", onDragMove);
    document.body.classList.remove("dsg-dragging");
    const d = drag;
    drag = null;
    if (!d || !d.moved) return;
    if (d.button) {
      d.button.dataset.dragged = "1"; // swallow the click that follows the drag
      setTimeout(() => delete d.button.dataset.dragged, 0);
    }
    const at = e.type === "pointerup" ? cellAt(e.clientX, e.clientY) : null;
    if (!at) return render();
    if (d.kind === "new") return addItem(d.type, at);
    const it = item(d.id);
    if (!it) return;
    if (d.kind === "move") {
      if (at[0] === it.row && at[1] === it.col) return render();
      change(`${it.name} moved to row ${at[0]}, column ${at[1]}.`, () => moveTo(it, at[0], at[1]));
    } else {
      const rs = Math.max(1, at[0] - it.row + 1), cs = Math.max(1, at[1] - it.col + 1);
      if (rs === it.rowSpan && cs === it.colSpan) return render();
      change(`${it.name} is now ${rs} rows by ${cs} columns.`, () => moveTo(it, it.row, it.col, rs, cs));
    }
    focusSelected();
  }

  /* ---------- properties ---------- */
  function field(label, control, help) {
    return el("label", { class: "dsg-field" }, el("span", {}, label), control, help ? el("small", { class: "muted" }, help) : null);
  }

  /** An input bound to a getter/setter; the design changes when the field is committed. */
  function bound(kind, get, set, attrs = {}) {
    const tag = kind === "area" ? "textarea" : kind === "select" ? "select" : "input";
    const input = el(tag, { class: "dsg-input", ...(tag === "input" ? { type: kind === "check" ? "checkbox" : kind === "number" ? "number" : "text" } : {}), ...attrs });
    if (kind === "select") for (const o of attrs.options) input.append(el("option", { value: o }, o));
    if (kind === "check") input.checked = !!get();
    else input.value = get() ?? "";
    input.addEventListener("change", () => {
      const before = JSON.stringify(design);
      const error = set(kind === "check" ? input.checked : input.value);
      if (typeof error === "string") {
        toast(error, "error");
        if (kind !== "check") input.value = get() ?? "";
        return;
      }
      if (JSON.stringify(design) === before) return;
      undo.push(before);
      redo = [];
      const keep = input.dataset.key;
      render();
      const again = keep && props.querySelector(`[data-key="${keep}"]`);
      if (again) again.focus();
    });
    return input;
  }

  function sizeList(list) {
    return list.map((s) => String(s)).join(", ");
  }

  function parseSizes(text, fallbackCount) {
    const parts = String(text).split(/[,\s]+/).filter(Boolean);
    if (!parts.length) return "Type one size per row or column, for example: fit, 1x, 120";
    const out = [];
    for (const p of parts) {
      if (/^fit$/i.test(p)) out.push("fit");
      else if (/^\d*\.?\d+x$/i.test(p)) out.push(p.toLowerCase());
      else if (/^\d+$/.test(p)) out.push(Number(p));
      else return `"${p}" is not a size. Use fit, a weight such as 1x, or a number of pixels.`;
    }
    return fallbackCount && out.length < fallbackCount ? `There are components in ${fallbackCount} of them; keep at least ${fallbackCount} sizes.` : out;
  }

  function renderProps() {
    props.textContent = "";
    const it = item(selected);
    const key = (k) => ({ "data-key": k });
    if (!it) {
      props.append(el("h3", {}, "App"));
      props.append(field("Function and file name", bound("text", () => design.name, (v) => {
        if (!isName(v)) return "The name must start with a letter and use only letters, digits and _.";
        design.name = v;
        file = (file && file.includes("/") ? file.slice(0, file.lastIndexOf("/") + 1) : "") + v + ".m";
      }, key("name"))));
      props.append(field("Window title", bound("text", () => design.title, (v) => void (design.title = v), key("title"))));
      props.append(field("Width (pixels)", bound("number", () => design.width, (v) => void (design.width = Math.min(Math.max(Math.round(num(v, 480)), 160), 2000)), { min: 160, max: 2000, ...key("width") })));
      props.append(field("Height (pixels)", bound("number", () => design.height, (v) => void (design.height = Math.min(Math.max(Math.round(num(v, 360)), 100), 2000)), { min: 100, max: 2000, ...key("height") })));
      const maxRow = Math.max(0, ...design.items.map((x) => x.row + x.rowSpan - 1));
      const maxCol = Math.max(0, ...design.items.map((x) => x.col + x.colSpan - 1));
      props.append(field("Row heights", bound("text", () => sizeList(design.rows), (v) => {
        const r = parseSizes(v, maxRow);
        if (typeof r === "string") return r;
        design.rows = r;
      }, key("rows")), "fit = as tall as the content, 1x = a share of the free space, or pixels"));
      props.append(field("Column widths", bound("text", () => sizeList(design.cols), (v) => {
        const r = parseSizes(v, maxCol);
        if (typeof r === "string") return r;
        design.cols = r;
      }, key("cols"))));
      const b = (id, label, fn) => el("button", { type: "button", class: "btn sm", id, onclick: fn }, label);
      props.append(
        el("div", { class: "dsg-buttons" },
          b("dsg-add-row", "Add a row", () => change("Row added.", () => design.rows.push("fit"))),
          b("dsg-add-col", "Add a column", () => change("Column added.", () => design.cols.push("1x"))),
          b("dsg-del-row", "Remove the last row", () => removeTrack("row", design.rows.length)),
          b("dsg-del-col", "Remove the last column", () => removeTrack("col", design.cols.length))
        )
      );
      props.append(el("p", { class: "muted small" }, "Select a component on the grid to change its text, items, limits, name and callback."));
      return;
    }
    const kind = KINDS[it.type];
    const p = it.props;
    props.append(el("h3", {}, kind.title));
    for (const f of kind.fields) {
      const label = FIELD_LABELS[f];
      if (f === "text") props.append(field(label, bound("text", () => p.text, (v) => void (p.text = v), key(f))));
      else if (f === "value") props.append(field(label, bound("text", () => p.value, (v) => void (p.value = v), key(f))));
      else if (f === "placeholder") props.append(field(label, bound("text", () => p.placeholder, (v) => void (p.placeholder = v), key(f))));
      else if (f === "plotTitle") props.append(field(label, bound("text", () => p.title, (v) => void (p.title = v), key(f))));
      else if (f === "color") props.append(field(label, bound("text", () => p.color, (v) => void (p.color = v.trim() || "green"), key(f))));
      else if (f === "lines") props.append(field(label, bound("area", () => p.value, (v) => void (p.value = v), { rows: 3, ...key(f) })));
      else if (f === "items" || f === "columns") {
        const k = f === "items" ? "items" : "columns";
        props.append(field(label, bound("area", () => p[k].join("\n"), (v) => {
          const list = String(v).split("\n").map((s) => s.trim()).filter(Boolean);
          if (!list.length) return "Type at least one line.";
          p[k] = list;
        }, { rows: 4, ...key(f) })));
      } else if (f === "checked" || f === "editable") {
        const k = f === "checked" ? "value" : "editable";
        props.append(el("label", { class: "dsg-field dsg-check" }, bound("check", () => p[k], (v) => void (p[k] = !!v), key(f)), el("span", {}, label)));
      } else if (f === "number" || f === "min" || f === "max" || f === "step") {
        const k = f === "number" ? "value" : f;
        props.append(field(label, bound("text", () => p[k], (v) => {
          if (String(v).trim() !== "" && Number.isNaN(Number(v))) return "Type a number.";
          p[k] = String(v).trim() === "" ? "" : Number(v);
        }, { inputmode: "decimal", ...key(f) })));
      }
    }
    props.append(field("Name in the code (Tag)", bound("text", () => it.name, (v) => {
      if (!isName(v) || RESERVED.has(v)) return "A name starts with a letter and uses only letters, digits and _ (and is not a word such as fig or end).";
      if (design.items.some((o) => o !== it && o.name === v)) return "Another component already has that name.";
      if (kind.event && it.callback === it.name + kind.event[1]) it.callback = v + kind.event[1];
      it.name = v;
    }, key("name")), "The callbacks use this name, for example " + it.name + (kind.fields.includes("text") ? ".Text" : ".Value")));
    if (kind.event) {
      props.append(field("Callback function", bound("text", () => it.callback, (v) => {
        if (v && (!isName(v) || RESERVED.has(v))) return "A function name starts with a letter and uses only letters, digits and _.";
        it.callback = v;
      }, key("callback")), `Runs when ${kind.event[2]}. Leave empty for none.`));
    }
    const pos = (label, k, min) => field(label, bound("number", () => it[k], (v) => {
      const n = Math.max(min, Math.round(num(v, it[k])));
      const next = { row: it.row, col: it.col, rowSpan: it.rowSpan, colSpan: it.colSpan, [k]: n };
      moveTo(it, next.row, next.col, next.rowSpan, next.colSpan);
    }, { min, max: 50, ...key(k) }));
    props.append(el("div", { class: "dsg-pos" }, pos("Row", "row", 1), pos("Column", "col", 1), pos("Rows tall", "rowSpan", 1), pos("Columns wide", "colSpan", 1)));
    props.append(el("div", { class: "dsg-buttons" }, el("button", { type: "button", class: "btn sm danger", id: "dsg-delete", onclick: () => removeItem(it.id) }, icon("trash"), "Delete"), el("button", { type: "button", class: "btn sm", id: "dsg-deselect", onclick: () => (select(null, false), canvas.focus()) }, "App settings")));
  }

  /* ---------- preview, code, saving ---------- */
  function setPreview(on) {
    preview = on;
    btnPreview.setAttribute("aria-pressed", String(on));
    render();
  }

  function previousCode() {
    return file && app.state.files.has(file) && app.isText(file) ? app.liveContent(file) : "";
  }

  function toggleCode() {
    codeView.hidden = !codeView.hidden;
    canvas.hidden = !codeView.hidden;
    btnCode.setAttribute("aria-pressed", String(!codeView.hidden));
    btnCode.querySelector("span").textContent = codeView.hidden ? "Show code" : "Show design";
    if (!codeView.hidden) codeView.textContent = generateCode(design, previousCode());
  }

  async function save({ close: closeAfter = false, run = false } = {}) {
    const code = generateCode(design, previousCode());
    if (app.state.files.has(file)) {
      if (app.state.active !== file) app.openFile(file, { focus: false });
      app.replaceText(file, code);
      await app.flushSave(file, { manual: true });
    } else {
      await app.createFile(file, code);
    }
    live.textContent = `Code written to ${file}.`;
    if (closeAfter || run) close();
    if (run) {
      if (!app.canRun()) return toast("The engine is still loading. Press Run when it is ready.");
      await app.runActiveFile();
    } else {
      toast(`Wrote ${file}. Fill in the callback functions at the end of the file.`);
    }
  }

  function open(d, path) {
    design = d;
    file = path;
    selected = null;
    undo = [];
    redo = [];
    preview = false;
    btnPreview.setAttribute("aria-pressed", "false");
    codeView.hidden = true;
    canvas.hidden = false;
    btnCode.setAttribute("aria-pressed", "false");
    btnCode.querySelector("span").textContent = "Show code";
    if (app.hideApp) app.hideApp();
    root.hidden = false;
    render();
    palette.querySelector("button").focus();
  }

  function close() {
    root.hidden = true;
    if (view) view.destroy();
    view = null;
    if (app.editor && app.state.active) app.editor.focus();
  }

  async function newApp() {
    let n = 1;
    while (app.state.files.has(`MyApp${n > 1 ? n : ""}.m`)) n++;
    const name = await promptDialog({
      title: "New app",
      text: "Name of the app. It becomes the name of the .m file and of its function.",
      value: `MyApp${n > 1 ? n : ""}`,
      okLabel: "Open the app builder",
      selectAll: true,
      validate: (v) => {
        const s = v.trim().replace(/\.m$/, "");
        if (!isName(s)) return { error: "Start with a letter; use only letters, digits and _." };
        if (app.state.files.has(s + ".m")) return { error: "A file with that name already exists." };
        return { name: s };
      },
    });
    if (!name) return;
    const folder = app.state.active && app.state.active.includes("/") ? app.dirName(app.state.active) + "/" : "";
    open(newDesign(name), folder + name + ".m");
  }

  function designOfActive() {
    const path = app.state.active;
    return path && app.isText(path) ? readDesign(app.liveContent(path)) : null;
  }

  function openActive() {
    const d = designOfActive();
    if (!d) return toast("This file was not made with the app builder (it has no stored design).");
    d.name = app.baseName(app.state.active).replace(/\.m$/, "");
    open(d, app.state.active);
  }

  /** After the canvas was redrawn from the keyboard, put the focus back on it. */
  const refocus = () => (selected && canvas.querySelector(`.dsg-item[data-id="${selected}"]`) ? focusSelected() : canvas.focus());

  root.addEventListener("keydown", (e) => {
    const mod = e.ctrlKey || e.metaKey;
    const typing = e.target.matches("input, textarea, select");
    if (mod && !e.altKey && e.key.toLowerCase() === "z" && !typing) {
      e.preventDefault();
      stepHistory(e.shiftKey ? 1 : -1);
      refocus();
    } else if (mod && e.key.toLowerCase() === "y" && !typing) {
      e.preventDefault();
      stepHistory(1);
      refocus();
    } else if (e.key === "Escape") {
      e.stopPropagation();
      if (typing) e.target.blur(), focusSelected();
      else if (selected) select(null, false), canvas.focus();
      else close();
    }
    if (mod) e.stopPropagation(); // the editor's shortcuts stay out of the builder
  });
  canvas.tabIndex = -1;

  // A hint in the editor: files made by the builder can go back into it.
  const reopen = el("button", { id: "btn-design", class: "btn", hidden: true, type: "button", title: "Open this file in the app builder to change its layout", onclick: openActive }, icon("pencil"), el("span", { class: "hide-sm" }, "Edit layout"));
  ($("btn-app") || $("btn-publish")).after(reopen);
  const refreshReopen = () => (reopen.hidden = !designOfActive());
  for (const name of ["fileOpened", "tabsChanged", "saved", "projectChanged"]) app.on(name, refreshReopen);

  app.commands.register([
    { id: "app.new", group: "App", title: "New app (open the app builder)", run: newApp },
    { id: "app.design", group: "App", title: "Open this file in the app builder", run: openActive, when: () => !!designOfActive() },
    { id: "app.designUndo", group: "App", title: "App builder: undo", run: () => stepHistory(-1), when: () => !root.hidden && undo.length > 0, hidden: true },
    { id: "app.designRedo", group: "App", title: "App builder: redo", run: () => stepHistory(1), when: () => !root.hidden && redo.length > 0, hidden: true },
    { id: "app.designCode", group: "App", title: "App builder: generate the code", run: () => save({ close: true }), when: () => !root.hidden },
    ...["unit_converter", "plot_explorer", "todo_list"].map((name) => ({
      id: "app.example." + name, group: "App", title: `Example app: ${name.replace("_", " ")}`,
      run: async () => {
        const res = await fetch(`samples/apps/${name}.m`);
        if (!res.ok) return toast("The example could not be loaded.", "error");
        const code = await res.text();
        if (!app.state.files.has(`${name}.m`)) await app.createFile(`${name}.m`, code);
        else app.openFile(`${name}.m`);
      },
    })),
  ]);

  Object.assign(app, { newApp, openDesigner: open, openDesignerForActive: openActive, designer: { get design() { return design; }, generateCode, readDesign } });
}
