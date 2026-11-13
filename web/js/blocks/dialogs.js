// Dialogs of the diagram editor: block parameters, simulation settings, pick lists, key help.

import { el } from "../ui.js";
import { findType, paramsOf, countPorts, signature } from "./model.js";

function openDialog({ title, body, okLabel = "OK", wide = false, onOk, cancelLabel = "Cancel", id = "" }) {
  return new Promise((resolve) => {
    const error = el("p", { class: "small bd-dlg-error", role: "alert", hidden: true });
    const ok = el("button", { class: "btn primary", type: "submit", value: "ok" }, okLabel);
    const form = el(
      "form",
      { method: "dialog", novalidate: true },
      el("h2", {}, title),
      ...[body].flat(),
      error,
      el("menu", {}, cancelLabel ? el("button", { class: "btn", type: "button", onclick: () => dlg.close("cancel") }, cancelLabel) : null, ok)
    );
    const dlg = el("dialog", { class: "dialog bd-dialog" + (wide ? " wide" : ""), id: id || null }, form);
    let result = null;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      ok.disabled = true;
      let out;
      try {
        out = onOk ? await onOk({ ok, error }) : { value: true };
      } finally {
        ok.disabled = false;
      }
      if (out && out.error) {
        error.textContent = out.error;
        error.hidden = false;
        return;
      }
      if (out && out.again) return;
      result = out ? out.value : true;
      dlg.close("ok");
    });
    dlg.addEventListener("close", () => {
      dlg.remove();
      resolve(dlg.returnValue === "ok" ? result : null);
    });
    document.body.appendChild(dlg);
    dlg.showModal();
    const first = dlg.querySelector("[data-first]") || dlg.querySelector("input, select, textarea");
    if (first) {
      first.focus();
      if (first.select && first.tagName === "INPUT") first.select();
    }
  });
}

/** Problems that can be seen without evaluating: empty values, unbalanced brackets or quotes. */
export function checkExpression(text) {
  const s = String(text).trim();
  if (!s) return "Type a value.";
  const pairs = { ")": "(", "]": "[", "}": "{" };
  const stack = [];
  let quote = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quote) {
      if (c === "'") {
        if (s[i + 1] === "'") i++;
        else quote = false;
      }
      continue;
    }
    if (c === '"') {
      const j = s.indexOf('"', i + 1);
      if (j < 0) return 'A " is not closed.';
      i = j;
    } else if (c === "'" && (i === 0 || /[\s,;([{=+\-*/<>~&|:]/.test(s[i - 1]))) quote = true;
    else if ("([{".includes(c)) stack.push(c);
    else if (pairs[c]) {
      if (stack.pop() !== pairs[c]) return `There is a "${c}" without its opening bracket.`;
    }
  }
  if (quote) return "A ' is not closed.";
  if (stack.length) return `A "${stack[stack.length - 1]}" is not closed.`;
  return null;
}

function checkParam(spec, value, type) {
  const v = String(value);
  if (spec.kind === "expr") return checkExpression(v);
  if (spec.kind === "code") {
    if (!/^\s*function\b/m.test(v)) return "The code must start with a function line, for example: function y = fcn(u)";
    return null;
  }
  if (spec.kind === "text") {
    if (!v.trim() && spec.name !== "Title") return "Type a value.";
    const counted = [type.in, type.out].some((rule) => JSON.stringify(rule).includes(`"${spec.name}"`));
    if (spec.name === "VariableName" && type.type === "ToWorkspace" && !/^[A-Za-z]\w*$/.test(v.trim())) return "A variable name starts with a letter and has only letters, digits and _.";
    if (spec.name === "Port" && !/^[1-9]\d*$/.test(v.trim())) return "The port number is a whole number, 1 or more.";
    if (counted && type.type !== "MATLABFunction") {
      const t = v.replace(/\s/g, "");
      const signs = ["Sum"].includes(type.type) ? /^[|+-]+$/ : ["Product", "Divide"].includes(type.type) ? /^[|*/]+$/ : null;
      const isCount = /^[1-9]\d?$/.test(t);
      if (type.type === "BusSelector") return /^[^,]+(,[^,]+)*$/.test(v.trim()) ? null : "List signal names separated by commas.";
      if (type.type === "Demux" && /^\[[\d\s,]+\]$/.test(t)) return null;
      if (!isCount && !(signs && signs.test(t))) {
        return signs ? `Type a number of inputs (1 to 99) or a list of signs such as ${type.type === "Sum" ? "+-" : "*/"}.` : "Type a whole number from 1 to 99.";
      }
    }
  }
  return null;
}

/**
 * Edit the parameters of a block. validate(values) may ask the engine: it resolves to { name: message }.
 * Resolves to { name, params, mask } or null.
 */
export function paramsDialog(block, { taken, validate }) {
  const type = findType(block.type);
  const values = paramsOf(block);
  const fields = new Map();
  const notes = new Map();
  const rows = [];
  const nameInput = el("input", { type: "text", value: block.name, spellcheck: "false", autocomplete: "off", "aria-label": "Block name" });
  rows.push(el("label", { class: "field" }, "Name", nameInput));
  const nameNote = el("span", { class: "bd-field-note", hidden: true });
  rows.push(nameNote);
  if (!type) {
    rows.push(el("p", { class: "small" }, `mylab Blocks has no block of the type "${block.type}". The model cannot run until this block is replaced by blocks that do the same job.`));
  }
  for (const spec of type ? type.params : []) {
    let input;
    const v = values[spec.name];
    const id = "bd-p-" + spec.name;
    if (spec.kind === "enum") {
      const opts = spec.options.includes(v) ? spec.options : [v, ...spec.options];
      input = el("select", { id }, opts.map((o) => el("option", { value: o, selected: o === v }, o)));
    } else if (spec.kind === "bool") {
      input = el("input", { id, type: "checkbox", checked: String(v).toLowerCase() === "on" });
    } else if (spec.kind === "code") {
      input = el("textarea", { id, class: "bd-code", rows: "10", spellcheck: "false", autocomplete: "off" });
      input.value = v;
    } else {
      input = el("input", { id, type: "text", value: v, spellcheck: "false", autocomplete: "off", class: spec.kind === "expr" ? "mono" : "" });
    }
    const note = el("span", { class: "bd-field-note", id: id + "-note", hidden: true });
    input.setAttribute("aria-describedby", id + "-note");
    fields.set(spec.name, { spec, input });
    notes.set(spec.name, note);
    if (spec.kind === "bool") rows.push(el("label", { class: "bd-check" }, input, spec.label), note);
    else rows.push(el("label", { class: "field", for: id }, el("span", {}, spec.label, spec.kind === "expr" ? el("span", { class: "muted bd-kind" }, " · expression") : null), input), note);
  }
  // mask variables of a subsystem
  let maskInput = null;
  if (type && type.type === "Subsystem") {
    maskInput = el("textarea", { class: "bd-code", rows: "3", spellcheck: "false", placeholder: "k = 2\ntau = 0.5", "aria-label": "Subsystem variables" });
    maskInput.value = Object.entries(block.mask || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
    rows.push(el("label", { class: "field" }, el("span", {}, "Variables of this subsystem ", el("span", { class: "muted bd-kind" }, "· one per line, name = expression")), maskInput));
    rows.push(el("p", { class: "small muted" }, "Blocks inside the subsystem can use these names in their parameters. Press Enter on the block (or double-click it) to look inside."));
  }
  if (type) rows.unshift(el("p", { class: "small muted" }, type.doc));
  const read = () => {
    const out = {};
    for (const [name, f] of fields) out[name] = f.spec.kind === "bool" ? (f.input.checked ? "on" : "off") : f.input.value;
    return out;
  };
  const show = (name, text, level = "error") => {
    const note = name === "__name" ? nameNote : notes.get(name);
    note.textContent = text || "";
    note.hidden = !text;
    note.className = "bd-field-note " + level;
    const input = name === "__name" ? nameInput : fields.get(name).input;
    input.setAttribute("aria-invalid", text && level === "error" ? "true" : "false");
  };
  let warned = "";
  return openDialog({
    title: type ? `${type.title}: ${block.name}` : block.name,
    body: el("div", { class: "bd-params" }, rows),
    wide: !!(type && type.params.some((p) => p.kind === "code")),
    id: "bd-dlg-params",
    onOk: async ({ ok }) => {
      const vals = read();
      let bad = 0;
      const name = nameInput.value.trim().replace(/\s+/g, " ");
      if (!name) (show("__name", "A block needs a name."), bad++);
      else if (name !== block.name && taken.has(name)) (show("__name", `There is already a block named "${name}" here.`), bad++);
      else show("__name", "");
      for (const [pname, f] of fields) {
        const msg = type ? checkParam(f.spec, vals[pname], type) : null;
        show(pname, msg);
        if (msg) bad++;
      }
      let mask = null;
      if (maskInput) {
        mask = {};
        for (const line of maskInput.value.split("\n")) {
          if (!line.trim()) continue;
          const m = /^\s*([A-Za-z]\w*)\s*=\s*(.+?)\s*;?\s*$/.exec(line);
          if (!m || checkExpression(m[2])) return { error: `Subsystem variables: "${line.trim()}" is not of the form name = expression.` };
          mask[m[1]] = m[2];
        }
      }
      if (bad) {
        const firstBad = [...fields.values()].map((f) => f.input).concat(nameInput).find((i) => i.getAttribute("aria-invalid") === "true");
        firstBad && firstBad.focus();
        return { again: true };
      }
      // ask the engine whether the expressions can be evaluated; a second OK saves anyway
      const exprs = {};
      for (const [pname, f] of fields) if (f.spec.kind === "expr") exprs[pname] = vals[pname];
      const sig = JSON.stringify(exprs);
      if (validate && Object.keys(exprs).length && warned !== sig) {
        let problems = null;
        try {
          problems = await validate(exprs);
        } catch {
          problems = null; // the engine is busy or still loading: nothing to add
        }
        const names = problems ? Object.keys(problems).filter((k) => problems[k] && fields.has(k)) : [];
        if (names.length) {
          for (const k of names) show(k, `Cannot be evaluated right now: ${problems[k]}`, "warn");
          warned = sig;
          ok.textContent = "Save anyway";
          return { error: "Some values cannot be evaluated with the variables that exist now. If you will define them before running, press Save anyway." };
        }
      }
      // keep only the values that differ from the defaults, like the file format does
      const params = {};
      for (const spec of type ? type.params : []) if (vals[spec.name] !== spec.default || spec.name in (block.params || {})) params[spec.name] = vals[spec.name];
      return { value: { name, params, mask } };
    },
  });
}

/** Port counts a block would have with other parameter values (to warn before lines are dropped). */
export function portsWith(block, params) {
  const type = findType(block.type);
  if (!type || type.type === "Subsystem") return null;
  const p = { ...paramsOf(block), ...params };
  return { in: countPorts(type.in, p), out: countPorts(type.out, p), sig: type.type === "MATLABFunction" ? signature(p.Script) : null };
}

const SOLVERS = [
  ["ode45", "ode45: variable step, the usual choice"], ["ode23", "ode23: variable step, lower order"], ["ode113", "ode113: variable step, high order"],
  ["ode15s", "ode15s: variable step, for stiff models"], ["ode23s", "ode23s: variable step, for stiff models"],
  ["ode4", "ode4: fixed step, Runge-Kutta"], ["ode1", "ode1: fixed step, Euler"], ["ode2", "ode2: fixed step, Heun"], ["ode3", "ode3: fixed step"], ["ode5", "ode5: fixed step"],
  ["FixedStepDiscrete", "Discrete: fixed step, no continuous states"],
];

export function settingsDialog(config) {
  const cfg = { ...config };
  const field = (name, label, hint) => {
    const input = el("input", { type: "text", value: cfg[name] ?? "", spellcheck: "false", autocomplete: "off", class: "mono", "data-cfg": name });
    return el("label", { class: "field" }, el("span", {}, label, hint ? el("span", { class: "muted bd-kind" }, " · " + hint) : null), input);
  };
  const solver = el("select", { "data-cfg": "Solver", "data-first": true }, (SOLVERS.some((s) => s[0] === cfg.Solver) ? SOLVERS : [[cfg.Solver, cfg.Solver], ...SOLVERS]).map(([v, text]) => el("option", { value: v, selected: v === cfg.Solver }, text)));
  const zc = el("input", { type: "checkbox", checked: String(cfg.ZeroCrossings).toLowerCase() !== "off" });
  const body = el(
    "div",
    { class: "bd-params" },
    el("div", { class: "bd-two" }, field("StartTime", "Start time", "s"), field("StopTime", "Stop time", "s")),
    el("label", { class: "field" }, "Solver", solver),
    el("div", { class: "bd-two" }, field("MaxStep", "Largest step", "auto = 1/50 of the run"), field("FixedStep", "Fixed step", "for fixed-step solvers")),
    el("div", { class: "bd-two" }, field("RelTol", "Relative tolerance"), field("AbsTol", "Absolute tolerance")),
    el("label", { class: "bd-check" }, zc, "Find the exact time of events (zero crossings)"),
    el("p", { class: "small muted" }, "Times and tolerances may be expressions with workspace variables. Variable-step solvers pick their step from the tolerances; fixed-step solvers use the fixed step.")
  );
  return openDialog({
    title: "Simulation settings",
    body,
    id: "bd-dlg-settings",
    onOk: () => {
      for (const input of body.querySelectorAll("[data-cfg]")) {
        const v = input.value.trim();
        if (input.tagName === "INPUT") {
          const msg = checkExpression(v);
          if (msg) {
            input.focus();
            return { error: `${input.dataset.cfg}: ${msg}` };
          }
        }
        cfg[input.dataset.cfg] = v;
      }
      cfg.ZeroCrossings = zc.checked ? "on" : "off";
      return { value: cfg };
    },
  });
}

/**
 * A list to pick from by typing. items: [{ label, hint, value, group, keywords }]. Resolves to the value or null.
 */
export function pickDialog({ title, placeholder, items, emptyText = "Nothing matches.", id = "" }) {
  return new Promise((resolve) => {
    const input = el("input", { type: "text", placeholder, spellcheck: "false", autocomplete: "off", role: "combobox", "aria-expanded": "true", "aria-controls": "bd-pick-list", "aria-label": title });
    const list = el("div", { class: "bd-pick-list", id: "bd-pick-list", role: "listbox", "aria-label": title });
    const dlg = el("dialog", { class: "dialog bd-dialog bd-pick", id: id || null }, el("form", { method: "dialog" }, el("h2", {}, title), input, list, el("p", { class: "small muted" }, "↑ ↓ to move, Enter to choose, Esc to cancel.")));
    let shown = [], active = 0, result = null;
    const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
    const draw = () => {
      const q = norm(input.value);
      const scored = [];
      for (const it of items) {
        const hay = norm(it.label), extra = norm((it.keywords || "") + " " + (it.group || "") + " " + (it.hint || ""));
        let s = -1;
        if (!q) s = 0;
        else if (hay.startsWith(q)) s = 3;
        else if (hay.includes(q)) s = 2;
        else if (extra.includes(q)) s = 1;
        else if (isSubsequence(q, hay)) s = 0.5;
        if (s >= 0) scored.push([s, it]);
      }
      scored.sort((a, b) => b[0] - a[0]);
      shown = scored.slice(0, 60).map((x) => x[1]);
      active = Math.min(active, Math.max(shown.length - 1, 0));
      list.replaceChildren(
        ...(shown.length
          ? shown.map((it, i) =>
              el(
                "div",
                { class: "bd-pick-item" + (i === active ? " active" : ""), role: "option", id: "bd-pick-" + i, "aria-selected": i === active ? "true" : "false", onclick: () => choose(i) },
                el("span", { class: "bd-pick-label" }, it.label),
                it.group ? el("span", { class: "pill small" }, it.group) : null,
                it.hint ? el("span", { class: "bd-pick-hint muted small" }, it.hint) : null
              )
            )
          : [el("p", { class: "muted small bd-pick-empty" }, emptyText)])
      );
      input.setAttribute("aria-activedescendant", shown.length ? "bd-pick-" + active : "");
      const act = list.querySelector(".active");
      act && act.scrollIntoView({ block: "nearest" });
    };
    const choose = (i) => {
      if (!shown[i]) return;
      result = shown[i].value;
      dlg.close("ok");
    };
    input.addEventListener("input", () => {
      active = 0;
      draw();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : -1) + shown.length) % Math.max(shown.length, 1);
        draw();
      } else if (e.key === "Enter") {
        e.preventDefault();
        choose(active);
      }
    });
    dlg.addEventListener("close", () => {
      dlg.remove();
      resolve(dlg.returnValue === "ok" ? result : null);
    });
    document.body.appendChild(dlg);
    dlg.showModal();
    draw();
    input.focus();
  });
}

function isSubsequence(q, s) {
  let i = 0;
  for (const c of s) if (c === q[i]) i++;
  return i === q.length;
}

export const KEYS = [
  ["Tab, Shift+Tab", "Select the next or previous block"],
  ["Alt+arrows", "Select the nearest block in that direction (with Shift: add it to the selection)"],
  ["Arrows", "Move the selected blocks (Shift: by 1 px). With nothing selected: scroll"],
  ["Enter", "Parameters of the block; open a subsystem or a scope"],
  ["Esc, Alt+↑", "Clear the selection; leave the subsystem"],
  ["A or /", "Add a block by typing its name"],
  ["C", "Connect: with two blocks selected, output to input; with one, choose the port from a list"],
  ["L", "Name the selected line"],
  ["R, Shift+R, F", "Rotate right or left, flip"],
  ["F2", "Rename the block"],
  ["Delete", "Delete the selection"],
  ["Ctrl+C, X, V, D", "Copy, cut, paste, duplicate"],
  ["Ctrl+Z, Ctrl+Y", "Undo, redo"],
  ["Ctrl+G", "Make a subsystem from the selection"],
  ["Ctrl+Shift+L", "Arrange the blocks automatically"],
  ["Ctrl+A", "Select all"],
  ["+, −, 0, 1", "Zoom in, zoom out, show everything, 100 %"],
  ["F5 or Ctrl+Enter", "Run the model"],
];

export function keysDialog() {
  const body = el("table", { class: "data-table bd-keys" }, el("tbody", {}, KEYS.map(([k, what]) => el("tr", {}, el("th", { scope: "row" }, k), el("td", {}, what)))));
  return openDialog({
    title: "Keys in the diagram editor",
    body: [body, el("p", { class: "small muted" }, "Mouse and touch: drag from a port to a port to draw a line, or click one port and then the other. Ctrl+drag from a line makes a branch. Drag the background to select (mouse) or to move the view (touch, or with Space held). Double-click the background to add a block. On a Mac use ⌘ for Ctrl.")],
    okLabel: "Close", cancelLabel: "", wide: true, id: "bd-dlg-keys",
  });
}

export function textDialog({ title, label, value = "", hint = "", okLabel = "OK", validate }) {
  const input = el("input", { type: "text", value, spellcheck: "false", autocomplete: "off", "data-first": true });
  return openDialog({
    title,
    body: [el("label", { class: "field" }, label, input), hint ? el("p", { class: "small muted" }, hint) : null],
    okLabel,
    onOk: () => {
      const err = validate ? validate(input.value) : null;
      return err ? { error: err } : { value: input.value };
    },
  });
}

export function noticeDialog({ title, text, lines = [] }) {
  return openDialog({
    title,
    body: [el("p", {}, text), lines.length ? el("ul", { class: "bd-notes small" }, lines.map((l) => el("li", {}, l))) : null],
    okLabel: "OK", cancelLabel: "", wide: true, id: "bd-dlg-notice",
  });
}
