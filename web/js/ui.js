// Small DOM helpers shared by the app's modules.

export const $ = (id) => document.getElementById(id);
export const isMac = /Mac|iPhone|iPad/.test(navigator.platform || "");

export function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast" + (kind ? " " + kind : "");
  t.textContent = msg;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "error" ? 6000 : 2800);
  return t;
}

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) e.append(c);
  return e;
}

export function icon(name, cls = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "ic" + (cls ? " " + cls : ""));
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.appendChild(use);
  return svg;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Copy text that is still being computed (an engine reply, a compressed link). The clipboard write starts
 * at once, inside the click, and receives the text when it is ready: Firefox and Safari allow a clipboard
 * write only shortly after the click, and the computation can take longer than that.
 */
export async function copyTextLater(textPromise) {
  if (typeof ClipboardItem === "function" && navigator.clipboard && navigator.clipboard.write) {
    try {
      const blob = Promise.resolve(textPromise).then((t) => new Blob([t], { type: "text/plain" }));
      await navigator.clipboard.write([new ClipboardItem({ "text/plain": blob })]);
      return true;
    } catch {
      // an older browser, or a refused write: copy the finished text the ordinary way
    }
  }
  return copyText(await textPromise);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Older browsers, or pages without clipboard permission.
    const ta = el("textarea", { style: "position:fixed;left:-999px;top:0", readonly: true });
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    ta.remove();
    return ok;
  }
}

export function dateStamp(d = new Date()) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* ---------- dialogs ---------- */

/**
 * Ask for a name. With `choices`, radio buttons above the field offer kinds of thing to make:
 * { legend, value, items: [{ value, label, detail }], pick(value, typed) -> { value?, text? },
 *   detect(text) -> value | null }. pick() is called when a choice is made (it may rewrite the
 * field and the explanation); detect() lets typing a name (for example its extension) select a choice.
 */
export function promptDialog({ title, text = "", value = "", okLabel = "OK", validate, selectAll = false, placeholder = "", choices = null }) {
  const dlg = $("dlg-prompt");
  $("prompt-title").textContent = title;
  $("prompt-text").textContent = text;
  $("prompt-text").hidden = !text;
  $("prompt-ok").textContent = okLabel;
  const input = $("prompt-input");
  const err = $("prompt-error");
  const box = $("prompt-choices");
  input.value = value;
  input.placeholder = placeholder;
  err.hidden = true;
  box.replaceChildren();
  box.hidden = !choices;
  let chosen = choices ? choices.value : null;
  /** Select the part of the name to type over: the name without its folder and extension. */
  const selectName = (v, ext = null) => {
    const slash = v.lastIndexOf("/") + 1;
    const end = ext && v.toLowerCase().endsWith(ext.toLowerCase()) ? v.length - ext.length : v.lastIndexOf(".");
    input.setSelectionRange(slash, end > slash ? end : v.length);
  };
  return new Promise((resolve) => {
    const check = () => (validate ? validate(input.value) : { name: input.value.trim() });
    const onInput = () => {
      const r = check();
      err.hidden = !(r.error || r.warn);
      err.textContent = r.error || r.warn || "";
      err.classList.toggle("muted", !r.error);
      $("prompt-ok").disabled = !!r.error;
    };
    const choose = (v, typed) => {
      chosen = v;
      const r = choices.pick(v, typed) || {};
      if (r.text !== undefined) {
        $("prompt-text").textContent = r.text;
        $("prompt-text").hidden = !r.text;
      }
      if (r.value !== undefined) input.value = r.value;
      onInput();
      if (!typed && !$("prompt-ok").disabled) err.hidden = true; // a choice is not typing: no tips yet
      return r;
    };
    if (choices) {
      box.append(el("legend", { class: "sr-only" }, choices.legend || "Kind"));
      for (const it of choices.items) {
        const radio = el("input", { type: "radio", name: "prompt-choice", id: "prompt-choice-" + it.value, value: it.value });
        radio.checked = it.value === chosen;
        radio.addEventListener("change", () => {
          if (!radio.checked) return;
          const r = choose(it.value, false);
          if (r.value !== undefined) selectName(input.value, r.ext);
        });
        box.append(el("label", { class: "prompt-choice", for: radio.id }, radio, el("span", {}, it.label, it.detail ? el("small", {}, it.detail) : null)));
      }
    }
    const onType = () => {
      const v = choices && choices.detect ? choices.detect(input.value) : null;
      if (v && v !== chosen) {
        $("prompt-choice-" + v).checked = true;
        choose(v, true);
      } else onInput();
    };
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      input.removeEventListener("input", onType);
      if (dlg.returnValue !== "ok") return resolve(null);
      const r = check();
      resolve(r.error ? null : r.name);
    };
    input.addEventListener("input", onType);
    dlg.addEventListener("close", onClose);
    dlg.returnValue = "";
    $("prompt-ok").disabled = false;
    dlg.showModal();
    onInput();
    err.hidden = true; // don't nag before typing
    input.focus();
    if (selectAll) input.select();
    else selectName(value);
  });
}

export function confirmDialog({ title, text, okLabel = "OK", danger = true }) {
  const dlg = $("dlg-confirm");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = okLabel;
  $("confirm-ok").className = "btn " + (danger ? "danger" : "primary");
  return new Promise((resolve) => {
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      resolve(dlg.returnValue === "ok");
    };
    dlg.addEventListener("close", onClose);
    dlg.returnValue = "";
    dlg.showModal();
  });
}

/** A dialog with a title, a short text and lists of lines (e.g. what an import skipped). */
export function summaryDialog({ title, text = "", groups = [] }) {
  $("summary-title").textContent = title;
  $("summary-text").textContent = text;
  $("summary-text").hidden = !text;
  const body = $("summary-body");
  body.innerHTML = "";
  for (const g of groups) {
    if (!g.items || !g.items.length) continue;
    const shown = g.items.slice(0, 200);
    body.append(
      el("h3", {}, `${g.title} (${g.items.length})`),
      el("ul", { class: "summary-list" }, shown.map((it) => el("li", {}, el("code", {}, it.path), it.reason ? ` — ${it.reason}` : ""))),
      g.items.length > shown.length ? el("p", { class: "muted small" }, `…and ${g.items.length - shown.length} more`) : null
    );
  }
  $("dlg-summary").showModal();
}

/* ---------- popup menu ---------- */

let openMenuEl = null;

export function closeMenu() {
  if (openMenuEl) {
    openMenuEl.remove();
    openMenuEl = null;
    document.removeEventListener("pointerdown", onDocDown, true);
    document.removeEventListener("keydown", onDocKey, true);
  }
}

function onDocDown(e) {
  if (openMenuEl && !openMenuEl.contains(e.target) && !openMenuEl._anchor.contains(e.target)) closeMenu();
}

function onDocKey(e) {
  if (e.key === "Escape") {
    e.preventDefault();
    closeMenu();
  } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const items = [...openMenuEl.querySelectorAll("button:not(:disabled)")];
    const i = items.indexOf(document.activeElement);
    const next = items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length];
    next && next.focus();
  }
}

/**
 * Show a menu under `anchor`. items: [{ label, icon, onclick, checked, danger, hint, id } | "sep" | { heading }]
 */
export function openMenu(anchor, items, { id = "" } = {}) {
  const wasOpenFor = openMenuEl && openMenuEl._anchor === anchor;
  closeMenu();
  if (wasOpenFor) return null;
  const menu = el("div", { class: "menu", role: "menu", id: id || null });
  menu._anchor = anchor;
  for (const it of items) {
    if (!it) continue;
    if (it === "sep") {
      menu.append(el("div", { class: "menu-sep", role: "separator" }));
    } else if (it.heading) {
      menu.append(el("div", { class: "menu-heading" }, it.heading));
    } else {
      menu.append(
        el(
          "button",
          {
            type: "button",
            role: "menuitem",
            class: "menu-item" + (it.danger ? " danger" : "") + (it.checked ? " checked" : ""),
            id: it.id || null,
            title: it.hint || null,
            "data-value": it.value || null,
            onclick: () => {
              closeMenu();
              it.onclick && it.onclick();
            },
          },
          it.checked ? icon("check") : it.icon ? icon(it.icon) : el("span", { class: "ic" }),
          el("span", { class: "menu-label" }, it.label),
          it.badge ? el("span", { class: "pill small" }, it.badge) : null
        )
      );
    }
  }
  document.body.appendChild(menu);
  const r = anchor.getBoundingClientRect();
  const w = menu.offsetWidth;
  const h = menu.offsetHeight;
  let left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8));
  let top = r.bottom + 4;
  if (top + h > window.innerHeight - 8) top = Math.max(8, r.top - h - 4);
  menu.style.left = left + window.scrollX + "px";
  menu.style.top = top + window.scrollY + "px";
  openMenuEl = menu;
  document.addEventListener("pointerdown", onDocDown, true);
  document.addEventListener("keydown", onDocKey, true);
  const first = menu.querySelector("button");
  first && first.focus({ preventScroll: true });
  return menu;
}
