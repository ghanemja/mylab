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

export function promptDialog({ title, text = "", value = "", okLabel = "OK", validate, selectAll = false, placeholder = "" }) {
  const dlg = $("dlg-prompt");
  $("prompt-title").textContent = title;
  $("prompt-text").textContent = text;
  $("prompt-text").hidden = !text;
  $("prompt-ok").textContent = okLabel;
  const input = $("prompt-input");
  const err = $("prompt-error");
  input.value = value;
  input.placeholder = placeholder;
  err.hidden = true;
  return new Promise((resolve) => {
    const check = () => (validate ? validate(input.value) : { name: input.value.trim() });
    const onInput = () => {
      const r = check();
      err.hidden = !(r.error || r.warn);
      err.textContent = r.error || r.warn || "";
      err.classList.toggle("muted", !r.error);
      $("prompt-ok").disabled = !!r.error;
    };
    const onClose = () => {
      dlg.removeEventListener("close", onClose);
      input.removeEventListener("input", onInput);
      if (dlg.returnValue !== "ok") return resolve(null);
      const r = check();
      resolve(r.error ? null : r.name);
    };
    input.addEventListener("input", onInput);
    dlg.addEventListener("close", onClose);
    dlg.returnValue = "";
    $("prompt-ok").disabled = false;
    dlg.showModal();
    onInput();
    err.hidden = true; // don't nag before typing
    const slash = value.lastIndexOf("/") + 1;
    const dot = value.lastIndexOf(".");
    input.focus();
    if (selectAll) input.select();
    else input.setSelectionRange(slash, dot > slash ? dot : value.length);
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
