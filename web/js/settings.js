// Appearance and comfort: themes (following the system setting by default), text size,
// the settings dialog and the "More" menu in the top bar.

import { $, openMenu } from "./ui.js";

const DEFAULTS = { autosave: true, lint: true, suggest: true, hover: true };
const THEMES = [
  { id: "system", label: "Follow my system" },
  { id: "light", label: "Light" },
  { id: "dark", label: "Dark" },
  { id: "hc", label: "High contrast" },
];
export const FONT_MIN = 10;
export const FONT_MAX = 24;
export const FONT_DEFAULT = 13.5;

export function initSettings(app) {
  const { prefs } = app;
  const values = { ...DEFAULTS, ...(prefs.get("settings", null) || {}) };

  app.settings = {
    get: (key) => values[key],
    set(key, value) {
      if (values[key] === value) return;
      values[key] = value;
      prefs.set("settings", values);
      app.emit("settingsChanged", key, value);
    },
  };

  /* ---------- theme ---------- */

  const darkQuery = matchMedia("(prefers-color-scheme: dark)");
  const contrastQuery = matchMedia("(prefers-contrast: more)");

  /** The saved choice: "system", "light", "dark" or "hc". */
  function themeChoice() {
    const t = prefs.get("theme", null);
    return t === "light" || t === "dark" || t === "hc" ? t : "system";
  }

  /** What is shown right now: "light", "dark" or "hc". */
  function shownTheme() {
    const c = themeChoice();
    if (c !== "system") return c;
    if (contrastQuery.matches) return "hc";
    return darkQuery.matches ? "dark" : "light";
  }

  function applyTheme() {
    const c = themeChoice();
    const root = document.documentElement;
    if (c !== "system") root.dataset.theme = c;
    else if (contrastQuery.matches) root.dataset.theme = "hc";
    else delete root.dataset.theme;
    const shown = shownTheme();
    root.dataset.shown = shown;
    const btn = $("btn-theme");
    btn.querySelector("use").setAttribute("href", shown === "light" ? "#i-moon" : shown === "dark" ? "#i-sun" : "#i-contrast");
    btn.title = `Theme: ${THEMES.find((t) => t.id === c).label}. Click to change.`;
    if ($("set-theme")) $("set-theme").value = c;
    app.editor && app.editor.refresh();
  }

  function setTheme(choice) {
    if (choice === "system") prefs.remove("theme");
    else prefs.set("theme", choice);
    applyTheme();
  }

  function themeMenu() {
    const c = themeChoice();
    openMenu(
      $("btn-theme"),
      THEMES.map((t) => ({ label: t.label, checked: t.id === c, value: t.id, id: "theme-" + t.id, onclick: () => setTheme(t.id) })),
      { id: "theme-menu" }
    );
  }

  darkQuery.addEventListener?.("change", applyTheme);
  contrastQuery.addEventListener?.("change", applyTheme);

  /* ---------- text size ---------- */

  function fontSize() {
    const n = Number(prefs.get("fontSize", FONT_DEFAULT));
    return Number.isFinite(n) ? Math.max(FONT_MIN, Math.min(FONT_MAX, n)) : FONT_DEFAULT;
  }

  function applyFont() {
    const px = fontSize();
    document.documentElement.style.setProperty("--code-size", px + "px");
    if ($("set-font")) {
      $("set-font").value = String(px);
      $("set-font-value").textContent = `${px} px`;
    }
    app.editor && app.editor.refresh();
  }

  function setFontSize(px) {
    prefs.set("fontSize", Math.max(FONT_MIN, Math.min(FONT_MAX, Math.round(px * 2) / 2)));
    applyFont();
  }

  /* ---------- settings dialog ---------- */

  function openSettings() {
    $("set-theme").value = themeChoice();
    $("set-keys").value = app.commands.preset;
    for (const key of Object.keys(DEFAULTS)) $("set-" + key).checked = !!values[key];
    applyFont();
    if (!$("dlg-settings").open) $("dlg-settings").showModal();
  }

  $("set-theme").addEventListener("change", (e) => setTheme(e.target.value));
  $("set-font").addEventListener("input", (e) => setFontSize(Number(e.target.value)));
  $("set-keys").addEventListener("change", (e) => app.commands.setPreset(e.target.value));
  for (const key of Object.keys(DEFAULTS)) $("set-" + key).addEventListener("change", (e) => app.settings.set(key, e.target.checked));
  $("set-layout").addEventListener("click", () => app.resetLayout());
  $("btn-theme").addEventListener("click", themeMenu);

  /* ---------- "More" menu ---------- */

  function moreMenu() {
    const item = (id, label, icon) => ({ label, icon, badge: app.commands.keyLabel(id) || null, id: "more-" + id.replace(/\./g, "-"), onclick: () => app.commands.run(id) });
    openMenu(
      $("btn-more"),
      [
        item("palette.commands", "All commands…", "menu"),
        item("palette.files", "Go to file…", "file"),
        item("palette.outline", "Outline of this file…", "outline"),
        "sep",
        item("find.open", "Find in this file", "search"),
        item("find.inFiles", "Search in all files", "search"),
        item("edit.format", "Fix indentation", "format"),
        item("file.history", "Version history", "clock"),
        "sep",
        item("view.settings", "Settings", "gear"),
        item("help.keys", "Keyboard shortcuts", "keyboard"),
        item("help.search", "Function help", "info"),
      ],
      { id: "more-menu" }
    );
  }
  $("btn-more").addEventListener("click", moreMenu);

  app.commands.register([
    { id: "view.settings", group: "View", title: "Settings", keys: "Mod-,", run: openSettings },
    { id: "view.theme", group: "View", title: "Change the color theme…", run: () => app.commands.openPalette("", { title: "Color theme", placeholder: "Pick a theme", items: (q) => THEMES.filter((t) => t.label.toLowerCase().includes(q.toLowerCase())).map((t) => ({ label: t.label, detail: t.id === themeChoice() ? "in use" : "", run: () => setTheme(t.id) })) }) },
    { id: "view.themeLight", group: "View", title: "Theme: Light", run: () => setTheme("light"), hidden: true },
    { id: "view.themeDark", group: "View", title: "Theme: Dark", run: () => setTheme("dark"), hidden: true },
    { id: "view.themeHc", group: "View", title: "Theme: High contrast", run: () => setTheme("hc"), hidden: true },
    { id: "view.fontBigger", group: "View", title: "Make the text bigger", keys: "Mod-Alt-=", run: () => setFontSize(fontSize() + 1) },
    { id: "view.fontSmaller", group: "View", title: "Make the text smaller", keys: "Mod-Alt--", run: () => setFontSize(fontSize() - 1) },
    { id: "view.fontReset", group: "View", title: "Reset the text size", run: () => setFontSize(FONT_DEFAULT) },
    { id: "view.keysDefault", group: "View", title: "Use the standard shortcuts", run: () => app.commands.setPreset("default"), when: () => app.commands.preset !== "default" },
    { id: "view.keysMatlab", group: "View", title: "Use MATLAB-style keys (F5, F9, Ctrl+Enter)", run: () => app.commands.setPreset("matlab"), when: () => app.commands.preset !== "matlab" },
  ]);

  applyTheme();
  applyFont();
  Object.assign(app, { setTheme, themeChoice, shownTheme, setFontSize, fontSize, openSettings });
}
