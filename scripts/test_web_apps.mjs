#!/usr/bin/env node
// Browser checks for app windows (uifigure / uicontrol) and the app builder.
//
// scripts/test_web.mjs runs these as part of the full suite. Run this file on its own while
// working on the App panel or the builder (it starts its own server):
//
//   node scripts/test_web_apps.mjs [--headed] [--no-screenshot]

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const IMAGES = path.join(ROOT, "docs", "images");
const READY = "body[data-engine='ready']";

const DEMO_APP = `% A small app used by the browser tests.
fig = uifigure('Name', 'Signal lab', 'Position', [100 100 560 470]);
g = uigridlayout(fig, [4 3]);
g.RowHeight = {'fit', '1x', 'fit', 90};
g.ColumnWidth = {'fit', '1x', 'fit'};
lbl = uilabel(g, 'Text', 'Pushed 0 times', 'Tag', 'count');
lbl.Layout.Column = [1 2];
btn = uibutton(g, 'Text', 'Push me', 'Tag', 'push');
ax = uiaxes(g, 'Tag', 'ax');
ax.Layout.Column = [1 3];
uilabel(g, 'Text', 'Frequency');
sld = uislider(g, 'Limits', [1 10], 'Value', 2, 'Tag', 'freq');
readout = uilabel(g, 'Text', '2.0 Hz', 'Tag', 'readout');
tbl = uitable(g, 'Data', {'gain', 1; 'offset', 0}, 'ColumnName', {'Setting', 'Value'}, ...
    'ColumnEditable', [false true], 'Tag', 'settings');
tbl.Layout.Column = [1 3];
fig.UserData = 0;
btn.ButtonPushedFcn = @(src, event) pushed(fig, lbl);
sld.ValueChangingFcn = @(src, event) set(readout, 'Text', sprintf('%.1f Hz', event.Value));
sld.ValueChangedFcn = @(src, event) draw(ax, readout, src.Value, tbl.Data);
tbl.CellEditCallback = @(src, event) draw(ax, readout, sld.Value, src.Data);
draw(ax, readout, sld.Value, tbl.Data);

function pushed(fig, lbl)
    fig.UserData = fig.UserData + 1;
    lbl.Text = sprintf('Pushed %d times', fig.UserData);
    fprintf('push number %d\\n', fig.UserData);
end

function draw(ax, readout, f, data)
    t = linspace(0, 1, 400);
    plot(ax, t, data{1, 2} * sin(2 * pi * f * t) + data{2, 2}, 'LineWidth', 1.5);
    title(ax, sprintf('gain %g, offset %g', data{1, 2}, data{2, 2}));
    ylim(ax, [-3 3]);
    readout.Text = sprintf('%.1f Hz', f);
end
`;

const CLASSIC_APP = `function classic_app
% A classic GUI: figure + uicontrol, written as a function file.
f = figure('Name', 'Classic adder', 'Position', [100 100 300 140], 'MenuBar', 'none', 'NumberTitle', 'off');
uicontrol(f, 'Style', 'text', 'String', 'a + b =', 'Position', [20 90 60 20]);
a = uicontrol(f, 'Style', 'edit', 'String', '2', 'Tag', 'a', 'Position', [90 90 60 22]);
b = uicontrol(f, 'Style', 'edit', 'String', '3', 'Tag', 'b', 'Position', [160 90 60 22]);
out = uicontrol(f, 'Style', 'text', 'String', '?', 'Tag', 'sum', 'Position', [230 90 50 20]);
uicontrol(f, 'Style', 'pushbutton', 'String', 'Add', 'Position', [90 40 130 30], 'Callback', @add);
    function add(~, ~)
        total = str2double(get(a, 'String')) + str2double(get(b, 'String'));
        set(out, 'String', num2str(total));
        if isnan(total), error('Type two numbers.'); end
    end
end
`;

/** The checks. `t` = { ok, check, shot(page, name, selector) }. */
export async function testApps(browser, url, t, consoleErrors = []) {
  const { ok, check } = t;
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 2, permissions: ["clipboard-read", "clipboard-write"] });
  const page = await context.newPage();
  page.on("console", (m) => m.type() === "error" && consoleErrors.push("[apps] " + m.text()));
  page.on("pageerror", (e) => consoleErrors.push("[apps] pageerror: " + e.message));
  const waitIdle = () => page.waitForSelector("#btn-run:not([hidden])", { timeout: 120000 });
  const newFile = async (name, code) => {
    await page.evaluate(([n, c]) => window.mylab.createFile(n, c), [name, code]);
    await page.waitForSelector(`.tab.active[data-name='${name}']`);
  };
  const run = async () => {
    await page.click("#btn-run");
    await page.waitForTimeout(50);
    await waitIdle();
  };
  const text = (sel) => page.locator(sel).innerText();
  const until = (fn, arg, timeout = 30000) => page.waitForFunction(fn, arg, { timeout });

  await page.goto(url);
  await page.waitForSelector(".CodeMirror");
  await page.waitForSelector("#dlg-welcome[open]");
  await page.click("#welcome-close");
  await page.waitForSelector(READY, { timeout: 240000 });
  await page.waitForSelector("#loader", { state: "hidden" });

  /* ---------------- running an app ---------------- */
  await newFile("signal_lab.m", DEMO_APP);
  check(await page.locator("#btn-app").isHidden(), "the App button should be hidden before an app runs");
  await run();
  await page.waitForSelector("#app-view:not([hidden]) .mui-window");
  check((await text("#app-view-title")) === "Signal lab", "the panel should show the window name: " + (await text("#app-view-title")));
  check((await page.locator("#plots .plot-card").count()) === 0, "an axes inside an app must not also appear in the Plots panel");
  const label = "#app-view .mui-label >> nth=0";
  check((await text(label)) === "Pushed 0 times", "label text: " + (await text(label)));
  const img = page.locator("#app-view .mui-axes-image");
  await until(() => document.querySelector("#app-view .mui-axes-image").naturalWidth > 0);
  // the engine redraws the axes once at the size it is shown at
  await page.waitForTimeout(600);
  const before = await img.getAttribute("src");
  check(before.startsWith("data:image/png;base64,"), "the axes should be a PNG image");
  check(/gain 1, offset 0/.test(await img.getAttribute("alt")), "the axes needs a text alternative: " + (await img.getAttribute("alt")));
  ok("Run opens the App panel with the window, its controls and the axes as an image");

  await page.click("#app-view .mui-button:has-text('Push me')");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Pushed 1 times");
  await page.click("#app-view .mui-button:has-text('Push me')");
  await page.click("#app-view .mui-button:has-text('Push me')");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Pushed 3 times");
  await until(() => /push number 3/.test(document.querySelector("#console-out").innerText));
  check(/push number 3/.test(await text("#app-log")), "what callbacks print should also show under the app");
  ok("a button click runs its callback; state lives on between events; output reaches the command window");

  // slider: drag with the mouse (ValueChangingFcn while dragging, ValueChangedFcn on release)
  const range = page.locator("#app-view input[type=range]");
  const box = await range.boundingBox();
  await page.mouse.move(box.x + box.width * 0.12, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2, { steps: 4 });
  await until(() => parseFloat([...document.querySelectorAll("#app-view .mui-label")].pop().textContent) > 3, null, 20000);
  await page.mouse.move(box.x + box.width * 0.85, box.y + box.height / 2, { steps: 4 });
  await page.mouse.up();
  await until((src) => document.querySelector("#app-view .mui-axes-image").getAttribute("src") !== src, before);
  const hz = parseFloat(await page.locator("#app-view .mui-label").last().innerText());
  check(hz > 7 && hz <= 10, "after the drag the readout should be near the right end: " + hz);
  ok("dragging a slider updates a label while it moves and redraws the plot on release");

  // keyboard: the slider is a native range input
  await range.focus();
  const v0 = Number(await range.inputValue());
  await page.keyboard.press("ArrowLeft");
  await until((v) => Number(document.querySelector("#app-view input[type=range]").value) < v, v0);
  await until((v) => parseFloat([...document.querySelectorAll("#app-view .mui-label")].pop().textContent) < v - 0.01, hz);
  await page.waitForTimeout(300);
  ok("controls are native form elements: the slider moves with the arrow keys");

  // table edit
  const afterSlider = await img.getAttribute("src");
  const cell = page.locator("#app-view .mui-table input.mui-cell-input").first();
  check((await cell.inputValue()) === "1", "editable cell should hold the gain");
  check((await page.locator("#app-view .mui-table td").first().locator("input").count()) === 0, "a column that is not editable must be plain text");
  await cell.fill("2.5");
  await cell.press("Enter");
  await until((src) => document.querySelector("#app-view .mui-axes-image").getAttribute("src") !== src, afterSlider);
  check(/gain 2.5, offset 0/.test(await img.getAttribute("alt")), "the plot should use the edited value: " + (await img.getAttribute("alt")));
  check((await page.locator("#app-view .mui-table input.mui-cell-input").first().inputValue()) === "2.5", "the table should keep the new value");
  ok("editing a table cell runs CellEditCallback with the new data");
  await t.shot(page, "app-run.png", "#app-view");

  // hide and show; the workspace is shared with the command window
  await page.click("#app-hide");
  await page.waitForSelector("#app-view", { state: "hidden" });
  check((await page.locator("#btn-app").getAttribute("aria-pressed")) === "false", "App button state");
  const input = page.locator("#console-input");
  await input.fill("lbl.Text = 'set from the command window';");
  await input.press("Enter");
  await waitIdle();
  await page.click("#btn-app");
  await page.waitForSelector("#app-view:not([hidden])");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "set from the command window");
  await page.click("#app-view .mui-button:has-text('Push me')");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Pushed 4 times");
  ok("the app keeps running while hidden, and the command window can change its components");

  // dark theme
  await page.evaluate(() => window.mylab.app.setTheme("dark"));
  await page.waitForTimeout(200);
  const colors = await page.evaluate(() => {
    const cs = (sel) => getComputedStyle(document.querySelector(sel));
    return { body: cs("#app-view .mui-window-body").backgroundColor, text: cs("#app-view .mui-label").color, input: cs("#app-view .mui-cell-input").color };
  });
  const lum = (c) => c.match(/\d+/g).slice(0, 3).reduce((a, b) => a + Number(b), 0) / 3;
  check(lum(colors.body) < 80 && lum(colors.text) > 170, "dark theme colors look wrong: " + JSON.stringify(colors));
  await t.shot(page, "app-dark.png", "#app-view");
  await page.evaluate(() => window.mylab.app.setTheme("system"));
  ok("the app follows the dark theme");

  // running the file again starts a fresh app (no second window piles up)
  await page.click("#app-restart");
  await waitIdle();
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Pushed 0 times");
  check((await page.locator("#app-view .mui-window").count()) === 1, "restarting must not leave the old window behind");
  ok("Restart runs the file again and replaces the old window");

  // closing the window ends the app
  await page.click("#app-view .mui-window-close");
  await page.waitForSelector("#app-view", { state: "hidden" });
  check(await page.locator("#btn-app").isHidden(), "the App button should go away when the window is closed");
  ok("the window's close button closes the app");

  /* ---------------- classic GUI in a function file, errors ---------------- */
  await newFile("classic_app.m", CLASSIC_APP);
  await run();
  await page.waitForSelector("#app-view:not([hidden]) .mui-window.mui-classic");
  await page.click("#app-view .mui-button:has-text('Add')");
  await until(() => [...document.querySelectorAll("#app-view .mui-label")].some((l) => l.textContent === "5"));
  const fields = page.locator("#app-view input.mui-input");
  await fields.nth(0).fill("40");
  await fields.nth(0).press("Tab");
  await page.click("#app-view .mui-button:has-text('Add')");
  await until(() => [...document.querySelectorAll("#app-view .mui-label")].some((l) => l.textContent === "43"));
  ok("Run calls a function file without inputs; figure + uicontrol with nested callbacks works");
  await fields.nth(1).fill("abc");
  await page.click("#app-view .mui-button:has-text('Add')");
  await until(() => /Type two numbers/.test(document.querySelector("#app-log").textContent));
  check(/Type two numbers/.test(await text("#console-out")), "callback errors should be shown in the command window");
  await fields.nth(1).fill("1");
  await page.click("#app-view .mui-button:has-text('Add')");
  await until(() => [...document.querySelectorAll("#app-view .mui-label")].some((l) => l.textContent === "41"));
  ok("an error in a callback is reported and the app keeps working");
  await page.evaluate(() => window.mylab.app.closeApp());
  await page.waitForSelector("#app-view", { state: "hidden" });

  /* ---------------- the app builder ---------------- */
  const cmd = (id) => page.evaluate((c) => void window.mylab.app.commands.run(c), id); // not awaited: some open a dialog
  const design = () => page.evaluate(() => JSON.parse(JSON.stringify(window.mylab.app.designer.design)));
  const itemBy = async (type) => (await design()).items.find((it) => it.type === type);
  await cmd("app.new");
  await page.waitForSelector("#dlg-prompt[open]");
  await page.fill("#prompt-input", "TipApp");
  await page.click("#prompt-ok");
  await page.waitForSelector("#designer:not([hidden]) .mui-window");
  check((await text("#designer-title")) === "App builder: TipApp.m", "builder title: " + (await text("#designer-title")));

  // add with the mouse (click), with the keyboard, and by dragging from the palette
  await page.click(".dsg-tool[data-type='label']");
  await page.click(".dsg-tool[data-type='numericfield']");
  await page.locator(".dsg-tool[data-type='button']").focus();
  await page.keyboard.press("Enter");
  let d = await design();
  check(d.items.map((it) => `${it.type}@${it.row},${it.col}`).join(" ") === "label@1,1 numericfield@1,2 button@2,1", "new components should fill the grid in order: " + JSON.stringify(d.items.map((it) => [it.type, it.row, it.col])));
  const tool = await page.locator(".dsg-tool[data-type='axes']").boundingBox();
  const target = await page.locator(".dsg-cell[data-row='4'][data-col='1']").boundingBox();
  await page.mouse.move(tool.x + 20, tool.y + 10);
  await page.mouse.down();
  await page.mouse.move(target.x + 30, target.y + 12, { steps: 6 });
  await page.waitForSelector(".dsg-cell.target[data-row='4'][data-col='1']");
  await page.mouse.up();
  await until(() => window.mylab.app.designer.design.items.length === 4);
  const ax = await itemBy("axes");
  check(ax.row === 4 && ax.col === 1, "the dropped axes should land in the cell under the pointer: " + JSON.stringify(ax));
  ok("the builder adds components by click, by keyboard and by dragging from the palette onto a grid cell");

  // keyboard: move and resize the selected component, announce it
  const axCover = page.locator(`.dsg-item[data-id='${ax.id}']`);
  await axCover.focus();
  await page.keyboard.press("Shift+ArrowRight");
  check((await itemBy("axes")).colSpan === 2, "Shift+ArrowRight should widen the component");
  check(/2 columns/.test(await page.locator("#designer [role=status]").innerText()), "changes should be announced to screen readers");
  check(/row 4, column 1, 1 rows by 2 columns/.test(await page.locator(`.dsg-item[data-id='${ax.id}']`).getAttribute("aria-label")), "the component needs a spoken position");
  const btnItem = await itemBy("button");
  await page.locator(`.dsg-item[data-id='${btnItem.id}']`).focus();
  await page.keyboard.press("ArrowRight");
  check((await itemBy("button")).col === 2, "ArrowRight should move the component one column");
  check((await page.evaluate(() => document.activeElement.dataset.id)) === btnItem.id, "the moved component should keep the focus");
  // mouse: drag the button back, and resize the label with its handle
  const from = await page.locator(`.dsg-item[data-id='${btnItem.id}']`).boundingBox();
  const to = await page.locator(".dsg-cell[data-row='3'][data-col='1']").boundingBox();
  await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
  await page.mouse.down();
  await page.mouse.move(to.x + 20, to.y + 10, { steps: 6 });
  await page.mouse.up();
  await until(() => { const b = window.mylab.app.designer.design.items.find((it) => it.type === "button"); return b.row === 3 && b.col === 1; });
  ok("components move and resize with the arrow keys (announced) and by dragging; they snap to grid cells");

  // properties, undo and redo
  const lab = await itemBy("label");
  await page.locator(`.dsg-item[data-id='${lab.id}']`).click();
  await page.fill("#dsg-props [data-key='text']", "Bill amount");
  await page.press("#dsg-props [data-key='text']", "Enter");
  await until(() => [...document.querySelectorAll("#dsg-canvas .mui-label")].some((l) => l.textContent === "Bill amount"));
  await page.locator(`.dsg-item[data-id='${(await itemBy("numericfield")).id}']`).click();
  await page.fill("#dsg-props [data-key='name']", "bill");
  await page.press("#dsg-props [data-key='name']", "Tab");
  await page.fill("#dsg-props [data-key='max']", "1000");
  await page.press("#dsg-props [data-key='max']", "Tab");
  await page.locator(`.dsg-item[data-id='${btnItem.id}']`).click();
  await page.fill("#dsg-props [data-key='text']", "15% tip");
  await page.press("#dsg-props [data-key='text']", "Tab");
  await page.fill("#dsg-props [data-key='callback']", "showTip");
  await page.press("#dsg-props [data-key='callback']", "Tab");
  await page.fill("#dsg-props [data-key='name']", "2bad");
  await page.press("#dsg-props [data-key='name']", "Tab");
  check((await itemBy("button")).name === "button1", "a name that is not valid must be refused");
  await page.click("#dsg-undo");
  check((await itemBy("button")).callback === "button1Pushed", "Undo should take back the last change");
  await page.click("#dsg-redo");
  check((await itemBy("button")).callback === "showTip", "Redo should bring it back");
  await page.locator(`.dsg-item[data-id='${btnItem.id}']`).focus();
  await page.keyboard.press("Control+z");
  await page.keyboard.press("Control+y");
  check((await itemBy("button")).callback === "showTip", "Ctrl+Z / Ctrl+Y should undo and redo");
  ok("the property panel edits text, limits, names and callbacks; bad names are refused; undo and redo work");
  await page.keyboard.press("Escape");
  await t.shot(page, "app-builder.png", "#designer");

  // preview and code
  await page.click("#dsg-preview");
  check((await page.locator("#dsg-canvas .dsg-cell").count()) === 0, "Preview should hide the grid lines");
  await page.click("#dsg-preview");
  await page.click("#dsg-code");
  const shown = await text("#dsg-code-view");
  check(/bill = uieditfield\(grid, 'numeric', 'Limits', \[-Inf 1000\], 'Value', 0\);/.test(shown), "the code view should show the generated code:\n" + shown);
  await page.click("#dsg-code");
  await page.click("#dsg-save");
  await page.waitForSelector("#designer", { state: "hidden" });
  await page.waitForSelector(".tab.active[data-name='TipApp.m']");
  let code = await page.evaluate(() => window.mylab.editor.getValue());
  for (const piece of ["function TipApp", "fig = uifigure('Name', 'My app'", "grid = uigridlayout(fig, [4 2]);", "label1 = uilabel(grid, 'Text', 'Bill amount');", "button1.ButtonPushedFcn = @showTip;",
    "    function showTip(src, event)", "ax1.Layout.Column = [1 2];", "mylab-app-design:"]) {
    check(code.includes(piece), `generated code lacks "${piece}":\n${code}`);
  }
  ok("Generate code writes a readable .m file with the layout, callback stubs and the stored design");

  // write a callback, run the app
  code = code.replace(/( {4}function showTip\(src, event\)\n)[\s\S]*?( {4}end)/, "$1        label1.Text = sprintf('Tip: %.2f', 0.15 * bill.Value);\n        plot(ax1, [0 bill.Value], [0 0.15 * bill.Value]);\n$2");
  await page.evaluate((c) => window.mylab.editor.setValue(c), code);
  await run();
  await page.waitForSelector("#app-view:not([hidden]) .mui-window");
  await page.fill("#app-view input.mui-numeric", "80");
  await page.press("#app-view input.mui-numeric", "Enter");
  await page.click("#app-view .mui-button:has-text('15% tip')");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Tip: 12.00");
  ok("the generated app runs: its callback reads the number field and updates the label and the plot");
  await page.click("#app-hide");

  // round trip: reopen the file in the builder, change the layout, keep the callback
  check(await page.locator("#btn-design").isVisible(), "files made by the builder should offer 'Edit layout'");
  await page.click("#btn-design");
  await page.waitForSelector("#designer:not([hidden]) .mui-window");
  d = await design();
  check(d.items.length === 4 && d.items.find((it) => it.type === "button").callback === "showTip" && d.items.find((it) => it.type === "axes").colSpan === 2, "the design was not restored: " + JSON.stringify(d.items));
  await page.click(".dsg-tool[data-type='checkbox']");
  await page.click("#dsg-run");
  await page.waitForSelector("#designer", { state: "hidden" });
  await waitIdle();
  await page.waitForSelector("#app-view:not([hidden]) .mui-check");
  code = await page.evaluate(() => window.mylab.editor.getValue());
  check(code.includes("label1.Text = sprintf('Tip: %.2f', 0.15 * bill.Value);"), "generating again must keep the callback the user wrote");
  check(/check1 = uicheckbox\(grid/.test(code) && /function check1Changed\(src, event\)/.test(code), "the new component and its stub should be in the code");
  await page.click("#app-view .mui-button:has-text('15% tip')");
  await until(() => document.querySelector("#app-view .mui-label").textContent === "Tip: 0.00");
  ok("reopening a generated file restores the design; generating again keeps hand-written callbacks");

  // the command palette knows the app actions
  await cmd("palette.commands");
  await page.waitForSelector("#dlg-palette[open]");
  await page.fill("#palette-input", ">app builder");
  try {
    await until(() => /New app \(open the app builder\)/.test(document.querySelector("#palette-list").textContent), null, 5000);
  } catch {
    throw new Error("palette shows: " + (await page.locator("#palette-list").textContent()).slice(0, 400));
  }
  await page.keyboard.press("Escape");
  ok("the app actions are in the command palette");

  /* ---------------- "Open as app" links, on a phone ---------------- */
  await page.evaluate(() => window.mylab.app.openFile("signal_lab.m"));
  await cmd("app.share");
  await page.waitForSelector("#dlg-share[open]");
  await until(() => /&app=1$/.test(document.querySelector("#share-link").value));
  const link = await page.inputValue("#share-link");
  await page.click("#dlg-share button[value='close']");
  await page.evaluate(() => window.mylab.app.closeApp());
  const phone = await browser.newContext({ viewport: { width: 360, height: 740 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const pp = await phone.newPage();
  pp.on("console", (m) => m.type() === "error" && consoleErrors.push("[app link] " + m.text()));
  pp.on("pageerror", (e) => consoleErrors.push("[app link] pageerror: " + e.message));
  await pp.goto(link);
  await pp.waitForSelector("body[data-appmode] #app-view:not([hidden])");
  check(await pp.locator(".CodeMirror").isHidden(), "an app link should hide the editor");
  check(await pp.locator("#dlg-welcome").isHidden(), "the welcome dialog should not cover an app link");
  await pp.waitForSelector("#app-view .mui-window", { timeout: 240000 });
  await pp.waitForFunction(() => document.querySelector("#app-view .mui-axes-image") && document.querySelector("#app-view .mui-axes-image").naturalWidth > 0);
  await pp.tap("#app-view .mui-button:has-text('Push me')");
  await pp.waitForFunction(() => document.querySelector("#app-view .mui-label").textContent === "Pushed 1 times");
  await pp.waitForTimeout(700);
  const over = await pp.evaluate(() => ({ page: document.documentElement.scrollWidth - window.innerWidth, win: document.querySelector("#app-view .mui-window").getBoundingClientRect().right - window.innerWidth }));
  check(over.page <= 1 && over.win <= 1, "the app should fit a 360 px screen: " + JSON.stringify(over));
  await t.shot(pp, "app-phone.png");
  ok("an 'Open as app' link opens straight into the running app, which fits a 360 px phone");
  await pp.tap("#app-code");
  await pp.waitForSelector("body:not([data-appmode]) .CodeMirror");
  check(/uifigure/.test(await pp.evaluate(() => window.mylab.editor.getValue())), "'Edit the code' should show the code of the app");
  ok("'Edit the code' leaves the app-only view and shows the editor");
  // the builder on a phone
  await pp.tap("#app-hide");
  await pp.evaluate(() => window.mylab.app.openDesigner(window.mylab.app.designer.readDesign("mylab-app-design:\n" + JSON.stringify({ format: "mylab-app-1", name: "P", title: "P", width: 480, height: 300, rows: ["fit", "1x"], cols: ["1x", "1x"], items: [] })), "P.m"));
  await pp.tap(".dsg-tool[data-type='slider']");
  await pp.waitForSelector("#dsg-canvas input[type=range]");
  const wide = await pp.evaluate(() => document.querySelector("#designer .dsg-body").scrollWidth - document.querySelector("#designer .dsg-body").clientWidth);
  check(wide <= 1, `the app builder scrolls sideways on a phone by ${wide}px`);
  ok("the app builder fits a phone screen");
  await phone.close();
  await context.close();
}

/* ------------------------------------------------------------------ standalone */

function startServer(py) {
  return new Promise((resolve, reject) => {
    const proc = spawn(py, ["-u", "scripts/build_web.py", "--serve", "--port", "0"], { cwd: ROOT });
    let out = "";
    const timer = setTimeout(() => reject(new Error("server did not start:\n" + out)), 120000);
    proc.stdout.on("data", (d) => {
      out += d.toString();
      const m = out.match(/http:\/\/127\.0\.0\.1:(\d+)\//);
      if (m) {
        clearTimeout(timer);
        resolve({ proc, url: `http://127.0.0.1:${m[1]}/` });
      }
    });
    proc.stderr.on("data", (d) => (out += d.toString()));
    proc.on("exit", (code) => reject(new Error(`server exited with ${code}:\n${out}`)));
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const require = createRequire(import.meta.url);
  const { chromium } = require("playwright");
  const args = new Set(process.argv.slice(2));
  const shots = !args.has("--no-screenshot");
  let passed = 0;
  const t = {
    ok: (name) => console.log(`  ok   ${name}`, ++passed && ""),
    check: (cond, message) => {
      if (!cond) throw new Error(message);
    },
    async shot(page, name, selector = null) {
      if (!shots) return;
      fs.mkdirSync(IMAGES, { recursive: true });
      await page.evaluate(() => document.querySelectorAll(".toast").forEach((x) => x.remove()));
      await page.waitForTimeout(350);
      const clip = selector ? await page.locator(selector).boundingBox() : undefined;
      await page.screenshot({ path: path.join(IMAGES, name), clip });
      t.ok(`screenshot saved to docs/images/${name}`);
    },
  };
  const { proc, url } = await startServer(process.env.PYTHON || "python3");
  const browser = await chromium.launch({ headless: !args.has("--headed") });
  const errors = [];
  try {
    await testApps(browser, url, t, errors);
    if (errors.length) throw new Error("console errors:\n" + errors.join("\n"));
    console.log(`\nAll ${passed} app checks passed.`);
  } catch (e) {
    console.error("\nFAILED:", e.message);
    process.exitCode = 1;
  } finally {
    await browser.close();
    proc.kill();
  }
}
