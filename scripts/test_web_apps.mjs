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

  return { context, page, newFile, run, waitIdle, until, text };
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
