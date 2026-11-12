#!/usr/bin/env node
// End-to-end test for the mylab web app (web/).
//
// Builds the wheel, serves web/ on a local port, opens the page in headless
// Chromium with Playwright and checks the main features. Needs network access
// to cdn.jsdelivr.net and cdnjs.cloudflare.com (Pyodide, CodeMirror, JSZip).
// GitHub and the AI providers are mocked: no request reaches them.
//
// Usage:  node scripts/test_web.mjs [--headed] [--no-screenshot] [--gif]
//
//   --no-screenshot  don't refresh the stills in docs/images/
//   --gif            only record docs/images/demo.gif (needs Pillow in $PYTHON)

import { spawn, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const IMAGES = path.join(ROOT, "docs", "images");
const args = new Set(process.argv.slice(2));
const PY = process.env.PYTHON || "python3";
const SHOTS = !args.has("--no-screenshot");

let passed = 0;
function ok(name) {
  passed++;
  console.log(`  ok   ${name}`);
}

function check(cond, message) {
  if (!cond) throw new Error(message);
}

function startServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn(PY, ["-u", "scripts/build_web.py", "--serve", "--port", "0"], { cwd: ROOT });
    let out = "";
    const timer = setTimeout(() => reject(new Error("server did not start:\n" + out)), 120000);
    const onData = (d) => {
      out += d.toString();
      const m = out.match(/http:\/\/127\.0\.0\.1:(\d+)\//);
      if (m) {
        clearTimeout(timer);
        resolve({ proc, url: `http://127.0.0.1:${m[1]}/` });
      }
    };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", (d) => (out += d.toString()));
    proc.on("exit", (code) => reject(new Error(`server exited with ${code}:\n${out}`)));
  });
}

/** Save a still for the docs. With `selector`, only that part of the page is kept. */
async function shot(page, name, selector = null) {
  if (!SHOTS) return;
  fs.mkdirSync(IMAGES, { recursive: true });
  await page.evaluate(() => document.querySelectorAll(".toast").forEach((t) => t.remove()));
  await page.waitForTimeout(350);
  const clip = selector ? await page.locator(selector).boundingBox() : undefined;
  await page.screenshot({ path: path.join(IMAGES, name), clip });
  ok(`screenshot saved to docs/images/${name}`);
}

/* ------------------------------------------------------------------ page helpers */

const READY = "body[data-engine='ready']";

function helpers(page) {
  const h = {
    setCode: (code) => page.evaluate((c) => window.mylab.editor.setValue(c), code),
    consoleText: () => page.locator("#console-out").innerText(),
    waitIdle: () => page.waitForSelector("#btn-run:not([hidden])", { timeout: 120000 }),
    waitConsole: (re, timeout = 60000) => page.waitForFunction((src) => new RegExp(src).test(document.querySelector("#console-out").innerText), re.source, { timeout }),
    openFile: (name) => page.click(`.file-item[data-name='${name}']`),
    async newFile(name, code) {
      await page.evaluate(([n, c]) => window.mylab.createFile(n, c), [name, code]);
      await page.waitForSelector(`.tab.active[data-name='${name}']`);
    },
    async run() {
      await page.click("#btn-run");
      await page.waitForTimeout(50);
      await h.waitIdle();
    },
    async menu(itemId) {
      await page.click("#project-btn");
      await page.waitForSelector("#project-menu");
      await page.click("#" + itemId);
    },
    projectName: () => page.locator("#project-name").innerText(),
    files: () => page.evaluate(() => window.mylab.app.sortedPaths()),
    varValue: (name) => page.locator(`#var-table tr[data-name='${name}'] td:nth-child(4)`).innerText(),
    stepInfo: () => page.evaluate(() => ({ ...document.querySelector("#step-bar").dataset })),
    async switchTo(name) {
      await page.click("#project-btn");
      await page.locator("#project-menu .menu-item", { hasText: name }).first().click();
      await page.waitForFunction((n) => document.querySelector("#project-name").textContent === n, name);
    },
    /** Wait until a function in the page (it may be async) returns something truthy. */
    async until(fn, arg, timeout = 30000) {
      const t0 = Date.now();
      for (;;) {
        if (await page.evaluate(fn, arg)) return;
        if (Date.now() - t0 > timeout) throw new Error("timed out waiting for: " + String(fn).slice(0, 160));
        await page.waitForTimeout(50);
      }
    },
    /** Start an assignment check and wait until that very check has finished and been drawn. */
    async checkAnswers(selector) {
      const before = await page.locator("#asg-score").getAttribute("data-when");
      await page.click(selector);
      await page.waitForFunction((b) => { const s = document.querySelector("#asg-score"); return s && s.dataset.state === "idle" && s.dataset.when !== b; }, before, { timeout: 240000 });
      await h.waitIdle();
    },
    markedLine: (cls) =>
      page.evaluate((c) => {
        const cm = window.mylab.editor;
        for (let i = 0; i < cm.lineCount(); i++) {
          const info = cm.lineInfo(i);
          if ((info.bgClass || "").includes(c) || (info.wrapClass || "").includes(c)) return i + 1;
        }
        return null;
      }, cls),
  };
  return h;
}

function watchErrors(page, list, tag = "") {
  page.on("console", (msg) => {
    // The mocked GitHub error replies (rate limit, missing repository) are expected.
    if (msg.type() === "error" && /^Failed to load resource/.test(msg.text()) && /api\.github\.com/.test(msg.location().url || "")) return;
    if (msg.type() === "error") list.push(tag + msg.text());
  });
  page.on("pageerror", (err) => list.push(tag + "pageerror: " + err.message));
}

const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Expose-Headers": "x-ratelimit-remaining, x-ratelimit-reset" };

/** A pretend public repository octo/course. Returns counters. */
async function mockGithub(page) {
  const repo = {
    "README.md": "# Course files\n",
    "week1/main.m": "%% Week 1\nv = readmatrix('data/values.csv');\nfprintf('week1 total: %d\\n', sum(v(:)) + bonus());\n",
    "week1/bonus.m": "function b = bonus()\n    b = 100;\nend\n",
    "week1/data/values.csv": "1,2,3\n4,5,6\n",
    "week1/slides.pptx": "not imported",
    "week2/lab.m": "disp('week 2')\n",
    ".github/workflows/ci.yml": "hidden",
  };
  const state = { mode: "ok", treeCalls: 0, rawCalls: 0 };
  await page.route("https://api.github.com/**", async (route) => {
    const url = route.request().url();
    if (route.request().method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    state.treeCalls++;
    state.lastTreeUrl = url;
    if (state.mode === "ratelimit") {
      return route.fulfill({
        status: 403,
        headers: { ...CORS, "Content-Type": "application/json", "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(Math.floor(Date.now() / 1000) + 1800) },
        body: JSON.stringify({ message: "API rate limit exceeded" }),
      });
    }
    if (state.mode === "missing" || !/\/repos\/octo\/course\/git\/trees\//.test(url)) {
      return route.fulfill({ status: 404, headers: { ...CORS, "Content-Type": "application/json", "x-ratelimit-remaining": "57" }, body: JSON.stringify({ message: "Not Found" }) });
    }
    const tree = Object.entries(repo).map(([p, text]) => ({ path: p, type: "blob", size: text.length }));
    tree.push({ path: "week1", type: "tree" }, { path: "week1/data", type: "tree" });
    await route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json", "x-ratelimit-remaining": "58" }, body: JSON.stringify({ sha: "abc", tree, truncated: false }) });
  });
  await page.route("https://raw.githubusercontent.com/**", async (route) => {
    state.rawCalls++;
    const m = /raw\.githubusercontent\.com\/octo\/course\/[^/]+\/(.+)$/.exec(route.request().url());
    const text = m && repo[decodeURIComponent(m[1])];
    if (text === undefined) return route.fulfill({ status: 404, headers: CORS, body: "404: Not Found" });
    await route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "text/plain; charset=utf-8" }, body: text });
  });
  return state;
}

/* ------------------------------------------------------------------ demo GIF */

async function recordGif(browser, url) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "mylab-gif-"));
  const context = await browser.newContext({ viewport: { width: 1200, height: 700 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const h = helpers(page);
  const frames = [];
  const frame = async (caption, hold = 1000) => {
    await page.evaluate((text) => {
      let c = document.getElementById("demo-caption");
      if (!c) {
        c = document.createElement("div");
        c.id = "demo-caption";
        c.style.cssText =
          "position:fixed;left:50%;bottom:34px;transform:translateX(-50%);z-index:999;background:#0f766e;color:#fff;font:600 17px/1.3 system-ui,sans-serif;padding:9px 18px;border-radius:999px;box-shadow:0 6px 24px rgba(0,0,0,.25);white-space:nowrap";
        document.body.appendChild(c);
      }
      c.textContent = text;
      document.querySelectorAll(".toast").forEach((t) => t.remove());
    }, caption);
    await page.waitForTimeout(250);
    const file = path.join(dir, `f${String(frames.length).padStart(3, "0")}.png`);
    await page.screenshot({ path: file });
    frames.push({ file, hold });
  };

  await page.goto(url);
  await page.waitForSelector("#dlg-welcome[open]");
  await page.waitForSelector(READY, { timeout: 240000 });
  await page.waitForSelector("#loader", { state: "hidden" });
  await frame("mylab: MATLAB-language code in your browser", 1600);
  await page.click("#welcome-examples");
  await page.waitForSelector(".tab.active[data-name='welcome.m']");
  await frame("Open the welcome example…");
  await page.click("#btn-run");
  await page.waitForSelector(".plot-card img", { timeout: 120000 });
  await h.waitIdle();
  await page.evaluate(() => (document.querySelector("#console-out").scrollTop = 1e6));
  await frame("…press Run: output, variables and a plot", 1800);

  await h.openFile("step_demo.m");
  await frame("Step through: walk through a loop line by line");
  await page.click("#btn-step");
  await page.waitForSelector("#step-bar:not([hidden])");
  await frame("Step through: about to run the first line");
  for (let i = 0; i < 9; i++) {
    await page.click("#step-fwd");
    if (i !== 0 && i !== 4) await frame("Step forward: watch k and total change");
  }
  for (let i = 0; i < 3; i++) {
    await page.click("#step-back");
    await frame("…and step backwards, too");
  }
  await page.click("#step-exit");

  await page.evaluate(() => (location.hash = "#assignment=samples/first-steps.json"));
  await page.waitForSelector("#asg-panel:not([hidden])");
  await page.waitForSelector(".tab.active[data-name='p1_rect_area/rect_area.m']");
  await frame("Assignments: read the problem, complete the starter file", 1500);
  await page.click("[data-check='p1_rect_area']");
  await page.waitForSelector("[data-problem='p1_rect_area'] .asg-test.fail");
  await h.waitIdle();
  await frame("Check my answers: not right yet", 1500);
  await h.setCode("function a = rect_area(w, h)\n% RECT_AREA  Area of a rectangle with width w and height h.\n\na = w .* h;\n\nend\n");
  await frame("Fix the code…", 1300);
  await page.click("[data-check='p1_rect_area']");
  await page.waitForFunction(() => document.querySelectorAll("[data-problem='p1_rect_area'] .asg-test.pass").length === 2);
  await h.waitIdle();
  await frame("…check again: the tests pass", 3200);
  await context.close();

  const outFile = path.join(IMAGES, "demo.gif");
  const script = `
import json, sys
from PIL import Image
frames = json.loads(sys.argv[1])
out = sys.argv[2]
images = [Image.open(f["file"]).convert("RGB") for f in frames]
# One shared palette keeps the file small and the colors steady from frame to frame.
sheet = Image.new("RGB", (images[0].width, images[0].height * min(len(images), 6)))
for i, im in enumerate(images[:: max(1, len(images) // 6)][:6]):
    sheet.paste(im, (0, i * im.height))
palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
quant = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
quant[0].save(out, save_all=True, append_images=quant[1:], duration=[f["hold"] for f in frames], loop=0, optimize=True, disposal=1)
print(len(quant), "frames", images[0].size)
`;
  const r = spawnSync(PY, ["-c", script, JSON.stringify(frames), outFile], { encoding: "utf8" });
  if (r.status !== 0) throw new Error("building demo.gif failed (is Pillow installed?):\n" + r.stderr);
  const mb = fs.statSync(outFile).size / 1024 / 1024;
  fs.rmSync(dir, { recursive: true, force: true });
  if (mb > 6) throw new Error(`demo.gif is ${mb.toFixed(1)} MB (limit 6 MB)`);
  ok(`docs/images/demo.gif built: ${r.stdout.trim()}, ${mb.toFixed(2)} MB`);
}

/* ------------------------------------------------------------------ main */

async function main() {
  console.log("Building and serving web/ …");
  const { proc, url } = await startServer();
  console.log("Serving at", url);
  const browser = await chromium.launch({ headless: !args.has("--headed") });
  const consoleErrors = [];
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mylab-web-"));
  try {
    if (args.has("--gif")) {
      await recordGif(browser, url);
      return;
    }
    const context = await browser.newContext({
      viewport: { width: 1366, height: 768 },
      deviceScaleFactor: 2,
      acceptDownloads: true,
      permissions: ["clipboard-read", "clipboard-write"],
    });
    const page = await context.newPage();
    watchErrors(page, consoleErrors);
    const h = helpers(page);
    const { setCode: cm, consoleText, waitIdle } = h;

    const t0 = Date.now();
    await page.goto(url);
    await page.waitForSelector(".CodeMirror");
    await page.waitForSelector(".file-item[data-name='welcome.m']");
    ok("page loads with editor and seeded files");

    // First-run welcome
    await page.waitForSelector("#dlg-welcome[open]");
    for (const id of ["welcome-examples", "welcome-github", "welcome-assignment", "welcome-step"]) {
      check(await page.locator("#" + id).isVisible(), `welcome card ${id} missing`);
    }
    await page.click("#welcome-close");
    await page.waitForSelector("#dlg-welcome", { state: "hidden" });
    await page.waitForFunction(() => localStorage.getItem("mylab.welcomed") === "2");
    ok("first-run welcome points to examples, GitHub, the example assignment and step-through");

    await page.waitForSelector("#loader .steps li.active, body[data-engine='ready']", { timeout: 60000 });
    ok("loading screen shows progress steps");

    await page.waitForSelector(READY, { timeout: 240000 });
    ok(`engine ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    await page.waitForSelector("#loader", { state: "hidden" });

    // 1. Command window
    const input = page.locator("#console-input");
    await input.fill("x = magic(3)");
    await input.press("Enter");
    await page.waitForFunction(() => /8\s+1\s+6\s*\n\s*3\s+5\s+7\s*\n\s*4\s+9\s+2/.test(document.querySelector("#console-out").innerText), null, { timeout: 60000 });
    ok("command window shows x = magic(3)");

    // 2. Workspace
    await page.waitForSelector("#var-table tr[data-name='x']");
    const row = await page.locator("#var-table tr[data-name='x']").innerText();
    if (!/3×3/.test(row) || !/double/.test(row)) throw new Error("workspace row for x looks wrong: " + row);
    ok("workspace lists x (3×3 double)");
    await page.click("#var-table tr[data-name='x']");
    await page.waitForSelector("#dlg-var[open]");
    const varText = await page.locator("#var-body").innerText();
    if (!/8\s+1\s+6/.test(varText)) throw new Error("variable viewer shows: " + varText);
    const gridText = (await page.locator("#var-grid tbody tr").first().innerText()).replace(/\s+/g, " ");
    if (gridText !== "1 8 1 6") throw new Error("variable grid shows: " + gridText);
    await page.click("#dlg-var button[value='close']");
    ok("clicking a variable shows its value");

    // 3. History
    await input.press("ArrowUp");
    if ((await input.inputValue()) !== "x = magic(3)") throw new Error("history up arrow did not recall the command");
    await input.fill("");
    ok("up arrow recalls command history");

    // 4. Run a file that plots (welcome.m) with Ctrl+Enter
    await page.click(".file-item[data-name='welcome.m']");
    await page.click(".CodeMirror");
    await page.keyboard.press("Control+Enter");
    await page.waitForSelector(".plot-card img", { timeout: 120000 });
    await waitIdle();
    const nat = await page.evaluate(() => document.querySelector(".plot-card img").naturalWidth);
    if (!(nat > 100)) throw new Error("plot image did not render");
    if (!/eigenvalues of A/.test(await consoleText())) throw new Error("welcome.m output missing");
    ok("running welcome.m prints output and shows a plot");

    // Screenshot of the main UI with a plot
    await page.evaluate(() => {
      const out = document.querySelector("#console-out");
      out.scrollTop = out.scrollHeight;
      document.querySelectorAll(".toast").forEach((t) => t.remove());
    });
    await shot(page, "web-app.png");

    // 5. Runtime error with a line number, highlighted in the editor
    await page.click("#btn-new");
    await page.waitForSelector("#dlg-prompt[open]");
    await page.fill("#prompt-input", "broken_demo");
    await page.click("#prompt-ok");
    await page.waitForSelector(".tab.active:has-text('broken_demo.m')");
    await cm("a = 1;\nb = [1 2 3];\nc = b(7);\ndisp(c)\n");
    await page.click("#btn-run");
    await page.waitForSelector("#console-out .err");
    const errText = await page.locator("#console-out .err").last().innerText();
    if (!/line 3/.test(errText)) throw new Error("error message has no line number: " + errText);
    await page.waitForSelector(".CodeMirror .cm-error-line");
    const errLine = await h.markedLine("cm-error-line");
    if (errLine !== 3) throw new Error("highlighted error line is " + errLine);
    ok("runtime error shows 'line 3' and highlights the line");

    // 6. A function in a second file, called from the first
    await page.click("#btn-new");
    await page.fill("#prompt-input", "add_squares.m");
    await page.click("#prompt-ok");
    await page.waitForSelector(".tab.active:has-text('add_squares.m')");
    await cm("function s = add_squares(a, b)\n%ADD_SQUARES  Sum of squares.\n    s = a^2 + b^2;\nend\n");
    await page.click("#btn-new");
    await page.fill("#prompt-input", "use_function.m");
    await page.click("#prompt-ok");
    await page.waitForSelector(".tab.active:has-text('use_function.m')");
    await cm("r = add_squares(3, 4)\nfprintf('total: %d\\n', r);\n");
    await page.click("#btn-run");
    await page.waitForFunction(() => /total: 25/.test(document.querySelector("#console-out").innerText), null, { timeout: 60000 });
    ok("script calls a function defined in another .m file");

    // 7. Run selection
    await cm("p = 10;\nq = p * 3\n");
    await page.evaluate(() => {
      const cmi = document.querySelector(".CodeMirror").CodeMirror;
      cmi.setSelection({ line: 0, ch: 0 }, { line: 0, ch: 6 });
    });
    await page.click("#btn-run-sel");
    await waitIdle();
    await page.waitForSelector("#var-table tr[data-name='p']");
    if (await page.locator("#var-table tr[data-name='q']").count()) throw new Error("run selection ran too much");
    ok("run selection runs only the selected code");
    await cm("r = add_squares(3, 4)\nfprintf('total: %d\\n', r);\n");

    // 8. input() gives a friendly error
    await input.fill("v = input('n? ')");
    await input.press("Enter");
    await page.waitForFunction(() => /not available in the browser/.test(document.querySelector("#console-out").innerText));
    ok("input() reports that it is not available in the browser");

    // 9. Assistant panel and settings (no real API calls)
    await page.click("#btn-ai");
    await page.waitForSelector("#ai-panel:not([hidden])");
    if (await page.locator("#ai-explain-error").isHidden()) throw new Error("'Explain this error' should be visible after an error");
    ok("assistant panel opens and offers 'Explain this error'");
    await page.click("#btn-ai-settings");
    await page.waitForSelector("#dlg-ai-settings[open]");
    await page.selectOption("#ai-provider", "anthropic");
    if (await page.locator("#key-warning").isHidden()) throw new Error("key warning should be visible for API-key providers");
    await page.fill("#anthropic-key", "sk-ant-test-not-a-real-key");
    await page.fill("#anthropic-model", "claude-opus-5-5");
    await page.selectOption("#ai-provider", "ollama");
    await page.fill("#ollama-model", "qwen2.5-coder");
    await page.click("#ai-settings-save");
    await page.waitForSelector("#dlg-ai-settings", { state: "hidden" });
    await page.waitForFunction(() => localStorage.getItem("mylab.ai") !== null);
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("mylab.ai")));
    if (saved.provider !== "ollama" || saved.ollama.model !== "qwen2.5-coder" || saved.anthropic.key !== "sk-ant-test-not-a-real-key")
      throw new Error("settings not saved: " + JSON.stringify(saved));
    const label = await page.locator("#ai-provider-label").innerText();
    if (!/Ollama/.test(label)) throw new Error("provider label: " + label);
    ok("assistant settings save to localStorage");

    // Mocked providers: requests are intercepted inside the browser, nothing reaches a real API.
    const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*" };
    let ollamaBody = null;
    await page.route("http://localhost:11434/api/chat", async (route) => {
      ollamaBody = route.request().postDataJSON();
      const lines = [
        { message: { role: "assistant", content: "Index 7 is past the end of `b`.\n\n```matlab\n" }, done: false },
        { message: { role: "assistant", content: "c = b(3)\n```\n" }, done: false },
        { message: { role: "assistant", content: "" }, done: true },
      ];
      await route.fulfill({ status: 200, headers: { ...cors, "Content-Type": "application/x-ndjson" }, body: lines.map((l) => JSON.stringify(l)).join("\n") + "\n" });
    });
    await page.click("#ai-explain-error");
    await page.waitForSelector("#ai-messages .msg.assistant .codeblock");
    if (!ollamaBody || ollamaBody.model !== "qwen2.5-coder" || ollamaBody.stream !== true || ollamaBody.messages[0].role !== "system")
      throw new Error("unexpected Ollama request: " + JSON.stringify(ollamaBody).slice(0, 300));
    const lastUser = ollamaBody.messages[ollamaBody.messages.length - 1].content;
    if (!/Last error/.test(lastUser) || !/not available in the browser/.test(lastUser)) throw new Error("error context missing from request");
    ok("Ollama streaming (mocked) renders a reply with a code block");

    const sse = (events) => events.map(([t, d]) => `event: ${t}\ndata: ${JSON.stringify(d)}\n\n`).join("");
    const anthropicReply = (text, stop) =>
      sse([
        ["message_start", { type: "message_start", message: { id: "msg_test", type: "message", role: "assistant", model: "claude-opus-5-5", content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 10, output_tokens: 1 } } }],
        ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
        ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } }],
        ["content_block_stop", { type: "content_block_stop", index: 0 }],
        ["message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: 5 } }],
        ["message_stop", { type: "message_stop" }],
      ]);
    let anthropicReq = null;
    let anthropicStop = "end_turn";
    await page.route("https://api.anthropic.com/**", async (route) => {
      if (route.request().method() === "OPTIONS") return route.fulfill({ status: 204, headers: cors });
      anthropicReq = { url: route.request().url(), headers: route.request().headers(), body: route.request().postDataJSON() };
      const text = anthropicStop === "refusal" ? "" : "Here you go:\n\n```matlab\nz_from_ai = 6 * 7\n```\n";
      await route.fulfill({ status: 200, headers: { ...cors, "Content-Type": "text/event-stream" }, body: anthropicReply(text, anthropicStop) });
    });
    await page.click("#btn-ai-settings");
    await page.selectOption("#ai-provider", "anthropic");
    await page.click("#ai-settings-save");
    await page.waitForFunction(() => /Claude/.test(document.querySelector("#ai-provider-label").textContent));
    await page.fill("#ai-input", "Give me a tiny example");
    await page.press("#ai-input", "Enter");
    await page.waitForSelector("#ai-messages .msg.assistant:last-of-type .codeblock button:has-text('Run')", { timeout: 60000 });
    const b = anthropicReq && anthropicReq.body;
    if (!b || b.model !== "claude-opus-5-5" || b.max_tokens !== 16000 || b.fallbacks !== "default" || !b.stream || "temperature" in b)
      throw new Error("unexpected Anthropic request body: " + JSON.stringify(b).slice(0, 300));
    if (!/server-side-fallback-2026-07-01/.test(anthropicReq.headers["anthropic-beta"] || ""))
      throw new Error("missing beta header: " + JSON.stringify(anthropicReq.headers));
    ok("Anthropic SDK streaming (mocked) sends the expected request and renders the reply");
    await page.locator("#ai-messages .codeblock button:has-text('Run')").last().click();
    await page.waitForFunction(() => /z_from_ai =\s*\n?\s*42/.test(document.querySelector("#console-out").innerText), null, { timeout: 60000 });
    ok("'Run' on an assistant code block runs it in the command window");
    anthropicStop = "refusal";
    await page.fill("#ai-input", "another question");
    await page.press("#ai-input", "Enter");
    await page.waitForSelector("#ai-messages .msg.notice:has-text('declined')", { timeout: 60000 });
    ok("a refusal shows a friendly note");
    await page.unroute("https://api.anthropic.com/**");
    await page.unroute("http://localhost:11434/api/chat");

    // Switch the assistant off again so nothing tries to reach a provider.
    await page.click("#btn-ai-settings");
    await page.selectOption("#ai-provider", "none");
    await page.click("#ai-settings-save");
    await page.waitForFunction(() => document.querySelector("#ai-provider-label").textContent === "Off");
    await page.fill("#ai-input", "hello");
    await page.press("#ai-input", "Enter");
    await page.waitForSelector("#ai-messages .msg.notice:has-text('assistant is off')");
    ok("assistant works safely with no provider configured");
    await page.click("#btn-ai-close");

    /* ---------------- editor conveniences ---------------- */
    const ed = (fn, arg) => page.evaluate(fn, arg);
    const edValue = () => ed(() => window.mylab.editor.getValue());
    const focusAt = (line, ch) => ed(([l, c]) => (window.mylab.editor.focus(), window.mylab.editor.setCursor(l, c)), [line, ch]);
    const hoverAt = async (line, ch) => {
      const c = await ed(([l, k]) => window.mylab.editor.charCoords({ line: l, ch: k }, "window"), [line, ch]);
      await page.mouse.move(c.left + 3, (c.top + c.bottom) / 2 - 1);
      await page.mouse.move(c.left + 4, (c.top + c.bottom) / 2);
    };
    const problems = () => ed(() => window.mylab.app.allProblems().map((p) => ({ path: p.path, line: p.line + 1, severity: p.severity, message: p.message, run: !!p.run })));
    await input.fill("clear");
    await input.press("Enter");
    await waitIdle();
    await ed(() =>
      window.mylab.app.createProject({
        name: "Editor tests",
        files: [
          { path: "main.m", content: "%% Load\nvalues = [3 1 2];\ntotal = sum(values);\nfor k = 1:3\n    total = total + area(k, 2); % total so far\nend\n%% Report\nfprintf('total is %d\\n', total);\nresult = twice(total);\n\nfunction y = twice(x)\n% TWICE  Two times x.\ny = 2 * x;\nend\n" },
          { path: "lib/area.m", content: "function a = area(w, h)\n% AREA  Area of a rectangle.\n%   a = area(w, h) multiplies the two sides.\na = w * h;\nend\n" },
          { path: "notes.txt", content: "The total area is what we want.\n" },
        ],
        open: "main.m",
      })
    );
    await page.waitForSelector(".tab.active[data-name='main.m']");
    await page.waitForFunction(() => !!window.mylab.app.helpIndex());

    // Command palette
    await page.click(".CodeMirror");
    await page.keyboard.press("Control+Shift+P");
    await page.waitForSelector("#dlg-palette[open]");
    const pal = await ed(() => ({ items: document.querySelectorAll("#palette-list .pal-item").length, keys: document.querySelectorAll("#palette-list .pal-keys").length, all: window.mylab.app.commands.all().filter((c) => !c.hidden).length }));
    check(pal.items >= 60 && pal.keys >= 30, "the palette should list every command with its shortcut: " + JSON.stringify(pal));
    await page.keyboard.type("fmt doc");
    check(/Format document/.test(await page.locator(".pal-item.active").innerText()), "fuzzy search should find 'Format document': " + (await page.locator(".pal-item.active").innerText()));
    check(/Shift\s*Alt\s*F/.test(await page.locator(".pal-item.active .pal-keys").innerText()), "the palette should show the shortcut of a command");
    await page.fill("#palette-input", ">cmd window");
    check(/Go to the command window/.test(await page.locator(".pal-item.active").innerText()), "fuzzy search for 'cmd window' failed");
    await page.keyboard.press("Enter");
    await page.waitForSelector("#dlg-palette", { state: "hidden" });
    await page.waitForFunction(() => document.activeElement === document.querySelector("#console-input"));
    ok("command palette (Ctrl+Shift+P) lists every action with its shortcut, finds them by fuzzy search and runs them");

    // Quick file switcher
    await page.keyboard.press("Control+p");
    await page.waitForSelector("#dlg-palette[open]");
    check((await page.locator("#palette-list .pal-item").count()) === 3, "the file switcher should list the 3 project files");
    await page.keyboard.type("are");
    check(/lib\/area\.m/.test(await page.locator(".pal-item.active").innerText()), "file switcher did not find lib/area.m");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".tab.active[data-name='lib/area.m']");
    await page.keyboard.press("Control+p");
    await page.keyboard.type(":4");
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => window.mylab.editor.getCursor().line === 3);
    ok("quick file switcher (Ctrl+P) opens a project file by fuzzy name, and ':4' goes to a line");

    // Find and replace in the file
    await h.openFile("main.m");
    await focusAt(0, 0);
    await page.keyboard.press("Control+f");
    await page.waitForSelector("#find-bar:not([hidden])");
    await page.fill("#find-input", "total");
    await page.waitForFunction(() => document.querySelector("#find-count").textContent === "1 of 7");
    check((await page.locator(".cm-find-match").count()) >= 7, "matches should be highlighted in the editor");
    await page.keyboard.press("Enter");
    check((await page.locator("#find-count").innerText()) === "2 of 7", "Enter should go to the next match");
    await page.keyboard.press("Shift+Enter");
    check((await page.locator("#find-count").innerText()) === "1 of 7", "Shift+Enter should go to the previous match");
    await page.fill("#find-input", "Total");
    await page.click("#find-case");
    await page.waitForFunction(() => document.querySelector("#find-count").textContent === "No results");
    await page.click("#find-case");
    await page.click("#find-regex");
    await page.fill("#find-input", "t(ot)al\\b");
    await page.waitForFunction(() => /of 7$/.test(document.querySelector("#find-count").textContent));
    await page.fill("#find-input", "t(ot");
    await page.waitForFunction(() => document.querySelector("#find-count").textContent === "Not valid");
    await page.fill("#find-input", "\\b(to)tal = ");
    await page.waitForFunction(() => /of 2$/.test(document.querySelector("#find-count").textContent));
    await page.keyboard.press("Control+h");
    await page.waitForSelector("#replace-row:not([hidden])");
    check((await page.inputValue("#find-input")) === "\\b(to)tal = ", "opening replace must keep what was typed in the find box");
    await page.fill("#replace-input", "sub$1tal = ");
    await page.click("#replace-one");
    check((await ed(() => window.mylab.editor.getLine(2))) === "subtotal = sum(values);", "replace with a regex group: " + (await ed(() => window.mylab.editor.getLine(2))));
    await page.waitForFunction(() => document.querySelector("#find-count").textContent === "1 of 1");
    await page.click("#replace-all");
    check((await ed(() => window.mylab.editor.getLine(4))) === "    subtotal = total + area(k, 2); % total so far", "replace all: " + (await ed(() => window.mylab.editor.getLine(4))));
    await ed(() => (window.mylab.editor.undo(), window.mylab.editor.undo()));
    check((await ed(() => window.mylab.editor.getLine(2))) === "total = sum(values);", "undo should take the replacements back");
    await page.click("#find-regex");
    await page.focus("#find-input");
    await page.keyboard.press("Escape");
    await page.waitForSelector("#find-bar", { state: "hidden" });
    ok("find and replace (Ctrl+F / Ctrl+H): match count, case and regex toggles, replace one, replace all");

    // Search in all files
    await page.keyboard.press("Control+Shift+F");
    await page.waitForSelector("#search-view:not([hidden])");
    await page.fill("#search-input", "area");
    await page.waitForFunction(() => document.querySelector("#search-results").dataset.total === "6");
    check(/6 results in 3 files/.test(await page.locator("#search-status").innerText()), "search status: " + (await page.locator("#search-status").innerText()));
    await page.click("#search-word");
    await page.click("#search-case");
    await page.waitForFunction(() => document.querySelector("#search-results").dataset.total === "4");
    await page.locator(".search-file[data-path='lib/area.m'] .search-hit[data-line='3']").click();
    await page.waitForSelector(".tab.active[data-name='lib/area.m']");
    check((await ed(() => [window.mylab.editor.getSelection(), window.mylab.editor.getCursor("from").line].join())) === "area,2", "clicking a search result should select the match");
    await page.click("#search-word");
    await page.click("#search-case");
    await page.click("#tab-console");
    ok("search across all project files with click-to-jump");

    // Problems as you type
    await h.openFile("main.m");
    await page.waitForFunction(() => window.mylab.app.problemsOf("main.m").length === 0 && document.querySelector("#problems-count").hidden);
    check((await problems()).length === 0, "area() and twice() are project functions, so they must not be reported: " + JSON.stringify(await problems()));
    await focusAt(8, 0);
    await page.keyboard.type("m = meen(values);\n", { delay: 10 });
    await page.waitForSelector(".CodeMirror .cm-lint-warning");
    let probs = await problems();
    check(probs.length === 1 && probs[0].line === 9 && probs[0].severity === "warning" && /meen/.test(probs[0].message), "problems: " + JSON.stringify(probs));
    check((await page.locator(".CodeMirror .cm-lint-warning").innerText()) === "meen", "the unknown name should be underlined");
    check((await page.locator(".lint-mark.warning").count()) === 1 && /meen/.test(await page.locator(".lint-mark.warning").getAttribute("aria-label")), "gutter marker missing");
    check((await page.locator("#problems-count").innerText()) === "1", "the Problems tab should show a count");
    await ed(() => window.mylab.editor.replaceRange("if total > 3\n", { line: 9, ch: 0 }));
    await page.waitForSelector(".lint-mark.error");
    probs = await problems();
    check(probs.length === 1 && probs[0].severity === "error", "a syntax error should be reported as an error: " + JSON.stringify(probs));
    await ed(() => window.mylab.editor.replaceRange("", { line: 9, ch: 0 }, { line: 10, ch: 0 }));
    await page.waitForSelector(".CodeMirror .cm-lint-warning");
    await page.click("#tab-problems");
    await page.waitForSelector("#problems-view:not([hidden])");
    await focusAt(0, 0);
    await page.locator(".problem-item[data-kind='check']").click();
    check((await ed(() => window.mylab.editor.getSelection())) === "meen", "clicking a problem should jump to it");
    await hoverAt(8, 5);
    await page.waitForSelector("#code-tip:not([hidden]) .tip-problem");
    check(/meen/.test(await page.locator("#code-tip").innerText()), "hovering an underlined name should explain the problem");
    await shot(page, "problems.png", "#center");
    await page.mouse.move(5, 5);
    ok("problems as you type: underline, gutter marker, Problems list with click-to-jump, hover text");
    await page.click("#btn-run");
    await page.waitForSelector("#console-out .err");
    await waitIdle();
    check((await h.markedLine("cm-error-line")) === 9, "the failing line should be marked after the run");
    check((await page.locator("#console-out .err").last().locator("button").allInnerTexts()).join("|") === "Go to line 9|Explain this error|Fix this error", "error actions: " + (await page.locator("#console-out .err").last().locator("button").allInnerTexts()).join("|"));
    await page.waitForFunction(() => window.mylab.app.allProblems().some((p) => p.run && p.line === 8));
    await focusAt(0, 0);
    await page.click("#tab-problems");
    await page.locator(".problem-item[data-kind='run']").click();
    check((await ed(() => window.mylab.editor.getCursor().line)) === 8, "the run error in the Problems list should jump to its line");
    await ed(() => window.mylab.editor.replaceRange("", { line: 8, ch: 0 }, { line: 9, ch: 0 }));
    await page.waitForFunction(() => window.mylab.app.problemsOf("main.m").length === 0);
    await h.run();
    await h.waitConsole(/total is 18/);
    check((await problems()).length === 0, "a clean run should leave no problems");
    ok("a failed run marks the error line, lists it under Problems and offers 'Explain' and 'Fix this error'");

    // Hover help, usage hints, F1
    await hoverAt(2, 9);
    await page.waitForFunction(() => document.querySelector("#code-tip").dataset.name === "sum" && !document.querySelector("#code-tip").hidden);
    let tipText = await page.locator("#code-tip").innerText();
    check(/s = sum\(x\)/.test(tipText) && /Sum of elements\./.test(tipText), "hover on sum: " + tipText);
    await hoverAt(4, 22);
    await page.waitForFunction(() => document.querySelector("#code-tip").dataset.name === "area" && !document.querySelector("#code-tip").hidden);
    tipText = await page.locator("#code-tip").innerText();
    check(/a = area\(w, h\)/.test(tipText) && /Area of a rectangle\./.test(tipText) && /lib\/area\.m/.test(tipText), "hover on a project function: " + tipText);
    await hoverAt(1, 2);
    await page.waitForFunction(() => document.querySelector("#code-tip").dataset.name === "values" && !document.querySelector("#code-tip").hidden);
    check(/values is a variable: 1×3 double/.test(await page.locator("#code-tip").innerText()), "hover on a workspace variable: " + (await page.locator("#code-tip").innerText()));
    await hoverAt(4, 40);
    await page.waitForFunction(() => document.querySelector("#code-tip").hidden);
    await page.mouse.move(5, 5);
    await focusAt(9, 0);
    await page.keyboard.type("q = linspace(", { delay: 10 });
    await page.waitForFunction(() => document.querySelector("#code-tip").dataset.kind === "sig" && !document.querySelector("#code-tip").hidden);
    check(/y = linspace\(a, b, n\)/.test(await page.locator("#code-tip").innerText()), "usage hint: " + (await page.locator("#code-tip").innerText()));
    await page.keyboard.type("0, 1", { delay: 10 });
    check(await page.locator("#code-tip").isVisible(), "the usage hint should stay while typing the arguments");
    await page.keyboard.press("ArrowRight");
    await page.waitForFunction(() => document.querySelector("#code-tip").hidden);
    check((await ed(() => window.mylab.editor.getLine(9))) === "q = linspace(0, 1)", "typed line: " + (await ed(() => window.mylab.editor.getLine(9))));
    await focusAt(9, 7);
    await page.keyboard.press("F1");
    await page.waitForSelector("#dlg-help[open]");
    await page.waitForFunction(() => /y = linspace\(a, b, n\)/.test(document.querySelector("#help-body").textContent));
    check(/evenly spaced/.test(await page.locator("#help-body").innerText()) && (await page.inputValue("#help-name")) === "linspace", "F1 help for linspace is wrong");
    await page.fill("#help-name", "twice");
    await page.click("#help-go");
    await page.waitForFunction(() => /Two times x\./.test(document.querySelector("#help-body").textContent));
    await page.fill("#help-name", "linspase");
    await page.click("#help-go");
    await page.waitForSelector("#help-also button:has-text('linspace')");
    await page.click("#help-close");
    await page.waitForTimeout(150);
    ok("hover shows usage and a one-line description, typing name( shows the usage, F1 opens the full help");

    // Autocomplete
    await focusAt(9, 0);
    await ed(() => window.mylab.editor.replaceRange("", { line: 9, ch: 0 }, { line: 9, ch: 100 }));
    await page.keyboard.type("va", { delay: 30 });
    await page.waitForSelector(".CodeMirror-hints");
    let hints = await page.locator(".CodeMirror-hints li").allInnerTexts();
    check(/^x\s+values\s+1×3 double$/.test(hints[0].replace(/\n/g, " ")), "the first suggestion should be the workspace variable with its size: " + JSON.stringify(hints.slice(0, 3)));
    check(hints.some((t) => /^ƒ\s+var\s+Variance/.test(t.replace(/\n/g, " "))), "built-ins should come with a description: " + JSON.stringify(hints.slice(0, 6)));
    await page.keyboard.press("Tab");
    check((await ed(() => window.mylab.editor.getLine(9))) === "values", "Tab should accept the suggestion");
    await page.keyboard.type(" = ar", { delay: 30 });
    await page.waitForSelector(".CodeMirror-hints");
    hints = await page.locator(".CodeMirror-hints li").allInnerTexts();
    check(/^ƒ\s+area\s+Area of a rectangle\.$/.test(hints[0].replace(/\n/g, " ")), "project functions should be suggested with their description: " + JSON.stringify(hints.slice(0, 3)));
    await page.keyboard.press("Enter");
    check((await ed(() => [window.mylab.editor.getLine(9), window.mylab.editor.getCursor().line].join("|"))) === "values = ar|10", "Enter should make a new line while suggestions pop up by themselves");
    await ed(() => window.mylab.editor.replaceRange("", { line: 9, ch: 0 }, { line: 10, ch: 0 }));
    await page.keyboard.type("tw", { delay: 30 });
    await page.keyboard.press("Control+Space");
    await page.waitForSelector(".CodeMirror-hints");
    await page.waitForTimeout(100);
    hints = await page.locator(".CodeMirror-hints li").allInnerTexts();
    check(/^ƒ\s+twice\s+Two times x\.$/.test(hints[0].replace(/\n/g, " ")), "Ctrl+Space should list the function of this file first: " + JSON.stringify(hints.slice(0, 3)));
    await page.keyboard.press("Enter");
    check((await ed(() => window.mylab.editor.getLine(9))) === "twice", "Enter should accept a suggestion asked for with Ctrl+Space: " + (await ed(() => window.mylab.editor.getLine(9))));
    await ed(() => window.mylab.editor.replaceRange("", { line: 9, ch: 0 }, { line: 9, ch: 100 }));
    ok("autocomplete suggests workspace variables, project functions and built-ins with descriptions");

    // Go to definition, outline, rename
    await focusAt(4, 22);
    await page.keyboard.press("F12");
    await page.waitForSelector(".tab.active[data-name='lib/area.m']");
    check((await ed(() => window.mylab.editor.getCursor().line)) === 0, "F12 should go to the function line");
    await h.openFile("main.m");
    const twiceAt = await ed(() => window.mylab.editor.charCoords({ line: 8, ch: 11 }, "window"));
    await page.keyboard.down("Control");
    await page.mouse.click(twiceAt.left + 2, (twiceAt.top + twiceAt.bottom) / 2);
    await page.keyboard.up("Control");
    await page.waitForFunction(() => window.mylab.editor.getCursor().line === 10);
    check((await ed(() => window.mylab.state.active)) === "main.m", "Ctrl+click on a local function should stay in the file");
    await focusAt(7, 27);
    await page.keyboard.press("F12");
    await page.waitForFunction(() => window.mylab.editor.getCursor().line === 2);
    ok("go to definition (F12, Ctrl+click) for functions in this file and in other project files, and for variables");
    await page.keyboard.press("Control+Shift+O");
    await page.waitForSelector("#dlg-palette[open]");
    const outline = (await page.locator("#palette-list .pal-item .pal-label").allInnerTexts()).join("|");
    check(outline === "%% Load|%% Report|y = twice(x)", "outline: " + outline);
    await page.keyboard.type("rep");
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => window.mylab.editor.getCursor().line === 6);
    ok("outline (Ctrl+Shift+O) lists the functions and %% sections of the file and jumps to them");
    await focusAt(2, 2);
    await page.keyboard.press("F2");
    await page.waitForSelector("#dlg-prompt[open]");
    check(/Changes 5 places/.test(await page.locator("#prompt-text").innerText()), "rename should say how many places change: " + (await page.locator("#prompt-text").innerText()));
    await page.fill("#prompt-input", "2bad");
    check(await page.locator("#prompt-ok").isDisabled(), "an invalid name must be refused");
    await page.fill("#prompt-input", "grand");
    await page.click("#prompt-ok");
    await page.waitForFunction(() => window.mylab.editor.getLine(2) === "grand = sum(values);");
    let text = await edValue();
    check(text.includes("    grand = grand + area(k, 2); % total so far") && text.includes("fprintf('total is %d\\n', grand);") && text.includes("result = twice(grand);"), "rename result:\n" + text);
    await ed(() => window.mylab.editor.undo());
    check((await edValue()).includes("total = sum(values);") && !(await edValue()).includes("grand"), "one undo should take the whole rename back");
    ok("rename (F2) changes every use of a name in the file, but not comments or text in quotes");

    // Editing shortcuts
    await h.newFile("edit.m", "a = 1;\nb = 2;\nc = a + b;\n");
    await focusAt(1, 2);
    await page.keyboard.press("Control+/");
    check((await ed(() => window.mylab.editor.getLine(1))) === "% b = 2;", "Ctrl+/ should comment the line");
    await page.keyboard.press("Control+/");
    await page.keyboard.press("Shift+Alt+ArrowDown");
    check((await edValue()) === "a = 1;\nb = 2;\nb = 2;\nc = a + b;\n", "copy line down: " + JSON.stringify(await edValue()));
    await page.keyboard.press("Control+Shift+K");
    await focusAt(1, 2);
    await page.keyboard.press("Alt+ArrowUp");
    check((await edValue()) === "b = 2;\na = 1;\nc = a + b;\n", "move line up: " + JSON.stringify(await edValue()));
    await page.keyboard.press("Alt+ArrowDown");
    await page.keyboard.press("Control+]");
    check((await ed(() => window.mylab.editor.getLine(1))) === "    b = 2;", "Ctrl+] should indent the line");
    await page.keyboard.press("Control+[");
    check((await ed(() => window.mylab.editor.getLine(1))) === "b = 2;", "Ctrl+[ should outdent the line");
    await focusAt(0, 0);
    for (let i = 0; i < 2; i++) await page.keyboard.press("Control+d");
    check((await ed(() => window.mylab.editor.listSelections().length)) === 2, "Ctrl+D should add the next match as a second cursor");
    await page.keyboard.type("alpha");
    check((await edValue()) === "alpha = 1;\nb = 2;\nc = alpha + b;\n", "typing with two cursors: " + JSON.stringify(await edValue()));
    await page.keyboard.press("Escape");
    ok("editing shortcuts: comment (Ctrl+/), copy and move lines, indent/outdent, several cursors (Ctrl+D)");

    // Auto-indent, format, folding
    await ed(() => window.mylab.editor.setValue(""));
    await focusAt(0, 0);
    await page.keyboard.type("function r = pick(x)\nif x > 1\nr = 1;\nelse\nr = 2;\nend\nswitch x\ncase 1\ndisp(1)\nend\nfor k = 1:2\nwhile true\nbreak\nend\nend\nend", { delay: 4 });
    await page.keyboard.press("Escape");
    const typed = await edValue();
    check(
      typed === "function r = pick(x)\n    if x > 1\n        r = 1;\n    else\n        r = 2;\n    end\n    switch x\n        case 1\n            disp(1)\n    end\n    for k = 1:2\n        while true\n            break\n        end\n    end\nend",
      "auto-indent while typing gave:\n" + typed
    );
    ok("typing indents after if/for/while/function/switch and moves 'end', 'else' and 'case' back out");
    const messy = "for k = 1:3\nif k > 1   \n      disp('if end % not code')\n  x = [1 2 ...\n       3];\n   end\n        end\n";
    await ed((c) => window.mylab.editor.setValue(c), messy);
    await page.keyboard.press("Shift+Alt+F");
    const tidy = "for k = 1:3\n    if k > 1\n        disp('if end % not code')\n        x = [1 2 ...\n             3];\n    end\nend\n";
    await page.waitForFunction((t) => window.mylab.editor.getValue() === t, tidy);
    check(tidy.replace(/[ \t]+/g, "") === messy.replace(/[ \t]+/g, ""), "formatting may only change white space");
    await ed(() => window.mylab.editor.undo());
    check((await edValue()) === messy, "one undo should take the formatting back");
    await ed(() => window.mylab.editor.redo());
    ok("'Format document' fixes the indentation by block structure and changes nothing else");
    await ed(() => window.mylab.editor.setValue("%% One\nfor k = 1:3\n    if k > 1\n        disp(k)\n    end\nend\n%% Two\ny = 2;\n"));
    await page.waitForSelector(".CodeMirror-foldgutter-open");
    await page.locator(".CodeMirror-foldgutter-open").nth(1).click();
    await page.waitForSelector(".CodeMirror-foldmarker");
    check((await ed(() => window.mylab.editor.getAllMarks().filter((m) => m.__isFold).map((m) => [m.find().from.line, m.find().to.line].join("-")).join())) === "1-4", "folding the for block should hide lines 3 to 5");
    check((await page.locator(".bp-dot").count()) === 0, "clicking the fold arrow must not set a breakpoint");
    await ed(() => window.mylab.app.commands.run("edit.unfoldAll"));
    await ed(() => window.mylab.app.commands.run("edit.foldAll"));
    check((await page.locator(".CodeMirror-foldmarker").count()) === 2, "fold all should fold both %% sections");
    await ed(() => window.mylab.app.commands.run("edit.unfoldAll"));
    await focusAt(3, 4);
    await page.keyboard.press("Control+Shift+BracketLeft");
    check((await ed(() => window.mylab.editor.getAllMarks().filter((m) => m.__isFold).map((m) => m.find().from.line).join())) === "2", "the fold shortcut should fold the innermost block around the cursor");
    await page.keyboard.press("Control+Shift+BracketRight");
    check((await page.locator(".CodeMirror-foldmarker").count()) === 0, "the unfold shortcut should open the block again");
    ok("code folding for blocks and %% sections (gutter arrows, shortcuts, fold all)");

    // Tabs: unsaved dot, keyboard, autosave off
    await page.waitForFunction(() => !document.querySelector(".tab.dirty"));
    await ed(() => window.mylab.app.settings.set("autosave", false));
    await page.keyboard.type("z");
    await page.waitForSelector(".tab.active.dirty[data-name='edit.m']");
    await page.waitForTimeout(900);
    check((await page.locator(".tab.dirty").count()) === 1 && (await page.locator("#save-state").innerText()) === "Not saved", "with autosave off the tab should keep its 'not saved' dot");
    check((await ed(() => window.mylab.state.files.get("edit.m").content === window.mylab.editor.getValue())) === false, "with autosave off nothing should be saved yet");
    await page.keyboard.press("Control+s");
    await page.waitForFunction(() => !document.querySelector(".tab.dirty") && window.mylab.state.files.get("edit.m").content === window.mylab.editor.getValue());
    await ed(() => window.mylab.app.settings.set("autosave", true));
    await page.keyboard.type("z");
    await page.waitForSelector(".tab.active.dirty");
    await page.waitForFunction(() => !document.querySelector(".tab.dirty"));
    await page.focus(".tab.active");
    await page.keyboard.press("ArrowLeft");
    await page.waitForSelector(".tab.active[data-name='lib/area.m']");
    check((await ed(() => document.activeElement.dataset.name)) === "lib/area.m", "arrow keys should move between tabs and keep the focus on the tab row");
    await page.keyboard.press("Alt+PageDown");
    await page.waitForSelector(".tab.active[data-name='edit.m']");
    await page.keyboard.press("Alt+w");
    await page.waitForFunction(() => !document.querySelector(".tab[data-name='edit.m']"));
    ok("tabs show a dot for unsaved changes, work with the keyboard, and autosave can be turned off");

    // Variable viewer
    await input.fill("M = magic(6) / 7; big = reshape(1:60000, 300, 200); T = table([1;2;3], {'a';'b';'c'}, 'VariableNames', {'n','name'}); s.a = 1; s.m = magic(3); c = {1, 'two'; [1 2 3], {4}};");
    await input.press("Enter");
    await waitIdle();
    await page.click("#var-table tr[data-name='M']");
    await page.waitForSelector("#dlg-var[open] #var-grid tbody tr");
    check((await page.locator("#var-grid tbody tr").count()) === 6 && (await page.locator("#var-grid tbody tr").first().innerText()).replace(/\s+/g, " ") === "1 5 0.1429 0.8571 3.7143 2.7143 3.4286", "matrix grid: " + (await page.locator("#var-grid tbody tr").first().innerText()));
    await shot(page, "variable-viewer.png", "#dlg-var");
    await page.click("#var-view-text");
    check(/M =\s+5\.0000\s+0\.1429/.test(await page.locator("#var-body").innerText()) && (await page.locator("#var-grid-wrap").isHidden()), "the Text view should show the printed form");
    await page.click("#var-view-grid");
    await page.click("#var-copy-csv");
    await h.until(() => navigator.clipboard.readText().then((t) => t.startsWith("5,0.142857142857143,")));
    check((await ed(() => navigator.clipboard.readText())).trim().split("\n").length === 6, "the CSV should have 6 rows");
    await page.click("#var-plots [data-plot='bar']");
    await page.waitForSelector("#dlg-var", { state: "hidden" });
    await h.waitConsole(/figure; bar\(M\)/);
    await page.waitForSelector(".plot-card img", { timeout: 60000 });
    await waitIdle();
    await page.click("#var-table tr[data-name='big']");
    await page.waitForSelector("#dlg-var[open] #var-pager:not([hidden])");
    check((await page.locator("#var-rows-text").innerText()) === "Rows 1–100 of 300" && (await page.locator("#var-cols-text").innerText()) === "Columns 1–25 of 200", "pager: " + (await page.locator("#var-rows-text").innerText()));
    check((await page.locator("#var-grid tbody tr").count()) === 100, "a large matrix should be shown one page at a time");
    await page.click("#var-rows-last");
    await page.click("#var-cols-next");
    await page.waitForFunction(() => document.querySelector("#var-rows-text").textContent === "Rows 201–300 of 300" && document.querySelector("#var-cols-text").textContent === "Columns 26–50 of 200");
    check((await page.locator("#var-grid tbody tr").first().locator("td").first().innerText()) === "7701", "page 3 should start at big(201, 26) = 7701");
    await page.click("#dlg-var button[value='close']");
    await page.click("#var-table tr[data-name='T']");
    await page.waitForFunction(() => document.querySelector("#dlg-var").dataset.kind === "table");
    check((await page.locator("#var-grid").innerText()).replace(/\s+/g, " ").trim() === "n name 1 1 a 2 2 b 3 3 c", "table grid: " + (await page.locator("#var-grid").innerText()));
    await page.click("#dlg-var button[value='close']");
    await page.click("#var-table tr[data-name='s']");
    await page.waitForFunction(() => document.querySelector("#dlg-var").dataset.kind === "struct");
    await page.locator("#var-grid td.open button").click();
    await page.waitForFunction(() => document.querySelector("#var-title").textContent === "s.m" && document.querySelector("#dlg-var").dataset.kind === "matrix");
    check((await page.locator("#var-grid tbody tr").first().innerText()).replace(/\s+/g, " ") === "1 8 1 6", "s.m should open as a grid");
    await page.locator("#var-crumbs button").click();
    await page.waitForFunction(() => document.querySelector("#var-title").textContent === "s");
    await page.click("#dlg-var button[value='close']");
    await page.click("#var-table tr[data-name='c']");
    await page.waitForFunction(() => document.querySelector("#dlg-var").dataset.kind === "cell");
    check((await page.locator("#var-grid tbody").innerText()).replace(/\s+/g, " ").trim() === "1 1 two 2 [1 2 3] 1×1 cell", "cell grid: " + (await page.locator("#var-grid tbody").innerText()));
    await page.click("#dlg-var button[value='close']");
    await page.click("#btn-clear-plots");
    ok("variable viewer: grid for matrices, tables, cells and structs, pages for large data, copy as CSV, quick plots");

    // Breakpoints: run to the next breakpoint
    await h.newFile("bp.m", "total = 0;\nfor k = 1:4\n    total = total + k;\nend\ndone = total * 2;\n");
    await focusAt(4, 0);
    await page.keyboard.press("F9");
    await page.waitForSelector(".bp-dot");
    await page.keyboard.press("Alt+F5");
    await page.waitForSelector("#step-bar:not([hidden])", { timeout: 60000 });
    const bpStop = await h.stepInfo();
    check(bpStop.line === "5" && (await h.varValue("total")) === "10" && (await h.varValue("k")) === "4", "run to breakpoint should stop at line 5 with total = 10, got " + JSON.stringify(bpStop));
    await page.keyboard.press("Escape");
    await page.waitForSelector("#step-bar", { state: "hidden" });
    ok("breakpoints: F9 sets one, 'Run to the next breakpoint' stops there and shows the workspace");

    // Version history
    await h.newFile("story.m", "a = 1;\nb = 2;\n");
    await focusAt(1, 6);
    await page.keyboard.type("\nc = 3;");
    await page.keyboard.press("Control+s");
    await page.waitForFunction(() => window.mylab.state.files.get("story.m").content === "a = 1;\nb = 2;\nc = 3;\n");
    await page.keyboard.type("\nd = 4;");
    await page.keyboard.press("Control+s");
    await h.until(() => window.mylab.app.historyStore.list(window.mylab.state.project.id, "story.m").then((v) => v.length === 3));
    await ed(() => window.mylab.app.commands.run("file.history"));
    await page.waitForSelector("#dlg-history[open] .history-item");
    const versions = (await page.locator(".history-item").allInnerTexts()).map((t) => t.replace(/\s+/g, " "));
    check(versions.length === 3 && /Saved · 5 lines/.test(versions[0]) && /Saved · 4 lines/.test(versions[1]) && /Before your edits · 3 lines/.test(versions[2]), "versions: " + JSON.stringify(versions));
    await page.locator(".history-item").nth(2).click();
    check(/2 lines added/.test(await page.locator("#history-summary").innerText()) && (await page.locator("#history-diff .diff-add").count()) === 2, "the diff against the first version should show 2 added lines");
    await shot(page, "version-history.png", "#dlg-history");
    await page.click("#history-restore");
    await page.waitForFunction(() => window.mylab.editor.getValue() === "a = 1;\nb = 2;\n");
    check(await ed(() => window.mylab.app.historyStore.db !== null), "versions should be kept in IndexedDB");
    ok("version history keeps the last saves of a file, shows a diff and restores a version");

    // Appearance: themes, text size, shortcut presets, layout
    check((await ed(() => document.documentElement.dataset.shown)) === "light", "the default theme should follow the system (light here)");
    await page.emulateMedia({ colorScheme: "dark" });
    await page.waitForFunction(() => document.documentElement.dataset.shown === "dark" && getComputedStyle(document.body).backgroundColor === "rgb(15, 20, 24)");
    await page.emulateMedia({ colorScheme: "light" });
    await page.click("#btn-theme");
    await page.click("#theme-hc");
    await page.waitForFunction(() => document.documentElement.dataset.theme === "hc" && getComputedStyle(document.body).backgroundColor === "rgb(0, 0, 0)");
    await page.click("#btn-theme");
    await page.click("#theme-dark");
    await page.waitForFunction(() => getComputedStyle(document.body).backgroundColor === "rgb(15, 20, 24)" && localStorage.getItem("mylab.theme") === '"dark"');
    await h.openFile("main.m");
    await shot(page, "web-app-dark.png");
    await page.keyboard.press("Control+,");
    await page.waitForSelector("#dlg-settings[open]");
    await page.selectOption("#set-theme", "system");
    await page.locator("#set-font").fill("17");
    check((await ed(() => getComputedStyle(document.querySelector(".CodeMirror")).fontSize)) === "17px", "the text size setting should change the editor");
    await page.selectOption("#set-keys", "matlab");
    await page.click("#dlg-settings button[value='close']");
    check((await ed(() => [document.documentElement.dataset.shown, localStorage.getItem("mylab.theme")].join())) === "light,", "'Follow my system' should clear the saved theme");
    await ed(() => window.mylab.app.commands.run("view.fontReset"));
    await page.click("#btn-clear-console");
    await ed(() => (window.mylab.editor.focus(), window.mylab.editor.setSelection({ line: 1, ch: 0 }, { line: 2, ch: 22 })));
    await page.keyboard.press("F9");
    await h.waitConsole(/run lines 2–3 of main\.m/);
    await waitIdle();
    await page.click("#btn-keys");
    await page.waitForSelector("#dlg-keys[open]");
    const sheet = await page.locator("#keys-body").innerText();
    check(/F9\s+Run the selection/.test(sheet) && /Ctrl\+Enter/.test(sheet) && /Find in this file/.test(sheet) && /Rename/.test(sheet), "the cheat sheet should list the shortcuts of the chosen style");
    await page.selectOption("#keys-preset", "default");
    check(/Ctrl\+Shift\+Enter\s+Run the selection/.test(await page.locator("#keys-body").innerText()), "switching back to the standard shortcuts failed");
    await page.click("#dlg-keys button[value='close']");
    const sideBefore = await ed(() => document.querySelector("#side").getBoundingClientRect().width);
    await page.focus(".gutter[data-resize='side']");
    for (let i = 0; i < 3; i++) await page.keyboard.press("ArrowLeft");
    check((await ed(() => document.querySelector("#side").getBoundingClientRect().width)) === sideBefore + 60 && (await ed(() => JSON.parse(localStorage.getItem("mylab.sizes")).side)) === sideBefore + 60, "panel sizes should change with the keyboard and be remembered");
    await ed(() => window.mylab.app.commands.run("view.resetLayout"));
    check((await ed(() => document.querySelector("#side").getBoundingClientRect().width)) === sideBefore, "'Reset the panel sizes' should restore the layout");
    ok("appearance: light, dark and high-contrast themes that follow the system, text size, MATLAB-style keys, cheat sheet, resizable panels");

    // Assistant tied to the editor (mocked provider)
    let aiPrompt = null;
    let aiReply = "";
    await page.route("http://localhost:11434/api/chat", async (route) => {
      const body = route.request().postDataJSON();
      aiPrompt = body.messages[body.messages.length - 1].content;
      const lines = [{ message: { role: "assistant", content: aiReply }, done: false }, { message: { role: "assistant", content: "" }, done: true }];
      await route.fulfill({ status: 200, headers: { ...cors, "Content-Type": "application/x-ndjson" }, body: lines.map((l) => JSON.stringify(l)).join("\n") + "\n" });
    });
    await ed(() => {
      const a = window.mylab.app.assistant;
      a.settings = { ...a.settings, provider: "ollama", ollama: { url: "http://localhost:11434", model: "test-model" } };
      a.clear();
    });
    const broken = "a = [1 2 3];\nb = a(5);\ndisp(b)\n";
    await h.newFile("fixme.m", broken);
    await h.run();
    await page.waitForSelector("#console-out .err");
    aiReply = "The index 5 is past the end of `a`.\n\n```matlab\na = [1 2 3];\nb = a(3);\ndisp(b)\n```\n";
    await page.locator("#console-out .err").last().locator("button:has-text('Fix this error')").click();
    await page.waitForSelector("#ai-panel:not([hidden]) .msg.assistant .codeblock button.review:not([disabled])");
    check(/Fix the error in my code/.test(aiPrompt) && /Last error in fixme\.m \(line 2\)/.test(aiPrompt) && aiPrompt.includes("b = a(5);"), "the 'Fix this error' request should carry the file and the error");
    await page.locator("#ai-messages .codeblock button.review").last().click();
    await page.waitForSelector("#dlg-diff[open]");
    check((await page.locator("#diff-body .diff-del .diff-text").innerText()) === "b = a(5);" && (await page.locator("#diff-body .diff-add .diff-text").innerText()) === "b = a(3);", "the proposed edit should be shown as a diff");
    check(/1 line added, 1 line removed/.test(await page.locator("#diff-summary").innerText()), "diff summary: " + (await page.locator("#diff-summary").innerText()));
    await shot(page, "ai-diff.png", "#dlg-diff");
    await page.click("#diff-reject");
    await page.waitForSelector("#dlg-diff", { state: "hidden" });
    check((await edValue()) === broken, "rejecting must leave the file alone");
    await page.locator("#ai-messages .codeblock button.review").last().click();
    await page.click("#diff-accept");
    await page.waitForFunction(() => window.mylab.editor.getValue() === "a = [1 2 3];\nb = a(3);\ndisp(b)\n");
    await h.until(() => window.mylab.app.historyStore.list(window.mylab.state.project.id, "fixme.m").then((v) => v.some((x) => /assistant/.test(x.reason) && x.content.includes("a(5)"))));
    check(await page.locator("#ai-explain-selection").isHidden(), "'Explain selection' should only show while something is selected");
    await ed(() => window.mylab.editor.setSelection({ line: 1, ch: 0 }, { line: 1, ch: 9 }));
    aiReply = "It takes the third element.";
    await page.click("#ai-explain-selection");
    await page.waitForSelector("#ai-messages .msg.assistant:has-text('third element')");
    check(/Explain the selected lines/.test(aiPrompt) && /Selected lines 2-2:\n```matlab\nb = a\(3\);/.test(aiPrompt), "'Explain selection' should send the selected lines");
    await page.unroute("http://localhost:11434/api/chat");
    await ed(() => {
      const a = window.mylab.app.assistant;
      a.settings = { ...a.settings, provider: "none" };
      a.clear();
    });
    await page.click("#btn-ai-close");
    ok("assistant: 'Fix this error' on a failed run, 'Explain selection', and proposed edits shown as a diff to accept or reject");

    // Accessibility: names, keyboard reach, visible focus
    const a11y = await ed(() => {
      const visible = (e) => !e.closest("[hidden]") && e.getClientRects().length > 0;
      const name = (e) => (e.getAttribute("aria-label") || e.textContent || e.title || "").trim() || (e.labels && e.labels[0] && e.labels[0].textContent.trim()) || e.placeholder || "";
      const controls = [...document.querySelectorAll("button, a[href], input, select, textarea, [role='tab'], [role='separator'][tabindex]")].filter(visible);
      return {
        count: controls.length,
        unnamed: controls.filter((e) => !name(e)).map((e) => e.outerHTML.slice(0, 90)),
        unreachable: controls.filter((e) => e.tabIndex < 0 && !e.closest("[role='tablist']") && !e.closest(".CodeMirror") && !e.disabled).map((e) => e.outerHTML.slice(0, 90)),
        rows: [...document.querySelectorAll(".file-item, .folder-item, #var-table tbody tr")].filter((e) => e.tabIndex !== 0).length,
      };
    });
    check(a11y.count > 30 && a11y.unnamed.length === 0, "controls without a name: " + a11y.unnamed.join("\n"));
    check(a11y.unreachable.length === 0 && a11y.rows === 0, "controls that the Tab key can't reach: " + a11y.unreachable.join("\n"));
    await page.focus("#btn-run");
    await page.keyboard.press("Tab");
    const ring = await ed(() => {
      const s = getComputedStyle(document.activeElement);
      return { id: document.activeElement.id, visible: document.activeElement.matches(":focus-visible"), width: parseFloat(s.outlineWidth), style: s.outlineStyle };
    });
    check(ring.visible && ring.width >= 2 && ring.style === "solid", "keyboard focus should have a visible ring: " + JSON.stringify(ring));
    check((await page.locator(".CodeMirror textarea, .CodeMirror [contenteditable]").first().getAttribute("aria-label")) === "Code editor", "the editor needs a label for screen readers");
    ok("accessibility: every control has a name, can be reached with the keyboard and shows a focus ring");

    await ed(() => window.mylab.app.commands.setPreset("default"));
    await h.switchTo("My files");
    await page.click("#tab-console");

    /* ---------------- sections ---------------- */
    await input.fill("clear");
    await input.press("Enter");
    await waitIdle();
    await page.click("#btn-clear-plots");
    await h.openFile("sections_demo.m");
    await page.waitForSelector("#btn-run-section:not([hidden])");
    await page.evaluate(() => {
      window.mylab.editor.focus();
      window.mylab.editor.setCursor({ line: 9, ch: 0 });
    });
    await page.waitForSelector(".CodeMirror .cm-section-cur");
    const curLines = await page.evaluate(() => {
      const cmi = window.mylab.editor;
      const lines = [];
      for (let i = 0; i < cmi.lineCount(); i++) if ((cmi.lineInfo(i).wrapClass || "").includes("cm-section-cur")) lines.push(i + 1);
      return [lines[0], lines[lines.length - 1]];
    });
    check(curLines[0] === 7 && curLines[1] === 13, "highlighted section is lines " + curLines.join("–") + ", expected 7–13");
    await page.keyboard.press("Control+Enter");
    await h.waitConsole(/The ball lands after about 2\.02 s/);
    await waitIdle();
    check((await page.locator(".plot-card").count()) === 0, "running one section must not run the plot section");
    await page.waitForSelector("#var-table tr[data-name='height']");
    check(/run section “Make some data” \(lines 7–13\)/.test(await consoleText()), "section run label missing");
    ok("Ctrl+Enter runs only the %% section under the cursor, which is highlighted");
    await page.keyboard.press("Alt+Enter"); // run and advance: "Make some data" again, cursor moves to "Plot it"
    await waitIdle();
    await page.waitForTimeout(100);
    check((await page.evaluate(() => window.mylab.editor.getCursor().line)) === 14, "run-and-advance did not move to the next section");
    await page.click("#btn-run-section");
    await page.waitForSelector(".plot-card img", { timeout: 60000 });
    await waitIdle();
    ok("'Run section' and 'run section and advance' work");
    await page.keyboard.press("F5");
    await h.waitConsole(/Highest point: 20\.0 m/);
    await waitIdle();
    ok("F5 still runs the whole file");

    /* ---------------- publish report ---------------- */
    await page.click("#btn-publish");
    await page.waitForSelector("#dlg-report[open]", { timeout: 120000 });
    await page.fill("#report-author", "Ada Student");
    const report = await page.evaluate(() => window.mylab.app.lastReport().html);
    check(/<h1>Sections and reports<\/h1>/.test(report), "report title missing");
    check(/<h2>Plot it<\/h2>/.test(report), "report section title missing");
    check(/A falling ball, measured every tenth of a second\./.test(report), "report prose missing");
    check(/The ball lands after about 2\.02 s/.test(report) && /Highest point: 20\.0 m/.test(report), "report output missing");
    check(/<img alt="Figure 1" src="data:image\/png;base64,iVBOR/.test(report), "report has no embedded figure");
    check(/class="t-number"/.test(report) && /class="t-comment"|class="t-string"/.test(report), "report code is not highlighted");
    check(/Ada Student/.test(report) && /@media print/.test(report), "report author or print styles missing");
    check(!/<script/i.test(report) && !/https?:\/\/(?!www\.w3\.org)/.test(report.replace(/data:[^"]+/g, "")), "report must be standalone (no scripts or outside links)");
    const frameText = await page.frameLocator("#report-frame").locator("h1").innerText();
    check(frameText === "Sections and reports", "report preview did not render");
    const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#report-download")]);
    check(dl.suggestedFilename() === "sections_demo-report.html", "report download name: " + dl.suggestedFilename());
    const dlText = fs.readFileSync(await dl.path(), "utf8");
    check(dlText === report, "downloaded report differs from the preview");
    check(await page.locator("#report-print").isVisible(), "print button missing");
    ok("published report has titles, prose, highlighted code, output and an embedded figure");
    await page.frameLocator("#report-frame").locator("#s2").evaluate((n) => n.ownerDocument.defaultView.scrollTo(0, n.offsetTop - 150));
    await shot(page, "report.png");
    await page.click("#dlg-report button[value='close']");

    /* ---------------- step through ---------------- */
    await input.fill("clear");
    await input.press("Enter");
    await waitIdle();
    await h.openFile("step_demo.m");
    await page.click("#btn-step");
    await page.waitForSelector("#step-bar:not([hidden])", { timeout: 60000 });
    let si = await h.stepInfo();
    check(si.index === "0" && si.line === "5", "first step should be line 5, got " + JSON.stringify(si));
    check((await h.markedLine("cm-step-line")) === 5, "line 5 is not highlighted");
    for (let i = 0; i < 3; i++) await page.click("#step-fwd");
    si = await h.stepInfo();
    check(si.index === "3" && si.line === "8", "step 4 should be at line 8, got " + JSON.stringify(si));
    check((await h.varValue("total")) === "1" && (await h.varValue("k")) === "1", "at step 4, total and k should be 1");
    check(await page.locator("#var-table tr[data-name='total'].step-changed").count(), "total should be marked as changed");
    check((await page.locator("#step-out").innerText()) === "", "nothing should be printed before the first fprintf");
    await page.click("#step-fwd");
    check(/^k = 1, total = 1\s*$/.test(await page.locator("#step-out").innerText()), "output up to step 5 is wrong");
    await page.click("#step-fwd");
    si = await h.stepInfo();
    check(si.line === "8" && (await h.varValue("total")) === "5" && (await h.varValue("k")) === "2", "second pass through line 8 should show total = 5, k = 2");
    await page.keyboard.press("ArrowRight");
    check(/k = 2, total = 5/.test(await page.locator("#step-out").innerText()), "arrow key did not step forward");
    await shot(page, "step-through.png");
    await page.keyboard.press("ArrowLeft");
    await page.click("#step-back");
    await page.click("#step-back");
    si = await h.stepInfo();
    check(si.index === "3" && si.line === "8", "stepping back should return to step 4, got " + JSON.stringify(si));
    check((await h.varValue("total")) === "1" && (await h.varValue("k")) === "1", "after stepping back, total and k should be 1 again");
    check((await h.markedLine("cm-step-line")) === 8, "after stepping back, line 8 should be highlighted");
    check((await page.locator("#step-out").innerText()) === "", "after stepping back, the output should be empty again");
    ok("step forward and back show the right line, variables and output");

    // Breakpoints: click beside line 11, then jump.
    await page.locator(".CodeMirror-linenumber", { hasText: /^11$/ }).click();
    await page.waitForSelector(".bp-dot");
    await page.click("#step-next-bp");
    si = await h.stepInfo();
    check(si.line === "11", "next breakpoint should stop at line 11, got line " + si.line);
    check((await h.varValue("total")) === "30", "at the breakpoint total should be 30");
    await page.click("#step-last");
    check(/finished/.test(await page.locator("#step-where").innerText()), "last step should say finished");
    check(/^7\.50*$/.test(await h.varValue("average")), "final workspace should show average = 7.5");
    await page.click("#step-prev-bp");
    check((await h.stepInfo()).line === "11", "previous breakpoint should go back to line 11");
    await page.evaluate(() => {
      const s = document.querySelector("#step-slider");
      s.value = "2";
      s.dispatchEvent(new Event("input", { bubbles: true }));
    });
    check((await h.stepInfo()).line === "7", "slider should move to step 3 (line 7)");
    await page.locator(".CodeMirror-linenumber", { hasText: /^11$/ }).click();
    check((await page.locator(".bp-dot").count()) === 0, "clicking again should remove the breakpoint");
    ok("breakpoints are set in the gutter and next/previous breakpoint jump to them");
    await page.click("#step-exit");
    await page.waitForSelector("#step-bar", { state: "hidden" });
    check((await page.locator("#ws-title").innerText()).toLowerCase() === "workspace", "workspace title not restored");
    check((await page.locator("#var-table tr[data-name='total']").count()) === 0, "step-through must not change the user's workspace");

    // Steps inside another file's function switch the editor to that file.
    await h.openFile("use_function.m");
    await page.click("#btn-step");
    await page.waitForSelector("#step-bar:not([hidden])");
    await page.click("#step-fwd");
    si = await h.stepInfo();
    check(si.file === "add_squares.m" && si.line === "3" && si.function === "add_squares", "step 2 should be inside add_squares.m, got " + JSON.stringify(si));
    await page.waitForSelector(".tab.active[data-name='add_squares.m']");
    check((await h.varValue("a")) === "3" && (await h.varValue("b")) === "4", "function workspace should show a = 3, b = 4");
    check(/inside add_squares\(\) · call depth 2/.test(await page.locator("#step-where").innerText()), "call depth missing");
    await page.keyboard.press("Escape");
    ok("stepping into a function shows its file, workspace and call depth");

    // An error is shown at the failing step; long runs are cut off with a notice.
    await h.openFile("broken_demo.m");
    await page.click("#btn-step");
    await page.waitForSelector("#step-bar:not([hidden])");
    await page.click("#step-last");
    check(/Index exceeds/.test(await page.locator("#step-msg .step-error").innerText()), "error missing at the end");
    await page.click("#step-back");
    si = await h.stepInfo();
    check(si.line === "3" && /This line fails/.test(await page.locator("#step-msg .step-error").innerText()), "failing step should be line 3");
    check((await h.markedLine("cm-step-error")) === 3, "failing line is not marked");
    await page.click("#step-exit");
    await h.newFile("long_loop.m", "k = 0;\nwhile k < 9000\n    k = k + 1;\nend\n");
    await page.click("#btn-step");
    await page.waitForSelector("#step-bar:not([hidden])", { timeout: 120000 });
    await page.click("#step-last");
    check((await h.stepInfo()).index === "4999", "a long run should stop at 5000 steps");
    check(/only the first 5,000 steps/.test(await page.locator("#step-msg .step-note").innerText()), "truncation notice missing");
    await page.click("#step-exit");
    ok("the error is shown at the failing step, and a long trace is cut off with a notice");

    /* ---------------- folders, data files, files written by code ---------------- */
    await input.fill("clear");
    await input.press("Enter");
    await waitIdle();
    await page.evaluate(async () => {
      const app = window.mylab.app;
      // A CSV stored as raw bytes (the way an imported binary file is stored).
      await app.writeFiles([
        { path: "data/bytes.csv", content: new TextEncoder().encode("1,2\n3,4\n") },
        { path: "lib/triple.m", content: "function y = triple(x)\n    y = 3 * x;\nend\n" },
        { path: "data/read_it.m", content: "m = readmatrix('bytes.csv');\ns = sum(m(:))\nt = triple(s)\nwritematrix(m * 2, 'doubled.csv');\nplot(m(:, 1), m(:, 2), 'o-')\nsaveas(gcf, 'plot.png');\nclose all\n" },
      ]);
      app.openFile("data/read_it.m");
    });
    await page.waitForSelector(".folder-item[data-folder='data']");
    await page.waitForSelector(".file-item[data-name='lib/triple.m']");
    await h.run();
    await h.waitConsole(/s =\s*\n?\s*10\b[\s\S]*t =\s*\n?\s*30\b/);
    ok("binary data file round trip: bytes written in JS are read with readmatrix (folder as working directory, subfolder on the path)");
    await page.waitForSelector(".file-item[data-name='data/doubled.csv']");
    await page.waitForSelector(".file-item[data-name='data/plot.png']");
    const written = await page.evaluate(() => {
      const files = window.mylab.state.files;
      const png = files.get("data/plot.png").content;
      return { csv: files.get("data/doubled.csv").content, png: png instanceof Uint8Array ? [...png.slice(0, 4)] : null, size: png.length };
    });
    check(written.csv.trim() === "2,4\n6,8", "doubled.csv content: " + JSON.stringify(written.csv));
    check(written.png && written.png.join() === "137,80,78,71" && written.size > 500, "plot.png is not a PNG stored as bytes");
    check(/Your code saved 2 files/.test(await consoleText()), "console note about saved files missing");
    ok("files written by code (writematrix, saveas) appear in the tree");
    await h.openFile("data/plot.png");
    await page.waitForSelector("#dlg-viewer[open] img.viewer-img");
    check((await page.evaluate(() => document.querySelector(".viewer-img").decode().then(() => document.querySelector(".viewer-img").naturalWidth))) > 100, "image viewer did not render");
    const [imgDl] = await Promise.all([page.waitForEvent("download"), page.click("#viewer-download")]);
    check(imgDl.suggestedFilename() === "plot.png" && fs.statSync(await imgDl.path()).size === written.size, "downloaded plot.png differs");
    await page.click("#dlg-viewer button[value='close']");
    await page.locator(".file-item[data-name='data/doubled.csv']").hover();
    await page.click(".file-item[data-name='data/doubled.csv'] button[title='Preview']");
    await page.waitForSelector("#dlg-viewer[open] .data-table");
    check((await page.locator("#dlg-viewer .data-table td").allInnerTexts()).join() === "2,4,6,8", "CSV viewer table is wrong");
    await page.click("#dlg-viewer button[value='close']");
    ok("images and CSV files open in a viewer and can be downloaded");
    await page.click(".folder-item[data-folder='data']");
    check((await page.locator(".file-item[data-name='data/plot.png']").count()) === 0, "collapsing a folder should hide its files");
    await page.click(".folder-item[data-folder='data']");
    await page.waitForSelector(".file-item[data-name='data/plot.png']");
    // A second run only sends what changed, and files survive an engine round trip unchanged.
    await h.newFile("check_bytes.m", "m2 = readmatrix('data/doubled.csv');\nfprintf('doubled sum: %d\\n', sum(m2(:)));\ninfo = dir('data/plot.png');\nfprintf('png bytes: %d\\n', info.bytes);\n");
    await h.run();
    await h.waitConsole(/doubled sum: 20/);
    await h.waitConsole(new RegExp(`png bytes: ${written.size}\\b`));
    ok("the file tree has collapsible folders and stays in sync with the engine");

    /* ---------------- animations (engine output extra) ---------------- */
    await page.click("#btn-clear-plots");
    await h.newFile("spin.m", "t = linspace(0, 2*pi, 60);\nfor k = 1:8\n    plot(t, sin(t + k/2))\n    drawnow\nend\nmsgbox('Animation done')\n");
    await h.run();
    await page.waitForSelector(".plot-card.animation img", { timeout: 120000 });
    const anim = await page.evaluate(async () => {
      const img = document.querySelector(".plot-card.animation img");
      await img.decode();
      return { src: img.src.slice(0, 28), width: img.naturalWidth, label: document.querySelector(".plot-card.animation .plot-meta span").innerText, stills: document.querySelectorAll(".plot-card:not(.animation)").length };
    });
    check(anim.src === "data:image/gif;base64,R0lGOD" && anim.width > 100 && /^Animation \d+ · spin\.m/.test(anim.label), "animation card: " + JSON.stringify(anim));
    check(anim.stills === 1, "the final still should be shown as a figure too");
    check(/\.gif$/.test(await page.locator(".plot-card.animation a[download]").getAttribute("download")), "animation should download as .gif");
    check(/Animation done/.test(await page.locator("#console-out .msg").last().innerText()), "msgbox text should be shown on the page");
    ok("a drawnow loop produces an animation in the Plots panel (downloadable as .gif); msgbox shows a note");

    /* ---------------- projects ---------------- */
    await h.menu("menu-new-project");
    await page.fill("#prompt-input", "Scratch pad");
    await page.click("#prompt-ok");
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "Scratch pad");
    check((await h.files()).join() === "main.m", "a new project should start with main.m");
    await cm("disp('from scratch pad')\n");
    await h.menu("menu-rename-project");
    await page.fill("#prompt-input", "Lab 1");
    await page.click("#prompt-ok");
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "Lab 1");
    await h.newFile("util/helper.m", "function helper()\n    disp('helper in a folder')\nend\n");
    const [zipDl] = await Promise.all([page.waitForEvent("download"), h.menu("menu-export-zip")]);
    check(/^mylab-Lab-1-\d{4}-\d\d-\d\d\.zip$/.test(zipDl.suggestedFilename()), "zip name: " + zipDl.suggestedFilename());
    const zipPath = path.join(tmp, "Lab 1 backup.zip");
    await zipDl.saveAs(zipPath);
    await page.setInputFiles("#zip-input", zipPath);
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "Lab 1 backup");
    check((await h.files()).join() === "main.m,util/helper.m", "zip import files: " + (await h.files()).join());
    await h.openFile("main.m");
    await h.run();
    await h.waitConsole(/from scratch pad/);
    await h.menu("menu-delete-project");
    await page.click("#confirm-ok");
    await page.waitForFunction(() => document.querySelector("#project-name").textContent !== "Lab 1 backup");
    await h.switchTo("Lab 1");
    await h.switchTo("My files");
    await page.waitForSelector(".file-item[data-name='welcome.m']");
    ok("projects: create, rename, switch, delete, export .zip and import .zip");

    /* ---------------- open folder ---------------- */
    const courseDir = path.join(tmp, "course");
    fs.mkdirSync(path.join(courseDir, "lib"), { recursive: true });
    fs.mkdirSync(path.join(courseDir, "data"));
    fs.mkdirSync(path.join(courseDir, ".git"));
    fs.writeFileSync(path.join(courseDir, "main.m"), "v = readmatrix('data/v.csv');\nfprintf('folder total: %d\\n', add_all(v));\n");
    fs.writeFileSync(path.join(courseDir, "lib", "add_all.m"), "function s = add_all(v)\n    s = sum(v(:));\nend\n");
    fs.writeFileSync(path.join(courseDir, "data", "v.csv"), "10,20\n30,40\n");
    fs.writeFileSync(path.join(courseDir, "data", "photo.png"), Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64"));
    fs.writeFileSync(path.join(courseDir, "notes.docx"), "not a mylab file");
    fs.writeFileSync(path.join(courseDir, "big.csv"), Buffer.alloc(5 * 1024 * 1024 + 10, "1"));
    fs.writeFileSync(path.join(courseDir, ".git", "config"), "hidden");
    await page.setInputFiles("#folder-input", courseDir);
    await page.waitForSelector("#dlg-summary[open]");
    const summary = await page.locator("#dlg-summary").innerText();
    check(/4 files imported, 2 skipped/.test(summary) && /notes\.docx/.test(summary) && /big\.csv[\s\S]*larger than 5 MB/.test(summary), "import summary: " + summary);
    check(!/config/.test(summary), "hidden folders should be ignored silently");
    await page.click("#dlg-summary button[value='close']");
    check((await h.projectName()) === "course", "folder project name: " + (await h.projectName()));
    check((await h.files()).join() === "data/photo.png,data/v.csv,lib/add_all.m,main.m", "folder files: " + (await h.files()).join());
    check(await page.evaluate(() => window.mylab.state.files.get("data/photo.png").content instanceof Uint8Array), "photo.png should be stored as bytes");
    await page.waitForSelector(".tab.active[data-name='main.m']");
    await h.run();
    await h.waitConsole(/folder total: 100/);
    ok("Open folder imports .m and data files into a new project and lists what was skipped");

    /* ---------------- open from GitHub (mocked) ---------------- */
    const gh = await mockGithub(page);
    await h.menu("menu-open-github");
    await page.waitForSelector("#dlg-github[open]");
    gh.mode = "ratelimit";
    await page.fill("#github-url", "https://github.com/octo/course");
    await page.click("#github-open");
    await page.waitForSelector("#github-status.error-text");
    check(/hourly limit/.test(await page.locator("#github-status").innerText()), "rate limit message: " + (await page.locator("#github-status").innerText()));
    gh.mode = "missing";
    await page.fill("#github-url", "octo/secret-repo");
    await page.click("#github-open");
    await page.waitForFunction(() => /no public repository octo\/secret-repo/.test(document.querySelector("#github-status").textContent));
    check(/Private repositories can't be opened/.test(await page.locator("#github-status").innerText()), "private repo hint missing");
    await page.fill("#github-url", "not a repo");
    await page.click("#github-open");
    await page.waitForFunction(() => /doesn't look like a GitHub repository/.test(document.querySelector("#github-status").textContent));
    ok("GitHub errors are explained: rate limit, missing or private repository, bad address");
    gh.mode = "ok";
    gh.treeCalls = 0;
    await page.fill("#github-url", "https://github.com/octo/course/tree/main/week1");
    await page.click("#github-open");
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "course/week1", null, { timeout: 30000 });
    check((await h.files()).join() === "bonus.m,data/values.csv,main.m", "GitHub files: " + (await h.files()).join());
    check(gh.treeCalls === 1 && /\/git\/trees\/main\?recursive=1$/.test(gh.lastTreeUrl) && gh.rawCalls === 3, `GitHub requests: ${gh.treeCalls} tree (${gh.lastTreeUrl}), ${gh.rawCalls} raw`);
    await page.waitForSelector(".tab.active[data-name='main.m']");
    await h.run();
    await h.waitConsole(/week1 total: 121/);
    ok("Open from GitHub (mocked) imports a folder of a repository with one API request, and the code runs");
    await page.evaluate(() => (location.hash = "#gh=octo/course@v2"));
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "course (2)", null, { timeout: 30000 });
    check((await h.files()).join() === "README.md,week1/bonus.m,week1/data/values.csv,week1/main.m,week2/lab.m", "#gh files: " + (await h.files()).join());
    check(/\/git\/trees\/v2\?recursive=1$/.test(gh.lastTreeUrl), "ref from #gh link not used: " + gh.lastTreeUrl);
    await page.evaluate(() => window.mylab.createFile("week2/lab.m", "disp('my edit')\n"));
    const callsBefore = gh.treeCalls;
    await h.switchTo("My files");
    await page.evaluate(() => (location.hash = "#gh=octo/course@v2"));
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "course (2)");
    check(gh.treeCalls === callsBefore, "opening the same #gh link again should reuse the saved project");
    check((await page.evaluate(() => window.mylab.app.liveContent("week2/lab.m"))) === "disp('my edit')\n", "re-opening a #gh link must keep local edits");
    ok("#gh=owner/repo@ref links open a repository in one click and keep local changes on later visits");

    /* ---------------- share links ---------------- */
    await h.switchTo("My files");
    await h.newFile("share_me.m", "% shared by the test\nanswer = 6 * 7;\nfprintf('The answer is %d\\n', answer);\n");
    await page.click("#btn-share");
    await page.waitForSelector("#dlg-share[open]");
    await page.waitForFunction(() => /#code=/.test(document.querySelector("#share-link").value));
    check(await page.locator("#share-warning").isHidden(), "a short link should not show the length warning");
    await page.check("#share-run");
    await page.waitForFunction(() => /&run=1$/.test(document.querySelector("#share-link").value));
    const codeLink = await page.inputValue("#share-link");
    await page.click("#share-copy");
    await page.waitForSelector("#share-copied:not([hidden])");
    check(/Copied/.test(await page.locator("#share-copied").innerText()), "copy confirmation missing");
    check((await page.evaluate(() => navigator.clipboard.readText())) === codeLink, "clipboard does not hold the link");
    await shot(page, "share.png");
    await page.click("#dlg-share button[value='close']");
    // A long project link shows the warning.
    await h.switchTo("Lab 1");
    await page.evaluate(() => {
      let noise = "";
      for (let i = 0; i < 1500; i++) noise += Math.random().toString(36).slice(2) + "\n";
      return window.mylab.app.writeFiles([
        { path: "noise.txt", content: noise },
        { path: "pixel.png", content: new Uint8Array([137, 80, 78, 71, 0, 1, 2, 250, 255]) },
      ]);
    });
    await page.click("#btn-share");
    await page.check("#share-scope-project");
    await page.waitForFunction(() => /#project=/.test(document.querySelector("#share-link").value));
    await page.waitForSelector("#share-warning:not([hidden])");
    check(/GitHub/.test(await page.locator("#share-warning").innerText()), "long-link warning should suggest GitHub");
    await page.click("#dlg-share button[value='close']");
    await page.evaluate(() => window.mylab.app.removePaths(["noise.txt"]));
    const projectLink = await page.evaluate(() => window.mylab.app.buildShareLink("project", false));
    ok("Share builds #code= and #project= links, copies them and warns when a link is long");

    // Open the links as a different visitor (a second browser context with its own storage).
    const visitor = await browser.newContext({ viewport: { width: 1280, height: 720 } });
    const vp = await visitor.newPage();
    watchErrors(vp, consoleErrors, "[visitor] ");
    const vh = helpers(vp);
    await vp.goto(codeLink);
    await vp.waitForSelector("#temp-banner:not([hidden])");
    check((await vh.projectName()) === "Shared: share_me.m", "shared project name: " + (await vh.projectName()));
    check(/temporary/.test(await vp.locator("#temp-text").innerText()), "the shared project must be labelled temporary");
    check(await vp.locator("#dlg-welcome").isHidden(), "the welcome dialog should not cover a shared link");
    check((await vp.evaluate(() => window.mylab.editor.getValue())) === "% shared by the test\nanswer = 6 * 7;\nfprintf('The answer is %d\\n', answer);\n", "shared code differs");
    const own = await vp.evaluate(async () => {
      const store = window.mylab.app.store;
      const projects = await store.listProjects();
      const main = await store.listFiles("main");
      return { saved: projects.filter((p) => !p.temp).map((p) => p.name), main: main.map((f) => f.path) };
    });
    check(own.saved.join() === "My files" && own.main.includes("welcome.m") && !own.main.includes("share_me.m"), "the visitor's own files were touched: " + JSON.stringify(own));
    await vp.waitForSelector(READY, { timeout: 240000 });
    await vh.waitConsole(/The answer is 42/);
    ok("share link round trip in a second browser: code arrives in a temporary project and &run=1 runs it");
    await vp.click("#temp-keep");
    await vp.waitForFunction(() => document.querySelector("#project-name").textContent === "share_me");
    check(await vp.locator("#temp-banner").isHidden(), "banner should go away after keeping a copy");
    check((await vp.evaluate(() => location.hash)) === "", "the link's code should leave the address bar after keeping a copy");
    await vp.reload();
    await vp.waitForSelector(".file-item[data-name='share_me.m']");
    check((await vh.projectName()) === "share_me", "kept copy should persist across reloads");
    ok("'Keep a copy' saves the shared code as a new project without touching other files");
    await vp.goto(projectLink);
    await vp.waitForFunction(() => document.querySelector("#project-name").textContent === "Shared: Lab 1");
    check((await vh.files()).join() === "main.m,pixel.png,util/helper.m", "shared project files: " + (await vh.files()).join());
    check((await vp.evaluate(() => [...window.mylab.state.files.get("pixel.png").content].join())) === "137,80,78,71,0,1,2,250,255", "binary file changed in the share link");
    ok("project links carry folders and binary files");

    /* ---------------- assignments ---------------- */
    await input.fill("clear");
    await input.press("Enter");
    await waitIdle();
    await page.click("#btn-clear-plots");
    await page.click("#btn-clear-console");
    await page.evaluate(() => (location.hash = "#assignment=samples/first-steps.json"));
    await page.waitForSelector("#asg-panel:not([hidden])");
    check((await h.projectName()) === "First steps: two small functions", "assignment project name: " + (await h.projectName()));
    check((await h.files()).join() === "p1_rect_area/rect_area.m,p2_mean_score/mean_score.m,p2_mean_score/scores.csv", "assignment files: " + (await h.files()).join());
    await page.waitForSelector(".tab.active[data-name='p1_rect_area/rect_area.m']");
    check(await page.locator("[data-problem='p1_rect_area'] .md-table").count(), "prompt markdown table not rendered");
    check(await page.locator("[data-problem='p1_rect_area'] pre.md-code").count(), "prompt code block not rendered");
    check(/hidden tests/.test(await page.locator(".asg-hint").innerText()), "the panel must mention hidden tests");
    ok("an assignment link creates a project with one folder per problem and shows the prompts");

    await h.checkAnswers("#asg-check");
    check((await page.locator(".asg-test.fail").count()) === 4, "the starter files should fail all 4 visible tests");
    const failMsg = await page.locator("[data-problem='p1_rect_area'] .asg-test.fail").first().innerText();
    check(/rect_area\(3, 4\) should be 12/.test(failMsg) && /line 3 of the test/.test(failMsg), "failure message: " + failMsg);
    check((await page.locator("#asg-score").getAttribute("data-earned")) === "0", "score should be 0 at first");
    ok("wrong answers fail with the test's message and line");

    // exit/quit in student code is a failure, never a pass.
    await cm("function a = rect_area(w, h)\nexit\nend\n");
    await h.checkAnswers("[data-check='p1_rect_area']");
    check((await page.locator("[data-problem='p1_rect_area'] .asg-test.fail").count()) === 2, "exit in student code should fail both tests");
    ok("exit in student code counts as a failed test");

    await cm("function a = rect_area(w, h)\n% RECT_AREA  Area of a rectangle with width w and height h.\n\na = w .* h;\n\nend\n");
    await h.checkAnswers("[data-check='p1_rect_area']");
    check((await page.locator("[data-problem='p1_rect_area'] .asg-test.pass").count()) === 2, "the fixed answer should pass both tests");
    const score1 = await page.evaluate(() => ({ ...document.querySelector("#asg-score").dataset, text: document.querySelector("#asg-score").innerText, pill: document.querySelector("[data-problem='p1_rect_area'] .asg-points").innerText }));
    check(score1.earned === "4" && score1.visible === "10" && /Visible tests: 4 of 10 points/.test(score1.text), "score after fixing problem 1: " + JSON.stringify(score1));
    check(/visible tests: 4 of 4 points/.test(score1.pill), "problem pill: " + score1.pill);
    ok("a correct answer passes and the points update");
    await page.evaluate(() => document.querySelectorAll(".toast").forEach((t) => t.remove()));
    await shot(page, "assignment.png");

    // Watchdog: an endless loop times out after 15 s, the engine restarts and the next test still runs.
    await h.openFile("p2_mean_score/mean_score.m");
    await cm("function m = mean_score(filename)\nif strcmp(filename, 'scores.csv')\n    while true\n    end\nend\nx = readmatrix(filename);\nm = mean(x(:, 2));\nend\n");
    const tw = Date.now();
    await page.click("[data-check='p2_mean_score']");
    await page.waitForSelector("[data-problem='p2_mean_score'] .asg-test.timeout", { timeout: 60000 });
    const waited = (Date.now() - tw) / 1000;
    check(waited > 13 && waited < 40, `the watchdog fired after ${waited.toFixed(1)} s, expected about 15 s`);
    await page.waitForSelector("[data-problem='p2_mean_score'] .asg-test.pass", { timeout: 240000 });
    await page.waitForSelector(READY);
    await waitIdle();
    check(/Timed out/.test(await page.locator("[data-problem='p2_mean_score'] .asg-test.timeout").innerText()), "timeout message missing");
    check((await page.locator("#asg-score").getAttribute("data-earned")) === "7", "score after the timeout run should be 7");
    await input.fill("after_timeout = 1 + 1");
    await input.press("Enter");
    await h.waitConsole(/after_timeout =\s*\n?\s*2/);
    ok(`timeout watchdog: an endless loop is marked "timed out" after ${waited.toFixed(0)} s, the engine restarts and the next test runs`);

    const [subDl] = await Promise.all([page.waitForEvent("download"), page.click("#asg-download")]);
    check(subDl.suggestedFilename() === "First-steps-two-small-functions-submission.zip", "submission name: " + subDl.suggestedFilename());
    const subPath = path.join(tmp, "submission.zip");
    await subDl.saveAs(subPath);
    const zipNames = await page.evaluate(async (b64) => {
      const zip = await JSZip.loadAsync(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)));
      return Object.keys(zip.files).filter((n) => !zip.files[n].dir).sort();
    }, fs.readFileSync(subPath).toString("base64"));
    check(zipNames.join() === "p1_rect_area/rect_area.m,p2_mean_score/mean_score.m", "submission contents: " + zipNames.join());
    ok("'Download my submission' makes a .zip with one folder per problem");

    // Reloading the assignment link keeps the student's work and the last results.
    await page.reload();
    await page.waitForSelector("#asg-panel:not([hidden])");
    check((await page.evaluate(() => window.mylab.app.liveContent("p1_rect_area/rect_area.m"))).includes("a = w .* h;"), "reloading the assignment overwrote student work");
    check((await page.locator("#asg-score").getAttribute("data-earned")) === "7", "results were not kept across a reload");
    check((await page.locator("[data-problem='p2_mean_score'] .asg-test.timeout").count()) === 1, "per-test results were not kept");
    ok("reloading an assignment keeps the student's files and the last results");

    // A bundle from the real export tool, loaded from a file; then passed on embedded in a link.
    await page.setInputFiles("#asg-file-input", path.join(ROOT, "web", "samples", "intro-assignment.json"));
    await page.waitForFunction(() => document.querySelector("#project-name").textContent === "Intro Assignment: First Functions");
    check((await page.locator(".asg-problem").count()) === 4, "the example assignment should have 4 problems");
    check(/the remaining 35 points come from hidden tests/.test(await page.locator(".asg-hint").innerText()), "hidden-test points not explained");
    await page.click("#asg-share");
    await page.waitForFunction(() => navigator.clipboard.readText().then((t) => /#assignmentdata=/.test(t)));
    const asgLink = await page.evaluate(() => navigator.clipboard.readText());
    await vp.goto(asgLink);
    await vp.waitForFunction(() => document.querySelector("#project-name").textContent === "Intro Assignment: First Functions");
    check((await vp.locator(".asg-problem").count()) === 4 && (await vh.files()).includes("p4_rain_summary/rain_week.csv"), "embedded assignment link did not load");
    await visitor.close();
    ok("assignments also load from a file and from a link with the bundle embedded");
    await page.waitForSelector(READY, { timeout: 240000 });
    await page.evaluate(() => history.replaceState(null, "", location.pathname));
    await h.switchTo("My files");

    /* ---------------- polish ---------------- */
    check((await page.locator(".footbar a").getAttribute("href")) === "https://github.com/ghanemja/mylab/issues", "Report a problem link is wrong");
    await page.click("#foot-about");
    await page.waitForSelector("#dlg-about[open]");
    check(/not affiliated with The MathWorks, Inc\. MATLAB is a registered trademark of The MathWorks, Inc\./.test(await page.locator("#dlg-about .disclaimer").innerText()), "trademark disclaimer changed");
    await page.click("#dlg-about button[value='close']");
    await page.click("#btn-keys");
    check(/Run the section/.test(await page.locator("#dlg-keys").innerText()) && /breakpoint/.test(await page.locator("#dlg-keys").innerText()), "shortcut help not updated");
    await page.click("#dlg-keys button[value='close']");
    await page.click("#foot-welcome");
    await page.waitForSelector("#dlg-welcome[open]");
    await page.click("#welcome-close");
    ok("footer has 'Report a problem', About keeps the trademark disclaimer, shortcut help is updated");

    // 10. Stop restarts the engine
    await input.fill("k = 0; while true, k = k + 1; end");
    await input.press("Enter");
    await page.waitForSelector("#btn-stop:not([hidden])");
    await page.waitForTimeout(800);
    await page.click("#btn-stop");
    await page.waitForFunction(() => /Stopped\. The engine was restarted/.test(document.querySelector("#console-out").innerText));
    await page.waitForSelector(READY, { timeout: 240000 });
    await input.fill("y = 2 + 2");
    await input.press("Enter");
    await page.waitForFunction(() => /y =\s*\n?\s*4/.test(document.querySelector("#console-out").innerText), null, { timeout: 60000 });
    ok("Stop interrupts an infinite loop and the engine restarts");
    // After a restart the whole project is copied into the new engine again.
    await h.openFile("data/read_it.m");
    await h.run();
    await h.waitConsole(/t =\s*\n?\s*30\b/);
    ok("after a restart, files (including binary ones) are synced into the new engine");

    // 11. Files persist after reload
    await h.openFile("step_demo.m");
    await page.evaluate(() => {
      window.mylab.editor.setCursor(7, 3);
      window.mylab.app.toggleBreakpoint(10);
    });
    await h.openFile("welcome.m");
    await page.reload();
    await page.waitForSelector(".file-item[data-name='add_squares.m']");
    await page.waitForSelector(".tab.active[data-name='welcome.m']");
    check(await page.locator(".tab[data-name='step_demo.m']").count(), "open tabs should be remembered");
    await page.click(".tab[data-name='step_demo.m']");
    await page.waitForSelector(".bp-dot");
    const back = await page.evaluate(() => ({ cursor: window.mylab.editor.getCursor(), breaks: window.mylab.app.breakpointMap() }));
    check(back.cursor.line === 7 && back.cursor.ch === 3, "the cursor position should be remembered per file: " + JSON.stringify(back.cursor));
    check(JSON.stringify(back.breaks) === '{"step_demo.m":[11]}', "breakpoints should be remembered: " + JSON.stringify(back.breaks));
    await page.evaluate(() => window.mylab.app.toggleBreakpoint(10));
    ok("open tabs, the cursor in each file and breakpoints are remembered across reloads");
    const kept = await page.evaluate(() => {
      const f = window.mylab.state.files.get("data/plot.png");
      return { png: f && f.content instanceof Uint8Array && f.content[1] === 80, projects: window.mylab.state.projects.map((p) => p.name) };
    });
    check(kept.png, "binary files should persist in IndexedDB");
    check(["My files", "Lab 1", "course", "course/week1"].every((n) => kept.projects.includes(n)), "projects after reload: " + kept.projects.join(", "));
    ok("files (text and binary), folders and projects persist in IndexedDB across reloads");

    // 12. Phone-sized layout has no horizontal scrolling
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const pp = await phone.newPage();
    watchErrors(pp, consoleErrors, "[phone] ");
    await pp.goto(url);
    await pp.waitForSelector(".CodeMirror");
    await pp.waitForSelector("#dlg-welcome[open]");
    const dlgBox = await pp.locator("#dlg-welcome").boundingBox();
    check(dlgBox.width <= 390 && dlgBox.x >= 0, "welcome dialog does not fit a phone screen");
    await pp.tap("#welcome-close");
    await pp.click("#loader-hide");
    const overflow = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 1) throw new Error(`phone layout scrolls horizontally by ${overflow}px`);
    ok("phone layout stacks panels without horizontal scrolling");
    await pp.evaluate(() => (location.hash = "#assignment=samples/first-steps.json"));
    await pp.waitForSelector("#asg-panel:not([hidden])");
    await pp.tap("#project-btn");
    await pp.waitForSelector("#project-menu");
    const menuBox = await pp.locator("#project-menu").boundingBox();
    check(menuBox.x >= 0 && menuBox.x + menuBox.width <= 390, "project menu does not fit a phone screen");
    await pp.keyboard.press("Escape");
    const overflow2 = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check(overflow2 <= 1, `phone layout with an assignment scrolls horizontally by ${overflow2}px`);
    ok("phone layout also fits the assignment panel, the project menu and the welcome dialog");
    await pp.setViewportSize({ width: 320, height: 640 });
    await pp.evaluate(() => window.mylab.app.switchProject("main"));
    await pp.waitForSelector(".tab.active");
    const narrow = async (what) => {
      const o = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      check(o <= 1, `at 320 px the page scrolls sideways by ${o}px (${what})`);
    };
    await narrow("main view");
    await pp.evaluate(() => window.mylab.app.commands.run("find.replace"));
    await pp.waitForSelector("#replace-row:not([hidden])");
    await narrow("find and replace bar");
    await pp.evaluate(() => window.mylab.app.closeFind());
    await pp.evaluate(() => window.mylab.app.commands.run("find.inFiles"));
    await pp.fill("#search-input", "plot");
    await pp.waitForSelector(".search-hit");
    await narrow("search results");
    for (const [cmd, dlg] of [["palette.commands", "dlg-palette"], ["help.keys", "dlg-keys"], ["view.settings", "dlg-settings"], ["help.search", "dlg-help"]]) {
      await pp.evaluate((c) => window.mylab.app.commands.run(c), cmd);
      await pp.waitForSelector(`#${dlg}[open]`);
      const box = await pp.locator("#" + dlg).boundingBox();
      check(box.x >= 0 && box.x + box.width <= 320 && box.y >= 0 && box.y + box.height <= 640, `${dlg} does not fit a 320 px screen: ${JSON.stringify(box)}`);
      await pp.evaluate((d) => document.getElementById(d).close(), dlg);
    }
    await pp.tap("#btn-more");
    await pp.waitForSelector("#more-menu");
    const moreBox = await pp.locator("#more-menu").boundingBox();
    check(moreBox.x >= 0 && moreBox.x + moreBox.width <= 320, "the More menu does not fit a 320 px screen");
    await pp.keyboard.press("Escape");
    ok("the app is usable at 320 px: no sideways scrolling; find, search, palette, shortcuts, settings and help fit");
    await phone.close();

    // 13. No console errors
    if (consoleErrors.length) throw new Error("console errors:\n" + consoleErrors.join("\n"));
    ok("no console errors");

    console.log(`\nAll ${passed} checks passed.`);
  } finally {
    await browser.close();
    proc.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("\nFAILED:", e.message);
  process.exit(1);
});
