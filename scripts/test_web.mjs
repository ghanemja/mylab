#!/usr/bin/env node
// End-to-end test for the mylab web app (web/).
//
// Builds the wheel, serves web/ on a local port, opens the page in headless
// Chromium with Playwright and checks the main features. Needs network access
// to cdn.jsdelivr.net and cdnjs.cloudflare.com (Pyodide, CodeMirror, JSZip).
//
// Usage:  node scripts/test_web.mjs [--headed] [--no-screenshot]

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SCREENSHOT = path.join(ROOT, "docs", "images", "web-app.png");
const args = new Set(process.argv.slice(2));
const PY = process.env.PYTHON || "python3";

let passed = 0;
const results = [];
function ok(name) {
  passed++;
  results.push(`  ok   ${name}`);
  console.log(`  ok   ${name}`);
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

async function main() {
  console.log("Building and serving web/ …");
  const { proc, url } = await startServer();
  console.log("Serving at", url);
  const browser = await chromium.launch({ headless: !args.has("--headed") });
  const consoleErrors = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 2 });
    const page = await context.newPage();
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push("pageerror: " + err.message));

    const t0 = Date.now();
    await page.goto(url);
    await page.waitForSelector(".CodeMirror");
    await page.waitForSelector(".file-item[data-name='welcome.m']");
    ok("page loads with editor and seeded files");

    await page.waitForSelector("#loader .steps li.active, body[data-engine='ready']", { timeout: 60000 });
    ok("loading screen shows progress steps");

    await page.waitForSelector("body[data-engine='ready']", { timeout: 240000 });
    ok(`engine ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    await page.waitForSelector("#loader", { state: "hidden" });

    const cm = (code) => page.evaluate((c) => document.querySelector(".CodeMirror").CodeMirror.setValue(c), code);
    const consoleText = () => page.locator("#console-out").innerText();
    const waitIdle = () => page.waitForSelector("#btn-run:not([hidden])", { timeout: 120000 });

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
    if (!args.has("--no-screenshot")) {
      await page.evaluate(() => {
        const out = document.querySelector("#console-out");
        out.scrollTop = out.scrollHeight;
      });
      await page.waitForTimeout(400);
      fs.mkdirSync(path.dirname(SCREENSHOT), { recursive: true });
      await page.screenshot({ path: SCREENSHOT });
      ok(`screenshot saved to ${path.relative(ROOT, SCREENSHOT)}`);
    }

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
    const errLine = await page.evaluate(() => {
      const cmi = document.querySelector(".CodeMirror").CodeMirror;
      for (let i = 0; i < cmi.lineCount(); i++) {
        const info = cmi.lineInfo(i);
        if (info.bgClass && info.bgClass.includes("cm-error-line")) return i + 1;
      }
      return null;
    });
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

    // Files persist in IndexedDB across reloads (checked later after reload)

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

    // 10. Stop restarts the engine
    await input.fill("k = 0; while true, k = k + 1; end");
    await input.press("Enter");
    await page.waitForSelector("#btn-stop:not([hidden])");
    await page.waitForTimeout(800);
    await page.click("#btn-stop");
    await page.waitForFunction(() => /Stopped\. The engine was restarted/.test(document.querySelector("#console-out").innerText));
    await page.waitForSelector("body[data-engine='ready']", { timeout: 240000 });
    await input.fill("y = 2 + 2");
    await input.press("Enter");
    await page.waitForFunction(() => /y =\s*\n?\s*4/.test(document.querySelector("#console-out").innerText), null, { timeout: 60000 });
    ok("Stop interrupts an infinite loop and the engine restarts");

    // 11. Files persist after reload
    await page.reload();
    await page.waitForSelector(".file-item[data-name='add_squares.m']");
    ok("files persist in IndexedDB across reloads");

    // 12. Phone-sized layout has no horizontal scrolling
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const pp = await phone.newPage();
    pp.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push("[phone] " + msg.text());
    });
    pp.on("pageerror", (err) => consoleErrors.push("[phone] pageerror: " + err.message));
    await pp.goto(url);
    await pp.waitForSelector(".CodeMirror");
    await pp.click("#loader-hide");
    const overflow = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 1) throw new Error(`phone layout scrolls horizontally by ${overflow}px`);
    ok("phone layout stacks panels without horizontal scrolling");
    await phone.close();

    // 13. No console errors
    if (consoleErrors.length) throw new Error("console errors:\n" + consoleErrors.join("\n"));
    ok("no console errors");

    console.log(`\nAll ${passed} checks passed.`);
  } finally {
    await browser.close();
    proc.kill();
  }
}

main().catch((e) => {
  console.error("\nFAILED:", e.message);
  process.exit(1);
});
