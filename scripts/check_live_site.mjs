#!/usr/bin/env node
// Check the deployed web app as a visitor would: load the page, wait for the engine, run code, draw a
// plot. Used by .github/workflows/post-release.yml against the GitHub Pages site.
//
// Usage:  node scripts/check_live_site.mjs <url> [--browser chromium|firefox|webkit] [--version X.Y.Z]
//
//   --version  also check that the site serves that mylab version (from wheels/manifest.json)
//
// Behind a TLS-intercepting proxy whose certificate Firefox does not trust, set MYLAB_TEST_IGNORE_HTTPS=1
// (local use only; integrity hashes still check every CDN file).

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const playwright = require("playwright");

const argv = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const url = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
const browserName = opt("--browser", "chromium");
const version = opt("--version", null);
if (!url || !playwright[browserName]) {
  console.error("usage: node scripts/check_live_site.mjs <url> [--browser chromium|firefox|webkit] [--version X.Y.Z]");
  process.exit(2);
}

const base = url.endsWith("/") ? url : url + "/";
let failures = 0;
const ok = (msg) => console.log(`  ok   ${msg}`);
const fail = (msg) => {
  failures++;
  console.log(`  FAIL ${msg}`);
};

const browser = await playwright[browserName].launch();
try {
  const context = await browser.newContext({ ignoreHTTPSErrors: process.env.MYLAB_TEST_IGNORE_HTTPS === "1" });
  const page = await context.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  console.log(`Checking ${base} in ${browserName} ${browser.version()}`);

  const t0 = Date.now();
  const resp = await page.goto(base);
  if (!resp || !resp.ok()) throw new Error(`the page answered HTTP ${resp && resp.status()}`);
  ok("the page loads");

  if (version) {
    const manifest = await page.evaluate(async () => (await fetch("wheels/manifest.json", { cache: "no-store" })).json());
    if (manifest.wheel && manifest.wheel.startsWith(`mylab_lang-${version}-`)) ok(`the site serves mylab ${version}`);
    else fail(`the site serves ${manifest.wheel}, not mylab ${version}`);
  }

  await page.waitForSelector("body[data-engine='ready'], body[data-engine='error']", { timeout: 300000 });
  if ((await page.getAttribute("body", "data-engine")) !== "ready") {
    throw new Error("the engine failed to start: " + (await page.locator("#loader-error").innerText()));
  }
  ok(`the engine starts (${((Date.now() - t0) / 1000).toFixed(0)} s, cold cache)`);

  const res = await page.evaluate(() => window.mylab.runCode("A = magic(4);\ndisp(sum(A(:)))", { kind: "command" }));
  if (res && !res.error && /\b136\b/.test(res.output || "")) ok("code runs: sum(magic(4)(:)) = 136");
  else fail("running code gave " + JSON.stringify(res && { output: res.output, error: res.error }));

  await page.evaluate(() => window.mylab.runCode("plot(1:10, (1:10).^2); title('check')", { kind: "command" }));
  try {
    await page.waitForSelector(".plot-card img", { timeout: 60000 });
    const w = await page.evaluate(() => document.querySelector(".plot-card img").naturalWidth);
    if (w > 100) ok(`a plot is drawn (${w} px wide)`);
    else fail(`the plot image is ${w} px wide`);
  } catch {
    fail("no plot appeared within 60 s");
  }

  if (errors.length) fail("console errors:\n    " + errors.join("\n    "));
  else ok("no console errors");
} catch (e) {
  fail(e.message);
} finally {
  await browser.close();
}
console.log(failures ? `\n${failures} check(s) failed.` : "\nThe live site works.");
process.exit(failures ? 1 : 0);
