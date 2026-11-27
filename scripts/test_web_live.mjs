#!/usr/bin/env node
// Browser checks for live collaboration (js/live.js): three people in three browser contexts of one
// browser, with a signalling server (scripts/signaling_server.mjs) and a STUN server started here, so
// that no public service is used.
//
// scripts/test_web.mjs runs these as part of the full suite. Run this file on its own while working on
// live collaboration (it starts its own web server):
//
//   node scripts/test_web_live.mjs [--browser chromium|firefox|webkit] [--headed]
//
// Why a STUN server: browsers hide their local addresses behind random "….local" names (mDNS).
// Chromium and Firefox resolve the names of another context on the same machine by themselves;
// WebKit's Linux port needs an mDNS responder, which CI containers don't run, and then never connects.
// A STUN server (here a few lines answering on 127.0.0.1) gives every browser a second, plain address,
// as a public STUN server does on the internet, so the three engines find each other the same way.
// The app takes the servers from ?signaling= and ?ice= (see docs/WEB.md).

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import dgram from "node:dgram";
import path from "node:path";

import { startSignalingServer } from "./signaling_server.mjs";
import { clipboardPermissions } from "./test_web_apps.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const READY = "body[data-engine='ready']";
const MARKER = "live_marker_5f0c2a"; // text in a shared file that must never reach the signalling server

/** A minimal STUN server: answers Binding requests (RFC 5389) with the address it sees. */
function startStun() {
  const sock = dgram.createSocket("udp4");
  sock.on("message", (msg, from) => {
    if (msg.length < 20 || msg.readUInt16BE(0) !== 0x0001 || msg.readUInt32BE(4) !== 0x2112a442) return;
    const res = Buffer.alloc(32);
    res.writeUInt16BE(0x0101, 0); // Binding success
    res.writeUInt16BE(12, 2);
    msg.copy(res, 4, 4, 20); // magic cookie and transaction id
    res.writeUInt16BE(0x0020, 20); // XOR-MAPPED-ADDRESS
    res.writeUInt16BE(8, 22);
    res.writeUInt8(1, 25); // IPv4
    res.writeUInt16BE(from.port ^ 0x2112, 26);
    from.address.split(".").forEach((b, i) => res.writeUInt8(Number(b) ^ [0x21, 0x12, 0xa4, 0x42][i], 28 + i));
    sock.send(res, from.port, from.address);
  });
  return new Promise((resolve) => sock.bind(0, "127.0.0.1", () => resolve({ port: sock.address().port, close: () => sock.close() })));
}

const hexToRgb = (hex) => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;

/** The checks. `t` = { ok, check, axePath? }. */
export async function testLive(browser, url, t, consoleErrors = []) {
  const { ok, check } = t;
  const relayed = []; // every message the signalling server received
  const sig = await startSignalingServer({ onMessage: (m) => relayed.push(m) });
  const stun = await startStun();
  const query = `?signaling=${encodeURIComponent(sig.url)}&ice=${encodeURIComponent(`stun:127.0.0.1:${stun.port}`)}`;
  const contexts = [];

  async function person(tag) {
    const context = await browser.newContext({ viewport: { width: 1366, height: 768 }, permissions: clipboardPermissions(browser) });
    await context.addInitScript(() => {
      window.__readClipboard = () =>
        Promise.race([navigator.clipboard.readText(), new Promise((r) => setTimeout(() => r("[the clipboard could not be read within 5 s]"), 5000))]);
    });
    contexts.push(context);
    const page = await context.newPage();
    page.on("console", (m) => m.type() === "error" && consoleErrors.push(`[live, ${tag}] ${m.text()}`));
    page.on("pageerror", (e) => consoleErrors.push(`[live, ${tag}] pageerror: ${e.message}`));
    return page;
  }
  const info = (p) => p.evaluate(() => window.mylab.app.live.info());
  const until = (p, fn, arg, timeout = 30000) => p.waitForFunction(fn, arg, { timeout, polling: 100 });
  const text = (p) => p.evaluate(() => window.mylab.editor.getValue());
  const content = (p, file) => p.evaluate((f) => (window.mylab.state.files.has(f) ? window.mylab.app.liveContent(f) : null), file);
  const peopleNames = (p) => p.evaluate(() => [...document.querySelectorAll("#live-people .live-person .sr-only")].map((e) => e.textContent));

  try {
    /* ---------------- the host starts a session ---------------- */
    const ana = await person("Ana");
    await ana.goto(url + query);
    await ana.waitForSelector("#dlg-welcome[open]");
    await ana.click("#welcome-close");
    await ana.evaluate(async (marker) => {
      const png = Uint8Array.from(atob("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="), (c) => c.charCodeAt(0));
      await window.mylab.app.createProject({
        name: "Live demo",
        files: [
          { path: "main.m", content: `% ${marker}\ntotal = sum(1:10);\nfprintf('main total %d\\n', total);\n` },
          { path: "together.m", content: "% typing together\n% line two\n% line three\n" },
          { path: "util/helper.m", content: "function y = helper(x)\n    y = 2 * x;\nend\n" },
          { path: "data/values.csv", content: "1,2,3\n4,5,6\n" },
          { path: "data/pixel.png", content: png },
        ],
        open: "main.m",
      });
    }, MARKER);
    await ana.click("#btn-live");
    await ana.waitForSelector("#dlg-live[open] #live-start-view:not([hidden])");
    check((await ana.locator("#live-project").innerText()) === "Live demo", "the dialog should name the project to share");
    await ana.fill("#live-name", "Ana");
    await ana.click("#live-start");
    await until(ana, () => document.querySelector("#live-link").value.includes("#live="));
    const link = await ana.inputValue("#live-link");
    const m = /#live=([A-Za-z0-9_-]{16})\.([A-Za-z0-9_-]{43})(&s=[^&]+)?$/.exec(link);
    check(m && link.startsWith(url + "#live="), "the link should be the app's address with #live=<session>.<key>: " + link);
    const [, room, key] = m;
    check(Buffer.from(key.replace(/-/g, "+").replace(/_/g, "/"), "base64").length === 32, "the key should hold 256 bits");
    await until(ana, () => document.activeElement && document.activeElement.id === "live-copy", null, 5000);
    await ana.click("#live-copy");
    await ana.waitForSelector("#live-copied:not([hidden])");
    check((await ana.evaluate(() => window.__readClipboard())) === link, "Copy link should put the link on the clipboard");
    check((await ana.evaluate(() => location.hash)) === new URL(link).hash, "the address bar should hold the link, so a reload joins again");
    const stored = await ana.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }));
    check(!stored.includes(key), "the key must not be saved in localStorage or sessionStorage");
    check(await ana.locator("#live-bar:not([hidden]) #live-people li").count() === 1, "the presence list should show the host");
    check((await ana.getAttribute("#btn-live", "aria-label")) === "Live collaboration: on, 1 person", "the Live button should say the session is on");
    await ana.click("#dlg-live button[value=close]");
    ok("starting a session makes a link with a random session id and a 256-bit key after #; Copy link; the key is not stored");

    /* ---------------- a second person joins with the link ---------------- */
    const ben = await person("Ben");
    await ben.goto(url + query + new URL(link).hash);
    await ben.waitForSelector("#dlg-prompt[open]");
    check(/name/i.test(await ben.locator("#prompt-title").innerText()), "joining should ask for a display name");
    await ben.fill("#prompt-input", "Ben");
    await ben.click("#prompt-ok");
    await until(ben, () => { const i = window.mylab.app.live.info(); return i && i.ready; }, null, 60000);
    check((await ben.locator("#project-name").innerText()) === "Live: Live demo", "the guest should get the files in a project named after the session");
    check(await ben.locator("#temp-banner:not([hidden]) #temp-keep").isVisible(), "the guest's copy should be temporary, with Keep a copy");
    const files = await ben.evaluate(() => window.mylab.app.sortedPaths());
    check(JSON.stringify(files) === JSON.stringify(["data/pixel.png", "data/values.csv", "main.m", "together.m", "util/helper.m"]), "the guest should get every file: " + files);
    check((await content(ben, "main.m")) === (await content(ana, "main.m")), "text files should arrive unchanged");
    const pixel = await ben.evaluate(() => Array.from(window.mylab.app.liveContent("data/pixel.png")));
    check(pixel.length === 70 && pixel[0] === 0x89 && pixel[1] === 0x50, "binary files should arrive byte for byte: " + pixel.length);
    check((await ben.evaluate(() => localStorage.getItem("mylab.liveName"))) === '"Ben"', "the display name should be remembered");
    await until(ana, () => document.querySelectorAll("#live-people li").length === 2);
    await until(ben, () => document.querySelectorAll("#live-people li").length === 2);
    const benSees = await peopleNames(ben);
    check(benSees.includes("Ben (you)") && benSees.some((n) => /^Ana, host, sharing their session$/.test(n)), "the presence list should name everyone for screen readers: " + benSees);
    await until(ana, () => /Ben joined the live session/.test(document.querySelector("#live-announce").textContent), null, 10000);
    const colors = (await info(ana)).people.map((p) => p.color);
    check(new Set(colors).size === 2, "the two people should have different colours: " + colors);
    ok("a second person opens the link, is asked for a name, and gets every file (binary too) in a temporary project");
    ok("everyone sees who is there: coloured avatars, names in a labelled list, and a polite announcement when someone joins");

    /* ---------------- typing at the same time ---------------- */
    await ana.click(".file-item[data-name='together.m']");
    await ben.click(".file-item[data-name='together.m']");
    await ana.evaluate(() => { const cm = window.mylab.editor; cm.focus(); cm.setCursor({ line: 0, ch: 999 }); });
    await ben.evaluate(() => { const cm = window.mylab.editor; cm.focus(); cm.setCursor({ line: 1, ch: 999 }); });
    await Promise.all([ana.keyboard.type(" AAAAAAAAAAAAAAAAAAAA", { delay: 12 }), ben.keyboard.type(" BBBBBBBBBBBBBBBBBBBB", { delay: 12 })]);
    // and at the very same spot
    await ana.evaluate(() => window.mylab.editor.setCursor({ line: 2, ch: 0 }));
    await ben.evaluate(() => window.mylab.editor.setCursor({ line: 2, ch: 0 }));
    await Promise.all([ana.keyboard.type("XXXXXXXXXX", { delay: 9 }), ben.keyboard.type("YYYYYYYYYY", { delay: 11 })]);
    const count = (s, c) => s.split(c).length - 1;
    let ta = "";
    let tb = "";
    for (let i = 0; i < 100; i++) {
      [ta, tb] = [await text(ana), await text(ben)];
      if (ta === tb && count(ta, "A") === 20 && count(ta, "B") === 20 && count(ta, "X") === 10 && count(ta, "Y") === 10) break;
      await ana.waitForTimeout(100);
    }
    check(ta === tb, `both editors should end with the same text:\n${ta}\n---\n${tb}`);
    check(count(ta, "A") === 20 && count(ta, "B") === 20 && count(ta, "X") === 10 && count(ta, "Y") === 10, "no keystroke may be lost:\n" + ta);
    check(/^% typing together A{20}\n% line two B{20}\n/.test(ta), "each person's typing should stay where they typed it:\n" + ta);
    check(/^(X{10}Y{10}|Y{10}X{10})% line three$/m.test(ta), "two people typing at the same spot should each keep their own words together:\n" + ta);
    ok("two people typing at once, also at the very same spot: both editors end with the same text and every keystroke");

    const anaColor = (await info(ana)).people.find((p) => p.me).color;
    await until(ben, () => document.querySelector(".CodeMirror .live-caret[data-name='Ana']"), null, 10000);
    const caret = await ben.evaluate(() => { const c = document.querySelector(".CodeMirror .live-caret[data-name='Ana']"); return { color: getComputedStyle(c).borderLeftColor, label: c.textContent }; });
    check(caret.color === hexToRgb(anaColor) && caret.label === "Ana", "Ana's cursor should show in Ben's editor in her colour with her name: " + JSON.stringify(caret));
    await until(ana, () => document.querySelector(".CodeMirror .live-caret[data-name='Ben']"), null, 10000);
    await ana.evaluate(() => window.mylab.editor.setSelection({ line: 1, ch: 0 }, { line: 1, ch: 6 }));
    await until(ben, () => [...document.querySelectorAll(".CodeMirror .live-sel")].some((e) => e.textContent.length > 0), null, 10000);
    ok("each sees the other's cursor and selection in their colour, with their name");

    // undo takes back only one's own typing (a pause of over 0.6 s starts a new undo step)
    await ben.waitForTimeout(700);
    await ben.evaluate(() => { const cm = window.mylab.editor; cm.focus(); cm.setCursor({ line: cm.lastLine(), ch: 0 }); });
    await ben.keyboard.type("QQQ");
    await ana.evaluate(() => { const cm = window.mylab.editor; cm.focus(); cm.setCursor({ line: 0, ch: 0 }); });
    await ana.keyboard.type("RRR");
    await until(ben, () => window.mylab.editor.getValue().startsWith("RRR"));
    await until(ana, () => window.mylab.editor.getValue().endsWith("QQQ"));
    await ben.waitForTimeout(700);
    await ben.keyboard.press("Control+Z");
    await until(ana, () => !window.mylab.editor.getValue().includes("QQQ") && window.mylab.editor.getValue().startsWith("RRR"));
    [ta, tb] = [await text(ana), await text(ben)];
    check(ta === tb && tb.includes("B".repeat(20)) && tb.includes("Y".repeat(10)), `Ctrl+Z should take back only Ben's last typing:\n${ta}\n---\n${tb}`);
    ok("undo takes back only your own typing, never someone else's");

    /* ---------------- new files, renames, deletions ---------------- */
    await ana.evaluate(() => window.mylab.openFile("util/helper.m"));
    await ana.evaluate(() => window.mylab.openFile("main.m"));
    await ana.evaluate(() => window.mylab.createFile("extra.m", "disp('an extra file')\n", { open: false }));
    await until(ben, () => document.querySelector(".file-item[data-name='extra.m']"));
    check((await content(ben, "extra.m")) === "disp('an extra file')\n", "a new file should arrive with its content");
    await ben.evaluate(() => window.mylab.app.movePaths([["util/helper.m", "tools/helper2.m"]]));
    await until(ana, () => document.querySelector(".file-item[data-name='tools/helper2.m']") && !document.querySelector(".file-item[data-name='util/helper.m']"));
    check(await ana.locator(".tab[data-name='tools/helper2.m']").count() === 1, "an open tab should follow a rename made by someone else");
    check((await content(ana, "tools/helper2.m")) === "function y = helper(x)\n    y = 2 * x;\nend\n", "a renamed file keeps its content");
    await ana.evaluate(() => window.mylab.app.removePaths(["extra.m"]));
    await until(ben, () => !document.querySelector(".file-item[data-name='extra.m']"));
    await ana.evaluate(() => window.mylab.app.writeFiles([{ path: "data/bytes.bin", content: new Uint8Array([0, 1, 2, 253, 254, 255]) }]));
    await until(ben, () => window.mylab.state.files.has("data/bytes.bin"));
    check(JSON.stringify(await ben.evaluate(() => Array.from(window.mylab.app.liveContent("data/bytes.bin")))) === "[0,1,2,253,254,255]", "a binary file written later should arrive too");
    // edits in a file the other has not opened
    await ben.evaluate(() => window.mylab.app.replaceText("data/values.csv", "1,2,3\n4,5,6\n7,8,9\n"));
    await until(ana, () => window.mylab.app.liveContent("data/values.csv") === "1,2,3\n4,5,6\n7,8,9\n");
    ok("new files, renames (an open tab follows), deletions, binary files and edits in unopened files reach everyone");

    /* ---------------- a wrong or missing key ---------------- */
    const cy = await person("Cy");
    const wrong = key.slice(0, -1) + (key.endsWith("A") ? "B" : "A");
    await cy.goto(url + query + `#live=${room}.${wrong}` + (m[3] || ""));
    await cy.waitForSelector("#dlg-prompt[open]");
    await cy.fill("#prompt-input", "Cy");
    await cy.click("#prompt-ok");
    await until(cy, () => { const i = window.mylab.app.live.info(); return i && i.status && /No one else is in this session/.test(i.status.text); }, null, 40000);
    check(!(await info(cy)).ready, "a wrong key must not get into the session");
    check(!(await cy.evaluate(() => window.mylab.state.projects.some((p) => /^Live: /.test(p.name)))), "a wrong key must not receive the project");
    check(!(await cy.evaluate((mk) => [...window.mylab.state.files.values()].some((f) => typeof f.content === "string" && f.content.includes(mk)), MARKER)), "a wrong key must not receive any file");
    await cy.click("#btn-live");
    await cy.waitForSelector("#dlg-live[open] #live-conn.problem");
    check(/No one else is in this session right now/.test(await cy.locator("#live-conn").innerText()), "the dialog should explain why no one is there");
    await cy.click("#dlg-live button[value=close]");
    await cy.evaluate((r) => (location.hash = "#live=" + r), room);
    await until(cy, () => [...document.querySelectorAll(".toast")].some((t) => /link is incomplete/.test(t.textContent)), null, 10000);
    // what the signalling server saw: hashed topics and encrypted blobs
    const kinds = new Set(relayed.map((r) => r.type));
    check(relayed.length > 5 && [...kinds].every((k) => ["subscribe", "unsubscribe", "publish", "ping"].includes(k)), "unexpected signalling messages: " + [...kinds]);
    const topics = new Set(relayed.flatMap((r) => r.topics || (r.topic ? [r.topic] : [])));
    check(topics.size === 2 && [...topics].every((tp) => /^mylab-[0-9a-f]{32}$/.test(tp)), "topics should be hashes (one per key): " + [...topics]);
    for (const r of relayed.filter((x) => x.type === "publish")) {
      check(Object.keys(r).sort().join() === "data,topic,type" && typeof r.data === "string", "a published message should only hold encrypted data: " + Object.keys(r));
      check(Buffer.from(r.data, "base64").subarray(0, 8).toString("latin1") === "\x07AES-GCM", "a published message should be AES-GCM encrypted");
    }
    const all = JSON.stringify(relayed);
    for (const secret of [key, wrong, room, MARKER, "Live demo", "together.m", "typing together"]) check(!all.includes(secret), "the signalling server saw " + secret);
    ok("a wrong key finds no one and gets nothing; an incomplete link is refused with a message");
    ok("the signalling server only saw hashed topics and AES-GCM encrypted messages: no key, session id, names or content");

    /* ---------------- a third person joins ---------------- */
    await cy.evaluate((h) => (location.hash = h), new URL(link).hash);
    await until(cy, () => { const i = window.mylab.app.live.info(); return i && i.ready; }, null, 60000);
    for (const p of [ana, ben, cy]) await until(p, () => document.querySelectorAll("#live-people li").length === 3);
    check((await content(cy, "together.m")) === (await content(ana, "together.m")) && (await content(cy, "data/values.csv")).endsWith("7,8,9\n"), "the third person should get the current text");
    ok("a third person joins: everyone sees three people and the latest files");

    /* ---------------- following a shared session ---------------- */
    await ana.waitForSelector(READY, { timeout: 240000 });
    await ben.waitForSelector(READY, { timeout: 240000 });
    await ben.evaluate(() => {
      window.__runs = 0;
      window.mylab.app.on("runStarting", () => window.__runs++);
    });
    await ana.fill("#console-input", "live_only = 41 + 1; disp('hello from Ana'); plot(1:5, (1:5).^2)");
    await ana.press("#console-input", "Enter");
    await until(ana, () => /hello from Ana/.test(document.querySelector("#console-out").innerText), null, 120000);
    check(await ben.locator("#tab-live").isVisible(), "the Live session tab should be shown during a session");
    await ben.click("#tab-live");
    await until(ben, () => /hello from Ana/.test(document.querySelector("#live-feed").innerText), null, 30000);
    await until(ben, () => { const img = document.querySelector("#live-feed img"); return img && img.naturalWidth > 0; }, null, 30000);
    check((await ben.locator("#live-follow option:checked").innerText()) === "Ana (host)", "Ben should follow the host, who shares by default");
    check(/>> live_only = 41 \+ 1/.test(await ben.locator("#live-feed").innerText()), "the command Ana typed should be shown");
    check((await ben.getAttribute("#live-feed img", "src")).startsWith("data:image/png;base64,iVBORw0KGgo"), "a figure should be shown as a PNG image");
    check((await ben.getAttribute("#live-feed img", "alt")) === "Figure from Ana's session", "the figure needs a text alternative");
    await until(ben, () => [...document.querySelectorAll("#live-vars tbody tr")].some((tr) => tr.textContent.startsWith("live_only") && /42/.test(tr.textContent)));
    check(await ben.evaluate(() => document.querySelector("#live-feed").innerHTML.includes("<script") === false), "output must be text");
    check(!/hello from Ana/.test(await ben.locator("#console-out").innerText()), "Ana's output must not appear in Ben's own command window");
    check((await ben.locator("#var-table tr[data-name='live_only']").count()) === 0, "nothing of Ana's may run in Ben's engine");
    check((await ben.evaluate(() => window.__runs)) === 0, "Ben's engine must not run anything he did not start");
    ok("following Ana: her command, output, figure (a PNG) and workspace appear in Ben's Live tab as text; nothing ran in his engine");

    // Ana changes the shared code; it runs in Ben's browser only when he presses Run
    await ana.evaluate(() => { const cm = window.mylab.editor; cm.focus(); cm.setCursor({ line: cm.lastLine(), ch: 0 }); });
    await ana.keyboard.type("disp('edited by Ana')\n");
    await until(ben, () => /edited by Ana/.test(window.mylab.app.liveContent("main.m")));
    await ben.waitForTimeout(500);
    check((await ben.evaluate(() => window.__runs)) === 0, "a received edit must not run by itself");
    await ben.click(".file-item[data-name='main.m']");
    await ben.click("#btn-run");
    await until(ben, () => /main total 55[\s\S]*edited by Ana/.test(document.querySelector("#console-out").innerText), null, 120000);
    check((await ben.evaluate(() => window.__runs)) === 1, "Run should run the code once, in Ben's engine");
    ok("shared code runs only when the person presses Run, in their own engine");

    /* ---------------- keeping a copy ---------------- */
    const projectId = (await info(ben)).projectId;
    await ben.click("#temp-keep");
    await until(ben, () => document.querySelector("#temp-banner").hidden && document.querySelector("#project-name").textContent === "Live demo");
    check((await info(ben)).ready && (await info(ben)).projectId === projectId, "the kept copy should stay in the session");
    check(await ben.evaluate(async (id) => (await window.mylab.app.store.listProjects()).some((p) => p.id === id && !p.temp), projectId), "the kept project should be saved in this browser");
    await ana.keyboard.type("disp('after the copy')\n");
    await until(ben, async (id) => {
      const f = (await window.mylab.app.store.listFiles(id)).find((x) => x.path === "main.m");
      return f && f.content.includes("after the copy");
    }, projectId, 20000);
    ok("Keep a copy saves the guest's copy in their browser; it stays in the session and keeps receiving changes");

    /* ---------------- accessibility of the live parts ---------------- */
    if (t.axePath) {
      const problems = [];
      const scan = async (p, label) => {
        await p.waitForTimeout(400);
        const found = await p.evaluate(async () => {
          const res = await axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] } });
          return res.violations
            .map((v) => ({ ...v, nodes: v.nodes.filter((n) => !(v.id === "scrollable-region-focusable" && n.target.join(" ") === ".CodeMirror-scroll")) }))
            .filter((v) => v.nodes.length)
            .map((v) => `${v.id} (${v.impact}, ${v.nodes.length}x): ${v.help} - ${v.nodes.slice(0, 2).map((n) => n.target.join(" ")).join(", ")}`);
        });
        for (const f of found) problems.push(`${label}: ${f}`);
      };
      await ana.addScriptTag({ path: t.axePath });
      await ben.addScriptTag({ path: t.axePath });
      await ana.click("#btn-live");
      await ana.waitForSelector("#dlg-live[open] #live-list li");
      await scan(ana, "live dialog during a session");
      await ana.click("#dlg-live button[value=close]");
      await scan(ben, "Live session tab, presence list and remote cursors");
      await ben.evaluate(() => (document.documentElement.dataset.theme = "dark"));
      await scan(ben, "Live session tab, dark theme");
      await ben.evaluate(() => delete document.documentElement.dataset.theme);
      check(!problems.length, "accessibility problems found by axe-core:\n  " + problems.join("\n  "));
      ok("axe-core finds no WCAG 2.2 A/AA problems in the live dialog, the presence list and the Live session tab");
    }

    /* ---------------- leaving, and ending the session ---------------- */
    await cy.click("#btn-live");
    await cy.click("#live-leave");
    await until(cy, () => !window.mylab.app.live.info() && document.querySelector("#live-bar").hidden);
    for (const p of [ana, ben]) await until(p, () => document.querySelectorAll("#live-people li").length === 2, null, 10000);
    await until(ana, () => /Cy left the live session/.test(document.querySelector("#live-announce").textContent), null, 10000);
    check((await content(cy, "together.m")) !== null, "the person who left keeps their copy");
    check((await cy.evaluate(() => location.hash)) === "", "leaving should take the link out of the address bar");
    ok("leaving: the others' lists update at once and announce it; the one who left keeps their copy");

    // coming back with the link: the copy catches up with what changed meanwhile
    await ana.evaluate(() => window.mylab.app.replaceText("together.m", window.mylab.app.liveContent("together.m") + "% while Cy was away\n"));
    await cy.evaluate((h) => (location.hash = h), new URL(link).hash);
    await until(cy, () => { const i = window.mylab.app.live.info(); return i && i.ready; }, null, 60000);
    await until(cy, () => /while Cy was away/.test(window.mylab.app.liveContent("together.m")));
    check((await cy.evaluate(() => window.mylab.state.projects.filter((p) => /^Live: /.test(p.name)).length)) === 1, "coming back should reuse the copy, not make another");
    for (const p of [ana, ben, cy]) await until(p, () => document.querySelectorAll("#live-people li").length === 3, null, 10000);
    ok("coming back with the link reuses the copy and brings it up to date with what changed meanwhile");

    await ana.click("#btn-live");
    await ana.waitForSelector("#live-end:not([hidden])");
    check(await ben.evaluate(() => { document.querySelector("#btn-live").click(); const hidden = document.querySelector("#live-end").hidden; document.querySelector("#dlg-live").close(); return hidden; }), "only the host may end the session for everyone");
    await ana.click("#live-end");
    await ana.waitForSelector("#dlg-confirm[open]");
    await ana.click("#confirm-ok");
    await until(ben, () => [...document.querySelectorAll(".toast")].some((x) => /Ana ended the live session/.test(x.textContent)), null, 15000);
    await until(ben, () => !window.mylab.app.live.info() && document.querySelector("#live-bar").hidden && document.querySelector("#tab-live").hidden);
    await until(cy, () => !window.mylab.app.live.info() && [...document.querySelectorAll(".toast")].some((x) => /Ana ended the live session/.test(x.textContent)), null, 15000);
    await until(ana, () => !window.mylab.app.live.info() && document.querySelector("#live-bar").hidden);
    check((await content(ben, "main.m")).includes("after the copy") && (await content(ana, "main.m")).includes("after the copy"), "everyone keeps their copy of the files");
    check((await ana.evaluate(() => location.hash)) === "", "ending should take the link out of the address bar");
    check(await ben.evaluate(() => !("live" in (window.mylab.state.project.meta || {}))), "an ended session should be forgotten by the kept project");
    ok("the host ends the session: everyone is told and disconnected, and keeps their copy");
  } finally {
    for (const c of contexts) await c.close().catch(() => {});
    await sig.close();
    stun.close();
  }
}

/* ------------------------------------------------------------------ on its own */

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
  const playwright = require("playwright");
  const argv = process.argv.slice(2);
  const b = argv.findIndex((a) => a === "--browser" || a.startsWith("--browser="));
  const browserName = b < 0 ? "chromium" : argv[b].includes("=") ? argv[b].slice("--browser=".length) : argv[b + 1];
  if (!["chromium", "firefox", "webkit"].includes(browserName)) {
    console.error(`--browser must be chromium, firefox or webkit, not ${browserName}`);
    process.exit(2);
  }
  let axePath = null;
  try {
    axePath = require.resolve("axe-core/axe.min.js");
  } catch {
    console.log("(axe-core is not installed: the accessibility check is skipped)");
  }
  let passed = 0;
  const t = {
    ok: (name) => console.log(`  ok   ${name}`, ++passed && ""),
    check: (cond, message) => {
      if (!cond) throw new Error(message);
    },
    axePath,
  };
  const { proc, url } = await startServer(process.env.PYTHON || "python3");
  const browser = await playwright[browserName].launch({ headless: !argv.includes("--headed") });
  console.log(`Testing live collaboration in ${browserName} ${browser.version()}`);
  const errors = [];
  try {
    await testLive(browser, url, t, errors);
    if (errors.length) throw new Error("console errors:\n" + errors.join("\n"));
    console.log(`\nAll ${passed} live collaboration checks passed.`);
  } catch (e) {
    if (errors.length && !/^console errors:/.test(e.message)) e.message += "\n\nconsole errors before the failure:\n" + errors.join("\n");
    console.error("\nFAILED:", e.message);
    process.exitCode = 1;
  } finally {
    await browser.close();
    proc.kill();
  }
}
