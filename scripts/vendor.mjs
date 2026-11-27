#!/usr/bin/env node
// Bundle the third-party JavaScript the web app loads from its own site into web/vendor/, so that no
// code from a CDN runs without an integrity hash. (jsDelivr's "+esm" modules are generated on demand
// and cannot carry a Subresource Integrity hash; a file served with the app needs none.)
//
//   web/vendor/anthropic-sdk.mjs   the Anthropic SDK, for the assistant (js/ai.js)
//   web/vendor/live-collab.mjs     Yjs, y-webrtc (with simple-peer) and y-protocols, for live
//                                  collaboration (js/live.js)
//
// The packages and their versions are pinned in scripts/vendor/package-lock.json; `npm ci` checks
// each package's SHA-512 from it. The licences of every bundled package are copied next to the
// bundles (web/vendor/licenses/), as the Apache-2.0, MIT, ISC and BSD licences require.
//
// Usage:  node scripts/vendor.mjs           # rebuild web/vendor/
//         node scripts/vendor.mjs --check   # fail if web/vendor/ differs from a fresh build
//
// To update a package: change its version in scripts/vendor/package.json, run
// `npm install --package-lock-only` in scripts/vendor, then this script, and run the browser tests.

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SPEC = path.join(ROOT, "scripts", "vendor");
const OUT = path.join(ROOT, "web", "vendor");
const check = process.argv.includes("--check");

// y-webrtc creates its WebRTC peers with `new Peer(options)` from simple-peer's prebuilt bundle. The
// app gives it a subclass that splits large messages (js/live-peer.js); this shim lets it do so
// without changing y-webrtc.
const PEER_SHIM = `import Peer from "simple-peer/simplepeer.min.js";
let Impl = Peer;
export function setPeerClass(cls) { Impl = cls || Peer; }
export { Peer as SimplePeer };
export default function MylabPeer(opts) { return new Impl(opts); }
`;

const BUNDLES = [
  {
    file: "anthropic-sdk.mjs",
    entry: 'export * from "@anthropic-ai/sdk";\nexport { default } from "@anthropic-ai/sdk";\n',
    title: (v) => `@anthropic-ai/sdk ${v("@anthropic-ai/sdk")}`,
  },
  {
    file: "live-collab.mjs",
    entry: [
      'export * as Y from "yjs";',
      'export { WebrtcProvider } from "y-webrtc";',
      'export * as awarenessProtocol from "y-protocols/awareness";',
      'export { SimplePeer, setPeerClass } from "./peer-shim.mjs";',
      "",
    ].join("\n"),
    files: { "peer-shim.mjs": PEER_SHIM },
    // every import of simple-peer's bundle except the shim's own goes to the shim
    redirect: { filter: /^simple-peer\/simplepeer\.min\.js$/, to: "peer-shim.mjs" },
    title: (v) => `yjs ${v("yjs")}, y-webrtc ${v("y-webrtc")}, y-protocols ${v("y-protocols")}, lib0 ${v("lib0")}, simple-peer ${v("simple-peer")}`,
    // simplepeer.min.js is a browserify bundle: it contains simple-peer's dependencies and
    // browserify's copy of the events module, whose licences must travel with it
    embeds: { "simple-peer": ["events"] },
  },
];

const work = fs.mkdtempSync(path.join(os.tmpdir(), "mylab-vendor-"));
try {
  for (const f of ["package.json", "package-lock.json"]) fs.copyFileSync(path.join(SPEC, f), path.join(work, f));
  const npm = spawnSync("npm", ["ci", "--no-audit", "--no-fund", "--ignore-scripts"], { cwd: work, stdio: "inherit", shell: process.platform === "win32" });
  if (npm.status !== 0) throw new Error("npm ci failed");

  const require = createRequire(path.join(work, "package.json"));
  const esbuild = require("esbuild");
  const pkgDir = (pkg, from = work) => {
    // the copy a package at `from` would load: its own node_modules first, then the top level
    for (let d = from; ; d = path.dirname(d)) {
      const p = path.join(d, "node_modules", pkg);
      if (fs.existsSync(path.join(p, "package.json"))) return p;
      if (d === work || d === path.dirname(d)) throw new Error(`${pkg} is not installed`);
    }
  };
  const readPkg = (dir) => JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"));
  const versionOf = (pkg) => readPkg(pkgDir(pkg)).version;
  const esbuildVersion = versionOf("esbuild");

  const built = path.join(work, "out");
  const packages = new Map(); // package directory -> name
  for (const b of BUNDLES) {
    const src = path.join(work, "src-" + b.file.replace(/\W/g, "_"));
    fs.mkdirSync(src);
    fs.writeFileSync(path.join(src, "entry.mjs"), b.entry);
    for (const [name, text] of Object.entries(b.files || {})) fs.writeFileSync(path.join(src, name), text);
    const plugins = [];
    if (b.redirect) {
      const target = path.join(src, b.redirect.to);
      plugins.push({
        name: "redirect",
        setup(build) {
          build.onResolve({ filter: b.redirect.filter }, (args) => (args.importer === target ? undefined : { path: target }));
        },
      });
    }
    const result = await esbuild.build({
      entryPoints: [path.join(src, "entry.mjs")],
      bundle: true,
      format: "esm",
      platform: "browser",
      target: "es2020",
      minify: true,
      legalComments: "eof",
      metafile: true,
      plugins,
      absWorkingDir: work,
      outfile: path.join(built, b.file),
      banner: { js: `/* ${b.title(versionOf)}, bundled with esbuild ${esbuildVersion} by scripts/vendor.mjs. Licences: licenses/ */` },
      logLevel: "warning",
    });
    // the packages that ended up in the bundle, and what their prebuilt files contain
    const add = (dir) => {
      if (packages.has(dir)) return;
      const pkg = readPkg(dir);
      packages.set(dir, pkg.name);
      return pkg;
    };
    for (const input of Object.keys(result.metafile.inputs)) {
      const m = input.replace(/\\/g, "/").match(/^(.*node_modules\/(?:@[^/]+\/)?[^/]+)\//);
      if (!m) continue;
      const dir = path.resolve(work, m[1]);
      const pkg = add(dir);
      const extra = pkg && b.embeds && b.embeds[pkg.name];
      if (!extra) continue;
      const walk = (d) => {
        for (const dep of Object.keys(readPkg(d).dependencies || {})) {
          const dd = pkgDir(dep, d);
          if (add(dd)) walk(dd);
        }
      };
      walk(dir);
      for (const name of extra) add(pkgDir(name));
    }
  }

  // the licence and notice files of every package that ended up in a bundle
  const names = [...new Set(packages.values())].sort();
  for (const [dir, pkg] of [...packages].sort((a, b) => a[1].localeCompare(b[1]))) {
    const version = readPkg(dir).version;
    const isLicence = (f) => /^(licen[cs]e|notice|copying|source)/i.test(f);
    // a package that ships no licence file: the copy kept in scripts/vendor/licenses/ (see its SOURCE.txt)
    const kept = path.join(SPEC, "licenses", pkg);
    let from = dir;
    let files = fs.readdirSync(dir).filter((f) => isLicence(f) && !/^source/i.test(f));
    if (!files.length && fs.existsSync(kept)) [from, files] = [kept, fs.readdirSync(kept).filter(isLicence)];
    if (!files.length) throw new Error(`${pkg} is bundled but ships no licence file; add one under scripts/vendor/licenses/${pkg}/`);
    const dest = path.join(built, "licenses", `${pkg.replace("/", "__")}@${version}`);
    fs.mkdirSync(dest, { recursive: true });
    for (const f of files) fs.copyFileSync(path.join(from, f), path.join(dest, f));
  }

  const listing = (dir) => {
    const out = {};
    const walk = (d) => {
      for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, e.name);
        if (e.isDirectory()) walk(p);
        else out[path.relative(dir, p).replace(/\\/g, "/")] = crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
      }
    };
    if (fs.existsSync(dir)) walk(dir);
    return out;
  };
  const fresh = listing(built);
  if (check) {
    const have = listing(OUT);
    const all = [...new Set([...Object.keys(fresh), ...Object.keys(have)])].sort();
    const bad = all.filter((n) => fresh[n] !== have[n]);
    for (const n of bad) console.log(`web/vendor/${n}: ${!have[n] ? "missing" : !fresh[n] ? "not produced by a fresh build" : "differs from a fresh build"}`);
    console.log(bad.length ? `web/vendor/ is out of date: run node scripts/vendor.mjs` : `web/vendor/ matches a fresh build (${all.length} files).`);
    process.exitCode = bad.length ? 1 : 0;
  } else {
    fs.rmSync(OUT, { recursive: true, force: true });
    fs.cpSync(built, OUT, { recursive: true });
    for (const b of BUNDLES) {
      const kb = (fs.statSync(path.join(OUT, b.file)).size / 1024).toFixed(0);
      console.log(`wrote web/vendor/${b.file} (${kb} KB, ${b.title(versionOf)})`);
    }
    console.log(`and the licences of: ${names.join(", ")}`);
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
