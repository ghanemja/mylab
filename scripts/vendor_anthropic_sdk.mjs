#!/usr/bin/env node
// Bundle the Anthropic SDK into web/vendor/anthropic-sdk.mjs, so the web app's assistant loads it from
// its own site instead of from a CDN. (jsDelivr's "+esm" modules are generated on demand and cannot
// carry a Subresource Integrity hash; a file served with the app needs none.)
//
// The packages and their versions are pinned in scripts/vendor/package-lock.json; `npm ci` checks
// each package's SHA-512 from it. The licences of every bundled package are copied next to the
// bundle (web/vendor/licenses/), as the Apache-2.0 and MIT licences require.
//
// Usage:  node scripts/vendor_anthropic_sdk.mjs           # rebuild web/vendor/
//         node scripts/vendor_anthropic_sdk.mjs --check   # fail if web/vendor/ differs from a fresh build
//
// To update the SDK: change its version in scripts/vendor/package.json, run
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

const work = fs.mkdtempSync(path.join(os.tmpdir(), "mylab-vendor-"));
try {
  for (const f of ["package.json", "package-lock.json"]) fs.copyFileSync(path.join(SPEC, f), path.join(work, f));
  const npm = spawnSync("npm", ["ci", "--no-audit", "--no-fund", "--ignore-scripts"], { cwd: work, stdio: "inherit", shell: process.platform === "win32" });
  if (npm.status !== 0) throw new Error("npm ci failed");

  const require = createRequire(path.join(work, "package.json"));
  const esbuild = require("esbuild");
  const versionOf = (pkg) => JSON.parse(fs.readFileSync(path.join(work, "node_modules", pkg, "package.json"), "utf8")).version;
  const sdkVersion = versionOf("@anthropic-ai/sdk");
  const esbuildVersion = versionOf("esbuild");
  fs.writeFileSync(path.join(work, "entry.mjs"), 'export * from "@anthropic-ai/sdk";\nexport { default } from "@anthropic-ai/sdk";\n');

  const built = path.join(work, "out");
  const result = await esbuild.build({
    entryPoints: [path.join(work, "entry.mjs")],
    bundle: true,
    format: "esm",
    platform: "browser",
    target: "es2020",
    minify: true,
    legalComments: "eof",
    metafile: true,
    outfile: path.join(built, "anthropic-sdk.mjs"),
    banner: {
      js: `/* @anthropic-ai/sdk ${sdkVersion}, bundled with esbuild ${esbuildVersion} by scripts/vendor_anthropic_sdk.mjs. Licences: licenses/ */`,
    },
    logLevel: "warning",
  });

  // the licence and notice files of every package that ended up in the bundle
  const packages = new Set();
  for (const input of Object.keys(result.metafile.inputs)) {
    const m = input.replace(/\\/g, "/").match(/node_modules\/((?:@[^/]+\/)?[^/]+)\//g);
    if (m) packages.add(m[m.length - 1].replace(/^node_modules\//, "").replace(/\/$/, ""));
  }
  for (const pkg of [...packages].sort()) {
    const dir = path.join(work, "node_modules", pkg);
    const version = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8")).version;
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
    const names = [...new Set([...Object.keys(fresh), ...Object.keys(have)])].sort();
    const bad = names.filter((n) => fresh[n] !== have[n]);
    for (const n of bad) console.log(`web/vendor/${n}: ${!have[n] ? "missing" : !fresh[n] ? "not produced by a fresh build" : "differs from a fresh build"}`);
    console.log(bad.length ? `web/vendor/ is out of date: run node scripts/vendor_anthropic_sdk.mjs` : `web/vendor/ matches a fresh build (${names.length} files).`);
    process.exitCode = bad.length ? 1 : 0;
  } else {
    fs.rmSync(OUT, { recursive: true, force: true });
    fs.cpSync(built, OUT, { recursive: true });
    const kb = (fs.statSync(path.join(OUT, "anthropic-sdk.mjs")).size / 1024).toFixed(0);
    console.log(`wrote web/vendor/anthropic-sdk.mjs (${kb} KB, @anthropic-ai/sdk ${sdkVersion}) and the licences of: ${[...packages].sort().join(", ")}`);
  }
} finally {
  fs.rmSync(work, { recursive: true, force: true });
}
