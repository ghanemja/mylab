// Read a public GitHub repository without a server: one API request for the file list,
// then plain downloads from raw.githubusercontent.com (which do not count against the
// 60-requests-per-hour limit of the unauthenticated API).

import { toStored } from "./codec.js";

export const GITHUB_EXT = /\.(m|csv|txt|json|md|png|jpe?g|mat)$/i;
export const GITHUB_MAX_FILE = 5 * 1024 * 1024;
export const GITHUB_MAX_TOTAL = 25 * 1024 * 1024;
export const GITHUB_MAX_FILES = 300;

const NAME = /^[A-Za-z0-9_.-]+$/;

/**
 * Accepts  https://github.com/owner/repo[/tree/ref[/sub/folder]],  .../blob/ref/path/file.m,
 * owner/repo[/sub/folder][@ref]  (the form used in #gh= links) and git@github.com:owner/repo.git.
 * Returns { owner, repo, ref, path } (ref "" means the default branch) or null.
 */
export function parseGithub(input) {
  let s = String(input || "").trim();
  if (!s) return null;
  let ref = "";
  let path = "";
  const ssh = /^git@github\.com:([^/]+)\/(.+?)(?:\.git)?$/.exec(s);
  if (ssh) s = `${ssh[1]}/${ssh[2]}`;
  const url = /^(?:https?:\/\/)?(?:www\.)?github\.com\/(.+)$/i.exec(s);
  let parts;
  if (url) {
    parts = url[1].split(/[?#]/)[0].split("/").filter(Boolean).map((p) => {
      try {
        return decodeURIComponent(p);
      } catch {
        return p;
      }
    });
    if (parts.length >= 4 && (parts[2] === "tree" || parts[2] === "blob")) {
      ref = parts[3];
      let rest = parts.slice(4);
      if (parts[2] === "blob") rest = rest.slice(0, -1); // a file: open its folder
      path = rest.join("/");
      parts = parts.slice(0, 2);
    } else {
      parts = parts.slice(0, 2);
    }
  } else {
    if (/^[a-z]+:\/\//i.test(s)) return null;
    const at = s.lastIndexOf("@");
    if (at > 0) {
      ref = s.slice(at + 1);
      s = s.slice(0, at);
    }
    parts = s.split("/").filter(Boolean);
    path = parts.slice(2).join("/");
    parts = parts.slice(0, 2);
  }
  if (parts.length < 2) return null;
  const owner = parts[0];
  const repo = parts[1].replace(/\.git$/, "");
  if (!NAME.test(owner) || !NAME.test(repo)) return null;
  if (path.split("/").some((p) => p === "..")) return null;
  return { owner, repo, ref, path: path.replace(/\/+$/, "") };
}

export function githubSpecText({ owner, repo, ref, path }) {
  return `${owner}/${repo}${path ? "/" + path : ""}${ref ? "@" + ref : ""}`;
}

export class GithubError extends Error {}

function describeFailure(resp, spec) {
  const remaining = resp.headers.get("x-ratelimit-remaining");
  if ((resp.status === 403 || resp.status === 429) && (remaining === "0" || remaining === null)) {
    const reset = Number(resp.headers.get("x-ratelimit-reset"));
    const when = reset ? ` Try again after ${new Date(reset * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.` : " Try again in an hour.";
    return (
      "GitHub's hourly limit for this network has been used up (60 file lists per hour without signing in)." +
      when +
      " In the meantime, download the repository as a .zip on github.com and use “Import .zip”."
    );
  }
  if (resp.status === 404) {
    return (
      `GitHub has no public repository ${spec.owner}/${spec.repo}` +
      (spec.ref ? ` with a branch or tag “${spec.ref}”` : "") +
      ". Check the spelling. Private repositories can't be opened here: download them as a .zip and use “Import .zip”."
    );
  }
  if (resp.status === 409) return "This repository is empty.";
  return `GitHub answered with an error (HTTP ${resp.status}). Please try again later.`;
}

async function pool(items, limit, worker) {
  let next = 0;
  const run = async () => {
    while (next < items.length) {
      const i = next++;
      await worker(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, run));
}

/**
 * Download the files of a repository (or one of its folders).
 * Returns { files: [{ path, content }], skipped: [{ path, reason }], total, ref }.
 * onProgress({ phase: "list" | "download", done, total })
 */
export async function fetchGithub(spec, { onProgress = () => {}, signal } = {}) {
  const ref = spec.ref || "HEAD";
  onProgress({ phase: "list", done: 0, total: 0 });
  let resp;
  try {
    resp = await fetch(`https://api.github.com/repos/${spec.owner}/${spec.repo}/git/trees/${encodeURIComponent(ref)}?recursive=1`, {
      headers: { Accept: "application/vnd.github+json" },
      signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new GithubError("Couldn't reach GitHub. Check your internet connection.");
  }
  if (!resp.ok) throw new GithubError(describeFailure(resp, spec));
  const tree = await resp.json();
  const prefix = spec.path ? spec.path + "/" : "";
  const skipped = [];
  const wanted = [];
  let total = 0;
  let other = 0;
  for (const item of tree.tree || []) {
    if (item.type !== "blob" || !item.path.startsWith(prefix)) continue;
    const rel = item.path.slice(prefix.length);
    if (rel.split("/").some((p) => p.startsWith("."))) continue;
    if (!GITHUB_EXT.test(rel)) {
      other++;
      continue;
    }
    if (item.size > GITHUB_MAX_FILE) {
      skipped.push({ path: rel, reason: "larger than 5 MB" });
    } else if (wanted.length >= GITHUB_MAX_FILES) {
      skipped.push({ path: rel, reason: `more than ${GITHUB_MAX_FILES} files` });
    } else if (total + (item.size || 0) > GITHUB_MAX_TOTAL) {
      skipped.push({ path: rel, reason: "over the 25 MB total" });
    } else {
      total += item.size || 0;
      wanted.push({ path: rel, full: item.path, size: item.size || 0 });
    }
  }
  if (!wanted.length) {
    throw new GithubError(
      spec.path && !(tree.tree || []).some((t) => t.path === spec.path || t.path.startsWith(prefix))
        ? `There is no folder “${spec.path}” in ${spec.owner}/${spec.repo}.`
        : "No .m or data files were found there (looked for .m, .csv, .txt, .json, .md, .png, .jpg and .mat)."
    );
  }
  const files = [];
  let done = 0;
  onProgress({ phase: "download", done, total: wanted.length });
  await pool(wanted, 6, async (item) => {
    const url = `https://raw.githubusercontent.com/${spec.owner}/${spec.repo}/${ref.split("/").map(encodeURIComponent).join("/")}/${item.full.split("/").map(encodeURIComponent).join("/")}`;
    try {
      const r = await fetch(url, { signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const bytes = new Uint8Array(await r.arrayBuffer());
      files.push({ path: item.path, content: toStored(item.path, bytes) });
    } catch (e) {
      if (e.name === "AbortError") throw e;
      skipped.push({ path: item.path, reason: "download failed" });
    }
    onProgress({ phase: "download", done: ++done, total: wanted.length });
  });
  files.sort((a, b) => a.path.localeCompare(b.path));
  return { files, skipped, other, truncated: !!tree.truncated, ref: spec.ref || "" };
}
