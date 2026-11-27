"""Subresource Integrity hashes for the files the web app loads from CDNs.

The browser refuses a file whose SHA-384 differs from the hash written next to its address, so a
compromised or changed CDN file cannot run in the app.

- web/index.html: every <script src> and <link rel=stylesheet href> on cdnjs.cloudflare.com or
  cdn.jsdelivr.net gets integrity="sha384-..." crossorigin="anonymous".
- web/worker.js: the Pyodide core files (pyodide.js, pyodide.asm.js, the WebAssembly binary,
  the standard library and the lock file) are listed with their hashes between the
  "BEGIN SRI" and "END SRI" markers. The lock file holds a SHA-256 for every package
  (NumPy, SciPy, ...), which Pyodide checks itself, so a verified lock file covers them too.

Usage:
    python scripts/sri.py           # download every file and write the hashes
    python scripts/sri.py --check   # download every file and fail if a hash is missing or differs

Run it after changing a CDN version in web/index.html or PYODIDE_VERSION in web/worker.js.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "web" / "index.html"
WORKER = ROOT / "web" / "worker.js"
CDN = ("https://cdnjs.cloudflare.com/", "https://cdn.jsdelivr.net/")
PYODIDE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]

TAG = re.compile(r"<(script|link)\b[^>]*>")
URL_ATTR = re.compile(r'\b(?:src|href)="([^"]+)"')
SRI_BLOCK = re.compile(r"(// BEGIN SRI[^\n]*\n)(.*?)(\s*// END SRI)", re.S)


def sri(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    return "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


def index_tags(html: str) -> list[tuple[str, str]]:
    """(tag, url) for every CDN script or stylesheet tag in index.html."""
    out = []
    for m in TAG.finditer(html):
        tag = m.group(0)
        if m.group(1) == "link" and 'rel="stylesheet"' not in tag:
            continue
        u = URL_ATTR.search(tag)
        if u and u.group(1).startswith(CDN):
            out.append((tag, u.group(1)))
    return out


def with_integrity(tag: str, value: str) -> str:
    tag = re.sub(r'\s+integrity="[^"]*"', "", tag)
    tag = re.sub(r'\s+crossorigin(="[^"]*")?', "", tag)
    return re.sub(r"\s*>$", f' integrity="{value}" crossorigin="anonymous">', tag)


def pyodide_base(js: str) -> str:
    version = re.search(r'const PYODIDE_VERSION = "([^"]+)"', js).group(1)
    return f"https://cdn.jsdelivr.net/pyodide/v{version}/full/"


def worker_hashes(js: str) -> dict:
    m = SRI_BLOCK.search(js)
    if m is None:
        raise SystemExit("web/worker.js has no '// BEGIN SRI' ... '// END SRI' block.")
    body = m.group(2).strip().rstrip(",")
    return json.loads("{" + body + "}") if body else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if a hash is missing or differs; change nothing")
    args = ap.parse_args()

    html, js = INDEX.read_text(encoding="utf-8"), WORKER.read_text(encoding="utf-8")
    problems: list[str] = []

    new_html = html
    for tag, url in index_tags(html):
        want = sri(url)
        have = re.search(r'integrity="([^"]*)"', tag)
        if args.check:
            if have is None or have.group(1) != want or 'crossorigin="anonymous"' not in tag:
                problems.append(f"web/index.html: {url} " + ("has no integrity hash" if have is None else
                                                             f"is {want}, not {have.group(1)}"))
        else:
            new_html = new_html.replace(tag, with_integrity(tag, want), 1)

    base = pyodide_base(js)
    wanted = {name: sri(base + name) for name in PYODIDE_FILES}
    have = worker_hashes(js)
    if args.check:
        for name, want in wanted.items():
            if have.get(name) != want:
                problems.append(f"web/worker.js: {base}{name} " + (f"is {want}, not {have[name]}" if name in have
                                                                   else "has no integrity hash"))
    else:
        lines = ",\n".join(f'  "{k}": "{v}"' for k, v in wanted.items())
        js = SRI_BLOCK.sub(lambda m: m.group(1) + lines + "\n  // END SRI", js, count=1)

    if args.check:
        for p in problems:
            print(p)
        n = len(index_tags(html)) + len(PYODIDE_FILES)
        print(f"{n - len(problems)} of {n} CDN files match their integrity hashes.")
        return 1 if problems else 0
    INDEX.write_text(new_html, encoding="utf-8")
    WORKER.write_text(js, encoding="utf-8")
    print(f"wrote hashes for {len(index_tags(html))} files in web/index.html and {len(PYODIDE_FILES)} in web/worker.js")
    return 0


if __name__ == "__main__":
    sys.exit(main())
