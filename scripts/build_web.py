#!/usr/bin/env python3
"""Build the browser app in ``web/``.

The web app runs the mylab engine inside Pyodide (Python compiled to
WebAssembly). This script builds a pure-Python wheel of mylab, copies it to
``web/wheels/`` and writes ``web/wheels/manifest.json`` so the Web Worker
knows which file to install.

Usage::

    python scripts/build_web.py            # build only
    python scripts/build_web.py --serve    # build, then serve web/ on localhost
    python scripts/build_web.py --serve --port 8123
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
WHEELS = WEB / "wheels"


def build_wheel() -> Path:
    """Build mylab as a wheel and copy it into web/wheels/. Returns the wheel path."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-q", "-w", tmp]
        print("Building wheel:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=tmp)
        built = sorted(Path(tmp).glob("mylab*.whl"))
        if not built:
            raise SystemExit("No mylab wheel was produced.")
        if WHEELS.exists():
            shutil.rmtree(WHEELS)
        WHEELS.mkdir(parents=True)
        dest = WHEELS / built[-1].name
        shutil.copy2(built[-1], dest)

    version = dest.name.split("-")[1]
    manifest = {"wheel": dest.name, "version": version}
    (WHEELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dest.relative_to(ROOT)} and {(WHEELS / 'manifest.json').relative_to(ROOT)}")
    return dest


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".whl": "application/zip",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".webmanifest": "application/manifest+json",
    }

    def end_headers(self) -> None:
        # Always serve fresh files while developing.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - signature from base class
        if not getattr(self.server, "quiet", False):
            super().log_message(format, *args)


def make_server(port: int = 8000, host: str = "127.0.0.1", quiet: bool = False) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(_Handler, directory=str(WEB))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    server.quiet = quiet  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="serve web/ on localhost after building")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="host for --serve (default 127.0.0.1)")
    parser.add_argument("--no-build", action="store_true", help="skip building the wheel (with --serve)")
    args = parser.parse_args(argv)

    if not args.no_build:
        build_wheel()
    if args.serve:
        server = make_server(args.port, args.host)
        print(f"Serving {WEB.relative_to(ROOT)}/ at http://{args.host}:{server.server_address[1]}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
