"""Showing app windows outside the browser app: a tiny local server and static snapshots.

``uirender.js`` and ``uirender.css`` in this folder draw the component tree of
:mod:`mylab.ui` as HTML controls. They are used three ways:

* the mylab web app loads its own copy (``web/js/uirender.js``, kept in sync by
  ``scripts/build_web.py``),
* :class:`AppServer` serves them from ``127.0.0.1`` for ``mylab app.m``,
* :func:`static_html` embeds them in one HTML file (Jupyter, MCP): a picture of
  the window whose controls do not run code.
"""

from __future__ import annotations

import html
import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import ui
from ..errors import MatlabError

_DIR = os.path.dirname(os.path.abspath(__file__))

_PAGE_CSS = """
:root { --panel: #ffffff; --panel-2: #f4f6f8; --text: #1c2730; --muted: #5d6b77; --border: #dde3e8;
  --border-strong: #c1cad2; --accent: #0f766e; --accent-text: #ffffff; --accent-soft: #d8f1ee; --focus: #14b8a6;
  --error: #c0262d; --warn: #9a6700; --page: #eef1f4; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --panel: #161d22; --panel-2: #1b2329; --text: #e3e9ed; --muted: #93a2ad; --border: #273239;
    --border-strong: #34424b; --accent: #2dd4bf; --accent-text: #04201d; --accent-soft: #133a37; --focus: #5eead4;
    --error: #ff8a8a; --warn: #facc15; --page: #0f1418; color-scheme: dark; }
}
html, body { margin: 0; }
body { background: var(--page); color: var(--text); font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto,
  sans-serif; }
main { padding: 16px; max-width: 100%; }
.note { margin: 0 0 12px; color: var(--muted); font-size: 13px; }
.log { margin: 12px 0 0; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px;
  background: var(--panel); font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap;
  max-height: 30vh; overflow: auto; }
.log:empty { display: none; }
.log .err { color: var(--error); }
.stopped { padding: 10px 12px; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--panel); }
"""

_LIVE_JS = """
const token = TOKEN;
const main = document.getElementById("app");
const log = document.getElementById("log");
let stopped = false;
function write(text, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = text.endsWith("\\n") ? text : text + "\\n";
  log.append(span);
  log.scrollTop = log.scrollHeight;
}
async function call(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-Mylab-Token": token, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error("The app answered with an error (" + res.status + ").");
  return res.json();
}
function stop(text) {
  if (stopped) return;
  stopped = true;
  view.destroy();
  const box = document.createElement("p");
  box.className = "stopped";
  box.setAttribute("role", "status");
  box.textContent = text;
  main.prepend(box);
}
const view = MylabUI.mount(main, {
  patience: 400,
  send: (event) => call("/event", event),
  onOutput: (text) => write(text),
  onError: (text) => write("Error: " + text, "err"),
  onEmpty: () => stop("The app has closed. You can close this tab."),
});
call("/state").then((s) => {
  document.title = (s.tree.figures[0] && s.tree.figures[0].props.name) || "mylab app";
  view.setTree(s.tree);
}, (e) => stop(String(e.message || e)));
setInterval(() => {
  if (stopped) return;
  call("/update").then((r) => {
    if (r.output) write(r.output);
    view.apply(r.update);
  }, () => stop("The app has stopped (the mylab program ended)."));
}, 1000);
addEventListener("pagehide", () => navigator.sendBeacon("/bye?t=" + encodeURIComponent(token)));
"""


def asset(name: str) -> str:
    with open(os.path.join(_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _document(title: str, body: str, script: str) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_CSS}\n{asset('uirender.css')}</style>\n</head>\n"
        f"<body>\n<main>{body}</main>\n<script>{asset('uirender.js')}</script>\n<script>{script}</script>\n"
        "</body>\n</html>\n"
    )


def _json_for_script(value: Any) -> str:
    return json.dumps(value).replace("</", "<\\/")


def static_html(tree: dict, note: str | None = None) -> str:
    """One self-contained HTML page that shows the app windows. Its controls do not run code."""
    figures = tree.get("figures") or []
    title = (figures[0]["props"].get("name") if figures else "") or "mylab app"
    if note is None:
        note = ("Snapshot of an app window. The controls do not run code here: open the file in the mylab web "
                "app, or run it with the mylab command, to use the app.")
    body = f"<p class=\"note\">{html.escape(note)}</p><div id=\"app\"></div>"
    script = (f"MylabUI.mount(document.getElementById('app'), {{ readOnly: true }})"
              f".setTree({_json_for_script(tree)});")
    return _document(title, body, script)


def outline(tree: dict) -> str:
    """The windows as indented text (for tools that cannot show HTML)."""
    lines: list[str] = []

    def walk(node: dict, depth: int) -> None:
        p = node["props"]
        label = p.get("name") or p.get("title") or p.get("text") or p.get("alt") or ""
        extra = []
        if "value" in p and node["type"] not in ("axes",):
            extra.append(f"value={p['value']!r}")
        if "items" in p:
            sel = p.get("selected")
            picked = [p["items"][i] for i in (sel if isinstance(sel, list) else [sel])
                      if isinstance(i, int) and 0 <= i < len(p["items"])]
            extra.append(f"items={p['items']!r} selected={picked!r}")
        if node["type"] == "table":
            extra.append(f"{len(p.get('rows', []))} rows x {len(p.get('columns', []))} columns")
        if p.get("tag"):
            extra.append(f"Tag={p['tag']!r}")
        text = f"{'  ' * depth}- {node['type']}"
        if label:
            text += f" \"{label}\""
        if extra:
            text += " (" + ", ".join(extra) + ")"
        lines.append(text)
        for c in node["children"]:
            walk(c, depth + 1)

    for fig in tree.get("figures") or []:
        walk(fig, 0)
    for d in tree.get("dialogs") or []:
        lines.append(f"- dialog ({d['kind']}) \"{d.get('title', '')}\": {d.get('message', '')}")
    return "\n".join(lines)


class AppServer:
    """Serves the app windows of one interpreter on 127.0.0.1 until they are closed.

    Everything runs on the calling thread: :meth:`serve` handles one request at a
    time, so callbacks never run at the same time as other code of the session.
    """

    def __init__(self, interp, open_browser: bool = True, announce=None):
        self.interp = interp
        self.open_browser = open_browser
        self.announce = announce or (lambda text: print(text, flush=True))
        self.token = secrets.token_urlsafe(18)
        self.httpd: HTTPServer | None = None
        self.bye: float | None = None
        self.seen = False
        self._buffer: list[str] = []
        ui.state(interp).wait_hook = self.wait

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def url(self) -> str:
        assert self.httpd is not None
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/?t={self.token}"

    def start(self) -> None:
        if self.httpd is not None:
            return
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - name required by http.server
                server.handle(self, None)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                server.handle(self, self.rfile.read(length) if 0 < length < 5_000_000 else b"")

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.timeout = 0.2
        self.announce(f"App window: {self.url}")
        if self.open_browser:
            try:
                import webbrowser
                if not webbrowser.open(self.url):
                    self.announce("Open that address in a web browser to use the app.")
            except Exception:  # no browser on this machine
                self.announce("Open that address in a web browser to use the app.")

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.server_close()
            self.httpd = None
        st = self.interp.state.get("ui")
        if st is not None and st.wait_hook == self.wait:
            st.wait_hook = None

    def gone(self) -> bool:
        """True when the page was closed and did not come back (a reload returns within seconds)."""
        return self.bye is not None and time.monotonic() - self.bye > 3.0

    def serve(self) -> None:
        """Handle requests until every app window is closed (or the browser tab is)."""
        if not ui.has_ui(self.interp):
            return
        self.start()
        try:
            while ui.has_ui(self.interp) and not self.gone():
                self.httpd.handle_request()
        finally:
            self.stop()

    def wait(self, interp, fig, dialog) -> Any:
        """The hook behind uiwait (fig) and a blocking uiconfirm (dialog)."""
        st = ui.state(interp)
        self.start()
        if dialog is not None:
            while dialog["id"] not in st.answers and not self.gone() and self.httpd is not None:
                self.httpd.handle_request()
            return st.answers.pop(dialog["id"], None)
        while fig.uid in st.waiting and not fig.deleted and not self.gone() and self.httpd is not None:
            self.httpd.handle_request()
        return None

    # -- requests ----------------------------------------------------------------------------
    def handle(self, req: BaseHTTPRequestHandler, body: bytes | None) -> None:
        url = urlparse(req.path)
        host = (req.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return self.reply(req, 403, "text/plain", "Forbidden")
        if body is None and url.path in ("/uirender.js", "/uirender.css"):
            kind = "text/javascript" if url.path.endswith(".js") else "text/css"
            return self.reply(req, 200, kind, asset(url.path[1:]))
        token = req.headers.get("X-Mylab-Token") or (parse_qs(url.query).get("t") or [""])[0]
        if not secrets.compare_digest(token, self.token):
            return self.reply(req, 403, "text/plain", "This address needs the token that mylab printed.")
        self.bye = None
        self.seen = True
        if url.path == "/bye":
            self.bye = time.monotonic()
            return self.reply(req, 200, "application/json", "{}")
        if body is None and url.path == "/":
            page = ("<div id=\"app\"></div><div id=\"log\" class=\"log\" role=\"log\" aria-label=\"Output\"></div>")
            script = _LIVE_JS.replace("TOKEN", json.dumps(self.token))
            return self.reply(req, 200, "text/html", _document("mylab app", page, script))
        if body is None and url.path == "/state":
            tree = self.guard(lambda: ui.snapshot(self.interp)) or {"figures": [], "dialogs": []}
            return self.reply(req, 200, "application/json", json.dumps({"tree": tree}))
        if body is None and url.path == "/update":
            res = {"update": self.guard(lambda: ui.update(self.interp)) or {"full": False, "patch": {}}}
            return self.reply(req, 200, "application/json", json.dumps(res))
        if body is not None and url.path == "/event":
            try:
                event = json.loads(body.decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                return self.reply(req, 400, "text/plain", "Bad request")
            return self.reply(req, 200, "application/json", json.dumps(self.event(event)))
        return self.reply(req, 404, "text/plain", "Not found")

    def guard(self, fn):
        try:
            return fn()
        except MatlabError as e:
            self.interp.out(f"Error: {e.message}\n")
            return None

    def event(self, event: dict) -> dict:
        """Run one event; what the callbacks print goes to the terminal and to the page."""
        interp = self.interp
        original = interp.out
        printed: list[str] = []

        def tee(text: str) -> None:
            printed.append(text)
            original(text)

        interp.out = tee
        try:
            res = ui.dispatch(interp, event if isinstance(event, dict) else {})
        except Exception as e:  # an engine bug must not end the app
            res = {"error": f"Internal error: {type(e).__name__}: {e}", "update": {"full": False, "patch": {}}}
        finally:
            interp.out = original
        if res.get("error"):
            original(f"Error: {res['error']}\n")
        res["output"] = "".join(printed)
        return res

    @staticmethod
    def reply(req: BaseHTTPRequestHandler, status: int, kind: str, text: str) -> None:
        data = text.encode("utf-8")
        try:
            req.send_response(status)
            req.send_header("Content-Type", kind + "; charset=utf-8")
            req.send_header("Content-Length", str(len(data)))
            req.send_header("Cache-Control", "no-store")
            req.send_header("X-Content-Type-Options", "nosniff")
            req.end_headers()
            req.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
