"""App windows outside the web app: the local server behind `mylab app.m`, and static snapshots."""

import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request

import pytest

from mylab.api import Session
from mylab.uiweb import asset, outline, static_html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP = """
    fig = uifigure('Name', 'Served');
    lbl = uilabel(fig, 'Text', '0', 'Tag', 'lbl');
    fig.UserData = 0;
    uibutton(fig, 'Text', 'Add', 'Tag', 'add', 'ButtonPushedFcn', @(s, e) bump(fig, lbl));
    function bump(fig, lbl)
        fig.UserData = fig.UserData + 1;
        lbl.Text = num2str(fig.UserData);
        disp('bumped');
    end
"""


def start(tmp_path, code):
    path = tmp_path / "app.m"
    path.write_text(textwrap.dedent(code))
    env = dict(os.environ, PYTHONPATH=ROOT, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-m", "mylab", "--no-browser", str(path)], cwd=tmp_path, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc


def read_url(proc) -> tuple[str, str]:
    for _ in range(50):
        line = proc.stdout.readline()
        m = re.search(r"App window: (http://127\.0\.0\.1:\d+)/\?t=(\S+)", line)
        if m:
            return m.group(1), m.group(2)
        if not line and proc.poll() is not None:
            break
    raise AssertionError("the address was not printed: " + proc.stderr.read())


def call(base, token, path, body=None, headers=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-Mylab-Token": token, **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as res:
        data = res.read().decode()
        return json.loads(data) if res.headers.get_content_type() == "application/json" else data


def find(tree, tag):
    def walk(n):
        if n["props"].get("tag") == tag:
            return n
        for c in n["children"]:
            r = walk(c)
            if r:
                return r
        return None
    return next(filter(None, (walk(f) for f in tree["figures"])))


def test_local_server_runs_an_app_until_the_window_closes(tmp_path):
    proc = start(tmp_path, APP)
    try:
        base, token = read_url(proc)
        page = call(base, token, "/?t=" + token)
        assert "MylabUI.mount" in page and "<title>mylab app</title>" in page
        assert "mui-root" in call(base, token, "/uirender.css")
        tree = call(base, token, "/state")["tree"]
        assert tree["figures"][0]["props"]["name"] == "Served"
        add, lbl = find(tree, "add"), find(tree, "lbl")
        r = call(base, token, "/event", {"id": add["id"], "event": "push"})
        assert r["error"] is None and r["output"] == "bumped\n"
        assert r["update"]["patch"] == {str(lbl["id"]): {"text": "1"}}
        r = call(base, token, "/event", {"id": add["id"], "event": "push"})
        assert r["update"]["patch"][str(lbl["id"])] == {"text": "2"}
        assert call(base, token, "/update")["update"]["patch"] == {}

        # no token, a wrong token or another Host: refused
        for bad in ({"X-Mylab-Token": "nope"}, {"Host": "evil.example"}):
            with pytest.raises(urllib.error.HTTPError) as e:
                call(base, token, "/state", headers=bad)
            assert e.value.code == 403

        r = call(base, token, "/event", {"id": tree["figures"][0]["id"], "event": "close"})
        assert r["update"]["figures"] == []
        assert proc.wait(timeout=20) == 0
        assert "bumped" in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_uiwait_blocks_the_script_until_uiresume(tmp_path):
    proc = start(tmp_path, """
        f = figure('Name', 'Ask');
        uicontrol('Style', 'edit', 'Tag', 'name', 'String', '');
        uicontrol('Style', 'pushbutton', 'Tag', 'ok', 'String', 'OK', 'Callback', 'uiresume(gcbf)');
        uiwait(f);
        h = findobj(f, 'Tag', 'name');
        fprintf('Hello %s\\n', get(h, 'String'));
        close(f);
    """)
    try:
        base, token = read_url(proc)
        tree = call(base, token, "/state")["tree"]
        call(base, token, "/event", {"id": find(tree, "name")["id"], "event": "value", "value": "Ada"})
        call(base, token, "/event", {"id": find(tree, "ok")["id"], "event": "push"})
        assert proc.wait(timeout=20) == 0
        assert "Hello Ada" in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_scripts_without_windows_do_not_start_a_server(tmp_path):
    proc = start(tmp_path, "disp(1 + 1)")
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 0 and out.strip() == "2" and "App window" not in out


def test_static_snapshot_is_self_contained():
    r = Session().run("f = uifigure('Name', 'Snap </script>'); uibutton(f, 'Text', 'Hi', 'Tag', 'b'); "
                      "uislider(f, 'Value', 30); uitable(f, 'Data', magic(3));")
    page = static_html(r.ui)
    assert page.startswith("<!doctype html>") and "readOnly: true" in page
    assert asset("uirender.js")[:200] in page and ".mui-root" in page
    assert "src=\"http" not in page and "href=\"http" not in page  # nothing is loaded from elsewhere
    assert "Snap <\\/script>" in page  # the tree cannot end the script element
    text = outline(r.ui)
    assert text.splitlines()[0] == '- figure "Snap </script>"'
    assert '- button "Hi" (Tag=\'b\')' in text and "3 rows x 3 columns" in text


def test_web_copy_of_the_renderer_is_in_sync():
    for name in ("uirender.js", "uirender.css"):
        folder = "js" if name.endswith(".js") else "css"
        path = os.path.join(ROOT, "web", folder, name)
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == asset(name), f"web/{folder}/{name} differs: run python scripts/build_web.py"


def test_web_copies_of_the_example_apps_are_in_sync():
    import glob
    paths = sorted(glob.glob(os.path.join(ROOT, "examples", "apps", "*.m")))
    assert paths
    for path in paths:
        copy = os.path.join(ROOT, "web", "samples", "apps", os.path.basename(path))
        with open(path, encoding="utf-8") as a, open(copy, encoding="utf-8") as b:
            assert a.read() == b.read(), f"{copy} differs: run python scripts/build_web.py"


def test_mcp_and_jupyter_get_a_note_and_a_snapshot():
    from mylab.mcp_server import _format_result
    r = Session().run("f = uifigure('Name', 'Tool'); uibutton(f, 'Text', 'Go');")
    parts = _format_result(r, None)
    note = parts[-1]
    assert "app window" in note and '- button "Go"' in note and "cannot be operated" in note
    path = re.search(r"saved to (\S+\.html)", note).group(1)
    try:
        with open(path, encoding="utf-8") as fh:
            assert "MylabUI.mount" in fh.read()
    finally:
        os.remove(path)
    pytest.importorskip("ipykernel")
    from mylab.kernel import app_snapshot
    data = app_snapshot(r.ui)["data"]
    assert data["text/html"].startswith("<iframe") and "srcdoc=" in data["text/html"]
    assert "snapshot" in data["text/plain"]
