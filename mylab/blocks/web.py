"""Entry points for the diagram editor of the web app.

The page sends a model as JSON text and gets plain data back, so nothing
here depends on the browser. ``handle`` is called from the engine worker.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np

from .engine import Sim, SimError, compile_model, run_callbacks
from .blocks_basic import Display, Scope, XYGraph
from .model import Model, ModelError

MAX_POINTS = 200_000  # per scope signal sent to the page


def _model(args: dict) -> Model:
    data = args.get("model")
    if isinstance(data, str):
        data = json.loads(data)
    return Model.from_json(data, args.get("name"))


def _clean(a: np.ndarray) -> list:
    """Numbers as JSON: NaN and infinities become null."""
    a = np.asarray(a, dtype=float)
    if np.all(np.isfinite(a)):
        return np.round(a, 12).tolist()
    return (
        [None if not np.isfinite(v) else float(v) for v in a.ravel()]
        if a.ndim == 1
        else [[None if not np.isfinite(v) else float(v) for v in row] for row in a]
    )


def check(session: Any, args: dict) -> dict:
    """Problems that can be found without running: for the markers on the canvas."""
    try:
        model = _model(args)
        comp = compile_model(model, session.interp, None, strict=False)
        return {"diagnostics": comp.diagnostics}
    except (ModelError, SimError, ValueError) as exc:
        return {
            "diagnostics": [
                {"level": "error", "message": str(getattr(exc, "message", exc)), "path": None, "kind": "model"}
            ]
        }


def _scope_chunk(nodes: list, sent: dict) -> list:
    out = []
    for n in nodes:
        k0 = sent.get(id(n), 0)
        k1 = len(n.log_t)
        if k1 <= k0:
            continue
        sent[id(n)] = k1
        item: dict = {"path": n.path, "t": _clean(np.array(n.log_t[k0:k1]))}
        sigs = []
        for col in n.log_u:
            rows = col[k0:k1]
            width = max(r.size for r in rows)
            sigs.append(_clean(np.array([r if r.size == width else np.resize(r, width) for r in rows])))
        item["signals"] = sigs
        out.append(item)
    return out


def run(session: Any, args: dict, emit: Callable | None = None) -> dict:
    """Simulate a model. Scope data is streamed through ``emit`` while the run goes on."""
    interp = session.interp
    out: dict = {"ok": False, "error": None, "diagnostics": [], "warnings": []}
    try:
        model = _model(args)
        notes = run_callbacks(model, interp)  # the model's own set-up code (InitFcn and the like)
        comp = compile_model(model, interp, None, strict=False, config=args.get("config"))
        for note in notes:
            comp.problem("warning", note, None, kind="callback")
        out["diagnostics"] = comp.diagnostics
        errors = comp.errors()
        if errors:
            first = errors[0]
            message = first["message"] + (" Earlier: " + notes[0] if notes else "")
            out["error"] = {"message": message, "path": first.get("path"), "paths": first.get("paths") or []}
            return out
        scopes = [n for n in comp.order if isinstance(n, (Scope, XYGraph))]
        sent: dict = {}
        last = [time.time()]
        meta = [
            {
                "path": n.path,
                "kind": "xy" if isinstance(n, XYGraph) else "scope",
                "title": n.text("Title") if isinstance(n, Scope) else "",
                "names": list(getattr(n, "in_names", [])),
                "stairs": isinstance(n.rate, tuple),
            }
            for n in scopes
        ]

        def tick(sim: Sim, t: float) -> None:
            now = time.time()
            if emit is not None and now - last[0] >= 0.08:
                last[0] = now
                emit(
                    json.dumps(
                        {"kind": "progress", "t": t, "t0": sim.t0, "tf": sim.tf, "scopes": _scope_chunk(scopes, sent)}
                    )
                )

        sim = Sim(comp, args.get("config"), tick)
        if emit is not None:
            emit(json.dumps({"kind": "start", "t0": sim.t0, "tf": sim.tf, "scopes": meta}))
        sim.run()
        res = sim.result()
        out["scopes"] = _scope_chunk(scopes, sent)
        out["meta"] = meta
        out["displays"] = {
            n.path: _clean(np.ravel(getattr(n, "last", np.zeros(0)))) for n in comp.order if isinstance(n, Display)
        }
        out["warnings"] = list(dict.fromkeys(res.warnings))
        out["steps"] = res.steps
        out["t_end"] = float(res.t[-1]) if res.t.size else 0.0
        out["stopped_by"] = res.stopped_by
        out["solver"] = res.solver
        # the results also go to the workspace, so ordinary plotting code works on them
        from ..builtins.blockdiagram import result_struct

        struct, variables = result_struct(res)
        base = interp.base
        base.set("out", struct)
        base.set("tout", res.t.reshape(-1, 1))
        base.set("yout", res.y)
        for name, v in variables.items():
            base.set(name, v)
        out["assigned"] = ["out", "tout", "yout", *variables]
        out["ok"] = True
    except SimError as exc:
        out["error"] = {"message": exc.message, "path": exc.path, "paths": exc.paths}
        out["diagnostics"] = out["diagnostics"] or exc.diagnostics
    except ModelError as exc:
        out["error"] = {"message": str(exc), "path": None, "paths": []}
    return out


def import_slx(session: Any, args: dict) -> dict:
    """Convert a .slx file of the project into model JSON text."""
    from .slx import import_slx as read

    path = str(args.get("path", ""))
    root = os.path.realpath(str(args.get("root") or os.getcwd()))
    full = os.path.realpath(os.path.join(root, path))
    if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
        return {"error": f"There is no file {path} in this project."}
    try:
        model = read(full)
    except ModelError as exc:
        return {"error": str(exc)}
    rep = getattr(model, "import_report", {})
    return {
        "model": model.dumps(),
        "name": model.name,
        "notes": model.notes,
        "blocks": rep.get("blocks", 0),
        "unsupported": rep.get("unsupported", {}),
    }


def validate(session: Any, args: dict) -> dict:
    """Check parameter expressions typed into a dialog: {name: error text or None}."""
    from .engine import Context

    ctx = Context(session.interp)
    result = {}
    for name, expr in (args.get("values") or {}).items():
        try:
            if str(expr).strip() == "":
                raise ValueError("a value is needed")
            ctx.evaluate(str(expr))
            result[name] = None
        except Exception as exc:
            result[name] = str(getattr(exc, "message", None) or exc)
    return {"errors": result}


OPS = {"check": check, "import_slx": import_slx, "validate": validate}


def handle(session: Any, op: str, args_json: str, emit: Callable | None = None) -> str:
    args = json.loads(args_json or "{}")
    session.interp.state["blocks_ui"] = True
    if op == "run":
        return json.dumps(run(session, args, emit))
    fn = OPS.get(op)
    if fn is None:
        return json.dumps({"error": f"unknown request '{op}'"})
    return json.dumps(fn(session, args))
