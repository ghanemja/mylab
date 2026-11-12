"""Differential conformance runner for mylab.

Every file under cases/<area>/*.m is a small script. It is run inside
support/cf_run_case.m, which writes all workspace variables to JSON. GNU
Octave (an independent implementation of the same language) provides the
reference values; overrides.json corrects the places where Octave is known
to differ from MATLAB.

Examples:
    python tests/conformance/run.py --update-golden      # needs octave-cli
    python tests/conformance/run.py --compare            # mylab vs golden + overrides
    python tests/conformance/run.py --compare --filter strings --verbose
    python tests/conformance/run.py --engine octave --out /tmp/octave.json
    python tests/conformance/run.py --stats
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
SUPPORT = HERE / "support"
FUNCS = SUPPORT / "funcs"
GOLDEN = HERE / "golden.json"
OVERRIDES = HERE / "overrides.json"
KNOWN = HERE / "known_failures.json"

DEFAULT_TOL = 1e-10
SINGLE_TOL = 5e-6
INT_CLASSES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
CASE_TIMEOUT = 30
ERROR_KEY = "__error__"
FLAKY = "FLAKY"  # a known failure whose reason starts with this word may also pass


# ---------------------------------------------------------------- discovery

def discover(filter_text: str | None = None, include_matlab_only: bool = True) -> list[str]:
    ids = sorted(str(p.relative_to(CASES).with_suffix("")).replace(os.sep, "/") for p in CASES.rglob("*.m"))
    if filter_text:
        ids = [i for i in ids if filter_text in i]
    if not include_matlab_only:
        ids = [i for i in ids if not i.startswith("matlab_only/")]
    return ids


def case_path(case_id: str) -> Path:
    return CASES / (case_id + ".m")


def case_tolerance(case_id: str) -> float:
    m = re.search(r"^%\s*tol:\s*([0-9.eE+-]+)", case_path(case_id).read_text(), re.M)
    return float(m.group(1)) if m else DEFAULT_TOL


def _quote(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _load_result(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as e:
        return {"error": 2, "crash": f"no usable output ({type(e).__name__})", "vars": {}}


# ------------------------------------------------------------------ engines

def run_octave(case_ids: list[str], jobs: int = 4) -> dict[str, dict]:
    if shutil.which("octave-cli") is None:
        raise SystemExit("octave-cli not found")
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="cf_oct_") as td:
        tmp = Path(td)

        def batch(n: int) -> None:
            ids = case_ids[n::jobs]
            if not ids:
                return
            lines = [f"addpath({_quote(SUPPORT)}); addpath({_quote(FUNCS)}); warning('off', 'all');"]
            for i, cid in enumerate(ids):
                out = tmp / f"b{n}_{i}.json"
                lines.append(f"try, cf_run_case({_quote(case_path(cid))}, {_quote(out)}); catch, end")
            driver = tmp / f"driver{n}.m"
            driver.write_text("\n".join(lines) + "\n")
            work = tmp / f"work{n}"
            work.mkdir()
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run(["octave-cli", "--norc", "--quiet", "--no-window-system", str(driver)],
                               cwd=work, capture_output=True, text=True, timeout=CASE_TIMEOUT * len(ids))
            for i, cid in enumerate(ids):
                results[cid] = _load_result(tmp / f"b{n}_{i}.json")

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(batch, range(jobs)))
    return {cid: results[cid] for cid in case_ids}


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def run_mylab_case(case_id: str, workdir: Path) -> dict:
    """Run one case in-process on a fresh interpreter."""
    from mylab import Interpreter

    out = workdir / "out.json"
    if out.exists():
        out.unlink()
    code = (f"addpath({_quote(SUPPORT)}); addpath({_quote(FUNCS)}); warning('off', 'all');\n"
            f"cf_run_case({_quote(case_path(case_id))}, {_quote(out)});")
    use_alarm = hasattr(signal, "SIGALRM")
    cwd = os.getcwd()
    crash = None
    try:
        os.chdir(workdir)
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(CASE_TIMEOUT)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Interpreter().execute(code)
    except _Timeout:
        crash = "timeout"
    except BaseException as e:  # noqa: BLE001 - any escape from the interpreter is a finding
        if isinstance(e, KeyboardInterrupt):
            raise
        crash = f"{type(e).__name__}: {e}"[:300]
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        os.chdir(cwd)
    if crash is not None:
        return {"error": 2, "crash": crash, "vars": {}}
    return _load_result(out)


def run_mylab(case_ids: list[str]) -> dict[str, dict]:
    results = {}
    with tempfile.TemporaryDirectory(prefix="cf_my_") as td:
        for cid in case_ids:
            results[cid] = run_mylab_case(cid, Path(td))
    return results


# --------------------------------------------------------------- comparison

def _nums(text: str) -> list[float]:
    # Octave prints its "missing" value as NA; MATLAB has only NaN.
    return [float("nan") if t == "NA" else float(t) for t in text.split()]


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _text(rec: dict) -> str:
    return "".join(chr(int(t)) for t in rec.get("text", "").split())


def compare_value(exp: dict, got: dict, tol: float, path: str = "") -> tuple[str, str] | None:
    """Return None when equal, else (category, detail)."""
    if exp.get("class") != got.get("class"):
        return "class", f"{path}class {got.get('class')} (expected {exp.get('class')})"
    if exp.get("size") != got.get("size"):
        return "size", f"{path}size {got.get('size')} (expected {exp.get('size')})"
    cls = exp["class"]
    if "re" in exp:
        if ("im" in exp) != ("im" in got):
            # A complex value whose imaginary part is all zero equals the real value: the engines
            # (and MATLAB) differ on when such results are narrowed. Cases that care call isreal().
            side = exp if "im" in exp else got
            if any(x != 0 for x in _nums(side["im"])):
                return "complexity", f"{path}{'complex' if 'im' in got else 'real'} " \
                                     f"(expected {'complex' if 'im' in exp else 'real'})"
            exp = {k: v for k, v in exp.items() if k != "im"}
            got = {k: v for k, v in got.items() if k != "im"}
        for part in ("re", "im"):
            if part not in exp:
                continue
            if cls in INT_CLASSES or cls in ("char", "logical"):
                if exp[part].split() != got[part].split():
                    return "value", f"{path}{_show(got)} (expected {_show(exp)})"
                continue
            e, g = _nums(exp[part]), _nums(got[part])
            t = max(tol, SINGLE_TOL) if cls == "single" else tol
            for x, y in zip(e, g, strict=True):
                if not _close(x, y, t):
                    return "value", f"{path}{_show(got)} (expected {_show(exp)})"
            for x, y in zip(e, g, strict=True):
                if x == 0 and y == 0 and math.copysign(1, x) != math.copysign(1, y):
                    return "negative-zero", f"{path}{_show(got)} (expected {_show(exp)})"
        return None
    if cls == "function_handle":
        a, b = (re.sub(r"\s+", "", _text(r)).lstrip("@") for r in (exp, got))
        if a != b:
            return "text", f"{path}handle {_text(got)!r} (expected {_text(exp)!r})"
        return None
    if cls == "struct":
        if exp["fields"] != got["fields"]:
            return "fields", f"{path}fields {got['fields']} (expected {exp['fields']})"
        for k, (ei, gi) in enumerate(zip(exp["items"], got["items"], strict=True)):
            for name, ev, gv in zip(exp["fields"], ei, gi, strict=True):
                d = compare_value(ev, gv, tol, f"{path}({k + 1}).{name}: ")
                if d:
                    return d
        return None
    if cls == "containers.Map":
        if exp.get("count") != got.get("count"):
            return "value", f"{path}Count {got.get('count')} (expected {exp.get('count')})"
        return (compare_value(exp["keys"], got["keys"], tol, path + "keys: ")
                or compare_value(exp["values"], got["values"], tol, path + "values: "))
    if "items" in exp:
        if "items" not in got or len(exp["items"]) != len(got["items"]):
            return "value", f"{path}contents differ"
        for k, (ev, gv) in enumerate(zip(exp["items"], got["items"], strict=True)):
            d = compare_value(ev, gv, tol, f"{path}{{{k + 1}}}: ")
            if d:
                return d
    return None


def _show(rec: dict, limit: int = 70) -> str:
    cls = rec.get("class")
    if "re" not in rec:
        return f"<{cls}>"
    if cls == "char":
        s = repr("".join(chr(int(t)) for t in rec["re"].split()))
    elif "im" in rec:
        s = "[" + ", ".join(f"{a}{'+' if not b.startswith('-') else ''}{b}i"
                            for a, b in zip(rec["re"].split(), rec["im"].split(), strict=False)) + "]"
    else:
        s = "[" + " ".join(rec["re"].split()) + "]"
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def is_exception(rec: dict | None) -> bool:
    """Caught error objects are skipped: MATLAB gives an MException, Octave a struct with file paths."""
    if not rec:
        return False
    return rec.get("class") == "MException" or (
        rec.get("class") == "struct" and {"message", "identifier", "stack"} <= set(rec.get("fields", [])))


def expected_for(case_id: str, golden: dict) -> dict:
    """Golden record of a case with the MATLAB overrides applied."""
    base = golden["cases"].get(case_id, {"error": 0, "vars": {}})
    exp = {"error": 1 if base.get("error") else 0, "vars": dict(base.get("vars", {}))}
    for name, rec in golden.get("overrides", {}).get(case_id, {}).items():
        if name == ERROR_KEY:
            exp["error"] = 1 if rec else 0
        elif rec is None:
            exp["vars"].pop(name, None)
        else:
            exp["vars"][name] = rec
    return exp


def compare_case(case_id: str, got: dict, golden: dict) -> dict[str, tuple[str, str]]:
    """Return {variable: (category, detail)} for every mismatch in one case."""
    exp = expected_for(case_id, golden)
    tol = case_tolerance(case_id)
    diffs: dict[str, tuple[str, str]] = {}
    if got.get("crash"):
        diffs[ERROR_KEY] = ("crash", got["crash"])
        return diffs
    msg = "".join(chr(int(t)) for t in got.get("message", {}).get("re", "").split())
    if bool(got.get("error")) != bool(exp["error"]):
        diffs[ERROR_KEY] = ("error-state", f"stopped with error: {msg}" if got.get("error")
                            else "ran to the end (expected an error)")
    elif "internal error" in msg.lower():
        diffs[ERROR_KEY] = ("crash", msg)
    for name, erec in exp["vars"].items():
        grec = got["vars"].get(name)
        if is_exception(grec):
            continue
        if grec is None:
            diffs[name] = ("missing", "variable not set")
            continue
        d = compare_value(erec, grec, tol)
        if d:
            diffs[name] = d
    for name, grec in got["vars"].items():
        if name not in exp["vars"] and case_id in golden["cases"] and not is_exception(grec):
            diffs[name] = ("extra", "variable not in expectation")
    return diffs


def count_checks(golden: dict, case_ids: list[str]) -> int:
    return sum(len(expected_for(c, golden)["vars"]) + 1 for c in case_ids)


# ------------------------------------------------------------------- golden

def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def update_golden() -> int:
    ids = discover(None, include_matlab_only=False)
    results = run_octave(ids)
    cases = {}
    for cid, r in results.items():
        if r.get("crash"):
            print(f"WARNING {cid}: Octave produced no output ({r['crash']})")
        cases[cid] = {"error": 1 if r.get("error") else 0,
                      "vars": {k: v for k, v in r.get("vars", {}).items() if not is_exception(v)}}
        if r.get("error"):
            cases[cid]["message"] = "".join(chr(int(t)) for t in r.get("message", {}).get("re", "").split())
    overrides = load_json(OVERRIDES)
    resolved: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="cf_ovr_") as td:
        exprs = []
        for cid, entries in overrides.items():
            for name, spec in entries.items():
                if name == ERROR_KEY:
                    resolved.setdefault(cid, {})[name] = bool(spec["value"])
                elif spec.get("skip"):
                    resolved.setdefault(cid, {})[name] = None
                else:
                    exprs.append((cid, name, spec["matlab"]))
        # Evaluate each MATLAB-correct literal expression with the same serializer.
        lines = [f"addpath({_quote(SUPPORT)}); addpath({_quote(FUNCS)}); warning('off', 'all');"]
        for i, (_cid, _name, expr) in enumerate(exprs):
            f = Path(td) / f"e{i}.m"
            f.write_text(f"v = {expr};\n")
            lines.append(f"try, cf_run_case({_quote(f)}, {_quote(Path(td) / f'e{i}.json')}); catch, end")
        (Path(td) / "driver.m").write_text("\n".join(lines) + "\n")
        subprocess.run(["octave-cli", "--norc", "--quiet", "--no-window-system", str(Path(td) / "driver.m")],
                       cwd=td, capture_output=True, text=True, timeout=600)
        for i, (cid, name, expr) in enumerate(exprs):
            r = _load_result(Path(td) / f"e{i}.json")
            if r.get("error") or "v" not in r.get("vars", {}):
                print(f"WARNING override {cid}:{name} could not be evaluated: {expr}")
                continue
            if cid not in cases and not cid.startswith("matlab_only/"):
                print(f"WARNING override for unknown case {cid}")
            resolved.setdefault(cid, {})[name] = r["vars"]["v"]
    golden = {"generator": "GNU Octave (black box) + overrides.json", "cases": cases, "overrides": resolved}
    GOLDEN.write_text(json.dumps(golden, indent=0, sort_keys=True, separators=(",", ":")) + "\n")
    nvars = sum(len(c["vars"]) for c in cases.values())
    print(f"golden.json: {len(cases)} cases, {nvars} variables, "
          f"{sum(len(v) for v in resolved.values())} overrides")
    return 0


# ------------------------------------------------------------------ reports

def report(case_ids: list[str], results: dict[str, dict], golden: dict, verbose: bool) -> int:
    known = load_json(KNOWN)
    by_cat: dict[str, list[str]] = {}
    total = failed = known_hit = 0
    stale = []
    for cid in case_ids:
        exp = expected_for(cid, golden)
        diffs = compare_case(cid, results[cid], golden)
        total += len(exp["vars"]) + 1
        failed += len(diffs)
        for name, (cat, detail) in sorted(diffs.items()):
            tag = "known" if name in known.get(cid, {}) else "NEW"
            known_hit += tag == "known"
            by_cat.setdefault(cat, []).append(f"[{tag}] {cid} :: {name}: {detail}")
        stale += [f"{cid} :: {n}" for n, why in known.get(cid, {}).items()
                  if n not in diffs and not why.startswith(FLAKY)]
    for cat in sorted(by_cat):
        print(f"\n== {cat} ({len(by_cat[cat])}) ==")
        for line in by_cat[cat]:
            if verbose or "[NEW]" in line:
                print("  " + line)
    if stale:
        print(f"\n== known failures that now pass ({len(stale)}) ==")
        for s in stale:
            print("  " + s)
    print(f"\n{len(case_ids)} cases, {total} checks, {total - failed} pass "
          f"({100 * (total - failed) / max(total, 1):.2f}%), {failed} differ "
          f"({known_hit} known, {failed - known_hit} new)")
    return 1 if failed - known_hit or stale else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["octave", "mylab"], help="run all cases on one engine")
    ap.add_argument("--out", type=Path, help="with --engine: write raw results JSON here")
    ap.add_argument("--update-golden", action="store_true", help="regenerate golden.json from Octave")
    ap.add_argument("--compare", action="store_true", help="run mylab and compare with golden + overrides")
    ap.add_argument("--write-known", action="store_true",
                    help="with --compare: rewrite known_failures.json for the selected cases from the "
                         "current mismatches (existing reasons are kept, new ones get a TODO reason)")
    ap.add_argument("--stats", action="store_true", help="print suite size")
    ap.add_argument("--filter", help="only cases whose id contains this text")
    ap.add_argument("--verbose", action="store_true", help="also list known failures")
    args = ap.parse_args(argv)

    if args.update_golden:
        return update_golden()
    ids = discover(args.filter)
    if args.stats:
        golden = load_json(GOLDEN)
        areas: dict[str, int] = {}
        for c in ids:
            areas[c.split("/")[0]] = areas.get(c.split("/")[0], 0) + 1
        for a, n in sorted(areas.items()):
            print(f"{a:16s} {n}")
        print(f"{len(ids)} cases, {count_checks(golden, ids)} checks "
              f"({count_checks(golden, ids) - len(ids)} variables + {len(ids)} error states)")
        return 0
    if args.engine:
        if args.engine == "octave":
            ids = [i for i in ids if not i.startswith("matlab_only/")]
        results = run_octave(ids) if args.engine == "octave" else run_mylab(ids)
        text = json.dumps(results, indent=0, sort_keys=True)
        if args.out:
            args.out.write_text(text)
        else:
            sys.stdout.write(text + "\n")
        return 0
    if args.compare:
        golden = load_json(GOLDEN)
        results = run_mylab(ids)
        if args.write_known:
            known = load_json(KNOWN)
            for cid in ids:
                old = known.pop(cid, {})
                diffs = compare_case(cid, results[cid], golden)
                entries = {name: why for name, why in old.items() if why.startswith(FLAKY)}
                entries.update({name: old.get(name, f"TODO {cat}: {detail}"[:160])
                                for name, (cat, detail) in sorted(diffs.items())})
                if entries:
                    known[cid] = entries
            KNOWN.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
        return report(ids, results, golden, args.verbose)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
