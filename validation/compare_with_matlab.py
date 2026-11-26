"""Compare mylab with the results of the validation kit run in MATLAB.

    python validation/compare_with_matlab.py mylab_validation_results.json
    python validation/compare_with_matlab.py results.json --report validation/report.md --jobs 3

Runs the same checks and expressions on this mylab checkout (several minutes), then writes a report
that lists every place where mylab disagrees with MATLAB and every check whose expectation MATLAB
itself does not meet (those checks are wrong about MATLAB and must be corrected, with the reason
recorded in tests/probes/heldout/CORRECTIONS.md). A JSON summary is written next to the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "validation", "matlab")
RUNNER = os.path.join(ROOT, "tests", "probes", "run_probes.py")
SETS = {"original": os.path.join(ROOT, "tests", "probes", "checks"),
        "heldout": os.path.join(ROOT, "tests", "probes", "heldout")}
MISSING_HINTS = ("Undefined function", "Unrecognized function or variable", "requires", "license", "Toolbox")


def run_mylab_checks(jobs: int, only: str | None) -> dict:
    out = {}
    for set_name, folder in SETS.items():
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        extra = ["--only", only] if only else []
        subprocess.run([sys.executable, RUNNER, "--src", ROOT, "--probes", folder, "--jobs", str(jobs),
                        "--quiet", "--out", path, *extra], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(path, encoding="utf-8") as fh:
            for r in json.load(fh)["results"]:
                out[(set_name, r["id"])] = r
        os.remove(path)
    return out


def run_mylab_exprs() -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    kit = KIT.replace("\\", "/")
    code = (f"addpath('{kit}'); ex = jsondecode(fileread('{kit}/exprs.json')); out = cell(1, numel(ex)); "
            f"for k = 1:numel(ex), out{{k}} = mlv_eval(ex{{k}}); end; "
            f"fid = fopen('{path.replace(chr(92), '/')}', 'w'); fwrite(fid, jsonencode(out)); fclose(fid);")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    with tempfile.TemporaryDirectory() as work:
        subprocess.run([sys.executable, "-m", "mylab", "-e", code], check=True, cwd=work, env=env,
                       stdout=subprocess.DEVNULL)
    with open(path, encoding="utf-8") as fh:
        vals = json.load(fh)
    os.remove(path)
    return vals


_NUM = re.compile(r"^(\w+)\[([\d ]*)\] (complex )?(.*)$")


def same_value(a: str, b: str, tol: float = 1e-10) -> bool:
    if a == b:
        return True
    if a.startswith("ERROR") and b.startswith("ERROR"):
        return True
    ma, mb = _NUM.match(a), _NUM.match(b)
    if not (ma and mb) or ma.group(1, 2, 3) != mb.group(1, 2, 3) or ma.group(1) in ("char", "cell", "string"):
        return False
    xa = ma.group(4).replace("|", " ").split()
    xb = mb.group(4).replace("|", " ").split()
    if len(xa) != len(xb):
        return False
    for p, q in zip(xa, xb):
        if p == q:
            continue
        try:
            u, v = float(p), float(q)
        except ValueError:
            return False
        if math.isnan(u) and math.isnan(v):
            continue
        if math.isinf(u) or math.isinf(v) or math.isnan(u) or math.isnan(v):
            return False
        if abs(u - v) > tol * max(1.0, abs(u), abs(v)):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="mylab_validation_results.json written by run_mylab_validation in MATLAB")
    ap.add_argument("--report", default=os.path.join(ROOT, "validation", "report.md"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default=None, help="regex on check ids, matching the 'Only' option of the kit run")
    a = ap.parse_args()
    with open(a.results, encoding="utf-8") as fh:
        ml = json.load(fh)
    with open(os.path.join(KIT, "checks.json"), encoding="utf-8") as fh:
        kit_checks = json.load(fh)
    code = {(c["set"], c["id"]): c["code"] for c in kit_checks}
    m_checks = ml.get("checks") or []
    if isinstance(m_checks, dict):
        m_checks = [m_checks]
    matlab = {(r["set"], r["id"]): r for r in m_checks}
    missing_in_run = [k for k in code if k not in matlab and not (a.only and not re.search(a.only, k[1]))]

    print("Running the checks on mylab...", flush=True)
    mylab = run_mylab_checks(a.jobs, a.only)
    print("Evaluating the expressions on mylab...", flush=True)
    my_vals = run_mylab_exprs()

    differs, wrong, unavailable, agree, octave_only = [], [], [], 0, []
    for key, r in matlab.items():
        if key not in mylab:
            continue
        mine = mylab[key]
        if key[1].startswith("octave_ext/"):
            octave_only.append(key)
            continue
        if r["ok"] and mine["ok"]:
            agree += 1
        elif r["ok"] and not mine["ok"]:
            differs.append((key, mine["msg"]))
        else:
            msg = r.get("message") or ""
            if r.get("identifier") == "MATLAB:UndefinedFunction" or any(h in msg for h in MISSING_HINTS):
                unavailable.append((key, msg, mine["ok"]))
            else:
                wrong.append((key, r.get("identifier") or "", msg, mine["ok"]))

    m_exprs = ml.get("exprs") or []
    if isinstance(m_exprs, dict):
        m_exprs = [m_exprs]
    expr_diffs = []
    for e in m_exprs:
        k = int(e["index"]) - 1
        if k < len(my_vals) and not same_value(e["value"], my_vals[k]):
            expr_diffs.append((e["expr"], e["value"], my_vals[k]))

    def block(key):
        return "```matlab\n" + code.get(key, "")[:1500] + "\n```"

    lines = [f"# mylab compared with MATLAB {ml.get('release', '?')}", "",
             f"MATLAB {ml.get('matlab_version', '?')} on {ml.get('computer', '?')}, run {ml.get('started', '?')}.", "",
             "| | Count |", "|---|---|",
             f"| Checks run in MATLAB | {len(matlab)} of {len(code)} |",
             f"| Same result in both | {agree} |",
             f"| MATLAB passes, mylab fails (mylab is wrong) | {len(differs)} |",
             f"| MATLAB fails the check (the check is wrong about MATLAB) | {len(wrong)} |",
             f"| Function not available in this MATLAB (toolbox or license) | {len(unavailable)} |",
             f"| Expressions with a different value | {len(expr_diffs)} of {len(m_exprs)} |", ""]
    if missing_in_run:
        lines += [f"{len(missing_in_run)} checks have no MATLAB result yet: run `run_mylab_validation` again "
                  "until it prints Finished.", ""]
    lines += ["## MATLAB passes, mylab fails", ""]
    for key, msg in differs:
        lines += [f"### {key[0]} `{key[1]}`", f"mylab: {msg}", "", block(key), ""]
    lines += ["## Checks MATLAB itself fails", "",
              "Each of these expects something MATLAB does not do. Correct the check (never to make an engine "
              "pass) and record the reason in tests/probes/heldout/CORRECTIONS.md. If mylab passes it, mylab "
              "matches the wrong expectation and needs fixing too.", ""]
    for key, ident, msg, mine_ok in wrong:
        lines += [f"### {key[0]} `{key[1]}`", f"MATLAB: {ident} {msg}".strip(),
                  f"mylab: {'passes (also needs fixing)' if mine_ok else 'fails'}", "", block(key), ""]
    lines += ["## Not available in this MATLAB", ""]
    for key, msg, _mine_ok in unavailable:
        lines.append(f"- {key[0]} `{key[1]}`: {msg}")
    lines += ["", "## Expressions with a different value", "", "| Expression | MATLAB | mylab |", "|---|---|---|"]
    for expr, mv, yv in expr_diffs:
        lines.append(f"| `{expr}` | `{mv[:120]}` | `{yv[:120]}` |")
    if octave_only:
        lines += ["", f"Not compared: {len(octave_only)} check(s) of Octave-only syntax, "
                  "which MATLAB rejects by design."]
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    summary = {"matlab": ml.get("release"), "checks_run": len(matlab), "agree": agree,
               "mylab_wrong": [list(k) for k, _ in differs], "check_wrong": [list(k) for k, *_ in wrong],
               "unavailable": [list(k) for k, *_ in unavailable], "expr_diffs": [e for e, _, _ in expr_diffs]}
    with open(os.path.splitext(a.report)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"Same result: {agree}. mylab wrong: {len(differs)}. Check wrong: {len(wrong)}. "
          f"Not available in MATLAB: {len(unavailable)}. Expression differences: {len(expr_diffs)}.")
    print(f"Report: {a.report}")


if __name__ == "__main__":
    main()
