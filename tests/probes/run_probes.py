"""Run the MATLAB-behaviour probe checks on a mylab checkout (or on GNU Octave).

    python tests/probes/run_probes.py                                    # all checks in checks/ on this checkout
    python tests/probes/run_probes.py --only '^(linalg|sparse)/'         # only ids matching a regex
    python tests/probes/run_probes.py --probes tests/probes/heldout      # the held-out set
    python tests/probes/run_probes.py --engine octave --only '^lang/'    # the same checks on GNU Octave

checks/ holds the checks used for the first comparison with GNU Octave; heldout/ holds a second
set written afterwards and kept out of sight while the gaps were being closed. baselines/ records
which checks passed on each engine at the time of the first comparison. The expected values come
from documented MATLAB behaviour; they have not been run on MATLAB itself.

Each check is a small script asserting documented MATLAB behaviour. It runs in its own
process and passes when it reaches the end without an error. Failing checks are listed with
the first error line and the first line of the check's code.
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKGS = ("statistics signal image optim control symbolic io struct splines geometry communications "
        "linear-algebra mapping financial general miscellaneous data-smoothing video audio parallel netcdf dicom")


def parse(path):
    checks, cur = [], None
    for raw in open(path, encoding="utf-8").read().split("\n"):
        if raw.startswith("### "):
            head = raw[4:].strip()
            per_line = head.endswith("*")
            head = head.rstrip("* ").strip()
            cat, name = head.split("/", 1)
            cur = {"cat": cat, "name": name, "per_line": per_line, "files": {}, "main": [], "target": "main"}
            checks.append(cur)
            continue
        if cur is None:
            continue
        if raw.startswith("@file "):
            cur["target"] = raw[6:].strip()
            cur["files"][cur["target"]] = []
            continue
        if raw.strip() == "@main":
            cur["target"] = "main"
            continue
        (cur["main"] if cur["target"] == "main" else cur["files"][cur["target"]]).append(raw)
    out = []
    for c in checks:
        files = {k: "\n".join(v).strip("\n") + "\n" for k, v in c["files"].items()}
        if c["per_line"]:
            k = 0
            for line in c["main"]:
                if not line.strip() or line.startswith("%%"):
                    continue
                k += 1
                out.append({"id": f"{c['cat']}/{c['name']}#{k:02d}", "cat": c["cat"], "group": c["name"],
                            "code": line, "files": files})
        else:
            out.append({"id": f"{c['cat']}/{c['name']}", "cat": c["cat"], "group": c["name"],
                        "code": "\n".join(c["main"]).strip("\n"), "files": files})
    return out


def run_one(engine, src, work, chk, octave_cmd="octave", pkgs=PKGS):
    d = os.path.join(work, re.sub(r"[^A-Za-z0-9_]+", "_", chk["id"]))
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for rel, body in chk["files"].items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(body)
    code = chk["code"]
    if "PROBE_OK" not in code:
        code += "\ndisp('PROBE_OK');\n"
    open(os.path.join(d, "probe_main.m"), "w", encoding="utf-8").write(code)
    env = dict(os.environ, DISPLAY=":99", MPLBACKEND="Agg")
    if engine == "mylab":
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "mylab", "probe_main.m"]
    else:
        cmd = [octave_cmd, *(["--no-gui"] if os.path.basename(octave_cmd) == "octave" else []),
               "--no-init-file", "--quiet", "--eval", f"warning('off','all'); pkg load {pkgs}; probe_main"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=d, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=180, errors="replace")
        out, err, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "TIMEOUT", -9
    ok = "PROBE_OK" in out
    msg = ""
    if not ok:
        lines = [ln for ln in (err + "\n" + out).split("\n") if ln.strip()
                 and not ln.startswith(("QStandardPaths", "Picked up", "warning:"))]
        errs = [ln for ln in lines if ln.lower().startswith("error")]
        msg = (errs[0] if errs else (lines[-1] if lines else f"rc={rc}"))[:300]
    return {"id": chk["id"], "cat": chk["cat"], "group": chk["group"], "engine": engine, "ok": ok, "msg": msg,
            "secs": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mylab", choices=["mylab", "octave"])
    ap.add_argument("--src", default=".", help="mylab checkout to test (folder containing the mylab package)")
    ap.add_argument("--only", default=None, help="regex on check ids, e.g. '^linalg/' or 'ode/solvers#1[45]'")
    ap.add_argument("--probes", default=os.path.join(HERE, "checks"), help="folder with the probe .txt files")
    ap.add_argument("--out", default=None, help="write full results as JSON")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--octave-cmd", default="octave", help="the Octave program (for --engine octave)")
    ap.add_argument("--octave-pkgs", default=PKGS, help="Octave packages to load before each check")
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    tag = a.engine + "-" + hashlib.md5((src + a.probes + a.octave_cmd).encode()).hexdigest()[:8]
    work = os.path.join(os.environ.get("PROBE_WORK", os.path.join(HERE, "work")), tag)
    checks = []
    for p in sorted(glob.glob(os.path.join(a.probes, "*.txt"))):
        checks += parse(p)
    if a.only:
        checks = [c for c in checks if re.search(a.only, c["id"])]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda c: run_one(a.engine, src, work, c, a.octave_cmd, a.octave_pkgs), checks))
    code = {c["id"]: c["code"] for c in checks}
    if not a.quiet:
        for r in res:
            if not r["ok"]:
                print(f"FAIL {r['id']} :: {r['msg'][:160]}\n       > {code[r['id']].splitlines()[0][:150]}")
    cats = {}
    for r in res:
        c = cats.setdefault(r["cat"], [0, 0])
        c[0] += 1
        c[1] += r["ok"]
    print("  ".join(f"{k} {v[1]}/{v[0]}" for k, v in cats.items()))
    print(f"{a.engine}: {sum(r['ok'] for r in res)} of {len(res)} checks pass")
    if a.out:
        json.dump({"checks": [{k: c[k] for k in ("id", "cat", "group", "code")} for c in checks], "results": res},
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
