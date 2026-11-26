"""Speed regression check for the benchmark cases in benchmarks/octave_compare.

    python benchmarks/speed_check.py              # compare with benchmarks/speed_baseline.json
    python benchmarks/speed_check.py --update     # record a new baseline on this machine
    python benchmarks/speed_check.py --only b0    # some cases only

Times differ between machines, so every case is stored relative to a calibration workload timed on
the same machine in the same run: a pure-Python loop for cases bound by the interpreter, a NumPy
workload for cases bound by numerical libraries, and the same LAPACK/BLAS operations called from
NumPy for the dense linear-algebra cases. A case fails when its relative time is
more than --limit (default 25%) above the baseline and still is when timed again twice, each time
against a freshly measured calibration (shared machines have slow spells).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = os.path.join(HERE, "octave_compare")
BASELINE = os.path.join(HERE, "speed_baseline.json")
# cases whose time goes mostly to NumPy, SciPy, BLAS/LAPACK or file and image libraries
LIBRARY_BOUND = {"b20_vec_elementwise", "b21_matmul_1500", "b22_solve_1500", "b23_eig_500", "b24_svd_600",
                 "b25_fft_4m", "b26_sort_unique_5m", "b27_logical_index_1e7", "b28_sparse_tridiag_solve",
                 "b30_accumarray_interp", "b31_csv_write_read", "b32_plot_to_png"}


# dense linear algebra: timed against the same LAPACK/BLAS operations called from NumPy, so the number
# of cores and the BLAS library (which change these far more than anything else) cancel out
LINALG_BOUND = {"b21_matmul_1500", "b22_solve_1500", "b23_eig_500", "b24_svd_600"}


def calibrate() -> dict:
    """Best of five timings of three fixed workloads, in seconds."""
    import numpy as np

    def py():
        s = 0
        for i in range(2_000_000):
            s += i * i
        return s

    a = np.random.default_rng(0).random((500, 500))
    x = np.random.default_rng(1).random(1 << 20)

    def lib():
        for _ in range(4):
            a @ a
        np.fft.fft(x)
        np.sort(x)

    def linalg():  # the operations of b21-b24, done directly in NumPy
        a = np.ones((1500, 1500)) + np.eye(1500)
        a @ a.T.copy()  # two different arrays, as in the case: a full product (A @ A.T alone would use syrk)
        k = np.arange(1, 1500 ** 2 + 1, dtype=float)
        m = np.mod(k * 7919, 1009).reshape(1500, 1500, order="F") + 2000 * np.eye(1500)
        np.linalg.solve(m, np.arange(1.0, 1501.0))
        k = np.arange(1, 500 ** 2 + 1, dtype=float)
        np.linalg.eigvals(np.mod(k * 7919, 1009).reshape(500, 500, order="F"))
        k = np.arange(1, 600 ** 2 + 1, dtype=float)
        np.linalg.svd(np.mod(k * 7919, 1009).reshape(600, 600, order="F"), compute_uv=False)

    out = {}
    for name, fn in (("python", py), ("numpy", lib), ("linalg", linalg)):
        best = float("inf")
        for _ in range(15):  # the best of many: on shared machines single timings vary by +-30%
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        out[name] = best
    return out


def run_case(name: str, repeats: int = 5) -> float:
    """Best wall time of the case inside mylab (the case's own tic/toc), over ``repeats`` runs."""
    code = f"best = Inf; for rep = 1:{repeats}, t0 = tic; feval('{name}'); best = min(best, toc(t0)); end; " \
           "fprintf('TIME %.6f\\n', best);"
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-m", "mylab", "-e", code], cwd=CASES, env=env, capture_output=True,
                       text=True, timeout=900)
    m = re.search(r"TIME ([0-9.]+)", r.stdout)
    if not m:
        raise RuntimeError(f"{name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return float(m.group(1))


def startup(repeats: int = 5) -> float:
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", "mylab", "-e", "x = 1;"], env=env, check=True, capture_output=True)
        best = min(best, time.perf_counter() - t0)
    return best


def unit(name: str) -> str:
    if name in LINALG_BOUND:
        return "linalg"
    return "numpy" if name in LIBRARY_BOUND else "python"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="record a new baseline instead of checking")
    ap.add_argument("--limit", type=float, default=0.25, help="allowed slowdown (0.25 = 25%%)")
    ap.add_argument("--only", default=None, help="regex on case names")
    a = ap.parse_args()
    names = sorted(os.path.basename(p)[:-2] for p in glob.glob(os.path.join(CASES, "b*.m")))
    if a.only:
        names = [n for n in names if re.search(a.only, n)]
    cal = calibrate()
    print(f"calibration: python loop {cal['python']:.3f} s, numpy workload {cal['numpy']:.3f} s, "
          f"linear algebra {cal['linalg']:.3f} s", flush=True)
    now = {}
    for n in names:
        now[n] = run_case(n)
        print(f"{n:28s} {now[n]:8.3f} s  ({now[n] / cal[unit(n)]:.2f} units)", flush=True)
    again = calibrate()  # once more after the cases: the machine's best state of the two
    cal = {k: min(cal[k], again[k]) for k in cal}
    start = startup()
    print(f"{'startup':28s} {start:8.3f} s  ({start / cal['python']:.2f} units)")
    if a.update:
        data = {"note": "Times relative to the calibration workloads of benchmarks/speed_check.py, "
                        "recorded with --update.",
                "calibration": cal, "startup": start / cal["python"],
                "cases": {n: {"unit": unit(n), "relative": now[n] / cal[unit(n)], "seconds": now[n]} for n in names}}
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        print(f"baseline written: {BASELINE}")
        return
    with open(BASELINE, encoding="utf-8") as fh:
        base = json.load(fh)
    rows, failed = [], []
    for n in names:
        if n not in base["cases"]:
            rows.append((n, None, now[n] / cal[unit(n)], "new"))
            continue
        if base["cases"][n].get("unit") != unit(n):
            rows.append((n, None, now[n] / cal[unit(n)], "re-record the baseline (--update)"))
            failed.append(n)
            continue
        ref = base["cases"][n]["relative"]
        rel = now[n] / cal[unit(n)]
        for _ in range(2):  # a slowdown counts only if it is still there twice more, against a fresh yardstick
            if rel <= ref * (1 + a.limit):
                break
            fresh = calibrate()
            rel = min(rel, run_case(n, repeats=9) / min(cal[unit(n)], fresh[unit(n)]))
        verdict = "SLOWER" if rel > ref * (1 + a.limit) else "ok"
        if verdict != "ok":
            failed.append(n)
        rows.append((n, ref, rel, verdict))
    ref = base.get("startup")
    rel = start / cal["python"]
    if ref is not None and rel > ref * (1 + a.limit):
        rel = min(rel, startup(repeats=9) / cal["python"])
    if ref is not None:
        verdict = "SLOWER" if rel > ref * (1 + a.limit) else "ok"
        if verdict != "ok":
            failed.append("startup")
        rows.append(("startup", ref, rel, verdict))
    lines = ["| Case | Baseline | Now | Change | |", "|---|---|---|---|---|"]
    for n, ref, rel, verdict in rows:
        change = "" if ref is None else f"{100 * (rel / ref - 1):+.0f}%"
        lines.append(f"| {n} | {'' if ref is None else f'{ref:.2f}'} | {rel:.2f} | {change} | {verdict} |")
    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Speed against the stored baseline\n\nRelative times (lower is faster).\n\n" + report + "\n")
    if failed:
        print(f"\n{len(failed)} case(s) more than {100 * a.limit:.0f}% slower than the baseline: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(rows)} cases within {100 * a.limit:.0f}% of the baseline.")


if __name__ == "__main__":
    main()
