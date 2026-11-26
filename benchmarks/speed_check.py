"""Speed regression check for the benchmark cases in benchmarks/octave_compare.

    python benchmarks/speed_check.py              # compare with benchmarks/speed_baseline.json
    python benchmarks/speed_check.py --update     # record a new baseline on this machine
    python benchmarks/speed_check.py --only b0    # some cases only

Times differ between machines, so every case is stored relative to a calibration workload timed on
the same machine in the same run: a pure-Python loop for cases bound by the interpreter, and for
each case bound by numerical libraries the same work done directly with NumPy/SciPy/Matplotlib (so
the ratio is what mylab adds, whatever the hardware). A case fails when its relative time is
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


def _equivalents() -> dict:
    """For each library-bound case, the same work done directly with NumPy, SciPy and Matplotlib.

    A case's time divided by its equivalent's is what mylab adds on top of the libraries it calls; that
    ratio hardly depends on the machine, unlike one shared calibration (GitHub's runners are 25-80%
    faster than a developer machine on FFT and indexing but slower on matrix products and sorting).
    """
    import tempfile

    import numpy as np

    def mod_seq(n, m):
        return np.mod(np.arange(1, n + 1, dtype=float) * 7919, m)

    def square(n):
        return mod_seq(n * n, 1009).reshape(n, n, order="F")

    def b20():
        x = np.linspace(0, 1, 10 ** 7)
        return float((np.sin(x) ** 2 + np.cos(x) ** 2 + x * x - 2 * x).sum())

    def b21():
        a = np.ones((1500, 1500)) + np.eye(1500)
        return float((a @ a.T.copy())[0, 0])

    def b22():
        return float(np.linalg.solve(square(1500) + 2000 * np.eye(1500), np.arange(1.0, 1501.0))[0])

    def b23():
        return float(abs(np.linalg.eigvals(square(500))[0]))

    def b24():
        return float(np.linalg.svd(square(600), compute_uv=False)[0])

    def b25():
        return float(abs(np.fft.fft(mod_seq(2 ** 22, 1009))[1]))

    def b26():
        x = mod_seq(5 * 10 ** 6, 1000003)
        return float(np.sort(x)[0] + np.unique(np.mod(x, 1000)).size)

    def b27():
        x = mod_seq(10 ** 7, 1009)
        y = x[x > 500]
        return float(y.size + np.flatnonzero(x == 7).size + (np.cumsum(y) > 0).sum())

    def b28():
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        e = np.ones(200000)
        a = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], format="csc")
        return float(spl.spsolve(a, e)[0])

    def b30():
        idx = mod_seq(2 * 10 ** 6, 1000).astype(np.int64)
        a = np.bincount(idx, minlength=1000).astype(float)
        return float(np.interp(np.linspace(1, 1000, 2 * 10 ** 6), np.arange(1.0, 1001.0), a).sum())

    def b31():
        m = mod_seq(300000, 1009).reshape(100000, 3, order="F") / 7
        fd, fn = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            np.savetxt(fn, m, fmt="%.10g", delimiter=",")
            return float(np.loadtxt(fn, delimiter=",")[-1, -1])
        finally:
            os.remove(fn)

    def b32():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        ax = fig.add_subplot()
        x = np.linspace(0, 10, 2000)
        for k in range(1, 11):
            ax.plot(x, np.sin(k * x))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("benchmark")
        ax.legend(["a"])
        fd, fn = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            fig.savefig(fn)
            return float(os.path.getsize(fn))
        finally:
            plt.close(fig)
            os.remove(fn)

    fns = {"b20": b20, "b21": b21, "b22": b22, "b23": b23, "b24": b24, "b25": b25, "b26": b26, "b27": b27,
           "b28": b28, "b30": b30, "b31": b31, "b32": b32}
    return {n: fns[n.split("_")[0]] for n in LIBRARY_BOUND}


_EQUIV: dict = {}


def calibrate(units=None) -> dict:
    """Best of several timings of the calibration workloads (or of the ``units`` named), in seconds:
    a pure-Python loop for the interpreter-bound cases, and each library-bound case's own equivalent."""
    if not _EQUIV:
        _EQUIV.update(_equivalents())

    def py():
        s = 0
        for i in range(2_000_000):
            s += i * i
        return s

    out = {}
    for name, fn in {"python": py, **_EQUIV}.items():
        if units is not None and name not in units:
            continue
        best = float("inf")
        for _ in range(15 if name == "python" else 5):  # the best of many: single timings vary by +-30%
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
    """What a case's time is divided by: its own NumPy equivalent, or the Python loop."""
    return name if name in LIBRARY_BOUND else "python"


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
    print(f"calibration: python loop {cal['python']:.3f} s; library cases against their NumPy equivalents",
          flush=True)
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
            fresh = calibrate({unit(n)})
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
