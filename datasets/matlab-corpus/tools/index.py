"""Index the downloaded corpus: per-file metadata, language features and function usage.

    python tools/index.py [--cache DIR]

Writes:
  files.jsonl.gz  one record per .m file (repo, path, size, kind, features, calls, mylab parse status)
  stats.json      corpus-wide totals, feature frequencies and parse rate
  GAPS.md         functions the corpus calls that mylab does not provide, ranked by how many
                  repositories use them (a to-do list for compatibility work)

Only parses files; nothing in the corpus is executed.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from mylab import ast as A
from mylab.builtins import BUILTINS
from mylab.parser import parse

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 1_000_000

FEATURE_CALLS = {
    "tables": {"table", "readtable", "writetable", "array2table", "groupsummary", "innerjoin"},
    "containers_map": {"containers.Map"},
    "cellfun_arrayfun": {"cellfun", "arrayfun", "structfun"},
    "regexp": {"regexp", "regexprep", "regexpi"},
    "plotting": {"plot", "figure", "subplot", "scatter", "surf", "bar", "histogram", "imagesc", "mesh"},
    "file_io": {"fopen", "fprintf", "fgetl", "fread", "fwrite", "textscan", "load", "save", "readmatrix",
                "csvread", "dlmread", "xlsread", "importdata"},
    "ode_solvers": {"ode45", "ode23", "ode15s", "ode113"},
    "optimization": {"fminsearch", "fminunc", "fmincon", "fzero", "lsqnonlin", "linprog", "quadprog"},
    "signal": {"fft", "ifft", "filter", "filtfilt", "butter", "conv", "spectrogram", "pwelch"},
    "eval": {"eval", "evalin", "assignin", "feval"},
}


def walk(node, visit):
    visit(node)
    if isinstance(node, A.Node):
        for v in vars(node).values():
            walk(v, visit)
    elif isinstance(node, (list, tuple)):
        for item in node:
            walk(item, visit)
    elif isinstance(node, dict):
        for item in node.values():
            walk(item, visit)


def analyze(program: A.Program) -> tuple[str, dict, collections.Counter, set[str]]:
    features: collections.Counter = collections.Counter()
    defined: set[str] = set()
    assigned: set[str] = set()
    used: collections.Counter = collections.Counter()

    def visit(n):
        t = type(n)
        if t is A.Function:
            defined.add(n.name)
            assigned.update(n.params)
            assigned.update(n.outputs)
            if n.subfunctions:
                features["nested_functions"] += 1
        elif t is A.ClassDef:
            features["classdef"] += 1
            defined.add(n.name)
        elif t is A.Assign:
            for target in n.targets:
                root = target
                while not isinstance(root, A.Ident) and hasattr(root, "base"):
                    root = root.base
                if isinstance(root, A.Ident):
                    assigned.add(root.name)
        elif t is A.For:
            assigned.add(n.var.name)
            features["for"] += 1
        elif t is A.While:
            features["while"] += 1
        elif t is A.Switch:
            features["switch"] += 1
        elif t is A.Try:
            features["try_catch"] += 1
            if n.ident:
                assigned.add(n.ident)
        elif t is A.Global:
            assigned.update(n.names)
            features["persistent" if getattr(n, "persistent", False) else "global"] += 1
        elif t is A.Lambda:
            features["anonymous_functions"] += 1
            assigned.update(n.params)
        elif t is A.CellArray or t is A.CellIndex:
            features["cell_arrays"] += 1
        elif t is A.Field or t is A.DynField:
            features["struct_fields"] += 1
        elif t is A.String and n.double:
            features["string_literals"] += 1
        elif t is A.End:
            features["end_in_index"] += 1
        elif t is A.Index and isinstance(n.base, A.Ident):
            used[n.base.name] += 1
        elif t is A.Index and isinstance(n.base, A.Field) and isinstance(n.base.base, A.Ident):
            used[f"{n.base.base.name}.{n.base.name}"] += 1
        elif t is A.FuncHandle:
            used[n.name] += 1
            features["function_handles"] += 1

    walk(program, visit)
    if program.classdefs:
        kind = "classdef"
    elif program.functions and not program.body:
        kind = "function"
    else:
        kind = "script"
    calls = collections.Counter({k: v for k, v in used.items() if k not in assigned})
    for feat, names in FEATURE_CALLS.items():
        if any(name in calls for name in names):
            features[feat] += 1
    return kind, dict(features), calls, defined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "mylab-corpus")
    args = ap.parse_args()
    manifest = [json.loads(line) for line in (ROOT / "manifest.jsonl").read_text().splitlines()]

    records = []
    feature_files: collections.Counter = collections.Counter()
    missing_repos: dict[str, set[str]] = collections.defaultdict(set)
    missing_calls: collections.Counter = collections.Counter()
    totals = collections.Counter()
    error_kinds: collections.Counter = collections.Counter()

    for row in manifest:
        repo_dir = args.cache / row["repo"].replace("/", "__")
        if not repo_dir.is_dir():
            continue
        totals["repos"] += 1
        # Skip git internals and macOS "._" resource-fork files (binary junk that ends in .m).
        files = sorted(p for p in repo_dir.rglob("*.m")
                       if p.is_file() and ".git" not in p.parts and not p.name.startswith("._")
                       and "__MACOSX" not in p.parts)
        repo_names = {p.stem for p in files}  # functions and classes the repository defines itself
        parsed = []
        for f in files:
            rel = f.relative_to(repo_dir).as_posix()
            data = f.read_bytes()
            rec = {"repo": row["repo"], "commit": row["commit"], "license": row["license"], "path": rel,
                   "bytes": len(data), "lines": data.count(b"\n") + 1}
            totals["files"] += 1
            totals["lines"] += rec["lines"]
            if len(data) > MAX_BYTES:
                rec["mylab_parse"] = "skipped (too large)"
                records.append(rec)
                continue
            text = data.decode("utf-8", errors="replace")
            try:
                program = parse(text)
            except Exception as e:  # noqa: BLE001 - record any parser failure
                msg = str(e)
                rec["mylab_parse"] = msg[:200]
                error_kinds[msg.split(": ", 1)[-1][:60]] += 1
                records.append(rec)
                continue
            totals["parsed"] += 1
            rec["mylab_parse"] = "ok"
            kind, features, calls, defined = analyze(program)
            rec["kind"] = kind
            rec["features"] = features
            rec["calls"] = dict(calls.most_common(50))
            for feat in features:
                feature_files[feat] += 1
            parsed.append((rec, calls, defined))
            records.append(rec)
        defined_here = set(repo_names)
        for _, _, defined in parsed:
            defined_here |= defined
        for _rec, calls, _ in parsed:
            for name, count in calls.items():
                if name in BUILTINS or name in defined_here or name.split(".")[0] in defined_here:
                    continue
                missing_repos[name].add(row["repo"])
                missing_calls[name] += count

    with gzip.open(ROOT / "files.jsonl.gz", "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    licenses = collections.Counter(r["license"] for r in manifest)
    stats = {
        "repositories": totals["repos"],
        "files": totals["files"],
        "lines": totals["lines"],
        "mylab_parse_rate": round(totals["parsed"] / max(totals["files"], 1), 4),
        "licenses": dict(licenses.most_common()),
        "feature_file_counts": dict(feature_files.most_common()),
        "top_parse_errors": dict(error_kinds.most_common(15)),
    }
    (ROOT / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    ranked = sorted(missing_repos, key=lambda n: (-len(missing_repos[n]), -missing_calls[n], n))
    lines = ["# Functions the corpus uses that mylab does not provide", "",
             "Generated by `tools/index.py`. Ranked by the number of repositories that call each name.",
             "Some names are false positives: variables created by `load`/`eval`, functions from",
             "third-party toolboxes or MEX files, or code for other MATLAB-like languages.", "",
             "| function | repositories | calls |", "|---|---|---|"]
    lines += [f"| `{n}` | {len(missing_repos[n])} | {missing_calls[n]} |" for n in ranked[:300]
              if len(missing_repos[n]) >= 2]
    (ROOT / "GAPS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
