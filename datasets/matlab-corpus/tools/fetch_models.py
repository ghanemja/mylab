"""Download the block-diagram model files (.slx) listed in models.jsonl.

    python tools/fetch_models.py [--cache DIR] [--jobs 4]

The files are test data for the .slx reader of mylab Blocks. They are not
part of this repository: each one is fetched from its own repository at the
commit pinned in manifest.jsonl, with a partial, sparse git clone that only
transfers the listed files, and checked against the SHA-256 in models.jsonl.
Nothing is executed. Files land in <cache>/<owner>__<repo>/<path>.
Default cache: ~/.cache/mylab-corpus-models (or $MYLAB_SLX_CACHE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def default_cache() -> Path:
    return Path(os.environ.get("MYLAB_SLX_CACHE") or Path.home() / ".cache" / "mylab-corpus-models")


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=900)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_repo(repo: str, rows: list[dict], cache: Path) -> tuple[str, str]:
    dest = cache / repo.replace("/", "__")
    if all((dest / r["path"]).is_file() and sha256(dest / r["path"]) == r["sha256"] for r in rows):
        return repo, "cached"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            run(["git", "init", "-q"], work)
            run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"], work)
            run(["git", "config", "core.sparseCheckout", "true"], work)
            (work / ".git" / "info").mkdir(exist_ok=True)
            # literal paths: escape the characters that sparse-checkout patterns treat specially
            patterns = ["/" + "".join("\\" + c if c in "*?[]\\!#" else c for c in r["path"]) for r in rows]
            (work / ".git" / "info" / "sparse-checkout").write_text("\n".join(patterns) + "\n")
            run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", rows[0]["commit"]], work)
            run(["git", "checkout", "-q", "FETCH_HEAD"], work)
            bad = []
            for r in rows:
                src = work / r["path"]
                if not src.is_file() or sha256(src) != r["sha256"]:
                    bad.append(r["path"])
                    continue
                target = dest / r["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
        return repo, "ok" if not bad else f"failed: {len(bad)} file(s) missing or changed, e.g. {bad[0]}"
    except subprocess.CalledProcessError as e:
        return repo, f"failed: {(e.stderr or '')[:200].strip()}"
    except subprocess.TimeoutExpired:
        return repo, "failed: timeout"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=default_cache())
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for line in (ROOT / "models.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_repo[row["repo"]].append(row)
    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for repo, status in pool.map(lambda kv: fetch_repo(kv[0], kv[1], args.cache), by_repo.items()):
            print(f"{repo}: {status} ({len(by_repo[repo])} files)")
    print(f"models in {args.cache}")


if __name__ == "__main__":
    main()
