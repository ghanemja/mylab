"""Download the corpus: the .m files (and license/notice files) of every pinned repository.

    python tools/fetch.py [--cache DIR] [--jobs 8]

Uses partial, sparse git clones so only .m sources and license files are
transferred, checked out at the exact pinned commit. Nothing is executed.
Default cache: ~/.cache/mylab-corpus
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["*.m", "LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "license*", "License*"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=600)


def fetch_one(row: dict, cache: Path) -> tuple[str, str]:
    dest = cache / row["repo"].replace("/", "__")
    marker = dest / ".mylab-corpus-commit"
    if marker.exists() and marker.read_text().strip() == row["commit"]:
        return row["repo"], "cached"
    try:
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.mkdir(parents=True)
        run(["git", "init", "-q"], dest)
        run(["git", "remote", "add", "origin", row["clone_url"]], dest)
        run(["git", "config", "core.sparseCheckout", "true"], dest)
        run(["git", "sparse-checkout", "set", "--no-cone", *PATTERNS], dest)
        run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", row["commit"]], dest)
        run(["git", "checkout", "-q", "FETCH_HEAD"], dest)
        marker.write_text(row["commit"] + "\n")
        return row["repo"], "ok"
    except subprocess.CalledProcessError as e:
        return row["repo"], f"failed: {(e.stderr or '')[:200].strip()}"
    except subprocess.TimeoutExpired:
        return row["repo"], "failed: timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "mylab-corpus")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(line) for line in (ROOT / "manifest.jsonl").read_text().splitlines()]
    args.cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        for repo, status in pool.map(lambda r: fetch_one(r, args.cache), rows):
            if status not in ("ok", "cached"):
                print(f"{repo}: {status}")
    print(f"corpus in {args.cache}")


if __name__ == "__main__":
    main()
