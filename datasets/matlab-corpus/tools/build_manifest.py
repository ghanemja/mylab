"""Select permissively licensed MATLAB repositories and pin each to a commit.

    python tools/build_manifest.py --max-repos 200 --min-stars 50

Uses the GitHub search API through the `gh` CLI (set GH_TOKEN / GITHUB_TOKEN
in CI) and `git ls-remote` to pin the current commit of the default branch.
Writes manifest.jsonl (one repository per line). Re-running keeps existing
pins unless --refresh is given, so the dataset stays reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.jsonl"

# Licenses that allow redistribution and modification with attribution only.
PERMISSIVE = ["mit", "bsd-2-clause", "bsd-3-clause", "apache-2.0", "isc", "0bsd", "unlicense",
              "cc0-1.0", "zlib", "bsl-1.0"]


def gh_search(query: str, per_page: int = 100, pages: int = 3) -> list[dict]:
    items = []
    for page in range(1, pages + 1):
        out = subprocess.run(
            ["gh", "api", "-X", "GET", "search/repositories", "-f", f"q={query}", "-f", "sort=stars",
             "-f", "order=desc", "-f", f"per_page={per_page}", "-f", f"page={page}"],
            capture_output=True, text=True, check=True).stdout
        batch = json.loads(out)["items"]
        items += batch
        if len(batch) < per_page:
            break
        time.sleep(2)  # stay well inside the search rate limit
    return items


def pin(url: str, branch: str) -> str | None:
    try:
        out = subprocess.run(["git", "ls-remote", url, f"refs/heads/{branch}"], capture_output=True,
                             text=True, timeout=60).stdout.split()
    except subprocess.TimeoutExpired:
        return None
    return out[0] if out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-repos", type=int, default=200)
    ap.add_argument("--min-stars", type=int, default=50)
    ap.add_argument("--refresh", action="store_true", help="re-pin repositories already in the manifest")
    args = ap.parse_args()

    existing = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            row = json.loads(line)
            existing[row["repo"]] = row

    candidates: dict[str, dict] = {}
    for lic in PERMISSIVE:
        q = f"language:MATLAB license:{lic} stars:>={args.min_stars} fork:false"
        try:
            for item in gh_search(q):
                candidates[item["full_name"]] = item
        except subprocess.CalledProcessError as e:
            print(f"search failed for {lic}: {e.stderr[:200]}")
        time.sleep(2)

    ranked = sorted(candidates.values(), key=lambda i: -i["stargazers_count"])
    rows = []
    for item in ranked:
        if len(rows) >= args.max_repos:
            break
        name = item["full_name"]
        if name in existing and not args.refresh:
            rows.append(existing[name])
            continue
        spdx = (item.get("license") or {}).get("spdx_id")
        if not spdx or spdx in ("NOASSERTION", "OTHER"):
            continue
        sha = pin(item["clone_url"], item["default_branch"])
        if not sha:
            continue
        rows.append({
            "repo": name,
            "url": item["html_url"],
            "clone_url": item["clone_url"],
            "commit": sha,
            "branch": item["default_branch"],
            "license": spdx,
            "stars": item["stargazers_count"],
            "description": (item.get("description") or "")[:300],
            "pinned_on": date.today().isoformat(),
        })
        print(f"{name:55s} {spdx:14s} {item['stargazers_count']:6d} {sha[:10]}")
    MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} repositories to {MANIFEST}")


if __name__ == "__main__":
    main()
