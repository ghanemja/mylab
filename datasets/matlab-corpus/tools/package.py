"""Build a redistributable archive of the corpus (code + licenses + attribution notice).

    python tools/package.py [--cache DIR] [--out dist/matlab-corpus.tar.gz]

Each repository keeps its own license file(s); NOTICE.md lists every
repository, its commit, license and URL. Repositories without a license file
in the checkout are left out, because their license text can't travel with
the code.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSE_NAMES = ("license", "licence", "copying", "notice")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "mylab-corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "matlab-corpus.tar.gz")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(line) for line in (ROOT / "manifest.jsonl").read_text().splitlines()]
    notice = ["# MATLAB Corpus - attribution", "",
              "Each folder is an unmodified copy of the .m files of one repository at the listed commit,",
              "distributed under that repository's own license (its license file is included).", "",
              "| repository | commit | license |", "|---|---|---|"]
    kept = skipped = 0
    with tarfile.open(args.out, "w:gz") as tar:
        for row in manifest:
            src = args.cache / row["repo"].replace("/", "__")
            if not src.is_dir():
                continue
            licenses = [p for p in src.iterdir() if p.is_file() and p.name.lower().startswith(LICENSE_NAMES)]
            if not licenses:
                skipped += 1
                continue
            prefix = f"matlab-corpus/{row['repo'].replace('/', '__')}"
            for p in sorted(src.rglob("*")):
                if p.is_file() and ".git" not in p.parts and not p.name.startswith("._") and (
                        p.suffix == ".m" or p in licenses) and p.name != ".mylab-corpus-commit":
                    tar.add(p, arcname=f"{prefix}/{p.relative_to(src).as_posix()}")
            notice.append(f"| [{row['repo']}]({row['url']}) | `{row['commit'][:12]}` | {row['license']} |")
            kept += 1
        notice_path = args.out.parent / "NOTICE.md"
        notice_path.write_text("\n".join(notice) + "\n")
        tar.add(notice_path, arcname="matlab-corpus/NOTICE.md")
        tar.add(ROOT / "README.md", arcname="matlab-corpus/README.md")
    print(f"packaged {kept} repositories ({skipped} skipped: no license file in checkout) -> {args.out}")


if __name__ == "__main__":
    main()
