"""Real code from the corpus: the selected repositories' tests and examples must keep running.

The repositories are not part of mylab. Fetch them once (about 1.7 GB):

    python datasets/matlab-corpus/tools/run_corpus.py fetch

It takes about ten minutes, so it runs only when asked for:

    MYLAB_CORPUS_REGRESSION=1 python -m pytest tests/test_corpus_regression.py

and is skipped until the repositories are there (in ~/.cache/mylab-corpus-runs or $MYLAB_CORPUS_RUNS).
It runs every selected script and fails if one that passed before (regression_expected.json)
fails now; datasets/matlab-corpus/tools/run_corpus.py run --verbose shows the details.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets" / "matlab-corpus"
CACHE = Path(os.environ.get("MYLAB_CORPUS_RUNS", Path.home() / ".cache" / "mylab-corpus-runs"))


def _fetched() -> bool:
    sel = json.loads((CORPUS / "regression.json").read_text())
    return all((CACHE / e["repo"].replace("/", "__") / ".mylab-corpus-commit").exists() for e in sel["repositories"])


@pytest.mark.skipif(os.environ.get("MYLAB_CORPUS_REGRESSION") != "1",
                    reason="set MYLAB_CORPUS_REGRESSION=1 to run the corpus repositories (about ten minutes)")
@pytest.mark.skipif(not CACHE.exists() or not _fetched(), reason="corpus repositories not fetched "
                    "(python datasets/matlab-corpus/tools/run_corpus.py fetch)")
def test_corpus_repositories_still_run():
    p = subprocess.run([sys.executable, str(CORPUS / "tools" / "run_corpus.py"), "run", "--jobs", "3"],
                       capture_output=True, text=True, timeout=3 * 3600)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
