"""Shared test fixtures.

Locates the committed golden scalar targets under ``paper/golden/`` and exposes a
``golden`` loader fixture. The golden files are small JSON snapshots extracted from the
archived published run (iteration 15); see ``paper/golden/README.md``.

Also puts the repo's ``examples/`` directory on ``sys.path`` so ``examples/`` and
``paper/`` (neither pip-installed -- see REFACTOR_PLAN.md §7.3) are importable in
dev/CI as top-level packages, e.g. ``import vle_distillation.activity_model``. This
also makes ``examples/vle_distillation/juliapkg.json`` discoverable by juliapkg (which
scans every ``<sys.path entry>/<subdir>/juliapkg.json``), pinning Clapeyron.jl for
anything that lazily imports ``juliacall``.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "paper" / "golden"
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture(scope="session")
def golden():
    """Return a callable that loads a golden JSON file by name."""
    def _load(name):
        with open(GOLDEN_DIR / name) as f:
            return json.load(f)
    return _load
