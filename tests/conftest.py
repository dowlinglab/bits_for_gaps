"""Shared test fixtures.

Locates the committed golden scalar targets under ``paper/golden/`` and exposes a
``golden`` loader fixture. The golden files are small JSON snapshots extracted from the
archived published run (iteration 15); see ``paper/golden/README.md``.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "paper" / "golden"


@pytest.fixture(scope="session")
def golden():
    """Return a callable that loads a golden JSON file by name."""
    def _load(name):
        with open(GOLDEN_DIR / name) as f:
            return json.load(f)
    return _load
