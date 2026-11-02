import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption("--run-heavy", action="store_true", help="run tests that need real model weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return
    skip = pytest.mark.skip(reason="heavy: needs model weights; use --run-heavy")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)
