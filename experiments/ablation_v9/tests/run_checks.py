"""Run synthetic regression checks through an absolute script path."""
from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve().parent), "-q", *sys.argv[1:]]))
