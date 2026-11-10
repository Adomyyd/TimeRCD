from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
for path in (REPO, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
