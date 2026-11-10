from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "generate": HERE / "src" / "generate_400k.py",
    "train": HERE / "src" / "train_400k.py",
    "compare": HERE / "src" / "compare_400k.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original TimeRCD 400k workflow")
    parser.add_argument("--mode", choices=sorted(SCRIPTS), required=True)
    parser.add_argument("--resume", help="Absolute checkpoint path for train mode; defaults to the latest checkpoint")
    args = parser.parse_args()
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid with --mode train")
    if args.resume and not Path(args.resume).is_absolute():
        parser.error("--resume must be an absolute path")

    script = SCRIPTS[args.mode]
    sys.path.insert(0, str(HERE.parents[1]))
    sys.argv = [str(script)]
    if args.resume:
        sys.argv.extend(["--resume", args.resume])
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
