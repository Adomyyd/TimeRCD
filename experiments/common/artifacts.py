import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from .config import ROOT


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_row(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode()


def create_run(config, config_path):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(config["output_root"]) / f"{timestamp}_{config['name']}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "arrays").mkdir()
    manifest = {
        "status": "initializing", "started_utc": timestamp, "command": sys.argv,
        "config": config, "config_source": str(Path(config_path).resolve()),
    }
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def capture_provenance(directory, manifest):
    config = manifest["config"]
    shutil.copyfile(manifest["config_source"], directory / "config.input.json")
    write_json(directory / "config.resolved.json", config)
    if "selection_manifest" in config["data"]:
        selection = Path(config["data"]["selection_manifest"])
        shutil.copyfile(selection, directory / "selection.json")
        manifest["selection_sha256"] = sha256(selection)
    manifest["git"] = {"commit": git("rev-parse", "HEAD").strip(), "status": git("status", "--short")}
    (directory / "working_tree.patch").write_text(git("diff", "HEAD", "--binary"))
    tracked = [ROOT / p for p in git("ls-files", "-z").split("\0") if p]
    experiment_root = ROOT / "experiments"
    experimental = list(experiment_root.glob("*"))
    for folder in ("common", "stage1", "stage2", "stage2_context_validation", "stage2_followup", "stage2_paths", "stage2_patch_types", "protocols", "configs", "tests"):
        experimental.extend((experiment_root / folder).rglob("*"))
    sources = sorted({p for p in tracked + experimental
                      if p.is_file() and p.suffix in {".py", ".json", ".toml", ".md"}
                      and directory not in p.parents})
    fingerprints = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        target = directory / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        fingerprints[str(relative)] = sha256(target)
    write_json(directory / "source.sha256.json", fingerprints)
    manifest["checkpoint"] = {
        "path": config["checkpoint"], "sha256": sha256(config["checkpoint"]),
        "bytes": Path(config["checkpoint"]).stat().st_size,
    }
    manifest["environment"] = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "pandas", "scikit-learn", "einops", "jaxtyping")},
    }
    packages = sorted(
        [{"name": distribution.metadata["Name"], "version": distribution.version}
         for distribution in importlib.metadata.distributions()],
        key=lambda package: package["name"].lower(),
    )
    write_json(directory / "environment.packages.json", packages)
