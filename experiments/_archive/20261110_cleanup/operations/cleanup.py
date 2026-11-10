"""Bounded experiment layout cleanup. No training/data/model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path("/diskA/YYD/TimeRCD")
EXP = ROOT / "experiments"
ARCH = EXP / "_archive/20261110_cleanup"
DOCS = ["AGENTS.md", "研究问题和思路.md", "方法设计.md", "实验记录.md", "README.md",
        "experiments/README.md", "experiments/original_400k/README.md",
        "experiments/ablation_v9/README.md", "experiments/ablation_v9/EXPERIMENT_SPEC.md",
        "experiments/ablation_v9/IMPLEMENTATION_DIFF.md"]
CODE = ["experiments/ablation_v9/src/audit.py", "experiments/ablation_v9/tests/test_parallel_contract.py"]
LINKS = {
    "train_400k.py": EXP / "original_400k/src/train_400k.py",
    "generate_400k.py": EXP / "original_400k/src/generate_400k.py",
    "compare_400k.py": EXP / "original_400k/src/compare_400k.py",
    "EXPERIMENT_SPEC.md": EXP / "ablation_v9/EXPERIMENT_SPEC.md",
}
MOVES = [
    (EXP / "original_400k/migrations", ARCH / "original_400k_migrations"),
    (EXP / "ablation_v9/audits/padding_20261110/before", ARCH / "padding_before"),
    (EXP / "original_400k/audits/stop_20261110/stop_and_inspect.py", ARCH / "one_off_scripts/stop_and_inspect.py"),
]
PROTECTED = [EXP / "original_400k/data", EXP / "original_400k/runs",
             EXP / "ablation_v9/runs", EXP / "ablation_v9/registry", ROOT / "best_model"]
plan_path = ARCH / "plan.json"
result_path = ARCH / "result.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files_at(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())

def identity(path, digest=False):
    st = path.stat()
    item = {"bytes": st.st_size, "inode": st.st_ino, "mtime_ns": st.st_mtime_ns}
    if digest:
        item["sha256"] = sha(path)
    return item

def assets():
    return {str(p.relative_to(ROOT)): identity(p, p.stat().st_size < 2_000_000)
            for root in PROTECTED for p in files_at(root)}

def production():
    result = {}
    for route in ("original_400k", "ablation_v9"):
        base = EXP / route
        paths = [base / "run_experiment.py"]
        paths += files_at(base / "src")
        if route == "ablation_v9":
            paths += files_at(base / "configs") + files_at(base / "tests")
        for p in paths:
            if "__pycache__" not in p.parts and p.suffix in (".py", ".json"):
                result[str(p.relative_to(ROOT))] = sha(p)
    return result

def live_training():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, ProcessLookupError):
            continue
        if not args or "python" not in Path(args[0]).name:
            continue
        if any(("TimeRCD" in a and a.endswith(("train_400k.py", "/training.py")))
               or ("TimeRCD" in a and a.endswith("run_experiment.py") and "train" in args)
               for a in args[1:]):
            found.append({"pid": int(proc.name), "args": args})
    return found

def cache_dirs():
    result = []
    for root, dirs, _ in os.walk(EXP, followlinks=False):
        current = Path(root)
        dirs[:] = [d for d in dirs if d not in ("runs", "data", "registry", "_archive")
                   and not (current / d).is_symlink()]
        for d in list(dirs):
            if d in ("__pycache__", ".pytest_cache"):
                result.append(current / d)
                dirs.remove(d)
    return result

def references():
    needles = [str(EXP / name) for name in LINKS]
    needles += ["original_400k/migrations", "padding_20261110/before", "stop_and_inspect.py"]
    hits = []
    for root, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in (".git", "_archive", "runs", "data", "datasets",
                   "best_model", "__pycache__", "audits", "migrations") and not (Path(root) / d).is_symlink()]
        for name in files:
            p = Path(root) / name
            if p.is_symlink() or p.suffix != ".py" or p.stat().st_size > 1_000_000:
                continue
            for number, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if any(n in line for n in needles):
                    hits.append({"file": str(p.relative_to(ROOT)), "line": number, "text": line.strip()})
    return hits

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=("plan", "apply", "verify", "purge_caches"))
args = parser.parse_args()
ARCH.mkdir(parents=True, exist_ok=True)
if args.action == "plan":
    if plan_path.exists():
        raise FileExistsError(plan_path)
    active = live_training()
    if active:
        raise RuntimeError(f"Training is active: {active}")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "protected": assets(), "production": production(), "references": references(),
            "moves": [{"source": str(a), "destination": str(b),
                       "files": {str(p.relative_to(a)) if a.is_dir() else p.name: sha(p)
                                 for p in files_at(a)}} for a, b in MOVES],
            "remove_links": {str(EXP / name): str(target) for name, target in LINKS.items()},
            "keep_links": {str(EXP / name): os.readlink(EXP / name) for name in
                           ("runs", "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set.pkl",
                            "dataset_400000_samples_Nonelen_1.0ratio_univariate_attr_set.manifest.json")}}
    for rel in DOCS + CODE:
        src = ROOT / rel
        dst = ARCH / "before" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"protected_files": len(plan["protected"]), "production_files": len(plan["production"]),
                      "references": plan["references"], "cache_dirs": [str(p) for p in cache_dirs()]}, ensure_ascii=False))
elif args.action == "apply":
    plan = json.loads(plan_path.read_text())
    if result_path.exists():
        raise FileExistsError(result_path)
    if live_training():
        raise RuntimeError("Training is active")
    if assets() != plan["protected"]:
        raise RuntimeError("Protected assets changed since plan")
    if plan["references"]:
        raise RuntimeError("Live code references old paths")
    for a, b in MOVES:
        if not a.exists() or b.exists():
            raise RuntimeError(f"Unexpected move paths: {a} -> {b}")
    for name, target in LINKS.items():
        link = EXP / name
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected compatibility link: {link}")
    for a, b in MOVES:
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(a), str(b))
    for name in LINKS:
        (EXP / name).unlink()
    result_path.write_text(json.dumps({"moved": [str(a) for a, b in MOVES],
                                      "removed_links": list(LINKS)}, ensure_ascii=False, indent=2) + "\n")
    print("Archived backups and one-off script; removed four obsolete entry links")
elif args.action == "purge_caches":
    removed = []
    for p in cache_dirs():
        if not p.resolve().is_relative_to(EXP.resolve()) or p.is_symlink():
            raise RuntimeError(f"Unsafe cache path: {p}")
        removed.append({"path": str(p), "files": len(files_at(p)),
                        "bytes": sum(f.stat().st_size for f in files_at(p))})
        shutil.rmtree(p)
    (ARCH / "removed_caches.json").write_text(json.dumps(removed, indent=2) + "\n")
    print(json.dumps(removed))
else:
    plan = json.loads(plan_path.read_text())
    assert assets() == plan["protected"], "Protected assets modified"
    after = production()
    changed = sorted(k for k, v in plan["production"].items() if after.get(k) != v)
    assert set(changed) <= set(CODE), changed
    for item in plan["moves"]:
        src, dst = Path(item["source"]), Path(item["destination"])
        assert not src.exists(), src
        for name, digest in item["files"].items():
            p = dst / name if dst.is_dir() else dst
            assert sha(p) == digest, p
    for name in LINKS:
        assert not (EXP / name).is_symlink() and not (EXP / name).exists()
    for name, target in plan["keep_links"].items():
        assert Path(name).is_symlink() and Path(name).exists() and os.readlink(name) == target
    out = {"protected_files_unchanged": len(plan["protected"]), "changed_production": changed,
           "archived_files_byte_identical": True, "legacy_asset_links_valid": True,
           "live_training": live_training(), "live_legacy_references": references()}
    assert not out["live_legacy_references"]
    (ARCH / "verification.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False))
