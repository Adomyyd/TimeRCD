import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "name", "seed", "device", "threads", "checkpoint", "output_root",
        "window_sizes", "batch_size", "atol", "rtol", "data",
    }
    optional = {"force_short_padding"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError(f"Required config keys: {sorted(required)}; optional: {sorted(optional)}")
    config.setdefault("force_short_padding", False)
    if type(config["force_short_padding"]) is not bool:
        raise ValueError("force_short_padding must be a boolean")
    name = config["name"]
    if not isinstance(name, str) or not name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name
    ):
        raise ValueError("name must contain only letters, digits, '-' or '_'")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda (first visible GPU)")
    for key in ("threads", "batch_size"):
        positive_int(config[key], key)
    windows = config["window_sizes"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("window_sizes must be a nonempty list")
    for value in windows:
        positive_int(value, "window_sizes entry")
    if len(set(windows)) != len(windows):
        raise ValueError("window_sizes must not contain duplicates")
    for key in ("atol", "rtol"):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("checkpoint", "output_root"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty path")
        config[key] = str(project_path(config[key]))
    if not Path(config["checkpoint"]).is_file():
        raise FileNotFoundError(config["checkpoint"])
    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    if data.get("kind") == "synthetic":
        if set(data) != {"kind", "lengths", "period", "noise_std", "anomaly_length", "anomaly_shift"}:
            raise ValueError("Unexpected or missing synthetic data setting")
        if not isinstance(data["lengths"], list) or not data["lengths"]:
            raise ValueError("lengths must be a nonempty list")
        for length in data["lengths"]:
            positive_int(length, "synthetic length")
        positive_int(data["anomaly_length"], "anomaly_length")
        for key in ("period", "noise_std", "anomaly_shift"):
            value = data[key]
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if data["period"] <= 0 or data["noise_std"] < 0:
            raise ValueError("period must be positive and noise_std nonnegative")
    elif data.get("kind") == "tsb":
        keys = {"kind", "root", "files", "split"}
        if not keys <= set(data) or set(data) - keys - {"selection_manifest"}:
            raise ValueError("Unexpected or missing TSB data setting")
        if data["split"] not in ("test", "full"):
            raise ValueError("split must be test or full")
        files = data["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("Explicit, nonempty TSB files list required")
        if any(not isinstance(f, str) or Path(f).name != f or not f.endswith(".csv") for f in files):
            raise ValueError("files must contain CSV basenames")
        if len(set(files)) != len(files):
            raise ValueError("files must not contain duplicates")
        data["root"] = str(project_path(data["root"]))
        for filename in files:
            if not (Path(data["root"]) / filename).is_file():
                raise FileNotFoundError(Path(data["root"]) / filename)
        if "selection_manifest" in data:
            data["selection_manifest"] = str(project_path(data["selection_manifest"]))
            if not Path(data["selection_manifest"]).is_file():
                raise FileNotFoundError(data["selection_manifest"])
    else:
        raise ValueError("data.kind must be synthetic or tsb")
    return config
