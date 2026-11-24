from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_current_training_configs_keep_patch_protocol():
    configs = sorted((ROOT / "configs").glob("p[012]_*.json"))
    variants = {"p0_mask_x", "p1_mask_x_normal"}
    assert {path.stem for path in configs} == (variants | {f"{name}_padding_official" for name in variants}
                                               | {f"{name}_official_stable" for name in variants}
                                               | {"p2_mask_x_normal_anomaly_patch_official_stable"})
    for path in configs:
        config = load_config(path)
        assert config.raw["model"]["num_features"] == 1
        assert config.raw["model"]["patch_size"] == 16
        assert config.raw["input"]["mask_patch_size"] == 16


def test_p2_matches_stable_p1_settings():
    p1 = load_config(ROOT / "configs/p1_mask_x_normal_official_stable.json").raw
    p2 = load_config(ROOT / "configs/p2_mask_x_normal_anomaly_patch_official_stable.json").raw
    assert p2 == dict(p1, experiment_id="p2_mask_x_normal_anomaly_patch")
