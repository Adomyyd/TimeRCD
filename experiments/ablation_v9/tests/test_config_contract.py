from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_variant_configs_match_single_gpu_training_patch_sizes():
    configs = sorted((ROOT / "configs").glob("p[012]_*.json"))
    variants = {"p0_mask_x", "p1_full_x", "p2_full_x0"}
    assert {path.stem for path in configs} == variants | {f"{name}_padding_official" for name in variants}
    for path in configs:
        config = load_config(path)
        assert config.raw["model"]["num_features"] == 1
        assert config.raw["model"]["patch_size"] == 16
        assert config.raw["input"]["mask_patch_size"] == 16
