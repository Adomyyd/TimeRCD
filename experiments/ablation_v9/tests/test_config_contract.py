from pathlib import Path

from src.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_variant_configs_validate():
    configs = sorted((ROOT / "configs").glob("*.json"))
    assert {path.stem for path in configs} == {"p0_mask_x", "p1_full_x", "p2_full_x0"}
    for path in configs:
        config = load_config(path)
        assert config.raw["model"]["num_features"] == 1
