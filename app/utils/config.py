from pathlib import Path
import yaml

_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
_cache = {}


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT
    key = str(p)
    if key not in _cache:
        with open(p, "r", encoding="utf-8") as f:
            _cache[key] = yaml.safe_load(f)
    return _cache[key]


def weights() -> dict:
    return load_config()["scoring"]["weights"]


def verdict_thresholds() -> dict:
    return load_config()["scoring"]["verdict"]
