from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _to_ns(d):
    """Recursively turn dicts into attribute-accessible namespaces."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def load_config(path: str | Path = ROOT / "config.yaml", overrides: dict | None = None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for k, v in (overrides or {}).items():
        if v is None:
            continue
        # support dotted keys: "vehicle.imgsz"
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

    # resolve relative weight paths against project root
    for key in ("vehicle", "plate"):
        w = cfg[key]["weights"]
        if not Path(w).is_absolute():
            cfg[key]["weights"] = str(ROOT / w)
    if not Path(cfg["enhance"]["model_dir"]).is_absolute():
        cfg["enhance"]["model_dir"] = str(ROOT / cfg["enhance"]["model_dir"])
    if not Path(cfg["output"]["dir"]).is_absolute():
        cfg["output"]["dir"] = str(ROOT / cfg["output"]["dir"])

    return _to_ns(cfg)
