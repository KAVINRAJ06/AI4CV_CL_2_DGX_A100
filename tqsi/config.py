from pathlib import Path
import copy
import os
import yaml


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    seen.add(path)
    cfg = yaml.safe_load(os.path.expandvars(path.read_text(encoding="utf-8")))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    parent = cfg.pop("extends", None)
    if parent:
        cfg = merge(load_config(path.parent / parent, seen), cfg)
    return cfg
