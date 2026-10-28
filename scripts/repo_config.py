"""
repo_config.py - configuration loading shared by all scripts.

configs/config.yaml stores every path relative to the repository root (or as an
absolute path). load_config() returns the configuration with these paths resolved,
so scripts can be started from any working directory. The split files in data/
list image file names; resolve_image() maps such a name to the image folder of the
local LSMA-PQR copy given by paths.images.
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
_IMAGES = None


def _resolve(node):
    if isinstance(node, dict):
        return {k: _resolve(v) for k, v in node.items()}
    if isinstance(node, str) and node:
        p = Path(node)
        return str(p if p.is_absolute() else REPO_ROOT / p)
    return node


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = _resolve(cfg["paths"])
    return cfg


def resolve_image(entry):
    """Split-file entry (file name or any path ending in the file name) -> absolute image path."""
    global _IMAGES
    if _IMAGES is None:
        _IMAGES = Path(load_config()["paths"]["images"])
    name = entry.strip().replace("\\", "/").split("/")[-1]
    return str(_IMAGES / name)
