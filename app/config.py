"""Loads config.yaml once and exposes it as a plain dict."""
from functools import lru_cache
from pathlib import Path
import os

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache
def settings() -> dict:
    path = Path(os.environ.get("APP_CONFIG", ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)
