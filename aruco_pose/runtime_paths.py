from __future__ import annotations

import os
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def default_runtime_dir() -> Path:
    return MODULE_DIR / "runtime"
