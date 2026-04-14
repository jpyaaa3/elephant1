from __future__ import annotations

import json
from pathlib import Path

try:
    from .runtime_paths import default_runtime_dir
except ImportError:
    from runtime_paths import default_runtime_dir


DEFAULT_RUNTIME_DIR = default_runtime_dir()
DEFAULT_CONTROL_PATH = DEFAULT_RUNTIME_DIR / "recording_control.json"


def sanitize_session_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "session"


def default_control_payload() -> dict:
    return {
        "csv_name": "session",
        "write_every": 2,
        "recording_active": False,
        "export_requested": False,
        "status": "idle",
        "last_export_path": None,
    }


def load_control_payload(path: Path = DEFAULT_CONTROL_PATH) -> dict:
    if not path.exists():
        return default_control_payload()
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return default_control_payload()
        payload = json.loads(raw)
    except Exception:
        return default_control_payload()
    merged = default_control_payload()
    merged.update(payload if isinstance(payload, dict) else {})
    try:
        merged["write_every"] = max(1, int(merged.get("write_every", 2)))
    except Exception:
        merged["write_every"] = 2
    return merged


def save_control_payload(payload: dict, path: Path = DEFAULT_CONTROL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
