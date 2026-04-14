from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .geometry import quat_rotate
    from .runtime_paths import default_runtime_dir
except ImportError:
    from geometry import quat_rotate
    from runtime_paths import default_runtime_dir


MODULE_DIR = Path(__file__).resolve().parent
EX_DIR = MODULE_DIR / "ex"
if str(EX_DIR) not in sys.path:
    sys.path.insert(0, str(EX_DIR))

from engine import protocol as proto  # type: ignore
from recording_control import DEFAULT_CONTROL_PATH, load_control_payload, sanitize_session_name, save_control_payload


DEFAULT_RUNTIME_DIR = default_runtime_dir()
DEFAULT_DETECTIONS_PATH = DEFAULT_RUNTIME_DIR / "latest_detections.json"
DEFAULT_CSV_PATH = DEFAULT_RUNTIME_DIR / "aruco_motor_dataset.csv"
MARKER_IDS = tuple(range(1, 16))


def _csv_headers() -> List[str]:
    headers = [
        "timestamp",
        "roll",
        "seg1",
        "seg2",
        "roll_current",
        "seg1_current",
        "seg2_current",
    ]
    for marker_id in MARKER_IDS:
        headers.extend(
            [
                f"aruco{marker_id}_px",
                f"aruco{marker_id}_py",
                f"aruco{marker_id}_pz",
                f"aruco{marker_id}_vx",
                f"aruco{marker_id}_vy",
                f"aruco{marker_id}_vz",
            ]
        )
    return headers


def _load_tracker_snapshot(path: Path) -> Tuple[Optional[int], Dict[int, dict]]:
    if not path.exists():
        return None, {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None, {}
    payload = json.loads(raw)
    frame_index_raw = payload.get("frame_index")
    frame_index: Optional[int]
    try:
        frame_index = int(frame_index_raw) if frame_index_raw is not None else None
    except Exception:
        frame_index = None
    markers = payload.get("markers_camera_frame", {})
    out: Dict[int, dict] = {}
    for marker_id_str, pose in markers.items():
        try:
            out[int(marker_id_str)] = pose
        except Exception:
            continue
    return frame_index, out


def _marker_position_and_xvec(marker_pose: dict) -> List[Optional[float]]:
    try:
        p = marker_pose.get("p", [None, None, None])
        q = marker_pose.get("q", None)
        if q is None:
            return [None, None, None, None, None, None]
        x_vec = quat_rotate(tuple(float(v) for v in q), (1.0, 0.0, 0.0))
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(x_vec[0]),
            float(x_vec[1]),
            float(x_vec[2]),
        ]
    except Exception:
        return [None, None, None, None, None, None]


def _invert_seg2_u(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return 360.0 - float(value)


class CsvRecorder:
    def __init__(self, csv_path: Path, detections_path: Path, endpoint: str, control_path: Path):
        self.csv_path = csv_path
        self.detections_path = detections_path
        self.endpoint = endpoint
        self.link = proto.LinkClient(endpoint=self.endpoint)
        self._headers = _csv_headers()
        self._last_markers: Dict[int, dict] = {}
        self._last_detection_frame_index: Optional[int] = None
        self.mapping_cfg = proto.SimMappingConfig()
        self.control_path = control_path
        self._active_session_name: Optional[str] = None
        self._active_partial_path: Optional[Path] = None
        self._recording_started_at: Optional[float] = None
        self._recording_was_active = False

    def _ensure_csv_header(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._headers)

    def _snapshot_markers(self) -> Tuple[Optional[int], Dict[int, dict]]:
        try:
            frame_index, markers = _load_tracker_snapshot(self.detections_path)
            self._last_markers = markers
            return frame_index, markers
        except (OSError, json.JSONDecodeError, ValueError):
            return self._last_detection_frame_index, self._last_markers

    def _row_from_state(self, markers: Dict[int, dict]) -> List[Optional[float]]:
        self.link.poll()
        q_cmd = self.link.last_q_cmd
        currents = self.link.last_currents or {}
        u_cmd = proto.gensim_q_to_control_u(q_cmd, self.mapping_cfg) if q_cmd is not None else None
        if self._recording_started_at is None:
            self._recording_started_at = time.time()
        elapsed_s = float(time.time() - self._recording_started_at)

        row: List[Optional[float]] = [
            elapsed_s,
            float(u_cmd.u_roll) if u_cmd is not None else None,
            float(u_cmd.u_s1) if u_cmd is not None else None,
            _invert_seg2_u(float(u_cmd.u_s2)) if u_cmd is not None else None,
            float(currents["roll"]) if currents.get("roll") is not None else None,
            float(currents["s1"]) if currents.get("s1") is not None else None,
            float(currents["s2"]) if currents.get("s2") is not None else None,
        ]
        for marker_id in MARKER_IDS:
            pose = markers.get(marker_id)
            if pose is None:
                row.extend([None, None, None, None, None, None])
                continue
            row.extend(_marker_position_and_xvec(pose))
        return row

    def _session_paths(self, session_name: str) -> tuple[Path, Path]:
        base = self.csv_path.parent
        safe_name = sanitize_session_name(session_name)
        return (base / f"{safe_name}.partial.csv", base / f"{safe_name}.csv")

    def _start_session_if_needed(self, session_name: str) -> None:
        if self._active_session_name == session_name and self._active_partial_path is not None:
            return
        partial_path, _ = self._session_paths(session_name)
        if partial_path.exists():
            partial_path.unlink()
        self._ensure_csv_header(partial_path)
        self._active_session_name = session_name
        self._active_partial_path = partial_path
        self._recording_started_at = time.time()
        print(f"[recorder] recording -> {partial_path}")

    def _append_row(self) -> None:
        if self._active_partial_path is None:
            return
        frame_index, markers = self._snapshot_markers()
        if frame_index is None or frame_index == self._last_detection_frame_index:
            return
        self._last_detection_frame_index = frame_index
        with self._active_partial_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._row_from_state(markers))

    def _export_session(self, session_name: str) -> Optional[Path]:
        partial_path, export_path = self._session_paths(session_name)
        if not partial_path.exists():
            return None
        shutil.copyfile(partial_path, export_path)
        return export_path

    def run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[recorder] detections={self.detections_path}")
        print(f"[recorder] bridge={self.endpoint} mode=on_new_detection_frame")
        try:
            while True:
                payload = load_control_payload(self.control_path)
                session_name = sanitize_session_name(str(payload.get("csv_name", "session")))
                recording_active = bool(payload.get("recording_active", False))

                if recording_active and (not self._recording_was_active):
                    self._active_session_name = None
                    self._active_partial_path = None
                    self._recording_started_at = None
                    self._last_detection_frame_index = None
                if recording_active:
                    self._start_session_if_needed(session_name)
                    self._append_row()
                self._recording_was_active = recording_active

                if bool(payload.get("export_requested", False)):
                    export_path = self._export_session(session_name)
                    payload["export_requested"] = False
                    payload["status"] = "exported" if export_path is not None else "export_failed"
                    payload["last_export_path"] = str(export_path) if export_path is not None else None
                    save_control_payload(payload, self.control_path)
                time.sleep(0.005)
        finally:
            self.link.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record commanded motor state, current, and ArUco observations to CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS_PATH)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--bridge-endpoint", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL_PATH)
    args = parser.parse_args()

    detections_path = args.detections if args.detections != DEFAULT_DETECTIONS_PATH else args.runtime_dir / "latest_detections.json"
    csv_path = args.csv if args.csv != DEFAULT_CSV_PATH else args.runtime_dir / "aruco_motor_dataset.csv"
    recorder = CsvRecorder(
        csv_path=csv_path,
        detections_path=detections_path,
        endpoint=str(args.bridge_endpoint),
        control_path=args.control,
    )
    recorder.run()


if __name__ == "__main__":
    main()
