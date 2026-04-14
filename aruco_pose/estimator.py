from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from .geometry import Pose, average_quaternions, average_vectors, pose_inverse, pose_multiply
except ImportError:
    from geometry import Pose, average_quaternions, average_vectors, pose_inverse, pose_multiply


@dataclass(frozen=True)
class MarkerObservation:
    marker_id: int
    pose_camera_marker: Pose


class Node9PoseEstimator:
    def __init__(self, layout_path: Path | str):
        path = Path(layout_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._node9_markers: Dict[int, Pose] = {}
        for marker in payload.get("markers", []):
            if marker["link_name"] != "node9":
                continue
            pose = marker["pose_link_marker"]
            self._node9_markers[int(marker["marker_id"])] = Pose(
                p=tuple(float(v) for v in pose["p"]),
                q=tuple(float(v) for v in pose["q"]),
            )

    def estimate(self, observations: Iterable[MarkerObservation]) -> Optional[Pose]:
        candidates: List[Pose] = []
        for obs in observations:
            pose_node9_marker = self._node9_markers.get(obs.marker_id)
            if pose_node9_marker is None:
                continue
            pose_camera_node9 = pose_multiply(obs.pose_camera_marker, pose_inverse(pose_node9_marker))
            candidates.append(pose_camera_node9)

        if not candidates:
            return None

        return Pose(
            p=average_vectors([candidate.p for candidate in candidates]),
            q=average_quaternions([candidate.q for candidate in candidates]),
        )


class AnchorMarkerEstimator:
    def __init__(self, layout_path: Path | str, *, anchor_marker_id: int = 0):
        path = Path(layout_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._pose_anchor_link_marker = None
        for marker in payload.get("markers", []):
            if int(marker["marker_id"]) != int(anchor_marker_id):
                continue
            pose = marker["pose_link_marker"]
            self._pose_anchor_link_marker = Pose(
                p=tuple(float(v) for v in pose["p"]),
                q=tuple(float(v) for v in pose["q"]),
            )
            break
        if self._pose_anchor_link_marker is None:
            raise ValueError(f"anchor marker {anchor_marker_id} not found in layout")

    @property
    def pose_anchor_link_marker(self) -> Pose:
        return self._pose_anchor_link_marker
