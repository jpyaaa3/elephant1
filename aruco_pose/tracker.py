from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from .estimator import AnchorMarkerEstimator, MarkerObservation, Node9PoseEstimator
    from .geometry import Pose, pose_inverse, pose_multiply, quat_dot, quat_normalize, quaternion_from_axes
    from .recording_control import DEFAULT_CONTROL_PATH, load_control_payload
    from .runtime_paths import default_runtime_dir
except ImportError:
    from estimator import AnchorMarkerEstimator, MarkerObservation, Node9PoseEstimator
    from geometry import Pose, pose_inverse, pose_multiply, quat_dot, quat_normalize, quaternion_from_axes
    from recording_control import DEFAULT_CONTROL_PATH, load_control_payload
    from runtime_paths import default_runtime_dir


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT_PATH = MODULE_DIR / "config" / "robot_markers.json"
DEFAULT_RUNTIME_DIR = default_runtime_dir()
DEFAULT_DETECTIONS_PATH = DEFAULT_RUNTIME_DIR / "latest_detections.json"
DEFAULT_DEBUG_PATH = DEFAULT_RUNTIME_DIR / "debug_diagnostics.json"
DEFAULT_DEBUG_TRACE_PATH = DEFAULT_RUNTIME_DIR / "debug_trace.jsonl"


def _write_runtime_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Runtime files are shared with long-lived readers on Windows.
    # Rewrite in place so readers do not race with unlink/replace semantics.
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _dict_name_to_cv_id(name: str) -> int:
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unsupported OpenCV ArUco dictionary: {name}")
    return int(getattr(cv2.aruco, name))


def _rvec_tvec_to_pose(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
    rot_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    x_axis = tuple(float(v) for v in rot_matrix[:, 0])
    y_axis = tuple(float(v) for v in rot_matrix[:, 1])
    z_axis = tuple(float(v) for v in rot_matrix[:, 2])
    quat = quaternion_from_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    return Pose(
        p=tuple(float(v) for v in np.asarray(tvec, dtype=np.float64).reshape(3)),
        q=quat,
    )


def _pose_to_rvec_tvec(pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z, w = quat_normalize(pose.q)
    rot_matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rot_matrix)
    tvec = np.asarray(pose.p, dtype=np.float64).reshape(3, 1)
    return rvec, tvec


def _pose_to_dict(pose: Pose) -> dict:
    return {
        "p": [float(v) for v in pose.p],
        "q": [float(v) for v in pose.q],
    }


def _load_layout(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _square_object_points(marker_size_m: float) -> np.ndarray:
    half = float(marker_size_m) * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pose: Pose,
) -> float:
    rvec, tvec = _pose_to_rvec_tvec(pose)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    err = projected - image_points
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _quat_angle_deg(a, b) -> float:
    dot = abs(quat_dot(quat_normalize(a), quat_normalize(b)))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_blend(prev, curr, alpha: float):
    p = quat_normalize(prev)
    c = quat_normalize(curr)
    if quat_dot(p, c) < 0.0:
        c = (-c[0], -c[1], -c[2], -c[3])
    mixed = tuple((1.0 - alpha) * p[i] + alpha * c[i] for i in range(4))
    return quat_normalize(mixed)


def _vec_dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


class PoseFilter:
    def __init__(self, alpha: float, max_translation_jump_m: float, max_rotation_jump_deg: float):
        self.alpha = float(alpha)
        self.max_translation_jump_m = float(max_translation_jump_m)
        self.max_rotation_jump_deg = float(max_rotation_jump_deg)
        self._state: Optional[Pose] = None

    def update(self, measurement: Optional[Pose]) -> Optional[Pose]:
        if measurement is None:
            return self._state
        if self._state is None:
            self._state = measurement
            return self._state
        if _vec_dist(self._state.p, measurement.p) > self.max_translation_jump_m:
            return self._state
        if _quat_angle_deg(self._state.q, measurement.q) > self.max_rotation_jump_deg:
            return self._state
        p = tuple((1.0 - self.alpha) * self._state.p[i] + self.alpha * measurement.p[i] for i in range(3))
        q = _quat_blend(self._state.q, measurement.q, self.alpha)
        self._state = Pose(p=p, q=q)
        return self._state


class RealSenseArucoTracker:
    def __init__(
        self,
        layout_path: Path,
        detections_path: Path,
        debug_path: Path,
        debug_trace_path: Path,
        width: int,
        height: int,
        fps: int,
        show_mode: Optional[str],
        detect_scale: float,
        enable_debug_output: bool,
        debug_every: int,
        control_path: Path,
    ):
        self.layout_path = layout_path
        self.detections_path = detections_path
        self.debug_path = debug_path
        self.debug_trace_path = debug_trace_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.show_mode = str(show_mode).lower() if show_mode is not None else None
        self.show = self.show_mode in {"high", "low"}
        self.detect_scale = max(0.1, min(1.0, float(detect_scale)))
        self.enable_debug_output = bool(enable_debug_output)
        self.debug_every = max(1, int(debug_every))
        self.control_path = control_path
        self.write_every = 1
        self._show_max_width = self.width if self.show_mode == "high" else 426
        self._show_max_height = self.height if self.show_mode == "high" else 240
        self._show_interval_s = 0.0 if self.show_mode == "high" else 0.1
        self._last_show_wall = 0.0

        self.layout = _load_layout(layout_path)
        self.marker_size_m = float(self.layout["aruco"]["marker_size_m"])
        self.dictionary_name = str(self.layout["aruco"]["dictionary"])
        self.dictionary = cv2.aruco.getPredefinedDictionary(_dict_name_to_cv_id(self.dictionary_name))
        detector_params = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        elif hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, detector_params)
        self.estimator = Node9PoseEstimator(layout_path)
        self.anchor_estimator = AnchorMarkerEstimator(layout_path, anchor_marker_id=0)
        self._anchor_marker_id = int(self.layout["tracking_strategy"]["anchor_marker_id"])
        self._node9_marker_ids = set(int(v) for v in self.layout["tracking_strategy"]["node9_marker_ids"])
        self._camera_filter = PoseFilter(alpha=0.25, max_translation_jump_m=0.12, max_rotation_jump_deg=45.0)
        self._node9_filter = PoseFilter(alpha=0.25, max_translation_jump_m=0.10, max_rotation_jump_deg=60.0)
        self._anchor_object_points = _square_object_points(self.marker_size_m)
        self._apply_control_write_every()

    def _apply_control_write_every(self) -> None:
        payload = load_control_payload(self.control_path)
        try:
            self.write_every = max(1, int(payload.get("write_every", self.write_every)))
        except Exception:
            pass

    def _start_pipeline(self) -> Tuple[rs.pipeline, rs.pipeline_profile, np.ndarray, np.ndarray]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        camera_matrix = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        return pipeline, profile, camera_matrix, dist_coeffs

    def _prepare_show_image(self, image: np.ndarray) -> np.ndarray:
        if self.show_mode != "low":
            return image
        src_h, src_w = image.shape[:2]
        scale = min(self._show_max_width / float(src_w), self._show_max_height / float(src_h))
        scale = min(1.0, scale)
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        return cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)

    def _estimate_marker_poses(
        self, image_bgr: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> Tuple[List[MarkerObservation], Dict[int, Pose], Optional[np.ndarray], Dict[int, List[dict]]]:
        detect_image = image_bgr
        scale_inv = 1.0
        if self.detect_scale < 0.999:
            detect_image = cv2.resize(image_bgr, None, fx=self.detect_scale, fy=self.detect_scale, interpolation=cv2.INTER_AREA)
            scale_inv = 1.0 / self.detect_scale
        corners, ids, _ = self.detector.detectMarkers(detect_image)
        debug = image_bgr.copy() if self.show else None
        observations: List[MarkerObservation] = []
        marker_pose_map: Dict[int, Pose] = {}
        marker_candidates_map: Dict[int, List[dict]] = {}

        if ids is None or len(ids) == 0:
            return observations, marker_pose_map, debug, marker_candidates_map

        if scale_inv != 1.0:
            corners = [np.asarray(corner, dtype=np.float64) * scale_inv for corner in corners]

        if debug is not None:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        for idx, marker_id in enumerate(ids.reshape(-1)):
            marker_id_int = int(marker_id)
            marker_corners = np.asarray(corners[idx], dtype=np.float64)
            pose, rvec, tvec, candidates = self._estimate_single_marker_pose(
                marker_corners=marker_corners,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
            marker_candidates_map[marker_id_int] = candidates
            if pose is None:
                continue
            marker_pose_map[marker_id_int] = pose
            if marker_id_int in self._node9_marker_ids:
                observations.append(MarkerObservation(marker_id=marker_id_int, pose_camera_marker=pose))
            if debug is not None and rvec is not None and tvec is not None:
                self._draw_axes(debug, camera_matrix, dist_coeffs, rvec, tvec)
                base_x = int(corners[idx][0][0][0])
                base_y = int(corners[idx][0][0][1])
                cv2.putText(
                    debug,
                    f"id={marker_id_int}",
                    (base_x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug,
                    "aruco_quat=(%.3f, %.3f, %.3f, %.3f)" % pose.q,
                    (base_x, base_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug,
                    f"aruco_depth={pose.p[2]:.3f}m",
                    (base_x, base_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
        return observations, marker_pose_map, debug, marker_candidates_map

    def _estimate_single_marker_pose(
        self,
        *,
        marker_corners: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[Pose], Optional[np.ndarray], Optional[np.ndarray], List[dict]]:
        image_points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self._anchor_object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except Exception:
            return None, None, None, []
        if not ok:
            return None, None, None, []

        best_pose = None
        best_rvec = None
        best_tvec = None
        best_rmse = None
        candidates: List[dict] = []
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            try:
                if hasattr(cv2, "solvePnPRefineVVS"):
                    cv2.solvePnPRefineVVS(
                        self._anchor_object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                    )
                elif hasattr(cv2, "solvePnPRefineLM"):
                    cv2.solvePnPRefineLM(
                        self._anchor_object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                    )
            except Exception:
                pass
            pose = _rvec_tvec_to_pose(rvec, tvec)
            rmse = _reprojection_rmse(self._anchor_object_points, image_points, camera_matrix, dist_coeffs, pose)
            candidates.append(
                {
                    "marker_pose_camera": _pose_to_dict(pose),
                    "rvec": [float(v) for v in rvec.reshape(3)],
                    "tvec": [float(v) for v in tvec.reshape(3)],
                    "reprojection_rmse_px": float(rmse),
                }
            )
            if best_rmse is None or rmse < best_rmse:
                best_pose = pose
                best_rvec = rvec
                best_tvec = tvec
                best_rmse = rmse
        return best_pose, best_rvec, best_tvec, candidates

    def _select_anchor_pose(
        self,
        candidate_payloads: List[dict],
        previous_camera_pose_housing: Optional[Pose],
    ) -> Tuple[Optional[Pose], List[dict], Optional[int]]:
        if not candidate_payloads:
            return None, [], None

        best_pose = None
        best_score = None
        best_index = None
        diagnostics: List[dict] = []
        for idx, candidate_payload in enumerate(candidate_payloads):
            candidate = Pose(
                p=tuple(float(v) for v in candidate_payload["marker_pose_camera"]["p"]),
                q=tuple(float(v) for v in candidate_payload["marker_pose_camera"]["q"]),
            )
            pose_camera_housing = pose_multiply(candidate, pose_inverse(self.anchor_estimator.pose_anchor_link_marker))
            camera_pose_housing = pose_inverse(pose_camera_housing)
            pos_score = 0.0
            rot_score = 0.0
            if previous_camera_pose_housing is not None:
                pos_score = _vec_dist(camera_pose_housing.p, previous_camera_pose_housing.p)
                rot_score = _quat_angle_deg(camera_pose_housing.q, previous_camera_pose_housing.q) / 180.0
            reproj_rmse = float(candidate_payload.get("reprojection_rmse_px", 0.0))
            reproj_score = reproj_rmse * 0.05
            score = pos_score + rot_score + reproj_score
            diagnostics.append(
                {
                    "candidate_index": idx,
                    "marker_pose_camera": _pose_to_dict(candidate),
                    "camera_pose_housing": _pose_to_dict(camera_pose_housing),
                    "position_delta_m": pos_score,
                    "rotation_delta_deg": rot_score * 180.0,
                    "score": score,
                    "reprojection_rmse_px": reproj_rmse,
                }
            )
            if best_score is None or score < best_score:
                best_score = score
                best_pose = candidate
                best_index = idx
        return best_pose, diagnostics, best_index

    def _draw_axes(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_len: Optional[float] = None,
    ) -> None:
        axis_len = float(axis_len if axis_len is not None else self.marker_size_m * 0.75)
        axes = np.float32(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, axis_len],
            ]
        )
        img_pts, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
        pts = np.round(img_pts.reshape(-1, 2)).astype(int)
        origin = tuple(pts[0])
        cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), 2)
        cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), 2)
        cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), 2)

    def _write_detections(
        self,
        marker_pose_map: Dict[int, Pose],
        housing_pose: Optional[Pose],
        camera_pose: Optional[Pose],
        node9_pose: Optional[Pose],
        frame_index: int,
    ) -> None:
        payload = {
            "schema_version": 1,
            "frame_index": int(frame_index),
            "world_frame": "housing_frame",
            "detected_marker_ids": sorted(int(k) for k in marker_pose_map.keys()),
            "markers_camera_frame": {str(k): _pose_to_dict(v) for k, v in marker_pose_map.items()},
            "housing_pose": _pose_to_dict(housing_pose) if housing_pose is not None else None,
            "camera_pose": _pose_to_dict(camera_pose) if camera_pose is not None else None,
            "node9_pose": _pose_to_dict(node9_pose) if node9_pose is not None else None,
        }
        _write_runtime_json(self.detections_path, payload)

    def _write_debug_diagnostics(
        self,
        *,
        frame_index: int,
        marker_pose_map: Dict[int, Pose],
        anchor_pose_camera_raw: Optional[Pose],
        anchor_pose_camera_candidates: List[dict],
        anchor_pose_camera_selected_index: Optional[int],
        camera_pose_housing_raw: Optional[Pose],
        camera_pose_housing_filtered: Optional[Pose],
        node9_pose_camera_raw: Optional[Pose],
        node9_pose_housing_raw: Optional[Pose],
        node9_pose_housing_filtered: Optional[Pose],
    ) -> None:
        payload = {
            "schema_version": 1,
            "frame_index": int(frame_index),
            "detected_marker_ids": sorted(int(k) for k in marker_pose_map.keys()),
            "markers_camera_frame": {str(k): _pose_to_dict(v) for k, v in marker_pose_map.items()},
            "anchor_marker_id": self._anchor_marker_id,
            "anchor_pose_camera_raw": _pose_to_dict(anchor_pose_camera_raw) if anchor_pose_camera_raw is not None else None,
            "anchor_pose_camera_candidates": anchor_pose_camera_candidates,
            "anchor_pose_camera_selected_index": anchor_pose_camera_selected_index,
            "camera_pose_housing_raw": _pose_to_dict(camera_pose_housing_raw) if camera_pose_housing_raw is not None else None,
            "camera_pose_housing_filtered": _pose_to_dict(camera_pose_housing_filtered)
            if camera_pose_housing_filtered is not None
            else None,
            "node9_pose_camera_raw": _pose_to_dict(node9_pose_camera_raw) if node9_pose_camera_raw is not None else None,
            "node9_pose_housing_raw": _pose_to_dict(node9_pose_housing_raw) if node9_pose_housing_raw is not None else None,
            "node9_pose_housing_filtered": _pose_to_dict(node9_pose_housing_filtered)
            if node9_pose_housing_filtered is not None
            else None,
        }
        _write_runtime_json(self.debug_path, payload)
        _append_jsonl(self.debug_trace_path, payload)

    def run(self, *, record_plate_once: bool) -> None:
        pipeline, _profile, camera_matrix, dist_coeffs = self._start_pipeline()
        frame_index = 0
        last_node9_pose: Optional[Pose] = None
        last_camera_pose: Optional[Pose] = None
        housing_identity = Pose(p=(0.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0))
        print(
            f"[tracker] dictionary={self.dictionary_name} marker_size={self.marker_size_m:.4f}m "
            f"detections={self.detections_path} anchor_marker={self._anchor_marker_id} "
            f"detect_scale={self.detect_scale:.2f} write_every={self.write_every} "
            f"debug={'on' if self.enable_debug_output else 'off'} show={self.show_mode or 'off'}"
        )
        if self.show:
            cv2.namedWindow("aruco_tracker", cv2.WINDOW_NORMAL)

        try:
            while True:
                self._apply_control_write_every()
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                observations, marker_pose_map, debug, marker_candidates_map = self._estimate_marker_poses(
                    color,
                    camera_matrix,
                    dist_coeffs,
                )

                anchor_camera = marker_pose_map.get(self._anchor_marker_id)
                anchor_candidate_debug: List[dict] = []
                anchor_selected_index: Optional[int] = None
                if anchor_camera is not None:
                    anchor_candidates = marker_candidates_map.get(self._anchor_marker_id, [])
                    if anchor_candidates:
                        anchor_camera, anchor_candidate_debug, anchor_selected_index = self._select_anchor_pose(
                            anchor_candidates,
                            last_camera_pose,
                        )
                        anchor_camera = anchor_camera or marker_pose_map[self._anchor_marker_id]
                        marker_pose_map[self._anchor_marker_id] = anchor_camera
                pose_camera_housing = None
                camera_pose_housing_raw = None
                if anchor_camera is not None:
                    pose_camera_housing = pose_multiply(
                        anchor_camera, pose_inverse(self.anchor_estimator.pose_anchor_link_marker)
                    )
                    camera_pose_housing_raw = pose_inverse(pose_camera_housing)
                    last_camera_pose = self._camera_filter.update(camera_pose_housing_raw)
                    if record_plate_once:
                        print("[tracker] anchor marker 0 visible; housing/world frame established")
                        record_plate_once = False

                node9_pose_camera = self.estimator.estimate(observations)
                node9_pose_housing_raw = None
                if node9_pose_camera is not None and pose_camera_housing is not None:
                    node9_pose_housing_raw = pose_multiply(pose_inverse(pose_camera_housing), node9_pose_camera)
                    last_node9_pose = self._node9_filter.update(node9_pose_housing_raw)

                should_write = (frame_index % self.write_every) == 0
                if should_write:
                    self._write_detections(marker_pose_map, housing_identity, last_camera_pose, last_node9_pose, frame_index)
                if self.enable_debug_output and ((frame_index % self.debug_every) == 0):
                    self._write_debug_diagnostics(
                        frame_index=frame_index,
                        marker_pose_map=marker_pose_map,
                        anchor_pose_camera_raw=anchor_camera,
                        anchor_pose_camera_candidates=anchor_candidate_debug,
                        anchor_pose_camera_selected_index=anchor_selected_index,
                        camera_pose_housing_raw=camera_pose_housing_raw,
                        camera_pose_housing_filtered=last_camera_pose,
                        node9_pose_camera_raw=node9_pose_camera,
                        node9_pose_housing_raw=node9_pose_housing_raw,
                        node9_pose_housing_filtered=last_node9_pose,
                    )

                if self.show and debug is not None:
                    self._draw_status(
                        debug,
                        anchor_visible=(anchor_camera is not None),
                        camera_pose=last_camera_pose,
                        node9_pose=last_node9_pose,
                        frame_index=frame_index,
                    )
                    show_image = debug
                    now = time.time()
                    if self._show_interval_s > 0.0 and (now - self._last_show_wall) < self._show_interval_s:
                        frame_index += 1
                        continue
                    self._last_show_wall = now
                    show_image = self._prepare_show_image(show_image)
                    cv2.imshow("aruco_tracker", show_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                frame_index += 1
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    def _draw_status(
        self,
        image: np.ndarray,
        anchor_visible: bool,
        camera_pose: Optional[Pose],
        node9_pose: Optional[Pose],
        frame_index: int,
    ) -> None:
        lines = [
            f"frame={frame_index}",
            f"anchor0={'yes' if anchor_visible else 'no'}",
            f"camera={'yes' if camera_pose is not None else 'no'}",
            f"node9={'yes' if node9_pose is not None else 'no'}",
            "q/esc: quit",
        ]
        if camera_pose is not None:
            lines.append(
                "camera p=(%.3f, %.3f, %.3f)m" % (camera_pose.p[0], camera_pose.p[1], camera_pose.p[2])
            )
        if node9_pose is not None:
            lines.append(
                "node9 p=(%.3f, %.3f, %.3f)m" % (node9_pose.p[0], node9_pose.p[1], node9_pose.p[2])
            )
        y = 24
        for line in lines:
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)
            y += 24


def main() -> None:
    parser = argparse.ArgumentParser(description="Track housing/node9 ArUco markers and infer moving camera pose.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS_PATH)
    parser.add_argument("--debug-json", type=Path, default=DEFAULT_DEBUG_PATH)
    parser.add_argument("--debug-trace", type=Path, default=DEFAULT_DEBUG_TRACE_PATH)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL_PATH)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--detect-scale", type=float, default=1.0, help="Scale factor for ArUco detection image.")
    parser.add_argument("--enable-debug-output", action="store_true", help="Write debug JSON/trace files.")
    parser.add_argument("--debug-every", type=int, default=10, help="Write debug files every N frames.")
    parser.add_argument(
        "--show",
        choices=("high", "low"),
        help="Show preview. 'high' keeps full-resolution preview; 'low' shows 240p preview at about 10 FPS.",
    )
    parser.add_argument(
        "--record-plate-once",
        action="store_true",
        help="Print a one-time message when anchor marker 0 first establishes the housing/world frame.",
    )
    args = parser.parse_args()

    runtime_dir = args.runtime_dir
    detections = args.detections if args.detections != DEFAULT_DETECTIONS_PATH else runtime_dir / "latest_detections.json"
    debug_json = args.debug_json if args.debug_json != DEFAULT_DEBUG_PATH else runtime_dir / "debug_diagnostics.json"
    debug_trace = args.debug_trace if args.debug_trace != DEFAULT_DEBUG_TRACE_PATH else runtime_dir / "debug_trace.jsonl"

    tracker = RealSenseArucoTracker(
        layout_path=args.layout,
        detections_path=detections,
        debug_path=debug_json,
        debug_trace_path=debug_trace,
        width=args.width,
        height=args.height,
        fps=args.fps,
        show_mode=args.show,
        detect_scale=args.detect_scale,
        enable_debug_output=args.enable_debug_output,
        debug_every=args.debug_every,
        control_path=args.control,
    )
    tracker.run(record_plate_once=bool(args.record_plate_once))


if __name__ == "__main__":
    main()
