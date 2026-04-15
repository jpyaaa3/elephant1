#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import zmq
import genesis as gs
from genesis.utils import geom as gs_geom

import engine.protocol as proto
import builder.json_builder as assembly_builder
from engine.config_loader import (
    HardwareConfig,
    IkConfig,
    JointLimit,
    SimConfig,
    SimParam,
    SpawnConfig,
    UrdfExportConfig,
    load_app_config_from_ini,
)
from engine.motor import estimate_ideal_sim_rates
from builder.urdf_converter import convert_manifest_file


def _to_numpy_1d(raw) -> np.ndarray:
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    return np.array(raw, dtype=float).reshape(-1)


def _as_single_dof_index(raw_idx) -> int:
    if isinstance(raw_idx, (list, tuple, np.ndarray)):
        arr = np.array(raw_idx).reshape(-1)
        if arr.size <= 0:
            raise ValueError("empty dof index list")
        return int(arr[0])
    return int(raw_idx)

@dataclass
class JointLayout:
    base_joint_name: str = "base_prismatic_x"
    roll_joint_name: str = "base_roll_x"
    bend_joint_names: List[str] = field(default_factory=list)
    base_axis_sign: float = 1.0
    roll_axis_sign: float = 1.0
    bend_axis_sign: float = -1.0
    chain_origin_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    tip_link_name: str = ""
    tip_local_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    control_mode: str = "commanded"
    part_control_mode: Dict[str, str] = field(default_factory=dict)
    part_pose_root: Dict[str, np.ndarray] = field(default_factory=dict)
    fk_root_link: str = "plate"
    fk_joint_chain: List[Dict[str, object]] = field(default_factory=list)
    no_clip_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class MarkerSet:
    _target_marker: object = None
    _ik_target_marker: object = None
    _sim_tip_marker: object = None
    _target_marker_pos: Optional[np.ndarray] = None
    _ik_target_marker_pos: Optional[np.ndarray] = None
    _sim_tip_marker_pos: Optional[np.ndarray] = None

    def draw(self, scene, attr_name: str, pos: np.ndarray, color) -> None:
        pos_arr = np.asarray(pos, dtype=float).reshape(3)
        prev_pos = getattr(self, f"{attr_name}_pos", None)
        marker = getattr(self, attr_name)
        if marker is not None and prev_pos is not None and np.array_equal(prev_pos, pos_arr):
            return
        if marker is not None:
            try:
                scene.clear_debug_object(marker)
            except Exception:
                pass
        setattr(self, attr_name, scene.draw_debug_sphere(pos=pos_arr, radius=0.012, color=color))
        setattr(self, f"{attr_name}_pos", pos_arr.copy())


@dataclass
class SimScene:
    scene: object = None
    mover: Optional["SimMover"] = None
    n_nodes: int = 0
    n_seg: int = 0

    @staticmethod
    def _to_numpy_1d(raw) -> np.ndarray:
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        return np.asarray(raw, dtype=float).reshape(-1)

    def draw_marker(self, markers: MarkerSet, attr_name: str, pos: np.ndarray, color) -> None:
        if self.scene is None:
            return
        markers.draw(self.scene, attr_name, pos, color)

    def actual_tip_world(self, layout: JointLayout) -> Optional[np.ndarray]:
        if self.mover is None or not layout.tip_link_name:
            return None
        try:
            link = self.mover.entity.get_link(layout.tip_link_name)
            p = self._to_numpy_1d(link.get_pos())[:3]
            q_wxyz = self._to_numpy_1d(link.get_quat())[:4]
            local = np.asarray(layout.tip_local_offset, dtype=float).reshape(3)
            tip = gs_geom.transform_by_trans_quat(local, p, q_wxyz)
            return np.array(tip, dtype=float)
        except Exception:
            return None

    def desired_tip_pos_from_cmd_target(
        self,
        layout: JointLayout,
        model: SpawnConfig,
        q_target_full: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.mover is None or not layout.tip_link_name or not layout.fk_joint_chain:
            return None
        try:
            q_vals = np.asarray(q_target_full, dtype=float).reshape(-1)
            q_map = {name: float(q_vals[i]) for i, name in enumerate(self.mover.dof_names()) if i < q_vals.size}

            spawn_pos = np.array(model.spawn_xyz, dtype=float).reshape(3)
            spawn_euler = np.array(model.spawn_euler_deg, dtype=float).reshape(3)
            R_spawn = Rot.from_euler("xyz", spawn_euler, degrees=True).as_matrix()

            link_tf: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            root = layout.fk_root_link
            p_root_local = layout.part_pose_root.get(root, np.array([0.0, 0.0, 0.0], dtype=float))
            link_tf[root] = (spawn_pos + R_spawn @ p_root_local, R_spawn.copy())

            for meta in layout.fk_joint_chain:
                parent = str(meta["parent"])
                child = str(meta["child"])
                if parent not in link_tf:
                    continue
                p_parent, R_parent = link_tf[parent]
                origin_parent = np.asarray(meta["origin_parent"], dtype=float).reshape(3)
                axis_parent = np.asarray(meta["axis_parent"], dtype=float).reshape(3)
                q = float(q_map.get(str(meta["name"]), 0.0))
                if str(meta["type"]) == "prismatic":
                    p_child = p_parent + R_parent @ (origin_parent + axis_parent * q)
                    R_child = R_parent
                elif str(meta["type"]) == "revolute":
                    p_child = p_parent + R_parent @ origin_parent
                    R_child = R_parent @ Rot.from_rotvec(axis_parent * q).as_matrix()
                else:
                    p_child = p_parent + R_parent @ origin_parent
                    R_child = R_parent
                link_tf[child] = (p_child, R_child)

            if layout.tip_link_name not in link_tf:
                return None
            p_tip, R_tip = link_tf[layout.tip_link_name]
            tip_world = p_tip + R_tip @ np.asarray(layout.tip_local_offset, dtype=float).reshape(3)
            return np.array(tip_world, dtype=float)
        except Exception:
            return None

    def apply_sim_q(self, q_errmodel: proto.SimQ) -> Optional[np.ndarray]:
        if self.mover is None:
            return None
        self.mover.control_4dof(
            float(q_errmodel.linear_m),
            float(q_errmodel.roll_rad),
            float(q_errmodel.theta1_rad),
            float(q_errmodel.theta2_rad),
        )
        return self.mover.target_from_4dof(
            float(q_errmodel.linear_m),
            float(q_errmodel.roll_rad),
            float(q_errmodel.theta1_rad),
            float(q_errmodel.theta2_rad),
        )

    def step(self) -> None:
        if self.scene is not None:
            self.scene.step()


class RateLimiter:
    def __init__(self, max_rate: np.ndarray):
        self._max_rate = np.array(max_rate, dtype=float)

    def step(self, q_cmd: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        dq = q_target - q_cmd
        max_step = self._max_rate * dt
        dq_clamped = np.clip(dq, -max_step, +max_step)
        return q_cmd + dq_clamped


class SimMover:
    def __init__(
        self,
        entity,
        params: SimParam,
        limit: JointLimit,
        n_nodes: int,
        n_seg: Optional[int] = None,
        *,
        base_joint_name: str = "base_prismatic_x",
        roll_joint_name: str = "base_roll_x",
        bend_joint_names: Optional[List[str]] = None,
    ):
        self.entity = entity
        self.p = params
        self.limit = limit
        self.n_nodes = int(n_nodes)
        self.n_seg = int(n_seg) if n_seg is not None else (self.n_nodes // 2)
        self._base_joint_name = str(base_joint_name)
        self._roll_joint_name = str(roll_joint_name)
        self._bend_joint_names = (
            [str(x) for x in bend_joint_names]
            if bend_joint_names is not None
            else [f"bend_{i}" for i in range(self.n_nodes)]
        )

        base_names = [self._base_joint_name, self._roll_joint_name]
        bend_names = list(self._bend_joint_names)

        pairs: List[Tuple[int, str]] = []
        for name in (base_names + bend_names):
            j = self.entity.get_joint(name)
            idx = getattr(j, "dofs_idx_local")
            pairs.append((_as_single_dof_index(idx), name))

        pairs.sort(key=lambda t: int(t[0]))
        self.joint_names: List[str] = [n for _, n in pairs]
        self.dofs_idx_local: List[int] = [int(i) for i, _ in pairs]

        self._name2pos: Dict[str, int] = {n: k for k, n in enumerate(self.joint_names)}
        self._bend_pos: List[int] = [self._name2pos[name] for name in self._bend_joint_names]

        self.bend_lim = float(limit.bend_lim_rad())
        max_rate = np.array(
            [float(params.linear_rate), float(params.roll_rate)] + [float(params.bend_rate)] * self.n_nodes,
            dtype=float,
        )
        self._rate = RateLimiter(max_rate=max_rate)

        try:
            raw0 = self.entity.get_dofs_position(dofs_idx_local=self.dofs_idx_local)
            q0 = _to_numpy_1d(raw0)
            if q0.shape[0] != len(self.dofs_idx_local):
                q0 = q0[: len(self.dofs_idx_local)]
            self._q_cmd = q0.copy()
        except Exception:
            self._q_cmd = np.zeros(len(self.dofs_idx_local), dtype=float)

        self._last_q_target: Optional[np.ndarray] = None
        self._last_q4: Optional[Tuple[float, float, float, float]] = None

    def idx_linear(self) -> Optional[int]:
        return self._name2pos.get(self._base_joint_name, None)

    def idx_roll(self) -> Optional[int]:
        return self._name2pos.get(self._roll_joint_name, None)

    def bend_indices(self) -> List[int]:
        return list(self._bend_pos)

    def dof_names(self) -> List[str]:
        return list(self.joint_names)

    def dof_count(self) -> int:
        return int(len(self.joint_names))

    def get_dofs_position(self) -> np.ndarray:
        raw = self.entity.get_dofs_position(dofs_idx_local=self.dofs_idx_local)
        q = _to_numpy_1d(raw)
        if q.shape[0] != len(self.dofs_idx_local):
            q = q[: len(self.dofs_idx_local)]
        return q

    def get_last_target_full(self) -> Optional[np.ndarray]:
        return None if self._last_q_target is None else self._last_q_target.copy()

    def get_last_command_full(self) -> np.ndarray:
        return self._q_cmd.copy()

    def _apply_q_direct(self, q_target: np.ndarray) -> None:
        try:
            self.entity.set_dofs_position(q_target, dofs_idx_local=self.dofs_idx_local)
        except Exception:
            self.entity.control_dofs_position(q_target, dofs_idx_local=self.dofs_idx_local)

    def target_from_4dof(self, linear: float, roll: float, theta1: float, theta2: float) -> np.ndarray:
        bx = float(np.clip(float(linear), self.limit.linear_min, self.limit.linear_max))
        rl = float(np.clip(float(roll), self.limit.roll_min_rad(), self.limit.roll_max_rad()))
        t1 = float(np.clip(float(theta1), -self.bend_lim, +self.bend_lim))
        t2 = float(np.clip(float(theta2), -self.bend_lim, +self.bend_lim))

        vals: Dict[str, float] = {self._base_joint_name: bx, self._roll_joint_name: rl}
        for i in range(self.n_nodes):
            vals[self._bend_joint_names[i]] = t1 if i < self.n_seg else t2

        return np.array([vals[n] for n in self.joint_names], dtype=float)

    def control_4dof(self, linear: float, roll: float, theta1: float, theta2: float):
        q_target = self.target_from_4dof(linear, roll, theta1, theta2)
        self._last_q_target = q_target
        self._last_q4 = (float(linear), float(roll), float(theta1), float(theta2))
        self._q_cmd = self._rate.step(self._q_cmd, q_target, dt=float(self.p.dt))
        self._apply_q_direct(self._q_cmd)

    def set_4dof_instant(self, linear: float, roll: float, theta1: float, theta2: float) -> None:
        q_target = self.target_from_4dof(linear, roll, theta1, theta2)
        self._last_q_target = q_target
        self._last_q4 = (float(linear), float(roll), float(theta1), float(theta2))
        self._q_cmd = q_target.copy()
        self._apply_q_direct(q_target)


class AssetProcessor:
    """Orchestrate asset prep: ensure manifest json exists, then convert JSON to URDF."""

    def __init__(self, app: "GenesisApp"):
        self.app = app

    def _json_path(self) -> str:
        c = self.app.cfg
        return os.path.join(c.build_dir, c.assy_build_json)

    def _urdf_path(self) -> str:
        c = self.app.cfg
        return os.path.join(c.build_dir, c.urdf_name)

    def prepare_assets(self) -> str:
        t0 = time.time()
        in_json = self._json_path()
        out_urdf = self._urdf_path()
        if self.app.cfg.rebuild_assembly or (not os.path.isfile(in_json)):
            os.makedirs(self.app.cfg.build_dir, exist_ok=True)
            try:
                assembly_builder.build_default_manifest(
                    self.app.cfg.build_dir,
                    use_hardware=bool(self.app.cfg.use_hardware),
                    use_go2=bool(getattr(self.app.cfg, "use_go2", False)),
                )
            except Exception as e:
                raise RuntimeError(f"Auto build failed for {self.app.cfg.assy_build_json}: {e}") from e
            if not os.path.isfile(in_json):
                raise FileNotFoundError(f"manifest json not found after auto-build: {in_json}")

        self._load_joint_layout(in_json)
        self.app._apply_ideal_rates_if_needed()
        convert_manifest_file(in_json, out_urdf, cfg=self.app.urdf_export_cfg)
        print(f"[runtime] use_hardware = {str(bool(self.app.cfg.use_hardware)).lower()}")
        print("[runtime] assets prepared in %.2fs" % (time.time() - t0))
        return out_urdf

    def _load_joint_layout(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            build = json.load(f)
        def _pick_manifest_value(mapping, *keys, default=None):
            for key in keys:
                if isinstance(mapping, dict) and key in mapping:
                    return mapping[key]
            return default

        joints = list(_pick_manifest_value(build, "joints", default=[]))
        parts = list(_pick_manifest_value(build, "parts", default=[]))
        raw_pairs = list(_pick_manifest_value(build, "no_clip_pairs", default=[]) or [])
        if not joints:
            raise RuntimeError("manifest json is missing joints")
        if not parts:
            raise RuntimeError("manifest json is missing parts")

        prismatic_names: List[str] = []
        revolute_names: List[str] = []
        for j in joints:
            jname = str(_pick_manifest_value(j, "name", default="")).strip()
            jtype = str(_pick_manifest_value(j, "type", default="")).strip().lower()
            if not jname:
                continue
            if jtype == "prismatic":
                prismatic_names.append(jname)
            elif jtype == "revolute":
                revolute_names.append(jname)

        if not prismatic_names or len(revolute_names) < 2:
            raise RuntimeError("manifest json does not provide enough control joints (need 1 prismatic + >=2 revolute).")

        self.app.layout.base_joint_name = prismatic_names[0]
        self.app.layout.roll_joint_name = revolute_names[0]
        self.app.layout.bend_joint_names = revolute_names[1:]
        joint_by_name = {str(_pick_manifest_value(j, "name", default="")): j for j in joints}
        first_bend = joint_by_name.get(self.app.layout.bend_joint_names[0]) if self.app.layout.bend_joint_names else None
        if first_bend is None:
            raise RuntimeError("manifest json is missing first bend joint metadata")
        ar = _pick_manifest_value(first_bend, "anchor_root", default=None)
        if not isinstance(ar, (list, tuple)) or len(ar) != 3:
            raise RuntimeError("manifest json is missing valid anchor_root for first bend joint")
        self.app.layout.chain_origin_local = np.array([float(ar[0]), float(ar[1]), float(ar[2])], dtype=float)

        tip_link_name = str(_pick_manifest_value(joints[-1], "child", default="")) if joints else ""
        if not tip_link_name:
            raise RuntimeError("manifest json is missing tip link child on last joint")
        tip_local_offset = np.array([0.0, 0.0, 0.0], dtype=float)
        part_control_mode: Dict[str, str] = {}
        controlled_modes: List[str] = []
        for p in parts:
            name = str(_pick_manifest_value(p, "name", default="")).strip()
            flags = _pick_manifest_value(p, "flags", default={}) or {}
            mode = str(_pick_manifest_value(flags, "control_mode", default=_pick_manifest_value(flags, "ControlMode", default="fixed"))).strip().lower() or "fixed"
            if name:
                part_control_mode[name] = mode
            kind = str(_pick_manifest_value(p, "kind", default="")).strip().lower()
            if kind in ("housing", "wedge", "node", "node_end"):
                controlled_modes.append(mode)
        part_by_name = {str(_pick_manifest_value(p, "name", default="")): p for p in parts}
        part = part_by_name.get(tip_link_name)
        if part is None:
            raise RuntimeError(f"manifest json is missing part entry for tip link '{tip_link_name}'")
        assets = _pick_manifest_value(part, "assets", default={}) or {}
        frame_rel = str(_pick_manifest_value(assets, "frame", default="") or "")
        if not frame_rel:
            raise RuntimeError(f"manifest json is missing frame asset for tip part '{tip_link_name}'")
        frame_abs = os.path.join(self.app.cfg.build_dir, frame_rel)
        with open(frame_abs, "r", encoding="utf-8") as ff:
            frame_json = json.load(ff)
        connectors = _pick_manifest_value(frame_json, "connectors", default={}) or {}
        to_raw = _pick_manifest_value(connectors, "to", default=None)
        if not isinstance(to_raw, (list, tuple)) or len(to_raw) != 3:
            raise RuntimeError(f"frame json is missing valid connectors.to for tip part '{tip_link_name}'")
        tip_local_offset = np.array([float(to_raw[0]), float(to_raw[1]), float(to_raw[2])], dtype=float)
        self.app.layout.tip_link_name = tip_link_name
        self.app.layout.tip_local_offset = tip_local_offset
        self.app.layout.part_control_mode = part_control_mode
        if controlled_modes:
            uniq = sorted(set(controlled_modes))
            self.app.layout.control_mode = uniq[0]
            if len(uniq) > 1:
                print(f"[runtime] mixed controlled part modes {uniq}; using chain mode '{self.app.layout.control_mode}'")
        else:
            self.app.layout.control_mode = "commanded"
        no_clip_pairs: List[Tuple[str, str]] = []
        for item in raw_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a0 = str(item[0]).strip()
                a1 = str(item[1]).strip()
                if a0 and a1 and a0 != a1:
                    no_clip_pairs.append((a0, a1))
        self.app.layout.no_clip_pairs = no_clip_pairs

        # IK sign convention: convert actual joint axis (+/-X, +/-Y) into scalar signs.
        def _axis_sign(raw_axis, axis_idx: int, *, name: str) -> float:
            a = np.asarray(raw_axis, dtype=float).reshape(-1)
            if a.size <= axis_idx:
                raise RuntimeError(f"manifest json is missing valid axis_root for {name}")
            v = float(a[axis_idx])
            if abs(v) < 1e-9:
                raise RuntimeError(f"manifest json axis_root for {name} has zero component on required axis")
            return -1.0 if v < 0.0 else 1.0

        roll_meta = joint_by_name.get(self.app.layout.roll_joint_name, {}) if self.app.layout.roll_joint_name else {}
        bend_meta = joint_by_name.get(self.app.layout.bend_joint_names[0], {}) if self.app.layout.bend_joint_names else {}
        base_meta = joint_by_name.get(self.app.layout.base_joint_name, {}) if self.app.layout.base_joint_name else {}
        self.app.layout.base_axis_sign = _axis_sign(_pick_manifest_value(base_meta, "axis_root", default=None), 0, name=self.app.layout.base_joint_name)
        self.app.layout.roll_axis_sign = _axis_sign(_pick_manifest_value(roll_meta, "axis_root", default=None), 0, name=self.app.layout.roll_joint_name)
        self.app.layout.bend_axis_sign = _axis_sign(_pick_manifest_value(bend_meta, "axis_root", default=None), 1, name=self.app.layout.bend_joint_names[0])

        part_pose_root: Dict[str, np.ndarray] = {}
        for p in parts:
            name = str(_pick_manifest_value(p, "name", default=""))
            pose_root = _pick_manifest_value(p, "pose_root", default={}) or {}
            pr = _pick_manifest_value(pose_root, "p", default=[0.0, 0.0, 0.0])
            if isinstance(pr, (list, tuple)) and len(pr) == 3:
                part_pose_root[name] = np.array([float(pr[0]), float(pr[1]), float(pr[2])], dtype=float)
        self.app.layout.part_pose_root = part_pose_root

        parent_of: Dict[str, str] = {}
        for j in joints:
            parent = str(_pick_manifest_value(j, "parent", default=""))
            child = str(_pick_manifest_value(j, "child", default=""))
            if parent and child:
                parent_of[child] = parent
        roots = [name for name in part_pose_root.keys() if name not in parent_of]
        if not roots:
            raise RuntimeError("manifest json does not provide a root link")
        self.app.layout.fk_root_link = roots[0]

        fk_chain = []
        for jn in [self.app.layout.base_joint_name, self.app.layout.roll_joint_name] + list(self.app.layout.bend_joint_names):
            j = joint_by_name.get(jn)
            if j is None:
                continue
            parent = str(_pick_manifest_value(j, "parent", default=""))
            child = str(_pick_manifest_value(j, "child", default=""))
            jtype = str(_pick_manifest_value(j, "type", default="")).strip().lower()
            anchor = _pick_manifest_value(j, "anchor_root", default=[0.0, 0.0, 0.0])
            axis = _pick_manifest_value(j, "axis_root", default=[1.0, 0.0, 0.0])
            p_parent = part_pose_root.get(parent, np.array([0.0, 0.0, 0.0], dtype=float))
            origin_parent = np.array(
                [float(anchor[0]) - float(p_parent[0]), float(anchor[1]) - float(p_parent[1]), float(anchor[2]) - float(p_parent[2])],
                dtype=float,
            )
            axis_parent = np.array([float(axis[0]), float(axis[1]), float(axis[2])], dtype=float)
            n = float(np.linalg.norm(axis_parent))
            if n > 1e-12:
                axis_parent /= n
            fk_chain.append(
                {
                    "name": jn,
                    "type": jtype,
                    "parent": parent,
                    "child": child,
                    "origin_parent": origin_parent,
                    "axis_parent": axis_parent,
                }
            )
        self.app.layout.fk_joint_chain = fk_chain


class StateSource:
    """Abstract source of 4-DOF chain state for the SIM runtime."""

    def poll(self) -> None:
        return None

    def estimate_q(self) -> Optional[proto.SimQ]:
        return None

    def ik_target_xyz(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        return None


class HardwareStateCache(StateSource):
    """
    Direct passthrough cache of the latest host-published state.
    Future IMU/AruCo/camera fusion should implement the same interface.
    """

    def __init__(self) -> None:
        self._last_q: Optional[proto.SimQ] = None
        self._last_ik_target_xyz: Optional[np.ndarray] = None

    def update(self, q: proto.SimQ, ik_target_xyz: Optional[np.ndarray] = None) -> None:
        self._last_q = q
        self._last_ik_target_xyz = None if ik_target_xyz is None else np.array(ik_target_xyz, dtype=float).reshape(3)

    def estimate_q(self) -> Optional[proto.SimQ]:
        return self._last_q

    def ik_target_xyz(self) -> Optional[np.ndarray]:
        return None if self._last_ik_target_xyz is None else self._last_ik_target_xyz.copy()


class HostStateSubscriber:
    """SIM-side subscriber that consumes host state broadcasts."""

    def __init__(self, endpoint: str) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required for sim host subscriber")
        self.endpoint = str(endpoint)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(self.endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)
        self.last_q: Optional[proto.SimQ] = None
        self.last_u: Optional[proto.ControlU] = None
        self.last_torque_enabled: bool = False
        self.last_state_ts: float = 0.0
        self.last_ik_target_xyz: Optional[np.ndarray] = None

    def close(self) -> None:
        try:
            self.poller.unregister(self.sock)
        except Exception:
            pass
        try:
            self.sock.close(0)
        except Exception:
            pass

    def poll(self) -> None:
        try:
            events = dict(self.poller.poll(timeout=0))
        except zmq.ZMQError:
            return
        if self.sock not in events:
            return
        while True:
            try:
                data = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError:
                return
            try:
                msg = proto.loads_msg(data)
            except Exception:
                continue
            if str(msg.get("t", "")).lower() != "state":
                continue
            self.last_state_ts = float(msg.get("ts", time.time()))
            if "q" in msg:
                try:
                    self.last_q = proto.unpack_q(msg["q"])
                except Exception:
                    pass
            if "u" in msg:
                try:
                    self.last_u = proto.unpack_u(msg["u"])
                except Exception:
                    pass
            if "torque_enabled" in msg:
                self.last_torque_enabled = bool(msg.get("torque_enabled", False))
            target_raw = msg.get("ik_target", None)
            if isinstance(target_raw, (list, tuple)) and len(target_raw) == 3:
                self.last_ik_target_xyz = np.array([float(target_raw[0]), float(target_raw[1]), float(target_raw[2])], dtype=float)


class HostStateSource(StateSource):
    """State source backed by host PUB/SUB updates."""

    def __init__(self, endpoint: str) -> None:
        self._sub = HostStateSubscriber(endpoint)
        self._cache = HardwareStateCache()

    def poll(self) -> None:
        self._sub.poll()
        if self._sub.last_q is not None:
            self._cache.update(self._sub.last_q, self._sub.last_ik_target_xyz)

    def estimate_q(self) -> Optional[proto.SimQ]:
        return self._cache.estimate_q()

    def ik_target_xyz(self) -> Optional[np.ndarray]:
        return self._cache.ik_target_xyz()

    def close(self) -> None:
        self._sub.close()


class RuntimePrep:
    """Scene wiring and runtime objects."""

    def __init__(self, app: "GenesisApp"):
        self.app = app
    def _detect_n_nodes(self, entity) -> int:
        a = self.app
        if a.layout.bend_joint_names:
            return len(a.layout.bend_joint_names)

        i = 0
        while True:
            try:
                entity.get_joint(f"bend_{i}")
                i += 1
            except Exception:
                break
        if i <= 0:
            raise RuntimeError("No bend_* joints found in loaded URDF")
        return i

    def _apply_no_clip_pairs(self, entity) -> None:
        a = self.app
        pairs = list(a.layout.no_clip_pairs)
        if not pairs:
            return

        methods = []
        for owner in (entity, a.sim_scene.scene):
            if owner is None:
                continue
            for name in (
                "disable_collision_between_links",
                "disable_collision_pair",
                "set_collision_between_links",
                "set_collision_pair",
                "set_pair_collision",
            ):
                fn = getattr(owner, name, None)
                if callable(fn):
                    methods.append((name, fn))

        applied = 0
        for la, lb in pairs:
            la = str(la)
            lb = str(lb)
            link_a = None
            link_b = None
            try:
                link_a = entity.get_link(la)
                link_b = entity.get_link(lb)
            except Exception:
                pass
            done = False
            for mname, fn in methods:
                patterns = []
                if mname.startswith("disable_"):
                    patterns = [(la, lb), (link_a, link_b)]
                else:
                    patterns = [(la, lb, False), (link_a, link_b, False)]
                for args in patterns:
                    if any(x is None for x in args):
                        continue
                    try:
                        fn(*args)
                        applied += 1
                        done = True
                        break
                    except Exception:
                        continue
                if done:
                    break

        if applied > 0:
            print(f"[Collision] no-clip pairs applied: {applied}/{len(pairs)}")
        else:
            print("[Collision] NoClipPairs present, but runtime collision-pair API was not found.")

    def init_genesis(self, urdf_path: str) -> None:
        a = self.app
        backend = gs.gpu if a.cfg.use_gpu else gs.cpu
        backend_name = "gpu" if a.cfg.use_gpu else "cpu"
        print(f"[runtime] genesis backend requested: {backend_name}")
        try:
            gs.init(backend=backend, logging_level="warning")
        except TypeError:
            gs.init(backend=backend)

        try:
            sim_opts = gs.options.SimOptions(dt=a.params.dt, gravity=a.params.gravity, substeps=int(a.params.substeps))
        except TypeError:
            try:
                sim_opts = gs.options.SimOptions(dt=a.params.dt, gravity=a.params.gravity)
            except TypeError:
                sim_opts = gs.options.SimOptions(dt=a.params.dt)

        bx, by, bz = map(float, a.spawn.spawn_xyz)
        cam_lookat = (bx + 0.25, by, bz)
        cam_pos = (bx + 1.10, by - 1.00, bz + 1.10)

        a.sim_scene.scene = gs.Scene(
            sim_options=sim_opts,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=cam_pos,
                camera_lookat=cam_lookat,
                camera_fov=35,
                max_FPS=60,
            ),
            show_viewer=bool(a.cfg.enable_viewer),
        )

        if a.cfg.floor:
            a.sim_scene.scene.add_entity(gs.morphs.Plane())

        spawn_pos = tuple(float(x) for x in a.spawn.spawn_xyz)
        spawn_euler = tuple(float(x) for x in a.spawn.spawn_euler_deg)
        spawn_q_xyzw = Rot.from_euler("xyz", np.array(spawn_euler, dtype=float), degrees=True).as_quat()
        spawn_q_wxyz = np.array([spawn_q_xyzw[3], spawn_q_xyzw[0], spawn_q_xyzw[1], spawn_q_xyzw[2]], dtype=float)
        morph = None
        try:
            morph = gs.morphs.URDF(
                file=urdf_path,
                pos=spawn_pos,
                euler=spawn_euler,
                fixed=True,
                prioritize_urdf_material=True,
                default_armature=0.0,
                merge_fixed_links=True,
                requires_jac_and_IK=False,
            )
        except TypeError:
            try:
                morph = gs.morphs.URDF(
                    file=urdf_path,
                    pos=spawn_pos,
                    euler=spawn_euler,
                    fixed=True,
                    prioritize_urdf_material=True,
                    default_armature=0.0,
                    merge_fixed_links=False,
                    requires_jac_and_IK=False,
                )
            except TypeError:
                morph = gs.morphs.URDF(
                    file=urdf_path,
                    pos=spawn_pos,
                    euler=spawn_euler,
                    fixed=True,
                    prioritize_urdf_material=True,
                    merge_fixed_links=False,
                    requires_jac_and_IK=False,
                )

        ent = a.sim_scene.scene.add_entity(morph)
        t_build = time.time()
        a.sim_scene.scene.build()
        print("[runtime] scene built in %.2fs" % (time.time() - t_build))

        n_nodes = self._detect_n_nodes(ent)
        n_seg = int(a.spawn.n_seg) if a.spawn.n_seg is not None else max(1, n_nodes // 2)

        a.sim_scene.mover = SimMover(
            ent,
            a.params,
            a.limit,
            n_nodes=n_nodes,
            n_seg=n_seg,
            base_joint_name=a.layout.base_joint_name,
            roll_joint_name=a.layout.roll_joint_name,
            bend_joint_names=a.layout.bend_joint_names,
        )
        a.sim_scene.n_nodes = n_nodes
        a.sim_scene.n_seg = n_seg



class SimRuntime:
    """Main loop: protocol sync, IK, control, debug markers."""

    def __init__(self, app: "GenesisApp"):
        self.app = app

    def _poll_host_and_update_model(self) -> None:
        a = self.app
        if a.state_source is not None:
            a.state_source.poll()

    def _cleanup(self) -> None:
        a = self.app
        if a.state_source is not None:
            a.state_source.close()

    def run(self) -> None:
        a = self.app
        assert a.sim_scene.scene is not None and a.sim_scene.mover is not None

        try:
            while True:
                self._poll_host_and_update_model()
                ik_target = a.state_source.ik_target_xyz() if a.state_source is not None else None
                if ik_target is not None and a.spawn.draw_debug_markers:
                    a.sim_scene.draw_marker(a.markers, "_ik_target_marker", ik_target, (1.0, 0.0, 0.0, 0.9))
                q_errmodel = a._errmodel_q() if a._has_state_source() else None
                if q_errmodel is not None:
                    q_des_full = a.sim_scene.apply_sim_q(q_errmodel)
                    if a.spawn.draw_debug_markers:
                        desired_tip = a.sim_scene.desired_tip_pos_from_cmd_target(a.layout, a.spawn, q_des_full)
                        sim_tip = a.sim_scene.actual_tip_world(a.layout)
                        if desired_tip is not None:
                            a.sim_scene.draw_marker(a.markers, "_target_marker", desired_tip, (0.0, 0.2, 1.0, 0.9))
                        if sim_tip is not None:
                            a.sim_scene.draw_marker(a.markers, "_sim_tip_marker", sim_tip, (0.0, 1.0, 0.0, 0.9))

                a.sim_scene.step()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()


class GenesisApp:
    """Thin orchestrator over asset/runtime/control components."""

    def __init__(
        self,
        params: Optional[SimParam] = None,
        cfg: Optional[SimConfig] = None,
        limit: Optional[JointLimit] = None,
        model: Optional[SpawnConfig] = None,
        *,
        urdf_export_cfg: Optional[UrdfExportConfig] = None,
        ik_cfg: Optional[IkConfig] = None,
        mapping_cfg: Optional[proto.SimMappingConfig] = None,
        endpoint: Optional[str] = None,
        enable_link: Optional[bool] = None,
        hardware_cfg: Optional[HardwareConfig] = None,
    ):
        self.params = params if params is not None else SimParam()
        self.cfg = cfg if cfg is not None else SimConfig()
        self.limit = limit if limit is not None else JointLimit(
            linear_min=-0.23,
            linear_max=0.01,
            roll_min_deg=-90.0,
            roll_max_deg=90.0,
            bend_deg=36.0,
        )
        self.spawn = model if model is not None else SpawnConfig()
        self.urdf_export_cfg = urdf_export_cfg if urdf_export_cfg is not None else UrdfExportConfig()
        self.ik_cfg = ik_cfg if ik_cfg is not None else IkConfig()
        self.hardware_cfg = hardware_cfg if hardware_cfg is not None else HardwareConfig()

        self._proto_cfg = mapping_cfg if mapping_cfg is not None else proto.SimMappingConfig()
        host_state_endpoint = str(self.cfg.host_sim_port).strip()

        self.layout = JointLayout()
        self.markers = MarkerSet()
        self.sim_scene = SimScene()
        self.state_source: Optional[StateSource] = HostStateSource(host_state_endpoint) if host_state_endpoint else None

    def _apply_ideal_rates_if_needed(self) -> None:
        if self.layout.control_mode != "commanded":
            return
        linear_rate = float(self.params.linear_rate)
        roll_rate = float(self.params.roll_rate)
        bend_rate = float(self.params.bend_rate)
        if np.isfinite(linear_rate) and np.isfinite(roll_rate) and np.isfinite(bend_rate):
            return
        try:
            est_linear, est_roll, est_bend = estimate_ideal_sim_rates(self._proto_cfg)
        except Exception:
            return
        self.params = replace(
            self.params,
            linear_rate=linear_rate if np.isfinite(linear_rate) else float(est_linear),
            roll_rate=roll_rate if np.isfinite(roll_rate) else float(est_roll),
            bend_rate=bend_rate if np.isfinite(bend_rate) else float(est_bend),
        )
        print(
            "[runtime] commanded rates matched to hardware profiles: "
            "linear=%.4f m/s roll=%.3f rad/s bend=%.3f rad/s"
            % (float(self.params.linear_rate), float(self.params.roll_rate), float(self.params.bend_rate))
        )

    def _has_state_source(self) -> bool:
        return bool(self.state_source is not None)

    def _errmodel_q(self) -> Optional[proto.SimQ]:
        if self.state_source is None:
            return None
        try:
            return self.state_source.estimate_q()
        except Exception:
            return None

    def run(self) -> None:
        urdf_path = AssetProcessor(self).prepare_assets()
        runtime = RuntimePrep(self)
        runtime.init_genesis(urdf_path)
        SimRuntime(self).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.ini"),
        help="path to ini config file",
    )
    args = ap.parse_args()

    bundle = load_app_config_from_ini(args.config)
    app = GenesisApp(
        params=bundle.sim_param,
        cfg=bundle.sim_config,
        hardware_cfg=bundle.hardware_config,
        limit=bundle.joint_limit,
        model=bundle.spawn_config,
        urdf_export_cfg=bundle.urdf_export_config,
        ik_cfg=bundle.ik_config,
        mapping_cfg=bundle.mapping_config,
    )
    app.run()


if __name__ == "__main__":
    main()
