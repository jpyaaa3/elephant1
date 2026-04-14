#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import genesis as gs
from genesis.utils import geom as gs_geom

from engine import protocol as proto
import assembly.builder as assembly_builder
from bridge import LinkClient
from robot_ui import ControlState, ImGuiController
from engine.config_loader import (
    AppModelConfig,
    HardwareConfig,
    IkConfig,
    JointLimit,
    SimConfig,
    SimParam,
    UrdfExportConfig,
    load_app_config_from_ini,
)
from engine.ik_solver import (
    Kinematics,
    Mover,
    is_tracking_full,
    TaskSpaceRefiner,
    TaskSpaceRefineConfig,
)
from engine.ik_new import (
    GoalSpec as IKNewGoalSpec,
    IKNewPipeline,
    Kinematics as IKNewKinematics,
    LinearJointPathPlanner,
    MultiSeedConfig as IKNewMultiSeedConfig,
    MultiSeedGoalFinder,
    make_default_search_bounds as ik_new_default_bounds,
)
from engine.motor import estimate_ideal_sim_rates
from assembly.urdf_converter import JSON2URDF


class AssetPipeline:
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
                assembly_builder.build_default(
                    self.app.cfg.build_dir,
                    use_hardware=bool(self.app.cfg.use_hardware),
                    use_go2=bool(getattr(self.app.cfg, "use_go2", False)),
                )
            except Exception as e:
                raise RuntimeError(f"Auto build failed for {self.app.cfg.assy_build_json}: {e}") from e
            if not os.path.isfile(in_json):
                raise FileNotFoundError(f"manifest json not found after auto-build: {in_json}")

        self._load_joint_layout(in_json)
        self.app._apply_hardware_matched_ideal_rates_if_needed()
        conv = JSON2URDF(self.app.urdf_export_cfg)
        conv.convert_file(in_json, out_urdf)
        print(
            "[runtime] assets prepared in %.2fs (rebuild_assembly=%s)"
            % (time.time() - t0, bool(self.app.cfg.rebuild_assembly))
        )
        return out_urdf

    def _load_joint_layout(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            build = json.load(f)
        def _pick(mapping, *keys, default=None):
            for key in keys:
                if isinstance(mapping, dict) and key in mapping:
                    return mapping[key]
            return default

        joints = list(_pick(build, "joints", default=[]))
        parts = list(_pick(build, "parts", default=[]))
        raw_pairs = list(_pick(build, "no_clip_pairs", default=[]) or [])

        prismatic_names: List[str] = []
        revolute_names: List[str] = []
        for j in joints:
            jname = str(_pick(j, "name", default="")).strip()
            jtype = str(_pick(j, "type", default="")).strip().lower()
            if not jname:
                continue
            if jtype == "prismatic":
                prismatic_names.append(jname)
            elif jtype == "revolute":
                revolute_names.append(jname)

        if not prismatic_names or len(revolute_names) < 2:
            raise RuntimeError("manifest json does not provide enough control joints (need 1 prismatic + >=2 revolute).")

        self.app._base_joint_name = prismatic_names[0]
        self.app._roll_joint_name = revolute_names[0]
        self.app._bend_joint_names = revolute_names[1:]
        joint_by_name = {str(_pick(j, "name", default="")): j for j in joints}
        first_bend = joint_by_name.get(self.app._bend_joint_names[0]) if self.app._bend_joint_names else None
        if first_bend is not None:
            ar = _pick(first_bend, "anchor_root", default=[0.0, 0.0, 0.0])
            self.app._chain_origin_local = np.array([float(ar[0]), float(ar[1]), float(ar[2])], dtype=float)
        else:
            self.app._chain_origin_local = np.array([0.0, 0.0, 0.0], dtype=float)

        tip_link_name = str(_pick(joints[-1], "child", default="")) if joints else ""
        tip_local_offset = np.array([0.0, 0.0, 0.0], dtype=float)
        part_control_mode: Dict[str, str] = {}
        controlled_modes: List[str] = []
        for p in parts:
            name = str(_pick(p, "name", default="")).strip()
            flags = _pick(p, "flags", default={}) or {}
            mode = str(_pick(flags, "ControlMode", default="fixed")).strip().lower() or "fixed"
            if name:
                part_control_mode[name] = mode
            kind = str(_pick(p, "kind", default="")).strip().lower()
            if kind in ("housing", "wedge", "node", "node_end"):
                controlled_modes.append(mode)
        if tip_link_name and parts:
            part_by_name = {str(_pick(p, "name", default="")): p for p in parts}
            part = part_by_name.get(tip_link_name)
            if part is not None:
                assets = _pick(part, "assets", default={}) or {}
                frame_rel = str(_pick(assets, "frame", default="") or "")
                if frame_rel:
                    frame_abs = os.path.join(self.app.cfg.build_dir, frame_rel)
                    try:
                        with open(frame_abs, "r", encoding="utf-8") as ff:
                            frame_json = json.load(ff)
                        connectors = _pick(frame_json, "connectors", default={}) or {}
                        to_raw = _pick(connectors, "to", default=None)
                        if isinstance(to_raw, (list, tuple)) and len(to_raw) == 3:
                            tip_local_offset = np.array([float(to_raw[0]), float(to_raw[1]), float(to_raw[2])], dtype=float)
                    except Exception:
                        pass
        self.app._tip_link_name = tip_link_name
        self.app._tip_local_offset = tip_local_offset
        self.app._part_control_mode = part_control_mode
        if controlled_modes:
            uniq = sorted(set(controlled_modes))
            self.app._chain_control_mode = uniq[0]
            if len(uniq) > 1:
                print(f"[runtime] mixed controlled part modes {uniq}; using chain mode '{self.app._chain_control_mode}'")
        else:
            self.app._chain_control_mode = "commanded"
        print(f"[runtime] chain control mode: {self.app._chain_control_mode}")
        no_clip_pairs: List[Tuple[str, str]] = []
        for item in raw_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a0 = str(item[0]).strip()
                a1 = str(item[1]).strip()
                if a0 and a1 and a0 != a1:
                    no_clip_pairs.append((a0, a1))
        self.app._no_clip_pairs = no_clip_pairs

        # IK sign convention: convert actual joint axis (+/-X, +/-Y) into scalar signs.
        def _axis_sign(raw_axis, axis_idx: int) -> float:
            try:
                a = np.asarray(raw_axis, dtype=float).reshape(-1)
                if a.size <= axis_idx:
                    return 1.0
                v = float(a[axis_idx])
                if abs(v) < 1e-9:
                    return 1.0
                return -1.0 if v < 0.0 else 1.0
            except Exception:
                return 1.0

        roll_meta = joint_by_name.get(self.app._roll_joint_name, {}) if self.app._roll_joint_name else {}
        bend_meta = joint_by_name.get(self.app._bend_joint_names[0], {}) if self.app._bend_joint_names else {}
        base_meta = joint_by_name.get(self.app._base_joint_name, {}) if self.app._base_joint_name else {}
        self.app._base_axis_sign = _axis_sign(_pick(base_meta, "axis_root", default=[1.0, 0.0, 0.0]), 0)
        self.app._roll_axis_sign = _axis_sign(_pick(roll_meta, "axis_root", default=[1.0, 0.0, 0.0]), 0)
        self.app._bend_axis_sign = _axis_sign(_pick(bend_meta, "axis_root", default=[0.0, 1.0, 0.0]), 1)

        part_pose_root: Dict[str, np.ndarray] = {}
        for p in parts:
            name = str(_pick(p, "name", default=""))
            pose_root = _pick(p, "pose_root", default={}) or {}
            pr = _pick(pose_root, "p", default=[0.0, 0.0, 0.0])
            if isinstance(pr, (list, tuple)) and len(pr) == 3:
                part_pose_root[name] = np.array([float(pr[0]), float(pr[1]), float(pr[2])], dtype=float)
        self.app._part_pose_root = part_pose_root

        parent_of: Dict[str, str] = {}
        for j in joints:
            parent = str(_pick(j, "parent", default=""))
            child = str(_pick(j, "child", default=""))
            if parent and child:
                parent_of[child] = parent
        roots = [name for name in part_pose_root.keys() if name not in parent_of]
        self.app._fk_root_link = roots[0] if roots else "plate"

        fk_chain = []
        for jn in [self.app._base_joint_name, self.app._roll_joint_name] + list(self.app._bend_joint_names):
            j = joint_by_name.get(jn)
            if j is None:
                continue
            parent = str(_pick(j, "parent", default=""))
            child = str(_pick(j, "child", default=""))
            jtype = str(_pick(j, "type", default="")).strip().lower()
            anchor = _pick(j, "anchor_root", default=[0.0, 0.0, 0.0])
            axis = _pick(j, "axis_root", default=[1.0, 0.0, 0.0])
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
        self.app._fk_joint_chain = fk_chain


class ChainStateModel:
    """Model path for simulated chain state injection."""

    def on_link_state(self, q: proto.SimQ) -> None:
        pass

    def estimate_q(self) -> Optional[proto.SimQ]:
        return None


class DirectHardwareModel(ChainStateModel):
    """
    Placeholder model: direct passthrough from hardware state.
    Future IMU/AruCo/camera fusion should implement the same interface.
    """

    def __init__(self) -> None:
        self._last_q: Optional[proto.SimQ] = None

    def on_link_state(self, q: proto.SimQ) -> None:
        self._last_q = q

    def estimate_q(self) -> Optional[proto.SimQ]:
        return self._last_q


class IKNewAdapter:
    def __init__(self, app: "GenesisApp", target_xyz: np.ndarray, q_seed: np.ndarray, ik_cfg: IkConfig):
        self._app = app
        self.cfg = ik_cfg
        self.target_xyz = np.asarray(target_xyz, dtype=float).reshape(3).copy()
        self.q = np.asarray(q_seed, dtype=float).reshape(4).copy()
        self.converged = False
        self.failed = False
        self.last_err_norm = float("inf")
        self._solved = False
        self._preferred_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        self._result = None

        kin_new = IKNewKinematics(
            pitch=float(app.model.pitch),
            n_nodes=int(app._n_nodes),
            n_seg=int(app._n_seg),
            origin_xyz=np.asarray(app._chain_origin_local, dtype=float).reshape(3),
            limit=app.limit,
            base_axis_sign=float(app._base_axis_sign),
            roll_axis_sign=float(app._roll_axis_sign),
            bend_axis_sign=float(app._bend_axis_sign),
        )
        goal_finder = MultiSeedGoalFinder(
            kin_new,
            ik_new_default_bounds(kin_new),
            cfg=IKNewMultiSeedConfig(
                max_seeds=16,
                n_random=8,
                seed_rng=0,
                max_iters_per_seed=max(int(ik_cfg.max_iters), 1),
                stall_limit=max(int(ik_cfg.stall_limit), 1),
                fd_eps=float(ik_cfg.fd_eps),
            ),
        )
        self._pipeline = IKNewPipeline(goal_finder=goal_finder, path_planner=LinearJointPathPlanner())

    def step(self, n_iters: int = 10) -> np.ndarray:
        if self._solved:
            return self.q
        self._solved = True
        self._result = self._pipeline.solve(
            q_start=self.q,
            goal=IKNewGoalSpec(
                target_world=self.target_xyz,
                preferred_dir_world=self._preferred_dir if bool(self.cfg.prefer_tip_plus_x) else None,
                position_tol_m=float(self.cfg.tol),
                direction_tol_deg=(float(self.cfg.direction_tol_deg) if bool(self.cfg.prefer_tip_plus_x) else None),
            ),
        )
        best = self._result.goal_search.best if self._result is not None else None
        if best is not None:
            self.q = np.asarray(best.q, dtype=float).reshape(4).copy()
            self.last_err_norm = float(best.position_error_m)
        else:
            self.last_err_norm = float("inf")
        self.converged = bool(self._result is not None and self._result.success)
        self.failed = not self.converged
        return self.q


class RuntimeBootstrap:
    """Scene wiring and runtime objects."""

    def __init__(self, app: "GenesisApp"):
        self.app = app
    def _detect_n_nodes(self, entity) -> int:
        a = self.app
        if a._bend_joint_names:
            return len(a._bend_joint_names)

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
        pairs = list(getattr(a, "_no_clip_pairs", []))
        if not pairs:
            return

        methods = []
        for owner in (entity, a._scene):
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

        bx, by, bz = map(float, a.model.spawn_xyz)
        cam_lookat = (bx + 0.25, by, bz)
        cam_pos = (bx + 1.10, by - 1.00, bz + 1.10)

        a._scene = gs.Scene(
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
            a._scene.add_entity(gs.morphs.Plane())

        spawn_pos = tuple(float(x) for x in a.model.spawn_xyz)
        spawn_euler = tuple(float(x) for x in a.model.spawn_euler_deg)
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

        ent = a._scene.add_entity(morph)
        t_build = time.time()
        a._scene.build()
        print("[runtime] scene built in %.2fs" % (time.time() - t_build))

        n_nodes = self._detect_n_nodes(ent)
        n_seg = int(a.model.n_seg) if a.model.n_seg is not None else max(1, n_nodes // 2)

        a._mover = Mover(
            ent,
            a.params,
            a.limit,
            n_nodes=n_nodes,
            n_seg=n_seg,
            base_joint_name=a._base_joint_name,
            roll_joint_name=a._roll_joint_name,
            bend_joint_names=a._bend_joint_names,
        )
        a._n_nodes = n_nodes
        a._n_seg = n_seg
        chain_origin_world = gs_geom.transform_by_trans_quat(a._chain_origin_local, np.array(spawn_pos, dtype=float), spawn_q_wxyz)
        a._kin = Kinematics(
            pitch=float(a.model.pitch),
            n_nodes=n_nodes,
            n_seg=n_seg,
            origin_xyz=np.array(chain_origin_world, dtype=float),
            limit=a.limit,
            base_axis_sign=float(a._base_axis_sign),
            roll_axis_sign=float(a._roll_axis_sign),
            bend_axis_sign=float(a._bend_axis_sign),
        )

        # Initialize IK target to current physical tip position so UI starts in a reachable frame.
        try:
            if a._tip_link_name:
                tip_link = ent.get_link(a._tip_link_name)
                p = np.asarray(tip_link.get_pos(), dtype=float).reshape(-1)[:3]
                q_wxyz = np.asarray(tip_link.get_quat(), dtype=float).reshape(-1)[:4]
                tip_world = gs_geom.transform_by_trans_quat(np.asarray(a._tip_local_offset, dtype=float).reshape(3), p, q_wxyz)
                a.state.target_x = float(tip_world[0])
                a.state.target_y = float(tip_world[1])
                a.state.target_z = float(tip_world[2])
                print(
                    "[IK] target initialized to current tip: (%.3f, %.3f, %.3f)"
                    % (a.state.target_x, a.state.target_y, a.state.target_z)
                )
        except Exception:
            pass

        a._wait_for_startup_pose_sync(timeout_s=1.0)

    def start_gui_thread(self) -> None:
        a = self.app
        assert a._mover is not None
        a._gui = ImGuiController(
            a.params,
            a._mover,
            a.state,
            link=a._link,
            mapping_cfg=a._proto_cfg,
            use_hardware=bool(a.cfg.use_hardware),
        )
        a._gui_thread = threading.Thread(target=a._gui.run, daemon=True)
        a._gui_thread.start()


class ControlLoopCoordinator:
    """Main loop: protocol sync, IK, control, debug markers."""

    def __init__(self, app: "GenesisApp"):
        self.app = app

    def _unpack_4dof_from_full(self, q_full: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
        a = self.app
        assert a._mover is not None
        q = np.asarray(q_full, dtype=float).reshape(-1)

        idx_bx = a._mover.idx_linear()
        idx_rl = a._mover.idx_roll()
        bends_idx = a._mover.bend_indices()
        bends = q[bends_idx] if len(bends_idx) > 0 else np.array([], dtype=float)

        bx = float(q[idx_bx]) if idx_bx is not None and idx_bx < q.size else 0.0
        rl = float(q[idx_rl]) if idx_rl is not None and idx_rl < q.size else 0.0

        n1 = int(a._n_seg)
        if bends.size == 0:
            t1 = 0.0
            t2 = 0.0
        else:
            t1 = float(np.mean(bends[:n1])) if bends.size >= 1 else 0.0
            t2 = float(np.mean(bends[n1:])) if bends.size > n1 else t1

        return bx, rl, t1, t2, bends

    def _sync_state_from_physics(self, paused: bool) -> Tuple[float, float, float, float]:
        a = self.app
        assert a._mover is not None
        q_phys = a._mover.get_dofs_position()
        bx, rl, t1, t2, _ = self._unpack_4dof_from_full(q_phys)
        a.state.set_all(bx, rl, t1, t2, paused)
        return bx, rl, t1, t2

    def _draw_marker(self, attr_name: str, pos: np.ndarray, color) -> None:
        a = self.app
        if not a.model.draw_debug_markers:
            return
        marker = getattr(a, attr_name)
        if marker is not None:
            try:
                a._scene.clear_debug_object(marker)
            except Exception:
                pass
        setattr(a, attr_name, a._scene.draw_debug_sphere(pos=pos, radius=0.012, color=color))

    @staticmethod
    def _to_numpy_1d(raw) -> np.ndarray:
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        return np.asarray(raw, dtype=float).reshape(-1)

    def _actual_tip_world(self) -> Optional[np.ndarray]:
        a = self.app
        if a._mover is None or not a._tip_link_name:
            return None
        try:
            link = a._mover.entity.get_link(a._tip_link_name)
            p = self._to_numpy_1d(link.get_pos())[:3]
            q_wxyz = self._to_numpy_1d(link.get_quat())[:4]
            local = np.asarray(a._tip_local_offset, dtype=float).reshape(3)
            tip = gs_geom.transform_by_trans_quat(local, p, q_wxyz)
            return np.array(tip, dtype=float)
        except Exception:
            return None

    def _start_ik_solver(self, target_xyz: np.ndarray, q_seed: np.ndarray) -> None:
        a = self.app
        a._ik = IKNewAdapter(a, target_xyz, q_seed=q_seed, ik_cfg=a.ik_cfg)
        a._ik_hold_q = None
        a._ik_traj_active = False
        a._ik_tip_ok_streak = 0
        a._ik_target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
        a._ik_logged_converged = False

    def _desired_tip_world_from_qdes(self, q_des_full: np.ndarray) -> Optional[np.ndarray]:
        a = self.app
        if a._mover is None or not a._tip_link_name or not a._fk_joint_chain:
            return None
        try:
            q_vals = np.asarray(q_des_full, dtype=float).reshape(-1)
            q_map = {name: float(q_vals[i]) for i, name in enumerate(a._mover.dof_names()) if i < q_vals.size}

            spawn_pos = np.array(a.model.spawn_xyz, dtype=float).reshape(3)
            spawn_euler = np.array(a.model.spawn_euler_deg, dtype=float).reshape(3)
            R_spawn = Rot.from_euler("xyz", spawn_euler, degrees=True).as_matrix()

            link_tf: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            root = a._fk_root_link
            p_root_local = a._part_pose_root.get(root, np.array([0.0, 0.0, 0.0], dtype=float))
            link_tf[root] = (spawn_pos + R_spawn @ p_root_local, R_spawn.copy())

            for meta in a._fk_joint_chain:
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

            if a._tip_link_name not in link_tf:
                return None
            p_tip, R_tip = link_tf[a._tip_link_name]
            tip_world = p_tip + R_tip @ np.asarray(a._tip_local_offset, dtype=float).reshape(3)
            return np.array(tip_world, dtype=float)
        except Exception:
            return None

    def _poll_link_and_update_model(self) -> None:
        a = self.app
        if a._link is None:
            return
        a._link.poll()
        current_device = str(getattr(a._link, "last_device", "") or "").strip()
        if current_device != a._startup_pose_device:
            a._startup_pose_device = current_device
            a._startup_pose_synced = False
        if a._link.last_q is None:
            return
        try:
            if a._chain_model is not None:
                a._chain_model.on_link_state(a._link.last_q)
            if not a._startup_pose_synced:
                q_model = a._model_q()
                if q_model is not None:
                    a._apply_startup_pose_from_q(q_model)
        except Exception:
            pass

    def _handle_ik_requests(
        self,
        *,
        linear: float,
        roll: float,
        theta1: float,
        theta2: float,
        paused: bool,
    ) -> None:
        a = self.app
        solve_req, stop_req, target_req = a.state.consume_ik_requests()

        if stop_req:
            if a._ik is not None:
                try:
                    self._sync_state_from_physics(paused)
                except Exception:
                    pass
            a._ik = None
            a._ik_hold_q = None
            a._ik_traj_active = False
            a._ik_tip_ok_streak = 0
            a._ik_target_xyz = None
            a._ik_logged_converged = False
            a.state.set_ik_status(running=False, converged=False, failed=False, err_m=0.0)

        if solve_req:
            q0 = np.array([linear, roll, theta1, theta2], dtype=float)
            a._ik_refiner = TaskSpaceRefiner(target_req, cfg=a._ik_refine_cfg)
            a._ik_refine_iter = 0
            self._start_ik_solver(target_req, q_seed=q0)
            print(
                "[IK] Solve Requested | target_world=(%.3f, %.3f, %.3f) | q0=(%.3f, %.3f, %.3f, %.3f)"
                % (
                    a._ik_target_xyz[0],
                    a._ik_target_xyz[1],
                    a._ik_target_xyz[2],
                    q0[0],
                    q0[1],
                    q0[2],
                    q0[3],
                )
            )
            a.state.set_ik_status(running=True, converged=False, failed=False, err_m=float("inf"))

    def _start_ik_traj(self, q_start: np.ndarray, q_goal: np.ndarray) -> None:
        a = self.app
        q0 = np.asarray(q_start, dtype=float).reshape(4).copy()
        q1 = np.asarray(q_goal, dtype=float).reshape(4).copy()
        dq = np.abs(q1 - q0)
        rates = np.array(
            [
                float(a.params.linear_rate),
                float(a.params.roll_rate),
                float(a.params.bend_rate),
                float(a.params.bend_rate),
            ],
            dtype=float,
        )
        # If rates are unbounded (inf) in config, use conservative defaults so
        # the commanded trajectory is still physically trackable.
        nominal = np.array([0.08, 2.0, 2.2, 2.2], dtype=float)  # [m/s, rad/s, rad/s, rad/s]
        for i in range(4):
            if (not np.isfinite(float(rates[i]))) or (float(rates[i]) <= 1e-9):
                rates[i] = nominal[i]
        t_need = []
        for i in range(4):
            r = float(rates[i])
            if (not np.isfinite(r)) or (r <= 1e-9):
                t_need.append(0.0)
            else:
                t_need.append(float(dq[i]) / r)
        T = float(max(t_need)) if t_need else 0.0
        T = max(T, 0.15)

        a._ik_traj_q0 = q0
        a._ik_traj_q1 = q1
        a._ik_traj_t0 = float(time.time())
        a._ik_traj_T = float(T)
        a._ik_traj_active = True

    def _eval_ik_traj(self) -> np.ndarray:
        a = self.app
        if not a._ik_traj_active:
            return a._ik_traj_q1.copy()
        T = max(float(a._ik_traj_T), 1e-9)
        u = (float(time.time()) - float(a._ik_traj_t0)) / T
        u = float(np.clip(u, 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)  # smoothstep
        q = a._ik_traj_q0 + s * (a._ik_traj_q1 - a._ik_traj_q0)
        if u >= 1.0 - 1e-12:
            a._ik_traj_active = False
            q = a._ik_traj_q1.copy()
        return q

    def _step_ik(
        self,
        *,
        linear: float,
        roll: float,
        theta1: float,
        theta2: float,
        paused: bool,
    ) -> Tuple[float, float, float, float]:
        a = self.app
        if a._ik is None:
            return linear, roll, theta1, theta2

        # While solving, do not feed intermediate IK iterates to motion control.
        # Start applying only after the optimizer has converged.
        q_curr = np.array([linear, roll, theta1, theta2], dtype=float)
        q_cmd = q_curr.copy()

        if not a._ik.failed:
            a._ik.step(n_iters=10)
            if a._ik.converged and (a._ik_hold_q is None):
                a._ik_hold_q = np.asarray(a._ik.q, dtype=float).reshape(4).copy()
                self._start_ik_traj(q_curr, a._ik_hold_q)

        if a._ik.converged and (not a._ik.failed) and (a._ik_hold_q is not None):
            if a._ik_traj_active:
                q_cmd = self._eval_ik_traj()
            else:
                q_cmd = a._ik_hold_q.copy()

        linear, roll, theta1, theta2 = map(float, q_cmd)
        q_sol = a._ik_hold_q if a._ik_hold_q is not None else np.asarray(a._ik.q, dtype=float).reshape(4)
        a.state.set_ik_solution(float(q_sol[0]), float(q_sol[1]), float(q_sol[2]), float(q_sol[3]))

        phys_ok = False
        tip_ok = False
        joint_ok = False
        sim_tip = None
        tip_err_m = float("inf")
        if a._ik.converged and (a._ik_hold_q is not None) and (not a._ik_traj_active):
            q_des_full = a._mover.target_from_4dof(linear, roll, theta1, theta2)
            joint_ok = is_tracking_full(
                a._mover,
                q_des_full,
                tol_linear=2e-5,
                tol_roll=2e-4,
                tol_bend=3e-4,
            )
            target_xyz = a._ik_target_xyz if a._ik_target_xyz is not None else np.array([0.0, 0.0, 0.0], dtype=float)
            sim_tip = self._actual_tip_world()
            if sim_tip is not None:
                tip_err_m = float(np.linalg.norm(np.asarray(sim_tip, dtype=float).reshape(3) - target_xyz))
                tip_ok = bool(tip_err_m <= float(a._ik_tip_tol_m))
            else:
                tip_ok = False

            if joint_ok and tip_ok:
                a._ik_tip_ok_streak += 1
            else:
                a._ik_tip_ok_streak = 0
            phys_ok = bool(a._ik_tip_ok_streak >= int(a._ik_tip_ok_need))
        running = (not a._ik.failed) and (not (a._ik.converged and phys_ok))
        a.state.set_ik_status(
            running=running,
            converged=a._ik.converged,
            failed=a._ik.failed,
            err_m=float(a._ik.last_err_norm),
        )

        if a._ik.converged and (not a._ik_logged_converged) and (not a._ik_traj_active) and phys_ok:
            target_xyz = a._ik_target_xyz if a._ik_target_xyz is not None else np.array([0.0, 0.0, 0.0], dtype=float)
            solved_q = a._ik_hold_q.copy() if a._ik_hold_q is not None else np.array([linear, roll, theta1, theta2], dtype=float)
            err_model_mm = float(a._ik.last_err_norm) * 1000.0
            tip_dir = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
            tip_align = float("nan")
            err_fk_mm = float("nan")
            err_sim_mm = float("nan")
            try:
                _tip_p, tip_dir = a._kin.tip_position_and_forward(solved_q, clamp=True, soft=False)
                pref_dir = getattr(a._ik, "_preferred_dir", np.array([1.0, 0.0, 0.0], dtype=float))
                tip_align = float(a._kin.tip_alignment(solved_q, pref_dir, clamp=True))
            except Exception:
                pass
            try:
                q_des_full = a._mover.target_from_4dof(linear, roll, theta1, theta2)
                desired_tip = self._desired_tip_world_from_qdes(q_des_full)
                if desired_tip is not None:
                    err_fk_mm = float(np.linalg.norm(np.asarray(desired_tip, dtype=float).reshape(3) - target_xyz)) * 1000.0
            except Exception:
                pass
            try:
                sim_tip = self._actual_tip_world()
                if sim_tip is not None:
                    err_sim_mm = float(np.linalg.norm(np.asarray(sim_tip, dtype=float).reshape(3) - target_xyz)) * 1000.0
            except Exception:
                pass
            print("Desired Position = (%.3f, %.3f, %.3f)" % (target_xyz[0], target_xyz[1], target_xyz[2]))
            print("Calculated Pose = (%.3f, %.3f, %.3f, %.3f)" % (solved_q[0], solved_q[1], solved_q[2], solved_q[3]))
            print(
                "err_model = %.3f mm | err_fk = %.3f mm | err_sim = %.3f mm | "
                "tip_dir = (%.3f, %.3f, %.3f) | tip_align = %.3f"
                % (err_model_mm, err_fk_mm, err_sim_mm, float(tip_dir[0]), float(tip_dir[1]), float(tip_dir[2]), tip_align)
            )
            a._ik_logged_converged = True

        needs_refine = (
            a._ik.converged
            and (not a._ik.failed)
            and (a._ik_hold_q is not None)
            and (not a._ik_traj_active)
            and joint_ok
            and (not tip_ok)
            and (sim_tip is not None)
            and (a._ik_refiner is not None)
            and (a._ik_refine_iter < int(a._ik_refine_max_iters))
        )
        if needs_refine:
            q_phys = a._mover.get_dofs_position()
            linear_phys, roll_phys, theta1_phys, theta2_phys, _ = self._unpack_4dof_from_full(q_phys)
            q_seed = np.array([linear_phys, roll_phys, theta1_phys, theta2_phys], dtype=float)
            next_target = a._ik_refiner.next_target(sim_tip)
            a._ik_refine_iter += 1
            print(
                "[IK] refine %d/%d | tip_err=%.3f mm | next_target=(%.3f, %.3f, %.3f)"
                % (
                    int(a._ik_refine_iter),
                    int(a._ik_refine_max_iters),
                    float(tip_err_m) * 1000.0,
                    float(next_target[0]),
                    float(next_target[1]),
                    float(next_target[2]),
                )
            )
            self._start_ik_solver(next_target, q_seed=q_seed)
            a.state.set_ik_status(running=True, converged=False, failed=False, err_m=float(tip_err_m))
            return linear_phys, roll_phys, theta1_phys, theta2_phys

        if a._ik.failed:
            target_xyz = a._ik_target_xyz if a._ik_target_xyz is not None else np.array([0.0, 0.0, 0.0], dtype=float)
            tip_dir = np.array([float("nan"), float("nan"), float("nan")], dtype=float)
            tip_align = float("nan")
            try:
                _tip_p, tip_dir = a._kin.tip_position_and_forward(q_sol, clamp=True, soft=False)
                pref_dir = getattr(a._ik, "_preferred_dir", np.array([1.0, 0.0, 0.0], dtype=float))
                tip_align = float(a._kin.tip_alignment(q_sol, pref_dir, clamp=True))
            except Exception:
                pass
            print(
                "[IK] Failed | target_world=(%.3f, %.3f, %.3f) | best_err=%.3f mm | "
                "q=(%.3f, %.3f, %.3f, %.3f) | tip_dir=(%.3f, %.3f, %.3f) | tip_align=%.3f"
                % (
                    target_xyz[0],
                    target_xyz[1],
                    target_xyz[2],
                    float(a._ik.last_err_norm) * 1000.0,
                    float(q_sol[0]),
                    float(q_sol[1]),
                    float(q_sol[2]),
                    float(q_sol[3]),
                    float(tip_dir[0]),
                    float(tip_dir[1]),
                    float(tip_dir[2]),
                    tip_align,
                )
            )

        if a._ik.failed or (a._ik.converged and phys_ok):
            try:
                linear, roll, theta1, theta2 = self._sync_state_from_physics(paused)
            except Exception:
                pass
            a._ik = None
            a._ik_hold_q = None
            a._ik_traj_active = False
            a._ik_tip_ok_streak = 0
            a._ik_target_xyz = None
            a._ik_logged_converged = False
            a._ik_refiner = None
            a._ik_refine_iter = 0

        return linear, roll, theta1, theta2

    def _update_debug_metrics(
        self,
        *,
        linear: float,
        roll: float,
        theta1: float,
        theta2: float,
        target: np.ndarray,
    ) -> None:
        a = self.app
        try:
            q_des_full = a._mover.target_from_4dof(linear, roll, theta1, theta2)
            desired_tip = self._desired_tip_world_from_qdes(q_des_full)
            if desired_tip is None:
                bx_des, roll_des, _t1_des, _t2_des, bends_des = self._unpack_4dof_from_full(q_des_full)
                desired_tip = a._kin.tip_from_bends(bx_des, roll_des, bends_des)

            q_full = a._mover.get_dofs_position()
            bx_sim, roll_sim, _t1_sim, _t2_sim, bends_sim = self._unpack_4dof_from_full(q_full)
            sim_tip_actual = self._actual_tip_world()
            sim_tip = sim_tip_actual if sim_tip_actual is not None else a._kin.tip_from_bends(bx_sim, roll_sim, bends_sim)
            sim_tip_err = float(np.linalg.norm(sim_tip - target))

            dq = q_full - q_des_full
            idx_bx = a._mover.idx_linear()
            idx_rl = a._mover.idx_roll()
            bx_err = float(dq[idx_bx]) if idx_bx is not None else 0.0
            roll_err = float(dq[idx_rl]) if idx_rl is not None else 0.0

            bend_idx = a._mover.bend_indices()
            bend_dq = dq[bend_idx] if len(bend_idx) > 0 else np.array([], dtype=float)
            n1 = int(a._n_seg)
            t1_err = float(np.mean(bend_dq[:n1])) if bend_dq.size >= 1 else 0.0
            t2_err = float(np.mean(bend_dq[n1:])) if bend_dq.size > n1 else 0.0
            bend_max_err = float(np.max(np.abs(bend_dq))) if bend_dq.size > 0 else 0.0

            a.state.set_ik_debug(
                sim_tip_err_m=sim_tip_err,
                linear_err_m=bx_err,
                roll_err_rad=roll_err,
                theta1_err_rad=t1_err,
                theta2_err_rad=t2_err,
                bend_max_err_rad=bend_max_err,
            )

            if a.model.draw_debug_markers:
                self._draw_marker("_des_tip_marker", desired_tip, (0.0, 0.2, 1.0, 0.9))
                self._draw_marker("_sim_tip_marker", sim_tip, (0.0, 1.0, 0.0, 0.9))
        except Exception:
            pass

    def _send_link_target(self, *, linear: float, roll: float, theta1: float, theta2: float, paused: bool) -> None:
        a = self.app
        if a._link is None or paused:
            return
        if a._startup_sync_required() and (not a._startup_pose_synced):
            return
        try:
            q_cmd = proto.SimQ(
                linear_m=float(linear),
                roll_rad=float(roll),
                theta1_rad=float(theta1),
                theta2_rad=float(theta2),
            )
            src = "ik" if (a._ik is not None) else "slider"
            a._link.maybe_send_target_q(q_cmd, source=src)
        except Exception:
            pass

    def _cleanup(self) -> None:
        a = self.app
        if a._link is not None:
            try:
                a._link.close()
            except Exception:
                pass
        if a._gui is not None:
            a._gui.stop()

    def run(self) -> None:
        a = self.app
        assert a._scene is not None and a._mover is not None and a._kin is not None

        _, _, _, _, _, txyz = a.state.snapshot()
        target = np.array(txyz, dtype=float)
        if a.model.draw_debug_markers:
            a._target_marker = a._scene.draw_debug_sphere(
                pos=target,
                radius=0.012,
                color=(1.0, 0.0, 0.0, 0.9),
            )
        last_target = target.copy()

        try:
            while True:
                self._poll_link_and_update_model()

                linear, roll, theta1, theta2, paused, (tx, ty, tz) = a.state.snapshot()
                target = np.array([tx, ty, tz], dtype=float)

                if np.linalg.norm(target - last_target) > 1e-9:
                    if a.model.draw_debug_markers:
                        self._draw_marker("_target_marker", target, (1.0, 0.0, 0.0, 0.9))
                    last_target = target
                if a.model.draw_debug_markers:
                    if a._ik_target_marker is not None:
                        try:
                            a._scene.clear_debug_object(a._ik_target_marker)
                        except Exception:
                            pass
                        a._ik_target_marker = None

                self._handle_ik_requests(linear=linear, roll=roll, theta1=theta1, theta2=theta2, paused=paused)
                linear, roll, theta1, theta2 = self._step_ik(
                    linear=linear,
                    roll=roll,
                    theta1=theta1,
                    theta2=theta2,
                    paused=paused,
                )
                self._update_debug_metrics(
                    linear=linear,
                    roll=roll,
                    theta1=theta1,
                    theta2=theta2,
                    target=target,
                )
                self._send_link_target(linear=linear, roll=roll, theta1=theta1, theta2=theta2, paused=paused)

                if not paused:
                    q_model = a._model_q() if a._render_from_model_required() else None
                    if q_model is not None:
                        a._mover.set_4dof_instant(
                            float(q_model.linear_m),
                            float(q_model.roll_rad),
                            float(q_model.theta1_rad),
                            float(q_model.theta2_rad),
                        )
                    elif a._chain_control_mode == "commanded":
                        a._mover.control_4dof(linear=linear, roll=roll, theta1=theta1, theta2=theta2)

                a._scene.step()
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
        model: Optional[AppModelConfig] = None,
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
        self.model = model if model is not None else AppModelConfig()
        self.urdf_export_cfg = urdf_export_cfg if urdf_export_cfg is not None else UrdfExportConfig()
        self.ik_cfg = ik_cfg if ik_cfg is not None else IkConfig()
        self.hardware_cfg = hardware_cfg if hardware_cfg is not None else HardwareConfig()
        self.state = ControlState()

        self._proto_cfg = mapping_cfg if mapping_cfg is not None else proto.SimMappingConfig()
        if endpoint is None:
            endpoint = self.cfg.zmq_endpoint
        if enable_link is None:
            enable_link = self.cfg.use_hardware

        self._link: Optional[LinkClient] = None
        self._bridge_proc: Optional[subprocess.Popen] = None
        if enable_link and endpoint:
            self._link = self._try_init_link_with_bridge(endpoint=endpoint)
            if self._link is None:
                print("[runtime] bridge unavailable -> commanded-only")

        self._scene = None
        self._mover: Optional[Mover] = None
        self._gui: Optional[ImGuiController] = None
        self._gui_thread: Optional[threading.Thread] = None
        self._kin: Optional[Kinematics] = None
        self._ik: Optional[IKNewAdapter] = None
        self._ik_refiner: Optional[TaskSpaceRefiner] = None
        self._ik_refine_cfg = TaskSpaceRefineConfig(ds=0.35, goal_tol_m=1e-4, max_step_m=0.03)
        self._ik_refine_iter: int = 0
        self._ik_refine_max_iters: int = 8
        self._ik_hold_q: Optional[np.ndarray] = None
        self._ik_traj_active: bool = False
        self._ik_tip_tol_m: float = 1e-4  # 0.1 mm
        self._ik_tip_ok_need: int = 3
        self._ik_tip_ok_streak: int = 0
        self._ik_traj_q0: np.ndarray = np.zeros(4, dtype=float)
        self._ik_traj_q1: np.ndarray = np.zeros(4, dtype=float)
        self._ik_traj_t0: float = 0.0
        self._ik_traj_T: float = 0.0
        self._ik_target_xyz: Optional[np.ndarray] = None
        self._ik_logged_converged: bool = False
        self._n_nodes: int = 0
        self._n_seg: int = 0
        self._base_joint_name: str = "base_prismatic_x"
        self._roll_joint_name: str = "base_roll_x"
        self._bend_joint_names: List[str] = []
        self._base_axis_sign: float = 1.0
        self._roll_axis_sign: float = 1.0
        self._bend_axis_sign: float = -1.0
        self._chain_origin_local: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float)
        self._tip_link_name: str = ""
        self._tip_local_offset: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=float)
        self._startup_pose_synced: bool = False
        self._startup_pose_device: str = ""
        self._chain_control_mode: str = "commanded"
        self._part_control_mode: Dict[str, str] = {}
        self._part_pose_root: Dict[str, np.ndarray] = {}
        self._fk_root_link: str = "plate"
        self._fk_joint_chain: List[Dict[str, object]] = []
        self._no_clip_pairs: List[Tuple[str, str]] = []
        self._target_marker = None
        self._ik_target_marker = None
        self._des_tip_marker = None
        self._sim_tip_marker = None
        self._chain_model: Optional[ChainStateModel] = DirectHardwareModel() if bool(self.cfg.use_hardware) else None

    def _startup_sync_required(self) -> bool:
        return bool((self._chain_control_mode == "simulated") and self.cfg.use_hardware and (self._link is not None))

    def _apply_hardware_matched_ideal_rates_if_needed(self) -> None:
        if self._chain_control_mode != "commanded":
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

    def _render_from_model_required(self) -> bool:
        return bool(self._chain_control_mode == "simulated")

    def _model_q(self) -> Optional[proto.SimQ]:
        if self._chain_model is None:
            return None
        try:
            return self._chain_model.estimate_q()
        except Exception:
            return None

    def _apply_startup_pose_from_q(self, q_from_real: proto.SimQ) -> bool:
        if self._mover is None:
            return False
        self.state.set_all(
            float(q_from_real.linear_m),
            float(q_from_real.roll_rad),
            float(q_from_real.theta1_rad),
            float(q_from_real.theta2_rad),
            self.state.paused,
        )
        self._mover.set_4dof_instant(
            float(q_from_real.linear_m),
            float(q_from_real.roll_rad),
            float(q_from_real.theta1_rad),
            float(q_from_real.theta2_rad),
        )
        self._startup_pose_synced = True
        print(
            "[runtime] initial Pose synced from hardware: "
            "linear=%.4f m roll=%.2f deg seg1=%.2f deg seg2=%.2f deg"
            % (
                float(q_from_real.linear_m),
                np.degrees(float(q_from_real.roll_rad)),
                np.degrees(float(q_from_real.theta1_rad)),
                np.degrees(float(q_from_real.theta2_rad)),
            )
        )
        return True

    def _wait_for_startup_pose_sync(self, *, timeout_s: float) -> None:
        if (not self._startup_sync_required()) or self._startup_pose_synced or self._link is None:
            return
        t0 = time.time()
        while (time.time() - t0) < float(timeout_s):
            try:
                self._link.poll()
                if self._link.last_q is not None and self._apply_startup_pose_from_q(self._link.last_q):
                    return
            except Exception:
                break
            time.sleep(0.01)
        print("[runtime] startup Pose sync pending: waiting for first hardware state before sending targets")

    def _build_bind_endpoint(self, endpoint: str) -> str:
        ep = str(endpoint).strip()
        if ep.startswith("tcp://127.0.0.1:"):
            return "tcp://*:" + ep.rsplit(":", 1)[-1]
        if ep.startswith("tcp://localhost:"):
            return "tcp://*:" + ep.rsplit(":", 1)[-1]
        return ep

    def _try_init_link_with_bridge(self, *, endpoint: str) -> Optional[LinkClient]:
        # First try: connect to already-running bridge.
        try:
            link = LinkClient(endpoint, cfg=self._proto_cfg)
            t0 = time.time()
            acked = False
            while time.time() - t0 < 0.15:
                link.poll()
                if np.isfinite(link.rx_age_s()) and link.rx_age_s() < 1.0:
                    acked = True
                    break
                time.sleep(0.01)
            if acked:
                return link
            try:
                link.close()
            except Exception:
                pass
        except Exception:
            pass

        bridge_path = os.path.join(os.path.dirname(__file__), "bridge.py")
        if not os.path.isfile(bridge_path):
            return None
        try:
            cmd = [
                sys.executable,
                bridge_path,
                "--bind",
                self._build_bind_endpoint(endpoint),
            ]
            self._bridge_proc = subprocess.Popen(cmd)
            time.sleep(0.25)
            link = LinkClient(endpoint, cfg=self._proto_cfg)
            return link
        except Exception:
            self._bridge_proc = None
            return None

    def run(self) -> None:
        try:
            urdf_path = AssetPipeline(self).prepare_assets()
            runtime = RuntimeBootstrap(self)
            runtime.init_genesis(urdf_path)
            runtime.start_gui_thread()
            ControlLoopCoordinator(self).run()
        finally:
            if self._bridge_proc is not None:
                try:
                    self._bridge_proc.terminate()
                except Exception:
                    pass
                self._bridge_proc = None


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
        params=bundle.SimParam,
        cfg=bundle.SimConfig,
        hardware_cfg=bundle.HardwareConfig,
        limit=bundle.JointLimit,
        model=bundle.model_config,
        urdf_export_cfg=bundle.UrdfExportConfig,
        ik_cfg=bundle.IkConfig,
        mapping_cfg=bundle.mapping_config,
    )
    app.run()


if __name__ == "__main__":
    main()
