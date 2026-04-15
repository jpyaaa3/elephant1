#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import builder.json_builder as assembly_builder
from engine.config_loader import AppConfigBundle, load_app_config_from_ini
from engine.joint_defs import JointLimit
from engine.iklib_src.kinematics import Kinematics
from engine.iklib_src.solver import OptimizeResult, IkPositionOptimizer


Q4 = np.ndarray
Vec3 = np.ndarray

Q_I = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
Q_C = np.array([0.0, 0.0, -math.radians(36.0), +math.radians(36.0)], dtype=float)


@dataclass(frozen=True)
class IkSolveRequest:
    target_world: Vec3
    position_tol_m: float = 1e-4


@dataclass(frozen=True)
class IkSolveResult:
    success: bool
    q: Optional[Q4]
    position_error_m: float
    seed_name: str
    iterations: int
    reason: str = ""


def _pick_manifest_value(mapping, *keys, default=None):
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return default


def load_ik_context(config_path: str) -> tuple[AppConfigBundle, dict[str, Any]]:
    bundle = load_app_config_from_ini(config_path)
    build_dir = str(bundle.sim_config.build_dir)
    manifest_path = os.path.join(build_dir, str(bundle.sim_config.assy_build_json))
    if not os.path.isfile(manifest_path):
        os.makedirs(build_dir, exist_ok=True)
        assembly_builder.build_default_manifest(
            build_dir,
            use_hardware=bool(bundle.sim_config.use_hardware),
            use_go2=bool(bundle.sim_config.use_go2),
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        build = json.load(f)

    joints = list(_pick_manifest_value(build, "joints", default=[]))
    if not joints:
        raise RuntimeError("manifest json is missing joints")

    prismatic_names: list[str] = []
    revolute_names: list[str] = []
    for joint in joints:
        jname = str(_pick_manifest_value(joint, "name", default="")).strip()
        jtype = str(_pick_manifest_value(joint, "type", default="")).strip().lower()
        if not jname:
            continue
        if jtype == "prismatic":
            prismatic_names.append(jname)
        elif jtype == "revolute":
            revolute_names.append(jname)
    if not prismatic_names or len(revolute_names) < 2:
        raise RuntimeError("manifest json does not provide enough control joints")

    base_joint_name = prismatic_names[0]
    roll_joint_name = revolute_names[0]
    bend_joint_names = revolute_names[1:]
    joint_by_name = {str(_pick_manifest_value(j, "name", default="")): j for j in joints}

    first_bend = joint_by_name.get(bend_joint_names[0]) if bend_joint_names else None
    if first_bend is None:
        raise RuntimeError("manifest json is missing first bend joint metadata")
    anchor_root = _pick_manifest_value(first_bend, "anchor_root", default=None)
    if not isinstance(anchor_root, (list, tuple)) or len(anchor_root) != 3:
        raise RuntimeError("manifest json is missing valid anchor_root for first bend joint")
    chain_origin_local = np.array([float(anchor_root[0]), float(anchor_root[1]), float(anchor_root[2])], dtype=float)

    def _axis_sign(raw_axis, axis_idx: int, *, name: str) -> float:
        axis = np.asarray(raw_axis, dtype=float).reshape(-1)
        if axis.size <= axis_idx:
            raise RuntimeError(f"manifest json is missing valid axis_root for {name}")
        value = float(axis[axis_idx])
        if abs(value) < 1e-9:
            raise RuntimeError(f"manifest json axis_root for {name} has zero component on required axis")
        return -1.0 if value < 0.0 else 1.0

    base_meta = joint_by_name.get(base_joint_name, {})
    roll_meta = joint_by_name.get(roll_joint_name, {})
    bend_meta = joint_by_name.get(bend_joint_names[0], {})

    spawn_xyz = np.array(bundle.spawn_config.spawn_xyz, dtype=float).reshape(3)
    spawn_euler = np.array(bundle.spawn_config.spawn_euler_deg, dtype=float).reshape(3)
    spawn_q_xyzw = Rot.from_euler("xyz", spawn_euler, degrees=True).as_quat()
    spawn_q_wxyz = np.array([spawn_q_xyzw[3], spawn_q_xyzw[0], spawn_q_xyzw[1], spawn_q_xyzw[2]], dtype=float)
    from genesis.utils import geom as gs_geom

    chain_origin_world = gs_geom.transform_by_trans_quat(chain_origin_local, spawn_xyz, spawn_q_wxyz)

    n_nodes = len(bend_joint_names)
    n_seg = int(bundle.spawn_config.n_seg) if bundle.spawn_config.n_seg is not None else max(1, n_nodes // 2)
    ik_context = {
        "pitch": float(bundle.spawn_config.pitch),
        "n_nodes": int(n_nodes),
        "n_seg": int(n_seg),
        "origin_xyz": np.array(chain_origin_world, dtype=float),
        "limit": bundle.joint_limit,
        "base_axis_sign": float(_axis_sign(_pick_manifest_value(base_meta, "axis_root", default=None), 0, name=base_joint_name)),
        "roll_axis_sign": float(_axis_sign(_pick_manifest_value(roll_meta, "axis_root", default=None), 0, name=roll_joint_name)),
        "bend_axis_sign": float(_axis_sign(_pick_manifest_value(bend_meta, "axis_root", default=None), 1, name=bend_joint_names[0])),
    }
    return bundle, ik_context


def solve_ik(
    *,
    target_world: Sequence[float],
    pitch: float,
    n_nodes: int,
    n_seg: int,
    origin_xyz: Sequence[float],
    limit: JointLimit,
    position_tol_m: float = 1e-4,
    max_iters: int = 120,
    base_axis_sign: float = 1.0,
    roll_axis_sign: float = 1.0,
    bend_axis_sign: float = -1.0,
    i_seed: Optional[Sequence[float]] = None,
    c_seed: Optional[Sequence[float]] = None,
) -> IkSolveResult:
    request = IkSolveRequest(
        target_world=np.asarray(target_world, dtype=float).reshape(3),
        position_tol_m=float(position_tol_m),
    )
    kin = Kinematics(
        pitch=float(pitch),
        n_nodes=int(n_nodes),
        n_seg=int(n_seg),
        origin_xyz=np.asarray(origin_xyz, dtype=float).reshape(3),
        limit=limit,
        base_axis_sign=float(base_axis_sign),
        roll_axis_sign=float(roll_axis_sign),
        bend_axis_sign=float(bend_axis_sign),
    )
    optimizer = IkPositionOptimizer(max_iters=int(max_iters))

    seeds = [
        ("I", kin.clamp_q(np.asarray(i_seed if i_seed is not None else Q_I, dtype=float).reshape(4))),
        ("C", kin.clamp_q(np.asarray(c_seed if c_seed is not None else Q_C, dtype=float).reshape(4))),
    ]
    tol = float(max(request.position_tol_m, 0.0))

    for seed_name, q_seed in seeds:
        result = optimizer.solve(
            q0=q_seed,
            tol=tol,
            error_vec_fn=lambda q, target=request.target_world: kin.position_error_vec(q, target),
            jacobian_fn=lambda q, eps: kin.numerical_jacobian(q, eps=eps),
            clamp_q_fn=kin.clamp_q,
        )
        if result.success:
            return _build_ik_solve_result(result, seed_name)

    return IkSolveResult(
        success=False,
        q=None,
        position_error_m=float("inf"),
        seed_name="C",
        iterations=int(max_iters),
        reason="position tolerance not reached",
    )


def _build_ik_solve_result(result: OptimizeResult, seed_name: str) -> IkSolveResult:
    return IkSolveResult(
        success=bool(result.success),
        q=np.asarray(result.q, dtype=float).reshape(4).copy() if result.success else None,
        position_error_m=float(result.error_norm),
        seed_name=str(seed_name),
        iterations=int(result.iterations),
        reason=str(result.reason),
    )


__all__ = [
    "Q_I",
    "Q_C",
    "IkSolveRequest",
    "IkSolveResult",
    "load_ik_context",
    "solve_ik",
]
