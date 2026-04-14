#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import configparser
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from engine import protocol as proto


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BUILD_DIR = os.path.join(PROJECT_ROOT, "build")


@dataclass(frozen=True)
class JointLimit:
    linear_min: float
    linear_max: float
    roll_min_deg: float
    roll_max_deg: float
    bend_deg: float

    def roll_min_rad(self) -> float:
        return math.radians(self.roll_min_deg)

    def roll_max_rad(self) -> float:
        return math.radians(self.roll_max_deg)

    def bend_lim_rad(self) -> float:
        return math.radians(self.bend_deg)

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.linear_min, self.roll_min_rad(), -self.bend_lim_rad(), -self.bend_lim_rad()], dtype=float)
        hi = np.array([self.linear_max, self.roll_max_rad(), +self.bend_lim_rad(), +self.bend_lim_rad()], dtype=float)
        return lo, hi


@dataclass(frozen=True)
class SimParam:
    dt: float = 0.01
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    linear_rate: float = float("inf")
    roll_rate: float = float("inf")
    bend_rate: float = float("inf")

    zmq_hwm: int = 1


@dataclass(frozen=True)
class SimConfig:
    use_gpu: bool = True
    enable_viewer: bool = True
    floor: bool = True
    use_hardware: bool = True
    use_go2: bool = False

    build_dir: str = DEFAULT_BUILD_DIR
    assy_build_json: str = "manifest.json"
    urdf_name: str = "robot.urdf"
    rebuild_assembly: bool = True

    zmq_endpoint: str = "tcp://127.0.0.1:5555"


@dataclass(frozen=True)
class HardwareConfig:
    command_direction: Tuple[int, int, int, int] = (1, 1, 1, 1)
    motor_direction: Tuple[int, int, int, int] = (1, 1, 1, 1)


@dataclass(frozen=True)
class UrdfExportConfig:
    robot_name: str = "Robot"
    default_effort: float = 200.0
    default_velocity: float = 3.0
    revolute_effort: Optional[float] = None
    revolute_velocity: Optional[float] = None
    prismatic_effort: Optional[float] = None
    prismatic_velocity: Optional[float] = None
    revolute_damping: float = 0.12
    revolute_friction: float = 0.06
    prismatic_damping: float = 60.0
    prismatic_friction: float = 20.0
    mesh_basename_only: bool = False
    part_color_rgba_by_name: dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class IkConfig:
    tol: float = 1e-4
    max_iters: int = 200
    stall_limit: int = 40

    damping_init: float = 1e-2
    damping_min: float = 1e-6
    damping_max: float = 1e+2
    damping_up: float = 10.0
    damping_down: float = 0.7

    step_scale: float = 1.0
    line_search_steps: int = 4
    line_search_shrink: float = 0.5
    fd_eps: float = 1e-4
    direction_weight: float = 0.1
    prefer_tip_plus_x: bool = True
    direction_tol_deg: float = 1.0
    orientation_tie_eps_m: float = 1e-3


@dataclass(frozen=True)
class AppModelConfig:
    pitch: float = 0.05
    n_seg: Optional[int] = None
    spawn_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    draw_debug_markers: bool = True
    base1_offset: Tuple[float, float, float] = (-0.010, 0.065, 0.010)
    base2_offset: Tuple[float, float, float] = (-0.030, 0.065, 0.010)
    base3_offset: Tuple[float, float, float] = (-0.050, 0.065, 0.010)


@dataclass(frozen=True)
class AppConfigBundle:
    SimParam: SimParam
    SimConfig: SimConfig
    HardwareConfig: HardwareConfig
    JointLimit: JointLimit
    model_config: AppModelConfig
    UrdfExportConfig: UrdfExportConfig
    IkConfig: IkConfig
    mapping_config: proto.SimMappingConfig


def _parse_vec3(text: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    raw = str(text).strip()
    if not raw:
        return default
    parts = [x.strip() for x in raw.split(",")]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return default


def _parse_optional_float(text: str, default: Optional[float]) -> Optional[float]:
    raw = str(text).strip()
    if raw == "":
        return default
    if raw.lower() in ("none", "null"):
        return None
    try:
        return float(raw)
    except Exception:
        return default


def _parse_color_rgba(text: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    raw = str(text).strip()
    if not raw:
        return default
    if raw.startswith("#"):
        h = raw[1:].strip()
        if len(h) == 6:
            try:
                r = int(h[0:2], 16) / 255.0
                g = int(h[2:4], 16) / 255.0
                b = int(h[4:6], 16) / 255.0
                return (r, g, b, 1.0)
            except Exception:
                return default
        if len(h) == 8:
            try:
                r = int(h[0:2], 16) / 255.0
                g = int(h[2:4], 16) / 255.0
                b = int(h[4:6], 16) / 255.0
                a = int(h[6:8], 16) / 255.0
                return (r, g, b, a)
            except Exception:
                return default
        return default
    parts = [x.strip() for x in raw.split(",")]
    if len(parts) != 4:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except Exception:
        return default


def _parse_direction4(text: str, *, key: str) -> Tuple[int, int, int, int]:
    raw = str(text).strip()
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError(f"{key} must contain exactly 4 comma-separated values in order: linear, roll, seg1, seg2")
    values = []
    for part in parts:
        try:
            value = int(part)
        except Exception as exc:
            raise ValueError(f"{key} must contain only integers 1 or -1") from exc
        if value not in (-1, 1):
            raise ValueError(f"{key} must contain only 1 or -1")
        values.append(value)
    return (values[0], values[1], values[2], values[3])


def _default_app_config_bundle() -> AppConfigBundle:
    return AppConfigBundle(
        SimParam=SimParam(),
        SimConfig=SimConfig(),
        HardwareConfig=HardwareConfig(),
        JointLimit=JointLimit(linear_min=-0.23, linear_max=0.01, roll_min_deg=-90.0, roll_max_deg=90.0, bend_deg=36.0),
        model_config=AppModelConfig(),
        UrdfExportConfig=UrdfExportConfig(),
        IkConfig=IkConfig(),
        mapping_config=proto.SimMappingConfig(),
    )


def load_app_config_from_ini(path: str) -> AppConfigBundle:
    defaults = _default_app_config_bundle()
    if not path or not os.path.isfile(path):
        return defaults

    config_dir = os.path.dirname(os.path.abspath(path))

    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(path, encoding="utf-8-sig")

    sp0 = defaults.SimParam
    sim_param_cfg = SimParam(
        dt=cp.getfloat("SimParam", "dt", fallback=sp0.dt),
        substeps=cp.getint("SimParam", "substeps", fallback=sp0.substeps),
        gravity=_parse_vec3(cp.get("SimParam", "gravity", fallback=""), sp0.gravity),
        linear_rate=cp.getfloat("SimParam", "linear_rate", fallback=sp0.linear_rate),
        roll_rate=cp.getfloat("SimParam", "roll_rate", fallback=sp0.roll_rate),
        bend_rate=cp.getfloat("SimParam", "bend_rate", fallback=sp0.bend_rate),
        zmq_hwm=cp.getint("SimParam", "zmq_hwm", fallback=sp0.zmq_hwm),
    )

    sc0 = defaults.SimConfig
    build_dir = os.path.abspath(os.path.join(config_dir, "build"))
    sim_config_cfg = SimConfig(
        use_gpu=cp.getboolean("runtime", "use_gpu", fallback=sc0.use_gpu),
        enable_viewer=cp.getboolean("runtime", "enable_viewer", fallback=sc0.enable_viewer),
        floor=cp.getboolean("runtime", "floor", fallback=sc0.floor),
        use_hardware=cp.getboolean("runtime", "use_hardware", fallback=sc0.use_hardware),
        use_go2=cp.getboolean("runtime", "use_go2", fallback=sc0.use_go2),
        build_dir=build_dir,
        assy_build_json=cp.get("runtime", "assy_build_json", fallback=sc0.assy_build_json),
        urdf_name=cp.get("runtime", "urdf_name", fallback=sc0.urdf_name),
        rebuild_assembly=cp.getboolean("model", "rebuild_robot_assets", fallback=sc0.rebuild_assembly),
        zmq_endpoint=cp.get("runtime", "zmq_endpoint", fallback=sc0.zmq_endpoint),
    )
    if cp.has_option("hardware", "dxl_dir_1") or cp.has_option("hardware", "dxl_dir_2") or cp.has_option("hardware", "dxl_dir_3") or cp.has_option("hardware", "dxl_dir_4"):
        raise ValueError("legacy hardware keys dxl_dir_1..4 are no longer supported; use command_direction and motor_direction")
    if not cp.has_option("hardware", "command_direction"):
        raise ValueError("missing required hardware.command_direction")
    if not cp.has_option("hardware", "motor_direction"):
        raise ValueError("missing required hardware.motor_direction")
    hardware_config_cfg = HardwareConfig(
        command_direction=_parse_direction4(cp.get("hardware", "command_direction"), key="hardware.command_direction"),
        motor_direction=_parse_direction4(cp.get("hardware", "motor_direction"), key="hardware.motor_direction"),
    )

    jl0 = defaults.JointLimit
    joint_limit_cfg = JointLimit(
        linear_min=cp.getfloat("model", "linear_min", fallback=jl0.linear_min),
        linear_max=cp.getfloat("model", "linear_max", fallback=jl0.linear_max),
        roll_min_deg=cp.getfloat("model", "roll_min_deg", fallback=jl0.roll_min_deg),
        roll_max_deg=cp.getfloat("model", "roll_max_deg", fallback=jl0.roll_max_deg),
        bend_deg=cp.getfloat("model", "bend_deg", fallback=jl0.bend_deg),
    )

    am0 = defaults.model_config
    n_seg_raw = cp.get("app_model", "n_seg", fallback="")
    n_seg = am0.n_seg if n_seg_raw.strip() == "" else int(n_seg_raw)
    model_config_cfg = AppModelConfig(
        pitch=cp.getfloat("app_model", "pitch", fallback=am0.pitch),
        n_seg=n_seg,
        spawn_xyz=_parse_vec3(cp.get("spawn", "spawn_position", fallback=""), am0.spawn_xyz),
        spawn_euler_deg=_parse_vec3(cp.get("spawn", "spawn_orientation_deg", fallback=""), am0.spawn_euler_deg),
        draw_debug_markers=cp.getboolean("spawn", "draw_debug_markers", fallback=am0.draw_debug_markers),
        base1_offset=_parse_vec3(cp.get("spawn", "base1_offset", fallback=""), am0.base1_offset),
        base2_offset=_parse_vec3(cp.get("spawn", "base2_offset", fallback=""), am0.base2_offset),
        base3_offset=_parse_vec3(cp.get("spawn", "base3_offset", fallback=""), am0.base3_offset),
    )

    ue0 = defaults.UrdfExportConfig
    part_color_rgba_by_name: dict[str, Tuple[float, float, float, float]] = {}
    if cp.has_section("colors"):
        for raw_name, raw_value in cp.items("colors"):
            part_name = str(raw_name).strip()
            if not part_name:
                continue
            part_color_rgba_by_name[part_name] = _parse_color_rgba(str(raw_value), default=(1.0, 1.0, 1.0, 1.0))
    urdf_export_config_cfg = UrdfExportConfig(
        robot_name=cp.get("urdf_export", "robot_name", fallback=ue0.robot_name),
        default_effort=cp.getfloat("urdf_export", "default_effort", fallback=ue0.default_effort),
        default_velocity=cp.getfloat("urdf_export", "default_velocity", fallback=ue0.default_velocity),
        revolute_effort=_parse_optional_float(cp.get("urdf_export", "revolute_effort", fallback=""), ue0.revolute_effort),
        revolute_velocity=_parse_optional_float(
            cp.get("urdf_export", "revolute_velocity", fallback=""), ue0.revolute_velocity
        ),
        prismatic_effort=_parse_optional_float(cp.get("urdf_export", "prismatic_effort", fallback=""), ue0.prismatic_effort),
        prismatic_velocity=_parse_optional_float(
            cp.get("urdf_export", "prismatic_velocity", fallback=""), ue0.prismatic_velocity
        ),
        revolute_damping=cp.getfloat("urdf_export", "revolute_damping", fallback=ue0.revolute_damping),
        revolute_friction=cp.getfloat("urdf_export", "revolute_friction", fallback=ue0.revolute_friction),
        prismatic_damping=cp.getfloat("urdf_export", "prismatic_damping", fallback=ue0.prismatic_damping),
        prismatic_friction=cp.getfloat("urdf_export", "prismatic_friction", fallback=ue0.prismatic_friction),
        mesh_basename_only=cp.getboolean("urdf_export", "mesh_basename_only", fallback=ue0.mesh_basename_only),
        part_color_rgba_by_name=part_color_rgba_by_name,
    )

    ik0 = defaults.IkConfig
    ik_config_cfg = IkConfig(
        tol=cp.getfloat("ik", "tol", fallback=ik0.tol),
        max_iters=cp.getint("ik", "max_iters", fallback=ik0.max_iters),
        stall_limit=cp.getint("ik", "stall_limit", fallback=ik0.stall_limit),
        damping_init=cp.getfloat("ik", "damping_init", fallback=ik0.damping_init),
        damping_min=cp.getfloat("ik", "damping_min", fallback=ik0.damping_min),
        damping_max=cp.getfloat("ik", "damping_max", fallback=ik0.damping_max),
        damping_up=cp.getfloat("ik", "damping_up", fallback=ik0.damping_up),
        damping_down=cp.getfloat("ik", "damping_down", fallback=ik0.damping_down),
        step_scale=cp.getfloat("ik", "step_scale", fallback=ik0.step_scale),
        line_search_steps=cp.getint("ik", "line_search_steps", fallback=ik0.line_search_steps),
        line_search_shrink=cp.getfloat("ik", "line_search_shrink", fallback=ik0.line_search_shrink),
        fd_eps=cp.getfloat("ik", "fd_eps", fallback=ik0.fd_eps),
        direction_weight=cp.getfloat("ik", "direction_weight", fallback=ik0.direction_weight),
        prefer_tip_plus_x=cp.getboolean("ik", "prefer_tip_plus_x", fallback=ik0.prefer_tip_plus_x),
        direction_tol_deg=cp.getfloat("ik", "direction_tol_deg", fallback=ik0.direction_tol_deg),
        orientation_tie_eps_m=cp.getfloat("ik", "orientation_tie_eps_m", fallback=ik0.orientation_tie_eps_m),
    )

    mapping_config_cfg = proto.SimMappingConfig(
        linear_q_min_m=joint_limit_cfg.linear_min,
        linear_q_max_m=joint_limit_cfg.linear_max,
        roll_q_min_rad=joint_limit_cfg.roll_min_rad(),
        roll_q_max_rad=joint_limit_cfg.roll_max_rad(),
        seg1_q_min_rad=-joint_limit_cfg.bend_lim_rad(),
        seg1_q_max_rad=+joint_limit_cfg.bend_lim_rad(),
        seg2_q_min_rad=-joint_limit_cfg.bend_lim_rad(),
        seg2_q_max_rad=+joint_limit_cfg.bend_lim_rad(),
        command_direction=hardware_config_cfg.command_direction,
    )

    return AppConfigBundle(
        SimParam=sim_param_cfg,
        SimConfig=sim_config_cfg,
        HardwareConfig=hardware_config_cfg,
        JointLimit=joint_limit_cfg,
        model_config=model_config_cfg,
        UrdfExportConfig=urdf_export_config_cfg,
        IkConfig=ik_config_cfg,
        mapping_config=mapping_config_cfg,
    )
