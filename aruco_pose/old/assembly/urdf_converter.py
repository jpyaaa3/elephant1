#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from engine.config_loader import UrdfExportConfig


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def _as_vec3(x: Any) -> Vec3:
    a = list(x)
    return (float(a[0]), float(a[1]), float(a[2]))


def _as_quat_xyzw(q: Any) -> Quat:
    a = list(q)
    return (float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def _fmt3(v: Tuple[float, float, float]) -> str:
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def _fmt4(v: Tuple[float, float, float, float]) -> str:
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g} {v[3]:.9g}"


def _norm3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class JSON2URDF:
    def __init__(self, cfg: UrdfExportConfig = UrdfExportConfig()):
        self.cfg = cfg

    def _mesh_filename(self, path: str) -> str:
        if not path:
            return ""
        norm = path.replace("\\", "/")
        return os.path.basename(norm) if self.cfg.mesh_basename_only else norm

    def _load_physics(self, build_dir: str, phy_path: str) -> Tuple[float, Tuple[float, float, float], Dict[str, float]]:
        mass = 0.001
        com = (0.0, 0.0, 0.0)
        inertia = {"ixx": 1e-6, "ixy": 0.0, "ixz": 0.0, "iyy": 1e-6, "iyz": 0.0, "izz": 1e-6}

        if not phy_path:
            return mass, com, inertia

        rel = str(phy_path).replace("\\", "/")
        abs_path = os.path.abspath(rel) if os.path.isabs(rel) else os.path.abspath(os.path.join(build_dir, rel))
        if not os.path.exists(abs_path):
            return mass, com, inertia

        with open(abs_path, "r", encoding="utf-8") as f:
            ph = json.load(f)

        if "mass" in ph:
            mass = float(ph["mass"])

        com_raw = ph.get("com")
        if isinstance(com_raw, (list, tuple)) and len(com_raw) == 3:
            com = (float(com_raw[0]), float(com_raw[1]), float(com_raw[2]))

        ip = ph.get("inertia", {}) or {}
        inertia = {
            "ixx": float(ip.get("ixx", inertia["ixx"])),
            "iyy": float(ip.get("iyy", inertia["iyy"])),
            "izz": float(ip.get("izz", inertia["izz"])),
            "ixy": float(ip.get("ixy", inertia["ixy"])),
            "ixz": float(ip.get("ixz", inertia["ixz"])),
            "iyz": float(ip.get("iyz", inertia["iyz"])),
        }

        if mass <= 0.0:
            mass = 0.001
        return mass, com, inertia

    @staticmethod
    def _normalize_joint_type(jtype_src: str) -> str:
        return jtype_src if jtype_src in ("revolute", "prismatic", "fixed") else "fixed"

    def _part_color_rgba(self, part_name: str) -> Tuple[float, float, float, float] | None:
        return self.cfg.part_color_rgba_by_name.get(str(part_name).strip())

    def _joint_limit_effort_velocity(self, JointType: str) -> Tuple[float, float]:
        if JointType == "revolute":
            effort = self.cfg.revolute_effort if self.cfg.revolute_effort is not None else self.cfg.default_effort
            velocity = self.cfg.revolute_velocity if self.cfg.revolute_velocity is not None else self.cfg.default_velocity
            return float(effort), float(velocity)
        if JointType == "prismatic":
            effort = self.cfg.prismatic_effort if self.cfg.prismatic_effort is not None else self.cfg.default_effort
            velocity = self.cfg.prismatic_velocity if self.cfg.prismatic_velocity is not None else self.cfg.default_velocity
            return float(effort), float(velocity)
        return float(self.cfg.default_effort), float(self.cfg.default_velocity)

    def _joint_dynamics(self, JointType: str) -> Tuple[float, float]:
        if JointType == "revolute":
            return float(self.cfg.revolute_damping), float(self.cfg.revolute_friction)
        if JointType == "prismatic":
            return float(self.cfg.prismatic_damping), float(self.cfg.prismatic_friction)
        return 0.0, 0.0

    def convert_file(self, assy_build_json_path: str, urdf_out_path: str) -> None:
        with open(assy_build_json_path, "r", encoding="utf-8") as f:
            build = json.load(f)

        build_dir = os.path.dirname(os.path.abspath(assy_build_json_path))
        urdf_text = self.convert_dict(build, build_dir=build_dir)

        os.makedirs(os.path.dirname(os.path.abspath(urdf_out_path)), exist_ok=True)
        with open(urdf_out_path, "w", encoding="utf-8") as f:
            f.write(urdf_text)

    def convert_dict(self, build: Dict[str, Any], *, build_dir: str) -> str:
        def _pick(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        parts: List[Dict[str, Any]] = list(_pick(build, "parts", default=[]))
        joints: List[Dict[str, Any]] = list(_pick(build, "joints", default=[]))
        part_flags: Dict[str, Any] = dict(_pick(build, "part_flags", default={}) or {})
        no_clip_pairs = list(_pick(build, "no_clip_pairs", default=[]) or [])

        parts_by: Dict[str, Dict[str, Any]] = {str(_pick(p, "name")): p for p in parts}
        if not parts_by:
            raise ValueError("manifest.json: parts is empty.")

        parent_of: Dict[str, str] = {}
        joint_by_child: Dict[str, Dict[str, Any]] = {}
        children_of: Dict[str, List[str]] = {name: [] for name in parts_by.keys()}
        for j in joints:
            parent = str(_pick(j, "parent"))
            child = str(_pick(j, "child"))
            if child in parent_of:
                raise ValueError(f"Joint tree error: child '{child}' has multiple parents.")
            parent_of[child] = parent
            joint_by_child[child] = j
            children_of.setdefault(parent, []).append(child)

        roots = [name for name in parts_by.keys() if name not in parent_of]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly 1 root link, got {roots}")
        root = roots[0]

        R_root: Dict[str, np.ndarray] = {}
        for name, p in parts_by.items():
            pose_root = _pick(p, "pose_root", default={}) or {}
            q_xyzw = _as_quat_xyzw(_pick(pose_root, "q"))
            R_root[name] = Rot.from_quat(q_xyzw).as_matrix()

        P_root: Dict[str, np.ndarray] = {}
        for name, p in parts_by.items():
            pose_root = _pick(p, "pose_root", default={}) or {}
            P_root[name] = np.array(_as_vec3(_pick(pose_root, "p")), dtype=float)

        robot = ET.Element("robot", attrib={"name": self.cfg.robot_name})

        for name, p in parts_by.items():
            link_el = ET.SubElement(robot, "link", attrib={"name": name})

            assets = _pick(p, "assets", default={}) or {}
            mesh_path = str(_pick(assets, "mesh", default="") or "")
            phy_path = str(_pick(assets, "physics", default="") or "")
            flags = (_pick(p, "flags", default=None) or part_flags.get(name, {}) or {})
            collision_enabled = bool(_pick(flags, "collision_enabled", default=True))
            mass, com, I = self._load_physics(build_dir, phy_path)

            inertial = ET.SubElement(link_el, "inertial")
            ET.SubElement(inertial, "origin", attrib={"xyz": _fmt3(com), "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", attrib={"value": f"{mass:.9g}"})
            ET.SubElement(
                inertial,
                "inertia",
                attrib={
                    "ixx": f"{I['ixx']:.9g}",
                    "ixy": f"{I['ixy']:.9g}",
                    "ixz": f"{I['ixz']:.9g}",
                    "iyy": f"{I['iyy']:.9g}",
                    "iyz": f"{I['iyz']:.9g}",
                    "izz": f"{I['izz']:.9g}",
                },
            )

            if mesh_path:
                visual = ET.SubElement(link_el, "visual")
                ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
                geom = ET.SubElement(visual, "geometry")
                ET.SubElement(geom, "mesh", attrib={"filename": self._mesh_filename(mesh_path)})
                rgba = self._part_color_rgba(name)
                if rgba is not None:
                    material = ET.SubElement(visual, "material", attrib={"name": f"{name}_mat"})
                    ET.SubElement(material, "color", attrib={"rgba": _fmt4(rgba)})

            # Collision is enabled by default unless explicitly disabled in PartFlags.
            if mesh_path and collision_enabled:
                collision = ET.SubElement(link_el, "collision")
                ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
                cgeom = ET.SubElement(collision, "geometry")
                ET.SubElement(cgeom, "mesh", attrib={"filename": self._mesh_filename(mesh_path)})

        for j in joints:
            jname = str(_pick(j, "name"))
            jtype_src = str(_pick(j, "type", default="fixed")).lower()
            parent = str(_pick(j, "parent"))
            child = str(_pick(j, "child"))
            jtype = self._normalize_joint_type(jtype_src)

            joint_el = ET.SubElement(robot, "joint", attrib={"name": jname, "type": jtype})
            ET.SubElement(joint_el, "parent", attrib={"link": parent})
            ET.SubElement(joint_el, "child", attrib={"link": child})

            Rp = R_root[parent]
            Rc = R_root[child]
            Pp = P_root[parent]
            Pc = P_root[child]
            d_root = (Pc - Pp).reshape(3)
            xyz_parent = (Rp.T @ d_root).reshape(3)
            R_rel = Rp.T @ Rc
            rpy = Rot.from_matrix(R_rel).as_euler("xyz", degrees=False)

            ET.SubElement(
                joint_el,
                "origin",
                attrib={
                    "xyz": _fmt3((float(xyz_parent[0]), float(xyz_parent[1]), float(xyz_parent[2]))),
                    "rpy": _fmt3((float(rpy[0]), float(rpy[1]), float(rpy[2]))),
                },
            )

            axis_root = np.array(_as_vec3(_pick(j, "axis_root", default=(1, 0, 0))), dtype=float).reshape(3)
            axis_child = _norm3(Rc.T @ axis_root)
            ET.SubElement(
                joint_el,
                "axis",
                attrib={"xyz": _fmt3((float(axis_child[0]), float(axis_child[1]), float(axis_child[2])))},
            )

            if jtype in ("revolute", "prismatic"):
                lo = 0.0
                hi = 0.0
                effort, velocity = self._joint_limit_effort_velocity(jtype)
                damping, friction = self._joint_dynamics(jtype)
                lim_deg = _pick(j, "limit_deg", default=None)
                if jtype == "revolute" and isinstance(lim_deg, (list, tuple)) and len(lim_deg) == 2:
                    lo = math.radians(float(lim_deg[0]))
                    hi = math.radians(float(lim_deg[1]))

                ET.SubElement(
                    joint_el,
                    "limit",
                    attrib={
                        "lower": f"{lo:.9g}",
                        "upper": f"{hi:.9g}",
                        "effort": f"{effort:.9g}",
                        "velocity": f"{velocity:.9g}",
                    },
                )
                ET.SubElement(
                    joint_el,
                    "dynamics",
                    attrib={
                        "damping": f"{damping:.9g}",
                        "friction": f"{friction:.9g}",
                    },
                )

        # Inject Mujoco contact excludes for NoClipPairs.
        # Genesis URDF pipeline parses this via mujoco.MjModel and preserves pair exclusions.
        if no_clip_pairs:
            mujoco_el = ET.SubElement(robot, "mujoco")
            contact_el = ET.SubElement(mujoco_el, "contact")
            valid_links = set(parts_by.keys())
            for item in no_clip_pairs:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    continue
                body1 = str(item[0]).strip()
                body2 = str(item[1]).strip()
                if (not body1) or (not body2) or (body1 == body2):
                    continue
                if (body1 not in valid_links) or (body2 not in valid_links):
                    continue
                ET.SubElement(contact_el, "exclude", attrib={"body1": body1, "body2": body2})

        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ")
        xml = ET.tostring(robot, encoding="unicode")
        return '<?xml version="1.0"?>\n' + xml + "\n"
