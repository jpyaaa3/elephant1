#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from engine.config_loader import UrdfExportConfig


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def _pick(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


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


@dataclass(frozen=True)
class ManifestData:
    parts: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]
    part_flags: Dict[str, Any]
    no_clip_pairs: List[Any]
    parts_by: Dict[str, Dict[str, Any]]
    root: str


@dataclass(frozen=True)
class JointTransformData:
    root: str
    part_rot_root: Dict[str, np.ndarray]
    part_pos_root: Dict[str, np.ndarray]
    valid_links: set[str]


class ManifestLoader:
    def load_file(self, manifest_path: str) -> Tuple[ManifestData, str]:
        with open(manifest_path, "r", encoding="utf-8") as f:
            build = json.load(f)
        build_dir = os.path.dirname(os.path.abspath(manifest_path))
        return self.load_dict(build), build_dir

    def load_dict(self, build: Dict[str, Any]) -> ManifestData:
        parts = list(_pick(build, "parts", default=[]))
        joints = list(_pick(build, "joints", default=[]))
        part_flags = dict(_pick(build, "part_flags", default={}) or {})
        no_clip_pairs = list(_pick(build, "no_clip_pairs", default=[]) or [])

        parts_by = {str(_pick(part, "name")): part for part in parts}
        if not parts_by:
            raise ValueError("manifest.json: parts is empty.")

        parent_of: Dict[str, str] = {}
        for joint in joints:
            parent = str(_pick(joint, "parent"))
            child = str(_pick(joint, "child"))
            if child in parent_of:
                raise ValueError(f"Joint tree error: child '{child}' has multiple parents.")
            parent_of[child] = parent

        roots = [name for name in parts_by.keys() if name not in parent_of]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly 1 root link, got {roots}")

        return ManifestData(
            parts=parts,
            joints=joints,
            part_flags=part_flags,
            no_clip_pairs=no_clip_pairs,
            parts_by=parts_by,
            root=roots[0],
        )


class PhysicsLoader:
    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Vec3, Dict[str, float]]] = {}

    def load(self, build_dir: str, physics_path: str) -> Tuple[float, Vec3, Dict[str, float]]:
        mass = 0.001
        com = (0.0, 0.0, 0.0)
        inertia = {"ixx": 1e-6, "ixy": 0.0, "ixz": 0.0, "iyy": 1e-6, "iyz": 0.0, "izz": 1e-6}

        if not physics_path:
            return mass, com, inertia

        rel = str(physics_path).replace("\\", "/")
        abs_path = os.path.abspath(rel) if os.path.isabs(rel) else os.path.abspath(os.path.join(build_dir, rel))
        cached = self._cache.get(abs_path)
        if cached is not None:
            return cached
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

        result = (mass, com, inertia)
        self._cache[abs_path] = result
        return result


class JointTranslator:
    def translate(self, manifest: ManifestData) -> JointTransformData:
        part_rot_root: Dict[str, np.ndarray] = {}
        part_pos_root: Dict[str, np.ndarray] = {}

        for name, part in manifest.parts_by.items():
            pose_root = _pick(part, "pose_root", default={}) or {}
            q_xyzw = _as_quat_xyzw(_pick(pose_root, "q"))
            p_xyz = _as_vec3(_pick(pose_root, "p"))
            part_rot_root[name] = Rot.from_quat(q_xyzw).as_matrix()
            part_pos_root[name] = np.array(p_xyz, dtype=float)

        return JointTransformData(
            root=manifest.root,
            part_rot_root=part_rot_root,
            part_pos_root=part_pos_root,
            valid_links=set(manifest.parts_by.keys()),
        )


class URDFWriter:
    def __init__(self, cfg: UrdfExportConfig, physics_loader: PhysicsLoader):
        self.cfg = cfg
        self.physics_loader = physics_loader

    def render(self, manifest: ManifestData, transform: JointTransformData, *, build_dir: str) -> str:
        robot = ET.Element("robot", attrib={"name": self.cfg.robot_name})

        for name, part in manifest.parts_by.items():
            self._write_link(robot, name, part, manifest.part_flags, build_dir)

        for joint in manifest.joints:
            self._write_joint(robot, joint, transform)

        self._write_no_clip_pairs(robot, manifest.no_clip_pairs, transform.valid_links)

        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ")
        xml = ET.tostring(robot, encoding="unicode")
        return '<?xml version="1.0"?>\n' + xml + "\n"

    def _mesh_filename(self, path: str) -> str:
        if not path:
            return ""
        norm = path.replace("\\", "/")
        return os.path.basename(norm) if self.cfg.mesh_basename_only else norm

    @staticmethod
    def _normalize_joint_type(jtype_src: str) -> str:
        return jtype_src if jtype_src in ("revolute", "prismatic", "fixed") else "fixed"

    def _part_color_rgba(self, part_name: str) -> Tuple[float, float, float, float] | None:
        return self.cfg.part_color_rgba_by_name.get(str(part_name).strip())

    def _joint_limit_effort_velocity(self, joint_type: str) -> Tuple[float, float]:
        if joint_type == "revolute":
            effort = self.cfg.revolute_effort if self.cfg.revolute_effort is not None else self.cfg.default_effort
            velocity = self.cfg.revolute_velocity if self.cfg.revolute_velocity is not None else self.cfg.default_velocity
            return float(effort), float(velocity)
        if joint_type == "prismatic":
            effort = self.cfg.prismatic_effort if self.cfg.prismatic_effort is not None else self.cfg.default_effort
            velocity = self.cfg.prismatic_velocity if self.cfg.prismatic_velocity is not None else self.cfg.default_velocity
            return float(effort), float(velocity)
        return float(self.cfg.default_effort), float(self.cfg.default_velocity)

    def _joint_dynamics(self, joint_type: str) -> Tuple[float, float]:
        if joint_type == "revolute":
            return float(self.cfg.revolute_damping), float(self.cfg.revolute_friction)
        if joint_type == "prismatic":
            return float(self.cfg.prismatic_damping), float(self.cfg.prismatic_friction)
        return 0.0, 0.0

    def _write_link(
        self,
        robot: ET.Element,
        name: str,
        part: Dict[str, Any],
        part_flags: Dict[str, Any],
        build_dir: str,
    ) -> None:
        link_el = ET.SubElement(robot, "link", attrib={"name": name})

        assets = _pick(part, "assets", default={}) or {}
        mesh_path = str(_pick(assets, "mesh", default="") or "")
        phy_path = str(_pick(assets, "physics", default="") or "")
        flags = (_pick(part, "flags", default=None) or part_flags.get(name, {}) or {})
        collision_enabled = bool(_pick(flags, "collision_enabled", default=True))
        mass, com, inertia = self.physics_loader.load(build_dir, phy_path)

        inertial = ET.SubElement(link_el, "inertial")
        ET.SubElement(inertial, "origin", attrib={"xyz": _fmt3(com), "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", attrib={"value": f"{mass:.9g}"})
        ET.SubElement(
            inertial,
            "inertia",
            attrib={
                "ixx": f"{inertia['ixx']:.9g}",
                "ixy": f"{inertia['ixy']:.9g}",
                "ixz": f"{inertia['ixz']:.9g}",
                "iyy": f"{inertia['iyy']:.9g}",
                "iyz": f"{inertia['iyz']:.9g}",
                "izz": f"{inertia['izz']:.9g}",
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

        if mesh_path and collision_enabled:
            collision = ET.SubElement(link_el, "collision")
            ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            cgeom = ET.SubElement(collision, "geometry")
            ET.SubElement(cgeom, "mesh", attrib={"filename": self._mesh_filename(mesh_path)})

    def _write_joint(self, robot: ET.Element, joint: Dict[str, Any], transform: JointTransformData) -> None:
        joint_name = str(_pick(joint, "name"))
        joint_type_src = str(_pick(joint, "type", default="fixed")).lower()
        parent = str(_pick(joint, "parent"))
        child = str(_pick(joint, "child"))
        joint_type = self._normalize_joint_type(joint_type_src)

        joint_el = ET.SubElement(robot, "joint", attrib={"name": joint_name, "type": joint_type})
        ET.SubElement(joint_el, "parent", attrib={"link": parent})
        ET.SubElement(joint_el, "child", attrib={"link": child})

        rot_parent = transform.part_rot_root[parent]
        rot_child = transform.part_rot_root[child]
        pos_parent = transform.part_pos_root[parent]
        pos_child = transform.part_pos_root[child]
        d_root = (pos_child - pos_parent).reshape(3)
        xyz_parent = (rot_parent.T @ d_root).reshape(3)
        rot_rel = rot_parent.T @ rot_child
        rpy = Rot.from_matrix(rot_rel).as_euler("xyz", degrees=False)

        ET.SubElement(
            joint_el,
            "origin",
            attrib={
                "xyz": _fmt3((float(xyz_parent[0]), float(xyz_parent[1]), float(xyz_parent[2]))),
                "rpy": _fmt3((float(rpy[0]), float(rpy[1]), float(rpy[2]))),
            },
        )

        axis_root = np.array(_as_vec3(_pick(joint, "axis_root", default=(1, 0, 0))), dtype=float).reshape(3)
        axis_child = _norm3(rot_child.T @ axis_root)
        ET.SubElement(
            joint_el,
            "axis",
            attrib={"xyz": _fmt3((float(axis_child[0]), float(axis_child[1]), float(axis_child[2])))},
        )

        if joint_type not in ("revolute", "prismatic"):
            return

        lo = 0.0
        hi = 0.0
        effort, velocity = self._joint_limit_effort_velocity(joint_type)
        damping, friction = self._joint_dynamics(joint_type)
        lim_deg = _pick(joint, "limit_deg", default=None)
        if joint_type == "revolute" and isinstance(lim_deg, (list, tuple)) and len(lim_deg) == 2:
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

    def _write_no_clip_pairs(self, robot: ET.Element, no_clip_pairs: List[Any], valid_links: set[str]) -> None:
        if not no_clip_pairs:
            return

        mujoco_el = ET.SubElement(robot, "mujoco")
        contact_el = ET.SubElement(mujoco_el, "contact")
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


def convert_manifest_file(
    assy_build_json_path: str,
    urdf_out_path: str,
    cfg: UrdfExportConfig = UrdfExportConfig(),
) -> None:
    manifest_loader = ManifestLoader()
    physics_loader = PhysicsLoader()
    joint_translator = JointTranslator()
    urdf_writer = URDFWriter(cfg, physics_loader)

    manifest, build_dir = manifest_loader.load_file(assy_build_json_path)
    transform = joint_translator.translate(manifest)
    urdf_text = urdf_writer.render(manifest, transform, build_dir=build_dir)

    os.makedirs(os.path.dirname(os.path.abspath(urdf_out_path)), exist_ok=True)
    with open(urdf_out_path, "w", encoding="utf-8") as f:
        f.write(urdf_text)


def convert_manifest_dict(
    build: Dict[str, Any],
    *,
    build_dir: str,
    cfg: UrdfExportConfig = UrdfExportConfig(),
) -> str:
    manifest_loader = ManifestLoader()
    physics_loader = PhysicsLoader()
    joint_translator = JointTranslator()
    urdf_writer = URDFWriter(cfg, physics_loader)

    manifest = manifest_loader.load_dict(build)
    transform = joint_translator.translate(manifest)
    return urdf_writer.render(manifest, transform, build_dir=build_dir)
