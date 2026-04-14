#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from builder.robot_defs import (
    AssemblyPlan,
    ConnectorSpec,
    ControlMode,
    JointAxisRule,
    JointLayout,
    JointLayoutResult,
    JointSpec,
    JointType,
    ManifestBuildResult,
    PartAssets,
    PartInstance,
    PartKind,
    PartOverride,
    PartSpec,
    Pose,
    RobotBuildConfig,
    RuntimePartProps,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BUILD_DIR = os.path.join(PROJECT_ROOT, "craft")
DEFAULT_ASSET_ROOT_DIR = os.path.join(PROJECT_ROOT, "assets")


def _pick_named_file(dir_path: str, names: List[str]) -> Optional[str]:
    lowered = {str(name).strip().lower() for name in names}
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        if os.path.isfile(full) and entry.strip().lower() in lowered:
            return full
    return None


def _apply_rot(rot: Rot, v: tuple[float, float, float]) -> tuple[float, float, float]:
    out = rot.apply(v)
    return (float(out[0]), float(out[1]), float(out[2]))


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.array(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (float(arr[0] / n), float(arr[1] / n), float(arr[2] / n))


def _load_connector_spec_from_static_frame(kind: PartKind) -> ConnectorSpec:
    frame_path = os.path.join(DEFAULT_ASSET_ROOT_DIR, kind.value, f"{kind.value}_frame.json")
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"missing frame.json for kind '{kind.value}': {frame_path}")

    try:
        with open(frame_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        connectors = data.get("connectors", {}) or {}
        from_raw = connectors.get("from")
        to_raw = connectors.get("to")
        if (
            isinstance(from_raw, (list, tuple))
            and len(from_raw) == 3
            and isinstance(to_raw, (list, tuple))
            and len(to_raw) == 3
        ):
            return ConnectorSpec(
                from_pos=(float(from_raw[0]), float(from_raw[1]), float(from_raw[2])),
                to_pos=(float(to_raw[0]), float(to_raw[1]), float(to_raw[2])),
            )
    except Exception as exc:
        raise ValueError(f"failed to parse frame.json for kind '{kind.value}': {frame_path}") from exc
    raise ValueError(f"frame.json for kind '{kind.value}' must define connectors.from and connectors.to as vec3: {frame_path}")


def make_default_config() -> RobotBuildConfig:
    connectors = {
        kind: _load_connector_spec_from_static_frame(kind)
        for kind in (
            PartKind.plate,
            PartKind.housing,
            PartKind.wedge,
            PartKind.node,
            PartKind.node_end,
        )
    }
    return RobotBuildConfig(
        plate=PartSpec(connectors=connectors[PartKind.plate]),
        housing=PartSpec(connectors=connectors[PartKind.housing]),
        wedge=PartSpec(connectors=connectors[PartKind.wedge]),
        node=PartSpec(connectors=connectors[PartKind.node]),
        node_end=PartSpec(connectors=connectors[PartKind.node_end]),
        joint_axis_rules={
            "plate_housing_linear": JointAxisRule(axis_parent_local=(1.0, 0.0, 0.0)),
            "housing_wedge": JointAxisRule(axis_parent_local=(1.0, 0.0, 0.0)),
            "wedge_node": JointAxisRule(axis_parent_local=(0.0, 1.0, 0.0)),
            "node_node": JointAxisRule(axis_parent_local=(0.0, 1.0, 0.0)),
        },
    )


class AssetFinder:
    def __init__(self, root_dir: str = DEFAULT_ASSET_ROOT_DIR):
        self._root_dir = root_dir
        self._cache: Dict[PartKind, PartAssets] = {}

    def resolve_assets(self, kind: PartKind) -> PartAssets:
        cached = self._cache.get(kind)
        if cached is not None:
            return cached

        source_dir = self._find_asset_dir(kind)
        if source_dir is None:
            raise FileNotFoundError(f"missing static asset directory for kind '{kind.value}'")

        source = self._resolve_asset_files(source_dir, kind)
        if source is None:
            raise FileNotFoundError(f"missing static asset files for kind '{kind.value}' in '{source_dir}'")

        self._cache[kind] = source
        return source

    def _find_asset_dir(self, kind: PartKind) -> Optional[str]:
        if not os.path.isdir(self._root_dir):
            return None
        full = os.path.join(self._root_dir, kind.value)
        return full if os.path.isdir(full) else None

    def _resolve_asset_files(self, dir_path: str, kind: PartKind) -> Optional[PartAssets]:
        base = kind.value
        mesh = _pick_named_file(dir_path, [f"{base}_mesh.obj"])
        frame = _pick_named_file(dir_path, [f"{base}_frame.json"])
        physics = _pick_named_file(dir_path, [f"{base}_physics.json"])
        if mesh is None or frame is None or physics is None:
            return None
        return PartAssets(mesh_path=mesh, frame_path=frame, physics_path=physics)


class PartPolicySetter:
    def __init__(self):
        self._use_hardware: bool = False
        self._use_go2: bool = False
        self._overrides: Dict[str, PartOverride] = {}
        self._no_clip_pairs: Set[tuple[str, str]] = set()

    def set_use_hardware(self, enabled: bool) -> None:
        self._use_hardware = bool(enabled)

    def get_use_hardware(self) -> bool:
        return self._use_hardware

    def set_use_go2(self, enabled: bool) -> None:
        self._use_go2 = bool(enabled)

    def get_use_go2(self) -> bool:
        return self._use_go2

    def add_no_clip(self, part_a: str, part_b: str) -> None:
        a, b = str(part_a), str(part_b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        self._no_clip_pairs.add((a, b))

    def get_no_clip_pairs(self) -> Set[tuple[str, str]]:
        return self._no_clip_pairs.copy()

    @staticmethod
    def _is_go2_part(part_name: str, kind: PartKind) -> bool:
        return str(part_name).strip().lower() == "go2"

    @staticmethod
    def _is_fixed_base_part(part_name: str, kind: PartKind) -> bool:
        return kind == PartKind.plate

    @staticmethod
    def _is_controlled_part(part_name: str, kind: PartKind) -> bool:
        return kind in (PartKind.housing, PartKind.wedge, PartKind.node, PartKind.node_end)

    def resolve_runtime_props(self, part_name: str, kind: PartKind) -> RuntimePartProps:
        name = str(part_name).strip().lower()
        collision = True
        mode = self._default_mode(name, kind)

        ov = self._overrides.get(part_name)
        if ov is not None:
            if ov.control_mode is not None:
                mode = ov.control_mode
            if ov.collision_enabled is not None:
                collision = ov.collision_enabled
        return RuntimePartProps(control_mode=mode, collision_enabled=collision)

    def _default_mode(self, part_name: str, kind: PartKind) -> ControlMode:
        if self._use_hardware:
            if self._use_go2:
                return ControlMode.simulated
            if self._is_fixed_base_part(part_name, kind):
                return ControlMode.fixed
            return ControlMode.simulated

        if self._is_go2_part(part_name, kind) and self._use_go2:
            return ControlMode.fixed
        if self._is_fixed_base_part(part_name, kind):
            return ControlMode.fixed
        if self._is_controlled_part(part_name, kind):
            return ControlMode.commanded
        return ControlMode.fixed


class AssemblyDesigner:
    def __init__(self, config: RobotBuildConfig):
        if config.joint_axis_rules is None:
            raise ValueError("config.joint_axis_rules must be provided")
        self._cfg = config

    def create_default_robot_graph(self) -> AssemblyPlan:
        robot_graph = AssemblyPlan(root_part_name="plate")
        robot_graph.add_part(PartInstance("plate", PartKind.plate, self._cfg.plate.connectors))
        robot_graph.add_part(PartInstance("housing", PartKind.housing, self._cfg.housing.connectors))
        robot_graph.add_part(PartInstance("wedge", PartKind.wedge, self._cfg.wedge.connectors))
        node_count = self._cfg.node_count
        for i in range(node_count):
            if i == node_count - 1:
                robot_graph.add_part(PartInstance(f"node{i}", PartKind.node_end, self._cfg.node_end.connectors))
            else:
                robot_graph.add_part(PartInstance(f"node{i}", PartKind.node, self._cfg.node.connectors))

        robot_graph.connect("plate", "housing", JointSpec(name="j_plate_housing", type=JointType.prismatic, axis_rule_key="plate_housing_linear"))
        robot_graph.connect("housing", "wedge", JointSpec(name="j_housing_wedge", type=JointType.revolute, axis_rule_key="housing_wedge"))
        robot_graph.connect("wedge", "node0", JointSpec(name="j_wedge_node0", type=JointType.revolute, axis_rule_key="wedge_node"))
        for i in range(node_count - 1):
            robot_graph.connect(
                f"node{i}",
                f"node{i+1}",
                JointSpec(name=f"j_node{i}_node{i+1}", type=JointType.revolute, axis_rule_key="node_node"),
            )
        return robot_graph

    def resolve_joint_layout(self, robot_graph: AssemblyPlan) -> JointLayoutResult:
        robot_graph.validate()
        parts = robot_graph.get_parts()
        edges = robot_graph.get_edges()
        root = robot_graph.get_root_name()
        poses: Dict[str, Pose] = {root: self._cfg.root_part_pose}
        rot_cache: Dict[str, Rot] = {root: Rot.from_quat(self._cfg.root_part_pose.q)}

        outgoing: Dict[str, List] = {}
        for edge in edges:
            outgoing.setdefault(edge.parent, []).append(edge)

        def get_rot(part_name: str) -> Rot:
            rot = rot_cache.get(part_name)
            if rot is None:
                rot = Rot.from_quat(poses[part_name].q)
                rot_cache[part_name] = rot
            return rot

        stack = [root]
        while stack:
            parent = stack.pop()
            if parent not in outgoing:
                continue
            parent_pose = poses[parent]
            parent_p, parent_q = parent_pose.p, parent_pose.q
            parent_rot = get_rot(parent)
            parent_part = parts[parent]
            for edge in outgoing[parent]:
                child = edge.child
                child_part = parts[child]
                parent_to_local = parent_part.get_connector_end_pos(edge.parent_to)
                parent_to_rot = _apply_rot(parent_rot, parent_to_local)
                parent_to_root = tuple(parent_p[i] + parent_to_rot[i] for i in range(3))
                child_q = parent_q
                child_rot = parent_rot
                child_from_local = child_part.get_connector_end_pos(edge.child_from)
                child_from_root = _apply_rot(child_rot, child_from_local)
                child_p = (
                    parent_to_root[0] - child_from_root[0],
                    parent_to_root[1] - child_from_root[1],
                    parent_to_root[2] - child_from_root[2],
                )
                poses[child] = Pose(p=child_p, q=child_q)
                rot_cache[child] = child_rot
                stack.append(child)

        joints: List[JointLayout] = []
        for edge in edges:
            parent_pose = poses[edge.parent]
            parent_p = parent_pose.p
            parent_rot = get_rot(edge.parent)
            parent_part = parts[edge.parent]
            parent_to_local = parent_part.get_connector_end_pos(edge.parent_to)
            parent_to_rot = _apply_rot(parent_rot, parent_to_local)
            anchor_root = (
                parent_p[0] + parent_to_rot[0],
                parent_p[1] + parent_to_rot[1],
                parent_p[2] + parent_to_rot[2],
            )
            rule = self._cfg.joint_axis_rules[edge.spec.axis_rule_key]
            axis_root = _normalize(_apply_rot(parent_rot, rule.axis_parent_local))
            joints.append(
                JointLayout(
                    name=edge.spec.name,
                    type=edge.spec.type.value,
                    parent=edge.parent,
                    child=edge.child,
                    anchor_root=anchor_root,
                    axis_root=axis_root,
                    limit_deg=edge.spec.limit_deg,
                )
            )
        return JointLayoutResult(part_poses_root=poses, joints=joints)


class ManifestWriter:
    def __init__(self, config: RobotBuildConfig, policy: PartPolicySetter, asset_finder: Optional[AssetFinder] = None):
        self._cfg = config
        self._policy = policy
        self._asset_finder = asset_finder or AssetFinder()

    def build(self, robot_graph: AssemblyPlan, layout: JointLayoutResult, out_dir: str) -> ManifestBuildResult:
        os.makedirs(out_dir, exist_ok=True)
        parts = robot_graph.get_parts()
        emitted = self._load_assets(parts)
        self._record_no_clip_pairs(layout.joints)
        flags = self._resolve_flags(parts)
        manifest = self._build_manifest_dict(parts, emitted, layout, flags, out_dir)
        manifest_path = self._write_manifest(out_dir, manifest)
        return ManifestBuildResult(build_dir=out_dir, manifest_path=manifest_path)

    def _load_assets(self, parts: Dict[str, PartInstance]) -> Dict[str, PartAssets]:
        emitted: Dict[str, PartAssets] = {}
        for name, part in parts.items():
            emitted[name] = self._asset_finder.resolve_assets(part.kind)
        return emitted

    def _record_no_clip_pairs(self, joints: List[JointLayout]) -> None:
        for joint in joints:
            self._policy.add_no_clip(joint.parent, joint.child)

    def _resolve_flags(self, parts: Dict[str, PartInstance]) -> Dict[str, Dict[str, object]]:
        flags: Dict[str, Dict[str, object]] = {}
        for name, part in parts.items():
            props = self._policy.resolve_runtime_props(name, part.kind)
            flags[name] = {"control_mode": props.control_mode.value, "collision_enabled": props.collision_enabled}
        return flags

    def _build_manifest_dict(
        self,
        parts: Dict[str, PartInstance],
        emitted: Dict[str, PartAssets],
        layout: JointLayoutResult,
        flags: Dict[str, Dict[str, object]],
        out_dir: str,
    ) -> Dict[str, object]:
        manifest: Dict[str, object] = {
            "meta": {
                "node_count": self._cfg.node_count,
                "use_hardware": self._policy.get_use_hardware(),
                "use_go2": self._policy.get_use_go2(),
                "notes": "All poses/joints are root-relative.",
            },
            "parts": [],
            "joints": [],
            "no_clip_pairs": sorted(list(self._policy.get_no_clip_pairs())),
        }

        for name, part in parts.items():
            pose = layout.part_poses_root.get(name)
            if pose is None:
                raise RuntimeError(f"resolver did not produce Pose for part '{name}'")
            assets = emitted[name]
            manifest["parts"].append(
                {
                    "name": name,
                    "kind": part.kind.value,
                    "assets": {
                        "mesh": os.path.relpath(assets.mesh_path, out_dir),
                        "frame": os.path.relpath(assets.frame_path, out_dir),
                        "physics": os.path.relpath(assets.physics_path, out_dir),
                    },
                    "flags": flags[name],
                    "pose_root": {"p": list(pose.p), "q": list(pose.q)},
                }
            )

        for joint in layout.joints:
            manifest["joints"].append(
                {
                    "name": joint.name,
                    "type": joint.type,
                    "parent": joint.parent,
                    "child": joint.child,
                    "anchor_root": list(joint.anchor_root),
                    "axis_root": list(joint.axis_root),
                    "limit_deg": list(joint.limit_deg) if joint.limit_deg is not None else None,
                }
            )
        return manifest

    def _write_manifest(self, out_dir: str, manifest: Dict[str, object]) -> str:
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


def build_default_manifest(
    output_dir: str,
    *,
    use_hardware: bool = False,
    use_go2: bool = False,
) -> str:
    config = make_default_config()
    asset_finder = AssetFinder()
    policy = PartPolicySetter()
    policy.set_use_hardware(use_hardware)
    policy.set_use_go2(use_go2)
    designer = AssemblyDesigner(config)
    robot_graph = designer.create_default_robot_graph()
    layout = designer.resolve_joint_layout(robot_graph)
    result = ManifestWriter(config, policy, asset_finder=asset_finder).build(robot_graph, layout, output_dir)
    return result.manifest_path
