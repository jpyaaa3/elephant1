#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .parts import housing as _housing
from .parts import node as _node
from .parts import node_end as _node_end
from .parts import plate as _plate
from .parts import wedge as _wedge


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x, y, z, w)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
DEFAULT_CUSTOM_ASSET_DIR = os.path.join(os.path.dirname(__file__), "custom_obj")
DEFAULT_DENSITY_BY_KIND = {
    "plate": 2700.0,
    "housing": 1240.0,
    "wedge": 1240.0,
    "node": 1240.0,
    "node_end": 1240.0,
}


@dataclass(frozen=True)
class Pose:
    p: Vec3 = (0.0, 0.0, 0.0)
    q: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PhysicsSpec:
    mass: Optional[float] = None
    com: Optional[Vec3] = None
    inertia: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class JointAxisRule:
    axis_parent_local: Vec3


@dataclass(frozen=True)
class ConnectorSpec:
    from_pos: Vec3
    to_pos: Vec3


@dataclass(frozen=True)
class NodeSpec:
    param: _node.NodeConfig
    connectors: ConnectorSpec
    physics: PhysicsSpec = PhysicsSpec()


@dataclass(frozen=True)
class NodeEndSpec:
    param: _node_end.NodeEndConfig
    connectors: ConnectorSpec
    physics: PhysicsSpec = PhysicsSpec()


@dataclass(frozen=True)
class HousingSpec:
    param: _housing.HousingConfig
    connectors: ConnectorSpec
    physics: PhysicsSpec = PhysicsSpec()


@dataclass(frozen=True)
class WedgeSpec:
    param: _wedge.WedgeConfig
    connectors: ConnectorSpec
    physics: PhysicsSpec = PhysicsSpec()


@dataclass(frozen=True)
class PlateSpec:
    param: _plate.PlateConfig
    connectors: ConnectorSpec
    physics: PhysicsSpec = PhysicsSpec()


@dataclass(frozen=True)
class AssemblyConfig:
    node_count: int = 10
    root_part_pose: Pose = Pose()
    plate: PlateSpec = PlateSpec(
        param=_plate.PlateConfig(),
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, -0.005), to_pos=(0.0, 0.0, 0.0)),
        physics=PhysicsSpec(),
    )
    node: NodeSpec = NodeSpec(
        param=_node.NodeConfig(),
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.05, 0.0, 0.0)),
        physics=PhysicsSpec(),
    )
    node_end: NodeEndSpec = NodeEndSpec(
        param=_node_end.NodeEndConfig(),
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.05, 0.0, 0.0)),
        physics=PhysicsSpec(),
    )
    housing: HousingSpec = HousingSpec(
        param=_housing.HousingConfig(),
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.0, 0.0, float(_housing.HousingConfig().height))),
        physics=PhysicsSpec(),
    )
    wedge: WedgeSpec = WedgeSpec(
        param=_wedge.WedgeConfig(),
        connectors=ConnectorSpec(
            from_pos=(0.0, 0.0, 0.0),
            to_pos=(float(_wedge.WedgeConfig().wedge_e), 0.0, 0.0),
        ),
        physics=PhysicsSpec(),
    )
    joint_axis_rules: Optional[Dict[str, JointAxisRule]] = None


def make_default_config() -> AssemblyConfig:
    return AssemblyConfig(
        joint_axis_rules={
            "plate_housing_linear": JointAxisRule(axis_parent_local=(1.0, 0.0, 0.0)),
            "housing_wedge": JointAxisRule(axis_parent_local=(1.0, 0.0, 0.0)),
            # Positive bend angles should map to the physical +Y bending convention.
            "wedge_node": JointAxisRule(axis_parent_local=(0.0, 1.0, 0.0)),
            "node_node": JointAxisRule(axis_parent_local=(0.0, 1.0, 0.0)),
        }
    )


class PartKind(Enum):
    plate = "plate"
    housing = "housing"
    wedge = "wedge"
    node = "node"
    node_end = "node_end"


@dataclass(frozen=True)
class PartAssets:
    mesh_path: str
    frame_path: str
    physics_path: str


class PartComponent:
    def __init__(self, name: str, kind: PartKind):
        self._name = name
        self._kind = kind

    def get_name(self) -> str:
        return self._name

    def get_kind(self) -> PartKind:
        return self._kind

    def get_connector_from_pos(self) -> Vec3:
        raise NotImplementedError

    def get_connector_to_pos(self) -> Vec3:
        raise NotImplementedError

    def get_connector_pos(self, key: str) -> Vec3:
        if key in ("From", "from"):
            return self.get_connector_from_pos()
        if key in ("To", "to"):
            return self.get_connector_to_pos()
        raise KeyError(f"unknown connector key: {key}")

    def emit_assets(self, out_dir: str) -> PartAssets:
        os.makedirs(out_dir, exist_ok=True)
        assets = self._emit_custom_assets(out_dir)
        if assets is None:
            assets = self.emit_generated_assets(out_dir)
        self.postprocess_emitted_assets(assets)
        return assets

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        raise NotImplementedError

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        pass

    def _asset_paths(self, out_dir: str) -> PartAssets:
        base = self.get_name().strip().lower()
        mesh_path = os.path.join(out_dir, f"{base}_mesh.obj")
        return PartAssets(
            mesh_path=mesh_path,
            frame_path=os.path.join(out_dir, f"{base}_frame.json"),
            physics_path=os.path.join(out_dir, f"{base}_physics.json"),
        )

    def _emit_custom_assets(self, out_dir: str) -> Optional[PartAssets]:
        custom_dir = _find_custom_asset_dir(self.get_name(), self.get_kind())
        if custom_dir is None:
            return None
        source = _resolve_custom_asset_files(custom_dir, self.get_name(), self.get_kind())
        if source is None:
            return None
        target = self._asset_paths(out_dir)
        shutil.copyfile(source.mesh_path, target.mesh_path)
        shutil.copyfile(source.frame_path, target.frame_path)
        if source.physics_path:
            shutil.copyfile(source.physics_path, target.physics_path)
        else:
            _write_estimated_physics_file(target.physics_path, target.mesh_path, self.get_kind())
        return target


def _write_frame_only_from_to(path: str, from_pos: Vec3, to_pos: Vec3) -> None:
    payload = {"connectors": {"from": [float(x) for x in from_pos], "to": [float(x) for x in to_pos]}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _override_frame_connectors_file(path: str, from_pos: Vec3, to_pos: Vec3) -> None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    data["connectors"] = {
        "from": [float(x) for x in from_pos],
        "to": [float(x) for x in to_pos],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _override_physics_file(path: str, spec: PhysicsSpec) -> None:
    if spec.mass is None and spec.com is None and spec.inertia is None:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if spec.mass is not None:
        data["mass"] = float(spec.mass)
    if spec.com is not None:
        data["com"] = [float(x) for x in spec.com]
    if spec.inertia is not None:
        data["inertia"] = {str(k).lower(): float(v) for k, v in spec.inertia.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _find_custom_asset_dir(part_name: str, kind: PartKind) -> Optional[str]:
    if not os.path.isdir(DEFAULT_CUSTOM_ASSET_DIR):
        return None
    part_name = str(part_name).strip().lower()
    wanted: List[str] = []
    if kind == PartKind.plate:
        wanted.append("plate")
    elif kind == PartKind.housing:
        wanted.append("housing")
    elif kind == PartKind.wedge:
        wanted.append("wedge")
    elif kind == PartKind.node:
        wanted.append("node")
    elif kind == PartKind.node_end:
        wanted.append("node_end")
    for entry in wanted:
        full = os.path.join(DEFAULT_CUSTOM_ASSET_DIR, entry)
        if os.path.isdir(full):
            return full
    return None


def _pick_named_file(dir_path: str, names: List[str]) -> Optional[str]:
    lowered = {str(name).strip().lower() for name in names}
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        if os.path.isfile(full) and entry.strip().lower() in lowered:
            return full
    return None


def _load_obj_triangles(path: str) -> tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if line.startswith("f "):
                toks = line.split()[1:]
                idxs: List[int] = []
                for tok in toks:
                    head = tok.split("/")[0].strip()
                    if not head:
                        continue
                    idx = int(head)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    idxs.append(idx)
                if len(idxs) < 3:
                    continue
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable mesh data: {path}")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _estimate_physics_from_mesh(mesh_path: str, kind: PartKind) -> Dict[str, object]:
    vertices, faces = _load_obj_triangles(mesh_path)
    density = float(DEFAULT_DENSITY_BY_KIND.get(kind.value, 1000.0))

    vol6 = 0.0
    first = np.zeros(3, dtype=float)
    second = np.zeros(3, dtype=float)
    prod = np.zeros(3, dtype=float)  # xy, yz, zx

    for tri in faces:
        a = vertices[int(tri[0])]
        b = vertices[int(tri[1])]
        c = vertices[int(tri[2])]
        det = float(np.dot(a, np.cross(b, c)))
        vol6 += det

        first += det * (a + b + c) / 24.0

        second[0] += det * (
            a[0] * a[0] + b[0] * b[0] + c[0] * c[0] + a[0] * b[0] + b[0] * c[0] + c[0] * a[0]
        ) / 60.0
        second[1] += det * (
            a[1] * a[1] + b[1] * b[1] + c[1] * c[1] + a[1] * b[1] + b[1] * c[1] + c[1] * a[1]
        ) / 60.0
        second[2] += det * (
            a[2] * a[2] + b[2] * b[2] + c[2] * c[2] + a[2] * b[2] + b[2] * c[2] + c[2] * a[2]
        ) / 60.0

        prod[0] += det * (
            2.0 * (a[0] * a[1] + b[0] * b[1] + c[0] * c[1])
            + (a[0] * b[1] + a[1] * b[0] + a[0] * c[1] + a[1] * c[0] + b[0] * c[1] + b[1] * c[0])
        ) / 120.0
        prod[1] += det * (
            2.0 * (a[1] * a[2] + b[1] * b[2] + c[1] * c[2])
            + (a[1] * b[2] + a[2] * b[1] + a[1] * c[2] + a[2] * c[1] + b[1] * c[2] + b[2] * c[1])
        ) / 120.0
        prod[2] += det * (
            2.0 * (a[2] * a[0] + b[2] * b[0] + c[2] * c[0])
            + (a[2] * b[0] + a[0] * b[2] + a[2] * c[0] + a[0] * c[2] + b[2] * c[0] + b[0] * c[2])
        ) / 120.0

    volume = vol6 / 6.0
    if abs(volume) <= 1e-12:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        size = np.maximum(maxs - mins, 1e-9)
        com = 0.5 * (mins + maxs)
        mass = density * float(size[0] * size[1] * size[2])
        inertia = {
            "ixx": (mass / 12.0) * float(size[1] * size[1] + size[2] * size[2]),
            "iyy": (mass / 12.0) * float(size[0] * size[0] + size[2] * size[2]),
            "izz": (mass / 12.0) * float(size[0] * size[0] + size[1] * size[1]),
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }
        return {
            "unit": {"mass": "kg", "length": "m"},
            "mass": float(mass),
            "com": [float(com[0]), float(com[1]), float(com[2])],
            "inertia": inertia,
            "notes": {"info": "Estimated from mesh AABB fallback with uniform density.", "density_kg_m3": density},
        }

    sign = 1.0 if volume > 0.0 else -1.0
    volume *= sign
    first *= sign
    second *= sign
    prod *= sign

    com = first / volume
    mass = density * volume

    i_origin = np.array(
        [
            [density * (second[1] + second[2]), -density * prod[0], -density * prod[2]],
            [-density * prod[0], density * (second[0] + second[2]), -density * prod[1]],
            [-density * prod[2], -density * prod[1], density * (second[0] + second[1])],
        ],
        dtype=float,
    )
    c = np.asarray(com, dtype=float)
    shift = mass * (((float(np.dot(c, c))) * np.eye(3)) - np.outer(c, c))
    i_com = i_origin - shift
    i_com = 0.5 * (i_com + i_com.T)

    return {
        "unit": {"mass": "kg", "length": "m"},
        "mass": float(mass),
        "com": [float(c[0]), float(c[1]), float(c[2])],
        "inertia": {
            "ixx": float(max(i_com[0, 0], 1e-12)),
            "iyy": float(max(i_com[1, 1], 1e-12)),
            "izz": float(max(i_com[2, 2], 1e-12)),
            "ixy": float(i_com[0, 1]),
            "ixz": float(i_com[0, 2]),
            "iyz": float(i_com[1, 2]),
        },
        "notes": {"info": "Estimated from closed triangle mesh with uniform density.", "density_kg_m3": density},
    }


def _write_estimated_physics_file(path: str, mesh_path: str, kind: PartKind) -> None:
    payload = _estimate_physics_from_mesh(mesh_path, kind)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_custom_asset_files(dir_path: str, part_name: str, kind: PartKind) -> Optional[PartAssets]:
    part_name = str(part_name).strip().lower()
    visual = _pick_named_file(dir_path, [f"{part_name}_mesh.obj"])
    frame = _pick_named_file(dir_path, [f"{part_name}_frame.json"])
    physics = _pick_named_file(dir_path, [f"{part_name}_physics.json"])
    if visual is None or frame is None:
        return None
    return PartAssets(mesh_path=visual, frame_path=frame, physics_path=physics or "")


class NodeComponent(PartComponent):
    def __init__(self, index: int, spec: NodeSpec):
        super().__init__(name=f"node{index}", kind=PartKind.node)
        self._index = int(index)
        self._spec = spec

    def get_index(self) -> int:
        return self._index

    def get_connector_from_pos(self) -> Vec3:
        return self._spec.connectors.from_pos

    def get_connector_to_pos(self) -> Vec3:
        return self._spec.connectors.to_pos

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        p = _node.NodeGenerator(self._spec.param).make(out_dir)
        return PartAssets(mesh_path=p.mesh, frame_path=p.frame, physics_path=p.physics)

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        _override_frame_connectors_file(assets.frame_path, self.get_connector_from_pos(), self.get_connector_to_pos())
        _override_physics_file(assets.physics_path, self._spec.physics)


class NodeEndComponent(PartComponent):
    def __init__(self, index: int, spec: NodeEndSpec):
        super().__init__(name=f"node{index}", kind=PartKind.node_end)
        self._index = int(index)
        self._spec = spec

    def get_index(self) -> int:
        return self._index

    def get_connector_from_pos(self) -> Vec3:
        return self._spec.connectors.from_pos

    def get_connector_to_pos(self) -> Vec3:
        return self._spec.connectors.to_pos

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        p = _node_end.NodeEndGenerator(self._spec.param).make(out_dir)
        return PartAssets(mesh_path=p.mesh, frame_path=p.frame, physics_path=p.physics)

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        _override_frame_connectors_file(assets.frame_path, self.get_connector_from_pos(), self.get_connector_to_pos())
        _override_physics_file(assets.physics_path, self._spec.physics)


class PlateComponent(PartComponent):
    def __init__(self, spec: PlateSpec):
        super().__init__(name="plate", kind=PartKind.plate)
        self._spec = spec

    def get_connector_from_pos(self) -> Vec3:
        return self._spec.connectors.from_pos

    def get_connector_to_pos(self) -> Vec3:
        return self._spec.connectors.to_pos

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        p = _plate.PlateGenerator(self._spec.param).make(out_dir)
        return PartAssets(mesh_path=p.mesh, frame_path=p.frame, physics_path=p.physics)

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        _override_frame_connectors_file(assets.frame_path, self.get_connector_from_pos(), self.get_connector_to_pos())
        _override_physics_file(assets.physics_path, self._spec.physics)


class HousingComponent(PartComponent):
    def __init__(self, spec: HousingSpec):
        super().__init__(name="housing", kind=PartKind.housing)
        self._spec = spec

    def get_connector_from_pos(self) -> Vec3:
        return self._spec.connectors.from_pos

    def get_connector_to_pos(self) -> Vec3:
        return self._spec.connectors.to_pos

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        p = _housing.HousingGenerator(self._spec.param).make(out_dir)
        return PartAssets(mesh_path=p.mesh, frame_path=p.frame, physics_path=p.physics)

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        _override_frame_connectors_file(assets.frame_path, self.get_connector_from_pos(), self.get_connector_to_pos())
        _override_physics_file(assets.physics_path, self._spec.physics)


class WedgeComponent(PartComponent):
    def __init__(self, spec: WedgeSpec):
        super().__init__(name="wedge", kind=PartKind.wedge)
        self._spec = spec

    def get_connector_from_pos(self) -> Vec3:
        return self._spec.connectors.from_pos

    def get_connector_to_pos(self) -> Vec3:
        return self._spec.connectors.to_pos

    def emit_generated_assets(self, out_dir: str) -> PartAssets:
        p = _wedge.WedgeGenerator(self._spec.param).make(out_dir)
        return PartAssets(mesh_path=p.mesh, frame_path=p.frame, physics_path=p.physics)

    def postprocess_emitted_assets(self, assets: PartAssets) -> None:
        _override_frame_connectors_file(assets.frame_path, self.get_connector_from_pos(), self.get_connector_to_pos())
        _override_physics_file(assets.physics_path, self._spec.physics)


class ControlMode(Enum):
    fixed = "fixed"
    commanded = "commanded"
    simulated = "simulated"


@dataclass
class PartOverride:
    ControlMode: Optional[ControlMode] = None
    collision_enabled: Optional[bool] = None


@dataclass(frozen=True)
class ResolvedPartProps:
    ControlMode: ControlMode
    collision_enabled: bool


class PartSimulationPolicy:
    def __init__(self):
        self._use_hardware: bool = False
        self._use_go2: bool = False
        self._overrides: Dict[str, PartOverride] = {}
        self._no_clip_pairs: Set[Tuple[str, str]] = set()

    def set_use_hardware(self, enabled: bool) -> None:
        self._use_hardware = bool(enabled)

    def get_use_hardware(self) -> bool:
        return bool(self._use_hardware)

    def set_use_go2(self, enabled: bool) -> None:
        self._use_go2 = bool(enabled)

    def get_use_go2(self) -> bool:
        return bool(self._use_go2)

    def override_control_mode(self, part_name: str, mode: ControlMode) -> None:
        ov = self._overrides.get(part_name, PartOverride())
        ov.ControlMode = mode
        self._overrides[part_name] = ov

    def override_collision(self, part_name: str, enabled: bool) -> None:
        ov = self._overrides.get(part_name, PartOverride())
        ov.collision_enabled = bool(enabled)
        self._overrides[part_name] = ov

    def add_no_clip(self, part_a: str, part_b: str) -> None:
        a, b = str(part_a), str(part_b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        self._no_clip_pairs.add((a, b))

    def get_no_clip_pairs(self) -> Set[Tuple[str, str]]:
        return set(self._no_clip_pairs)

    @staticmethod
    def _is_go2_part(part_name: str, kind: PartKind) -> bool:
        return str(part_name).strip().lower() == "go2"

    @staticmethod
    def _is_fixed_base_part(part_name: str, kind: PartKind) -> bool:
        return kind == PartKind.plate

    @staticmethod
    def _is_controlled_part(part_name: str, kind: PartKind) -> bool:
        return kind in (PartKind.housing, PartKind.wedge, PartKind.node, PartKind.node_end)

    def resolve_part_props(self, part_name: str, kind: PartKind) -> ResolvedPartProps:
        name = str(part_name).strip().lower()
        mode = ControlMode.fixed
        collision = True

        if self._use_hardware:
            if self._use_go2:
                mode = ControlMode.simulated
            else:
                if self._is_fixed_base_part(name, kind):
                    mode = ControlMode.fixed
                else:
                    mode = ControlMode.simulated
        else:
            if self._is_go2_part(name, kind) and self._use_go2:
                mode = ControlMode.fixed
            elif self._is_fixed_base_part(name, kind):
                mode = ControlMode.fixed
            elif self._is_controlled_part(name, kind):
                mode = ControlMode.commanded

        ov = self._overrides.get(part_name)
        if ov is not None:
            if ov.ControlMode is not None:
                mode = ov.ControlMode
            if ov.collision_enabled is not None:
                collision = ov.collision_enabled
        return ResolvedPartProps(ControlMode=mode, collision_enabled=collision)


class JointType(Enum):
    fixed = "fixed"
    revolute = "revolute"
    prismatic = "prismatic"


@dataclass(frozen=True)
class JointSpec:
    name: str
    type: JointType
    axis_rule_key: str
    limit_deg: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class JointEdge:
    parent: str
    child: str
    spec: JointSpec
    parent_to: str = "to"
    child_from: str = "from"


class AssemblyGraph:
    def __init__(self, root_part_name: str):
        self._root = root_part_name
        self._parts: Dict[str, PartComponent] = {}
        self._edges: List[JointEdge] = []

    def get_root_part_name(self) -> str:
        return self._root

    def add_part(self, part: PartComponent) -> None:
        self._parts[part.get_name()] = part

    def get_parts(self) -> Dict[str, PartComponent]:
        return dict(self._parts)

    def get_edges(self) -> List[JointEdge]:
        return list(self._edges)

    def connect(self, parent_name: str, child_name: str, spec: JointSpec, parent_to: str = "to", child_from: str = "from") -> None:
        self._edges.append(JointEdge(parent=parent_name, child=child_name, spec=spec, parent_to=parent_to, child_from=child_from))

    def validate(self) -> None:
        if self._root not in self._parts:
            raise ValueError(f"root part '{self._root}' is not added")
        for edge in self._edges:
            if edge.parent not in self._parts:
                raise ValueError(f"missing parent part: {edge.parent}")
            if edge.child not in self._parts:
                raise ValueError(f"missing child part: {edge.child}")


def _rotate(q: Quat, v: Vec3) -> Vec3:
    out = Rot.from_quat(q).apply(np.array(v, dtype=float))
    return (float(out[0]), float(out[1]), float(out[2]))


def _normalize(v: Vec3) -> Vec3:
    arr = np.array(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (float(arr[0] / n), float(arr[1] / n), float(arr[2] / n))


@dataclass(frozen=True)
class ResolvedJoint:
    name: str
    type: str
    parent: str
    child: str
    anchor_root: Vec3
    axis_root: Vec3
    limit_deg: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class ResolveResult:
    part_poses_root: Dict[str, Pose]
    joints: List[ResolvedJoint]


class AssemblyResolver:
    def __init__(self, config: AssemblyConfig):
        if config.joint_axis_rules is None:
            raise ValueError("config.joint_axis_rules must be provided")
        self._cfg = config

    def resolve(self, assembly: AssemblyGraph) -> ResolveResult:
        assembly.validate()
        parts = assembly.get_parts()
        edges = assembly.get_edges()
        root = assembly.get_root_part_name()
        poses: Dict[str, Pose] = {root: self._cfg.root_part_pose}

        outgoing: Dict[str, List[JointEdge]] = {}
        for edge in edges:
            outgoing.setdefault(edge.parent, []).append(edge)

        stack = [root]
        while stack:
            parent = stack.pop()
            if parent not in outgoing:
                continue
            parent_pose = poses[parent]
            parent_p, parent_q = parent_pose.p, parent_pose.q
            parent_part = parts[parent]
            for edge in outgoing[parent]:
                child = edge.child
                child_part = parts[child]
                parent_to_local = parent_part.get_connector_pos(edge.parent_to)
                parent_to_root = tuple(parent_p[i] + _rotate(parent_q, parent_to_local)[i] for i in range(3))
                child_q = parent_q
                child_from_local = child_part.get_connector_pos(edge.child_from)
                child_from_root = _rotate(child_q, child_from_local)
                child_p = (
                    parent_to_root[0] - child_from_root[0],
                    parent_to_root[1] - child_from_root[1],
                    parent_to_root[2] - child_from_root[2],
                )
                poses[child] = Pose(p=child_p, q=child_q)
                stack.append(child)

        joints: List[ResolvedJoint] = []
        for edge in edges:
            parent_pose = poses[edge.parent]
            parent_p, parent_q = parent_pose.p, parent_pose.q
            parent_part = parts[edge.parent]
            parent_to_local = parent_part.get_connector_pos(edge.parent_to)
            parent_to_rot = _rotate(parent_q, parent_to_local)
            anchor_root = (
                parent_p[0] + parent_to_rot[0],
                parent_p[1] + parent_to_rot[1],
                parent_p[2] + parent_to_rot[2],
            )
            rule = self._cfg.joint_axis_rules[edge.spec.axis_rule_key]
            axis_root = _normalize(_rotate(parent_q, rule.axis_parent_local))
            joints.append(
                ResolvedJoint(
                    name=edge.spec.name,
                    type=edge.spec.type.value,
                    parent=edge.parent,
                    child=edge.child,
                    anchor_root=anchor_root,
                    axis_root=axis_root,
                    limit_deg=edge.spec.limit_deg,
                )
            )
        return ResolveResult(part_poses_root=poses, joints=joints)


@dataclass(frozen=True)
class BuildResult:
    build_dir: str
    manifest_path: str


class AssemblyBuilder:
    def __init__(self, config: AssemblyConfig, policy: PartSimulationPolicy):
        self._cfg = config
        self._policy = policy
        self._resolver = AssemblyResolver(config)

    def build(self, assembly: AssemblyGraph, out_dir: str) -> BuildResult:
        os.makedirs(out_dir, exist_ok=True)
        assets_dir = os.path.join(out_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        parts = assembly.get_parts()
        emitted: Dict[str, PartAssets] = {}
        for name, part in parts.items():
            emitted[name] = part.emit_assets(os.path.join(assets_dir, name))

        resolved = self._resolver.resolve(assembly)
        for joint in resolved.joints:
            self._policy.add_no_clip(joint.parent, joint.child)

        flags: Dict[str, Dict[str, object]] = {}
        for name, part in parts.items():
            props = self._policy.resolve_part_props(name, part.get_kind())
            flags[name] = {"ControlMode": props.ControlMode.value, "collision_enabled": bool(props.collision_enabled)}

        manifest = {
                "meta": {
                    "node_count": int(self._cfg.node_count),
                    "use_hardware": self._policy.get_use_hardware(),
                    "use_go2": self._policy.get_use_go2(),
                    "notes": "All poses/joints are root-relative.",
                },
            "parts": [],
            "joints": [],
            "no_clip_pairs": sorted(list(self._policy.get_no_clip_pairs())),
        }

        for name, part in parts.items():
            Pose = resolved.part_poses_root.get(name)
            if Pose is None:
                raise RuntimeError(f"resolver did not produce Pose for part '{name}'")
            assets = emitted[name]
            manifest["parts"].append(
                {
                    "name": name,
                    "kind": part.get_kind().value,
                    "assets": {
                        "mesh": os.path.relpath(assets.mesh_path, out_dir),
                        "frame": os.path.relpath(assets.frame_path, out_dir),
                        "physics": os.path.relpath(assets.physics_path, out_dir),
                    },
                    "flags": flags[name],
                    "pose_root": {"p": list(Pose.p), "q": list(Pose.q)},
                }
            )

        for joint in resolved.joints:
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

        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return BuildResult(build_dir=out_dir, manifest_path=manifest_path)


def create_default_graph(config: AssemblyConfig) -> AssemblyGraph:
    assembly = AssemblyGraph(root_part_name="plate")
    assembly.add_part(PlateComponent(config.plate))
    assembly.add_part(HousingComponent(config.housing))
    assembly.add_part(WedgeComponent(config.wedge))
    for i in range(int(config.node_count)):
        if i == int(config.node_count) - 1:
            assembly.add_part(NodeEndComponent(i, config.node_end))
        else:
            assembly.add_part(NodeComponent(i, config.node))

    assembly.connect("plate", "housing", JointSpec(name="j_plate_housing", type=JointType.prismatic, axis_rule_key="plate_housing_linear"))
    assembly.connect("housing", "wedge", JointSpec(name="j_housing_wedge", type=JointType.revolute, axis_rule_key="housing_wedge"))
    assembly.connect("wedge", "node0", JointSpec(name="j_wedge_node0", type=JointType.revolute, axis_rule_key="wedge_node"))
    for i in range(int(config.node_count) - 1):
        assembly.connect(
            f"node{i}",
            f"node{i+1}",
            JointSpec(name=f"j_node{i}_node{i+1}", type=JointType.revolute, axis_rule_key="node_node"),
        )
    return assembly


def build_default_manifest(
    output_dir: str,
    *,
    use_hardware: bool = False,
    use_go2: bool = False,
) -> str:
    config = make_default_config()
    policy = PartSimulationPolicy()
    policy.set_use_hardware(use_hardware)
    policy.set_use_go2(use_go2)
    result = AssemblyBuilder(config, policy).build(create_default_graph(config), output_dir)
    return result.manifest_path


def build_default(output_dir: str, *, use_hardware: bool = False, use_go2: bool = False) -> str:
    return build_default_manifest(output_dir, use_hardware=use_hardware, use_go2=use_go2)


def build_default_assembly(output_dir: str, *, use_hardware: bool = False, use_go2: bool = False) -> str:
    return build_default_manifest(output_dir, use_hardware=use_hardware, use_go2=use_go2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_BUILD_DIR, help="output directory for manifest.json + assets/")
    ap.add_argument("--use-hardware", action="store_true", help="generate manifest for hardware-backed simulated chain mode")
    ap.add_argument("--use-go2", action="store_true", help="include go2 policy semantics in manifest generation")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_default_manifest(out_dir, use_hardware=bool(args.use_hardware), use_go2=bool(args.use_go2))
    print("Wrote manifest.json at", manifest)


if __name__ == "__main__":
    main()

