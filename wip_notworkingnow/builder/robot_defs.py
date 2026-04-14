#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x, y, z, w)


@dataclass(frozen=True)
class Pose:
    p: Vec3 = (0.0, 0.0, 0.0)
    q: Quat = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class JointAxisRule:
    axis_parent_local: Vec3


@dataclass(frozen=True)
class ConnectorSpec:
    from_pos: Vec3
    to_pos: Vec3


@dataclass(frozen=True)
class PartSpec:
    connectors: ConnectorSpec


@dataclass(frozen=True)
class RobotBuildConfig:
    node_count: int = 10
    root_part_pose: Pose = Pose()
    plate: PartSpec = PartSpec(
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, -0.005), to_pos=(0.0, 0.0, 0.0)),
    )
    node: PartSpec = PartSpec(
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.05, 0.0, 0.0)),
    )
    node_end: PartSpec = PartSpec(
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.05, 0.0, 0.0)),
    )
    housing: PartSpec = PartSpec(
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.0, 0.0, 0.128)),
    )
    wedge: PartSpec = PartSpec(
        connectors=ConnectorSpec(from_pos=(0.0, 0.0, 0.0), to_pos=(0.03, 0.0, 0.0)),
    )
    joint_axis_rules: Optional[Dict[str, JointAxisRule]] = None


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


class ControlMode(Enum):
    fixed = "fixed"
    commanded = "commanded"
    simulated = "simulated"


@dataclass
class PartOverride:
    control_mode: Optional[ControlMode] = None
    collision_enabled: Optional[bool] = None


@dataclass(frozen=True)
class RuntimePartProps:
    control_mode: ControlMode
    collision_enabled: bool


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


@dataclass(frozen=True)
class PartInstance:
    name: str
    kind: PartKind
    connectors: ConnectorSpec

    def get_from_pos(self) -> Vec3:
        return self.connectors.from_pos

    def get_to_pos(self) -> Vec3:
        return self.connectors.to_pos

    def get_connector_end_pos(self, key: str) -> Vec3:
        if key == "from":
            return self.get_from_pos()
        if key == "to":
            return self.get_to_pos()
        raise KeyError(f"unknown connector key: {key}")


class AssemblyPlan:
    def __init__(self, root_part_name: str):
        self._root = root_part_name
        self._parts: Dict[str, PartInstance] = {}
        self._edges: List[JointEdge] = []

    def get_root_name(self) -> str:
        return self._root

    def add_part(self, part: PartInstance) -> None:
        self._parts[part.name] = part

    def get_parts(self) -> Dict[str, PartInstance]:
        return self._parts

    def get_edges(self) -> List[JointEdge]:
        return self._edges

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


@dataclass(frozen=True)
class JointLayout:
    name: str
    type: str
    parent: str
    child: str
    anchor_root: Vec3
    axis_root: Vec3
    limit_deg: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class JointLayoutResult:
    part_poses_root: Dict[str, Pose]
    joints: List[JointLayout]


@dataclass(frozen=True)
class ManifestBuildResult:
    build_dir: str
    manifest_path: str
