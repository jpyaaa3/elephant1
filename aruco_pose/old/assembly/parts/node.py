#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .common import PartArtifacts, compute_face_normals, write_json, write_obj


@dataclass(frozen=True)
class NodeConfig:
    pitch: float = 0.05
    height: float = 0.07
    width: float = 0.088
    radius: float = 0.025
    mass: float = 0.10
    com: Optional[Tuple[float, float, float]] = None
    inertia: Optional[Dict[str, float]] = None
    sharpness: bool = True


NodeArtifacts = PartArtifacts

_NODE_FRONT_CAP_FACES = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], dtype=np.int32)
_NODE_BACK_CAP_FACES = np.array([[6, 8, 7], [6, 9, 8], [6, 10, 9], [6, 11, 10]], dtype=np.int32)
_NODE_EDGE_INDICES = np.arange(6, dtype=np.int32)
_NODE_NEXT_EDGE_INDICES = (_NODE_EDGE_INDICES + 1) % 6
_NODE_SIDE_FACES_UPPER = np.column_stack([6 + _NODE_EDGE_INDICES, 6 + _NODE_NEXT_EDGE_INDICES, _NODE_NEXT_EDGE_INDICES])
_NODE_SIDE_FACES_LOWER = np.column_stack([6 + _NODE_EDGE_INDICES, _NODE_NEXT_EDGE_INDICES, _NODE_EDGE_INDICES])
_NODE_FACES = np.vstack(
    [_NODE_FRONT_CAP_FACES, _NODE_BACK_CAP_FACES, _NODE_SIDE_FACES_UPPER, _NODE_SIDE_FACES_LOWER]
).astype(np.int32, copy=False)


class NodeGeometry:
    def __init__(self, radius: float, height: float):
        self._radius = float(radius)
        self._height = float(height)
        self._edge_offset = self._solve_edge_offset()
        self._cross_section_cache: Optional[np.ndarray] = None
        self._mesh_cache: Optional[Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None

    def cross_section_points(self) -> np.ndarray:
        if self._cross_section_cache is not None:
            return self._cross_section_cache

        radius, height, edge_offset = self._radius, self._height, self._edge_offset
        self._cross_section_cache = np.array(
            [
                [0.0, 0.0],
                [radius - edge_offset, +height / 2.0],
                [radius + edge_offset, +height / 2.0],
                [2.0 * radius, 0.0],
                [radius + edge_offset, -height / 2.0],
                [radius - edge_offset, -height / 2.0],
            ],
            dtype=float,
        )
        return self._cross_section_cache

    def _solve_edge_offset(self) -> float:
        radius, height = self._radius, self._height
        edge_offset = radius - (height / 2.0) * math.tan(math.radians(18.0))
        if not (0.0 < edge_offset < radius):
            raise ValueError(f"invalid edge offset: edge_offset={edge_offset}, radius={radius}, height={height}")
        return float(edge_offset)

    def mesh_data(self, width: float, sharpness: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not sharpness:
            raise NotImplementedError("sharpness=False is not implemented")

        cached = self._mesh_cache
        width = float(width)
        if cached is not None and math.isclose(cached[0], width, rel_tol=0.0, abs_tol=0.0):
            return cached[1]

        cross_section_points = self.cross_section_points()
        half_w = width * 0.5
        left_ring_vertices = np.column_stack(
            [cross_section_points[:, 0], np.full(cross_section_points.shape[0], +half_w), cross_section_points[:, 1]]
        )
        right_ring_vertices = np.column_stack(
            [cross_section_points[:, 0], np.full(cross_section_points.shape[0], -half_w), cross_section_points[:, 1]]
        )
        vertices = np.vstack([left_ring_vertices, right_ring_vertices])
        mesh = (vertices, _NODE_FACES, compute_face_normals(vertices, _NODE_FACES))
        self._mesh_cache = (width, mesh)
        return mesh

    def write_mesh_obj(self, path: str, width: float, sharpness: bool = True) -> None:
        vertices, faces, normals = self.mesh_data(width=width, sharpness=sharpness)
        write_obj(path=path, vertices=vertices, faces=faces, normals=normals)


class NodeValidator:
    @staticmethod
    def okay(param: NodeConfig) -> None:
        if param.pitch <= 0:
            raise ValueError(f"negative pitch ({param.pitch})")
        if param.height <= 0:
            raise ValueError(f"negative height ({param.height})")
        if param.width <= 0:
            raise ValueError(f"negative width ({param.width})")
        if param.radius <= 0:
            raise ValueError(f"negative radius ({param.radius})")
        if abs(param.pitch - 2.0 * param.radius) > 1e-9:
            raise ValueError("pitch must be 2 * radius")
        if not param.sharpness:
            raise ValueError("sharpness=False is not supported")
        if param.mass <= 0:
            raise ValueError("mass must be positive")


class NodeGenerator:
    def __init__(self, param: NodeConfig):
        NodeValidator.okay(param)
        self._param = param
        self._geo = NodeGeometry(radius=param.radius, height=param.height)
        self._frame_payload_cache: Optional[Dict[str, object]] = None
        self._physics_payload_cache: Optional[Dict[str, object]] = None

    def make(self, output_dir: str) -> NodeArtifacts:
        os.makedirs(output_dir, exist_ok=True)
        mesh = os.path.join(output_dir, "node_mesh.obj")
        frame = os.path.join(output_dir, "node_frame.json")
        physics = os.path.join(output_dir, "node_physics.json")
        self.write_mesh(mesh)
        write_json(frame, self.build_frame_payload())
        write_json(physics, self.build_physics_payload())
        return NodeArtifacts(mesh=mesh, frame=frame, physics=physics)

    def build_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._geo.mesh_data(width=self._param.width, sharpness=self._param.sharpness)

    def write_mesh(self, mesh_path: str) -> None:
        vertices, faces, normals = self.build_mesh_data()
        write_obj(path=mesh_path, vertices=vertices, faces=faces, normals=normals)

    def build_frame_payload(self) -> Dict[str, object]:
        if self._frame_payload_cache is not None:
            return self._frame_payload_cache

        config = self._param
        self._frame_payload_cache = {
            "unit": "m",
            "part_frame": {
                "name": "node_part_frame",
                "origin_in_part": [0.0, 0.0, 0.0],
                "axes_note": "+X points from rear connector (from) to front connector (to).",
            },
            "connectors": {"from": [0.0, 0.0, 0.0], "to": [float(config.pitch), 0.0, 0.0]},
            "notes": {"definition": "Node body spans one pitch along +X between its connector frames."},
        }
        return self._frame_payload_cache

    def build_physics_payload(self) -> Dict[str, object]:
        if self._physics_payload_cache is not None:
            return self._physics_payload_cache

        config = self._param
        com = config.com if config.com is not None else (float(config.radius), 0.0, 0.0)
        inertia = config.inertia
        if inertia is None:
            ref_mass = 0.1
            s = float(config.mass) / ref_mass
            inertia = {
                "ixx": 9.93559147e-05 * s,
                "iyy": 4.83345352e-05 * s,
                "izz": 7.80452872e-05 * s,
                "ixy": 0.0,
                "ixz": 0.0,
                "iyz": 0.0,
            }
        self._physics_payload_cache = {
            "unit": {"mass": "kg", "length": "m"},
            "mass": float(config.mass),
            "com": [float(com[0]), float(com[1]), float(com[2])],
            "inertia": {
                "ixx": float(inertia["ixx"]),
                "iyy": float(inertia["iyy"]),
                "izz": float(inertia["izz"]),
                "ixy": float(inertia.get("ixy", 0.0)),
                "ixz": float(inertia.get("ixz", 0.0)),
                "iyz": float(inertia.get("iyz", 0.0)),
            },
            "notes": {"info": "Inertia defaults are mesh-derived and mass-scaled."},
        }
        return self._physics_payload_cache
def main() -> None:
    out_dir = os.path.join(".", "assets_preview", "node")
    artifacts = NodeGenerator(NodeConfig()).make(out_dir)
    print("Generated:")
    print(" -", artifacts.visual)
    print(" -", artifacts.collision)
    print(" -", artifacts.frame)
    print(" -", artifacts.physics)


if __name__ == "__main__":
    main()

