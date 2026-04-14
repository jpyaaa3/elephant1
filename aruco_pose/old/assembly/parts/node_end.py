#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .common import PartArtifacts, compute_face_normals, write_json, write_obj
from .node import NodeConfig


@dataclass(frozen=True)
class NodeEndConfig(NodeConfig):
    pass


NodeArtifacts = PartArtifacts

_NODE_END_FACES = np.array(
    [
        [0, 1, 4],
        [4, 1, 3],
        [3, 1, 2],
        [5, 6, 1],
        [1, 6, 2],
        [6, 7, 2],
        [2, 7, 3],
        [7, 8, 3],
        [3, 8, 4],
        [8, 9, 4],
        [4, 9, 0],
        [9, 5, 0],
        [0, 5, 1],
        [5, 9, 6],
        [6, 9, 7],
        [7, 9, 8],
    ],
    dtype=np.int32,
)


class NodeEndGeometry:
    def __init__(self, param: NodeEndConfig):
        self._param = param
        self._cross_section_cache: Optional[np.ndarray] = None
        self._mesh_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._section_props_cache: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    def cross_section_points(self) -> np.ndarray:
        if self._cross_section_cache is not None:
            return self._cross_section_cache

        config = self._param
        half_height = float(config.height) * 0.5
        bevel_depth = half_height * math.tan(math.radians(18.0))
        self._cross_section_cache = np.array(
            [
                [float(config.pitch), -half_height],
                [float(config.pitch), +half_height],
                [bevel_depth, +half_height],
                [0.0, 0.0],
                [bevel_depth, -half_height],
            ],
            dtype=float,
        )
        return self._cross_section_cache

    def section_properties(self) -> Tuple[float, np.ndarray, np.ndarray]:
        if self._section_props_cache is not None:
            return self._section_props_cache

        points = self.cross_section_points()
        x = points[:, 0]
        z = points[:, 1]
        x_next = np.roll(x, -1)
        z_next = np.roll(z, -1)
        cross = x * z_next - x_next * z

        area = 0.5 * np.sum(cross)
        if area <= 0.0:
            raise ValueError("node end cross-section area must be positive")

        centroid_x = np.sum((x + x_next) * cross) / (6.0 * area)
        centroid_z = np.sum((z + z_next) * cross) / (6.0 * area)
        centroid = np.array([centroid_x, centroid_z], dtype=float)

        area_ix_origin = np.sum((z * z + z * z_next + z_next * z_next) * cross) / 12.0
        area_iz_origin = np.sum((x * x + x * x_next + x_next * x_next) * cross) / 12.0
        area_ixz_origin = np.sum((x * z_next + 2.0 * x * z + 2.0 * x_next * z_next + x_next * z) * cross) / 24.0
        centroidal_area_moments = np.array(
            [
                area_ix_origin - area * centroid_z * centroid_z,
                area_iz_origin - area * centroid_x * centroid_x,
                area_ixz_origin - area * centroid_x * centroid_z,
            ],
            dtype=float,
        )
        self._section_props_cache = (area, centroid, centroidal_area_moments)
        return self._section_props_cache

    def mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mesh_cache is not None:
            return self._mesh_cache

        config = self._param
        cross_section_points = self.cross_section_points()
        half_width = float(config.width) * 0.5

        negative_side_vertices = np.column_stack(
            [cross_section_points[:, 0], np.full(cross_section_points.shape[0], -half_width), cross_section_points[:, 1]]
        )
        positive_side_vertices = np.column_stack(
            [cross_section_points[:, 0], np.full(cross_section_points.shape[0], +half_width), cross_section_points[:, 1]]
        )
        vertices = np.vstack([negative_side_vertices, positive_side_vertices])

        self._mesh_cache = (vertices, _NODE_END_FACES, compute_face_normals(vertices, _NODE_END_FACES))
        return self._mesh_cache


class NodeEndValidator:
    @staticmethod
    def okay(param: NodeEndConfig) -> None:
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
        bevel_depth = (float(param.height) * 0.5) * math.tan(math.radians(18.0))
        if bevel_depth >= float(param.pitch):
            raise ValueError("height produces an invalid node_end bevel depth")
        if param.mass <= 0:
            raise ValueError("mass must be positive")


class NodeEndGenerator:
    def __init__(self, param: NodeEndConfig):
        NodeEndValidator.okay(param)
        self._param = param
        self._geo = NodeEndGeometry(param)
        self._frame_payload_cache: Optional[Dict[str, object]] = None
        self._physics_payload_cache: Optional[Dict[str, object]] = None

    def make(self, output_dir: str) -> NodeArtifacts:
        os.makedirs(output_dir, exist_ok=True)
        mesh = os.path.join(output_dir, "node_end_mesh.obj")
        frame = os.path.join(output_dir, "node_end_frame.json")
        physics = os.path.join(output_dir, "node_end_physics.json")
        self.write_mesh(mesh)
        write_json(frame, self.build_frame_payload())
        write_json(physics, self.build_physics_payload())
        return NodeArtifacts(mesh=mesh, frame=frame, physics=physics)

    def build_mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._geo.mesh_data()

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
                "name": "node_end_part_frame",
                "origin_in_part": [0.0, 0.0, 0.0],
                "axes_note": "+X points from the tip connector (from) to the flat rear face connector (to).",
            },
            "connectors": {"from": [0.0, 0.0, 0.0], "to": [float(config.pitch), 0.0, 0.0]},
            "notes": {"definition": "Node end body uses the tapered pentagonal prism defined by node_end.obj."},
        }
        return self._frame_payload_cache

    def build_physics_payload(self) -> Dict[str, object]:
        if self._physics_payload_cache is not None:
            return self._physics_payload_cache

        config = self._param
        area, centroid_xz, centroidal_area_moments = self._geo.section_properties()
        width = float(config.width)
        mass = float(config.mass)

        com = config.com if config.com is not None else (float(centroid_xz[0]), 0.0, float(centroid_xz[1]))
        inertia = config.inertia
        if inertia is None:
            area_ix, area_iz, area_ixz = centroidal_area_moments
            ixx = mass * ((area_ix / area) + (width * width) / 12.0)
            iyy = mass * ((area_ix + area_iz) / area)
            izz = mass * ((area_iz / area) + (width * width) / 12.0)
            inertia = {
                "ixx": float(ixx),
                "iyy": float(iyy),
                "izz": float(izz),
                "ixy": 0.0,
                "ixz": float(-mass * (area_ixz / area)),
                "iyz": 0.0,
            }

        self._physics_payload_cache = {
            "unit": {"mass": "kg", "length": "m"},
            "mass": mass,
            "com": [float(com[0]), float(com[1]), float(com[2])],
            "inertia": {
                "ixx": float(inertia["ixx"]),
                "iyy": float(inertia["iyy"]),
                "izz": float(inertia["izz"]),
                "ixy": float(inertia.get("ixy", 0.0)),
                "ixz": float(inertia.get("ixz", 0.0)),
                "iyz": float(inertia.get("iyz", 0.0)),
            },
            "notes": {"info": "Inertia defaults use the tapered node-end prism analytic section model."},
        }
        return self._physics_payload_cache


def main() -> None:
    out_dir = os.path.join(".", "assets_preview", "node_end")
    artifacts = NodeEndGenerator(NodeEndConfig()).make(out_dir)
    print("Generated:")
    print(" -", artifacts.visual)
    print(" -", artifacts.collision)
    print(" -", artifacts.frame)
    print(" -", artifacts.physics)


if __name__ == "__main__":
    main()
