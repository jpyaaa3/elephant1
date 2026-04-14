#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .common import PartArtifacts, write_json, write_obj


@dataclass(frozen=True)
class PlateConfig:
    x_min: float = -0.590
    x_max: float = +0.010
    y_min: float = -0.150
    y_max: float = +0.150
    z_min: float = -0.005
    z_max: float = +0.000
    mass: float = 10.0
    com: Optional[Tuple[float, float, float]] = None
    inertia: Optional[Dict[str, float]] = None
    from_pos: Tuple[float, float, float] = (0.0, 0.0, -0.005)
    to_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)


PlateArtifacts = PartArtifacts

_PLATE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ],
    dtype=np.int32,
)


def _box_mesh(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ],
        dtype=float,
    )
    return vertices, _PLATE_FACES
class PlateValidator:
    @staticmethod
    def okay(param: PlateConfig) -> None:
        if not (param.x_max > param.x_min and param.y_max > param.y_min and param.z_max > param.z_min):
            raise ValueError("invalid box extents")
        if param.mass <= 0:
            raise ValueError("mass must be positive")


class PlateGenerator:
    def __init__(self, param: PlateConfig):
        PlateValidator.okay(param)
        self._param = param
        self._mesh_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._frame_payload_cache: Optional[Dict[str, object]] = None
        self._physics_payload_cache: Optional[Dict[str, object]] = None

    def make(self, output_dir: str) -> PlateArtifacts:
        os.makedirs(output_dir, exist_ok=True)
        mesh = os.path.join(output_dir, "plate_mesh.obj")
        frame = os.path.join(output_dir, "plate_frame.json")
        physics = os.path.join(output_dir, "plate_physics.json")
        self.write_mesh(mesh)
        write_json(frame, self.build_frame_payload())
        write_json(physics, self.build_physics_payload())
        return PlateArtifacts(mesh=mesh, frame=frame, physics=physics)

    def build_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._mesh_cache is not None:
            return self._mesh_cache

        config = self._param
        self._mesh_cache = _box_mesh(config.x_min, config.x_max, config.y_min, config.y_max, config.z_min, config.z_max)
        return self._mesh_cache

    def write_mesh(self, mesh_path: str) -> None:
        vertices, faces = self.build_mesh_data()
        write_obj(mesh_path, vertices, faces)

    def build_frame_payload(self) -> Dict[str, object]:
        if self._frame_payload_cache is not None:
            return self._frame_payload_cache

        config = self._param
        self._frame_payload_cache = {
            "unit": "m",
            "part_frame": {
                "name": "plate_part_frame",
                "origin_in_part": [0.0, 0.0, 0.0],
                "axes_note": "Thin plate box. Connectors are defined in plate local coordinates.",
            },
            "connectors": {
                "from": [float(config.from_pos[0]), float(config.from_pos[1]), float(config.from_pos[2])],
                "to": [float(config.to_pos[0]), float(config.to_pos[1]), float(config.to_pos[2])],
            },
            "mesh_offset_in_part": {"visual": [0.0, 0.0, 0.0], "collision": [0.0, 0.0, 0.0]},
            "geometry_note": {
                "box_extents": {
                    "x_min": float(config.x_min), "x_max": float(config.x_max),
                    "y_min": float(config.y_min), "y_max": float(config.y_max),
                    "z_min": float(config.z_min), "z_max": float(config.z_max),
                }
            },
        }
        return self._frame_payload_cache

    def build_physics_payload(self) -> Dict[str, object]:
        if self._physics_payload_cache is not None:
            return self._physics_payload_cache

        config = self._param
        com = (
            config.com
            if config.com is not None
            else ((config.x_min + config.x_max) * 0.5, (config.y_min + config.y_max) * 0.5, (config.z_min + config.z_max) * 0.5)
        )
        inertia = config.inertia
        if inertia is None:
            dx = float(config.x_max - config.x_min)
            dy = float(config.y_max - config.y_min)
            dz = float(config.z_max - config.z_min)
            m = float(config.mass)
            inertia = {
                "ixx": (m / 12.0) * (dy * dy + dz * dz),
                "iyy": (m / 12.0) * (dx * dx + dz * dz),
                "izz": (m / 12.0) * (dx * dx + dy * dy),
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
            "notes": {"info": "Inertia defaults use box analytic formula."},
        }
        return self._physics_payload_cache
def main() -> None:
    out_dir = os.path.join(".", "assets_preview", "plate")
    artifacts = PlateGenerator(PlateConfig()).make(out_dir)
    print("Generated:")
    print(" -", artifacts.visual)
    print(" -", artifacts.collision)
    print(" -", artifacts.frame)
    print(" -", artifacts.physics)


if __name__ == "__main__":
    main()

