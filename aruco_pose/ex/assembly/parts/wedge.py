#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .common import PartArtifacts, write_json, write_obj


@dataclass(frozen=True)
class WedgeConfig:
    wedge_h: float = 0.084
    wedge_w: float = 0.088
    wedge_e: float = 0.028
    wedge_m: float = 0.014353
    mass: float = 0.10
    com: Optional[Tuple[float, float, float]] = None
    inertia: Optional[Dict[str, float]] = None


WedgeArtifacts = PartArtifacts

_WEDGE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [0, 4, 3],
        [5, 6, 7],
        [5, 7, 8],
        [5, 8, 9],
        [0, 6, 5],
        [0, 1, 6],
        [1, 7, 6],
        [1, 2, 7],
        [2, 8, 7],
        [2, 3, 8],
        [3, 9, 8],
        [3, 4, 9],
        [4, 5, 9],
        [4, 0, 5],
    ],
    dtype=np.int32,
)[:, [0, 2, 1]].copy()


class WedgeGeometry:
    def __init__(self, param: WedgeConfig):
        self._param = param
        self._mesh_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def mesh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mesh_cache is not None:
            return self._mesh_cache

        config = self._param
        mid_x = config.wedge_m
        left_y = -config.wedge_w / 2.0
        right_y = +config.wedge_w / 2.0
        bottom_z = -config.wedge_h / 2.0
        top_z = +config.wedge_h / 2.0

        left_side_vertices = np.array(
            [
                [0.0, left_y, bottom_z],
                [mid_x, left_y, bottom_z],
                [config.wedge_e, left_y, 0.0],
                [mid_x, left_y, top_z],
                [0.0, left_y, top_z],
            ],
            dtype=float,
        )
        right_side_vertices = np.array(
            [
                [0.0, right_y, bottom_z],
                [mid_x, right_y, bottom_z],
                [config.wedge_e, right_y, 0.0],
                [mid_x, right_y, top_z],
                [0.0, right_y, top_z],
            ],
            dtype=float,
        )

        vertices = np.vstack([left_side_vertices, right_side_vertices]).astype(float)
        tip_position = np.array([config.wedge_e, 0.0, 0.0], dtype=float)
        self._mesh_cache = (vertices, _WEDGE_FACES, tip_position)
        return self._mesh_cache


class WedgeValidator:
    @staticmethod
    def okay(param: WedgeConfig) -> None:
        if param.wedge_w <= 0 or param.wedge_h <= 0 or param.wedge_e <= 0 or param.wedge_m <= 0:
            raise ValueError("wedge params must be positive")
        if param.wedge_m >= param.wedge_e:
            raise ValueError("wedge_m must satisfy 0 < wedge_m < wedge_e")
        if param.mass <= 0:
            raise ValueError("mass must be positive")


class WedgeGenerator:
    def __init__(self, param: WedgeConfig):
        WedgeValidator.okay(param)
        self._param = param
        self._geo = WedgeGeometry(param)
        self._frame_payload_cache: Optional[Dict[str, object]] = None
        self._physics_payload_cache: Optional[Dict[str, object]] = None

    def make(self, output_dir: str) -> WedgeArtifacts:
        os.makedirs(output_dir, exist_ok=True)
        mesh = os.path.join(output_dir, "wedge_mesh.obj")
        frame = os.path.join(output_dir, "wedge_frame.json")
        physics = os.path.join(output_dir, "wedge_physics.json")
        self.write_mesh(mesh)
        write_json(frame, self.build_frame_payload())
        write_json(physics, self.build_physics_payload())
        return WedgeArtifacts(mesh=mesh, frame=frame, physics=physics)

    def build_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices, faces, _ = self._geo.mesh_data()
        return vertices, faces

    def write_mesh(self, mesh_path: str) -> None:
        vertices, faces = self.build_mesh_data()
        write_obj(mesh_path, vertices, faces)

    def build_frame_payload(self) -> Dict[str, object]:
        if self._frame_payload_cache is not None:
            return self._frame_payload_cache

        _, _, tip_position = self._geo.mesh_data()
        self._frame_payload_cache = {
            "unit": "m",
            "part_frame": {
                "name": "wedge_part_frame",
                "origin_in_part": [0.0, 0.0, 0.0],
                "axes_note": "x=0 is the interface plane; wedge tip extends toward +X.",
            },
            "connectors": {
                "from": [0.0, 0.0, 0.0],
                "to": [float(tip_position[0]), float(tip_position[1]), float(tip_position[2])],
            },
            "mesh_offset_in_part": {"visual": [0.0, 0.0, 0.0], "collision": [0.0, 0.0, 0.0]},
        }
        return self._frame_payload_cache

    def build_physics_payload(self) -> Dict[str, object]:
        if self._physics_payload_cache is not None:
            return self._physics_payload_cache

        config = self._param
        # Mesh-derived centroid ratio along +X for the default pentagonal wedge section.
        com = config.com if config.com is not None else (0.391239111 * float(config.wedge_e), 0.0, 0.0)
        inertia = config.inertia
        if inertia is None:
            ref_mass = 0.1
            s = float(config.mass) / ref_mass
            inertia = {
                "ixx": 0.000113860054 * s,
                "iyy": 5.38263296e-05 * s,
                "izz": 6.90329423e-05 * s,
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
        }
        return self._physics_payload_cache

def main() -> None:
    out_dir = os.path.join(".", "assets_preview", "wedge")
    artifacts = WedgeGenerator(WedgeConfig()).make(out_dir)
    print("Generated:")
    print(" -", artifacts.visual)
    print(" -", artifacts.collision)
    print(" -", artifacts.frame)
    print(" -", artifacts.physics)


if __name__ == "__main__":
    main()

