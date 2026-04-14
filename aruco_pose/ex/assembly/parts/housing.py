#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .common import PartArtifacts, write_json, write_obj


@dataclass(frozen=True)
class HousingConfig:
    height: float = 0.128
    width: float = 0.130
    diameter: float = 0.130
    segment: int = 72
    length: float = 0.250
    mass: float = 3.0
    com: Optional[Tuple[float, float, float]] = None
    inertia: Optional[Dict[str, float]] = None


HousingArtifacts = PartArtifacts


class HousingGeometry:
    def __init__(self, param: HousingConfig):
        self._param = param
        self._profile_cache: Optional[np.ndarray] = None
        self._body_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def profile_points(self) -> np.ndarray:
        if self._profile_cache is not None:
            return self._profile_cache

        config = self._param
        width, height, radius = config.width, config.height, 0.5 * config.diameter
        end_points = np.array(
            [
                [-width / 2.0, 0.0],
                [+width / 2.0, 0.0],
                [+width / 2.0, height],
            ],
            dtype=float,
        )
        arc_indices = np.arange(1, config.segment, dtype=float)
        theta = (arc_indices / float(config.segment)) * math.pi
        arc_points = np.column_stack([radius * np.cos(theta), height + radius * np.sin(theta)])
        closing_point = np.array([[-width / 2.0, height]], dtype=float)
        self._profile_cache = np.vstack([end_points, arc_points, closing_point])
        return self._profile_cache

    def mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._body_cache is None:
            self._body_cache = extrude_profile(self.profile_points(), self._param.length)
        return self._body_cache


class HousingValidator:
    @staticmethod
    def okay(param: HousingConfig) -> None:
        if param.width <= 0 or param.height <= 0 or param.diameter <= 0 or param.length <= 0:
            raise ValueError("body params must be positive")
        if param.segment < 8:
            raise ValueError("segment too small")
        if param.diameter < param.width:
            raise ValueError("diameter should be >= width")
        if param.mass <= 0:
            raise ValueError("mass must be positive")


class HousingGenerator:
    def __init__(self, param: HousingConfig):
        HousingValidator.okay(param)
        self._param = param
        self._geo = HousingGeometry(param)
        self._frame_payload_cache: Optional[Dict[str, object]] = None
        self._physics_payload_cache: Optional[Dict[str, object]] = None

    def make(self, output_dir: str) -> HousingArtifacts:
        os.makedirs(output_dir, exist_ok=True)
        mesh = os.path.join(output_dir, "housing_mesh.obj")
        frame = os.path.join(output_dir, "housing_frame.json")
        physics = os.path.join(output_dir, "housing_physics.json")
        self.write_mesh(mesh)
        write_json(frame, self.build_frame_payload())
        write_json(physics, self.build_physics_payload())
        return HousingArtifacts(mesh=mesh, frame=frame, physics=physics)

    def build_mesh_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._geo.mesh_data()

    def write_mesh(self, mesh_path: str) -> None:
        vertices, faces = self.build_mesh_data()
        write_obj(mesh_path, vertices, faces)

    def build_frame_payload(self) -> Dict[str, object]:
        if self._frame_payload_cache is not None:
            return self._frame_payload_cache

        config = self._param
        from_pos = [-float(config.length), 0.0, float(config.height)]
        to_pos = [0.0, 0.0, float(config.height)]
        self._frame_payload_cache = {
            "unit": "m",
            "part_frame": {
                "name": "housing_part_frame",
                "origin_in_part": [0.0, 0.0, 0.0],
                "axes_note": "x=0 is the front face; the body extends toward -X (so it faces +X).",
            },
            "connectors": {"from": from_pos, "to": to_pos},
            "mesh_offset_in_part": {"visual": [0.0, 0.0, 0.0], "collision": [0.0, 0.0, 0.0]},
        }
        return self._frame_payload_cache

    def build_physics_payload(self) -> Dict[str, object]:
        if self._physics_payload_cache is not None:
            return self._physics_payload_cache

        config = self._param
        # Mesh-derived default CoM for current housing profile/extrusion.
        com = config.com if config.com is not None else (-float(config.length) * 0.5, 0.0, 0.090106019)
        inertia = config.inertia
        if inertia is None:
            ref_mass = 5.0
            s = float(config.mass) / ref_mass
            inertia = {
                "ixx": 0.0203873448 * s,
                "iyy": 0.0398896376 * s,
                "izz": 0.0325810406 * s,
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
def convex_cap_fan(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size = poly.shape[0]
    center = poly.mean(axis=0)
    idx = np.arange(size, dtype=np.int32)
    nxt = (idx + 1) % size
    faces = np.column_stack([np.full(size, size, dtype=np.int32), idx, nxt])
    return center, faces


def extrude_profile(poly_yz: np.ndarray, length: float) -> Tuple[np.ndarray, np.ndarray]:
    size = poly_yz.shape[0]
    v0 = np.column_stack([np.zeros(size), poly_yz[:, 0], poly_yz[:, 1]])
    v1 = np.column_stack([-np.full(size, length), poly_yz[:, 0], poly_yz[:, 1]])
    center, cap_faces = convex_cap_fan(poly_yz)
    vc0 = np.array([[0.0, center[0], center[1]]], dtype=float)
    vc1 = np.array([[-length, center[0], center[1]]], dtype=float)
    vertices = np.vstack([v0, v1, vc0, vc1])

    cap_front = cap_faces.copy()
    cap_front[:, 0] = 2 * size

    cap_back = np.empty_like(cap_faces)
    cap_back[:, 0] = 2 * size + 1
    cap_back[:, 1] = size + cap_faces[:, 2]
    cap_back[:, 2] = size + cap_faces[:, 1]

    idx = np.arange(size, dtype=np.int32)
    nxt = (idx + 1) % size
    side_a = np.column_stack([idx, size + nxt, nxt])
    side_b = np.column_stack([idx, size + idx, size + nxt])
    faces = np.vstack([cap_front, cap_back, side_a, side_b]).astype(np.int32, copy=False)
    return vertices, faces


def main() -> None:
    out_dir = os.path.join(".", "assets_preview", "housing")
    artifacts = HousingGenerator(HousingConfig()).make(out_dir)
    print("Generated:")
    print(" -", artifacts.visual)
    print(" -", artifacts.collision)
    print(" -", artifacts.frame)
    print(" -", artifacts.physics)


if __name__ == "__main__":
    main()

