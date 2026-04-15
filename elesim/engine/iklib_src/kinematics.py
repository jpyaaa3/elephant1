#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from engine.joint_defs import JointLimit


class Kinematics:
    """
    GenSimPlus-style 4DOF abstraction:
    q = [linear, roll, theta1, theta2], where theta1/theta2 are per-joint bend angles.
    """

    def __init__(
        self,
        pitch: float,
        n_nodes: int,
        n_seg: int,
        origin_xyz: np.ndarray,
        limit: JointLimit,
        base_axis_sign: float = 1.0,
        roll_axis_sign: float = 1.0,
        bend_axis_sign: float = -1.0,
        soft_k: float = 2.5,
    ):
        self.pitch = float(pitch)
        self.n_nodes = int(n_nodes)
        self.n_seg = int(n_seg)
        self.origin_xyz = np.array(origin_xyz, dtype=float).reshape(3)
        self.limits = limit

        self.linear_min = float(limit.linear_min)
        self.linear_max = float(limit.linear_max)
        self.roll_min = float(limit.roll_min_rad())
        self.roll_max = float(limit.roll_max_rad())
        self.bend_lim = float(limit.bend_lim_rad())
        self.base_axis_sign = -1.0 if float(base_axis_sign) < 0.0 else 1.0
        self.roll_axis_sign = -1.0 if float(roll_axis_sign) < 0.0 else 1.0
        self.bend_axis_sign = -1.0 if float(bend_axis_sign) < 0.0 else 1.0
        self.soft_k = float(soft_k)

    @staticmethod
    def _rot_x(phi: float) -> np.ndarray:
        c, s = math.cos(phi), math.sin(phi)
        return np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

    @staticmethod
    def _rot_y(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return np.array(
            [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

    @staticmethod
    def _trans(x: float, y: float, z: float) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[0, 3] = float(x)
        T[1, 3] = float(y)
        T[2, 3] = float(z)
        return T

    def _clamp_all(self, q: np.ndarray) -> np.ndarray:
        linear, roll, t1, t2 = map(float, np.asarray(q, dtype=float).reshape(4))
        linear = float(np.clip(linear, self.linear_min, self.linear_max))
        roll = float(np.clip(roll, self.roll_min, self.roll_max))
        t1 = float(np.clip(t1, -self.bend_lim, +self.bend_lim))
        t2 = float(np.clip(t2, -self.bend_lim, +self.bend_lim))
        return np.array([linear, roll, t1, t2], dtype=float)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return self._clamp_all(q)

    @staticmethod
    def _T_from_R_t(rot: Rot, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = np.array(t, dtype=float).reshape(3)
        return T

    def tip_position(self, q: np.ndarray, *, clamp: bool = True, soft: bool = True) -> np.ndarray:
        p, _dirx = self.tip_position_and_dir_x(q, clamp=clamp, soft=soft)
        return p

    def tip_position_and_forward(self, q: np.ndarray, *, clamp: bool = True, soft: bool = True) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float).reshape(4)
        if clamp:
            q = self._clamp_all(q)
        linear, roll, theta1, theta2 = map(float, q)
        if not soft:
            theta1 = float(np.clip(theta1, -self.bend_lim, +self.bend_lim))
            theta2 = float(np.clip(theta2, -self.bend_lim, +self.bend_lim))
        n_nodes = int(self.n_nodes)
        n1 = int(self.n_seg)
        pitch = float(self.pitch)

        phi = self.roll_axis_sign * roll
        cphi, sphi = math.cos(phi), math.sin(phi)
        c0 = np.array([1.0, 0.0, 0.0], dtype=float)
        c2 = np.array([0.0, -sphi, cphi], dtype=float)
        p = self.origin_xyz + np.array([self.base_axis_sign * linear, 0.0, 0.0], dtype=float)

        a1 = self.bend_axis_sign * theta1
        a2 = self.bend_axis_sign * theta2
        c1, s1 = math.cos(a1), math.sin(a1)
        c2a, s2 = math.cos(a2), math.sin(a2)

        for i in range(n_nodes):
            if i > 0:
                p = p + pitch * c0
            if i < n1:
                ct, st = c1, s1
            else:
                ct, st = c2a, s2
            c0_old = c0
            c0 = ct * c0_old - st * c2
            c2 = st * c0_old + ct * c2

        p_tip = p + pitch * c0
        return p_tip.astype(float, copy=True), c0.astype(float, copy=True)

    def tip_position_and_dir_x(self, q: np.ndarray, *, clamp: bool = True, soft: bool = True) -> tuple[np.ndarray, float]:
        p_tip, forward = self.tip_position_and_forward(q, clamp=clamp, soft=soft)
        dirx = float(np.clip(float(forward[0]), -1.0, 1.0))
        return p_tip, dirx

    def position_error_vec(self, q: np.ndarray, target_world: np.ndarray) -> np.ndarray:
        return self.tip_position(q) - np.asarray(target_world, dtype=float).reshape(3)

    def position_error(self, q: np.ndarray, target_world: np.ndarray) -> float:
        return float(np.linalg.norm(self.position_error_vec(q, target_world)))

    def tip_from_bends(self, linear: float, roll: float, bends: np.ndarray) -> np.ndarray:
        bends = np.array(bends, dtype=float).reshape(-1)
        n = int(self.n_nodes)
        if bends.size < n:
            bends = np.concatenate([bends, np.zeros(n - bends.size, dtype=float)], axis=0)
        else:
            bends = bends[:n]

        linear = float(np.clip(float(linear), self.linear_min, self.linear_max))
        roll = float(np.clip(float(roll), self.roll_min, self.roll_max))

        T = np.eye(4, dtype=float)
        T = T @ self._T_from_R_t(
            Rot.identity(), self.origin_xyz + np.array([self.base_axis_sign * linear, 0.0, 0.0], dtype=float)
        )
        T = T @ self._T_from_R_t(Rot.from_euler("x", self.roll_axis_sign * roll), np.array([0.0, 0.0, 0.0], dtype=float))

        for i in range(n):
            a = float(np.clip(float(bends[i]), -self.bend_lim, +self.bend_lim))
            step_x = 0.0 if i == 0 else float(self.pitch)
            T = T @ self._T_from_R_t(
                Rot.from_euler("y", self.bend_axis_sign * a),
                np.array([step_x, 0.0, 0.0], dtype=float),
            )
        T_tip = T @ self._trans(self.pitch, 0.0, 0.0)
        return T_tip[:3, 3].copy()

    def tip_dir_x(self, q: np.ndarray, *, clamp: bool = True) -> float:
        _p, dirx = self.tip_position_and_dir_x(q, clamp=clamp, soft=False)
        return dirx

    def tip_alignment(self, q: np.ndarray, preferred_dir: np.ndarray, *, clamp: bool = True) -> float:
        _p, forward = self.tip_position_and_forward(q, clamp=clamp, soft=False)
        v = np.asarray(preferred_dir, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return float(np.clip(float(forward[0]), -1.0, 1.0))
        v /= n
        return float(np.clip(float(np.dot(forward, v)), -1.0, 1.0))

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        q0 = self._clamp_all(q)
        eps_theta = max(float(eps), 1e-4)
        eps_vec = np.array([5e-4, 5e-4, eps_theta, eps_theta], dtype=float)
        J = np.zeros((3, 4), dtype=float)
        for i in range(4):
            e = float(eps_vec[i])
            qp = q0.copy()
            qm = q0.copy()
            qp[i] += e
            qm[i] -= e
            pp = self.tip_position(qp, clamp=True, soft=True)
            pm = self.tip_position(qm, clamp=True, soft=True)
            J[:, i] = (pp - pm) / (2.0 * e)
        return J
