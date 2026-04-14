#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from engine.config_loader import JointLimit
from engine.kinematics import Kinematics


Vec3 = np.ndarray
Q4 = np.ndarray

Q_I = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
Q_C = np.array([0.0, 0.0, -math.radians(36.0), +math.radians(36.0)], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= 1e-12:
        return np.zeros_like(vv)
    return vv / n


@dataclass(frozen=True)
class SolveRequest:
    target_world: Vec3
    position_tol_m: float = 1e-4


@dataclass(frozen=True)
class SolveResult:
    success: bool
    q: Optional[Q4]
    position_error_m: float
    seed_name: str
    iterations: int
    reason: str = ""


class PositionSolver:

    def __init__(
        self,
        kin: Kinematics,
        *,
        max_iters: int = 120,
        damping: float = 1e-2,
        line_search_shrink: float = 0.5,
        line_search_steps: int = 6,
        fd_eps: float = 1e-4,
        i_seed: Optional[Sequence[float]] = None,
        c_seed: Optional[Sequence[float]] = None,
    ):
        self.kin = kin
        self.max_iters = max(int(max_iters), 1)
        self.damping = float(max(damping, 1e-9))
        self.line_search_shrink = float(np.clip(line_search_shrink, 1e-3, 0.999))
        self.line_search_steps = max(int(line_search_steps), 1)
        self.fd_eps = float(max(fd_eps, 1e-6))
        self.i_seed = self.kin.clamp_q(i_seed if i_seed is not None else self._default_i_seed())
        self.c_seed = self.kin.clamp_q(c_seed if c_seed is not None else self._default_c_seed())

    def solve(self, request: SolveRequest) -> SolveResult:
        target = np.asarray(request.target_world, dtype=float).reshape(3)
        tol = float(max(request.position_tol_m, 0.0))

        for seed_name, q_seed in self._canonical_seeds():
            result = self._solve_from_seed(target, q_seed, tol, seed_name)
            if result.success:
                return result

        return SolveResult(
            success=False,
            q=None,
            position_error_m=float("inf"),
            seed_name="C",
            iterations=self.max_iters,
            reason="position tolerance not reached",
        )

    def _canonical_seeds(self) -> list[tuple[str, Q4]]:
        return [
            ("I", self.i_seed.copy()),
            ("C", self.c_seed.copy()),
        ]

    def _default_i_seed(self) -> Q4:
        return Q_I.copy()

    def _default_c_seed(self) -> Q4:
        return Q_C.copy()

    def _solve_from_seed(self, target_world: Vec3, q_seed: Q4, tol: float, seed_name: str) -> SolveResult:
        q = self.kin.clamp_q(q_seed)
        err = self.kin.position_error(q, target_world)
        if err <= tol:
            return SolveResult(True, q.copy(), err, seed_name, 0, "seed already satisfies tolerance")

        for iteration in range(1, self.max_iters + 1):
            err_vec = self.kin.position_error_vec(q, target_world)
            err = float(np.linalg.norm(err_vec))
            if err <= tol:
                return SolveResult(True, q.copy(), err, seed_name, iteration - 1, "converged")

            J = self.kin.numerical_jacobian(q, eps=self.fd_eps)
            H = J.T @ J + self.damping * np.eye(4, dtype=float)
            g = J.T @ err_vec
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.pinv(H) @ g

            if float(np.linalg.norm(step)) <= 1e-12:
                break

            accepted = False
            step_dir = _normalize(step)
            step_norm = float(np.linalg.norm(step))
            for ls_idx in range(self.line_search_steps):
                alpha = self.line_search_shrink ** ls_idx
                q_try = self.kin.clamp_q(q + (alpha * step_norm) * step_dir)
                err_try = self.kin.position_error(q_try, target_world)
                if err_try < err:
                    q = q_try
                    err = err_try
                    accepted = True
                    break

            if not accepted:
                break

        return SolveResult(
            success=bool(err <= tol),
            q=q.copy(),
            position_error_m=float(err),
            seed_name=seed_name,
            iterations=self.max_iters,
            reason="max_iters reached" if err > tol else "converged",
        )
