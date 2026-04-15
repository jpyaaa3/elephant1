#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


Q4 = np.ndarray
Vec3 = np.ndarray


def _normalize(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(vv))
    if n <= 1e-12:
        return np.zeros_like(vv)
    return vv / n


@dataclass(frozen=True)
class OptimizeResult:
    success: bool
    q: Q4
    error_norm: float
    iterations: int
    reason: str = ""


class IkPositionOptimizer:
    """Pure optimizer: reduce position error norm for a given problem definition."""

    def __init__(
        self,
        *,
        max_iters: int = 120,
        damping: float = 1e-2,
        line_search_shrink: float = 0.5,
        line_search_steps: int = 6,
        fd_eps: float = 1e-4,
    ):
        self.max_iters = max(int(max_iters), 1)
        self.damping = float(max(damping, 1e-9))
        self.line_search_shrink = float(np.clip(line_search_shrink, 1e-3, 0.999))
        self.line_search_steps = max(int(line_search_steps), 1)
        self.fd_eps = float(max(fd_eps, 1e-6))

    def solve(
        self,
        *,
        q0: Sequence[float],
        tol: float,
        error_vec_fn: Callable[[Q4], Vec3],
        jacobian_fn: Callable[[Q4, float], np.ndarray],
        clamp_q_fn: Optional[Callable[[Q4], Q4]] = None,
    ) -> OptimizeResult:
        q = np.asarray(q0, dtype=float).reshape(4)
        if clamp_q_fn is not None:
            q = clamp_q_fn(q)
        err_vec = np.asarray(error_vec_fn(q), dtype=float).reshape(3)
        err = float(np.linalg.norm(err_vec))
        if err <= tol:
            return OptimizeResult(True, q.copy(), err, 0, "seed already satisfies tolerance")

        for iteration in range(1, self.max_iters + 1):
            err_vec = np.asarray(error_vec_fn(q), dtype=float).reshape(3)
            err = float(np.linalg.norm(err_vec))
            if err <= tol:
                return OptimizeResult(True, q.copy(), err, iteration - 1, "converged")

            J = np.asarray(jacobian_fn(q, self.fd_eps), dtype=float).reshape(3, 4)
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
                q_try = q + (alpha * step_norm) * step_dir
                if clamp_q_fn is not None:
                    q_try = clamp_q_fn(q_try)
                err_try_vec = np.asarray(error_vec_fn(q_try), dtype=float).reshape(3)
                err_try = float(np.linalg.norm(err_try_vec))
                if err_try < err:
                    q = q_try
                    err = err_try
                    accepted = True
                    break

            if not accepted:
                break

        return OptimizeResult(
            success=bool(err <= tol),
            q=q.copy(),
            error_norm=float(err),
            iterations=self.max_iters,
            reason="max_iters reached" if err > tol else "converged",
        )
