#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.config_loader import JointLimit
from engine.kinematics import Kinematics
from engine.iklib_src.solver import Q_C, Q_I, PositionSolver, SolveRequest, SolveResult


def solve(
    *,
    target_world: Sequence[float],
    pitch: float,
    n_nodes: int,
    n_seg: int,
    origin_xyz: Sequence[float],
    limit: JointLimit,
    position_tol_m: float = 1e-4,
    max_iters: int = 120,
    base_axis_sign: float = 1.0,
    roll_axis_sign: float = 1.0,
    bend_axis_sign: float = -1.0,
    i_seed: Optional[Sequence[float]] = None,
    c_seed: Optional[Sequence[float]] = None,
) -> SolveResult:
    kin = Kinematics(
        pitch=float(pitch),
        n_nodes=int(n_nodes),
        n_seg=int(n_seg),
        origin_xyz=np.asarray(origin_xyz, dtype=float).reshape(3),
        limit=limit,
        base_axis_sign=float(base_axis_sign),
        roll_axis_sign=float(roll_axis_sign),
        bend_axis_sign=float(bend_axis_sign),
    )
    solver = PositionSolver(
        kin,
        max_iters=int(max_iters),
        i_seed=i_seed,
        c_seed=c_seed,
    )
    return solver.solve(
        SolveRequest(
            target_world=np.asarray(target_world, dtype=float).reshape(3),
            position_tol_m=float(position_tol_m),
        )
    )


__all__ = [
    "Q_I",
    "Q_C",
    "SolveRequest",
    "SolveResult",
    "solve",
]
