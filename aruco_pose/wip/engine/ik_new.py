#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from engine.config_loader import JointLimit


Vec3 = np.ndarray
Q4 = np.ndarray


@dataclass(frozen=True)
class GoalSpec:
    target_world: Vec3
    preferred_dir_world: Optional[Vec3] = None
    position_tol_m: float = 1e-4
    direction_tol_deg: Optional[float] = None


@dataclass(frozen=True)
class SearchBounds:
    linear_min: float
    linear_max: float
    roll_min: float
    roll_max: float
    seg1_min: float
    seg1_max: float
    seg2_min: float
    seg2_max: float


@dataclass(frozen=True)
class MultiSeedConfig:
    max_seeds: int = 16
    n_random: int = 8
    seed_rng: int = 0
    max_iters_per_seed: int = 120
    stall_limit: int = 30
    fd_eps: float = 1e-4
    damping: float = 1e-2
    line_search_shrink: float = 0.5
    line_search_steps: int = 6
    align_rounds: int = 3
    align_pos_tol_scale: float = 1.05


@dataclass(frozen=True)
class GoalCandidate:
    q: Q4
    tip_world: Vec3
    tip_forward_world: Vec3
    position_error_m: float
    direction_alignment: float
    direction_error_deg: Optional[float]
    position_ok: bool
    direction_ok: bool
    goal_ok: bool
    seed_index: int
    notes: str = ""


@dataclass(frozen=True)
class GoalSearchResult:
    success: bool
    best: Optional[GoalCandidate]
    candidates: tuple[GoalCandidate, ...]
    reason: str = ""


@dataclass(frozen=True)
class PathSpec:
    q_start: Q4
    q_goal: Q4
    max_step_linear_m: float = 0.01
    max_step_roll_rad: float = math.radians(5.0)
    max_step_bend_rad: float = math.radians(5.0)


@dataclass(frozen=True)
class PathWaypoint:
    q: Q4


@dataclass(frozen=True)
class PathPlan:
    success: bool
    waypoints: tuple[PathWaypoint, ...]
    cost: float
    reason: str = ""


class GoalStateFinder(Protocol):
    def solve(self, goal: GoalSpec) -> GoalSearchResult:
        ...


class PathPlanner(Protocol):
    def plan(self, spec: PathSpec) -> PathPlan:
        ...


def _normalize(v: Optional[Vec3]) -> Optional[Vec3]:
    if v is None:
        return None
    vv = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(vv))
    if n <= 1e-12:
        return None
    return vv / n


def _wrap_pi(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _direction_error_deg(alignment: float) -> float:
    return float(np.degrees(math.acos(float(np.clip(alignment, -1.0, 1.0)))))


class Kinematics:
    """
    Independent copy of the 4DOF continuum kinematics needed by ik_new.

    This intentionally does not import engine.ik_solver.
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
    ):
        self.pitch = float(pitch)
        self.n_nodes = int(n_nodes)
        self.n_seg = int(n_seg)
        self.origin_xyz = np.asarray(origin_xyz, dtype=float).reshape(3).copy()
        self.linear_min = float(limit.linear_min)
        self.linear_max = float(limit.linear_max)
        self.roll_min = float(limit.roll_min_rad())
        self.roll_max = float(limit.roll_max_rad())
        self.bend_lim = float(limit.bend_lim_rad())
        self.base_axis_sign = -1.0 if float(base_axis_sign) < 0.0 else 1.0
        self.roll_axis_sign = -1.0 if float(roll_axis_sign) < 0.0 else 1.0
        self.bend_axis_sign = -1.0 if float(bend_axis_sign) < 0.0 else 1.0

    def clamp_q(self, q: Sequence[float]) -> Q4:
        linear, roll, t1, t2 = map(float, np.asarray(q, dtype=float).reshape(4))
        return np.array(
            [
                np.clip(linear, self.linear_min, self.linear_max),
                np.clip(roll, self.roll_min, self.roll_max),
                np.clip(t1, -self.bend_lim, +self.bend_lim),
                np.clip(t2, -self.bend_lim, +self.bend_lim),
            ],
            dtype=float,
        )

    def tip_position_and_forward(self, q: Sequence[float]) -> tuple[Vec3, Vec3]:
        linear, roll, theta1, theta2 = map(float, self.clamp_q(q))
        phi = self.roll_axis_sign * roll
        cphi, sphi = math.cos(phi), math.sin(phi)
        c0 = np.array([1.0, 0.0, 0.0], dtype=float)
        c2 = np.array([0.0, -sphi, cphi], dtype=float)
        p = self.origin_xyz + np.array([self.base_axis_sign * linear, 0.0, 0.0], dtype=float)
        a1 = self.bend_axis_sign * theta1
        a2 = self.bend_axis_sign * theta2
        c1, s1 = math.cos(a1), math.sin(a1)
        c2a, s2 = math.cos(a2), math.sin(a2)

        for i in range(int(self.n_nodes)):
            if i > 0:
                p = p + self.pitch * c0
            ct, st = (c1, s1) if i < int(self.n_seg) else (c2a, s2)
            c0_old = c0
            c0 = ct * c0_old - st * c2
            c2 = st * c0_old + ct * c2

        p_tip = p + self.pitch * c0
        return p_tip.astype(float, copy=True), c0.astype(float, copy=True)

    def tip_alignment(self, q: Sequence[float], preferred_dir: Optional[Vec3]) -> float:
        _p, fwd = self.tip_position_and_forward(q)
        pref = _normalize(preferred_dir)
        if pref is None:
            return 1.0
        return float(np.clip(np.dot(fwd, pref), -1.0, 1.0))

    def position_error(self, q: Sequence[float], target_world: Sequence[float]) -> float:
        p, _fwd = self.tip_position_and_forward(q)
        return float(np.linalg.norm(p - np.asarray(target_world, dtype=float).reshape(3)))

    def numerical_jacobian(self, q: Sequence[float], eps: float) -> np.ndarray:
        q0 = self.clamp_q(q)
        eps_vec = np.array([max(5e-4, eps), max(5e-4, eps), max(1e-4, eps), max(1e-4, eps)], dtype=float)
        J = np.zeros((3, 4), dtype=float)
        for i, e in enumerate(eps_vec):
            qp = q0.copy()
            qm = q0.copy()
            qp[i] += float(e)
            qm[i] -= float(e)
            pp, _ = self.tip_position_and_forward(qp)
            pm, _ = self.tip_position_and_forward(qm)
            J[:, i] = (pp - pm) / (2.0 * float(e))
        return J


class SeedLocalSolver:
    """
    Phase 1: pure position solve.

    This stage intentionally ignores direction preference.
    It only tries to reach the target position accurately.
    """

    def __init__(self, kin: Kinematics, goal: GoalSpec, seed_q: Sequence[float], cfg: MultiSeedConfig):
        self.kin = kin
        self.goal = goal
        self.cfg = cfg
        self.q = self.kin.clamp_q(seed_q)
        self.best_q = self.q.copy()
        self.best_err = float("inf")
        self._stall = 0
    def _dls_step(self, J: np.ndarray, e: np.ndarray) -> np.ndarray:
        lam = float(self.cfg.damping)
        A = J @ J.T + (lam * lam) * np.eye(J.shape[0], dtype=float)
        y = np.linalg.solve(A, e)
        return (J.T @ y).reshape(4)

    def _objective(self, q: Q4) -> tuple[float, Vec3]:
        tip, _fwd = self.kin.tip_position_and_forward(q)
        err = np.asarray(self.goal.target_world, dtype=float).reshape(3) - tip
        return float(np.dot(err, err)), err

    def solve(self, *, seed_index: int) -> GoalCandidate:
        obj_best = float("inf")

        for _ in range(int(self.cfg.max_iters_per_seed)):
            obj0, e = self._objective(self.q)
            err_norm = float(math.sqrt(obj0))
            if err_norm < self.best_err:
                self.best_err = err_norm
                self.best_q = self.q.copy()
                self._stall = 0
            else:
                self._stall += 1
                if self._stall >= int(self.cfg.stall_limit):
                    break
            if obj0 < obj_best:
                obj_best = obj0

            if err_norm <= float(self.goal.position_tol_m):
                break

            J = self.kin.numerical_jacobian(self.q, eps=float(self.cfg.fd_eps))
            dq = self._dls_step(J, e)
            alpha = 1.0
            accepted = False
            for _ in range(int(self.cfg.line_search_steps)):
                q_try = self.kin.clamp_q(self.q + alpha * dq)
                obj_try, _ = self._objective(q_try)
                if obj_try < obj0:
                    self.q = q_try
                    accepted = True
                    break
                alpha *= float(self.cfg.line_search_shrink)
            if not accepted:
                break

        qf = self.best_q.copy()
        tip, fwd = self.kin.tip_position_and_forward(qf)
        pos_err = float(np.linalg.norm(tip - np.asarray(self.goal.target_world, dtype=float).reshape(3)))
        pos_ok = bool(pos_err <= float(self.goal.position_tol_m))
        return GoalCandidate(
            q=qf,
            tip_world=tip,
            tip_forward_world=fwd,
            position_error_m=pos_err,
            direction_alignment=1.0,
            direction_error_deg=None,
            position_ok=pos_ok,
            direction_ok=True,
            goal_ok=pos_ok,
            seed_index=int(seed_index),
            notes="multiseed position solve",
        )


class CandidateRanker:
    """
    Terminal-state ranking only.

    Path cost is intentionally excluded.
    """

    @staticmethod
    def better_position(a: GoalCandidate, b: Optional[GoalCandidate]) -> bool:
        if b is None:
            return True
        if a.position_ok != b.position_ok:
            return bool(a.position_ok and (not b.position_ok))
        if a.position_error_m < b.position_error_m - 1e-12:
            return True
        if abs(a.position_error_m - b.position_error_m) <= 1e-12 and a.seed_index < b.seed_index:
            return True
        return False

    @staticmethod
    def better_direction(a: GoalCandidate, b: Optional[GoalCandidate]) -> bool:
        if b is None:
            return True
        if a.direction_ok != b.direction_ok:
            return bool(a.direction_ok and (not b.direction_ok))
        if a.direction_alignment > b.direction_alignment + 1e-9:
            return True
        if abs(a.direction_alignment - b.direction_alignment) <= 1e-9:
            if a.position_error_m < b.position_error_m - 1e-12:
                return True
            if abs(a.position_error_m - b.position_error_m) <= 1e-12 and a.seed_index < b.seed_index:
                return True
        return False


class HoverAlignSolver:
    """
    Phase 2: after position is reached, adjust posture while staying near the
    reached position, similar to a hover/align mode.
    """

    def __init__(self, kin: Kinematics, goal: GoalSpec, q_init: Sequence[float], cfg: MultiSeedConfig):
        self.kin = kin
        self.goal = goal
        self.cfg = cfg
        self.q = self.kin.clamp_q(q_init)
        self._preferred_dir = _normalize(goal.preferred_dir_world)

    def solve(self, *, seed_index: int) -> GoalCandidate:
        tip, fwd = self.kin.tip_position_and_forward(self.q)
        pos_err0 = float(np.linalg.norm(tip - np.asarray(self.goal.target_world, dtype=float).reshape(3)))
        align0 = self.kin.tip_alignment(self.q, self._preferred_dir)
        best_q = self.q.copy()
        best_tip = tip.copy()
        best_fwd = fwd.copy()
        best_pos_err = pos_err0
        best_align = align0

        tol_margin = max(float(self.goal.position_tol_m) * float(self.cfg.align_pos_tol_scale), float(self.goal.position_tol_m) + 1e-7)
        step_sets = (
            np.array([0.01, math.radians(10.0), math.radians(8.0), math.radians(8.0)], dtype=float),
            np.array([0.005, math.radians(5.0), math.radians(4.0), math.radians(4.0)], dtype=float),
            np.array([0.002, math.radians(2.0), math.radians(2.0), math.radians(2.0)], dtype=float),
        )

        for steps in step_sets[: max(0, int(self.cfg.align_rounds))]:
            improved = True
            while improved:
                improved = False
                q_base = best_q.copy()
                for i in range(4):
                    for sgn in (-1.0, 1.0):
                        q_try = q_base.copy()
                        q_try[i] += sgn * float(steps[i])
                        q_try = self.kin.clamp_q(q_try)
                        tip_try, fwd_try = self.kin.tip_position_and_forward(q_try)
                        pos_err = float(np.linalg.norm(tip_try - np.asarray(self.goal.target_world, dtype=float).reshape(3)))
                        if pos_err > tol_margin:
                            continue
                        align = self.kin.tip_alignment(q_try, self._preferred_dir)
                        if (align > best_align + 1e-6) or (abs(align - best_align) <= 1e-6 and pos_err < best_pos_err - 1e-9):
                            best_q = q_try
                            best_tip = tip_try
                            best_fwd = fwd_try
                            best_pos_err = pos_err
                            best_align = align
                            improved = True

        dir_err = None if self._preferred_dir is None else _direction_error_deg(best_align)
        pos_ok = bool(best_pos_err <= float(self.goal.position_tol_m))
        dir_ok = True
        if self.goal.direction_tol_deg is not None and dir_err is not None:
            dir_ok = bool(dir_err <= float(self.goal.direction_tol_deg))
        return GoalCandidate(
            q=best_q,
            tip_world=best_tip,
            tip_forward_world=best_fwd,
            position_error_m=best_pos_err,
            direction_alignment=best_align,
            direction_error_deg=dir_err,
            position_ok=pos_ok,
            direction_ok=dir_ok,
            goal_ok=bool(pos_ok and dir_ok),
            seed_index=int(seed_index),
            notes="hover align after reach",
        )


class MultiSeedGoalFinder:
    """
    New goal-state search:
    - no single-first / fallback split
    - always multiseed from the start
    - each seed solves terminal state only
    """

    def __init__(self, kin: Kinematics, bounds: SearchBounds, cfg: Optional[MultiSeedConfig] = None):
        self.kin = kin
        self.bounds = bounds
        self.cfg = cfg if cfg is not None else MultiSeedConfig()

    def _linear_guess(self, target_x: float) -> float:
        reach = float((self.kin.n_nodes - 1) * self.kin.pitch + self.kin.pitch)
        bx = float(target_x - self.kin.origin_xyz[0] - reach)
        return float(np.clip(bx, self.bounds.linear_min, self.bounds.linear_max))

    def _dedup(self, seeds: list[Q4]) -> list[Q4]:
        out: list[Q4] = []
        seen: set[tuple[int, int, int, int]] = set()
        for s in seeds:
            q = self.kin.clamp_q(s)
            key = tuple(int(round(float(v) * 10000.0)) for v in q)
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    def _make_seeds(self, goal: GoalSpec) -> list[Q4]:
        rng = np.random.default_rng(int(self.cfg.seed_rng))
        t_lim = float(self.kin.bend_lim)
        bxg = self._linear_guess(float(np.asarray(goal.target_world, dtype=float).reshape(3)[0]))
        t_big = 0.75 * t_lim
        t_med = 0.45 * t_lim
        seeds: list[Q4] = [
            np.array([bxg, 0.0, 0.0, 0.0], dtype=float),
            np.array([bxg, 0.0, +t_big, -t_big], dtype=float),
            np.array([bxg, 0.0, -t_big, +t_big], dtype=float),
            np.array([bxg, 0.0, +t_med, -t_med], dtype=float),
            np.array([bxg, 0.0, -t_med, +t_med], dtype=float),
            np.array([bxg, 0.5 * math.pi, +t_big, -t_big], dtype=float),
            np.array([bxg, -0.5 * math.pi, -t_big, +t_big], dtype=float),
        ]
        for _ in range(max(0, int(self.cfg.n_random))):
            seeds.append(
                np.array(
                    [
                        rng.uniform(self.bounds.linear_min, self.bounds.linear_max),
                        rng.uniform(self.bounds.roll_min, self.bounds.roll_max),
                        rng.uniform(self.bounds.seg1_min, self.bounds.seg1_max),
                        rng.uniform(self.bounds.seg2_min, self.bounds.seg2_max),
                    ],
                    dtype=float,
                )
            )
        seeds = self._dedup(seeds)
        return seeds[: max(1, int(self.cfg.max_seeds))]

    def solve(self, goal: GoalSpec) -> GoalSearchResult:
        seeds = self._make_seeds(goal)
        best_reach: Optional[GoalCandidate] = None
        reach_candidates: list[GoalCandidate] = []
        for i, seed in enumerate(seeds):
            cand = SeedLocalSolver(self.kin, goal, seed, self.cfg).solve(seed_index=i)
            reach_candidates.append(cand)
            if CandidateRanker.better_position(cand, best_reach):
                best_reach = cand

        if best_reach is None:
            return GoalSearchResult(success=False, best=None, candidates=(), reason="no seed candidates")

        if (goal.preferred_dir_world is None) or (goal.direction_tol_deg is None):
            ranked_reach = tuple(sorted(reach_candidates, key=lambda c: (0 if c.position_ok else 1, c.position_error_m, c.seed_index)))
            return GoalSearchResult(
                success=bool(best_reach.position_ok),
                best=best_reach,
                candidates=ranked_reach,
                reason="position goal found" if best_reach.position_ok else "no seed satisfied position tolerance",
            )

        position_ok = [c for c in reach_candidates if c.position_ok]
        if not position_ok:
            ranked_reach = tuple(sorted(reach_candidates, key=lambda c: (c.position_error_m, c.seed_index)))
            return GoalSearchResult(
                success=False,
                best=best_reach,
                candidates=ranked_reach,
                reason="no seed satisfied position tolerance",
            )

        aligned: list[GoalCandidate] = []
        best_final: Optional[GoalCandidate] = None
        for cand in position_ok:
            aligned_cand = HoverAlignSolver(self.kin, goal, cand.q, self.cfg).solve(seed_index=cand.seed_index)
            aligned.append(aligned_cand)
            if CandidateRanker.better_direction(aligned_cand, best_final):
                best_final = aligned_cand

        assert best_final is not None
        ranked = tuple(
            sorted(
                aligned,
                key=lambda c: (
                    0 if c.direction_ok else 1,
                    -c.direction_alignment,
                    c.position_error_m,
                    c.seed_index,
                ),
            )
        )
        return GoalSearchResult(
            success=bool(best_final.goal_ok),
            best=best_final,
            candidates=ranked,
            reason="goal-state found" if best_final.goal_ok else "position reached but direction align failed",
        )


class LinearJointPathPlanner:
    def plan(self, spec: PathSpec) -> PathPlan:
        q0 = np.asarray(spec.q_start, dtype=float).reshape(4)
        q1 = np.asarray(spec.q_goal, dtype=float).reshape(4)
        dq = q1 - q0
        steps = max(
            1,
            int(
                math.ceil(
                    max(
                        abs(float(dq[0])) / max(float(spec.max_step_linear_m), 1e-9),
                        abs(float(dq[1])) / max(float(spec.max_step_roll_rad), 1e-9),
                        abs(float(dq[2])) / max(float(spec.max_step_bend_rad), 1e-9),
                        abs(float(dq[3])) / max(float(spec.max_step_bend_rad), 1e-9),
                    )
                )
            ),
        )
        waypoints = []
        for i in range(steps + 1):
            u = float(i) / float(max(steps, 1))
            waypoints.append(PathWaypoint(q=(q0 + u * dq).astype(float, copy=True)))
        cost = float(np.sum(np.abs(dq)))
        return PathPlan(success=True, waypoints=tuple(waypoints), cost=cost, reason="linear joint interpolation")


@dataclass(frozen=True)
class IKNewResult:
    success: bool
    goal_search: GoalSearchResult
    path_plan: Optional[PathPlan]
    q_goal: Optional[Q4]
    reason: str = ""


class IKNewPipeline:
    def __init__(self, goal_finder: GoalStateFinder, path_planner: PathPlanner):
        self.goal_finder = goal_finder
        self.path_planner = path_planner

    def solve(self, *, q_start: Sequence[float], goal: GoalSpec) -> IKNewResult:
        goal_result = self.goal_finder.solve(goal)
        if (not goal_result.success) or (goal_result.best is None):
            return IKNewResult(False, goal_result, None, None, goal_result.reason)
        q_goal = np.asarray(goal_result.best.q, dtype=float).reshape(4).copy()
        path = self.path_planner.plan(PathSpec(q_start=np.asarray(q_start, dtype=float).reshape(4), q_goal=q_goal))
        if not path.success:
            return IKNewResult(False, goal_result, path, q_goal, path.reason)
        return IKNewResult(True, goal_result, path, q_goal, "goal-state and path both found")


def make_default_search_bounds(kin: Kinematics) -> SearchBounds:
    return SearchBounds(
        linear_min=float(kin.linear_min),
        linear_max=float(kin.linear_max),
        roll_min=float(kin.roll_min),
        roll_max=float(kin.roll_max),
        seg1_min=-float(kin.bend_lim),
        seg1_max=+float(kin.bend_lim),
        seg2_min=-float(kin.bend_lim),
        seg2_max=+float(kin.bend_lim),
    )


def demo_pipeline(
    kin: Kinematics,
    *,
    q_start: Sequence[float],
    target_world: Sequence[float],
    preferred_dir_world: Optional[Sequence[float]] = None,
    direction_tol_deg: Optional[float] = None,
) -> IKNewResult:
    pipeline = IKNewPipeline(
        goal_finder=MultiSeedGoalFinder(kin, make_default_search_bounds(kin)),
        path_planner=LinearJointPathPlanner(),
    )
    return pipeline.solve(
        q_start=np.asarray(q_start, dtype=float).reshape(4),
        goal=GoalSpec(
            target_world=np.asarray(target_world, dtype=float).reshape(3),
            preferred_dir_world=None if preferred_dir_world is None else np.asarray(preferred_dir_world, dtype=float).reshape(3),
            direction_tol_deg=direction_tol_deg,
        ),
    )
