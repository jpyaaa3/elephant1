#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as Rot
try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None

from engine.config_loader import IkConfig, JointLimit, SimParam


class RateLimiter:
    def __init__(self, max_rate: np.ndarray):
        self._max_rate = np.array(max_rate, dtype=float)

    def step(self, q_cmd: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        dq = q_target - q_cmd
        max_step = self._max_rate * dt
        dq_clamped = np.clip(dq, -max_step, +max_step)
        return q_cmd + dq_clamped


def _to_numpy_1d(raw: Any) -> np.ndarray:
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    return np.array(raw, dtype=float).reshape(-1)


def _as_single_dof_index(raw_idx: Any) -> int:
    if isinstance(raw_idx, (list, tuple, np.ndarray)):
        arr = np.array(raw_idx).reshape(-1)
        if arr.size <= 0:
            raise ValueError("empty dof index list")
        return int(arr[0])
    return int(raw_idx)


class Mover:
    def __init__(
        self,
        entity: Any,
        params: SimParam,
        limit: JointLimit,
        n_nodes: int,
        n_seg: Optional[int] = None,
        *,
        base_joint_name: str = "base_prismatic_x",
        roll_joint_name: str = "base_roll_x",
        bend_joint_names: Optional[List[str]] = None,
    ):
        self.entity = entity
        self.p = params
        self.limit = limit
        self.n_nodes = int(n_nodes)
        self.n_seg = int(n_seg) if n_seg is not None else (self.n_nodes // 2)
        self._base_joint_name = str(base_joint_name)
        self._roll_joint_name = str(roll_joint_name)
        self._bend_joint_names = (
            [str(x) for x in bend_joint_names]
            if bend_joint_names is not None
            else [f"bend_{i}" for i in range(self.n_nodes)]
        )

        base_names = [self._base_joint_name, self._roll_joint_name]
        bend_names = list(self._bend_joint_names)

        pairs: List[Tuple[int, str]] = []
        for name in (base_names + bend_names):
            j = self.entity.get_joint(name)
            idx = getattr(j, "dofs_idx_local")
            pairs.append((_as_single_dof_index(idx), name))

        pairs.sort(key=lambda t: int(t[0]))
        self.joint_names: List[str] = [n for _, n in pairs]
        self.dofs_idx_local: List[int] = [int(i) for i, _ in pairs]

        self._name2pos: Dict[str, int] = {n: k for k, n in enumerate(self.joint_names)}
        self._bend_pos: List[int] = [self._name2pos[name] for name in self._bend_joint_names]

        self.bend_lim = float(limit.bend_lim_rad())
        max_rate = np.array(
            [float(params.linear_rate), float(params.roll_rate)] + [float(params.bend_rate)] * self.n_nodes,
            dtype=float,
        )
        self._rate = RateLimiter(max_rate=max_rate)

        try:
            raw0 = self.entity.get_dofs_position(dofs_idx_local=self.dofs_idx_local)
            q0 = _to_numpy_1d(raw0)
            if q0.shape[0] != len(self.dofs_idx_local):
                q0 = q0[: len(self.dofs_idx_local)]
            self._q_cmd = q0.copy()
        except Exception:
            self._q_cmd = np.zeros(len(self.dofs_idx_local), dtype=float)

        self._last_q_target: Optional[np.ndarray] = None
        self._last_q4: Optional[Tuple[float, float, float, float]] = None

    def idx_linear(self) -> Optional[int]:
        return self._name2pos.get(self._base_joint_name, None)

    def idx_roll(self) -> Optional[int]:
        return self._name2pos.get(self._roll_joint_name, None)

    def bend_indices(self) -> List[int]:
        return list(self._bend_pos)

    def dof_names(self) -> List[str]:
        return list(self.joint_names)

    def dof_count(self) -> int:
        return int(len(self.joint_names))

    def get_dofs_position(self) -> np.ndarray:
        raw = self.entity.get_dofs_position(dofs_idx_local=self.dofs_idx_local)
        q = _to_numpy_1d(raw)
        if q.shape[0] != len(self.dofs_idx_local):
            q = q[: len(self.dofs_idx_local)]
        return q

    def get_last_target_full(self) -> Optional[np.ndarray]:
        return None if self._last_q_target is None else self._last_q_target.copy()

    def get_last_command_full(self) -> np.ndarray:
        return self._q_cmd.copy()

    def _apply_q_direct(self, q_target: np.ndarray) -> None:
        try:
            self.entity.set_dofs_position(q_target, dofs_idx_local=self.dofs_idx_local)
        except Exception:
            self.entity.control_dofs_position(q_target, dofs_idx_local=self.dofs_idx_local)

    def target_from_4dof(self, linear: float, roll: float, theta1: float, theta2: float) -> np.ndarray:
        bx = float(np.clip(float(linear), self.limit.linear_min, self.limit.linear_max))
        rl = float(np.clip(float(roll), self.limit.roll_min_rad(), self.limit.roll_max_rad()))
        t1 = float(np.clip(float(theta1), -self.bend_lim, +self.bend_lim))
        t2 = float(np.clip(float(theta2), -self.bend_lim, +self.bend_lim))

        vals: Dict[str, float] = {self._base_joint_name: bx, self._roll_joint_name: rl}
        for i in range(self.n_nodes):
            vals[self._bend_joint_names[i]] = t1 if i < self.n_seg else t2

        return np.array([vals[n] for n in self.joint_names], dtype=float)

    def control_4dof(self, linear: float, roll: float, theta1: float, theta2: float):
        q_target = self.target_from_4dof(linear, roll, theta1, theta2)
        self._last_q_target = q_target
        self._last_q4 = (float(linear), float(roll), float(theta1), float(theta2))
        self._q_cmd = self._rate.step(self._q_cmd, q_target, dt=float(self.p.dt))
        self._apply_q_direct(self._q_cmd)

    def set_4dof_instant(self, linear: float, roll: float, theta1: float, theta2: float) -> None:
        """
        Snap DOFs to target immediately (simulation-only convenience).
        Falls back to position control command if direct setter is unavailable.
        """
        q_target = self.target_from_4dof(linear, roll, theta1, theta2)
        self._last_q_target = q_target
        self._last_q4 = (float(linear), float(roll), float(theta1), float(theta2))
        self._q_cmd = q_target.copy()
        self._apply_q_direct(q_target)


def is_tracking_full(
    Mover: Mover,
    q_des: np.ndarray,
    tol_linear: float = 5e-4,
    tol_roll: float = 1e-3,
    tol_bend: float = 2e-3,
) -> bool:
    q_cur = Mover.get_dofs_position()
    if q_cur.shape != q_des.shape:
        return False

    dq = q_cur - q_des
    idx_bx = Mover.idx_linear()
    idx_rl = Mover.idx_roll()
    if idx_bx is None or idx_rl is None:
        return False
    if abs(float(dq[idx_bx])) > tol_linear:
        return False
    if abs(float(dq[idx_rl])) > tol_roll:
        return False
    for pos in Mover.bend_indices():
        if abs(float(dq[pos])) > tol_bend:
            return False
    return True


@dataclass
class IkInfo:
    status: str
    iters: int
    err_norm: float
    damping: float
    alpha: float
    accepted: bool
    stall_count: int


@dataclass(frozen=True)
class TaskSpaceRefineConfig:
    ds: float = 0.35
    goal_tol_m: float = 1e-4
    max_step_m: Optional[float] = 0.03


class TaskSpaceRefiner:
    """
    Outer task-space correction loop.

    Given the measured tip position x_i and the user goal x_goal, this
    computes an intermediate target

        x_{i+1} = x_i + ds * (x_goal - x_i)

    and lets the existing IK solve for that softened target. This keeps the
    low-level IK unchanged while providing a stable feedback-friendly target
    update rule.
    """

    def __init__(self, goal_xyz: np.ndarray, cfg: Optional[TaskSpaceRefineConfig] = None):
        self.goal = np.asarray(goal_xyz, dtype=float).reshape(3).copy()
        self.cfg = cfg if cfg is not None else TaskSpaceRefineConfig()

    def next_target(self, current_tip_xyz: np.ndarray) -> np.ndarray:
        current = np.asarray(current_tip_xyz, dtype=float).reshape(3)
        err = self.goal - current
        err_norm = float(np.linalg.norm(err))
        if err_norm <= float(self.cfg.goal_tol_m):
            return self.goal.copy()

        ds = float(np.clip(float(self.cfg.ds), 1e-6, 1.0))
        step = ds * err

        max_step_m = self.cfg.max_step_m
        if (max_step_m is not None) and np.isfinite(float(max_step_m)) and (float(max_step_m) > 0.0):
            step_norm = float(np.linalg.norm(step))
            if step_norm > float(max_step_m):
                step *= float(max_step_m) / max(step_norm, 1e-12)

        return (current + step).astype(float, copy=True)

    def make_solver(
        self,
        kin: "Kinematics",
        current_tip_xyz: np.ndarray,
        q_seed: np.ndarray,
        *,
        max_solvers: int = 8,
        n_random: int = 2,
        seed_rng: int | None = None,
        ik_cfg: Optional[IkConfig] = None,
    ) -> "MultiSeedIk":
        target_xyz = self.next_target(current_tip_xyz)
        return MultiSeedIk(
            kin,
            target_xyz,
            q_seed,
            max_solvers=max_solvers,
            n_random=n_random,
            seed_rng=seed_rng,
            ik_cfg=ik_cfg,
        )


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

    def _soft_clamp(self, x: float, lo: float, hi: float) -> float:
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return lo
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        return mid + half * math.tanh(self.soft_k * (float(x) - mid) / max(half, 1e-12))

    def _clamp_all(self, q: np.ndarray) -> np.ndarray:
        linear, roll, t1, t2 = map(float, np.asarray(q, dtype=float).reshape(4))
        linear = float(np.clip(linear, self.linear_min, self.linear_max))
        roll = float(np.clip(roll, self.roll_min, self.roll_max))
        t1 = float(np.clip(t1, -self.bend_lim, +self.bend_lim))
        t2 = float(np.clip(t2, -self.bend_lim, +self.bend_lim))
        return np.array([linear, roll, t1, t2], dtype=float)

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
        # World columns of local frame after Rx(phi).
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


class _SingleIKSolver:
    def __init__(
        self,
        kin: Kinematics,
        target_xyz: np.ndarray,
        q_init: np.ndarray,
        cfg: IkConfig,
        *,
        q_path_ref: Optional[np.ndarray] = None,
        preferred_dir_world: Optional[np.ndarray] = None,
    ):
        self.kin = kin
        self.target = np.array(target_xyz, dtype=float).reshape(3)
        self.q = self.kin._clamp_all(np.array(q_init, dtype=float).reshape(4))
        if q_path_ref is None:
            self._q_ref = self.q.copy()
        else:
            self._q_ref = self.kin._clamp_all(np.array(q_path_ref, dtype=float).reshape(4))
        if preferred_dir_world is None:
            self._preferred_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self._preferred_dir = np.asarray(preferred_dir_world, dtype=float).reshape(3).copy()
        pref_norm = float(np.linalg.norm(self._preferred_dir))
        if pref_norm <= 1e-12:
            self._preferred_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self._preferred_dir /= pref_norm
        self.cfg = cfg

        self.lmbd = float(cfg.damping_init)
        self.iters = 0
        self.converged = False
        self.failed = False
        self.last_err_norm = float("inf")
        self._best_err = float("inf")
        self._stall = 0
        self._lo = np.array(
            [self.kin.linear_min, self.kin.roll_min, -self.kin.bend_lim, -self.kin.bend_lim],
            dtype=float,
        )
        self._hi = np.array(
            [self.kin.linear_max, self.kin.roll_max, +self.kin.bend_lim, +self.kin.bend_lim],
            dtype=float,
        )
        self._dir_jac_cache: Optional[np.ndarray] = None
        self._dir_jac_cache_q: Optional[np.ndarray] = None
        self._dir_jac_cache_age: int = 999
        self._dir_jac_reuse_steps: int = 2
        self._dir_jac_reuse_tol: float = 2e-3
        self._q_span = np.maximum(self._hi - self._lo, np.array([1e-6, 1e-6, 1e-6, 1e-6], dtype=float))
        self._dir_improve_eps: float = 1e-4
        self._posture_improve_eps: float = 1e-6

    @staticmethod
    def _dls_step(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
        J = np.asarray(J, dtype=float)
        e = np.asarray(e, dtype=float).reshape(-1)
        A = J @ J.T + (float(damping) ** 2) * np.eye(J.shape[0], dtype=float)
        try:
            y = np.linalg.solve(A, e)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(A, e, rcond=None)[0]
        return (J.T @ y).reshape(4)

    @staticmethod
    def _damped_pinv(J: np.ndarray, damping: float) -> np.ndarray:
        J = np.asarray(J, dtype=float)
        A = J @ J.T + (float(damping) ** 2) * np.eye(J.shape[0], dtype=float)
        try:
            A_inv = np.linalg.solve(A, np.eye(J.shape[0], dtype=float))
        except np.linalg.LinAlgError:
            A_inv = np.linalg.lstsq(A, np.eye(J.shape[0], dtype=float), rcond=None)[0]
        return J.T @ A_inv

    @staticmethod
    def _wrap_roll(r: float) -> float:
        return float(((float(r) + math.pi) % (2.0 * math.pi)) - math.pi)

    def posture_delta_norm(self, q: np.ndarray) -> np.ndarray:
        qv = np.asarray(q, dtype=float).reshape(4)
        dq = qv - self._q_ref
        dq[1] = self._wrap_roll(dq[1])
        return dq / self._q_span

    def posture_cost(self, q: np.ndarray) -> float:
        dq_norm = self.posture_delta_norm(q)
        return float(np.dot(dq_norm, dq_norm))

    def posture_gradient(self, q: np.ndarray) -> np.ndarray:
        dq = np.asarray(q, dtype=float).reshape(4) - self._q_ref
        dq[1] = self._wrap_roll(dq[1])
        denom = np.maximum(self._q_span * self._q_span, np.array([1e-12, 1e-12, 1e-12, 1e-12], dtype=float))
        return 2.0 * dq / denom

    def _jacobian_fd_shared(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        q0 = self.kin._clamp_all(q)
        p0, forward0 = self.kin.tip_position_and_forward(q0, clamp=True, soft=False)
        dir0 = float(np.clip(float(np.dot(forward0, self._preferred_dir)), -1.0, 1.0))
        e_pos = (self.target - p0).reshape(3)
        e_dir = float(1.0 - dir0)
        eps_base = max(1e-5, float(self.cfg.fd_eps) * 0.5)
        eps_theta = max(5e-5, float(self.cfg.fd_eps))
        eps = np.array([eps_base, eps_base, eps_theta, eps_theta], dtype=float)
        J = np.zeros((3, 4), dtype=float)
        Jd = np.zeros(4, dtype=float)

        for i in range(4):
            e = float(eps[i])

            if i == 1:
                qp = q0.copy()
                qm = q0.copy()
                qp[i] = self._wrap_roll(q0[i] + e)
                qm[i] = self._wrap_roll(q0[i] - e)
                pp, fp = self.kin.tip_position_and_forward(qp, clamp=False, soft=False)
                pm, fm = self.kin.tip_position_and_forward(qm, clamp=False, soft=False)
                dp = float(np.clip(float(np.dot(fp, self._preferred_dir)), -1.0, 1.0))
                dm = float(np.clip(float(np.dot(fm, self._preferred_dir)), -1.0, 1.0))
                J[:, i] = (pp - pm) / (2.0 * e)
                Jd[i] = -(dp - dm) / (2.0 * e)
                continue

            if q0[i] - e < self._lo[i] + 1e-12:
                qp = q0.copy()
                qp[i] = min(q0[i] + e, self._hi[i])
                pp, fp = self.kin.tip_position_and_forward(qp, clamp=True, soft=False)
                dp = float(np.clip(float(np.dot(fp, self._preferred_dir)), -1.0, 1.0))
                denom = float(qp[i] - q0[i])
                d = (denom if abs(denom) > 1e-12 else 1e-12)
                J[:, i] = (pp - p0) / d
                Jd[i] = -(dp - dir0) / d
            elif q0[i] + e > self._hi[i] - 1e-12:
                qm = q0.copy()
                qm[i] = max(q0[i] - e, self._lo[i])
                pm, fm = self.kin.tip_position_and_forward(qm, clamp=True, soft=False)
                dm = float(np.clip(float(np.dot(fm, self._preferred_dir)), -1.0, 1.0))
                denom = float(q0[i] - qm[i])
                d = (denom if abs(denom) > 1e-12 else 1e-12)
                J[:, i] = (p0 - pm) / d
                Jd[i] = -(dir0 - dm) / d
            else:
                qp = q0.copy()
                qm = q0.copy()
                qp[i] += e
                qm[i] -= e
                pp, fp = self.kin.tip_position_and_forward(qp, clamp=True, soft=False)
                pm, fm = self.kin.tip_position_and_forward(qm, clamp=True, soft=False)
                dp = float(np.clip(float(np.dot(fp, self._preferred_dir)), -1.0, 1.0))
                dm = float(np.clip(float(np.dot(fm, self._preferred_dir)), -1.0, 1.0))
                J[:, i] = (pp - pm) / (2.0 * e)
                Jd[i] = -(dp - dm) / (2.0 * e)
        reused_dir_jac = False
        if (self._dir_jac_cache is not None) and (self._dir_jac_cache_q is not None):
            dq_inf = float(np.max(np.abs(q0 - self._dir_jac_cache_q)))
            if (self._dir_jac_cache_age < self._dir_jac_reuse_steps) and (dq_inf <= self._dir_jac_reuse_tol):
                self._dir_jac_cache_age += 1
                Jd = self._dir_jac_cache.copy()
                reused_dir_jac = True
        if not reused_dir_jac:
            self._dir_jac_cache = Jd.copy()
            self._dir_jac_cache_q = q0.copy()
            self._dir_jac_cache_age = 0
        err_norm = float(np.linalg.norm(e_pos))
        return e_pos, J, e_dir, Jd, err_norm

    def _position_terms(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        e_pos, J_pos, _e_dir, _J_dir, err_norm = self._jacobian_fd_shared(q)
        obj = float(np.dot(e_pos, e_pos))
        return e_pos, J_pos, obj

    def _objective_only(self, q: np.ndarray) -> tuple[float, float]:
        """
        Evaluate position-only objective value without Jacobians.
        Returns (position_squared_error, position_error_norm).
        """
        p = self.kin.tip_position(q, clamp=True, soft=False)
        e_pos = (self.target - p).reshape(3)
        pos_sq = float(np.dot(e_pos, e_pos))
        pos_norm = float(math.sqrt(pos_sq))
        return pos_sq, pos_norm

    def _refine_once(self) -> bool:
        p = self.kin.tip_position(self.q, clamp=True, soft=False)
        err = self.target - p
        err0 = float(np.linalg.norm(err))
        if err0 <= 1e-12:
            return False

        e_pos, J_pos, obj0 = self._position_terms(self.q)
        dq = self._dls_step(J_pos, e_pos, max(float(self.cfg.damping_min), self.lmbd * 0.2))

        alpha = 1.0
        for _ in range(8):
            q_try = self.q + alpha * dq
            q_try[1] = self._wrap_roll(q_try[1])
            q_try = self.kin._clamp_all(q_try)
            obj_try, e_try = self._objective_only(q_try)
            if (obj_try < obj0) and (e_try <= err0 + max(float(self.cfg.tol), 1e-6)):
                self.q = q_try
                self.last_err_norm = e_try
                return True
            alpha *= 0.5
        return False

    def _secondary_refine_once(self) -> bool:
        if (not bool(self.cfg.prefer_tip_plus_x)) or (float(self.cfg.direction_weight) <= 0.0):
            return False
        e_pos, J_pos, e_dir, J_dir, err_norm = self._jacobian_fd_shared(self.q)
        tol = float(self.cfg.tol)
        if err_norm > tol:
            return False

        dirx0 = float(self.kin.tip_alignment(self.q, self._preferred_dir, clamp=True))
        posture0 = self.posture_cost(self.q)
        damping = max(float(self.cfg.damping_min), min(self.lmbd, 1e-2))
        J_pinv = self._damped_pinv(J_pos, damping)
        N = np.eye(4, dtype=float) - (J_pinv @ J_pos)

        w_dir = max(float(self.cfg.direction_weight), 1e-6)
        w_post = 0.05
        grad_dir = float(e_dir) * np.asarray(J_dir, dtype=float).reshape(4)
        grad_post = self.posture_gradient(self.q)
        dq = -(N @ (w_dir * grad_dir + w_post * grad_post))
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm <= 1e-12:
            return False

        if dq_norm > 0.25:
            dq *= 0.25 / dq_norm

        alpha = 1.0
        tol_margin = max(tol * 1.02, tol + 1e-7)
        for _ in range(8):
            q_try = self.q + alpha * dq
            q_try[1] = self._wrap_roll(q_try[1])
            q_try = self.kin._clamp_all(q_try)
            p_try = self.kin.tip_position(q_try, clamp=True, soft=False)
            err_try = float(np.linalg.norm(self.target - p_try))
            if err_try > tol_margin:
                alpha *= 0.5
                continue

            dirx_try = float(self.kin.tip_alignment(q_try, self._preferred_dir, clamp=True))
            posture_try = self.posture_cost(q_try)
            dir_better = dirx_try > (dirx0 + self._dir_improve_eps)
            dir_tied = abs(dirx_try - dirx0) <= self._dir_improve_eps
            posture_better = posture_try < (posture0 - self._posture_improve_eps)
            if dir_better or (dir_tied and posture_better):
                self.q = q_try
                self.last_err_norm = err_try
                return True
            alpha *= 0.5
        return False

    def step(self, n_iters: int = 1) -> np.ndarray:
        if self.converged or self.failed:
            return self.q

        for _ in range(max(1, int(n_iters))):
            p = self.kin.tip_position(self.q, clamp=True, soft=False)
            err = self.target - p
            err_norm = float(np.linalg.norm(err))
            self.last_err_norm = err_norm
            if err_norm <= float(self.cfg.tol):
                # First reduce residual, then improve +X preference inside the
                # position-feasible set while staying close to the actual start posture.
                refine_tol = max(float(self.cfg.tol) * 0.25, 1e-6)
                for _ in range(6):
                    if self.last_err_norm <= refine_tol:
                        break
                    if not self._refine_once():
                        break
                for _ in range(12):
                    if not self._secondary_refine_once():
                        break
                self.converged = True
                return self.q

            if err_norm < self._best_err * 0.999:
                self._best_err = err_norm
                self._stall = 0
            else:
                self._stall += 1
                if self._stall >= int(self.cfg.stall_limit):
                    self.failed = True
                    return self.q

            e_pos, J_pos, obj0 = self._position_terms(self.q)
            dq = self._dls_step(J_pos, e_pos, self.lmbd) * float(self.cfg.step_scale)

            alpha = float(self.cfg.step_scale)
            accepted = False
            q_best = self.q.copy()
            e_best = err_norm
            for _ls in range(max(1, int(self.cfg.line_search_steps))):
                q_try = self.q + alpha * dq
                q_try[1] = self._wrap_roll(q_try[1])
                q_try = self.kin._clamp_all(q_try)
                obj_try, e_try = self._objective_only(q_try)
                if (obj_try < obj0) and (e_try <= e_best + max(float(self.cfg.tol), 1e-6)):
                    accepted = True
                    q_best = q_try
                    e_best = e_try
                    break
                alpha *= float(self.cfg.line_search_shrink)

            if accepted:
                self.q = q_best
                self.lmbd = max(float(self.cfg.damping_min), self.lmbd * float(self.cfg.damping_down))
            else:
                self.lmbd = min(float(self.cfg.damping_max), self.lmbd * float(self.cfg.damping_up))

            self.iters += 1
            if self.iters >= int(self.cfg.max_iters):
                self.failed = True
                return self.q

        return self.q


class MultiSeedIk:
    def __init__(
        self,
        kin: Kinematics,
        target_xyz: np.ndarray,
        q_seed: np.ndarray,
        *,
        max_solvers: int = 8,
        n_random: int = 2,
        seed_rng: int | None = None,
        ik_cfg: Optional[IkConfig] = None,
        preferred_dir_world: Optional[np.ndarray] = None,
    ):
        self.kin = kin
        self.target = np.array(target_xyz, dtype=float).reshape(3)
        self.converged = False
        self.failed = False
        self.last_err_norm = float("inf")
        self._cfg = ik_cfg if ik_cfg is not None else IkConfig()
        if preferred_dir_world is None:
            self._preferred_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self._preferred_dir = np.asarray(preferred_dir_world, dtype=float).reshape(3).copy()
        pref_norm = float(np.linalg.norm(self._preferred_dir))
        if pref_norm <= 1e-12:
            self._preferred_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            self._preferred_dir /= pref_norm
        self._max_solvers = int(max_solvers)
        self._n_random = int(n_random)
        self._rng = np.random.default_rng(seed_rng)

        q0 = self.kin._clamp_all(np.array(q_seed, dtype=float).reshape(4))
        self._q_seed0 = q0.copy()
        seeds = self._make_seeds(q0, max_solvers=max_solvers, n_random=n_random)
        self._primary = _SingleIKSolver(
            self.kin,
            self.target,
            q0,
            self._cfg,
            q_path_ref=self._q_seed0,
            preferred_dir_world=self._preferred_dir,
        )
        self._fallback_seed_qs = [s.copy() for s in seeds[1:]]
        self._fallback_solvers: list[_SingleIKSolver] = []
        self._fallback_active = False
        self._rr_index = 0
        self.q = q0.copy()
        self._best_q = q0.copy()
        self._best_err = float("inf")
        self._prefer_plus_x = bool(self._cfg.prefer_tip_plus_x)
        self._dir_tol_deg = max(0.0, float(self._cfg.direction_tol_deg))
        self._dir_tol_cos = float(math.cos(math.radians(self._dir_tol_deg)))
        self._tie_pos_eps_m = float(self._cfg.orientation_tie_eps_m)
        self._best_conv_q: Optional[np.ndarray] = None
        self._best_conv_err: float = float("inf")
        self._best_conv_dirx: float = -float("inf")
        self._best_conv_posture: float = float("inf")
        self._best_conv_dir_ok: bool = False
        self._dir_tie_eps: float = 1e-2
        self._update_best_initial()

    def _linear_guess(self, target_x: float) -> float:
        reach = float((self.kin.n_nodes - 1) * self.kin.pitch + self.kin.pitch)
        bx = float(target_x - self.kin.origin_xyz[0] - reach)
        return float(np.clip(bx, self.kin.linear_min, self.kin.linear_max))

    def _dedup(self, seeds: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        seen = set()
        for s in seeds:
            s = self.kin._clamp_all(s)
            key = tuple(int(round(v * 10000.0)) for v in s)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    def _make_seeds(self, q0: np.ndarray, *, max_solvers: int, n_random: int) -> list[np.ndarray]:
        bx0, r0, _t10, _t20 = map(float, q0)
        t_lim = float(self.kin.bend_lim)
        t_big = 0.75 * t_lim
        t_med = 0.45 * t_lim
        bxg = self._linear_guess(float(self.target[0]))
        bx_candidates = [bx0, bxg, bxg - 0.05, bxg + 0.05, bxg - 0.02, bxg + 0.02]
        bx_candidates = [float(np.clip(b, self.kin.linear_min, self.kin.linear_max)) for b in bx_candidates]
        roll_candidates = [r0, 0.0, 0.5 * math.pi, -0.5 * math.pi]
        patterns = [
            (0.0, 0.0),
            (+t_big, -t_big),
            (-t_big, +t_big),
            (+t_med, -t_med),
            (-t_med, +t_med),
            (+t_big, +t_big),
            (-t_big, -t_big),
        ]

        seeds: list[np.ndarray] = [q0.copy()]
        for bx in bx_candidates[:4]:
            seeds.append(np.array([bx, r0, 0.0, 0.0], dtype=float))
            seeds.append(np.array([bx, 0.0, 0.0, 0.0], dtype=float))
        for k1, k2 in patterns[1:]:
            seeds.append(np.array([bxg, r0, k1, k2], dtype=float))
        for k1, k2 in patterns[1:5]:
            for rr in roll_candidates[:3]:
                seeds.append(np.array([bxg, rr, k1, k2], dtype=float))
        for i in range(max(0, int(n_random))):
            bx = float(np.clip(bxg + self._rng.normal(0.0, 0.06), self.kin.linear_min, self.kin.linear_max))
            roll = float(self._rng.uniform(-math.pi, math.pi))
            if i < n_random // 2:
                k1 = float(self._rng.uniform(-t_lim, +t_lim))
                k2 = float(np.clip(-k1 + self._rng.normal(0.0, 0.15 * t_lim), -t_lim, +t_lim))
            else:
                k1 = float(self._rng.uniform(-t_lim, +t_lim))
                k2 = float(self._rng.uniform(-t_lim, +t_lim))
            seeds.append(np.array([bx, roll, k1, k2], dtype=float))

        seeds = self._dedup(seeds)
        if len(seeds) > int(max_solvers):
            seeds = seeds[: int(max_solvers)]
        return seeds if seeds else [q0.copy()]

    def _update_best_initial(self) -> None:
        p = self.kin.tip_position(self._primary.q, clamp=True, soft=False)
        best_e = float(np.linalg.norm(self.target - p))
        self._primary.last_err_norm = best_e
        self.q = self._primary.q.copy()
        self.last_err_norm = best_e
        self._best_q = self.q.copy()
        self._best_err = float(best_e)

    def _activate_fallback(self) -> None:
        if self._fallback_active:
            return
        self._fallback_active = True
        self._fallback_solvers = [
            _SingleIKSolver(
                self.kin,
                self.target,
                s,
                self._cfg,
                q_path_ref=self._q_seed0,
                preferred_dir_world=self._preferred_dir,
            )
            for s in self._fallback_seed_qs
        ]
        self._rr_index = 0

    def _posture_cost_from_start(self, q: np.ndarray) -> float:
        return float(self._primary.posture_cost(q))

    def _direction_ok(self, dir_score: float) -> bool:
        if not self._prefer_plus_x:
            return True
        return bool(float(dir_score) >= self._dir_tol_cos)

    def _is_better_candidate(self, *, dir_ok: bool, dir_score: float, posture: float, err: float) -> bool:
        if self._best_conv_q is None:
            return True
        if not self._prefer_plus_x:
            if abs(err - self._best_conv_err) <= float(self._tie_pos_eps_m):
                if posture < self._best_conv_posture:
                    return True
                if abs(posture - self._best_conv_posture) <= 1e-12 and dir_score > (self._best_conv_dirx + self._dir_tie_eps):
                    return True
                return False
            return bool(err < self._best_conv_err)

        if dir_ok != self._best_conv_dir_ok:
            return bool(dir_ok and (not self._best_conv_dir_ok))
        if dir_score > (self._best_conv_dirx + self._dir_tie_eps):
            return True
        if abs(dir_score - self._best_conv_dirx) <= self._dir_tie_eps:
            if posture < self._best_conv_posture:
                return True
            if abs(posture - self._best_conv_posture) <= 1e-12 and err < self._best_conv_err:
                return True
        return False

    def _update_global_best(self, q: np.ndarray, err: float) -> None:
        if float(err) < float(self._best_err):
            self._best_err = float(err)
            self._best_q = np.asarray(q, dtype=float).reshape(4).copy()

    def _consider_converged(self, q: np.ndarray, err: float) -> None:
        q = np.asarray(q, dtype=float).reshape(4)
        err = float(err)
        dirx = float(self.kin.tip_alignment(q, self._preferred_dir, clamp=True))
        posture = self._posture_cost_from_start(q)
        dir_ok = self._direction_ok(dirx)
        if self._is_better_candidate(dir_ok=dir_ok, dir_score=dirx, posture=posture, err=err):
            self._best_conv_q = q.copy()
            self._best_conv_err = err
            self._best_conv_dirx = dirx
            self._best_conv_posture = posture
            self._best_conv_dir_ok = dir_ok

    def _least_squares_refine(self) -> bool:
        if least_squares is None:
            return False

        lo = np.array(
            [self.kin.linear_min, self.kin.roll_min, -self.kin.bend_lim, -self.kin.bend_lim],
            dtype=float,
        )
        hi = np.array(
            [self.kin.linear_max, self.kin.roll_max, +self.kin.bend_lim, +self.kin.bend_lim],
            dtype=float,
        )

        x0_list = [self._best_q.copy(), self.q.copy(), self._primary.q.copy()]
        for s in self._fallback_solvers:
            if s.last_err_norm < max(self._best_err * 1.5, self._cfg.tol * 10.0):
                x0_list.append(s.q.copy())
            if len(x0_list) >= 4:
                break

        best_q = self._best_q.copy()
        best_err = float(self._best_err)
        best_dirx = float(self.kin.tip_alignment(best_q, self._preferred_dir, clamp=True))
        best_posture = self._posture_cost_from_start(best_q)
        best_dir_ok = self._direction_ok(best_dirx)

        def residual(x: np.ndarray) -> np.ndarray:
            q = self.kin._clamp_all(np.asarray(x, dtype=float).reshape(4))
            return self.kin.tip_position(q, clamp=True, soft=False) - self.target

        for x0 in x0_list:
            try:
                res = least_squares(
                    residual,
                    np.asarray(x0, dtype=float).reshape(4),
                    bounds=(lo, hi),
                    method="trf",
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                    max_nfev=300,
                )
            except Exception:
                continue

            q_try = self.kin._clamp_all(np.asarray(res.x, dtype=float).reshape(4))
            err_try = float(np.linalg.norm(self.kin.tip_position(q_try, clamp=True, soft=False) - self.target))
            dirx_try = float(self.kin.tip_alignment(q_try, self._preferred_dir, clamp=True))
            posture_try = self._posture_cost_from_start(q_try)
            dir_ok_try = self._direction_ok(dirx_try)
            better = False
            if not self._prefer_plus_x:
                if abs(err_try - best_err) <= float(self._tie_pos_eps_m):
                    if posture_try < best_posture:
                        better = True
                    elif abs(posture_try - best_posture) <= 1e-12 and dirx_try > (best_dirx + self._dir_tie_eps):
                        better = True
                elif err_try < best_err:
                    better = True
            else:
                if dir_ok_try != best_dir_ok:
                    better = bool(dir_ok_try and (not best_dir_ok))
                elif dirx_try > (best_dirx + self._dir_tie_eps):
                    better = True
                elif abs(dirx_try - best_dirx) <= self._dir_tie_eps and posture_try < best_posture:
                    better = True
                elif abs(dirx_try - best_dirx) <= self._dir_tie_eps and abs(posture_try - best_posture) <= 1e-12 and err_try < best_err:
                    better = True
            if better:
                best_err = err_try
                best_q = q_try
                best_dirx = dirx_try
                best_posture = posture_try
                best_dir_ok = dir_ok_try

        self.q = best_q.copy()
        self.last_err_norm = float(best_err)
        self._update_global_best(best_q, best_err)
        if best_err > float(self._cfg.tol):
            return False
        if self._prefer_plus_x and (not self._direction_ok(best_dirx)):
            return False
        return True

    def step(self, n_iters: int = 10) -> np.ndarray:
        if self.converged or self.failed:
            return self.q

        budget = max(1, int(n_iters))
        for _ in range(budget):
            if not self._fallback_active:
                self._primary.step(1)
                self.q = self._primary.q.copy()
                self.last_err_norm = float(self._primary.last_err_norm)
                self._update_global_best(self._primary.q, self._primary.last_err_norm)
                if self._primary.converged:
                    primary_dir = float(self.kin.tip_alignment(self._primary.q, self._preferred_dir, clamp=True))
                    if self._direction_ok(primary_dir):
                        self.converged = True
                        self.failed = False
                        return self.q
                    self._consider_converged(self._primary.q, float(self._primary.last_err_norm))
                    self._activate_fallback()
                    if not self._fallback_solvers:
                        self.converged = True
                        self.failed = False
                        return self.q
                if self._primary.failed:
                    self._activate_fallback()
                    if not self._fallback_solvers:
                        self.failed = True
                        self.q = self._best_q.copy()
                        self.last_err_norm = float(self._best_err)
                        return self.q
                continue

            n = len(self._fallback_solvers)
            solver = None
            for _t in range(n):
                s = self._fallback_solvers[self._rr_index % n]
                self._rr_index = (self._rr_index + 1) % n
                if not (s.converged or s.failed):
                    solver = s
                    break
            if solver is None:
                break

            solver.step(1)
            self._update_global_best(solver.q, solver.last_err_norm)
            if solver.converged:
                self._consider_converged(solver.q, float(solver.last_err_norm))

        if not self._fallback_active:
            self.q = self._primary.q.copy()
            self.last_err_norm = float(self._primary.last_err_norm)
            return self.q

        if self._best_conv_q is not None:
            self.q = self._best_conv_q.copy()
            self.last_err_norm = float(self._best_conv_err)
        else:
            self.q = self._best_q.copy()
            self.last_err_norm = float(self._best_err)

        all_done = all((s.converged or s.failed) for s in self._fallback_solvers)
        if all_done:
            if self._best_conv_q is not None and ((not self._prefer_plus_x) or self._best_conv_dir_ok):
                self.converged = True
                self.failed = False
                return self.q
            self.failed = True
        return self.q
