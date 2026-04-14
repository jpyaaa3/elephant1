#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from scipy.spatial.transform import Rotation as Rot

try:
    import zmq  # type: ignore
except ImportError:
    zmq = None  # type: ignore

from engine.protocol import (
    ControlU,
    SimMappingConfig,
    SimQ,
    control_u_to_gensim_q,
    dumps_msg,
    gensim_q_to_control_u,
    pack_ui_state,
)
import engine.iklib as ik
from engine.config_loader import IkConfig, load_app_config_from_ini
import builder.json_builder as assembly_builder


@dataclass(frozen=True)
class LinkState:
    connected: bool
    tx_seq: int
    rx_age_s: float
    device: str
    ports: tuple[str, ...]
    torque_enabled: bool
    reply_ok: bool
    reply_reason: str
    q: Optional[SimQ]
    u: Optional[ControlU]


class Link:
    """Controller-side bridge client."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        send_hz: float = 30.0,
        cfg: Optional[SimMappingConfig] = None,
    ) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required for Link")
        self.endpoint = str(endpoint)
        self.cfg = cfg or SimMappingConfig()
        self.send_hz = float(send_hz)
        self._send_period = (1.0 / self.send_hz) if self.send_hz > 0 else 0.0

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.linger = 0
        self.sock.setsockopt(zmq.IDENTITY, f"gensim-{os.getpid()}-{int(time.time()*1000)}".encode("utf-8"))
        self.sock.connect(self.endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.is_connected = True
        self.tx_seq = 0
        self._t_last_tx = 0.0

        self.last_ack_ts = 0.0
        self.last_state_ts = 0.0
        self._t_last_rx_wall = 0.0
        self.last_q: SimQ | None = None
        self.last_u: ControlU | None = None
        self.last_ports: list[str] = []
        self.last_device: str = ""
        self.torque_enabled: bool = False
        self.last_reply_ok: bool = True
        self.last_reply_reason: str = ""

        self._send({"t": "hello", "ts": time.time()})

    def close(self) -> None:
        try:
            self.poller.unregister(self.sock)
        except KeyError:
            pass
        except AttributeError:
            pass
        try:
            self.sock.close(0)
        except AttributeError:
            pass
        self.is_connected = False

    def rx_age_s(self) -> float:
        if self._t_last_rx_wall <= 0.0:
            return float("inf")
        return float(time.time() - self._t_last_rx_wall)

    def get_state(self) -> LinkState:
        return LinkState(
            connected=bool(self.is_connected),
            tx_seq=int(self.tx_seq),
            rx_age_s=float(self.rx_age_s()),
            device=str(self.last_device),
            ports=tuple(str(x) for x in self.last_ports),
            torque_enabled=bool(self.torque_enabled),
            reply_ok=bool(self.last_reply_ok),
            reply_reason=str(self.last_reply_reason),
            q=self.last_q,
            u=self.last_u,
        )

    def _send(self, msg: dict) -> None:
        try:
            self.sock.send_json(msg, flags=zmq.NOBLOCK)
        except zmq.ZMQError as exc:
            self.is_connected = False
            self.last_reply_ok = False
            self.last_reply_reason = f"transport send failed: {exc}"

    def poll(self) -> None:
        try:
            events = dict(self.poller.poll(timeout=0))
        except zmq.ZMQError as exc:
            self.is_connected = False
            self.last_reply_ok = False
            self.last_reply_reason = f"transport poll failed: {exc}"
            return
        if self.sock not in events:
            return
        try:
            msg = self.sock.recv_json(flags=zmq.NOBLOCK)
        except ValueError as exc:
            self.last_reply_ok = False
            self.last_reply_reason = f"transport recv decode failed: {exc}"
            return
        except zmq.ZMQError as exc:
            self.is_connected = False
            self.last_reply_ok = False
            self.last_reply_reason = f"transport recv failed: {exc}"
            return

        self._t_last_rx_wall = time.time()
        t = str(msg.get("t", "")).lower()
        if t == "ack":
            self.last_ack_ts = float(msg.get("ts", time.time()))
            self.last_reply_ok = bool(msg.get("ok", True))
            self.last_reply_reason = str(msg.get("reason", ""))
            if "ports" in msg and isinstance(msg.get("ports"), list):
                self.last_ports = [str(v) for v in msg.get("ports", [])]
            if "device" in msg:
                new_device = str(msg.get("device", ""))
                if new_device != self.last_device:
                    self.last_q = None
                    self.last_u = None
                    self.last_state_ts = 0.0
                self.last_device = new_device
            if "torque_enabled" in msg:
                self.torque_enabled = bool(msg.get("torque_enabled", False))
            self.is_connected = True
            return

        if t == "state":
            self.last_state_ts = float(msg.get("ts", time.time()))
            if "q" in msg:
                try:
                    self.last_q = SimQ(**msg["q"])
                except (TypeError, ValueError) as exc:
                    self.last_reply_ok = False
                    self.last_reply_reason = f"state q decode failed: {exc}"
            if "u" in msg:
                try:
                    self.last_u = ControlU(**msg["u"])
                except (TypeError, ValueError) as exc:
                    self.last_reply_ok = False
                    self.last_reply_reason = f"state u decode failed: {exc}"
            if "torque_enabled" in msg:
                self.torque_enabled = bool(msg.get("torque_enabled", False))
            self.is_connected = True
            if self.last_reply_reason == "":
                self.last_reply_ok = True

    def refresh_state(self) -> LinkState:
        self.poll()
        return self.get_state()

    def estop(self) -> None:
        self._send({"t": "estop", "ts": time.time()})

    def torque_on(self) -> None:
        self._send({"t": "torque_on", "ts": time.time()})

    def torque_off(self) -> None:
        self._send({"t": "torque_off", "ts": time.time()})

    def request_ports(self) -> None:
        self._send({"t": "ports", "ts": time.time()})

    def set_device(self, device: str) -> None:
        self._send({"t": "set_device", "ts": time.time(), "device": str(device)})

    def disconnect_device(self) -> None:
        self._send({"t": "disconnect_device", "ts": time.time()})

    def maybe_send_target_q(self, q: SimQ, *, source: str = "sim") -> None:
        now = time.time()
        if self._send_period > 0 and (now - self._t_last_tx) < self._send_period:
            return
        self._t_last_tx = now
        self.tx_seq += 1
        msg = {
            "t": "target",
            "ts": now,
            "seq": self.tx_seq,
            "source": str(source),
            "q": {
                "linear_m": float(q.linear_m),
                "roll_rad": float(q.roll_rad),
                "theta1_rad": float(q.theta1_rad),
                "theta2_rad": float(q.theta2_rad),
            },
        }
        self._send(msg)

    def send_target_q(self, q: SimQ, *, source: str = "ui") -> None:
        self.maybe_send_target_q(q, source=source)

    def send_target_values(
        self,
        *,
        linear_m: float,
        roll_rad: float,
        theta1_rad: float,
        theta2_rad: float,
        source: str = "ui",
    ) -> None:
        self.send_target_q(
            SimQ(
                linear_m=float(linear_m),
                roll_rad=float(roll_rad),
                theta1_rad=float(theta1_rad),
                theta2_rad=float(theta2_rad),
            ),
            source=source,
        )

    def q_to_control_u(
        self,
        *,
        linear_m: float,
        roll_rad: float,
        theta1_rad: float,
        theta2_rad: float,
    ) -> ControlU:
        return gensim_q_to_control_u(
            SimQ(
                linear_m=float(linear_m),
                roll_rad=float(roll_rad),
                theta1_rad=float(theta1_rad),
                theta2_rad=float(theta2_rad),
            ),
            self.cfg,
        )

    def control_u_to_q(
        self,
        *,
        u_linear: float,
        u_roll: float,
        u_s1: float,
        u_s2: float,
    ) -> SimQ:
        return control_u_to_gensim_q(
            ControlU(
                u_linear=float(u_linear),
                u_roll=float(u_roll),
                u_s1=float(u_s1),
                u_s2=float(u_s2),
            ),
            self.cfg,
        )


class SimPublisher:
    def __init__(self, endpoint: str) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required for SimPublisher")
        self.endpoint = str(endpoint)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.linger = 0
        self.sock.connect(self.endpoint)

    def close(self) -> None:
        self.sock.close(0)

    def publish_ui_state(
        self,
        *,
        linear: float,
        roll: float,
        theta1: float,
        theta2: float,
        paused: bool,
        target_xyz: tuple[float, float, float],
    ) -> None:
        msg = pack_ui_state(
            q=SimQ(
                linear_m=float(linear),
                roll_rad=float(roll),
                theta1_rad=float(theta1),
                theta2_rad=float(theta2),
            ),
            paused=bool(paused),
            target_xyz=(float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])),
        )
        self.sock.send(dumps_msg(msg), flags=zmq.NOBLOCK)


@dataclass
class ControlState:
    linear: float = 0.0
    roll: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    paused: bool = False

    # IK target (world coordinates)
    target_x: float = 0.50
    target_y: float = 0.00
    target_z: float = 1.00

    # IK runtime status (read-only for UI thread)
    ik_running: bool = False
    ik_converged: bool = False
    ik_failed: bool = False
    ik_err_m: float = 0.0
    # Debug: physics tracking + sim tip error (to diagnose 'fake converge')
    ik_sim_tip_err_m: float = 0.0
    ik_track_linear_err_m: float = 0.0
    ik_track_roll_err_rad: float = 0.0
    ik_track_theta1_err_rad: float = 0.0
    ik_track_theta2_err_rad: float = 0.0
    ik_track_bend_max_err_rad: float = 0.0

    # IK best/current solution (for UI display)
    ik_sol_linear: float = 0.0
    ik_sol_roll: float = 0.0
    ik_sol_theta1: float = 0.0
    ik_sol_theta2: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def snapshot(self) -> Tuple[float, float, float, float, bool, Tuple[float, float, float]]:
        with self._lock:
            return (
                self.linear,
                self.roll,
                self.theta1,
                self.theta2,
                self.paused,
                (self.target_x, self.target_y, self.target_z),
            )

    def set_all(self, linear: float, roll: float, theta1: float, theta2: float, paused: bool) -> None:
        with self._lock:
            self.linear = float(linear)
            self.roll = float(roll)
            self.theta1 = float(theta1)
            self.theta2 = float(theta2)
            self.paused = bool(paused)

    def set_ik_status(self, running: bool, converged: bool, failed: bool, err_m: float) -> None:
        with self._lock:
            self.ik_running = bool(running)
            self.ik_converged = bool(converged)
            self.ik_failed = bool(failed)
            self.ik_err_m = float(err_m)

    def set_ik_solution(self, linear: float, roll: float, theta1: float, theta2: float) -> None:
        with self._lock:
            self.ik_sol_linear = float(linear)
            self.ik_sol_roll = float(roll)
            self.ik_sol_theta1 = float(theta1)
            self.ik_sol_theta2 = float(theta2)

    def set_ik_debug(
        self,
        *,
        sim_tip_err_m: float,
        linear_err_m: float,
        roll_err_rad: float,
        theta1_err_rad: float,
        theta2_err_rad: float,
        bend_max_err_rad: float,
    ) -> None:
        """UI display helpers: sim-thread writes debug error metrics for 'fake converge' diagnosis."""
        with self._lock:
            self.ik_sim_tip_err_m = float(sim_tip_err_m)
            self.ik_track_linear_err_m = float(linear_err_m)
            self.ik_track_roll_err_rad = float(roll_err_rad)
            self.ik_track_theta1_err_rad = float(theta1_err_rad)
            self.ik_track_theta2_err_rad = float(theta2_err_rad)
            self.ik_track_bend_max_err_rad = float(bend_max_err_rad)

class ImGuiController:
    """External ImGui window to edit ControlState.

    UI/UX is intentionally kept the same; this class only encapsulates drawing logic
    and fixes the IK tab crash (pyimgui input_float3 return shape).
    """

    def __init__(
        self,
        params: Any,
        robot: Any,
        state: ControlState,
        link: Optional[Link] = None,
        sim_pub: Optional[SimPublisher] = None,
        mapping_cfg: Optional[SimMappingConfig] = None,
        *,
        use_hardware: bool = False,
        ik_cfg: Optional[IkConfig] = None,
        ik_context: Optional[dict[str, Any]] = None,
    ):
        self.p = params
        self.robot = robot
        self.state = state
        self.link = link
        self.sim_pub = sim_pub
        self._mapping_cfg = mapping_cfg or SimMappingConfig()
        self._use_hardware = bool(use_hardware)
        self._ik_cfg = ik_cfg or IkConfig()
        self._ik_context = dict(ik_context or {})
        self._stop = False
        self._ik_header_init_open = False
        self._ctrl_window_init = False
        self._port_input = ""
        self._ik_worker: Optional[threading.Thread] = None
        self._link_state: Optional[LinkState] = None

    def _send_current_target(self, *, source: str) -> None:
        if self.sim_pub is not None:
            self.sim_pub.publish_ui_state(
                linear=float(self.state.linear),
                roll=float(self.state.roll),
                theta1=float(self.state.theta1),
                theta2=float(self.state.theta2),
                paused=bool(self.state.paused),
                target_xyz=(float(self.state.target_x), float(self.state.target_y), float(self.state.target_z)),
            )
        if self.link is not None and (not self.state.paused):
            self.link.send_target_values(
                linear_m=float(self.state.linear),
                roll_rad=float(self.state.roll),
                theta1_rad=float(self.state.theta1),
                theta2_rad=float(self.state.theta2),
                source=source,
            )

    def _start_ik_solve(self) -> None:
        if self.state.ik_running:
            return
        ctx = dict(self._ik_context)
        required = (
            "pitch",
            "n_nodes",
            "n_seg",
            "origin_xyz",
            "limit",
            "base_axis_sign",
            "roll_axis_sign",
            "bend_axis_sign",
        )
        if any(k not in ctx for k in required):
            print("[UI] IK solve rejected | missing ik_context fields")
            self.state.set_ik_status(running=False, converged=False, failed=True, err_m=float("inf"))
            return

        target = np.array([self.state.target_x, self.state.target_y, self.state.target_z], dtype=float)
        self.state.set_ik_status(running=True, converged=False, failed=False, err_m=float("inf"))

        def _worker() -> None:
            try:
                result = ik.solve(
                    target_world=target,
                    pitch=float(ctx["pitch"]),
                    n_nodes=int(ctx["n_nodes"]),
                    n_seg=int(ctx["n_seg"]),
                    origin_xyz=np.asarray(ctx["origin_xyz"], dtype=float).reshape(3),
                    limit=ctx["limit"],
                    position_tol_m=float(self._ik_cfg.tol),
                    max_iters=max(int(self._ik_cfg.max_iters), 1),
                    base_axis_sign=float(ctx["base_axis_sign"]),
                    roll_axis_sign=float(ctx["roll_axis_sign"]),
                    bend_axis_sign=float(ctx["bend_axis_sign"]),
                )
                if result.success and result.q is not None:
                    q = np.asarray(result.q, dtype=float).reshape(4)
                    self.state.linear = float(q[0])
                    self.state.roll = float(q[1])
                    self.state.theta1 = float(q[2])
                    self.state.theta2 = float(q[3])
                    self.state.set_ik_solution(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                    self.state.set_ik_status(
                        running=False,
                        converged=True,
                        failed=False,
                        err_m=float(result.position_error_m),
                    )
                    self._send_current_target(source="ik")
                else:
                    print(
                        "[UI] IK solve failed | target=(%.3f, %.3f, %.3f) | err=%s"
                        % (float(target[0]), float(target[1]), float(target[2]), float(result.position_error_m))
                    )
                    self.state.set_ik_status(
                        running=False,
                        converged=False,
                        failed=True,
                        err_m=float(result.position_error_m),
                    )
            finally:
                self._ik_worker = None

        self._ik_worker = threading.Thread(target=_worker, daemon=True)
        self._ik_worker.start()

    def stop(self) -> None:
        self._stop = True

    def _begin_disabled_ui(self, disabled: bool) -> str | None:
        if not disabled:
            return None
        begin_disabled = getattr(imgui, "begin_disabled", None)
        if callable(begin_disabled):
            begin_disabled()
            return "begin_disabled"
        item_disabled = getattr(imgui, "ITEM_DISABLED", None)
        push_item_flag = getattr(imgui, "push_item_flag", None)
        push_style_var = getattr(imgui, "push_style_var", None)
        style_alpha = getattr(imgui, "STYLE_ALPHA", None)
        if item_disabled is not None and callable(push_item_flag):
            push_item_flag(item_disabled, True)
            if style_alpha is not None and callable(push_style_var):
                push_style_var(style_alpha, imgui.get_style().alpha * 0.5)
                return "push_item_flag+alpha"
            return "push_item_flag"
        return None

    def _end_disabled_ui(self, token: str | None) -> None:
        if token is None:
            return
        if token == "begin_disabled":
            end_disabled = getattr(imgui, "end_disabled", None)
            if callable(end_disabled):
                end_disabled()
            return
        if token == "push_item_flag+alpha":
            pop_style_var = getattr(imgui, "pop_style_var", None)
            if callable(pop_style_var):
                pop_style_var()
        pop_item_flag = getattr(imgui, "pop_item_flag", None)
        if callable(pop_item_flag):
            pop_item_flag()

    def _draw_ik_panel(self) -> None:
        # --- IK Target (world xyz) ---
        if not self._ik_header_init_open:
            # 珥덇린?먮쭔 ??踰??닿린
            cond = getattr(imgui, "ONCE", getattr(imgui, "FIRST_USE_EVER", 1))
            imgui.set_next_item_open(True, cond)
            self._ik_header_init_open = True

        if imgui.collapsing_header("IK Target (world xyz)", visible=True)[0]:
            _ret = imgui.input_float3(
                "target [m]",
                self.state.target_x, self.state.target_y, self.state.target_z,
                format="%.4f",
            )

            # pyimgui returns (changed, (x, y, z)) for input_float3
            if isinstance(_ret, tuple) and len(_ret) == 2:
                changed, (x, y, z) = _ret
            else:
                # Fallback for unexpected bindings
                changed, x, y, z = _ret

            if changed:
                self.state.target_x = float(x)
                self.state.target_y = float(y)
                self.state.target_z = float(z)

            if imgui.button("Solve IK"):
                self._start_ik_solve()
            imgui.same_line()
            if imgui.button("Stop IK"):
                self.state.set_ik_status(running=False, converged=False, failed=False, err_m=0.0)

            # Status line
            status = "idle"
            if self.state.ik_running:
                status = "running"
            if self.state.ik_converged:
                status = "converged"
            if self.state.ik_failed:
                status = "failed"
            imgui.text(f"IK status: {status} | err: {self.state.ik_err_m*1000:.2f} mm")
            imgui.text(
                f"IK q*: linear={self.state.ik_sol_linear:.4f} m | "
                f"roll={math.degrees(self.state.ik_sol_roll):.1f} deg | "
                f"theta1={math.degrees(self.state.ik_sol_theta1):.1f} deg | "
                f"theta2={math.degrees(self.state.ik_sol_theta2):.1f} deg"
            )

            imgui.text(
                f"sliders q: linear={self.state.linear:.4f} m | "
                f"roll={math.degrees(self.state.roll):.1f} deg | "
                f"theta1={math.degrees(self.state.theta1):.1f} deg | "
                f"theta2={math.degrees(self.state.theta2):.1f} deg"
            )


    def _draw_controls_window(self) -> None:
        if not self._ctrl_window_init:
            cond = getattr(imgui, "ONCE", getattr(imgui, "FIRST_USE_EVER", 1))
            imgui.set_next_window_size(880, 520, cond)
            self._ctrl_window_init = True
        imgui.begin("4-DOF Controls", True)

        link_state = self._link_state if self._link_state is not None else None
        sliders_locked = bool(self._use_hardware and (self.link is None or link_state is None or not bool(link_state.torque_enabled)))
        disable_token = self._begin_disabled_ui(sliders_locked)

        # --- Control-unit sliders (0..360 position mode) ---
        if self.link is not None:
            u_now = self.link.q_to_control_u(
                linear_m=float(self.state.linear),
                roll_rad=float(self.state.roll),
                theta1_rad=float(self.state.theta1),
                theta2_rad=float(self.state.theta2),
            )
            cfg = self.link.cfg
        else:
            u_now = gensim_q_to_control_u(
                SimQ(
                    linear_m=float(self.state.linear),
                    roll_rad=float(self.state.roll),
                    theta1_rad=float(self.state.theta1),
                    theta2_rad=float(self.state.theta2),
                ),
                self._mapping_cfg,
            )
            cfg = self._mapping_cfg

        changed_lin, u_linear = imgui.slider_float(
            "linear [u]", float(u_now.u_linear),
            float(cfg.linear_u_min), float(cfg.linear_u_max),
            format="%.1f"
        )
        changed_rdeg, u_roll = imgui.slider_float(
            "roll [u]", float(u_now.u_roll),
            float(cfg.roll_u_min), float(cfg.roll_u_max),
            format="%.1f"
        )
        changed_s1, u_s1 = imgui.slider_float(
            "seg1 [u]", float(u_now.u_s1),
            float(cfg.seg_u_min), float(cfg.seg_u_max),
            format="%.1f"
        )
        changed_s2, u_s2 = imgui.slider_float(
            "seg2 [u]", float(u_now.u_s2),
            float(cfg.seg_u_min), float(cfg.seg_u_max),
            format="%.1f"
        )
        self._end_disabled_ui(disable_token)

        changed_any = bool((not sliders_locked) and (changed_lin or changed_rdeg or changed_s1 or changed_s2))

        # If user touches sliders while IK is running, treat it as a manual override.
        if self.state.ik_running and changed_any:
            self.state.set_ik_status(running=False, converged=False, failed=False, err_m=0.0)

        if changed_any:
            if self.link is not None:
                q_new = self.link.control_u_to_q(
                    u_linear=float(u_linear),
                    u_roll=float(u_roll),
                    u_s1=float(u_s1),
                    u_s2=float(u_s2),
                )
            else:
                q_new = control_u_to_gensim_q(
                    ControlU(
                        u_linear=float(u_linear),
                        u_roll=float(u_roll),
                        u_s1=float(u_s1),
                        u_s2=float(u_s2),
                    ),
                    self._mapping_cfg,
                )
            self.state.linear = float(q_new.linear_m)
            self.state.roll = float(q_new.roll_rad)
            self.state.theta1 = float(q_new.theta1_rad)
            self.state.theta2 = float(q_new.theta2_rad)
            self._send_current_target(source="slider")
        if sliders_locked:
            imgui.text("Sliders locked until Torque On")

        if self._use_hardware and self.link is not None:
            imgui.separator()
            imgui.text("Hardware")
            state = link_state if link_state is not None else self.link.get_state()
            imgui.text(f"Link: {'OK' if state.connected else 'OFF'}")
            imgui.text(f"tx_seq={state.tx_seq} rx_age={state.rx_age_s:.2f}s")
            current_device = str(state.device or "").strip()
            if current_device:
                imgui.text(f"Current Port: {current_device}")
                if not self._port_input:
                    self._port_input = current_device
            changed_port, new_port = imgui.input_text("Port", self._port_input, 256)
            if changed_port:
                self._port_input = str(new_port)
            if imgui.button("Search Ports"):
                self.link.request_ports()
            imgui.same_line()
            if imgui.button("Apply Port"):
                self.link.set_device(self._port_input.strip())
            imgui.same_line()
            if imgui.button("Disconnect Port"):
                self.link.disconnect_device()
                self._port_input = ""
            ports = list(state.ports)
            if ports:
                imgui.text("Detected Ports:")
                imgui.same_line()
                for idx, port in enumerate(ports):
                    if imgui.small_button(f"{port}##port_{idx}"):
                        self._port_input = str(port)
                    if (idx + 1) < len(ports):
                        imgui.same_line()
            reply_reason = str(state.reply_reason or "").strip()
            if reply_reason:
                if bool(state.reply_ok):
                    if reply_reason == "ports":
                        if not ports:
                            imgui.text("No serial ports found")
                    else:
                        imgui.text(f"Bridge: {reply_reason}")
                else:
                    imgui.text_colored(f"Bridge: {reply_reason}", 1.0, 0.35, 0.35)
            if imgui.button("Torque On"):
                self.link.torque_on()
            imgui.same_line()
            if imgui.button("Torque Off"):
                self.link.torque_off()

        _, self.state.paused = imgui.checkbox("pause control updates", self.state.paused)
        imgui.same_line()
        if imgui.button("Reset (0,0,0,0)"):
            self.state.set_ik_status(running=False, converged=False, failed=False, err_m=0.0)
            self.state.linear = 0.0
            self.state.roll = 0.0
            self.state.theta1 = 0.0
            self.state.theta2 = 0.0
            self._send_current_target(source="slider")

        imgui.separator()
        self._draw_ik_panel()
        imgui.end()
    def run(self) -> None:
        if not glfw.init():
            raise SystemExit("glfw.init() failed.")

        glfw.window_hint(glfw.RESIZABLE, True)
        window = glfw.create_window(640, 320, "Snake Controls (ImGui)", None, None)
        if not window:
            glfw.terminate()
            raise SystemExit("Failed to create GLFW window.")

        glfw.make_context_current(window)

        imgui.create_context()
        impl = GlfwRenderer(window)


        try:
            while not glfw.window_should_close(window) and not self._stop:
                if self.link is not None:
                    self._link_state = self.link.refresh_state()
                if self.sim_pub is not None:
                    self.sim_pub.publish_ui_state(
                        linear=float(self.state.linear),
                        roll=float(self.state.roll),
                        theta1=float(self.state.theta1),
                        theta2=float(self.state.theta2),
                        paused=bool(self.state.paused),
                        target_xyz=(float(self.state.target_x), float(self.state.target_y), float(self.state.target_z)),
                    )
                glfw.poll_events()
                impl.process_inputs()

                imgui.new_frame()
                self._draw_controls_window()
                imgui.render()

                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(window)
                time.sleep(0.01)
        finally:
            if self.sim_pub is not None:
                self.sim_pub.close()
            impl.shutdown()
            glfw.terminate()



