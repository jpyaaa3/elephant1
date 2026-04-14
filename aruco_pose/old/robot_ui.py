#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import zmq
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from engine import protocol as proto


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

    # UI-thread -> sim-thread commands
    _ik_solve_request: bool = field(default=False, init=False, repr=False)
    _ik_stop_request: bool = field(default=False, init=False, repr=False)

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

    def request_ik_solve(self) -> None:
        """UI-thread: request starting IK (sim-thread will consume)."""
        with self._lock:
            self._ik_solve_request = True
            self._ik_stop_request = False

    def request_ik_stop(self) -> None:
        """UI-thread: request stopping IK (sim-thread will consume)."""
        with self._lock:
            self._ik_stop_request = True
            self._ik_solve_request = False

    def consume_ik_requests(self) -> Tuple[bool, bool, np.ndarray]:
        """Sim-thread: atomically fetch & clear IK requests + current target xyz."""
        with self._lock:
            solve = bool(self._ik_solve_request)
            stop = bool(self._ik_stop_request)
            self._ik_solve_request = False
            self._ik_stop_request = False
            target = np.array([self.target_x, self.target_y, self.target_z], dtype=float)
        return solve, stop, target




# ZMQ Link: GenSim <-> Control_2
# ----------------------------

class LinkClient:
    """
    Lightweight DEALER client for Control_2.py (ROUTER).

    - Sends: hello, estop, target
    - Receives: ack, state
    """
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        send_hz: float = 30.0,
        cfg: proto.SimMappingConfig | None = None,
    ) -> None:
        self.endpoint = str(endpoint)
        self.cfg = cfg or proto.SimMappingConfig()
        self.send_hz = float(send_hz)
        self._send_period = (1.0 / self.send_hz) if self.send_hz > 0 else 0.0

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.linger = 0
        # identity helps debugging with ROUTER
        self.sock.setsockopt(zmq.IDENTITY, f"gensim-{os.getpid()}-{int(time.time()*1000)}".encode("utf-8"))
        self.sock.connect(self.endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.is_connected = True
        self.tx_seq = 0
        self._t_last_tx = 0.0

        self.last_ack_ts = 0.0
        self.last_state_ts = 0.0  # ts field from server state
        self._t_last_rx_wall = 0.0  # wall time when we last received anything

        self.last_q: proto.SimQ | None = None
        self.last_u: proto.ControlU | None = None
        self.last_ports: list[str] = []
        self.last_device: str = ""
        self.torque_enabled: bool = False
        self.last_reply_ok: bool = True
        self.last_reply_reason: str = ""

        # handshake
        self._send({"t": "hello", "ts": proto.now_s()})

    def close(self) -> None:
        try:
            self.poller.unregister(self.sock)
        except Exception:
            pass
        try:
            self.sock.close(0)
        except Exception:
            pass
        self.is_connected = False

    def rx_age_s(self) -> float:
        if self._t_last_rx_wall <= 0.0:
            return float("inf")
        return float(time.time() - self._t_last_rx_wall)

    def _send(self, msg: dict) -> None:
        try:
            self.sock.send_json(msg, flags=zmq.NOBLOCK)
        except Exception:
            # best-effort
            self.is_connected = False

    def poll(self) -> None:
        """Non-blocking receive; updates last_state/ack."""
        try:
            events = dict(self.poller.poll(timeout=0))
        except Exception:
            return
        if self.sock not in events:
            return
        try:
            msg = self.sock.recv_json(flags=zmq.NOBLOCK)
        except Exception:
            return

        self._t_last_rx_wall = time.time()
        t = str(msg.get("t", "")).lower()

        if t == "ack":
            self.last_ack_ts = float(msg.get("ts", proto.now_s()))
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
            self.last_state_ts = float(msg.get("ts", proto.now_s()))
            if "q" in msg:
                try:
                    self.last_q = proto.unpack_q(msg["q"])
                except Exception:
                    pass
            if "u" in msg:
                try:
                    self.last_u = proto.unpack_u(msg["u"])
                except Exception:
                    pass
            if "torque_enabled" in msg:
                self.torque_enabled = bool(msg.get("torque_enabled", False))
            self.is_connected = True
            return

    def estop(self) -> None:
        self._send({"t": "estop", "ts": proto.now_s()})

    def torque_on(self) -> None:
        self._send({"t": "torque_on", "ts": proto.now_s()})

    def torque_off(self) -> None:
        self._send({"t": "torque_off", "ts": proto.now_s()})

    def request_ports(self) -> None:
        self._send({"t": "ports", "ts": proto.now_s()})

    def set_device(self, device: str) -> None:
        self._send({"t": "set_device", "ts": proto.now_s(), "device": str(device)})

    def disconnect_device(self) -> None:
        self._send({"t": "disconnect_device", "ts": proto.now_s()})

    def maybe_send_target_q(self, q: proto.SimQ, *, source: str = "sim") -> None:
        now = time.time()
        if self._send_period > 0 and (now - self._t_last_tx) < self._send_period:
            return
        self._t_last_tx = now
        self.tx_seq += 1
        msg = proto.pack_target_q(q, source=source, seq=self.tx_seq, ts=proto.now_s())
        self._send(msg)




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
        link: Optional[Any] = None,
        mapping_cfg: Optional[proto.SimMappingConfig] = None,
        *,
        use_hardware: bool = False,
    ):
        self.p = params
        self.robot = robot
        self.state = state
        self.link = link
        self._proto_cfg = mapping_cfg or proto.SimMappingConfig()
        self._use_hardware = bool(use_hardware)
        self._stop = False
        self._ik_header_init_open = False
        self._ctrl_window_init = False
        self._port_input = ""

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
                self.state.request_ik_solve()
            imgui.same_line()
            if imgui.button("Stop IK"):
                self.state.request_ik_stop()

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

        sliders_locked = bool(self._use_hardware and (self.link is None or not bool(getattr(self.link, "torque_enabled", False))))
        disable_token = self._begin_disabled_ui(sliders_locked)

        # --- Control-unit sliders (0..360 position mode) ---
        q_now = proto.SimQ(
            linear_m=float(self.state.linear),
            roll_rad=float(self.state.roll),
            theta1_rad=float(self.state.theta1),
            theta2_rad=float(self.state.theta2),
        )
        u_now = proto.gensim_q_to_control_u(q_now, self._proto_cfg)

        changed_lin, u_linear = imgui.slider_float(
            "linear [u]", float(u_now.u_linear),
            float(self._proto_cfg.linear_u_min), float(self._proto_cfg.linear_u_max),
            format="%.1f"
        )
        changed_rdeg, u_roll = imgui.slider_float(
            "roll [u]", float(u_now.u_roll),
            float(self._proto_cfg.roll_u_min), float(self._proto_cfg.roll_u_max),
            format="%.1f"
        )
        changed_s1, u_s1 = imgui.slider_float(
            "seg1 [u]", float(u_now.u_s1),
            float(self._proto_cfg.seg_u_min), float(self._proto_cfg.seg_u_max),
            format="%.1f"
        )
        changed_s2, u_s2 = imgui.slider_float(
            "seg2 [u]", float(u_now.u_s2),
            float(self._proto_cfg.seg_u_min), float(self._proto_cfg.seg_u_max),
            format="%.1f"
        )
        self._end_disabled_ui(disable_token)

        changed_any = bool((not sliders_locked) and (changed_lin or changed_rdeg or changed_s1 or changed_s2))

        # If user touches sliders while IK is running, treat it as a manual override.
        if self.state.ik_running and changed_any:
            self.state.request_ik_stop()

        if changed_any:
            q_new = proto.control_u_to_gensim_q(
                proto.ControlU(
                    u_linear=float(u_linear),
                    u_roll=float(u_roll),
                    u_s1=float(u_s1),
                    u_s2=float(u_s2),
                ),
                self._proto_cfg,
            )
            self.state.linear = float(q_new.linear_m)
            self.state.roll = float(q_new.roll_rad)
            self.state.theta1 = float(q_new.theta1_rad)
            self.state.theta2 = float(q_new.theta2_rad)
        if sliders_locked:
            imgui.text("Sliders locked until Torque On")

        if self._use_hardware and self.link is not None:
            imgui.separator()
            imgui.text("Hardware")
            imgui.text(f"Link: {'OK' if self.link.is_connected else 'OFF'}")
            imgui.text(f"tx_seq={self.link.tx_seq} rx_age={self.link.rx_age_s():.2f}s")
            current_device = str(getattr(self.link, "last_device", "") or "").strip()
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
            ports = list(getattr(self.link, "last_ports", []))
            if ports:
                imgui.text("Detected Ports:")
                imgui.same_line()
                for idx, port in enumerate(ports):
                    if imgui.small_button(f"{port}##port_{idx}"):
                        self._port_input = str(port)
                    if (idx + 1) < len(ports):
                        imgui.same_line()
            reply_reason = str(getattr(self.link, "last_reply_reason", "") or "").strip()
            if reply_reason:
                if bool(getattr(self.link, "last_reply_ok", True)):
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
            # Manual override: stop IK and return to neutral
            self.state.request_ik_stop()
            self.state.linear = 0.0
            self.state.roll = 0.0
            self.state.theta1 = 0.0
            self.state.theta2 = 0.0

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
                glfw.poll_events()
                impl.process_inputs()

                imgui.new_frame()
                self._draw_controls_window()
                imgui.render()

                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(window)
                time.sleep(0.01)
        finally:
            impl.shutdown()
            glfw.terminate()



