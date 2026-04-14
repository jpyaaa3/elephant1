#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import zmq
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

from engine import protocol as proto
from engine.config_loader import AppConfigBundle, JointLimit, load_app_config_from_ini
from engine.ik_new import (
    GoalSpec as IKGoalSpec,
    IKNewPipeline,
    Kinematics as IKKinematics,
    LinearJointPathPlanner,
    MultiSeedConfig as IKMultiSeedConfig,
    MultiSeedGoalFinder,
    make_default_search_bounds,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recording_control import DEFAULT_CONTROL_PATH, load_control_payload, sanitize_session_name, save_control_payload


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.ini"


@dataclass
class ControlState:
    linear: float = 0.0
    roll: float = 0.0
    theta1: float = 0.0
    theta2: float = 0.0
    paused: bool = False
    aruco_filtering: bool = True

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

    def snapshot(self) -> Tuple[float, float, float, float, bool, bool, Tuple[float, float, float]]:
        with self._lock:
            return (
                self.linear,
                self.roll,
                self.theta1,
                self.theta2,
                self.paused,
                self.aruco_filtering,
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
        backend: Optional[Any] = None,
        *,
        use_hardware: bool = False,
    ):
        self.p = params
        self.robot = robot
        self.state = state
        self.link = link
        self.backend = backend
        self._proto_cfg = mapping_cfg or proto.SimMappingConfig()
        self._use_hardware = bool(use_hardware)
        self._stop = False
        self._ik_header_init_open = False
        self._ctrl_window_init = False
        self._port_input = ""
        self._recording_control_path = DEFAULT_CONTROL_PATH
        control_payload = load_control_payload(self._recording_control_path)
        self._recording_csv_name = str(control_payload.get("csv_name", "session"))
        self._recording_write_every = max(1, int(control_payload.get("write_every", 2)))

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

    def _draw_recording_panel(self) -> None:
        imgui.separator()
        imgui.text("CSV Recording")
        changed_name, new_name = imgui.input_text("CSV Name", self._recording_csv_name, 256)
        if changed_name:
            self._recording_csv_name = str(new_name)

        payload = load_control_payload(self._recording_control_path)
        dirty = False
        payload["csv_name"] = sanitize_session_name(self._recording_csv_name)
        if changed_name:
            dirty = True
        changed_write_every, new_write_every = imgui.input_int("Write Every [frames]", int(self._recording_write_every))
        if changed_write_every:
            self._recording_write_every = max(1, int(new_write_every))
            dirty = True
        payload["write_every"] = int(self._recording_write_every)
        if imgui.button("Start Recording"):
            payload["recording_active"] = True
            payload["export_requested"] = False
            payload["status"] = "recording"
            dirty = True
        imgui.same_line()
        if imgui.button("Stop Recording"):
            payload["recording_active"] = False
            payload["status"] = "stopped"
            dirty = True
        imgui.same_line()
        if imgui.button("Export CSV"):
            payload["recording_active"] = False
            payload["export_requested"] = True
            payload["status"] = "export_requested"
            dirty = True

        if dirty:
            save_control_payload(payload, self._recording_control_path)
        imgui.text(f"Control File: {self._recording_control_path}")
        imgui.text(f"Status: {payload.get('status', 'idle')}")
        last_export = payload.get("last_export_path", None)
        if last_export:
            imgui.text(f"Last Export: {last_export}")


    def _draw_controls_window(self) -> None:
        if not self._ctrl_window_init:
            cond = getattr(imgui, "ONCE", getattr(imgui, "FIRST_USE_EVER", 1))
            imgui.set_next_window_size(1120, 760, cond)
            self._ctrl_window_init = True
        imgui.begin("4-DOF Controls", True)
        imgui.push_item_width(420)

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
            raw_currents = getattr(self.link, "last_currents", {}) or {}
            imgui.text(
                "Current [mA]: roll=%s seg1=%s seg2=%s"
                % (
                    f"{float(raw_currents.get('roll')):.1f}" if raw_currents.get("roll") is not None else "n/a",
                    f"{float(raw_currents.get('s1')):.1f}" if raw_currents.get("s1") is not None else "n/a",
                    f"{float(raw_currents.get('s2')):.1f}" if raw_currents.get("s2") is not None else "n/a",
                )
            )

        _, self.state.paused = imgui.checkbox("pause control updates", self.state.paused)
        imgui.same_line()
        _, self.state.aruco_filtering = imgui.checkbox("aruco filtering", self.state.aruco_filtering)
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
        self._draw_recording_panel()
        imgui.pop_item_width()
        imgui.end()
    def run(self) -> None:
        if not glfw.init():
            raise SystemExit("glfw.init() failed.")

        glfw.window_hint(glfw.RESIZABLE, True)
        window = glfw.create_window(1120, 760, "Snake Controls (ImGui)", None, None)
        if not window:
            glfw.terminate()
            raise SystemExit("Failed to create GLFW window.")

        glfw.make_context_current(window)

        imgui.create_context()
        imgui.get_io().font_global_scale = 1.12
        impl = GlfwRenderer(window)


        try:
            while not glfw.window_should_close(window) and not self._stop:
                if self.backend is not None:
                    try:
                        self.backend.poll()
                    except Exception:
                        pass
                if self.link is not None:
                    try:
                        self.link.poll()
                    except Exception:
                        pass
                glfw.poll_events()
                impl.process_inputs()

                imgui.new_frame()
                self._draw_controls_window()
                imgui.render()

                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(window)
                time.sleep(0.01)
        finally:
            if self.backend is not None:
                try:
                    self.backend.close()
                except Exception:
                    pass
            impl.shutdown()
            glfw.terminate()


@dataclass(frozen=True)
class RobotKinematicsLayout:
    n_nodes: int
    n_seg: int
    chain_origin_local: np.ndarray
    base_axis_sign: float
    roll_axis_sign: float
    bend_axis_sign: float


class StandaloneRobotUiBackend:
    def __init__(
        self,
        *,
        state: ControlState,
        config_path: Path,
        bridge_endpoint: Optional[str] = None,
        spawn_recorder: bool = True,
    ) -> None:
        self.state = state
        self.config_path = Path(config_path).resolve()
        self.bundle: AppConfigBundle = load_app_config_from_ini(str(self.config_path))
        self.mapping_cfg = self.bundle.mapping_config
        self.limit: JointLimit = self.bundle.JointLimit
        self.model = self.bundle.model_config
        self.ik_cfg = self.bundle.IkConfig
        self.endpoint = str(bridge_endpoint or self.bundle.SimConfig.zmq_endpoint)
        self._bridge_proc: Optional[subprocess.Popen] = None
        self._recorder_proc: Optional[subprocess.Popen] = None
        self.link = self._try_init_link_with_bridge(endpoint=self.endpoint)
        self._startup_pose_synced = False
        self._target_initialized = False
        self._last_sent_q: Optional[tuple[float, float, float, float]] = None
        self._preferred_dir_world = self._spawn_rotation_matrix() @ np.array([1.0, 0.0, 0.0], dtype=float)
        self.layout = self._load_layout()
        self.kin = IKKinematics(
            pitch=float(self.model.pitch),
            n_nodes=int(self.layout.n_nodes),
            n_seg=int(self.layout.n_seg),
            origin_xyz=np.asarray(self.layout.chain_origin_local, dtype=float).reshape(3),
            limit=self.limit,
            base_axis_sign=float(self.layout.base_axis_sign),
            roll_axis_sign=float(self.layout.roll_axis_sign),
            bend_axis_sign=float(self.layout.bend_axis_sign),
        )
        self.pipeline = IKNewPipeline(
            goal_finder=MultiSeedGoalFinder(
                self.kin,
                make_default_search_bounds(self.kin),
                cfg=IKMultiSeedConfig(
                    max_seeds=16,
                    n_random=8,
                    seed_rng=0,
                    max_iters_per_seed=max(int(self.ik_cfg.max_iters), 1),
                    stall_limit=max(int(self.ik_cfg.stall_limit), 1),
                    fd_eps=float(self.ik_cfg.fd_eps),
                ),
            ),
            path_planner=LinearJointPathPlanner(),
        )
        if spawn_recorder:
            self._start_recorder()

    def _try_init_link_with_bridge(self, *, endpoint: str) -> LinkClient:
        try:
            link = LinkClient(endpoint=endpoint, cfg=self.mapping_cfg)
            t0 = time.time()
            while time.time() - t0 < 0.15:
                link.poll()
                if np.isfinite(link.rx_age_s()) and link.rx_age_s() < 1.0:
                    return link
                time.sleep(0.01)
            try:
                link.close()
            except Exception:
                pass
        except Exception:
            pass

        bridge_path = Path(__file__).resolve().parent / "bridge.py"
        cmd = [sys.executable, str(bridge_path), "--bind", self._build_bind_endpoint(endpoint)]
        self._bridge_proc = subprocess.Popen(cmd)
        time.sleep(0.25)
        return LinkClient(endpoint=endpoint, cfg=self.mapping_cfg)

    @staticmethod
    def _build_bind_endpoint(endpoint: str) -> str:
        ep = str(endpoint).strip()
        if ep.startswith("tcp://127.0.0.1:"):
            return "tcp://*:" + ep.rsplit(":", 1)[-1]
        if ep.startswith("tcp://localhost:"):
            return "tcp://*:" + ep.rsplit(":", 1)[-1]
        return ep

    @staticmethod
    def _axis_sign(raw_axis: Any, axis_idx: int) -> float:
        try:
            axis = np.asarray(raw_axis, dtype=float).reshape(-1)
            if axis.size <= axis_idx:
                return 1.0
            value = float(axis[axis_idx])
            if abs(value) < 1e-9:
                return 1.0
            return -1.0 if value < 0.0 else 1.0
        except Exception:
            return 1.0

    def _load_layout(self) -> RobotKinematicsLayout:
        manifest_path = Path(self.bundle.SimConfig.build_dir) / self.bundle.SimConfig.assy_build_json
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest json not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            build = json.load(handle)

        joints = list(build.get("joints", []) or [])
        prismatic = [j for j in joints if str(j.get("type", "")).strip().lower() == "prismatic"]
        revolute = [j for j in joints if str(j.get("type", "")).strip().lower() == "revolute"]
        if not prismatic or len(revolute) < 2:
            raise RuntimeError("manifest json does not contain enough control joints for standalone robot_ui")

        base_joint = prismatic[0]
        roll_joint = revolute[0]
        bend_joints = revolute[1:]
        first_bend = bend_joints[0]
        anchor = first_bend.get("anchor_root", [0.0, 0.0, 0.0])
        chain_origin_local = np.array([float(anchor[0]), float(anchor[1]), float(anchor[2])], dtype=float)
        n_nodes = len(bend_joints)
        n_seg = int(self.model.n_seg) if self.model.n_seg is not None else max(1, int(math.ceil(float(n_nodes) / 2.0)))
        n_seg = min(max(1, n_seg), n_nodes)

        return RobotKinematicsLayout(
            n_nodes=n_nodes,
            n_seg=n_seg,
            chain_origin_local=chain_origin_local,
            base_axis_sign=self._axis_sign(base_joint.get("axis_root", [1.0, 0.0, 0.0]), 0),
            roll_axis_sign=self._axis_sign(roll_joint.get("axis_root", [1.0, 0.0, 0.0]), 0),
            bend_axis_sign=self._axis_sign(first_bend.get("axis_root", [0.0, 1.0, 0.0]), 1),
        )

    def _spawn_rotation_matrix(self) -> np.ndarray:
        rx, ry, rz = [math.radians(float(v)) for v in self.model.spawn_euler_deg]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        return rot_z @ rot_y @ rot_x

    def _spawn_translation(self) -> np.ndarray:
        return np.asarray(self.model.spawn_xyz, dtype=float).reshape(3)

    def _q_to_vec(self, q: proto.SimQ) -> np.ndarray:
        return np.array([float(q.linear_m), float(q.roll_rad), float(q.theta1_rad), float(q.theta2_rad)], dtype=float)

    def _vec_to_q(self, q_vec: np.ndarray) -> proto.SimQ:
        qv = np.asarray(q_vec, dtype=float).reshape(4)
        return proto.SimQ(
            linear_m=float(qv[0]),
            roll_rad=float(qv[1]),
            theta1_rad=float(qv[2]),
            theta2_rad=float(qv[3]),
        )

    def _state_q(self) -> proto.SimQ:
        return proto.SimQ(
            linear_m=float(self.state.linear),
            roll_rad=float(self.state.roll),
            theta1_rad=float(self.state.theta1),
            theta2_rad=float(self.state.theta2),
        )

    def _set_state_from_q(self, q: proto.SimQ) -> None:
        self.state.set_all(float(q.linear_m), float(q.roll_rad), float(q.theta1_rad), float(q.theta2_rad), self.state.paused)

    def _tip_world_from_q(self, q: proto.SimQ) -> np.ndarray:
        tip_local, _fwd_local = self.kin.tip_position_and_forward(self._q_to_vec(q))
        return self._spawn_translation() + self._spawn_rotation_matrix() @ tip_local

    def _preferred_dir_local(self) -> np.ndarray:
        return self._spawn_rotation_matrix().T @ np.asarray(self._preferred_dir_world, dtype=float).reshape(3)

    def _target_world_to_local(self, target_world: np.ndarray) -> np.ndarray:
        return self._spawn_rotation_matrix().T @ (np.asarray(target_world, dtype=float).reshape(3) - self._spawn_translation())

    def _sync_from_link_once(self) -> None:
        if self._startup_pose_synced:
            return
        q_src = self.link.last_q
        if q_src is None and self.link.last_u is not None:
            q_src = proto.control_u_to_gensim_q(self.link.last_u, self.mapping_cfg)
        if q_src is None:
            return
        self._set_state_from_q(q_src)
        self._startup_pose_synced = True

    def _initialize_target_from_state(self) -> None:
        if self._target_initialized:
            return
        tip_world = self._tip_world_from_q(self._state_q())
        self.state.target_x = float(tip_world[0])
        self.state.target_y = float(tip_world[1])
        self.state.target_z = float(tip_world[2])
        self._target_initialized = True

    def _send_state_target(self, *, force: bool = False, source: str = "slider") -> None:
        if self.state.paused:
            return
        q = self._state_q()
        q_key = (float(q.linear_m), float(q.roll_rad), float(q.theta1_rad), float(q.theta2_rad))
        if (not force) and self._last_sent_q == q_key:
            return
        self.link.maybe_send_target_q(q, source=source)
        self._last_sent_q = q_key

    def _solve_ik(self, target_world: np.ndarray) -> None:
        q_seed = self._q_to_vec(self.link.last_q if self.link.last_q is not None else self._state_q())
        target_local = self._target_world_to_local(target_world)
        self.state.set_ik_status(True, False, False, float("inf"))
        result = self.pipeline.solve(
            q_start=q_seed,
            goal=IKGoalSpec(
                target_world=np.asarray(target_local, dtype=float).reshape(3),
                preferred_dir_world=self._preferred_dir_local() if bool(self.ik_cfg.prefer_tip_plus_x) else None,
                position_tol_m=float(self.ik_cfg.tol),
                direction_tol_deg=float(self.ik_cfg.direction_tol_deg) if bool(self.ik_cfg.prefer_tip_plus_x) else None,
            ),
        )
        if result.goal_search.best is not None:
            q_best = self._vec_to_q(result.goal_search.best.q)
            self.state.set_ik_solution(q_best.linear_m, q_best.roll_rad, q_best.theta1_rad, q_best.theta2_rad)
            self.state.set_ik_debug(
                sim_tip_err_m=float(result.goal_search.best.position_error_m),
                linear_err_m=0.0,
                roll_err_rad=0.0,
                theta1_err_rad=0.0,
                theta2_err_rad=0.0,
                bend_max_err_rad=0.0,
            )
            if result.success:
                self._set_state_from_q(q_best)
                self._send_state_target(force=True, source="ik")
        self.state.set_ik_status(
            False,
            bool(result.success),
            not bool(result.success),
            float(result.goal_search.best.position_error_m) if result.goal_search.best is not None else float("inf"),
        )

    def poll(self) -> None:
        self.link.poll()
        self._sync_from_link_once()
        self._initialize_target_from_state()
        solve, stop, target_world = self.state.consume_ik_requests()
        if stop:
            self.state.set_ik_status(False, False, False, 0.0)
        if solve:
            self._solve_ik(np.asarray(target_world, dtype=float).reshape(3))
        self._send_state_target()

    def close(self) -> None:
        try:
            self.link.close()
        except Exception:
            pass
        if self._recorder_proc is not None:
            try:
                self._recorder_proc.terminate()
            except Exception:
                pass
            self._recorder_proc = None
        if self._bridge_proc is not None:
            try:
                self._bridge_proc.terminate()
            except Exception:
                pass
            self._bridge_proc = None

    def _start_recorder(self) -> None:
        recorder_path = ROOT_DIR / "recorder.py"
        if not recorder_path.is_file():
            return
        try:
            self._recorder_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(recorder_path),
                    "--endpoint",
                    self.endpoint,
                ]
            )
        except Exception:
            self._recorder_proc = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone robot UI for 4-DOF hardware control and CSV recording.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--bridge-endpoint", type=str, default=None)
    parser.add_argument("--no-recorder", action="store_true", help="Do not auto-start recorder.py")
    args = parser.parse_args()

    backend = StandaloneRobotUiBackend(
        state=ControlState(),
        config_path=Path(args.config),
        bridge_endpoint=args.bridge_endpoint,
        spawn_recorder=not bool(args.no_recorder),
    )
    controller = ImGuiController(
        params=backend.bundle.SimParam,
        robot=None,
        state=backend.state,
        link=backend.link,
        mapping_cfg=backend.mapping_cfg,
        backend=backend,
        use_hardware=True,
    )
    controller.run()


if __name__ == "__main__":
    main()



