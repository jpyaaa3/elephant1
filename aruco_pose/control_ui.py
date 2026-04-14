from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

MODULE_DIR = Path(__file__).resolve().parent
EX_DIR = MODULE_DIR / "ex"
if str(EX_DIR) not in sys.path:
    sys.path.insert(0, str(EX_DIR))

from engine import protocol as proto  # type: ignore
from engine.config_loader import load_app_config_from_ini  # type: ignore
from recording_control import DEFAULT_CONTROL_PATH, load_control_payload, sanitize_session_name, save_control_payload


DEFAULT_CONFIG_PATH = EX_DIR / "config.ini"


@dataclass
class CommandState:
    linear_u: float = 180.0
    roll_u: float = 180.0
    seg1_u: float = 180.0
    seg2_u: float = 180.0


@dataclass
class AutoMoveScenario:
    points: List[Tuple[float, float, float]]
    repeats: int
    index: int = 0
    active: bool = False
    arrival_time: Optional[float] = None

    def current_target(self) -> Optional[Tuple[float, float, float]]:
        if not self.active or self.index >= len(self.points):
            return None
        return self.points[self.index]


class ControlUiApp:
    def __init__(self, *, endpoint: str, config_path: Path, linear_default_m: float):
        bundle = load_app_config_from_ini(str(config_path))
        self.mapping_cfg = bundle.mapping_config
        self.endpoint = endpoint
        self.link = proto.LinkClient(endpoint=endpoint, cfg=self.mapping_cfg)
        default_q = proto.SimQ(linear_m=float(linear_default_m), roll_rad=0.0, theta1_rad=0.0, theta2_rad=0.0)
        default_u = proto.gensim_q_to_control_u(default_q, self.mapping_cfg)
        self.state = CommandState(
            linear_u=float(default_u.u_linear),
            roll_u=float(default_u.u_roll),
            seg1_u=float(default_u.u_s1),
            seg2_u=float(default_u.u_s2),
        )
        self._port_input = ""
        self._last_send_wall = 0.0
        self._send_period_s = 1.0 / 30.0
        self.imgui = None
        self._startup_pose_synced = False
        self.recording_control_path = DEFAULT_CONTROL_PATH
        control_payload = load_control_payload(self.recording_control_path)
        self._recording_csv_name = str(control_payload.get("csv_name", "session"))
        self._write_every = max(1, int(control_payload.get("write_every", 2)))
        self._auto_a = [float(self.state.roll_u), float(self.state.seg1_u), float(self.state.seg2_u)]
        self._auto_b = [float(self.state.roll_u), float(self.state.seg1_u), float(self.state.seg2_u)]
        self._auto_repeats = 0
        self._auto_summary = "No automove scenario applied."
        self._auto_scenario: Optional[AutoMoveScenario] = None
        self._auto_tolerance_u = 1.0
        self._auto_settle_s = 0.20

    def close(self) -> None:
        self.link.close()

    def _send_target_if_due(self, *, force: bool = False) -> None:
        now = time.time()
        if (not force) and (now - self._last_send_wall) < self._send_period_s:
            return
        self._last_send_wall = now
        q_cmd = proto.control_u_to_gensim_q(
            proto.ControlU(
                u_linear=float(self.state.linear_u),
                u_roll=float(self.state.roll_u),
                u_s1=float(self.state.seg1_u),
                u_s2=float(self.state.seg2_u),
            ),
            self.mapping_cfg,
        )
        self.link.maybe_send_target_q(
            q_cmd,
            source="slider",
        )

    def _cancel_automove(self, *, reason: Optional[str] = None) -> None:
        if self._auto_scenario is None:
            return
        self._auto_scenario.active = False
        if reason:
            self._auto_summary = reason

    def _apply_automove(self) -> None:
        repeats = max(0, int(self._auto_repeats))
        point_a = (float(self._auto_a[0]), float(self._auto_a[1]), float(self._auto_a[2]))
        point_b = (float(self._auto_b[0]), float(self._auto_b[1]), float(self._auto_b[2]))
        points = [point_a, point_b] * (repeats + 1)
        self._auto_scenario = AutoMoveScenario(points=points, repeats=repeats, index=0, active=True, arrival_time=None)
        self._auto_summary = (
            f"({point_a[0]:.1f}, {point_a[1]:.1f}, {point_a[2]:.1f}) <-> "
            f"({point_b[0]:.1f}, {point_b[1]:.1f}, {point_b[2]:.1f}) [{repeats} times]"
        )
        self._set_state_u(*points[0])
        self._send_target_if_due(force=True)

    def _set_state_u(self, roll_u: float, seg1_u: float, seg2_u: float) -> None:
        self.state.roll_u = float(roll_u)
        self.state.seg1_u = float(seg1_u)
        self.state.seg2_u = float(seg2_u)

    def _actual_u(self) -> Optional[Tuple[float, float, float]]:
        u_src = self.link.last_u
        if u_src is None:
            return None
        return (float(u_src.u_roll), float(u_src.u_s1), float(u_src.u_s2))

    def _step_automove(self) -> None:
        scenario = self._auto_scenario
        if scenario is None or (not scenario.active):
            return
        target = scenario.current_target()
        actual = self._actual_u()
        if target is None or actual is None:
            return
        errors = [abs(actual[i] - target[i]) for i in range(3)]
        now = time.time()
        if all(err <= self._auto_tolerance_u for err in errors):
            if scenario.arrival_time is None:
                scenario.arrival_time = now
                return
            if (now - scenario.arrival_time) < self._auto_settle_s:
                return
            scenario.index += 1
            scenario.arrival_time = None
            next_target = scenario.current_target()
            if next_target is None:
                scenario.active = False
                self._auto_summary = f"{self._auto_summary} [completed]"
                return
            self._set_state_u(*next_target)
            self._send_target_if_due(force=True)
            return
        scenario.arrival_time = None

    def _sync_from_bridge(self, *, prefer_actual: bool = True) -> bool:
        self.link.poll()
        u_src = self.link.last_u if prefer_actual else None
        if u_src is None and self.link.last_q_cmd is not None:
            u_src = proto.gensim_q_to_control_u(self.link.last_q_cmd, self.mapping_cfg)
        if u_src is None:
            return False
        self.state.linear_u = float(u_src.u_linear)
        self.state.roll_u = float(u_src.u_roll)
        self.state.seg1_u = float(u_src.u_s1)
        self.state.seg2_u = float(u_src.u_s2)
        return True

    def _draw_hardware_panel(self) -> None:
        imgui = self.imgui
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
                if reply_reason == "ports" and not ports:
                    imgui.text("No serial ports found")
                elif reply_reason != "ports":
                    imgui.text(f"Bridge: {reply_reason}")
            else:
                imgui.text_colored(f"Bridge: {reply_reason}", 1.0, 0.35, 0.35)

        if imgui.button("Torque On"):
            self.link.torque_on()
        imgui.same_line()
        if imgui.button("Torque Off"):
            self.link.torque_off()
        imgui.same_line()
        if imgui.button("Sync From Motor"):
            self._sync_from_bridge(prefer_actual=True)

        currents = getattr(self.link, "last_currents", {}) or {}
        roll_current = currents.get("roll")
        seg1_current = currents.get("s1")
        seg2_current = currents.get("s2")
        imgui.text(
            "Current [mA]: roll=%s seg1=%s seg2=%s"
            % (
                f"{float(roll_current):.1f}" if roll_current is not None else "n/a",
                f"{float(seg1_current):.1f}" if seg1_current is not None else "n/a",
                f"{float(seg2_current):.1f}" if seg2_current is not None else "n/a",
            )
        )

    def _draw_command_panel(self) -> None:
        imgui = self.imgui
        imgui.separator()
        imgui.text("Commands")
        imgui.text(f"linear fixed [u] = {self.state.linear_u:.1f}")

        changed_roll, roll_u = imgui.slider_float(
            "roll [u]",
            float(self.state.roll_u),
            float(self.mapping_cfg.roll_u_min),
            float(self.mapping_cfg.roll_u_max),
            format="%.1f",
        )
        changed_seg1, seg1_u = imgui.slider_float(
            "seg1 [u]",
            float(self.state.seg1_u),
            float(self.mapping_cfg.seg_u_min),
            float(self.mapping_cfg.seg_u_max),
            format="%.1f",
        )
        changed_seg2, seg2_u = imgui.slider_float(
            "seg2 [u]",
            float(self.state.seg2_u),
            float(self.mapping_cfg.seg_u_min),
            float(self.mapping_cfg.seg_u_max),
            format="%.1f",
        )

        changed_any = bool(changed_roll or changed_seg1 or changed_seg2)
        if changed_any:
            self._set_state_u(float(roll_u), float(seg1_u), float(seg2_u))
            self._cancel_automove(reason="Automove cancelled by manual slider input.")
            self._send_target_if_due(force=True)

        if imgui.button("Reset Angles"):
            self._set_state_u(180.0, 180.0, 180.0)
            self._cancel_automove(reason="Automove cancelled by reset.")
            self._send_target_if_due(force=True)

    def _draw_automove_panel(self) -> None:
        imgui = self.imgui
        imgui.separator()
        imgui.text("AutoMove")

        changed, value = imgui.input_float("A roll [u]", float(self._auto_a[0]), format="%.1f")
        if changed:
            self._auto_a[0] = float(value)
        changed, value = imgui.input_float("A seg1 [u]", float(self._auto_a[1]), format="%.1f")
        if changed:
            self._auto_a[1] = float(value)
        changed, value = imgui.input_float("A seg2 [u]", float(self._auto_a[2]), format="%.1f")
        if changed:
            self._auto_a[2] = float(value)
        changed, value = imgui.input_float("B roll [u]", float(self._auto_b[0]), format="%.1f")
        if changed:
            self._auto_b[0] = float(value)
        changed, value = imgui.input_float("B seg1 [u]", float(self._auto_b[1]), format="%.1f")
        if changed:
            self._auto_b[1] = float(value)
        changed, value = imgui.input_float("B seg2 [u]", float(self._auto_b[2]), format="%.1f")
        if changed:
            self._auto_b[2] = float(value)
        changed, value = imgui.input_int("Repeat Count", int(self._auto_repeats))
        if changed:
            self._auto_repeats = max(0, int(value))

        if imgui.button("Apply AutoMove"):
            self._apply_automove()
        imgui.same_line()
        if imgui.button("Cancel AutoMove"):
            self._cancel_automove(reason="Automove cancelled.")

        scenario = self._auto_scenario
        status = "active" if (scenario is not None and scenario.active) else "idle"
        imgui.text(f"Scenario: {self._auto_summary}")
        imgui.text(f"AutoMove Status: {status}")

    def _draw_recording_panel(self) -> None:
        imgui = self.imgui
        imgui.separator()
        imgui.text("Recording")

        changed_name, new_name = imgui.input_text("CSV Name", self._recording_csv_name, 256)
        if changed_name:
            self._recording_csv_name = str(new_name)

        payload = load_control_payload(self.recording_control_path)
        payload["csv_name"] = sanitize_session_name(self._recording_csv_name)
        changed_write_every, new_write_every = imgui.input_int("Write Every [frames]", int(self._write_every))
        if changed_write_every:
            self._write_every = max(1, int(new_write_every))
        payload["write_every"] = int(self._write_every)

        if imgui.button("Start Recording"):
            payload["recording_active"] = True
            payload["export_requested"] = False
            payload["status"] = "recording"
            save_control_payload(payload, self.recording_control_path)

        imgui.same_line()
        if imgui.button("Stop Recording"):
            payload["recording_active"] = False
            payload["status"] = "stopped"
            save_control_payload(payload, self.recording_control_path)

        imgui.same_line()
        if imgui.button("Export"):
            payload["export_requested"] = True
            payload["recording_active"] = False
            payload["status"] = "export_requested"
            save_control_payload(payload, self.recording_control_path)

        imgui.text(f"Control File: {self.recording_control_path}")
        imgui.text(f"Tracker write-every: {int(payload.get('write_every', self._write_every))} frame(s)")
        imgui.text(f"Status: {payload.get('status', 'idle')}")
        last_export = payload.get("last_export_path")
        if last_export:
            imgui.text(f"Last Export: {last_export}")

    def run(self) -> None:
        import glfw  # type: ignore
        import imgui  # type: ignore
        from imgui.integrations.glfw import GlfwRenderer  # type: ignore
        self.imgui = imgui

        if not glfw.init():
            raise SystemExit("glfw.init() failed.")

        glfw.window_hint(glfw.RESIZABLE, True)
        window = glfw.create_window(720, 420, "Motor Controls", None, None)
        if not window:
            glfw.terminate()
            raise SystemExit("Failed to create GLFW window.")

        glfw.make_context_current(window)
        imgui.create_context()
        impl = GlfwRenderer(window)

        self._sync_from_bridge()
        if self.link.last_q is not None:
            self._startup_pose_synced = True
        self._send_target_if_due(force=True)

        try:
            while not glfw.window_should_close(window):
                glfw.poll_events()
                impl.process_inputs()
                self.link.poll()
                if (not self._startup_pose_synced) and self._sync_from_bridge(prefer_actual=True):
                    self._startup_pose_synced = True
                self._step_automove()

                imgui.new_frame()
                imgui.begin("3-DOF Controls", True)
                self._draw_hardware_panel()
                self._draw_command_panel()
                self._draw_automove_panel()
                self._draw_recording_panel()
                imgui.end()
                imgui.render()

                impl.render(imgui.get_draw_data())
                glfw.swap_buffers(window)
                time.sleep(0.01)
        finally:
            impl.shutdown()
            glfw.terminate()
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone ImGui controller for roll/seg1/seg2 commands.")
    parser.add_argument("--bridge-endpoint", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--linear-default", type=float, default=0.0, help="Fixed linear command in meters.")
    args = parser.parse_args()

    app = ControlUiApp(
        endpoint=str(args.bridge_endpoint),
        config_path=args.config,
        linear_default_m=float(args.linear_default),
    )
    app.run()


if __name__ == "__main__":
    main()
