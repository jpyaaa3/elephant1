#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Set
import zmq

from engine.config_loader import load_app_config_from_ini
from engine.config_loader import HardwareConfig
from engine.motor import load_hardware, tick_to_deg_0_360
import engine.protocol as proto

from serial.tools import list_ports as serial_list_ports


class ControlHost:
    """ROUTER-side host that receives controller requests and drives hardware."""

    def __init__(
        self,
        *,
        bind_addr: str,
        sim_pub_addr: str,
        hw: Any,
        direction_by_id: Dict[int, int],
        device: str,
        hardware_cfg: Optional[HardwareConfig],
        cfg: proto.SimMappingConfig = proto.SimMappingConfig(),
        state_hz: float = 10.0,
        hw_read_hz: float = 20.0,
        hw_cmd_hz: float = 30.0,
    ) -> None:
        if zmq is None:
            raise SystemExit("pyzmq is required. Install: pip install pyzmq")
        self.cfg = cfg
        self.hw = hw
        self.direction_by_id = direction_by_id
        self.device = str(device)
        self.hardware_cfg = hardware_cfg

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(bind_addr)
        self.sim_pub = self.ctx.socket(zmq.PUB)
        self.sim_pub.setsockopt(zmq.LINGER, 0)
        self.sim_pub.bind(str(sim_pub_addr))

        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        self.clients: Set[bytes] = set()
        self.last_u: Optional[proto.ControlU] = None
        self.last_q: Optional[proto.SimQ] = None
        self.last_state_ts: float = 0.0
        self.torque_enabled: bool = False
        self.last_ik_target_xyz: Optional[tuple[float, float, float]] = None

        self._state_period = 1.0 / max(0.1, float(state_hz))
        self._read_period = 1.0 / max(0.1, float(hw_read_hz))
        self._cmd_period = 1.0 / max(0.1, float(hw_cmd_hz))
        self._t_read = 0.0
        self._t_state = 0.0
        self._t_cmd = 0.0

        self._pending_target_q: Optional[proto.SimQ] = None
        self._pending_target_seq: int = -1

        self._ids = getattr(hw, "ids", [])
        self._hw_lock = threading.RLock()
        self._stop_event = threading.Event()

    def _has_hw(self) -> bool:
        return self.hw is not None

    def _list_ports(self) -> list[str]:
        if serial_list_ports is None:
            return []
        try:
            return [str(p.device) for p in serial_list_ports.comports()]
        except Exception:
            return []

    def set_device(self, device: str) -> None:
        new_device = str(device).strip()
        if not new_device:
            raise ValueError("empty device")
        with self._hw_lock:
            old_hw = self.hw
            old_direction = dict(self.direction_by_id)
            old_ids = list(self._ids)
            old_device = str(self.device)
            self._pending_target_q = None
            self._pending_target_seq = -1
            self.last_u = None
            self.last_q = None
            self.last_state_ts = 0.0
            self.torque_enabled = False
            self.last_ik_target_xyz = None
            if old_hw is not None:
                try:
                    old_hw.close()
                except Exception:
                    pass
            try:
                new_hw, new_direction = load_hardware(new_device, hardware_cfg=self.hardware_cfg)
                new_hw.open()
            except Exception as exc:
                if old_hw is not None:
                    try:
                        old_hw.open()
                    except Exception:
                        pass
                self.hw = old_hw
                self.direction_by_id = old_direction
                self._ids = old_ids
                self.device = old_device
                raise RuntimeError(f"failed to open device {new_device}: {exc}") from exc
            self.hw = new_hw
            self.direction_by_id = new_direction
            self._ids = list(getattr(new_hw, "ids", []))
            self.device = new_device

    def clear_device(self) -> None:
        with self._hw_lock:
            old_hw = self.hw
            self._pending_target_q = None
            self._pending_target_seq = -1
            self.last_u = None
            self.last_q = None
            self.last_state_ts = 0.0
            self.torque_enabled = False
            self.last_ik_target_xyz = None
            self.hw = None
            self.direction_by_id = {}
            self._ids = []
            self.device = ""
            if old_hw is not None:
                try:
                    old_hw.close()
                except Exception:
                    pass

    def _is_allowed_source(self, source: str) -> bool:
        return str(source) in ("slider", "ik", "sim")

    def _reply(self, ident: bytes, msg: Dict[str, Any]) -> None:
        try:
            self.sock.send_multipart([ident, proto.dumps_msg(msg)], flags=0)
        except Exception:
            pass

    def _broadcast(self, msg: Dict[str, Any]) -> None:
        data = proto.dumps_msg(msg)
        dead: Set[bytes] = set()
        for ident in list(self.clients):
            try:
                self.sock.send_multipart([ident, data], flags=zmq.NOBLOCK)
            except Exception:
                dead.add(ident)
        self.clients.difference_update(dead)
        try:
            self.sim_pub.send(data, flags=zmq.NOBLOCK)
        except Exception:
            pass

    def _read_hw_state(self) -> None:
        if not self._has_hw():
            return
        try:
            with self._hw_lock:
                ticks_by_id = self.hw.get_present_positions()
        except Exception:
            return
        if not self._ids or len(self._ids) < 4:
            return
        motor_deg_vals = []
        for dxl_id in self._ids[:4]:
            tick = int(ticks_by_id.get(dxl_id, 0))
            direction = int(self.direction_by_id.get(dxl_id, +1))
            motor_deg_vals.append(tick_to_deg_0_360(tick, direction))
        motor_deg = proto.ControlU(
            u_linear=motor_deg_vals[0],
            u_roll=motor_deg_vals[1],
            u_s1=motor_deg_vals[2],
            u_s2=motor_deg_vals[3],
        )
        self.last_q = proto.motor_deg_to_sim_q(motor_deg, self.cfg)
        self.last_u = proto.sim_q_to_control_u(self.last_q, self.cfg)
        self.last_state_ts = time.time()

    def _apply_sim_q_target(self, q: proto.SimQ) -> bool:
        if not self._has_hw():
            return False
        motor_deg = proto.sim_q_to_motor_deg(q, self.cfg)
        try:
            with self._hw_lock:
                self.hw.command_4dof_deg(motor_deg.u_linear, motor_deg.u_roll, motor_deg.u_s1, motor_deg.u_s2)
            return True
        except Exception:
            return False

    def torque_on(self, *, configure_modes: bool = True, set_profiles: bool = True, go_mid: bool = False) -> None:
        if not self._has_hw():
            raise RuntimeError("no device selected")
        with self._hw_lock:
            if configure_modes:
                self.hw.set_operating_modes()
            if set_profiles:
                self.hw.set_profiles()
            self.hw.torque_on_all()
            self.torque_enabled = True
            if go_mid:
                self.hw.go_mid_pose()

    def torque_off(self) -> None:
        if not self._has_hw():
            raise RuntimeError("no device selected")
        with self._hw_lock:
            self._pending_target_q = None
            self._pending_target_seq = -1
            self.hw.torque_off_all()
            self.torque_enabled = False

    def close(self) -> None:
        try:
            self.poller.unregister(self.sock)
        except Exception:
            pass
        try:
            self.sock.close(0)
        except Exception:
            pass
        try:
            self.sim_pub.close(0)
        except Exception:
            pass

    def _handle_msg(self, ident: bytes, msg: Dict[str, Any]) -> None:
        self.clients.add(ident)
        t = str(msg.get("t", "")).lower()
        if t in ("hello", "hi"):
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": True, "device": self.device, "torque_enabled": self.torque_enabled})
            return
        if t == "estop":
            ok = True
            try:
                self.torque_off()
            except Exception:
                ok = False
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": ok, "device": self.device, "torque_enabled": self.torque_enabled})
            return
        if t == "torque_on":
            ok = True
            try:
                self.torque_on()
            except Exception:
                ok = False
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": ok, "device": self.device, "torque_enabled": self.torque_enabled})
            return
        if t == "torque_off":
            ok = True
            try:
                self.torque_off()
            except Exception:
                ok = False
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": ok, "device": self.device, "torque_enabled": self.torque_enabled})
            return
        if t == "ports":
            ports = self._list_ports()
            ports_text = ", ".join(ports) if ports else "None"
            print(f"[host] ports searched: {ports_text}")
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": True, "device": self.device, "ports": ports, "reason": "ports", "torque_enabled": self.torque_enabled})
            return
        if t == "set_device":
            device = str(msg.get("device", "")).strip()
            ok = True
            reason = f"device set to {device}" if device else "device unchanged"
            try:
                self.set_device(device)
            except Exception as exc:
                ok = False
                reason = str(exc)
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": ok, "device": self.device, "ports": self._list_ports(), "reason": reason, "torque_enabled": self.torque_enabled})
            return
        if t == "disconnect_device":
            ok = True
            reason = "device disconnected"
            try:
                self.clear_device()
            except Exception as exc:
                ok = False
                reason = str(exc)
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": ok, "device": self.device, "ports": self._list_ports(), "reason": reason, "torque_enabled": self.torque_enabled})
            return
        if t == "target":
            source = str(msg.get("source", "sim"))
            if not self._is_allowed_source(source):
                self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": False, "reason": "source_reject", "device": self.device, "torque_enabled": self.torque_enabled})
                return
            seq = int(msg.get("seq", -1))
            q: Optional[proto.SimQ] = None
            if "u" in msg:
                q = proto.control_u_to_sim_q(proto.unpack_u(msg["u"]), self.cfg)
            elif "q" in msg:
                q = proto.unpack_q(msg["q"])
            if q is None:
                self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": False, "reason": "bad_target", "device": self.device, "torque_enabled": self.torque_enabled})
                return
            self._pending_target_q = q
            self._pending_target_seq = seq
            target_raw = msg.get("target", None)
            if isinstance(target_raw, (list, tuple)) and len(target_raw) == 3:
                self.last_ik_target_xyz = (float(target_raw[0]), float(target_raw[1]), float(target_raw[2]))
            if not self._has_hw():
                self.last_q = q
                self.last_u = proto.sim_q_to_control_u(q, self.cfg)
                self.last_state_ts = time.time()
            self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": True, "seq": seq, "device": self.device, "torque_enabled": self.torque_enabled})
            return
        self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": False, "reason": "unknown_type", "device": self.device, "torque_enabled": self.torque_enabled})

    def loop_forever(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            try:
                events = dict(self.poller.poll(timeout=10))
            except KeyboardInterrupt:
                break
            if self.sock in events and events[self.sock] & zmq.POLLIN:
                while True:
                    try:
                        ident, data = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    except Exception:
                        break
                    try:
                        msg = proto.loads_msg(data)
                    except Exception:
                        self._reply(ident, {"t": "ack", "ts": proto.now_s(), "ok": False, "reason": "json", "torque_enabled": self.torque_enabled})
                        continue
                    self._handle_msg(ident, msg)
            if (now - self._t_read) >= self._read_period:
                self._t_read = now
                self._read_hw_state()
            if self._pending_target_q is not None and (now - self._t_cmd) >= self._cmd_period:
                self._t_cmd = now
                if self._apply_sim_q_target(self._pending_target_q):
                    self._pending_target_q = None
            if (now - self._t_state) >= self._state_period:
                self._t_state = now
                self._broadcast(
                    proto.pack_state(
                        u=self.last_u,
                        q=self.last_q,
                        ts=self.last_state_ts or now,
                        torque_enabled=self.torque_enabled,
                        ik_target_xyz=self.last_ik_target_xyz,
                    )
                )


def run_host(
    *,
    config_path: str,
    bind_addr: str,
    device: str,
) -> None:
    bundle = load_app_config_from_ini(str(config_path))
    hw_cfg: HardwareConfig | None = bundle.hardware_config
    hw = None
    direction: Dict[int, int] = {}
    device = str(device).strip()
    if device:
        hw, direction = load_hardware(device, hardware_cfg=hw_cfg)
    try:
        if hw is not None:
            hw.open()
        server = ControlHost(
            bind_addr=str(bind_addr),
            sim_pub_addr=str(bundle.sim_config.host_sim_port),
            hw=hw,
            direction_by_id=direction,
            device=device,
            hardware_cfg=hw_cfg,
            cfg=bundle.mapping_config,
        )
        print(f"[host] comm with ctrl by {bind_addr}")
        print(f"[host] comm with sim by {bundle.sim_config.host_sim_port}")
        try:
            server.loop_forever()
        finally:
            server.close()
    finally:
        try:
            if hw is not None:
                hw.close()
        except Exception:
            pass


def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), "config.ini")
    bundle = load_app_config_from_ini(config_path)

    run_host(
        config_path=config_path,
        bind_addr=str(bundle.sim_config.host_ctrl_port),
        device="",
    )


if __name__ == "__main__":
    main()
