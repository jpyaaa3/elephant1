#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import zmq  # type: ignore
except Exception:
    zmq = None  # type: ignore


@dataclass(frozen=True)
class ControlU:
    u_linear: float
    u_roll: float
    u_s1: float
    u_s2: float


@dataclass(frozen=True)
class SimQ:
    linear_m: float
    roll_rad: float
    theta1_rad: float
    theta2_rad: float


@dataclass(frozen=True)
class SimMappingConfig:
    linear_u_min: float = 0.0
    linear_u_max: float = 360.0
    roll_u_min: float = 0.0
    roll_u_max: float = 360.0
    seg_u_min: float = 0.0
    seg_u_max: float = 360.0

    linear_q_min_m: float = -0.23
    linear_q_max_m: float = +0.01
    roll_q_min_rad: float = -math.pi / 2.0
    roll_q_max_rad: float = +math.pi / 2.0
    seg1_q_min_rad: float = -math.radians(36.0)
    seg1_q_max_rad: float = +math.radians(36.0)
    seg2_q_min_rad: float = -math.radians(36.0)
    seg2_q_max_rad: float = +math.radians(36.0)

    command_direction: tuple[int, int, int, int] = (1, 1, 1, 1)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _apply_axis_direction(u_value: float, direction: int, u_min: float, u_max: float) -> float:
    if int(direction) < 0:
        return float(u_min) + float(u_max) - float(u_value)
    return float(u_value)


def _map_axis_to_u(q_value: float, q_min: float, q_max: float, u_min: float, u_max: float) -> float:
    q_value = _clamp(q_value, q_min, q_max)
    if abs(float(q_max) - float(q_min)) < 1e-12:
        return float(u_min)
    ratio = (float(q_value) - float(q_min)) / (float(q_max) - float(q_min))
    return _clamp(float(u_min) + ratio * (float(u_max) - float(u_min)), u_min, u_max)


def _map_u_to_axis(u_value: float, u_min: float, u_max: float, q_min: float, q_max: float) -> float:
    u_value = _clamp(u_value, u_min, u_max)
    if abs(float(u_max) - float(u_min)) < 1e-12:
        return float(q_min)
    ratio = (float(u_value) - float(u_min)) / (float(u_max) - float(u_min))
    return _clamp(float(q_min) + ratio * (float(q_max) - float(q_min)), q_min, q_max)


def gensim_q_to_motor_deg(q: SimQ, cfg: SimMappingConfig = SimMappingConfig()) -> ControlU:
    return ControlU(
        u_linear=_map_axis_to_u(q.linear_m, cfg.linear_q_min_m, cfg.linear_q_max_m, cfg.linear_u_min, cfg.linear_u_max),
        u_roll=_map_axis_to_u(q.roll_rad, cfg.roll_q_min_rad, cfg.roll_q_max_rad, cfg.roll_u_min, cfg.roll_u_max),
        u_s1=_map_axis_to_u(q.theta1_rad, cfg.seg1_q_min_rad, cfg.seg1_q_max_rad, cfg.seg_u_min, cfg.seg_u_max),
        u_s2=_map_axis_to_u(q.theta2_rad, cfg.seg2_q_min_rad, cfg.seg2_q_max_rad, cfg.seg_u_min, cfg.seg_u_max),
    )


def motor_deg_to_gensim_q(u: ControlU, cfg: SimMappingConfig = SimMappingConfig()) -> SimQ:
    return SimQ(
        linear_m=_map_u_to_axis(u.u_linear, cfg.linear_u_min, cfg.linear_u_max, cfg.linear_q_min_m, cfg.linear_q_max_m),
        roll_rad=_map_u_to_axis(u.u_roll, cfg.roll_u_min, cfg.roll_u_max, cfg.roll_q_min_rad, cfg.roll_q_max_rad),
        theta1_rad=_map_u_to_axis(u.u_s1, cfg.seg_u_min, cfg.seg_u_max, cfg.seg1_q_min_rad, cfg.seg1_q_max_rad),
        theta2_rad=_map_u_to_axis(u.u_s2, cfg.seg_u_min, cfg.seg_u_max, cfg.seg2_q_min_rad, cfg.seg2_q_max_rad),
    )


def control_u_to_gensim_q(u: ControlU, cfg: SimMappingConfig = SimMappingConfig()) -> SimQ:
    dirs = tuple(int(v) for v in cfg.command_direction)
    motor_u = ControlU(
        u_linear=_clamp(_apply_axis_direction(u.u_linear, dirs[0], cfg.linear_u_min, cfg.linear_u_max), cfg.linear_u_min, cfg.linear_u_max),
        u_roll=_clamp(_apply_axis_direction(u.u_roll, dirs[1], cfg.roll_u_min, cfg.roll_u_max), cfg.roll_u_min, cfg.roll_u_max),
        u_s1=_clamp(_apply_axis_direction(u.u_s1, dirs[2], cfg.seg_u_min, cfg.seg_u_max), cfg.seg_u_min, cfg.seg_u_max),
        u_s2=_clamp(_apply_axis_direction(u.u_s2, dirs[3], cfg.seg_u_min, cfg.seg_u_max), cfg.seg_u_min, cfg.seg_u_max),
    )
    return motor_deg_to_gensim_q(motor_u, cfg)


def gensim_q_to_control_u(q: SimQ, cfg: SimMappingConfig = SimMappingConfig()) -> ControlU:
    dirs = tuple(int(v) for v in cfg.command_direction)
    motor_u = gensim_q_to_motor_deg(q, cfg)

    return ControlU(
        u_linear=_apply_axis_direction(motor_u.u_linear, dirs[0], cfg.linear_u_min, cfg.linear_u_max),
        u_roll=_apply_axis_direction(motor_u.u_roll, dirs[1], cfg.roll_u_min, cfg.roll_u_max),
        u_s1=_apply_axis_direction(motor_u.u_s1, dirs[2], cfg.seg_u_min, cfg.seg_u_max),
        u_s2=_apply_axis_direction(motor_u.u_s2, dirs[3], cfg.seg_u_min, cfg.seg_u_max),
    )


def now_s() -> float:
    return time.time()


def dumps_msg(msg: Dict[str, Any]) -> bytes:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def loads_msg(buf: bytes) -> Dict[str, Any]:
    return json.loads(buf.decode("utf-8"))


def pack_state(
    *,
    u: Optional[ControlU] = None,
    q: Optional[SimQ] = None,
    ts: Optional[float] = None,
    torque_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    ts = now_s() if ts is None else float(ts)
    out: Dict[str, Any] = {"t": "state", "ts": ts}
    if u is not None:
        out["u"] = {"linear": u.u_linear, "roll": u.u_roll, "s1": u.u_s1, "s2": u.u_s2}
    if q is not None:
        out["q"] = {"linear_m": q.linear_m, "roll_rad": q.roll_rad, "theta1_rad": q.theta1_rad, "theta2_rad": q.theta2_rad}
    if torque_enabled is not None:
        out["torque_enabled"] = bool(torque_enabled)
    return out


def pack_target_q(q: SimQ, *, source: str, seq: int, ts: Optional[float] = None) -> Dict[str, Any]:
    ts = now_s() if ts is None else float(ts)
    return {
        "t": "target",
        "ts": ts,
        "seq": int(seq),
        "source": str(source),
        "q": {"linear_m": q.linear_m, "roll_rad": q.roll_rad, "theta1_rad": q.theta1_rad, "theta2_rad": q.theta2_rad},
    }


def unpack_u(d: Dict[str, Any]) -> ControlU:
    return ControlU(
        u_linear=float(d.get("linear", 0.0)),
        u_roll=float(d.get("roll", 0.0)),
        u_s1=float(d.get("s1", 0.0)),
        u_s2=float(d.get("s2", 0.0)),
    )


def unpack_q(d: Dict[str, Any]) -> SimQ:
    return SimQ(
        linear_m=float(d.get("linear_m", 0.0)),
        roll_rad=float(d.get("roll_rad", 0.0)),
        theta1_rad=float(d.get("theta1_rad", 0.0)),
        theta2_rad=float(d.get("theta2_rad", 0.0)),
    )


class LinkClient:
    """DEALER client for bridge ROUTER."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        send_hz: float = 30.0,
        cfg: Optional[SimMappingConfig] = None,
    ) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq is required for LinkClient")
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

        self._send({"t": "hello", "ts": now_s()})

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
            self.is_connected = False

    def poll(self) -> None:
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
            self.last_ack_ts = float(msg.get("ts", now_s()))
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
            self.last_state_ts = float(msg.get("ts", now_s()))
            if "q" in msg:
                try:
                    self.last_q = unpack_q(msg["q"])
                except Exception:
                    pass
            if "u" in msg:
                try:
                    self.last_u = unpack_u(msg["u"])
                except Exception:
                    pass
            if "torque_enabled" in msg:
                self.torque_enabled = bool(msg.get("torque_enabled", False))
            self.is_connected = True

    def estop(self) -> None:
        self._send({"t": "estop", "ts": now_s()})

    def torque_on(self) -> None:
        self._send({"t": "torque_on", "ts": now_s()})

    def torque_off(self) -> None:
        self._send({"t": "torque_off", "ts": now_s()})

    def request_ports(self) -> None:
        self._send({"t": "ports", "ts": now_s()})

    def set_device(self, device: str) -> None:
        self._send({"t": "set_device", "ts": now_s(), "device": str(device)})

    def disconnect_device(self) -> None:
        self._send({"t": "disconnect_device", "ts": now_s()})

    def maybe_send_target_q(self, q: SimQ, *, source: str = "sim") -> None:
        now = time.time()
        if self._send_period > 0 and (now - self._t_last_tx) < self._send_period:
            return
        self._t_last_tx = now
        self.tx_seq += 1
        msg = pack_target_q(q, source=source, seq=self.tx_seq, ts=now_s())
        self._send(msg)
