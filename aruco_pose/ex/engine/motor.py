#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-level Dynamixel driver and motor conversion helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from engine import protocol as proto
from engine.config_loader import HardwareConfig

try:
    from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler
except Exception:  # pragma: no cover - optional on dev machines
    GroupSyncRead = None  # type: ignore
    GroupSyncWrite = None  # type: ignore
    PacketHandler = None  # type: ignore
    PortHandler = None  # type: ignore


ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_PROFILE_ACCEL = 108
ADDR_PROFILE_VEL = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
LEN_2 = 2
LEN_4 = 4
TORQUE_ON, TORQUE_OFF = 1, 0
OP_MODE_POSITION = 3
TICK_MAX = 4095
DXL_PROFILE_VEL_UNIT_RPM = 0.229
DXL_PRESENT_CURRENT_UNIT_MA = 2.69


def signed32(x: int) -> int:
    x &= 0xFFFFFFFF
    if x & 0x80000000:
        return -((~x & 0xFFFFFFFF) + 1)
    return x


def signed16(x: int) -> int:
    x &= 0xFFFF
    if x & 0x8000:
        return -((~x & 0xFFFF) + 1)
    return x


def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def int_to_le4(x: int) -> List[int]:
    x &= 0xFFFFFFFF
    return [x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF]


def deg_to_tick_0_360(deg: float) -> int:
    deg = clamp_float(deg, 0.0, 360.0)
    tick = int(round(deg * (TICK_MAX / 360.0)))
    return clamp_int(tick, 0, TICK_MAX)


def tick_to_deg_0_360(tick: int, direction: int = +1) -> float:
    tick = signed32(int(tick))
    tick = clamp_int(tick, 0, TICK_MAX)
    if int(direction) < 0:
        tick = TICK_MAX - tick
    return float(tick) * (360.0 / float(TICK_MAX))


@dataclass(frozen=True)
class JointProfile:
    profile_vel: int
    profile_acc: int


@dataclass(frozen=True)
class JointConstraintDeg:
    min_deg: float
    max_deg: float


@dataclass(frozen=True)
class DxlConfig:
    device_name: str = "/dev/ttyUSB0"
    baudrate: int = 57600
    protocol_version: float = 2.0
    id1_slider: int = 1
    id2_base: int = 2
    id3_seg1: int = 3
    id4_seg2: int = 4


def default_joint_profiles(cfg: DxlConfig) -> Dict[int, JointProfile]:
    return {
        cfg.id1_slider: JointProfile(profile_vel=150, profile_acc=5),
        cfg.id2_base: JointProfile(profile_vel=150, profile_acc=5),
        cfg.id3_seg1: JointProfile(profile_vel=120, profile_acc=5),
        cfg.id4_seg2: JointProfile(profile_vel=120, profile_acc=5),
    }


def estimate_ideal_sim_rates(
    mapping_cfg: proto.SimMappingConfig,
    *,
    cfg: Optional[DxlConfig] = None,
) -> Tuple[float, float, float]:
    cfg = cfg if cfg is not None else DxlConfig()
    profiles = default_joint_profiles(cfg)

    def _profile_deg_s(raw: int) -> float:
        return float(raw) * float(DXL_PROFILE_VEL_UNIT_RPM) * 6.0

    slider_deg_s = _profile_deg_s(profiles[cfg.id1_slider].profile_vel)
    roll_deg_s = _profile_deg_s(profiles[cfg.id2_base].profile_vel)
    seg1_deg_s = _profile_deg_s(profiles[cfg.id3_seg1].profile_vel)
    seg2_deg_s = _profile_deg_s(profiles[cfg.id4_seg2].profile_vel)

    linear_m_per_u = (float(mapping_cfg.linear_q_max_m) - float(mapping_cfg.linear_q_min_m)) / max(
        1e-9, float(mapping_cfg.linear_u_max) - float(mapping_cfg.linear_u_min)
    )
    roll_rad_per_u = (float(mapping_cfg.roll_q_max_rad) - float(mapping_cfg.roll_q_min_rad)) / max(
        1e-9, float(mapping_cfg.roll_u_max) - float(mapping_cfg.roll_u_min)
    )
    seg1_rad_per_u = (float(mapping_cfg.seg1_q_max_rad) - float(mapping_cfg.seg1_q_min_rad)) / max(
        1e-9, float(mapping_cfg.seg_u_max) - float(mapping_cfg.seg_u_min)
    )
    seg2_rad_per_u = (float(mapping_cfg.seg2_q_max_rad) - float(mapping_cfg.seg2_q_min_rad)) / max(
        1e-9, float(mapping_cfg.seg_u_max) - float(mapping_cfg.seg_u_min)
    )
    linear_m_s = abs(slider_deg_s * linear_m_per_u)
    roll_rad_s = abs(roll_deg_s * roll_rad_per_u)
    bend1_rad_s = abs(seg1_deg_s * seg1_rad_per_u)
    bend2_rad_s = abs(seg2_deg_s * seg2_rad_per_u)
    bend_rad_s = min(bend1_rad_s, bend2_rad_s)
    return float(linear_m_s), float(roll_rad_s), float(bend_rad_s)


class Dynamixel4dofDriver:
    """Independent hardware driver (no ControlLegacy dependency)."""

    def __init__(self, cfg: DxlConfig) -> None:
        if PortHandler is None or PacketHandler is None or GroupSyncWrite is None or GroupSyncRead is None:
            raise RuntimeError("dynamixel_sdk is not installed.")

        self.cfg = cfg
        self.ids = [cfg.id1_slider, cfg.id2_base, cfg.id3_seg1, cfg.id4_seg2]
        self.direction: Dict[int, int] = {dxl_id: +1 for dxl_id in self.ids}
        self.profiles: Dict[int, JointProfile] = default_joint_profiles(cfg)
        self.constraints_deg: Dict[int, JointConstraintDeg] = {
            cfg.id1_slider: JointConstraintDeg(0.0, 360.0),
            cfg.id2_base: JointConstraintDeg(0.0, 360.0),
            cfg.id3_seg1: JointConstraintDeg(0.0, 360.0),
            cfg.id4_seg2: JointConstraintDeg(0.0, 360.0),
        }

        self.port = PortHandler(cfg.device_name)
        self.packet = PacketHandler(cfg.protocol_version)
        self.sync_write_pos = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_4)
        self.sync_read_pos = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_POSITION, LEN_4)

    def _write1(self, dxl_id: int, addr: int, value: int) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, dxl_id, addr, value)
        if comm != 0:
            raise RuntimeError(f"[ID {dxl_id}] write1 comm fail: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {dxl_id}] write1 dxl error: {self.packet.getRxPacketError(err)}")

    def _write4(self, dxl_id: int, addr: int, value: int) -> None:
        comm, err = self.packet.write4ByteTxRx(self.port, dxl_id, addr, value)
        if comm != 0:
            raise RuntimeError(f"[ID {dxl_id}] write4 comm fail: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            raise RuntimeError(f"[ID {dxl_id}] write4 dxl error: {self.packet.getRxPacketError(err)}")

    def open(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port: {self.cfg.device_name}")
        if not self.port.setBaudRate(self.cfg.baudrate):
            raise RuntimeError(f"Failed to set baudrate: {self.cfg.baudrate}")
        self.sync_read_pos.clearParam()
        for dxl_id in self.ids:
            if not self.sync_read_pos.addParam(dxl_id):
                raise RuntimeError(f"sync_read addParam failed: ID={dxl_id}")

    def close(self) -> None:
        try:
            self.port.closePort()
        except Exception:
            pass

    def torque_off_all(self) -> None:
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_OFF)

    def torque_on_all(self) -> None:
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ON)

    def set_operating_modes(self) -> None:
        self.torque_off_all()
        for dxl_id in self.ids:
            self._write1(dxl_id, ADDR_OPERATING_MODE, OP_MODE_POSITION)

    def set_profiles(self) -> None:
        for dxl_id, prof in self.profiles.items():
            self._write4(dxl_id, ADDR_PROFILE_VEL, prof.profile_vel)
            self._write4(dxl_id, ADDR_PROFILE_ACCEL, prof.profile_acc)

    def _apply_constraint_deg(self, dxl_id: int, deg: float) -> float:
        c = self.constraints_deg[dxl_id]
        return clamp_float(deg, c.min_deg, c.max_deg)

    def deg_to_goal_tick(self, dxl_id: int, deg: float) -> int:
        deg_c = self._apply_constraint_deg(dxl_id, deg)
        tick = deg_to_tick_0_360(deg_c)
        if self.direction.get(dxl_id, +1) == -1:
            tick = TICK_MAX - tick
        return clamp_int(tick, 0, TICK_MAX)

    def get_present_positions(self) -> Dict[int, int]:
        comm = self.sync_read_pos.txRxPacket()
        if comm != 0:
            raise RuntimeError(f"sync_read comm fail: {self.packet.getTxRxResult(comm)}")
        out: Dict[int, int] = {}
        for dxl_id in self.ids:
            if not self.sync_read_pos.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_4):
                raise RuntimeError(f"present pos unavailable: ID={dxl_id}")
            raw = self.sync_read_pos.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_4)
            out[dxl_id] = signed32(raw)
        return out

    def get_present_currents_ma(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for dxl_id in self.ids:
            raw, comm, err = self.packet.read2ByteTxRx(self.port, dxl_id, ADDR_PRESENT_CURRENT)
            if comm != 0:
                raise RuntimeError(f"[ID {dxl_id}] read current comm fail: {self.packet.getTxRxResult(comm)}")
            if err != 0:
                raise RuntimeError(f"[ID {dxl_id}] read current dxl error: {self.packet.getRxPacketError(err)}")
            out[dxl_id] = float(signed16(int(raw))) * float(DXL_PRESENT_CURRENT_UNIT_MA)
        return out

    def sync_set_goal_positions(self, goals_tick: Dict[int, int]) -> None:
        self.sync_write_pos.clearParam()
        for dxl_id, tick in goals_tick.items():
            if not self.sync_write_pos.addParam(dxl_id, int_to_le4(clamp_int(tick, 0, TICK_MAX))):
                raise RuntimeError(f"sync_write addParam failed: ID={dxl_id}")
        comm = self.sync_write_pos.txPacket()
        if comm != 0:
            raise RuntimeError(f"sync_write comm fail: {self.packet.getTxRxResult(comm)}")

    def command_4dof_deg(self, slider_deg: float, base_deg: float, seg1_deg: float, seg2_deg: float) -> None:
        goals = {
            self.cfg.id1_slider: self.deg_to_goal_tick(self.cfg.id1_slider, slider_deg),
            self.cfg.id2_base: self.deg_to_goal_tick(self.cfg.id2_base, base_deg),
            self.cfg.id3_seg1: self.deg_to_goal_tick(self.cfg.id3_seg1, seg1_deg),
            self.cfg.id4_seg2: self.deg_to_goal_tick(self.cfg.id4_seg2, seg2_deg),
        }
        self.sync_set_goal_positions(goals)

    def go_mid_pose(self) -> None:
        self.command_4dof_deg(180.0, 180.0, 180.0, 180.0)


def load_hardware(
    device: str,
    *,
    hardware_cfg: HardwareConfig | None = None,
) -> Tuple[Any, Dict[int, int]]:
    """Return (hardware_driver, direction_by_id)."""
    hw = Dynamixel4dofDriver(DxlConfig(device_name=device))
    if hardware_cfg is not None:
        motor_dir = tuple(int(v) for v in getattr(hardware_cfg, "motor_direction"))
        raw = {
            hw.cfg.id1_slider: motor_dir[0],
            hw.cfg.id2_base: motor_dir[1],
            hw.cfg.id3_seg1: motor_dir[2],
            hw.cfg.id4_seg2: motor_dir[3],
        }
        for k, v in raw.items():
            hw.direction[k] = -1 if int(v) < 0 else 1
    return hw, dict(hw.direction)
