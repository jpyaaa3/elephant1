#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointLimit:
    linear_min: float
    linear_max: float
    roll_min_deg: float
    roll_max_deg: float
    bend_deg: float

    def roll_min_rad(self) -> float:
        return math.radians(self.roll_min_deg)

    def roll_max_rad(self) -> float:
        return math.radians(self.roll_max_deg)

    def bend_lim_rad(self) -> float:
        return math.radians(self.bend_deg)

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.linear_min, self.roll_min_rad(), -self.bend_lim_rad(), -self.bend_lim_rad()], dtype=float)
        hi = np.array([self.linear_max, self.roll_max_rad(), +self.bend_lim_rad(), +self.bend_lim_rad()], dtype=float)
        return lo, hi
