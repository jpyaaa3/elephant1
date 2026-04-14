from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (x, y, z, w)


@dataclass(frozen=True)
class Pose:
    p: Vec3
    q: Quat

    def to_dict(self) -> dict:
        return {"p": [float(v) for v in self.p], "q": [float(v) for v in self.q]}


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def vec_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec_norm(v: Vec3) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_normalize(v: Vec3) -> Vec3:
    n = vec_norm(v)
    if n <= 0.0:
        raise ValueError("zero-length vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def quat_conjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(v * v for v in q))
    if n <= 0.0:
        raise ValueError("zero-length quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_multiply(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_dot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    qv = (v[0], v[1], v[2], 0.0)
    qr = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return (qr[0], qr[1], qr[2])


def quat_to_matrix(q: Quat) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def pose_inverse(pose: Pose) -> Pose:
    q_inv = quat_conjugate(quat_normalize(pose.q))
    p_inv = quat_rotate(q_inv, vec_scale(pose.p, -1.0))
    return Pose(p=p_inv, q=q_inv)


def pose_multiply(a: Pose, b: Pose) -> Pose:
    return Pose(
        p=vec_add(a.p, quat_rotate(a.q, b.p)),
        q=quat_normalize(quat_multiply(a.q, b.q)),
    )


def average_quaternions(quaternions: Sequence[Quat]) -> Quat:
    if not quaternions:
        raise ValueError("no quaternions to average")
    ref = quat_normalize(quaternions[0])
    accum = [0.0, 0.0, 0.0, 0.0]
    for q_raw in quaternions:
        q = quat_normalize(q_raw)
        if sum(ref[i] * q[i] for i in range(4)) < 0.0:
            q = (-q[0], -q[1], -q[2], -q[3])
        for i in range(4):
            accum[i] += q[i]
    return quat_normalize((accum[0], accum[1], accum[2], accum[3]))


def average_vectors(vectors: Sequence[Vec3]) -> Vec3:
    if not vectors:
        raise ValueError("no vectors to average")
    inv = 1.0 / float(len(vectors))
    return (
        sum(v[0] for v in vectors) * inv,
        sum(v[1] for v in vectors) * inv,
        sum(v[2] for v in vectors) * inv,
    )


def quaternion_from_axes(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Quat:
    x_axis = vec_normalize(x_axis)
    y_axis = vec_normalize(y_axis)
    z_axis = vec_normalize(z_axis)
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return quat_normalize((qx, qy, qz, qw))


def pose_from_face(center: Vec3, normal: Vec3, x_hint: Vec3) -> Pose:
    z_axis = vec_normalize(normal)
    x_axis_projected = vec_sub(x_hint, vec_scale(z_axis, vec_dot(x_hint, z_axis)))
    x_axis = vec_normalize(x_axis_projected)
    y_axis = vec_cross(z_axis, x_axis)
    return Pose(p=center, q=quaternion_from_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis))


def as_vec3(values: Iterable[float]) -> Vec3:
    items = tuple(float(v) for v in values)
    if len(items) != 3:
        raise ValueError("expected 3 values")
    return items  # type: ignore[return-value]


def pose_from_axes(origin: Vec3, x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Pose:
    return Pose(p=origin, q=quaternion_from_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis))


def pose_to_matrix4(pose: Pose) -> Tuple[Tuple[float, float, float, float], ...]:
    r = quat_to_matrix(pose.q)
    return (
        (r[0][0], r[0][1], r[0][2], pose.p[0]),
        (r[1][0], r[1][1], r[1][2], pose.p[1]),
        (r[2][0], r[2][1], r[2][2], pose.p[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
