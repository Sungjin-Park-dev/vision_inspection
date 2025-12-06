#!/usr/bin/env python3
"""
Kinematics Utilities for Forward Kinematics and Quaternion Operations

Provides GPU-accelerated (Numba JIT when available) forward kinematics
for UR20 robot and quaternion/rotation matrix conversion utilities.

Functions consolidated from:
- config.py: quaternion_to_rotation_matrix
- fk_gtsp_gpu_claude2.py: FK and quaternion batch operations
"""

import math
from typing import Tuple

import numpy as np

# ============================================================================
# Numba Setup (optional acceleration)
# ============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco


# ============================================================================
# Quaternion & Rotation Utilities
# ============================================================================

def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        quat: Quaternion in (w, x, y, z) format

    Returns:
        3x3 rotation matrix

    Example:
        >>> quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        >>> R = quaternion_to_rotation_matrix(quat)
        >>> print(R)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def rot_to_quat_batch_np(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to quaternions (NumPy version)

    Args:
        R: (N, 3, 3) array of rotation matrices

    Returns:
        Q: (N, 4) array of quaternions in (w, x, y, z) format

    Note:
        Pure NumPy implementation without JIT compilation.
        Use rot_to_quat_batch() for automatic Numba acceleration.
    """
    N = R.shape[0]
    Q = np.empty((N, 4), dtype=np.float64)
    for k in range(N):
        r = R[k]
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0.0:
            t = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * t
            qx = (r[2, 1] - r[1, 2]) / t
            qy = (r[0, 2] - r[2, 0]) / t
            qz = (r[1, 0] - r[0, 1]) / t
        else:
            if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                t = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                qw = (r[2, 1] - r[1, 2]) / t
                qx = 0.25 * t
                qy = (r[0, 1] + r[1, 0]) / t
                qz = (r[0, 2] + r[2, 0]) / t
            elif r[1, 1] > r[2, 2]:
                t = math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2]) * 2.0
                qw = (r[0, 2] - r[2, 0]) / t
                qx = (r[0, 1] + r[1, 0]) / t
                qy = 0.25 * t
                qz = (r[1, 2] + r[2, 1]) / t
            else:
                t = math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2]) * 2.0
                qw = (r[1, 0] - r[0, 1]) / t
                qx = (r[0, 2] + r[2, 0]) / t
                qy = (r[1, 2] + r[2, 1]) / t
                qz = 0.25 * t
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        Q[k, 0] = qw / n
        Q[k, 1] = qx / n
        Q[k, 2] = qy / n
        Q[k, 3] = qz / n
    return Q


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def rot_to_quat_batch(R):
        """
        Convert rotation matrices to quaternions (Numba JIT version)

        Args:
            R: (N, 3, 3) array of rotation matrices

        Returns:
            Q: (N, 4) array of quaternions in (w, x, y, z) format

        Note:
            Numba-compiled for performance. Falls back to NumPy if Numba unavailable.
        """
        N = R.shape[0]
        Q = np.empty((N, 4), np.float64)
        for k in range(N):
            r = R[k]
            tr = r[0, 0] + r[1, 1] + r[2, 2]
            if tr > 0.0:
                t = math.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * t
                qx = (r[2, 1] - r[1, 2]) / t
                qy = (r[0, 2] - r[2, 0]) / t
                qz = (r[1, 0] - r[0, 1]) / t
            else:
                if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
                    t = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                    qw = (r[2, 1] - r[1, 2]) / t
                    qx = 0.25 * t
                    qy = (r[0, 1] + r[1, 0]) / t
                    qz = (r[0, 2] + r[2, 0]) / t
                elif r[1, 1] > r[2, 2]:
                    t = math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2]) * 2.0
                    qw = (r[0, 2] - r[2, 0]) / t
                    qx = (r[0, 1] + r[1, 0]) / t
                    qy = 0.25 * t
                    qz = (r[1, 2] + r[2, 1]) / t
                else:
                    t = math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2]) * 2.0
                    qw = (r[1, 0] - r[0, 1]) / t
                    qx = (r[0, 2] + r[2, 0]) / t
                    qy = (r[1, 2] + r[2, 1]) / t
                    qz = 0.25 * t
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            Q[k, 0] = qw / n
            Q[k, 1] = qx / n
            Q[k, 2] = qy / n
            Q[k, 3] = qz / n
        return Q
else:
    def rot_to_quat_batch(R):
        """Fallback to NumPy version if Numba not available"""
        return rot_to_quat_batch_np(R)


def rotation_angle_from_quats(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """
    Compute rotation angle between quaternion arrays

    Args:
        Qa: (..., 4) array of quaternions
        Qb: (..., 4) array of quaternions (same shape as Qa)

    Returns:
        angles: (...,) array of rotation angles in radians

    Formula:
        angle = 2 * arccos(|<qa, qb>|)

    Example:
        >>> Qa = np.array([[1, 0, 0, 0]])  # Identity
        >>> Qb = np.array([[1, 0, 0, 0]])  # Identity
        >>> angle = rotation_angle_from_quats(Qa, Qb)
        >>> print(angle)
        [0.]
    """
    dots = np.abs(np.sum(Qa * Qb, axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def rot_angle_ignore_tool_yaw(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """
    Compute rotation angle between two rotations ignoring tool yaw (tilt only)

    Args:
        Ra: (..., 3, 3) array of rotation matrices
        Rb: (..., 3, 3) array of rotation matrices

    Returns:
        angles: (...,) array of angles between tool Z-axes in radians

    Note:
        Computes angle between tool Z-axes (last column of rotation matrices).
        Useful for tool orientation comparison while ignoring spin around tool axis.

    Example:
        >>> Ra = np.eye(3)[None, ...]  # Identity rotation
        >>> Rb = np.eye(3)[None, ...]  # Identity rotation
        >>> angle = rot_angle_ignore_tool_yaw(Ra, Rb)
        >>> print(angle)
        [0.]
    """
    za = Ra[..., :, 2]   # Extract Z-axis (tool direction)
    zb = Rb[..., :, 2]
    dots = np.sum(za * zb, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


# ============================================================================
# Forward Kinematics (UR20-specific DH parameters)
# ============================================================================

# UR20 DH parameters (Denavit-Hartenberg convention)
_A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0], dtype=np.float64)
_D = np.array([0.1807, 0.0, 0.0, 0.163941, 0.1157, 0.0922], dtype=np.float64)
_AL = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0], dtype=np.float64)


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _fk_single(q, tool_z: float):
        """
        Compute forward kinematics for single configuration (Numba JIT)

        Args:
            q: Joint angles (6,) in radians
            tool_z: Tool offset along Z-axis in meters

        Returns:
            T: 4x4 transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        for i in range(6):
            th = q[i]
            ca = math.cos(_AL[i]); sa = math.sin(_AL[i])
            ct = math.cos(th);     st = math.sin(th)
            # DH transformation matrix A_i
            A = np.empty((4,4), np.float64)
            A[0,0]=ct;   A[0,1]=-st*ca; A[0,2]= st*sa; A[0,3]=_A[i]*ct
            A[1,0]=st;   A[1,1]= ct*ca; A[1,2]=-ct*sa; A[1,3]=_A[i]*st
            A[2,0]=0.0;  A[2,1]= sa;    A[2,2]= ca;    A[2,3]=_D[i]
            A[3,0]=0.0;  A[3,1]= 0.0;   A[3,2]= 0.0;   A[3,3]=1.0
            T = T @ A
        # Apply tool offset along Z-axis
        if tool_z != 0.0:
            T[0,3] += T[0,2]*tool_z
            T[1,3] += T[1,2]*tool_z
            T[2,3] += T[2,2]*tool_z
        return T

    @njit(cache=True, fastmath=True, parallel=False)
    def fk_batch(qs, tool_z: float):
        """
        Batch forward kinematics for multiple configurations (Numba JIT)

        Args:
            qs: (N, 6) array of joint angles in radians
            tool_z: Tool offset along Z-axis in meters

        Returns:
            R: (N, 3, 3) array of rotation matrices
            p: (N, 3) array of positions in meters
        """
        N = qs.shape[0]
        R = np.empty((N,3,3), np.float64)
        p = np.empty((N,3),   np.float64)
        for k in range(N):
            T = _fk_single(qs[k], tool_z)
            R[k,0,0]=T[0,0]; R[k,0,1]=T[0,1]; R[k,0,2]=T[0,2]
            R[k,1,0]=T[1,0]; R[k,1,1]=T[1,1]; R[k,1,2]=T[1,2]
            R[k,2,0]=T[2,0]; R[k,2,1]=T[2,1]; R[k,2,2]=T[2,2]
            p[k,0]=T[0,3]; p[k,1]=T[1,3]; p[k,2]=T[2,3]
        return R, p
else:
    def _fk_single_np(q: np.ndarray, tool_z: float) -> np.ndarray:
        """
        Compute forward kinematics for single configuration (NumPy version)

        Args:
            q: Joint angles (6,) in radians
            tool_z: Tool offset along Z-axis in meters

        Returns:
            T: 4x4 transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        for i in range(6):
            th = q[i]
            ca = np.cos(_AL[i]); sa = np.sin(_AL[i])
            ct = np.cos(th);     st = np.sin(th)
            A = np.array([
                [ct,   -st*ca,  st*sa, _A[i]*ct],
                [st,    ct*ca, -ct*sa, _A[i]*st],
                [0.0,     sa,      ca, _D[i]],
                [0.0,    0.0,     0.0, 1.0]
            ], dtype=np.float64)
            T = T @ A
        if tool_z != 0.0:
            T[:3, 3] += T[:3, 2] * tool_z
        return T

    def fk_batch(qs: np.ndarray, tool_z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch forward kinematics for multiple configurations (NumPy version)

        Args:
            qs: (N, 6) array of joint angles in radians
            tool_z: Tool offset along Z-axis in meters

        Returns:
            R: (N, 3, 3) array of rotation matrices
            p: (N, 3) array of positions in meters
        """
        N = qs.shape[0]
        R = np.empty((N,3,3), dtype=np.float64)
        p = np.empty((N,3),   dtype=np.float64)
        for k in range(N):
            T = _fk_single_np(qs[k], tool_z)
            R[k] = T[:3,:3]
            p[k] = T[:3, 3]
        return R, p
