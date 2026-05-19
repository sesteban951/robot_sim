##
#
# Usefule math utilities for the G1 humanoid simulation and policy.
#
##

import numpy as np
import torch


def gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame from base quaternion (w,x,y,z)."""
    qw, qx, qy, qz = quat
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return np.array([gx, gy, gz], dtype=np.float32)


def get_gravity_orientation_batch(quat: torch.Tensor) -> torch.Tensor:
    """Projected gravity in body frame from base quaternion (w,x,y,z).
    Batched torch version of `gravity_from_quat`. Input (B,4), output (B,3).
    """
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return torch.stack([gx, gy, gz], dim=-1)


def body_to_world(quat: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Rotate a vector from body to world frame: v_world = R(q) @ v_body.

    Uses the closed-form  R(q) v = v + 2w (q_vec x v) + 2 q_vec x (q_vec x v).
    Broadcasts over arbitrary leading dims.

    Args:
        quat:   (..., 4) unit quaternion in (w, x, y, z) order.
        v_body: (..., 3) vector(s) expressed in the body frame.
    Returns:
        v_world: (..., 3) same vector(s) expressed in the world frame.
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_body, axis=-1)
    return v_body + 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)


def world_to_body(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate a vector from world to body frame: v_body = R(q)^T @ v_world.

    Inverse of `body_to_world`. quat is (..., 4) in (w, x, y, z), v_world is (..., 3).
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_world, axis=-1)
    return v_world - 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)
