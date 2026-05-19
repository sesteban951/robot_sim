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
    Batched torch version of `gravity_from_quat`. Input (N,4), output (N,3).
    """
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return torch.stack([gx, gy, gz], dim=-1)
