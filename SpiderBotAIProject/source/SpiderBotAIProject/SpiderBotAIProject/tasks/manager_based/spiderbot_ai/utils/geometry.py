from __future__ import annotations

import torch


def quat_xyzw_to_rotation_matrix_inplace(quat_xyzw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Convert XYZW quaternions into rotation matrices."""
    x, y, z, w = quat_xyzw.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    out[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[:, 0, 1] = 2.0 * (xy - wz)
    out[:, 0, 2] = 2.0 * (xz + wy)

    out[:, 1, 0] = 2.0 * (xy + wz)
    out[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[:, 1, 2] = 2.0 * (yz - wx)

    out[:, 2, 0] = 2.0 * (xz - wy)
    out[:, 2, 1] = 2.0 * (yz + wx)
    out[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def transform_world_to_local_inplace(
    world_points: torch.Tensor,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
    out: torch.Tensor,
    *,
    rotation_world_to_local: torch.Tensor | None = None,
    relative_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform world-frame points into a local frame using XYZW quaternions."""
    if rotation_world_to_local is None:
        rotation_world_to_local = torch.empty(
            (world_points.shape[0], 3, 3),
            device=world_points.device,
            dtype=world_points.dtype,
        )
    if relative_points is None:
        relative_points = torch.empty_like(world_points)

    quat_xyzw_to_rotation_matrix_inplace(frame_quat_w, rotation_world_to_local)
    relative_points.copy_(world_points)
    relative_points.sub_(frame_pos_w[:, None, :])
    torch.bmm(relative_points, rotation_world_to_local.transpose(1, 2), out=out)
    return out
