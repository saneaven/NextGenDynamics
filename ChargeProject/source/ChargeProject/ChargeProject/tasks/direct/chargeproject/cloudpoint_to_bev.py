import torch
from typing import Optional, Sequence, Tuple

# -------------------------------
# Geometry helpers
# -------------------------------

def quaternion_to_rotation_matrix_wxyz(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to rotation matrices.
    Args:
        quaternion_wxyz: (batch_size, 4) in [w, x, y, z] order.
    Returns:
        rotation_matrix: (batch_size, 3, 3)
    """
    w, x, y, z = quaternion_wxyz.unbind(-1)
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotation_matrix = torch.empty(
        (quaternion_wxyz.shape[0], 3, 3),
        device=quaternion_wxyz.device,
        dtype=quaternion_wxyz.dtype
    )

    rotation_matrix[:, 0, 0] = 1 - 2 * (y2 + z2)
    rotation_matrix[:, 0, 1] = 2 * (xy - wz)
    rotation_matrix[:, 0, 2] = 2 * (xz + wy)

    rotation_matrix[:, 1, 0] = 2 * (xy + wz)
    rotation_matrix[:, 1, 1] = 1 - 2 * (x2 + z2)
    rotation_matrix[:, 1, 2] = 2 * (yz - wx)

    rotation_matrix[:, 2, 0] = 2 * (xz - wy)
    rotation_matrix[:, 2, 1] = 2 * (yz + wx)
    rotation_matrix[:, 2, 2] = 1 - 2 * (x2 + y2)
    return rotation_matrix


def transform_world_to_ego(
    world_points: torch.Tensor,                 # (batch_size, num_points, 3)
    sensor_position_world: torch.Tensor,        # (batch_size, 3)
    sensor_quaternion_world_wxyz: torch.Tensor  # (batch_size, 4) [w,x,y,z]
) -> torch.Tensor:
    """
    Transform points from world frame into the ego (sensor/base) frame.
    Returns:
        ego_points: (batch_size, num_points, 3)
    """
    rotation_world_to_sensor = quaternion_to_rotation_matrix_wxyz(
        sensor_quaternion_world_wxyz
    )  # (B, 3, 3)

    # Translate, then rotate by R^T to go into sensor frame
    relative_points = world_points - sensor_position_world[:, None, :]   # (B, P, 3)
    ego_points = torch.einsum(
        "bij,bpj->bpi", rotation_world_to_sensor.transpose(1, 2), relative_points
    )
    return ego_points

# -------------------------------
# BEV builder
# -------------------------------

@torch.no_grad()
def build_bev(
    ego_points: torch.Tensor,                          # (batch_size, num_points, 3) -> [x, y, z]
    intensities: Optional[torch.Tensor] = None,        # (batch_size, num_points) or None
    x_limits_meters: Tuple[float, float] = (-12.8, 12.8),
    y_limits_meters: Tuple[float, float] = (-12.8, 12.8),
    z_limits_meters: Tuple[float, float] = (-3.0, 2.0),
    resolution_meters: float = 0.40, # grid will be 64x64 for these limits
    density_normalization: float = 10.0,
    channels: Sequence[str] = ("max_height", "mean_height", "mean_intensity", "density"),
) -> torch.Tensor:
    """
    Create a BEV tensor suitable for 2D CNN input.

    Args:
        ego_points: (B, P, 3) points in ego frame [x, y, z].
        intensities: (B, P) optional reflectance/intensity values in [0, 1].
        x_limits_meters, y_limits_meters, z_limits_meters: (min, max) per axis.
        resolution_meters: grid cell size in meters.
        density_normalization: divisor to scale density to ~[0, 1].
        channels: list from {"max_height", "mean_height", "mean_intensity", "density"}.

    Returns:
        bev: (B, C, H, W) float32, channels in the provided order,
             laid out as channels_last memory format for better Conv2d kernels.
    """
    device = ego_points.device
    dtype = ego_points.dtype
    batch_size, num_points, _ = ego_points.shape

    x_min, x_max = x_limits_meters
    y_min, y_max = y_limits_meters
    z_min, z_max = z_limits_meters

    width_cells  = int((x_max - x_min) / resolution_meters)   # along +x (columns)
    height_cells = int((y_max - y_min) / resolution_meters)   # along +y (rows)
    total_cells_per_sample = height_cells * width_cells
    total_cells_all_batches = batch_size * total_cells_per_sample

    x_values, y_values, z_values = ego_points.unbind(-1)

    # Mask points that fall within the configured region of interest
    valid_mask = (
        (x_values >= x_min) & (x_values < x_max) &
        (y_values >= y_min) & (y_values < y_max) &
        (z_values >= z_min) & (z_values <= z_max)
    )

    # Map (x, y) to integer grid indices (column, row)
    grid_x = torch.clamp(((x_values - x_min) / resolution_meters).floor().long(), 0, width_cells  - 1)
    grid_y = torch.clamp(((y_values - y_min) / resolution_meters).floor().long(), 0, height_cells - 1)

    # Keep only valid points
    batch_indices, point_indices = torch.where(valid_mask)
    if batch_indices.numel() == 0:
        output = torch.zeros(
            (batch_size, len(channels), height_cells, width_cells),
            device=device, dtype=torch.float32
        )
        return output.contiguous(memory_format=torch.channels_last)

    grid_x_valid = grid_x[batch_indices, point_indices]
    grid_y_valid = grid_y[batch_indices, point_indices]
    linear_index_within_sample = grid_y_valid * width_cells + grid_x_valid              # (N_valid,)

    z_valid = z_values[batch_indices, point_indices]

    # Convert (sample_id, cell_id) into a single global linear index for vectorized scatter
    global_linear_index = linear_index_within_sample + batch_indices * total_cells_per_sample
    global_size = total_cells_all_batches

    channel_outputs = []
    z_range_span = max(1e-6, (z_max - z_min))  # safe normalization denominator

    for channel_name in channels:
        if channel_name == "max_height":
            # Per-cell max z, normalized to [0, 1] using [z_min, z_max]
            max_buffer = torch.full((global_size,), z_min, device=device, dtype=dtype)
            max_buffer.scatter_reduce_(0, global_linear_index, z_valid, reduce="amax")
            max_height_channel = max_buffer.view(batch_size, height_cells, width_cells)
            max_height_channel = (max_height_channel - z_min) / z_range_span
            channel_outputs.append(max_height_channel)

        elif channel_name == "mean_height":
            # Per-cell mean z; empty cells become z_min; then normalize to [0, 1]
            sum_z = torch.zeros((global_size,), device=device, dtype=dtype)
            count_points = torch.zeros((global_size,), device=device, dtype=dtype)
            sum_z.scatter_add_(0, global_linear_index, z_valid)
            count_points.scatter_add_(0, global_linear_index, torch.ones_like(z_valid))

            mean_z = sum_z / torch.clamp(count_points, min=1.0)
            mean_z = torch.where(count_points > 0, mean_z, torch.full_like(mean_z, z_min))
            mean_height_channel = (mean_z.view(batch_size, height_cells, width_cells) - z_min) / z_range_span
            channel_outputs.append(mean_height_channel)

        elif channel_name == "mean_intensity":
            if intensities is None:
                intensity_values = torch.zeros((batch_size, num_points), device=device, dtype=dtype)
            else:
                intensity_values = intensities

            intensity_valid = intensity_values[batch_indices, point_indices]

            # Per-cell mean intensity in [0, 1]
            sum_intensity = torch.zeros((global_size,), device=device, dtype=dtype)
            count_points = torch.zeros((global_size,), device=device, dtype=dtype)
            sum_intensity.scatter_add_(0, global_linear_index, intensity_valid)
            count_points.scatter_add_(0, global_linear_index, torch.ones_like(intensity_valid))
            mean_intensity_channel = (sum_intensity / torch.clamp(count_points, min=1.0)).view(
                batch_size, height_cells, width_cells
            )
            channel_outputs.append(mean_intensity_channel)

        elif channel_name == "density":
            # Per-cell point counts scaled to ~[0,1]
            counts = torch.bincount(global_linear_index, minlength=global_size).to(dtype)
            density_channel = (counts / max(1.0, density_normalization)).clamp(max=1.0).view(
                batch_size, height_cells, width_cells
            )
            channel_outputs.append(density_channel)

        else:
            raise ValueError(f"Unknown channel name: {channel_name}")

    bev_tensor = torch.stack(channel_outputs, dim=1).to(torch.float32)  # (B, C, H, W)
    # Keep NCHW semantics but set NHWC (channels_last) strides for faster Conv2d kernels
    bev_tensor = bev_tensor.contiguous(memory_format=torch.channels_last)
    return bev_tensor
