from __future__ import annotations

from collections.abc import Sequence

import torch


class BEVWorkspaceBuilder:
    """Workspace that reuses buffers for BEV rasterization."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_points: int,
        device: torch.device | str,
        dtype: torch.dtype,
        x_limits_meters: tuple[float, float] = (-12.8, 12.8),
        y_limits_meters: tuple[float, float] = (-12.8, 12.8),
        z_limits_meters: tuple[float, float] = (-3.0, 2.0),
        resolution_meters: float = 0.40,
        density_normalization: float = 10.0,
        channels: Sequence[str] = ("max_height", "mean_height", "density"),
    ) -> None:
        self.x_limits_meters = x_limits_meters
        self.y_limits_meters = y_limits_meters
        self.z_limits_meters = z_limits_meters
        self.resolution_meters = float(resolution_meters)
        self.density_normalization = float(density_normalization)
        self.channels = tuple(channels)

        self.width_cells = int((x_limits_meters[1] - x_limits_meters[0]) / self.resolution_meters)
        self.height_cells = int((y_limits_meters[1] - y_limits_meters[0]) / self.resolution_meters)
        self.cells_per_sample = self.width_cells * self.height_cells

        self.batch_size = int(batch_size)
        self.num_points = int(num_points)
        self.device = torch.device(device)
        self.dtype = dtype
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        b = self.batch_size
        p = self.num_points
        c = self.cells_per_sample

        self.batch_offsets = (
            torch.arange(b, device=self.device, dtype=torch.long).view(b, 1) * self.cells_per_sample
        )

        self.grid_x = torch.zeros((b, p), device=self.device, dtype=torch.long)
        self.linear_idx = torch.zeros((b, p), device=self.device, dtype=torch.long)

        self.valid_mask = torch.zeros((b, p), device=self.device, dtype=torch.bool)
        self.valid_mask_tmp = torch.zeros((b, p), device=self.device, dtype=torch.bool)

        self.tmp_x = torch.zeros((b, p), device=self.device, dtype=self.dtype)
        self.tmp_y = torch.zeros((b, p), device=self.device, dtype=self.dtype)

        self.max_buf = torch.zeros((b * c,), device=self.device, dtype=self.dtype)
        self.sum_buf = torch.zeros((b * c,), device=self.device, dtype=self.dtype)
        self.count_buf = torch.zeros((b * c,), device=self.device, dtype=self.dtype)
        self.count_nonzero = torch.zeros((b * c,), device=self.device, dtype=torch.bool)

        # Reused for world->ego transform.
        self.ego_points = torch.zeros((b, p, 3), device=self.device, dtype=self.dtype)
        self.relative_points = torch.zeros((b, p, 3), device=self.device, dtype=self.dtype)
        self.rotation_matrix = torch.zeros((b, 3, 3), device=self.device, dtype=self.dtype)

    def rebuild(self, *, batch_size: int, num_points: int, device: torch.device | str, dtype: torch.dtype) -> None:
        self.batch_size = int(batch_size)
        self.num_points = int(num_points)
        self.device = torch.device(device)
        self.dtype = dtype
        self._allocate_buffers()

    def ensure_shape(self, *, batch_size: int, num_points: int, device: torch.device | str, dtype: torch.dtype) -> None:
        if (
            int(batch_size) != self.batch_size
            or int(num_points) != self.num_points
            or torch.device(device) != self.device
            or dtype != self.dtype
        ):
            self.rebuild(batch_size=batch_size, num_points=num_points, device=device, dtype=dtype)


@torch.no_grad()
def build_bev_inplace(
    builder: BEVWorkspaceBuilder,
    ego_points: torch.Tensor,
    out: torch.Tensor,
    *,
    density_normalization: float | None = None,
) -> torch.Tensor:
    """Rasterize ego-frame points into BEV channels using preallocated workspace."""
    batch_size, num_points, _ = ego_points.shape
    builder.ensure_shape(batch_size=batch_size, num_points=num_points, device=ego_points.device, dtype=ego_points.dtype)

    x_min, x_max = builder.x_limits_meters
    y_min, y_max = builder.y_limits_meters
    z_min, z_max = builder.z_limits_meters

    x_values, y_values, z_values = ego_points.unbind(-1)

    torch.ge(x_values, x_min, out=builder.valid_mask)
    torch.lt(x_values, x_max, out=builder.valid_mask_tmp)
    builder.valid_mask.logical_and_(builder.valid_mask_tmp)
    torch.ge(y_values, y_min, out=builder.valid_mask_tmp)
    builder.valid_mask.logical_and_(builder.valid_mask_tmp)
    torch.lt(y_values, y_max, out=builder.valid_mask_tmp)
    builder.valid_mask.logical_and_(builder.valid_mask_tmp)
    torch.ge(z_values, z_min, out=builder.valid_mask_tmp)
    builder.valid_mask.logical_and_(builder.valid_mask_tmp)
    torch.le(z_values, z_max, out=builder.valid_mask_tmp)
    builder.valid_mask.logical_and_(builder.valid_mask_tmp)

    inv_resolution = 1.0 / builder.resolution_meters

    builder.tmp_x.copy_(x_values)
    builder.tmp_x.sub_(x_min)
    builder.tmp_x.mul_(inv_resolution)
    builder.tmp_x.floor_()
    builder.grid_x.copy_(builder.tmp_x)
    builder.grid_x.clamp_(0, builder.width_cells - 1)

    builder.tmp_y.copy_(y_values)
    builder.tmp_y.sub_(y_min)
    builder.tmp_y.mul_(inv_resolution)
    builder.tmp_y.floor_()
    builder.linear_idx.copy_(builder.tmp_y)
    builder.linear_idx.clamp_(0, builder.height_cells - 1)
    builder.linear_idx.mul_(builder.width_cells)
    builder.linear_idx.add_(builder.grid_x)
    builder.linear_idx.add_(builder.batch_offsets)

    # Reuse tmp_x/tmp_y as sources for count / sum / max.
    builder.tmp_x.zero_()
    builder.tmp_x.masked_fill_(builder.valid_mask, 1.0)

    flat_idx = builder.linear_idx.view(-1)

    builder.max_buf.fill_(z_min)
    builder.tmp_y.copy_(z_values)
    builder.tmp_y.masked_fill_(~builder.valid_mask, z_min)
    builder.max_buf.scatter_reduce_(0, flat_idx, builder.tmp_y.view(-1), reduce="amax", include_self=True)

    builder.sum_buf.zero_()
    builder.tmp_y.copy_(z_values)
    builder.tmp_y.masked_fill_(~builder.valid_mask, 0.0)
    builder.sum_buf.scatter_add_(0, flat_idx, builder.tmp_y.view(-1))

    builder.count_buf.zero_()
    builder.count_buf.scatter_add_(0, flat_idx, builder.tmp_x.view(-1))

    z_span_inv = 1.0 / max(1e-6, (z_max - z_min))
    density_divisor = max(1.0, density_normalization or builder.density_normalization)

    for channel_index, channel_name in enumerate(builder.channels):
        channel_view = out[:, channel_index, :, :]

        if channel_name == "max_height":
            channel_view.copy_(builder.max_buf.view(builder.batch_size, builder.height_cells, builder.width_cells))
            channel_view.sub_(z_min)
            channel_view.mul_(z_span_inv)
        elif channel_name == "mean_height":
            channel_view.copy_(builder.sum_buf.view(builder.batch_size, builder.height_cells, builder.width_cells))
            channel_view.div_(builder.count_buf.view(builder.batch_size, builder.height_cells, builder.width_cells))
            torch.gt(builder.count_buf, 0.0, out=builder.count_nonzero)
            channel_view.masked_fill_(
                ~builder.count_nonzero.view(builder.batch_size, builder.height_cells, builder.width_cells),
                z_min,
            )
            channel_view.sub_(z_min)
            channel_view.mul_(z_span_inv)
        elif channel_name == "density":
            channel_view.copy_(builder.count_buf.view(builder.batch_size, builder.height_cells, builder.width_cells))
            channel_view.div_(density_divisor)
            channel_view.clamp_(max=1.0)
        else:
            raise ValueError(f"Unknown channel name: {channel_name}")

    return out


@torch.no_grad()
def build_bev(
    ego_points: torch.Tensor,
    x_limits_meters: tuple[float, float] = (-12.8, 12.8),
    y_limits_meters: tuple[float, float] = (-12.8, 12.8),
    z_limits_meters: tuple[float, float] = (-3.0, 2.0),
    resolution_meters: float = 0.40,
    density_normalization: float = 10.0,
    channels: Sequence[str] = ("max_height", "mean_height", "density"),
) -> torch.Tensor:
    """Create a BEV tensor suitable for 2D CNN input."""
    batch_size, num_points, _ = ego_points.shape
    builder = BEVWorkspaceBuilder(
        batch_size=batch_size,
        num_points=num_points,
        device=ego_points.device,
        dtype=ego_points.dtype,
        x_limits_meters=x_limits_meters,
        y_limits_meters=y_limits_meters,
        z_limits_meters=z_limits_meters,
        resolution_meters=resolution_meters,
        density_normalization=density_normalization,
        channels=channels,
    )
    output = torch.empty(
        (batch_size, len(channels), builder.height_cells, builder.width_cells),
        device=ego_points.device,
        dtype=torch.float32,
    )
    build_bev_inplace(builder, ego_points, output, density_normalization=density_normalization)
    return output.contiguous(memory_format=torch.channels_last)

