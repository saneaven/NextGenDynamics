"""
MapManager: Manages exploration/staleness maps and generates nav_data for patrol mode.

Ported from the direct env implementation and adapted for manager-based usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.sensors import RayCaster

from ..utils.cloudpoint_to_bev import build_bev, transform_world_to_ego


@dataclass
class MapManagerOutput:
    """Return type for MapManager.update()."""

    nav_data: torch.Tensor  # (N, 1, nav_dim, nav_dim)
    bev_data: torch.Tensor  # (N, 3, 64, 64)
    height_data: torch.Tensor  # (N, 64, 64)
    far_staleness: torch.Tensor  # (N, 8)
    exploration_bonus: torch.Tensor  # (N,)


class MapManager:
    def __init__(self, config, num_envs, device, height_scanner: RayCaster, lidar_sensor: RayCaster):
        self.device = device
        self.num_envs = num_envs
        self.config = config

        self._height_scanner = height_scanner
        self._lidar_sensor = lidar_sensor

        x = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        y = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(grid_x**2 + grid_y**2)

        fade_width = 0.15
        self.patrol_mask = torch.clamp((1.0 - dist) / fade_width, min=0.0, max=1.0)
        self.patrol_mask = self.patrol_mask.view(1, 1, self.config.staleness_dim, self.config.staleness_dim)

        self.staleness_maps = torch.zeros(
            self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim, device=device
        )

        scan_radius = 12.0
        angles = torch.arange(0, 2 * np.pi, 2 * np.pi / 8, device=device)
        self.far_sensor_offsets = torch.stack([scan_radius * torch.cos(angles), scan_radius * torch.sin(angles)], dim=1)

    def update(self, env_origins, robot_pos_w, robot_yaw_w, dt) -> MapManagerOutput:
        lidar_hits_w = self._lidar_sensor.data.ray_hits_w

        exploration_bonus = self._update_staleness_map(lidar_hits_w, env_origins, dt)
        far_staleness = self._get_far_staleness(robot_pos_w, robot_yaw_w, env_origins)
        nav_data = self._sample_egocentric_maps(robot_pos_w, robot_yaw_w, env_origins)

        height_data = self._compute_height_data()
        bev_data = self._compute_bev_data()

        return MapManagerOutput(
            nav_data=nav_data,
            bev_data=bev_data,
            height_data=height_data,
            far_staleness=far_staleness,
            exploration_bonus=exploration_bonus,
        )

    def _update_staleness_map(self, lidar_hits_w, env_origins, dt) -> torch.Tensor:
        """Update staleness (decay + clear seen areas) and return exploration reward."""
        self.staleness_maps += dt * self.config.staleness_decay_rate
        self.staleness_maps = torch.minimum(self.staleness_maps, self.patrol_mask)

        rel_hits = lidar_hits_w - env_origins.unsqueeze(1)

        half_size = self.config.patrol_size / 2
        col = ((rel_hits[..., 0] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()
        row = ((rel_hits[..., 1] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()

        mask = (col >= 0) & (col < self.config.staleness_dim) & (row >= 0) & (row < self.config.staleness_dim)

        batch_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(col)
        flat_idx = batch_ids * (self.config.staleness_dim**2) + row * self.config.staleness_dim + col

        valid_idx = flat_idx[mask]

        total_cleared_value = torch.zeros(self.num_envs, device=self.device)

        if valid_idx.numel() > 0:
            flat_map = self.staleness_maps.view(-1)

            unique_idx, _ = torch.unique(valid_idx, return_inverse=True)
            env_ids_for_unique = torch.div(unique_idx, (self.config.staleness_dim**2), rounding_mode="floor")
            values_to_clear = flat_map[unique_idx]
            total_cleared_value.index_add_(0, env_ids_for_unique, values_to_clear)

            flat_map[valid_idx] = 0.0
            self.staleness_maps = flat_map.view(
                self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim
            )

        return total_cleared_value

    def _get_far_staleness(self, robot_pos_w, robot_yaw_w, env_origins) -> torch.Tensor:
        """Get average staleness in 8 cardinal directions around the robot."""
        b = self.num_envs
        h = w = self.config.staleness_dim
        device = self.device

        xs = torch.linspace(-1, 1, w, device=device)
        ys = torch.linspace(-1, 1, h, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        grid = torch.stack([gx, gy], dim=-1)
        grid = grid.view(1, h, w, 2).expand(b, -1, -1, -1)

        half = self.config.patrol_size / 2
        pixel_x = grid[..., 0] * half + env_origins[:, 0].view(b, 1, 1)
        pixel_y = grid[..., 1] * half + env_origins[:, 1].view(b, 1, 1)

        dx = pixel_x - robot_pos_w[:, 0].view(b, 1, 1)
        dy = pixel_y - robot_pos_w[:, 1].view(b, 1, 1)

        cos = torch.cos(robot_yaw_w).view(b, 1, 1)
        sin = torch.sin(robot_yaw_w).view(b, 1, 1)

        rx = dx * cos + dy * sin
        ry = -dx * sin + dy * cos

        angles = torch.atan2(ry, rx)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        oct_idx = (angles / (np.pi / 4)).long() % 8

        flat_map = self.staleness_maps.view(b, h * w)
        flat_idx = oct_idx.view(b, h * w)

        out = torch.zeros(b, 8, device=device)
        counts = torch.zeros(b, 8, device=device)

        out.scatter_add_(1, flat_idx, flat_map)
        counts.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))

        return out / (counts + 1e-6)

    def _sample_egocentric_maps(self, robot_pos_w, robot_yaw_w, env_origins) -> torch.Tensor:
        """Generate egocentric nav observation (1 channel: staleness)."""
        cos = torch.cos(robot_yaw_w).squeeze(-1)
        sin = torch.sin(robot_yaw_w).squeeze(-1)

        rel_x = (robot_pos_w[:, 0] - env_origins[:, 0]) / (self.config.patrol_size / 2)
        rel_y = (robot_pos_w[:, 1] - env_origins[:, 1]) / (self.config.patrol_size / 2)

        tx = (-rel_x * cos - rel_y * sin).view(-1, 1)
        ty = (rel_x * sin - rel_y * cos).view(-1, 1)

        zoom_s = (self.config.nav_size / self.config.patrol_size) * torch.ones_like(tx)

        theta_s = self._get_affine_matrix(zoom_s, cos.view(-1, 1), sin.view(-1, 1), tx, ty)
        grid_s = F.affine_grid(
            theta_s,
            [self.num_envs, 1, self.config.nav_dim, self.config.nav_dim],
            align_corners=False,
        )
        nav_staleness = F.grid_sample(self.staleness_maps, grid_s, align_corners=False, padding_mode="border")

        return nav_staleness

    def _get_affine_matrix(self, scale, cos, sin, tx, ty) -> torch.Tensor:
        """Generate 2x3 affine transformation matrix for grid_sample."""
        theta = torch.zeros(self.num_envs, 2, 3, device=self.device)
        theta[:, 0, 0] = scale.squeeze(-1) * cos.squeeze(-1)
        theta[:, 0, 1] = -scale.squeeze(-1) * sin.squeeze(-1)
        theta[:, 0, 2] = tx.squeeze(-1)
        theta[:, 1, 0] = scale.squeeze(-1) * sin.squeeze(-1)
        theta[:, 1, 1] = scale.squeeze(-1) * cos.squeeze(-1)
        theta[:, 1, 2] = ty.squeeze(-1)
        return theta

    def reset(self, env_ids):
        """Reset staleness maps for specific environments."""
        self.staleness_maps[env_ids] = 0.0

    def _compute_height_data(self) -> torch.Tensor:
        """Compute height scanner observation from internal sensor reference."""
        height_scanner_pos_z = self._height_scanner.data.pos_w[:, 2]
        height_scanner_ray_hits_z = self._height_scanner.data.ray_hits_w[..., 2]

        height_data = (height_scanner_pos_z.unsqueeze(1) - height_scanner_ray_hits_z - 0.5).clip(-1.0, 1.0)
        return height_data.view(self.num_envs, 64, 64)

    def _compute_bev_data(self) -> torch.Tensor:
        """Compute BEV observation from internal LiDAR sensor reference."""
        world_points = self._lidar_sensor.data.ray_hits_w
        sensor_position_world = self._lidar_sensor.data.pos_w
        sensor_quaternion_world_wxyz = self._lidar_sensor.data.quat_w

        ego_points = transform_world_to_ego(world_points, sensor_position_world, sensor_quaternion_world_wxyz)
        bev_data = build_bev(ego_points, channels=("max_height", "mean_height", "density"))

        return bev_data

