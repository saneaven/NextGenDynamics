# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg

from ..terrain_gen.custom_terrain_generator import CustomTerrainCfg, CustomTerrainGenerator
from ..terrain_gen.heightmap_utils import sample_height_torch


class TerrainCommandTerm(CommandTerm):
    """Owns terrain-derived data (height-map, obstacles, spawn points) and exposes sampling APIs."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        seed = int(getattr(self._env.cfg, "seed", 42) or 42)
        terrain_cfg = CustomTerrainCfg(
            size=(float(self._env.cfg.height_map_size_x), float(self._env.cfg.height_map_size_y)),
            meter_per_grid=float(self._env.cfg.height_map_meter_per_grid),
            seed=seed,
        )
        generator = CustomTerrainGenerator(terrain_cfg)
        generator.initialize()

        assert generator.height_map is not None
        assert generator.spawn_points is not None

        height_map_np = generator.height_map
        self.height_map = torch.from_numpy(height_map_np).to(device=self.device, dtype=torch.float32)

        self.meter_per_grid = float(terrain_cfg.meter_per_grid)
        self.size_x = float(terrain_cfg.size[0])
        self.size_y = float(terrain_cfg.size[1])
        self.origin_xy = torch.tensor([0.0, 0.0], device=self.device)

        self.spawn_points = torch.from_numpy(generator.spawn_points).to(device=self.device, dtype=torch.float32)

        self.obstacle_circles = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
        if generator.obstacle_placement:
            circles = []
            for data in generator.obstacle_placement.values():
                positions = data.get("positions")
                scales = data.get("scales")
                if positions is None or scales is None:
                    continue
                # Conservative radius approximation from XY scale.
                radii = 0.5 * torch.from_numpy(scales[:, :2].max(axis=1)).to(device=self.device, dtype=torch.float32)
                pos = torch.from_numpy(positions[:, :2]).to(device=self.device, dtype=torch.float32)
                circles.append(torch.cat([pos, radii.unsqueeze(1)], dim=1))
            if circles:
                self.obstacle_circles = torch.cat(circles, dim=0)

    @property
    def command(self) -> torch.Tensor:
        # Expose a compact (N, 1) tensor so the manager can query/print something.
        return torch.zeros(self.num_envs, 1, device=self.device)

    def height_at_xy(self, xy_w: torch.Tensor) -> torch.Tensor:
        if xy_w.shape[-1] != 2:
            xy_w = xy_w[..., :2]
        return sample_height_torch(self.height_map, self.meter_per_grid, xy_w)

    def collides(self, xy_w: torch.Tensor, margin: float) -> torch.Tensor:
        if xy_w.shape[-1] != 2:
            xy_w = xy_w[..., :2]
        if self.obstacle_circles.numel() == 0:
            return torch.zeros(xy_w.shape[0], device=self.device, dtype=torch.bool)

        obs_xy = self.obstacle_circles[:, :2]
        obs_r = self.obstacle_circles[:, 2]
        diff = xy_w[:, None, :] - obs_xy[None, :, :]
        dist2 = torch.sum(diff * diff, dim=-1)
        thresh2 = torch.square(obs_r[None, :] + float(margin))
        return torch.any(dist2 < thresh2, dim=1)

    def sample_spawn(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample spawn positions near environment origins."""
        env_ids_t = env_ids.to(device=self.device, dtype=torch.int64)
        env_origins_xy = self._env.scene.env_origins[env_ids_t, :2]

        # Prefer spawn points within patrol radius around env origin.
        patrol_r = float(self._env.cfg.patrol_size) / 2.0
        spawn_xy = self.spawn_points[:, :2]
        out_xy = torch.zeros(env_ids_t.numel(), 2, device=self.device, dtype=torch.float32)

        for i, env_id in enumerate(env_ids_t):
            center = env_origins_xy[i]
            d = torch.norm(spawn_xy - center, dim=1)
            candidates = (d < patrol_r).nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                out_xy[i] = center
            else:
                idx = candidates[torch.randint(0, candidates.numel(), (1,), device=self.device)]
                out_xy[i] = spawn_xy[idx].squeeze(0)

        z = self.height_at_xy(out_xy) + float(self._env.cfg.spawn_z_offset_small)
        return torch.cat([out_xy, z.unsqueeze(1)], dim=1)

    def sample_target(self, anchor_pos_w: torch.Tensor) -> torch.Tensor:
        """Sample obstacle-avoiding target positions around anchors."""
        n = anchor_pos_w.shape[0]
        anchor_xy = anchor_pos_w[:, :2]

        x_min = -self.size_x / 2.0 + float(self._env.cfg.spawn_padding)
        x_max = self.size_x / 2.0 - float(self._env.cfg.spawn_padding)
        y_min = -self.size_y / 2.0 + float(self._env.cfg.spawn_padding)
        y_max = self.size_y / 2.0 - float(self._env.cfg.spawn_padding)

        out_xy = anchor_xy.clone()
        valid = torch.zeros(n, device=self.device, dtype=torch.bool)

        attempts = int(self._env.cfg.target_sample_attempts)
        for _ in range(attempts):
            remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if remaining.numel() == 0:
                break

            m = remaining.numel()
            r = float(self._env.cfg.point_max_distance) + (
                float(self._env.cfg.point_min_distance) - float(self._env.cfg.point_max_distance)
            ) * torch.rand(m, device=self.device)
            a = 2.0 * torch.pi * torch.rand(m, device=self.device)
            cand_xy = anchor_xy[remaining] + torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)

            in_bounds = (
                (cand_xy[:, 0] >= x_min)
                & (cand_xy[:, 0] <= x_max)
                & (cand_xy[:, 1] >= y_min)
                & (cand_xy[:, 1] <= y_max)
            )
            not_collide = ~self.collides(cand_xy, margin=float(self._env.cfg.target_obstacle_margin))
            ok = in_bounds & not_collide

            ok_ids = remaining[ok]
            out_xy[ok_ids] = cand_xy[ok]
            valid[ok_ids] = True

        z = self.height_at_xy(out_xy) + float(self._env.cfg.target_z_offset)
        return torch.cat([out_xy, z.unsqueeze(1)], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None):
        # Static data (height-map/obstacles/spawns) are global; no per-env reset required.
        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        return
