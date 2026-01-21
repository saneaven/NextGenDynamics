# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heightmap_utils import sample_height_np


class ObstacleType:
    CUSTOM_MESH = "custom_mesh"
    CUBE = "cube"
    SPHERE = "sphere"


@dataclass
class Obstacle:
    type: str
    path: None | str = None
    scale_range: tuple[float, float] = (0.5, 2.0)
    num_instances: int = 100
    radius: float | None = None


@dataclass
class CustomTerrainCfg:
    """Configuration for the custom terrain generator.

    Note:
        This is a project-local port of the ChargeProject terrain generator config.
        It intentionally avoids non-core dependencies (pxr/opensimplex) so the generator can run
        in minimal python contexts.
    """

    # Terrain size in meters (x, y)
    size: tuple[float, float] = (196.0, 196.0)
    # Grid resolution in meters
    meter_per_grid: float = 0.15
    seed: int = 42

    # Height-map synthesis
    roughness: float = 0.05
    hill_scale: float = 768.0
    hill_height: float = 8.0
    hill_noise_lacunarity: float = 2.5
    hill_noise_persistence: float = 0.5
    hill_noise_octaves: int = 8

    # Obstacles (optional)
    obstacles: tuple[Obstacle, ...] | None = None

    # Spawn sampling
    num_points: int = 1024
    sample_radius: float = 0.5
    flatness_threshold: float = 0.5
    max_attempts: int = 1024
    margin: float = 32.0

    def __post_init__(self):
        self.grid_size = (
            int(self.size[1] / self.meter_per_grid),
            int(self.size[0] / self.meter_per_grid),
        )


def _smooth_noise_fft(rng: np.random.Generator, shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """Create smooth zero-mean noise via low-pass filtering in the frequency domain."""
    noise = rng.standard_normal(shape).astype(np.float32)
    f = np.fft.rfft2(noise)

    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    fr = np.sqrt(fx * fx + fy * fy)

    cutoff = float(np.clip(cutoff, 1e-6, 0.5))
    filt = 1.0 / (1.0 + (fr / cutoff) ** 4)
    f *= filt

    out = np.fft.irfft2(f, s=shape).astype(np.float32)
    out -= out.mean()
    std = out.std()
    if std > 1e-6:
        out /= std
    return out


def generate_height_map(cfg: CustomTerrainCfg) -> np.ndarray:
    """Generate a deterministic height-map for the given config."""
    rng = np.random.default_rng(cfg.seed)

    rough = rng.random(cfg.grid_size, dtype=np.float32) * float(cfg.roughness)

    hills = np.zeros(cfg.grid_size, dtype=np.float32)
    frequency = 1.0 / float(cfg.hill_scale)
    amplitude = 1.0
    for _ in range(int(cfg.hill_noise_octaves)):
        cutoff = np.clip(frequency * 4.0, 1e-4, 0.25)
        hills += _smooth_noise_fft(rng, cfg.grid_size, cutoff=cutoff) * amplitude
        frequency *= float(cfg.hill_noise_lacunarity)
        amplitude *= float(cfg.hill_noise_persistence)

    hills *= float(cfg.hill_height)
    return rough + hills


def _obstacle_radii(obs_type: str, scales: np.ndarray, base_radius: float | None) -> np.ndarray:
    """Approximate obstacle radius from type and (N, 3) scales."""
    s_xy = np.max(scales[:, :2], axis=1)
    if obs_type == ObstacleType.CUBE:
        return 0.5 * s_xy
    if obs_type == ObstacleType.SPHERE:
        return 0.5 * s_xy
    if obs_type == ObstacleType.CUSTOM_MESH:
        if base_radius is None:
            base_radius = 1.0
        return base_radius * s_xy
    return 0.5 * s_xy


def mesh_placer(cfg: CustomTerrainCfg, height_map: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Randomly sample obstacle placements (positions + scales)."""
    if cfg.obstacles is None:
        return {}

    rng = np.random.default_rng(cfg.seed)

    x_range = (-cfg.size[0] / 2, cfg.size[0] / 2)
    y_range = (-cfg.size[1] / 2, cfg.size[1] / 2)

    placements: dict[str, dict[str, np.ndarray]] = {}
    for obstacle in cfg.obstacles:
        num = int(obstacle.num_instances)
        coords = np.zeros((num, 2), dtype=np.float32)
        coords[:, 0] = rng.uniform(x_range[0], x_range[1], size=num)
        coords[:, 1] = rng.uniform(y_range[0], y_range[1], size=num)

        z = sample_height_np(height_map, cfg.meter_per_grid, coords)
        coords_3d = np.concatenate([coords, z.reshape(-1, 1)], axis=1).astype(np.float32)

        scales = rng.uniform(obstacle.scale_range[0], obstacle.scale_range[1], size=(num, 3)).astype(np.float32)
        placements[obstacle.type] = {"positions": coords_3d, "scales": scales}

    return placements


def spawn_point_sampler(
    height_map: np.ndarray, obstacle_placement: dict[str, dict[str, np.ndarray]] | None, cfg: CustomTerrainCfg
) -> np.ndarray:
    """Sample valid spawn points across the terrain (N, 3)."""
    rng = np.random.default_rng(cfg.seed)

    obstacle_circles = []
    if obstacle_placement and cfg.obstacles:
        obs_cfg_map = {obs.type: obs for obs in cfg.obstacles}
        for obs_type, data in obstacle_placement.items():
            positions = data.get("positions")
            scales = data.get("scales")
            if positions is None or scales is None:
                continue
            base_radius = obs_cfg_map.get(obs_type).radius if obs_type in obs_cfg_map else None
            radii = _obstacle_radii(obs_type, scales, base_radius)
            for i in range(len(positions)):
                obstacle_circles.append([positions[i, 0], positions[i, 1], radii[i]])

    obs_data = np.asarray(obstacle_circles, dtype=np.float32)  # (M, 3)

    valid_points: list[list[float]] = []

    x_min = -cfg.size[0] / 2 + cfg.margin
    x_max = cfg.size[0] / 2 - cfg.margin
    y_min = -cfg.size[1] / 2 + cfg.margin
    y_max = cfg.size[1] / 2 - cfg.margin

    rows, cols = height_map.shape

    for _ in range(int(cfg.num_points)):
        found = False
        for _attempt in range(int(cfg.max_attempts)):
            rx = rng.uniform(x_min, x_max)
            ry = rng.uniform(y_min, y_max)

            if obs_data.size:
                dists = np.linalg.norm(obs_data[:, :2] - np.array([rx, ry], dtype=np.float32), axis=1)
                min_dists = obs_data[:, 2] + cfg.sample_radius + 0.2
                if np.any(dists < min_dists):
                    continue

            grid_x = int((rx + cfg.size[0] / 2) / cfg.meter_per_grid)
            grid_y = int((ry + cfg.size[1] / 2) / cfg.meter_per_grid)
            grid_r = int(np.ceil(cfg.sample_radius / cfg.meter_per_grid))

            gx_min = max(0, grid_x - grid_r)
            gx_max = min(cols, grid_x + grid_r + 1)
            gy_min = max(0, grid_y - grid_r)
            gy_max = min(rows, grid_y + grid_r + 1)

            patch = height_map[gy_min:gy_max, gx_min:gx_max]
            if patch.size == 0:
                continue

            if (float(np.max(patch)) - float(np.min(patch))) > float(cfg.flatness_threshold):
                continue

            rz = float(sample_height_np(height_map, cfg.meter_per_grid, np.array([[rx, ry]], dtype=np.float32))[0])
            valid_points.append([float(rx), float(ry), float(rz + 0.2)])
            found = True
            break

        if not found:
            valid_points.append([0.0, 0.0, 1.0])

    return np.asarray(valid_points, dtype=np.float32)


class CustomTerrainGenerator:
    """Project-local terrain generator (height-map + spawn points + obstacle placement)."""

    def __init__(self, cfg: CustomTerrainCfg):
        self.cfg = cfg
        self.height_map: np.ndarray | None = None
        self.obstacle_placement: dict[str, dict[str, np.ndarray]] | None = None
        self.spawn_points: np.ndarray | None = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self.height_map = generate_height_map(self.cfg)
        self.obstacle_placement = mesh_placer(self.cfg, self.height_map)
        self.spawn_points = spawn_point_sampler(self.height_map, self.obstacle_placement, self.cfg)
        self._initialized = True

