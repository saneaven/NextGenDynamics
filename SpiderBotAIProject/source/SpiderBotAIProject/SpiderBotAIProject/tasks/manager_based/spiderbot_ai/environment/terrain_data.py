# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pxr import Usd, UsdGeom

from ..custom_terrain_gen.custom_terrain_config import CustomTerrainCfg
from ..custom_terrain_gen.height_sampling import sample_height_torch
from ..custom_terrain_gen.obstacles import compute_obstacle_circles, mesh_placer
from ..custom_terrain_gen.spawnpoint_sampler import spawn_point_sampler


class TerrainData:
    """Static terrain data provider (height map, spawn points, obstacle collision).

    Loaded once at environment init time. Replaces TerrainCommandTerm which
    was abusing the CommandTerm pattern for static data storage.
    """

    def __init__(self, cfg, device: torch.device | str):
        seed = int(getattr(cfg, "seed", 42))
        terrain_cfg = CustomTerrainCfg(
            size=(float(cfg.height_map_size_x), float(cfg.height_map_size_y)),
            meter_per_grid=float(cfg.height_map_meter_per_grid),
            seed=seed,
        )

        usd_path = Path(terrain_cfg.usd_path)
        if not usd_path.exists():
            raise FileNotFoundError(
                f"Missing terrain USD: {usd_path}. "
                "Run `scripts/skrlcustom/train.py` or `scripts/skrlcustom/play.py` once to generate it."
            )

        stage = Usd.Stage.Open(str(usd_path))
        mesh = UsdGeom.Mesh.Get(stage, "/World/terrain")
        if mesh is None or not mesh.GetPrim().IsValid():
            raise RuntimeError(f"Terrain USD is missing prim '/World/terrain': {usd_path}")

        points = mesh.GetPointsAttr().Get()
        points_np = np.array(points, dtype=np.float32, copy=True)
        rows, cols = terrain_cfg.grid_size
        if points_np.shape[0] < rows * cols:
            raise RuntimeError(
                f"Terrain mesh has too few points. Need at least {rows * cols}, got {points_np.shape[0]} ({usd_path})."
            )

        # First rows*cols points are the height-map grid; extras are obstacle verts.
        height_map_np = points_np[: rows * cols, 2].reshape(rows, cols)
        self.height_map = torch.from_numpy(height_map_np).to(device=device, dtype=torch.float32)

        self.device = device
        self.meter_per_grid = float(terrain_cfg.meter_per_grid)
        self.size_x = float(terrain_cfg.size[0])
        self.size_y = float(terrain_cfg.size[1])
        self.origin_xy = torch.tensor([0.0, 0.0], device=device)

        # Recompute obstacle placement (deterministic: same seed + terrain size)
        obstacle_placement = mesh_placer(terrain_cfg, height_map_np)
        obstacle_circles_np = compute_obstacle_circles(obstacle_placement, terrain_cfg)

        spawn_points_np = None
        spawn_prim = UsdGeom.Points.Get(stage, "/World/debug/spawn_points")
        if spawn_prim is not None and spawn_prim.GetPrim().IsValid():
            spawn_points = spawn_prim.GetPointsAttr().Get()
            spawn_points_np = np.array(spawn_points, dtype=np.float32, copy=True)

        if spawn_points_np is None or spawn_points_np.size == 0:
            spawn_points_np = spawn_point_sampler(height_map_np, obstacle_placement=obstacle_placement, cfg=terrain_cfg)

        self.spawn_points = torch.from_numpy(spawn_points_np).to(device=device, dtype=torch.float32)
        self.obstacle_circles = torch.from_numpy(obstacle_circles_np).to(device=device, dtype=torch.float32)

        # Traversability grid for BFS pathfinding (coarse resolution)
        nav_res = float(getattr(cfg, "nav_grid_resolution", 1.0))
        robot_radius = float(getattr(cfg, "robot_collision_radius", 0.3))
        self.nav_meter_per_grid = nav_res
        self.nav_cols = int(self.size_x / nav_res)
        self.nav_rows = int(self.size_y / nav_res)

        traversable = np.ones((self.nav_rows, self.nav_cols), dtype=np.bool_)
        for obs in obstacle_circles_np:
            ox, oy, orad = float(obs[0]), float(obs[1]), float(obs[2])
            gj = int((ox + self.size_x / 2.0) / nav_res)
            gi = int((oy + self.size_y / 2.0) / nav_res)
            r_cells = int(np.ceil((orad + robot_radius) / nav_res))
            for di in range(-r_cells, r_cells + 1):
                for dj in range(-r_cells, r_cells + 1):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < self.nav_rows and 0 <= nj < self.nav_cols:
                        if di * di + dj * dj <= r_cells * r_cells:
                            traversable[ni, nj] = False
        self.traversable = traversable
        self._nav_graph = self._build_nav_graph()

    def _build_nav_graph(self) -> csr_matrix:
        """Build sparse adjacency matrix from traversability grid (once at init)."""
        rows, cols = self.nav_rows, self.nav_cols
        nav_res = self.nav_meter_per_grid

        # Flat indices of all traversable cells
        trav_mask = self.traversable
        cell_i, cell_j = np.nonzero(trav_mask)

        # 8-connected neighbors: (di, dj, cost)
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
        ]

        src_list = []
        dst_list = []
        weight_list = []

        for di, dj, step_cost in directions:
            ni = cell_i + di
            nj = cell_j + dj
            # In-bounds and traversable neighbor
            valid = (ni >= 0) & (ni < rows) & (nj >= 0) & (nj < cols)
            valid[valid] &= trav_mask[ni[valid], nj[valid]]

            src_flat = cell_i[valid] * cols + cell_j[valid]
            dst_flat = ni[valid] * cols + nj[valid]

            src_list.append(src_flat)
            dst_list.append(dst_flat)
            weight_list.append(np.full(src_flat.shape[0], step_cost * nav_res, dtype=np.float32))

        src_all = np.concatenate(src_list)
        dst_all = np.concatenate(dst_list)
        weights_all = np.concatenate(weight_list)

        n_nodes = rows * cols
        return csr_matrix((weights_all, (src_all, dst_all)), shape=(n_nodes, n_nodes))

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

    def sample_spawn(self, env_origins: torch.Tensor, patrol_size: float) -> torch.Tensor:
        """Sample spawn XY positions uniformly across the full spawn-point pool.

        Returns:
            (N, 2) XY world positions (caller adds Z).
        """
        n = env_origins.shape[0]
        spawn_xy = self.spawn_points[:, :2]
        if spawn_xy.shape[0] == 0:
            return env_origins[:, :2].clone()
        idx = torch.randint(0, spawn_xy.shape[0], (n,), device=self.device)
        return spawn_xy[idx].clone()

    def sample_target(self, anchor_pos_w: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample target positions with pathfinding validation.

        Each candidate must pass bounds, obstacle collision, AND BFS
        reachability from the anchor before being accepted.

        Returns:
            positions: (N, 3) target positions with Z at terrain height + offset.
            valid: (N,) bool tensor — False for environments where no reachable target was found.
            distance_fields: (N, nav_rows, nav_cols) numpy float32 BFS distance fields.
                             Valid environments contain the distance field from their target;
                             invalid environments contain inf.
        """
        n = anchor_pos_w.shape[0]
        anchor_xy = anchor_pos_w[:, :2]

        x_min = -self.size_x / 2.0 + float(cfg.spawn_padding)
        x_max = self.size_x / 2.0 - float(cfg.spawn_padding)
        y_min = -self.size_y / 2.0 + float(cfg.spawn_padding)
        y_max = self.size_y / 2.0 - float(cfg.spawn_padding)

        out_xy = anchor_xy.clone()
        valid = torch.zeros(n, device=self.device, dtype=torch.bool)
        fields = np.full((n, self.nav_rows, self.nav_cols), np.inf, dtype=np.float32)

        anchor_xy_np = anchor_xy.detach().cpu().numpy()
        nav_res = self.nav_meter_per_grid

        attempts = int(cfg.target_sample_attempts)
        for _ in range(attempts):
            remaining = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if remaining.numel() == 0:
                break

            m = remaining.numel()
            r = float(cfg.point_max_distance) + (
                float(cfg.point_min_distance) - float(cfg.point_max_distance)
            ) * torch.rand(m, device=self.device)
            a = 2.0 * torch.pi * torch.rand(m, device=self.device)
            cand_xy = anchor_xy[remaining] + torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=1)

            in_bounds = (
                (cand_xy[:, 0] >= x_min)
                & (cand_xy[:, 0] <= x_max)
                & (cand_xy[:, 1] >= y_min)
                & (cand_xy[:, 1] <= y_max)
            )
            not_collide = ~self.collides(cand_xy, margin=float(cfg.target_obstacle_margin))
            geom_ok = in_bounds & not_collide

            if not geom_ok.any():
                continue

            # BFS for geometry-ok candidates to check reachability from anchor
            geom_ok_local = geom_ok.nonzero(as_tuple=False).squeeze(-1)
            geom_ok_global = remaining[geom_ok_local]
            cand_ok_xy_np = cand_xy[geom_ok_local].detach().cpu().numpy()

            cand_fields = self._bfs_distance_field(cand_ok_xy_np)

            # Vectorized anchor→candidate reachability check
            geom_ok_global_np = geom_ok_global.cpu().numpy()
            anchor_ok_xy = anchor_xy_np[geom_ok_global_np]
            aj = np.clip(((anchor_ok_xy[:, 0] + self.size_x / 2.0) / nav_res).astype(int), 0, self.nav_cols - 1)
            ai = np.clip(((anchor_ok_xy[:, 1] + self.size_y / 2.0) / nav_res).astype(int), 0, self.nav_rows - 1)
            geo_dist = cand_fields[np.arange(len(ai)), ai, aj]
            reachable = ~np.isinf(geo_dist)

            for k in np.where(reachable)[0]:
                gidx = geom_ok_global_np[k]
                if not valid[gidx]:
                    out_xy[gidx] = cand_xy[geom_ok_local[k]]
                    fields[gidx] = cand_fields[k]
                    valid[gidx] = True

        z = self.height_at_xy(out_xy) + float(cfg.target_z_offset)
        positions = torch.cat([out_xy, z.unsqueeze(1)], dim=1)
        return positions, valid, fields

    # ------------------------------------------------------------------
    # Pathfinding (BFS on traversability grid)
    # ------------------------------------------------------------------

    def _xy_to_grid(self, xy_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world XY to grid (row, col) indices."""
        gj = ((xy_w[:, 0] + self.size_x / 2.0) / self.nav_meter_per_grid).long().clamp(0, self.nav_cols - 1)
        gi = ((xy_w[:, 1] + self.size_y / 2.0) / self.nav_meter_per_grid).long().clamp(0, self.nav_rows - 1)
        return gi, gj

    def _bfs_distance_field(self, target_xy_np: np.ndarray) -> np.ndarray:
        """Thread-safe shortest-path via scipy Dijkstra. No torch/GPU dependency.

        Args:
            target_xy_np: (N, 2) numpy array of target XY world positions.
        Returns:
            (N, nav_rows, nav_cols) numpy float32 array of geodesic distances.
        """
        n = target_xy_np.shape[0]
        nav_res = self.nav_meter_per_grid

        # Convert target XY to flat node indices
        source_indices = np.empty(n, dtype=np.intp)
        for i in range(n):
            tj = int(np.clip((target_xy_np[i, 0] + self.size_x / 2.0) / nav_res, 0, self.nav_cols - 1))
            ti = int(np.clip((target_xy_np[i, 1] + self.size_y / 2.0) / nav_res, 0, self.nav_rows - 1))
            source_indices[i] = ti * self.nav_cols + tj

        dist_matrix = dijkstra(self._nav_graph, directed=False, indices=source_indices)

        return dist_matrix.reshape(n, self.nav_rows, self.nav_cols).astype(np.float32)

    def compute_distance_field(self, target_xy: torch.Tensor) -> torch.Tensor:
        """BFS distance field (blocking convenience wrapper).

        For non-blocking usage, call compute_distance_field_cpu directly.
        """
        target_np = target_xy.detach().cpu().numpy()
        fields = self._bfs_distance_field(target_np)
        return torch.from_numpy(fields).to(self.device)

    def geodesic_distance_at(self, xy_w: torch.Tensor, distance_field: torch.Tensor) -> torch.Tensor:
        """Look up geodesic distance at world positions from precomputed distance fields.

        Args:
            xy_w: (N, 2) world XY positions.
            distance_field: (N, nav_rows, nav_cols) precomputed fields.
        Returns:
            (N,) geodesic distances in meters.
        """
        gi, gj = self._xy_to_grid(xy_w)
        return distance_field[torch.arange(xy_w.shape[0], device=self.device), gi, gj]

    def pathfinding_direction(self, xy_w: torch.Tensor, distance_field: torch.Tensor) -> torch.Tensor:
        """Compute next-move direction by following the distance field gradient.

        Args:
            xy_w: (N, 2) world XY positions (robot positions).
            distance_field: (N, nav_rows, nav_cols) precomputed fields.
        Returns:
            (N, 3) unit direction vectors in world frame (z=0).
        """
        gi, gj = self._xy_to_grid(xy_w)
        n = xy_w.shape[0]
        arange = torch.arange(n, device=self.device)

        best_dir = torch.zeros(n, 2, device=self.device)
        best_dist = distance_field[arange, gi, gj].clone()

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni = (gi + di).clamp(0, self.nav_rows - 1)
            nj = (gj + dj).clamp(0, self.nav_cols - 1)
            neighbor_dist = distance_field[arange, ni, nj]
            better = neighbor_dist < best_dist
            if better.any():
                best_dir[better, 0] = float(dj)  # dj -> x direction
                best_dir[better, 1] = float(di)  # di -> y direction
                best_dist[better] = neighbor_dist[better]

        norm = best_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        best_dir = best_dir / norm
        return torch.cat([best_dir, torch.zeros(n, 1, device=self.device)], dim=1)
