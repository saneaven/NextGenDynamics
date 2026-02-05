# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg

from ..environment.map_manager import MapManager, MapManagerOutput


class MapCommandTerm(CommandTerm):
    """Owns the MapManager and computes its outputs once per step."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        self._robot = self._env.scene.articulations["robot"]
        self._height_scanner = self._env.scene.sensors["height_scanner"]
        self._lidar_sensor = self._env.scene.sensors["lidar_sensor"]

        self._map_manager = MapManager(
            config=self._env.cfg,
            num_envs=self.num_envs,
            device=self.device,
            height_scanner=self._height_scanner,
            lidar_sensor=self._lidar_sensor,
        )

        self.output = MapManagerOutput(
            nav_data=torch.zeros(self.num_envs, 1, self._env.cfg.nav_dim, self._env.cfg.nav_dim, device=self.device),
            bev_data=torch.zeros(self.num_envs, 3, 64, 64, device=self.device),
            height_data=torch.zeros(self.num_envs, 64, 64, device=self.device),
            far_staleness=torch.zeros(self.num_envs, 8, device=self.device),
            exploration_bonus=torch.zeros(self.num_envs, device=self.device),
        )
        self.can_see = torch.zeros(self.num_envs, 1, device=self.device)
        self._last_update_step = -1

    @property
    def command(self) -> torch.Tensor:
        # Expose something with (N, D) shape for manager APIs; far_staleness works as a compact "command".
        return self.output.far_staleness

    def ensure_updated(self) -> None:
        """Updates cached map values once per environment step."""
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        self._map_manager.update_into(
            self.output,
            env_origins=self._env.scene.env_origins,
            robot_pos_w=self._robot.data.root_pos_w,
            robot_yaw_w=self._robot.data.heading_w.unsqueeze(-1),
            dt=self._env.step_dt,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._map_manager.reset(env_ids)
        self.output.nav_data[env_ids] = 0.0
        self.output.bev_data[env_ids] = 0.0
        self.output.height_data[env_ids] = 0.0
        self.output.far_staleness[env_ids] = 0.0
        self.output.exploration_bonus[env_ids] = 0.0
        self.can_see[env_ids] = 0.0
        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()
