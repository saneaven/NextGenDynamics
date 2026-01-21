# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg


class RobotCacheCommandTerm(CommandTerm):
    """Caches indices and frequently used per-step signals for other terms."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        self.robot = self._env.scene.articulations["robot"]
        self.contact_sensor = self._env.scene.sensors["contact_sensor"]

        self.dof_idx, _ = self.robot.find_joints(self._env.cfg.actions.joint_pos.joint_names)

        self.contact_sensor_base_ids, _ = self.contact_sensor.find_bodies(self._env.cfg.base_name)
        self.contact_sensor_feet_ids, _ = self.contact_sensor.find_bodies(self._env.cfg.foot_names)
        self.undesired_contact_body_ids, _ = self.contact_sensor.find_bodies(self._env.cfg.undesired_contact_body_names)
        self.body_ids, _ = self.robot.find_bodies(self._env.cfg.base_name)

        self.is_contact = torch.zeros(
            self.num_envs,
            self.contact_sensor.data.net_forces_w_history.shape[-2],
            device=self.device,
            dtype=torch.float32,
        )
        self._last_update_step = -1

    @property
    def command(self) -> torch.Tensor:
        return self.is_contact

    def ensure_updated(self) -> None:
        """Updates cached values once per environment step."""
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > float(self._env.cfg.contact_threshold)
        self.is_contact[:] = is_contact.to(dtype=torch.float32)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.is_contact[env_ids] = 0.0
        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()
