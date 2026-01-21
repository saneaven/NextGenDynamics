# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg


class ModeCommandTerm(CommandTerm):
    """Robot mode state owned by CommandsManager.

    Modes:
        0 = WAYPOINT
        1 = PATROL
    """

    WAYPOINT = 0
    PATROL = 1

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        self.robot_mode = torch.full((self.num_envs,), self.PATROL, dtype=torch.int64, device=self.device)
        self._one_hot = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._last_update_step = -1

    @property
    def command(self) -> torch.Tensor:
        return self._one_hot

    def ensure_updated(self) -> None:
        """Updates cached one-hot mode once per environment step."""
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        idx = torch.clamp(self.robot_mode, min=0, max=1).to(dtype=torch.int64).unsqueeze(1)
        self._one_hot.zero_()
        self._one_hot.scatter_(1, idx, 1.0)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.robot_mode[env_ids] = self.PATROL
        self._one_hot[env_ids] = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float32)
        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()

