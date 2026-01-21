# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG


class SpawnCommandTerm(CommandTerm):
    """Applies spawn logic on reset.

    Note:
        This is intentionally implemented as a CommandTerm so that reset ordering is:
        scene reset -> spawn -> other command resets.
    """

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self.robot = self._env.scene.articulations["robot"]

        self._yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self.spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._debug_vis_z_offset = torch.tensor([0.0, 0.0, 0.1], device=self.device).view(1, 3)

    @property
    def command(self) -> torch.Tensor:
        # Not used by the policy; exposed only to satisfy the CommandTerm interface.
        return self._yaw

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        terrain_term = self._env.command_manager.get_term("terrain")
        spawn_pos_w = terrain_term.sample_spawn(env_ids_t)

        default_root_state = self.robot.data.default_root_state[env_ids_t].clone()
        default_root_state[:, :3] = spawn_pos_w

        # Random yaw.
        yaw = (torch.rand(env_ids_t.numel(), device=self.device) - 0.5) * float(self._env.cfg.spawn_yaw_range)
        qw = torch.cos(0.5 * yaw)
        qz = torch.sin(0.5 * yaw)
        quat_w = torch.stack([qw, torch.zeros_like(qw), torch.zeros_like(qw), qz], dim=-1)
        default_root_state[:, 3:7] = quat_w

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_t)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_t)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids_t],
            self.robot.data.default_joint_vel[env_ids_t],
            None,
            env_ids_t,
        )

        self._yaw[env_ids_t] = yaw.view(-1, 1)
        self.spawn_pos_w[env_ids_t] = spawn_pos_w
        return super().reset(env_ids=env_ids_t)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        return

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_spawn_visualizer"):
                spawn_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Commands/spawn_points")
                spawn_cfg.markers["sphere"].radius = 0.06
                spawn_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.8, 0.0)
                )
                self._spawn_visualizer = VisualizationMarkers(spawn_cfg)

            self._spawn_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_spawn_visualizer"):
                self._spawn_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self._spawn_visualizer.visualize(translations=self.spawn_pos_w + self._debug_vis_z_offset)
