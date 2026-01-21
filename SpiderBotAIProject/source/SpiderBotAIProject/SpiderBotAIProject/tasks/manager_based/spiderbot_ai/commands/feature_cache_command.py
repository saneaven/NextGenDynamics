# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg


class FeatureCacheCommandTerm(CommandTerm):
    """Caches frequently used reward/termination features once per step."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)

        self.robot = self._env.scene.articulations["robot"]
        self.contact_sensor = self._env.scene.sensors["contact_sensor"]
        self.lidar_sensor = self._env.scene.sensors["lidar_sensor"]

        self.life_time = torch.zeros(self.num_envs, device=self.device)

        # Navigation / target shaping
        self.target_distance = torch.zeros(self.num_envs, device=self.device)
        self.progress_metric = torch.zeros(self.num_envs, device=self.device)
        self.velocity_alignment = torch.zeros(self.num_envs, device=self.device)

        self.horizontal_speed = torch.zeros(self.num_envs, device=self.device)
        self.body_angular_velocity = torch.zeros(self.num_envs, device=self.device)
        self.body_vertical_acceleration = torch.zeros(self.num_envs, device=self.device)

        self.joint_torques_l2 = torch.zeros(self.num_envs, device=self.device)
        self.joint_accel_l2 = torch.zeros(self.num_envs, device=self.device)
        self.action_rate_l2 = torch.zeros(self.num_envs, device=self.device)

        self.feet_air_time = torch.zeros(self.num_envs, device=self.device)
        self.feet_ground_max_time = torch.zeros(self.num_envs, device=self.device)
        self.jump_penalty = torch.zeros(self.num_envs, device=self.device)

        self.feet_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.undesired_contacts = torch.zeros(self.num_envs, device=self.device)

        self.flat_orientation_l2 = torch.zeros(self.num_envs, device=self.device)
        self.wall_proximity_score = torch.zeros(self.num_envs, device=self.device)

        self.base_contact_time = torch.zeros(self.num_envs, device=self.device)
        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 3)
        self._last_update_step = -1

    @property
    def command(self) -> torch.Tensor:
        # Not a policy command; return a stable compact vector for debugging/inspection.
        return torch.stack([self.horizontal_speed, self.body_angular_velocity], dim=-1)

    def ensure_updated(self) -> None:
        """Updates cached values once per environment step."""
        step = int(self._env.common_step_counter)
        if self._last_update_step == step:
            return
        self._last_update_step = step

        self.life_time += self._env.step_dt

        robot_cache = self._env.command_manager.get_term("robot_cache")
        robot_cache.ensure_updated()

        waypoint = self._env.command_manager.get_term("waypoint")
        waypoint.ensure_updated()

        # Target distance (used by progress shaping).
        self.target_distance = torch.linalg.norm(waypoint.desired_pos - self.robot.data.root_pos_w, dim=1)

        # Progress shaping (ported from direct env).
        buffer = waypoint.get_distance_buffer()
        sum_valid = torch.nansum(buffer, dim=1)
        count_valid = torch.clamp(torch.sum(~torch.isnan(buffer), dim=1), min=1.0)
        previous_buffered_distance = sum_valid / count_valid

        difference = (previous_buffered_distance - self.target_distance) * (1 + 0.5 * (count_valid - 1))
        progress = torch.sign(difference) * torch.pow(torch.abs(difference), float(self._env.cfg.progress_pow))
        progress *= (waypoint.targets_reached * 0.5) + 1.0
        self.progress_metric = progress

        # Velocity alignment (ported from direct env).
        relative_target_pos_w = waypoint.desired_pos - self.robot.data.root_pos_w
        distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)
        relative_target_pos_b = math_utils.quat_apply_inverse(self.robot.data.root_quat_w, relative_target_pos_w)
        target_unit_vector = relative_target_pos_b / (distance + 1e-6)
        self.velocity_alignment = torch.nn.functional.cosine_similarity(
            self.robot.data.root_lin_vel_w[:, :2], target_unit_vector[:, :2], dim=1
        )

        # Body angular velocity (base).
        self.body_angular_velocity = torch.linalg.norm(
            self.robot.data.body_link_ang_vel_w[:, robot_cache.body_ids, :], dim=(-1, 1)
        )

        # Horizontal speed (base).
        v_w = self.robot.data.body_link_lin_vel_w[:, robot_cache.body_ids]
        dot = (v_w * self._up_dir).sum(dim=-1, keepdim=True)
        v_horizontal = v_w - dot * self._up_dir
        self.horizontal_speed = v_horizontal.norm(dim=-1).squeeze(1)

        # Body vertical acceleration.
        acc_lin_w = self.robot.data.body_com_lin_acc_w
        body_vertical_acc = torch.mean(acc_lin_w[:, robot_cache.body_ids, 2], dim=1)
        self.body_vertical_acceleration = torch.pow(body_vertical_acc, 2.0).clip(max=20.0)

        # Joint torques/accel (mean-square across joints).
        joint_num = float(self.robot.data.joint_acc.shape[1])
        self.joint_torques_l2 = torch.sum(torch.square(self.robot.data.applied_torque / joint_num), dim=1)
        self.joint_accel_l2 = torch.sum(torch.square(self.robot.data.joint_acc / joint_num), dim=1)

        # Action rate (L2 squared).
        delta_action = self._env.action_manager.action - self._env.action_manager.prev_action
        self.action_rate_l2 = torch.sum(torch.square(delta_action), dim=1)

        # Feet air-time shaping.
        first_contact = self.contact_sensor.compute_first_contact(self._env.step_dt)[:, robot_cache.contact_sensor_feet_ids]
        last_air_time = self.contact_sensor.data.last_air_time[:, robot_cache.contact_sensor_feet_ids]
        self.feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)

        # Feet ground max time.
        contact_time = self.contact_sensor.data.current_contact_time[:, robot_cache.contact_sensor_feet_ids]
        in_contact = (contact_time > 0.0).to(contact_time.dtype)
        self.feet_ground_max_time = torch.max(contact_time * in_contact, dim=1).values

        # Jump penalty (many feet in air).
        current_air_times = self.contact_sensor.data.current_air_time[:, robot_cache.contact_sensor_feet_ids]
        air_feet_per_agent = (current_air_times > 0).float().sum(dim=1)
        total_foot_num = float(len(robot_cache.contact_sensor_feet_ids))
        normalized_air_feet = air_feet_per_agent / total_foot_num
        normalized_air_feet[normalized_air_feet < 1e-7] = 2.0 / total_foot_num
        self.jump_penalty = normalized_air_feet * normalized_air_feet * normalized_air_feet

        # Feet contact force penalty.
        feet_contact_forces = self.contact_sensor.data.net_forces_w[:, robot_cache.contact_sensor_feet_ids]
        self.feet_contact_force = torch.mean(torch.norm(feet_contact_forces, dim=-1), dim=1) ** 2.0

        # Undesired contacts.
        self.undesired_contacts = torch.sum(robot_cache.is_contact[:, robot_cache.undesired_contact_body_ids], dim=1)

        # Flat orientation penalty.
        self.flat_orientation_l2 = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)

        # Wall proximity penalty.
        lidar_hits_w = self.lidar_sensor.data.ray_hits_w
        lidar_hits_w = torch.nan_to_num(lidar_hits_w, nan=0.0, posinf=1000.0, neginf=-1000.0)
        rel_hits_w = lidar_hits_w - self.lidar_sensor.data.pos_w.unsqueeze(1)

        batch_size, num_points, _ = rel_hits_w.shape
        rel_hits_w_flat = rel_hits_w.view(-1, 3)
        quat_w_expanded = self.lidar_sensor.data.quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)

        rel_hits_b_flat = math_utils.quat_apply_inverse(quat_w_expanded, rel_hits_w_flat)
        rel_hits_b = rel_hits_b_flat.view(batch_size, num_points, 3)

        dists = torch.norm(rel_hits_b, dim=-1)
        is_close = dists < float(self._env.cfg.wall_close_threshold)
        is_obstacle = rel_hits_b[:, :, 2] > float(self._env.cfg.wall_height_threshold)
        valid_wall_hits = is_close & is_obstacle

        wall_score = torch.sum((float(self._env.cfg.wall_close_threshold) - dists) * valid_wall_hits.float(), dim=1)
        wall_score = wall_score / float(num_points)
        self.wall_proximity_score = wall_score * wall_score * torch.sign(wall_score)

        # Base contact time (for on-ground termination).
        self.base_contact_time = self.contact_sensor.data.current_contact_time[:, robot_cache.contact_sensor_base_ids].squeeze(
            -1
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.life_time[env_ids] = 0.0
        self.target_distance[env_ids] = 0.0
        self.progress_metric[env_ids] = 0.0
        self.velocity_alignment[env_ids] = 0.0
        self.horizontal_speed[env_ids] = 0.0
        self.body_angular_velocity[env_ids] = 0.0
        self.body_vertical_acceleration[env_ids] = 0.0
        self.joint_torques_l2[env_ids] = 0.0
        self.joint_accel_l2[env_ids] = 0.0
        self.action_rate_l2[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_ground_max_time[env_ids] = 0.0
        self.jump_penalty[env_ids] = 0.0
        self.feet_contact_force[env_ids] = 0.0
        self.undesired_contacts[env_ids] = 0.0
        self.flat_orientation_l2[env_ids] = 0.0
        self.wall_proximity_score[env_ids] = 0.0
        self.base_contact_time[env_ids] = 0.0
        return super().reset(env_ids=env_ids)

    def _update_metrics(self):
        return

    def _resample_command(self, env_ids):
        return

    def _update_command(self):
        self.ensure_updated()
