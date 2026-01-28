# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def life_time_reward(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.life_time * env.step_dt


def progress_reward(env) -> torch.Tensor:
    mode_term = env.command_manager.get_term("mode")
    mode_term.ensure_updated()
    waypoint_mask = mode_term.command[:, 0]

    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.progress_metric * waypoint_mask * env.step_dt


def velocity_alignment_reward(env) -> torch.Tensor:
    mode_term = env.command_manager.get_term("mode")
    mode_term.ensure_updated()
    waypoint_mask = mode_term.command[:, 0]

    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.velocity_alignment * waypoint_mask * env.step_dt


def reach_target_reward(env) -> torch.Tensor:
    mode_term = env.command_manager.get_term("mode")
    mode_term.ensure_updated()
    waypoint_mask = mode_term.command[:, 0]

    waypoint = env.command_manager.get_term("waypoint")
    waypoint.ensure_updated()
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[waypoint.reached_target] = 0.5 + waypoint.targets_reached[waypoint.reached_target] * 0.5
    return reward * waypoint_mask


def death_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    died = env.scene.articulations["robot"].data.projected_gravity_b[:, 2] > 0.0
    on_ground = features.base_contact_time > float(env.cfg.base_on_ground_time)
    penalty = died.to(torch.float32) + on_ground.to(torch.float32)
    return penalty


def feet_ground_time_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.feet_ground_max_time * env.step_dt


def jump_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.jump_penalty * env.step_dt


def body_angular_velocity_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.body_angular_velocity * env.step_dt


def speed_reward(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.horizontal_speed * env.step_dt


def body_vertical_acceleration_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.body_vertical_acceleration * env.step_dt


def dof_torques_l2(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.joint_torques_l2 * env.step_dt


def dof_acc_l2(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.joint_accel_l2 * env.step_dt


def action_rate_l2(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.action_rate_l2 * env.step_dt


def feet_air_time_reward(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.feet_air_time * env.step_dt


def undesired_contacts_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.undesired_contacts * env.step_dt


def feet_contact_force_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.feet_contact_force * env.step_dt


def flat_orientation_l2(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.flat_orientation_l2 * env.step_dt


def wall_proximity_penalty(env) -> torch.Tensor:
    features = env.command_manager.get_term("features")
    features.ensure_updated()
    return features.wall_proximity_score * env.step_dt


def patrol_exploration_reward(env) -> torch.Tensor:
    mode_term = env.command_manager.get_term("mode")
    mode_term.ensure_updated()
    patrol_mask = mode_term.command[:, 1]

    map_term = env.command_manager.get_term("map")
    map_term.ensure_updated()
    return map_term.output.exploration_bonus * patrol_mask * env.step_dt


def patrol_boundary_penalty(env) -> torch.Tensor:
    mode_term = env.command_manager.get_term("mode")
    mode_term.ensure_updated()
    patrol_mask = mode_term.command[:, 1]

    robot = env.scene.articulations["robot"]
    rel_pos = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dist_from_center = torch.norm(rel_pos, dim=1)
    outside_patrol = torch.clamp(dist_from_center - (float(env.cfg.patrol_size) / 2), min=0.0)
    return outside_patrol * patrol_mask * env.step_dt
