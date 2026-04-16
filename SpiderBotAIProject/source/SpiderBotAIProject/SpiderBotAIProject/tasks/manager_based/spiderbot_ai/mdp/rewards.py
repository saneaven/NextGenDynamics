# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for SpiderBotAIProject.

Each function computes its value inline from sensors/robot data.
"""

from __future__ import annotations

import math

import torch


def _mode_scale(env, reward_name: str) -> torch.Tensor:
    """Per-env reward scale factor based on current mode.

    Returns (num_envs,) tensor. Falls back to 1.0 if reward_name
    is not in env.cfg.mode_reward_scales.
    """
    mode_term = env.command_manager.get_term("mode")
    scales_tuple = env.cfg.mode_reward_scales.get(reward_name)
    if scales_tuple is None:
        return torch.ones(env.num_envs, device=env.device)
    scales_tensor = torch.tensor(scales_tuple, device=env.device, dtype=torch.float32)
    return scales_tensor[mode_term.current_mode]


def life_time_reward(env) -> torch.Tensor:
    # life_time was: accumulator += step_dt each step = episode_length_buf * step_dt
    return env.episode_length_buf.float() * env.step_dt * env.step_dt


def _shape_geodesic_progress_rate(v_geo: torch.Tensor, env) -> torch.Tensor:
    """Shape geodesic progress speed around a target speed with asymmetric slopes."""
    v_ref = float(env.cfg.progress_target_speed)
    tau_lo = float(env.cfg.progress_low_speed_tau)
    tau_hi = float(env.cfg.progress_high_speed_tau)
    backward_gain = float(env.cfg.progress_backward_gain)

    low_scale = v_ref + tau_lo
    anchor = low_scale * math.log1p(v_ref / tau_lo)

    backward_speed = (-v_geo).clamp(min=0.0)
    forward_speed = v_geo.clamp(min=0.0)
    excess_speed = (v_geo - v_ref).clamp(min=0.0)

    backward_reward = -backward_gain * low_scale * torch.log1p(backward_speed / tau_lo)
    low_speed_reward = low_scale * torch.log1p(forward_speed / tau_lo)
    high_speed_reward = anchor + tau_hi * torch.log1p(excess_speed / tau_hi)

    return torch.where(
        v_geo < 0.0,
        backward_reward,
        torch.where(v_geo <= v_ref, low_speed_reward, high_speed_reward),
    )


def progress_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")

    prev_geo = waypoint.get_previous_geodesic()
    curr_geo = waypoint._geodesic_distance

    # Geodesic step-to-step delta: positive = approaching target via valid path
    delta = prev_geo - curr_geo
    delta = torch.nan_to_num(delta, nan=0.0)  # First step after reset → 0
    # Zero out on target-transition step to avoid spike from mismatched targets
    delta = torch.where(waypoint.reached_target, torch.zeros_like(delta), delta)
    # Unreachable targets (inf distance) → no progress signal
    delta = torch.where(torch.isinf(prev_geo) | torch.isinf(curr_geo), torch.zeros_like(delta), delta)

    v_geo = delta / max(float(env.step_dt), 1.0e-6)
    progress = _shape_geodesic_progress_rate(v_geo, env)
    progress = progress * (1.0 + 0.25 * waypoint.targets_reached.to(dtype=progress.dtype))

    return progress * _mode_scale(env, "progress") * env.step_dt


def velocity_alignment_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")
    robot = env.state.robot

    # Use pathfinding direction (wall-aware) instead of straight-line to target
    pathfind_dir_w = waypoint._pathfinding_dir_w[:, :2]
    velocity_alignment = torch.nn.functional.cosine_similarity(
        robot.root_lin_vel_w[:, :2], pathfind_dir_w, dim=1
    )

    return velocity_alignment * _mode_scale(env, "velocity_alignment") * env.step_dt


def reach_target_reward(env) -> torch.Tensor:
    waypoint = env.command_manager.get_term("waypoint")

    reward = (0.5 + waypoint.targets_reached * 0.5) * waypoint.reached_target.to(dtype=waypoint.targets_reached.dtype)
    return reward * _mode_scale(env, "reach_target")


def death_penalty(env) -> torch.Tensor:
    died = env.state.robot.projected_gravity_b[:, 2] > 0.0
    on_ground = env.state.contact.base_contact_time > float(env.cfg.base_on_ground_time)
    return died.to(torch.float32) + on_ground.to(torch.float32)


def feet_ground_time_penalty(env) -> torch.Tensor:
    contact_time = env.state.contact.current_contact_time[:, env.robot_idx.contact_sensor_feet_ids]
    in_contact = (contact_time > 0.0).to(contact_time.dtype)
    return torch.max(contact_time * in_contact, dim=1).values * env.step_dt


def jump_penalty(env) -> torch.Tensor:
    current_air_times = env.state.contact.current_air_time[:, env.robot_idx.contact_sensor_feet_ids]
    air_feet_per_agent = (current_air_times > 0).float().sum(dim=1)
    total_foot_num = float(len(env.robot_idx.contact_sensor_feet_ids))
    normalized_air_feet = air_feet_per_agent / total_foot_num
    normalized_air_feet[normalized_air_feet < 1e-7] = 2.0 / total_foot_num
    return (normalized_air_feet ** 3) * env.step_dt


def body_angular_velocity_penalty(env) -> torch.Tensor:
    robot = env.state.robot
    return torch.linalg.norm(
        robot.body_link_ang_vel_w[:, env.robot_idx.body_ids, :], dim=(-1, 1)
    ) * env.step_dt


def speed_reward(env) -> torch.Tensor:
    robot = env.state.robot
    up_dir = torch.tensor([0.0, 0.0, 1.0], device=env.device).view(1, 1, 3)
    v_w = robot.body_link_lin_vel_w[:, env.robot_idx.body_ids]
    dot = (v_w * up_dir).sum(dim=-1, keepdim=True)
    v_horizontal = v_w - dot * up_dir
    return v_horizontal.norm(dim=-1).squeeze(1) * _mode_scale(env, "speed") * env.step_dt


def body_vertical_acceleration_penalty(env) -> torch.Tensor:
    robot = env.state.robot
    body_vertical_acc = torch.mean(robot.body_com_lin_acc_w[:, env.robot_idx.body_ids, 2], dim=1)
    return torch.pow(body_vertical_acc, 2.0).clip(max=20.0) * env.step_dt


def dof_torques_l2(env) -> torch.Tensor:
    robot = env.state.robot
    joint_num = float(robot.joint_acc.shape[1])
    return torch.sum(torch.square(robot.applied_torque / joint_num), dim=1) * env.step_dt


def dof_acc_l2(env) -> torch.Tensor:
    robot = env.state.robot
    joint_num = float(robot.joint_acc.shape[1])
    return torch.sum(torch.square(robot.joint_acc / joint_num), dim=1) * env.step_dt


def action_rate_l2(env) -> torch.Tensor:
    delta_action = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta_action), dim=1) * env.step_dt


def feet_air_time_reward(env) -> torch.Tensor:
    first_contact = env.state.contact.first_contact[:, env.robot_idx.contact_sensor_feet_ids]
    last_air_time = env.state.contact.last_air_time[:, env.robot_idx.contact_sensor_feet_ids]
    return torch.sum((1.0 - torch.log1p(4.0 * (last_air_time - 0.5) ** 2)) * first_contact, dim=1) * env.step_dt


def undesired_contacts_penalty(env) -> torch.Tensor:
    return torch.sum(env.state.contact.is_contact[:, env.robot_idx.undesired_contact_body_ids], dim=1) * env.step_dt


def feet_contact_force_penalty(env) -> torch.Tensor:
    feet_forces = env.state.contact.net_forces_w[:, env.robot_idx.contact_sensor_feet_ids]
    return (torch.mean(torch.norm(feet_forces, dim=-1), dim=1) ** 2.0) * env.step_dt


def flat_orientation_l2(env) -> torch.Tensor:
    robot = env.state.robot
    return torch.sum(torch.square(robot.projected_gravity_b[:, :2]), dim=1) * env.step_dt


def wall_proximity_penalty(env) -> torch.Tensor:
    lidar_hits_w = env.state.sensors.lidar_hits_w
    lidar_hits_w = torch.nan_to_num(lidar_hits_w, nan=1e6, posinf=1e6, neginf=-1e6)
    rel_hits_w = lidar_hits_w - env.state.sensors.lidar_pos_w.unsqueeze(1)

    # Distance is rotation-invariant
    dists = torch.norm(rel_hits_w, dim=-1)

    threshold = float(env.cfg.wall_close_threshold)
    is_close = dists < threshold

    # Height map ground truth: compare hit Z against terrain surface height.
    # Hits on terrain (hills/terraces) have delta_z ≈ 0 → not walls.
    # Hits on obstacles (cubes/spheres) have delta_z >> 0 → walls.
    batch_size, num_points, _ = lidar_hits_w.shape
    # Convert world coords to terrain-local coords (height_at_xy expects (0,0)-centered)
    env_origins = env.scene.env_origins.unsqueeze(1)  # (B, 1, 3)
    local_hits = lidar_hits_w - env_origins  # (B, N, 3)
    hit_xy = local_hits[..., :2].reshape(-1, 2)
    terrain_z = env.terrain_data.height_at_xy(hit_xy).reshape(batch_size, num_points)
    height_above_terrain = local_hits[..., 2] - terrain_z
    is_obstacle = height_above_terrain > float(env.cfg.wall_obstacle_height)

    valid_wall_hits = is_close & is_obstacle

    if env.debug_plot.enabled:
        valid_lidar = dists[0] < threshold * 3
        env.debug_plot.scatter("Wall Detection", rel_hits_w[0, valid_lidar, :2], is_obstacle[0, valid_lidar].float())

    proximity = (threshold - dists) / threshold
    wall_score = torch.sum(proximity * valid_wall_hits.float(), dim=1) / valid_wall_hits.float().sum(dim=1).clamp(min=1.0)
    return wall_score * wall_score * env.step_dt


def patrol_exploration_reward(env) -> torch.Tensor:
    return env.state.map.exploration_bonus * _mode_scale(env, "patrol_exploration") * env.step_dt


def patrol_boundary_penalty(env) -> torch.Tensor:
    rel_pos = env.state.robot.root_pos_w[:, :2] - env.state.env_origins[:, :2]
    dist_from_center = torch.norm(rel_pos, dim=1)
    outside_patrol = torch.clamp(dist_from_center - (float(env.cfg.patrol_size) / 2), min=0.0)
    return outside_patrol * _mode_scale(env, "patrol_boundary") * env.step_dt


def chase_proximity_reward(env) -> torch.Tensor:
    """Continuous reward for staying close to the moving target in CHASE mode."""
    waypoint = env.command_manager.get_term("waypoint")
    dist = torch.linalg.norm(waypoint.desired_pos - env.state.robot.root_pos_w, dim=1)
    proximity = 1.0 / (dist + 0.5)
    return proximity * _mode_scale(env, "chase_proximity") * env.step_dt


def stillness_penalty(env) -> torch.Tensor:
    """Penalize low effective horizontal speed (EMA-filtered to reject oscillation)."""
    speed = torch.linalg.norm(env.state.smoothed_vel_xy, dim=1)
    min_speed = float(env.cfg.stillness_min_speed)
    penalty = torch.clamp(min_speed - speed, min=0.0) / min_speed
    return penalty * penalty * _mode_scale(env, "stillness") * env.step_dt
