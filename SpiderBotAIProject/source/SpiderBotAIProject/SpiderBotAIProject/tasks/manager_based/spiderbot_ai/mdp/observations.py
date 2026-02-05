# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _relative_target_info(robot, target_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (unit_vector_body_frame, distance_world_frame) to a target position."""
    relative_target_pos_w = target_pos_w - robot.data.root_pos_w
    distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)
    relative_target_pos_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, relative_target_pos_w)
    unit_vector_b = relative_target_pos_b / (distance + 1e-6)
    return unit_vector_b, distance


def policy_observations(env) -> torch.Tensor:
    """Main policy vector observation (concatenated)."""
    robot = env.scene.articulations["robot"]

    robot_cache = env.command_manager.get_term("robot_cache")
    waypoint = env.command_manager.get_term("waypoint")
    map_term = env.command_manager.get_term("map")
    mode_term = env.command_manager.get_term("mode")

    robot_cache.ensure_updated()
    waypoint.ensure_updated()
    map_term.ensure_updated()
    mode_term.ensure_updated()

    target_unit_vector, target_distance = _relative_target_info(robot, waypoint.desired_pos)
    next_target_unit_vector, next_target_distance = _relative_target_info(robot, waypoint.next_desired_pos)

    can_see = map_term.can_see
    one_hot_state = mode_term.command

    obs = torch.cat(
        [
            robot.data.root_lin_vel_b,
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            target_unit_vector,
            target_distance,
            next_target_unit_vector,
            next_target_distance,
            robot_cache.is_contact,
            robot.data.joint_pos[:, robot_cache.dof_idx] - robot.data.default_joint_pos[:, robot_cache.dof_idx],
            robot.data.joint_vel[:, robot_cache.dof_idx],
            env.action_manager.action,
            map_term.output.far_staleness,
            can_see,
            one_hot_state,
        ],
        dim=-1,
    )
    return obs


def height_data(env) -> torch.Tensor:
    env.command_manager.get_term("map").ensure_updated()
    return env.command_manager.get_term("map").output.height_data


def bev_data(env) -> torch.Tensor:
    env.command_manager.get_term("map").ensure_updated()
    return env.command_manager.get_term("map").output.bev_data


def nav_data(env) -> torch.Tensor:
    env.command_manager.get_term("map").ensure_updated()
    return env.command_manager.get_term("map").output.nav_data
