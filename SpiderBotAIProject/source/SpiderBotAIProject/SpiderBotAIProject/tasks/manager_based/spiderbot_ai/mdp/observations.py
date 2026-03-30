# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _relative_target_info(robot_state, target_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (unit_vector_body_frame, distance_world_frame) to a target position."""
    relative_target_pos_w = target_pos_w - robot_state.root_pos_w
    distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)
    relative_target_pos_b = math_utils.quat_apply_inverse(robot_state.root_quat_w, relative_target_pos_w)
    unit_vector_b = relative_target_pos_b / (distance + 1e-6)
    return unit_vector_b, distance


def policy_observations(env) -> torch.Tensor:
    """Main policy vector observation (concatenated)."""
    robot = env.state.robot
    waypoint = env.command_manager.get_term("waypoint")
    mode_term = env.command_manager.get_term("mode")

    target_unit_vector, target_distance = _relative_target_info(robot, waypoint.desired_pos)
    next_target_unit_vector, next_target_distance = _relative_target_info(robot, waypoint.next_desired_pos)

    obs = torch.cat(
        [
            robot.root_lin_vel_b,
            robot.root_ang_vel_b,
            robot.projected_gravity_b,
            target_unit_vector,
            target_distance,
            next_target_unit_vector,
            next_target_distance,
            env.state.contact.is_contact,
            robot.joint_pos[:, env.robot_idx.dof_idx] - robot.default_joint_pos[:, env.robot_idx.dof_idx],
            robot.joint_vel[:, env.robot_idx.dof_idx],
            env.action_manager.action,
            env.state.map.far_staleness,
            torch.zeros(env.num_envs, 1, device=env.device),  # placeholder (model compat)
            mode_term.command,
        ],
        dim=-1,
    )
    return obs


def height_data(env) -> torch.Tensor:
    return env.state.map.height_data


def bev_data(env) -> torch.Tensor:
    return env.state.map.bev_data


def nav_data(env) -> torch.Tensor:
    return env.state.map.nav_data
