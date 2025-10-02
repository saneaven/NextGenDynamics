# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 60.0
    # - spaces definition
    action_space = 12
    # observation_space = 48
    observation_space = 51
    state_space = 0
    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Unitree Go2
    base_name = "base"
    foot_names = ".*_foot"
    undesired_contact_body_names = ".*_thigh"

    # Spot
    # base_name = "body"
    # foot_names = ".*_foot"
    # undesired_contact_body_names = ".*_uleg"

    # Anymal
    # base_name = "base"
    # foot_names = ".*FOOT"
    # undesired_contact_body_names = ".*THIGH"

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True
    )

    point_max_distance = 10  # 20 #6.0
    point_min_distance = 5  # 10 #4.0
    success_tolerance = 0.25  # 1  # meters
    time_out_per_target = 5.0  # seconds
    time_out_decrease_per_target = 0.075  # seconds

    marker_colors = 57

    """ rewards for forward (with sqrt forward vel)
    Start with sqrt then move to linear
    action_scale = 0.2

    progress_reward_scale = 0
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 0
    # Multiplied by targets hit reward
    reach_target_reward = 0
    forward_vel_reward_scale = 1.2 # 9 with linear
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -5e-05
    joint_accel_reward_scale = -1.5e-7
    dof_vel_reward_scale = 0#-0.0005
    action_rate_reward_scale = -0.003
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -0.75
    flat_orientation_reward_scale = -1
    """

    """ rewards for start of velocity alignment
    action_scale = 0.2

    progress_reward_scale = 100
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 1 #0#1.25*10
    # Multiplied by targets hit reward
    reach_target_reward = 200
    forward_vel_reward_scale = 0.036
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -5e-05
    joint_accel_reward_scale = -1.5e-7
    dof_vel_reward_scale = 0#-0.0005
    action_rate_reward_scale = -0.003
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -0.75
    flat_orientation_reward_scale = -1
    """

    """ rewards for training point to point
    action_scale = 0.2
    
    # chage learning_rate from 5.0e-04 to 3.0e-04
    progress_reward_scale = 100
    #progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 2# 6
    # Multiplied by targets hit reward
    reach_target_reward = 1000#250
    forward_vel_reward_scale = 0.075
    #lin_vel_reward_scale = 1.5
    #yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -5e-05
    joint_accel_reward_scale = -1.5e-7
    dof_vel_reward_scale = 0#-0.0005
    action_rate_reward_scale = -0.003
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -0.75
    flat_orientation_reward_scale = -1
    """

    # Final rewards
    action_scale = 0.2

    progress_reward_scale = 50 / 7.5
    # progress_target_divisor = 7.5
    velocity_alignment_reward_scale = 0
    # Multiplied by targets hit reward
    reach_target_reward_scale = 500 / 15
    forward_vel_reward_scale = 1.2 / 30  # 1.2 # 0
    time_penalty_scale = 0  # -5
    death_penalty_scale = -500
    still_penalty_scale = -5
    # lin_vel_reward_scale = 1.5
    # yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -2
    ang_vel_reward_scale = -0.0375
    joint_torque_reward_scale = -5e-05
    joint_accel_reward_scale = -1.5e-7
    dof_vel_reward_scale = 0
    action_rate_reward_scale = -0.003 * 3
    feet_air_time_reward_scale = 1.5
    undesired_contact_reward_scale = -0.75 * 3
    flat_orientation_reward_scale = -1 * 5
